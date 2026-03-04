import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator

from config import Config, IGNORE_INDEX
from model import HybridMambaEncoderDecoder
from evaluation import perplexity_loop, mqar_accuracy_loop, _LOG_EXP_MAX
from kernels.loss import fused_cross_entropy_loss

# Optimizer states are always float32; epsilon matches that arithmetic dtype.
_ADAM_EPS = torch.finfo(torch.float32).eps


def _calibrate_lr(
    model: HybridMambaEncoderDecoder,
    batch: dict[str, torch.Tensor],
    config: Config,
    accelerator: Accelerator,
) -> float:
    """Derive learning rate from initial gradient norm.

    Runs one forward-backward pass and sets lr = 1 / (grad_norm * sqrt(d_model)).
    The 1/sqrt(d_model) factor is the muP width scaling (Yang et al. 2022).
    """
    model.train()
    model.zero_grad()
    with accelerator.autocast():
        if config.task == "lm":
            logits = model(None, batch["input_ids"][:, :-1])
            labels = batch["labels"][:, 1:]
        else:
            logits = model(batch["src_ids"], batch["tgt_ids"][:, :-1])
            labels = batch["labels"]
        loss = fused_cross_entropy_loss(
            logits, labels,
            smoothing=config.label_smoothing,
            ignore_index=IGNORE_INDEX,
        )
    accelerator.backward(loss)

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    model.zero_grad(set_to_none=True)

    lr = 1.0 / (total_norm.item() * math.sqrt(config.d_model))
    return lr


class CosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, *, max_steps: int):
        self.optimizer = optimizer
        # Warmup = Adam beta2 convergence timescale (Kingma & Ba 2015, Section 3).
        beta2 = optimizer.param_groups[0]["betas"][1]
        self.warmup = math.ceil(1.0 / (1.0 - beta2))
        self.max_steps = max_steps
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup:
            scale = self.step_count / self.warmup
        else:
            progress = (self.step_count - self.warmup) / (self.max_steps - self.warmup)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base_lr * scale

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


class Trainer:
    def __init__(
        self,
        model: HybridMambaEncoderDecoder,
        train_loader: DataLoader,
        config: Config,
        accelerator: Accelerator,
        eval_loader: DataLoader,
    ):
        self.config = config
        self.accelerator = accelerator
        self.is_main = accelerator.is_main_process
        self.output_dir = Path(config.output_dir)
        self.is_lm = config.task == "lm"

        # Calibrate LR before torch.compile (avoids recompilation from probe).
        first_batch = next(iter(train_loader))
        first_batch = {k: v.to("cuda", non_blocking=True) for k, v in first_batch.items()}
        lr = _calibrate_lr(model, first_batch, config, accelerator)

        # Weight decay: scale-invariant equilibrium WD = LR (Loshchilov & Hutter).
        wd = lr

        model = torch.compile(model, mode="reduce-overhead")

        no_decay = ["bias", "norm"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_no_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        groups = [
            {"params": params_decay, "weight_decay": wd},
            {"params": params_no_decay, "weight_decay": 0.0},
        ]

        betas = (config.adam_beta1, config.adam_beta2)
        optimizer = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=_ADAM_EPS, fused=True)

        self.model, self.optimizer, self.train_loader, self.eval_loader = (
            accelerator.prepare(model, optimizer, train_loader, eval_loader)
        )

        self.scheduler = CosineScheduler(self.optimizer, max_steps=config.max_steps)

        if self.is_lm:
            self.epoch_steps = len(train_loader.dataset) // config.batch_size
        else:
            self.epoch_steps = config.num_samples // config.batch_size

        self.global_step = 0
        self.epoch = 0
        self._best_metric = float("inf") if self.is_lm else 0.0

        # EMA clipping follows the observed gradient scale for this run.
        self._grad_ema_decay = 1.0 - 1.0 / self.epoch_steps
        self._grad_norm_ema = math.sqrt(config.d_model)

        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[derived] lr={lr:.6e} wd={wd:.6e} betas=({betas[0]:.4f}, {betas[1]:.6f}) "
                  f"label_smoothing={config.label_smoothing:.4f}")

    def resume_from(self, path: str) -> None:
        ckpt = torch.load(Path(path) / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.accelerator.unwrap_model(self.model).load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["global_step"]
        self.epoch = ckpt["epoch"]
        self._best_metric = ckpt["best_metric"]
        self.scheduler.step_count = ckpt["scheduler_step"]

        if self.is_main:
            print(f"[resume] checkpoint={path} step={self.global_step} best_metric={self._best_metric:.6f}")

    def train(self):
        self.model.train()
        data_iter = iter(self.train_loader)
        loss_accum = torch.tensor(0.0, device="cuda")

        while self.global_step < self.config.max_steps:
            batch = next(data_iter, None)
            if batch is None:
                self.epoch += 1
                self.train_loader.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}

            with self.accelerator.autocast():
                if self.is_lm:
                    logits = self.model(None, batch["input_ids"][:, :-1])
                    labels = batch["labels"][:, 1:]
                else:
                    logits = self.model(batch["src_ids"], batch["tgt_ids"][:, :-1])
                    labels = batch["labels"]
                loss = fused_cross_entropy_loss(
                    logits,
                    labels,
                    smoothing=self.config.label_smoothing,
                    ignore_index=IGNORE_INDEX,
                )

            self.accelerator.backward(loss)
            loss_accum += loss.detach()
            self.global_step += 1

            # Sub-Gaussian concentration: clip at EMA * (1 + 1/sqrt(t)).
            clip_val = self._grad_norm_ema * (1.0 + 1.0 / math.sqrt(self.global_step))
            grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), clip_val)
            self._grad_norm_ema = (
                self._grad_ema_decay * self._grad_norm_ema
                + (1 - self._grad_ema_decay) * grad_norm.item()
            )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.global_step % self.epoch_steps == 0:
                avg_loss = self.accelerator.gather(loss_accum).mean().item() / self.epoch_steps

                if self.is_main:
                    print(f"[train] epoch={self.epoch + 1} step={self.global_step}/{self.config.max_steps} loss={avg_loss:.6f} lr={self.scheduler.get_lr():.6e}")

                loss_accum = torch.tensor(0.0, device="cuda")
                self._eval()
                self._save(f"checkpoint-{self.global_step}")

        if self.is_main:
            print(f"[complete] step={self.global_step}/{self.config.max_steps} best_metric={self._best_metric:.6f}")

    def _eval(self):
        self.model.eval()

        if self.is_lm:
            total_loss, total_tokens = perplexity_loop(self.model, self.eval_loader)
            stats = self.accelerator.gather(torch.tensor([total_loss, float(total_tokens)], device="cuda"))
            total_loss = stats[::2].sum().item()
            total_tokens = int(stats[1::2].sum().item())
            avg_loss = total_loss / total_tokens
            ppl = math.exp(min(avg_loss, _LOG_EXP_MAX))
            if self.is_main:
                print(f"[eval] step={self.global_step} epoch={self.epoch + 1} val_loss={avg_loss:.6f} perplexity={ppl:.2f}")
            if ppl < self._best_metric:
                self._best_metric = ppl
                self._save("best")
        else:
            acc = mqar_accuracy_loop(self.model, self.eval_loader, self.config.vocab_size)
            if self.is_main:
                print(f"[eval] step={self.global_step} epoch={self.epoch + 1} token_accuracy={acc:.6f}")
            if acc > self._best_metric:
                self._best_metric = acc
                self._save("best")

        self.model.train()

    def _save(self, name: str):
        if not self.is_main:
            return

        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(
            {
                "model_state_dict": self.accelerator.get_state_dict(self.model),
                "config": asdict(unwrapped.config),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_step": self.scheduler.step_count,
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_metric": self._best_metric,
            },
            path / "checkpoint.pt",
        )

