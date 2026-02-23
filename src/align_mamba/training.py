import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from torchmetrics.classification import MulticlassAccuracy

from align_mamba.config import Config, IGNORE_INDEX
from align_mamba.model import HybridMambaEncoderDecoder
from align_mamba.evaluation import perplexity_loop
from align_mamba.kernels.loss import fused_cross_entropy_loss

_ADAM_BETAS = (0.9, 0.999)
_ADAM_EPS = 1e-8

_WARMUP_RATIO = 0.05
_LR_FLOOR_SCALE = _WARMUP_RATIO

_GRAD_CLIP_FACTOR = 2.0

_LABEL_SMOOTHING = 0.1
_WEIGHT_DECAY = 0.01
_MAX_CHECKPOINTS = 3

_BASE_LR = 3e-4
_BASE_D_MODEL = 256


def emit_log(**payload) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


class CosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, *, max_steps: int):
        self.optimizer = optimizer
        self.warmup = int(_WARMUP_RATIO * max_steps)
        self.max_steps = max_steps
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup:
            scale = self.step_count / self.warmup
        else:
            progress = (self.step_count - self.warmup) / (self.max_steps - self.warmup)
            scale = max(_LR_FLOOR_SCALE, 0.5 * (1 + math.cos(math.pi * progress)))
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
        self.device = accelerator.device
        self.is_main = accelerator.is_main_process
        self.output_dir = Path(config.output_dir)
        self.is_lm = config.task == "lm"

        model = torch.compile(model, mode="reduce-overhead")

        no_decay = ["bias", "norm"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_no_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        groups = [
            {"params": params_decay, "weight_decay": _WEIGHT_DECAY},
            {"params": params_no_decay, "weight_decay": 0.0},
        ]

        # Scale LR by width to keep update magnitudes stable across model sizes.
        lr = _BASE_LR * math.sqrt(_BASE_D_MODEL / config.d_model)
        optimizer = torch.optim.AdamW(groups, lr=lr, betas=_ADAM_BETAS, eps=_ADAM_EPS, fused=True)

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
        self._grad_ema_decay = 1.0 - 1.0 / max(self.epoch_steps, 1)
        self._grad_norm_ema = math.sqrt(config.d_model)

        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def resume_from(self, path: str) -> None:
        ckpt = torch.load(Path(path) / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.accelerator.unwrap_model(self.model).load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["global_step"]
        self.epoch = ckpt["epoch"]
        self._best_metric = ckpt["best_metric"]
        self.scheduler.step_count = ckpt["scheduler_step"]

        if self.is_main:
            emit_log(
                event="train_resume",
                checkpoint=str(path),
                step=self.global_step,
                best_metric=round(self._best_metric, 6),
            )

    def train(self):
        self.model.train()
        data_iter = iter(self.train_loader)
        loss_accum = torch.tensor(0.0, device=self.device)

        while self.global_step < self.config.max_steps:
            batch = next(data_iter, None)
            if batch is None:
                self.epoch += 1
                self.train_loader.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

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
                    smoothing=_LABEL_SMOOTHING,
                    ignore_index=IGNORE_INDEX,
                )

            self.accelerator.backward(loss)
            loss_accum += loss.detach()

            clip_val = self._grad_norm_ema * _GRAD_CLIP_FACTOR
            grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), clip_val)
            self._grad_norm_ema = (
                self._grad_ema_decay * self._grad_norm_ema
                + (1 - self._grad_ema_decay) * grad_norm.item()
            )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

            if self.global_step % self.epoch_steps == 0:
                avg_loss = self.accelerator.gather(loss_accum).mean().item() / self.epoch_steps

                if self.is_main:
                    emit_log(
                        event="train_epoch",
                        epoch=self.epoch + 1,
                        step=self.global_step,
                        max_steps=self.config.max_steps,
                        loss=round(avg_loss, 6),
                        lr=self.scheduler.get_lr(),
                    )

                loss_accum = torch.tensor(0.0, device=self.device)
                self._eval()
                self._save()

        if self.is_main:
            emit_log(
                event="train_complete",
                step=self.global_step,
                max_steps=self.config.max_steps,
                best_metric=round(self._best_metric, 6),
            )

    def _eval(self):
        self.model.eval()

        if self.is_lm:
            ppl, avg_loss, _ = perplexity_loop(self.model, self.eval_loader, self.device)
            if self.is_main:
                emit_log(
                    event="eval",
                    step=self.global_step,
                    epoch=self.epoch + 1,
                    val_loss=round(avg_loss, 6),
                    perplexity=round(ppl, 2),
                )
            if ppl < self._best_metric:
                self._best_metric = ppl
                self._save("best")
        else:
            accuracy = MulticlassAccuracy(
                num_classes=self.config.vocab_size,
                ignore_index=IGNORE_INDEX,
                average="micro",
            ).to(self.device)

            with torch.no_grad():
                for batch in self.eval_loader:
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    logits = self.model(batch["src_ids"], batch["tgt_ids"][:, :-1])
                    labels = batch["labels"]
                    accuracy.update(logits.view(-1, logits.size(-1)), labels.reshape(-1))

            acc = accuracy.compute().item()
            if self.is_main:
                emit_log(
                    event="eval",
                    step=self.global_step,
                    epoch=self.epoch + 1,
                    token_accuracy=round(acc, 6),
                )
            if acc > self._best_metric:
                self._best_metric = acc
                self._save("best")

        self.model.train()

    def _save(self, name: str = ""):
        if not self.is_main:
            return

        if not name:
            name = f"checkpoint-{self.global_step}"
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

        ckpts = sorted(
            [d for d in self.output_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1]),
        )
        while len(ckpts) > _MAX_CHECKPOINTS:
            shutil.rmtree(ckpts.pop(0))
