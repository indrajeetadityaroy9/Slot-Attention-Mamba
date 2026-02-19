"""Training loop for Align-Mamba."""

import math
import os
import random
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from align_mamba.config import Config, load_yaml
from align_mamba.data import create_dataloaders
from align_mamba.model import HybridMambaEncoderDecoder, get_unwrapped_model
from align_mamba.kernels.loss import fused_cross_entropy_loss

torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def setup_distributed():
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return {
        "rank": dist.get_rank(),
        "local_rank": local_rank,
        "world_size": dist.get_world_size(),
        "device": torch.device(f"cuda:{local_rank}"),
        "is_main": dist.get_rank() == 0,
    }


class CosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, *, warmup: int, max_steps: int):
        self.optimizer = optimizer
        self.warmup = warmup
        self.max_steps = max_steps
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup:
            scale = self.step_count / max(1, self.warmup)
        else:
            progress = (self.step_count - self.warmup) / max(1, self.max_steps - self.warmup)
            scale = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g['lr'] = base_lr * scale

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    def __init__(
        self,
        model: HybridMambaEncoderDecoder,
        train_loader: DataLoader,
        config: Config,
        dist_info: dict,
        eval_loader: DataLoader,
    ):
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = dist_info["device"]
        self.is_main = dist_info["is_main"]
        self.world_size = dist_info["world_size"]
        self.output_dir = Path(config.output_dir)

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        model = torch.compile(model, mode="reduce-overhead").to(self.device)
        self.model = DDP(model, device_ids=[self.device.index], gradient_as_bucket_view=True)

        no_decay = ["bias", "norm"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_no_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        groups = [
            {"params": params_decay, "weight_decay": config.weight_decay},
            {"params": params_no_decay, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(groups, lr=config.learning_rate,
                                           betas=(0.9, 0.95), eps=1e-8, fused=True)
        self.scheduler = CosineScheduler(self.optimizer, warmup=config.warmup_steps, max_steps=config.max_steps)

        steps_per_epoch = config.num_samples // config.batch_size
        self.log_steps = max(1, steps_per_epoch // 100)
        self.eval_steps = max(1, steps_per_epoch // 10)
        self.save_steps = max(1, steps_per_epoch // 2)

        self.global_step = 0
        self.epoch = 0
        self.best_acc = 0.0

        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        self.model.train()
        data_iter = iter(self.train_loader)
        loss_accum = torch.tensor(0.0, device=self.device)
        start_time = time.time()

        while self.global_step < self.config.max_steps:
            batch = next(data_iter, None)
            if batch is None:
                self.epoch += 1
                self.train_loader.sampler.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(batch["src_ids"], batch["tgt_ids"][:, :-1])
                loss = fused_cross_entropy_loss(logits, batch["labels"], smoothing=self.config.label_smoothing)

            loss.backward()
            loss_accum += loss.detach()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

            if self.global_step % self.log_steps == 0:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                elapsed = time.time() - start_time
                throughput = self.log_steps * self.config.batch_size * self.world_size / elapsed

                if self.is_main:
                    print(f"step={self.global_step}/{self.config.max_steps} loss={loss_accum.item():.4f} "
                          f"lr={self.scheduler.get_lr():.2e} throughput={throughput:.0f}/s "
                          f"mem={torch.cuda.memory_allocated() / 1e9:.1f}GB")

                loss_accum = torch.tensor(0.0, device=self.device)
                start_time = time.time()

            if self.global_step % self.eval_steps == 0:
                self._eval()

            if self.global_step % self.save_steps == 0:
                self._save()

        if self.is_main:
            print(f"training_complete step={self.global_step}")
        dist.destroy_process_group()

    def _eval(self):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in self.eval_loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                logits = self.model(batch["src_ids"], batch["tgt_ids"][:, :-1])
                preds = logits.argmax(dim=-1)
                labels = batch["labels"]
                mask = labels != -100
                correct += ((preds == labels) & mask).sum().item()
                total += mask.sum().item()

        acc = correct / total if total > 0 else 0
        if self.is_main:
            print(f"eval step={self.global_step} acc={acc:.4f}")

        if acc > self.best_acc:
            self.best_acc = acc
            self._save("best")

        self.model.train()

    def _save(self, name=None):
        if not self.is_main:
            return

        name = name or f"checkpoint-{self.global_step}"
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)

        model = get_unwrapped_model(self.model)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model.config.to_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_acc': self.best_acc,
            'timestamp': datetime.now().isoformat(),
        }, path / "checkpoint.pt")

        print(f"saved {path}")

        ckpts = sorted([d for d in self.output_dir.iterdir() if d.name.startswith("checkpoint-")],
                      key=lambda x: int(x.name.split("-")[1]))
        while len(ckpts) > 3:
            shutil.rmtree(ckpts.pop(0))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    config, _ = load_yaml(args.config)
    dist_info = setup_distributed()

    model = HybridMambaEncoderDecoder(config, device=str(dist_info["device"]), dtype=torch.bfloat16)

    train_loader, val_loader = create_dataloaders(config, world_size=dist_info["world_size"], rank=dist_info["rank"])

    if dist_info["is_main"]:
        params = sum(p.numel() for p in model.parameters())
        print(f"Polar-Mem-Mamba | params={params/1e6:.1f}M | d_state={config.d_state} | num_pairs={config.num_pairs}")

    Trainer(model, train_loader, config, dist_info, val_loader).train()


if __name__ == "__main__":
    main()
