"""Training loop for Align-Mamba."""

import os
import math
import time
import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from align_mamba.config import Config
from align_mamba.model import HybridMambaEncoderDecoder, get_unwrapped_model
from align_mamba.kernels.loss import fused_cross_entropy_loss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


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
    def __init__(self, optimizer, warmup: int, max_steps: int):
        self.optimizer = optimizer
        self.warmup = warmup
        self.max_steps = max_steps
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup:
            scale = self.step_count / self.warmup
        else:
            progress = (self.step_count - self.warmup) / (self.max_steps - self.warmup)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g['lr'] = base_lr * scale

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    def __init__(self, model, train_loader, config, dist_info, eval_loader):
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
        groups = [
            {"params": [p], "lr": config.learning_rate,
             "weight_decay": 0.0 if any(nd in n for nd in no_decay) else 0.01 / p.norm().item()}
            for n, p in model.named_parameters()
        ]
        self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, fused=True)
        self.scheduler = CosineScheduler(self.optimizer, 100, config.max_steps)
        self.smoothing = config.label_smoothing

        steps_per_epoch = config.num_samples // config.batch_size
        self.log_steps = steps_per_epoch // 100
        self.eval_steps = steps_per_epoch // 10
        self.save_steps = steps_per_epoch // 2

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
                loss = fused_cross_entropy_loss(logits, batch["labels"], self.smoothing)

            loss.backward()
            loss_accum += loss.detach()

            for p in self.model.parameters():
                pn = p.data.norm(p=2).clamp(min=1e-3)
                gn = p.grad.data.norm(p=2)
                clip = pn * p.data.std().item() / (p.shape[-1] ** 0.5 + 1e-4) if p.dim() > 1 else pn * 0.01
                if gn > clip:
                    p.grad.data.mul_(clip / gn)

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

        acc = correct / total
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
    config = Config.from_args()
    dist_info = setup_distributed()

    model = HybridMambaEncoderDecoder.from_config(config, str(dist_info["device"]), torch.bfloat16)

    from align_mamba.data import create_dataloaders
    train_loader, val_loader = create_dataloaders(config, dist_info["world_size"], dist_info["rank"])

    if dist_info["is_main"]:
        params = sum(p.numel() for p in model.parameters())
        print(f"params={params / 1e6:.1f}M cross_attn={sorted(model.decoder.cross_attn_positions)}")

    Trainer(model, train_loader, config, dist_info, val_loader).train()


if __name__ == "__main__":
    main()
