"""Training objectives and schedulers for State Capacity experiments."""

import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from kernels.loss import fused_cross_entropy_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy with fused Triton kernel on GPU."""

    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        assert logits.is_cuda, "LabelSmoothingCrossEntropy requires CUDA tensors"
        return fused_cross_entropy_loss(
            logits,
            targets,
            smoothing=self.smoothing,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class CosineAnnealingWarmupScheduler(LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 4000,
        max_steps: int = 100000,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        return [self.min_lr + (base_lr - self.min_lr) * scale for base_lr in self.base_lrs]
