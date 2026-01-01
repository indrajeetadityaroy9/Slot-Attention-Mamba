"""
Training infrastructure for Document-Level NMT.

Provides:
- LabelSmoothingCrossEntropy for NMT training
- CosineAnnealingWarmupScheduler for LR scheduling
- H100-optimized Trainer with bf16 and torch.compile
- Multi-GPU DDP/FSDP distributed training
"""

from .objectives import (
    LabelSmoothingCrossEntropy,
    SequenceLoss,
    PackedSequenceLoss,
    create_loss_fn,
)
from .schedulers import (
    CosineAnnealingWarmupScheduler,
    InverseSqrtScheduler,
    LinearWarmupDecayScheduler,
    PolynomialDecayScheduler,
    create_scheduler,
)
from .trainer import Trainer, TrainerConfig
from .distributed import (
    DistributedConfig,
    setup_distributed,
    cleanup_distributed,
    wrap_model_distributed,
    get_nvlink_info,
)

__all__ = [
    # Objectives
    "LabelSmoothingCrossEntropy",
    "SequenceLoss",
    "PackedSequenceLoss",
    "create_loss_fn",
    # Schedulers
    "CosineAnnealingWarmupScheduler",
    "InverseSqrtScheduler",
    "LinearWarmupDecayScheduler",
    "PolynomialDecayScheduler",
    "create_scheduler",
    # Trainer
    "Trainer",
    "TrainerConfig",
    # Distributed
    "DistributedConfig",
    "setup_distributed",
    "cleanup_distributed",
    "wrap_model_distributed",
    "get_nvlink_info",
]
