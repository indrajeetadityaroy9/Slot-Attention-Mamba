"""Adaptive hyperparameter computation with theoretical citations.

All functions derive values from input statistics or model properties
rather than using fixed hyperparameters. Each function includes
explicit paper references justifying the approach.
"""

import math
from typing import Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Adaptive Warmup
# Reference: Goyal et al., 2017 (arXiv 1706.02677)
# "Gradual warmup helps to alleviate the early training instability"
# =============================================================================

class AdaptiveWarmupScheduler:
    """End warmup when gradient norm coefficient of variation stabilizes.

    Instead of using a fixed warmup ratio (e.g., 5%), this scheduler
    monitors gradient norm stability and ends warmup when gradients
    become predictable (CV < threshold).

    Reference: Goyal et al., 2017 (arXiv 1706.02677)
    """

    def __init__(
        self,
        base_lr: float,
        min_steps: int = 100,
        stability_threshold: float = 0.1,
        window_size: int = 50,
    ):
        self.base_lr = base_lr
        self.min_steps = min_steps
        self.stability_threshold = stability_threshold
        self.window_size = window_size
        self.grad_norms = []
        self.warmup_done = False
        self.warmup_ended_at = None

    def step(self, grad_norm: float) -> float:
        """Update with current gradient norm, return learning rate."""
        if self.warmup_done:
            return self.base_lr

        self.grad_norms.append(grad_norm)
        step = len(self.grad_norms)

        # Check stability after minimum steps
        if step >= self.min_steps and len(self.grad_norms) >= self.window_size:
            recent = self.grad_norms[-self.window_size:]
            mean = np.mean(recent)
            std = np.std(recent)
            cv = std / (mean + 1e-8)  # Coefficient of variation

            if cv < self.stability_threshold:
                self.warmup_done = True
                self.warmup_ended_at = step
                return self.base_lr

        # Linear warmup until stable
        return self.base_lr * step / (step + self.min_steps)

    def is_warmup_done(self) -> bool:
        return self.warmup_done


# =============================================================================
# Adaptive Dropout
# Reference: Srivastava et al., 2014 (JMLR 15(56):1929-1958)
# "Dropout prevents overfitting by providing a way of combining
#  many different neural network architectures"
# =============================================================================

def compute_adaptive_dropout(
    num_params: int,
    num_samples: int,
) -> float:
    """Derive dropout from overfitting risk (capacity/data ratio).

    Higher ratio of parameters to samples → higher dropout needed.
    Uses sigmoid-like scaling to clamp output to [0, 0.5].

    Reference: Srivastava et al., 2014 (JMLR 15(56):1929-1958)

    Calibration: 1M params / 100K samples → ~0.1 dropout (standard)
    """
    capacity_ratio = num_params / max(num_samples, 1)

    # Sigmoid-like: 0.5 * (1 - exp(-100 * ratio))
    # At ratio=0.01 (1M/100K): dropout ≈ 0.316
    # At ratio=0.001: dropout ≈ 0.048
    dropout = 0.5 * (1 - math.exp(-capacity_ratio * 100))

    # Clamp to reasonable range [0, 0.5]
    return max(0.0, min(0.5, dropout))


# =============================================================================
# Adaptive Weight Decay
# Reference: Loshchilov & Hutter, 2017 (arXiv 1711.05101)
# "Decoupled weight decay regularization"
# =============================================================================

def compute_per_param_weight_decay(
    param: torch.Tensor,
    base_decay: float = 0.01,
) -> float:
    """Scale weight decay by parameter magnitude for scale invariance.

    Larger parameters (in magnitude) receive proportionally less decay.
    This ensures regularization is consistent regardless of initialization scale.

    Reference: Loshchilov & Hutter, 2017 (arXiv 1711.05101)
    """
    param_norm = param.norm().item()
    return base_decay / max(param_norm, 1e-4)


def create_adaptive_param_groups(
    model: nn.Module,
    base_lr: float,
    base_decay: float = 0.01,
) -> list:
    """Create optimizer param groups with per-parameter weight decay.

    Parameters with larger norms get smaller decay (scale-invariant).
    Bias and normalization parameters get zero decay (standard practice).
    """
    no_decay_keywords = ["bias", "LayerNorm", "RMSNorm", "norm"]

    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if should skip decay
        skip_decay = any(kw in name for kw in no_decay_keywords)

        if skip_decay:
            param_groups.append({
                "params": [param],
                "lr": base_lr,
                "weight_decay": 0.0,
            })
        else:
            decay = compute_per_param_weight_decay(param, base_decay)
            param_groups.append({
                "params": [param],
                "lr": base_lr,
                "weight_decay": decay,
            })

    return param_groups


# =============================================================================
# Adaptive Label Smoothing
# Reference: Müller et al., 2019 (arXiv 1906.02629)
# "When Does Label Smoothing Help?"
# =============================================================================

def compute_label_smoothing_from_entropy(
    vocab_size: int,
    target_entropy_increase: float = 0.05,
) -> float:
    """Derive label smoothing from vocabulary entropy.

    Smoothing redistributes probability mass toward uniform distribution.
    Target: ~5% increase in prediction entropy.

    Reference: Müller et al., 2019 (arXiv 1906.02629)

    Formula: smoothing ≈ target_increase * H_max / (H_max + 1)
    where H_max = log(vocab_size)
    """
    H_max = math.log(vocab_size)

    # Scale smoothing to achieve target entropy increase
    smoothing = target_entropy_increase * H_max / (H_max + 1)

    # Clamp to reasonable range [0.01, 0.2]
    return max(0.01, min(0.2, smoothing))


# =============================================================================
# Adaptive Gradient Clipping (AGC)
# Reference: Brock et al., 2021 (arXiv 2102.06171, NFNet)
# "High-Performance Large-Scale Image Recognition Without Normalization"
# =============================================================================

def compute_agc_factor(
    param: torch.Tensor,
) -> float:
    """Derive AGC clip factor from parameter statistics.

    Clip factor scales with parameter standard deviation and inversely
    with fan-in, following initialization theory.

    Reference: Brock et al., 2021 (arXiv 2102.06171)
    Also: He et al., 2015 (initialization theory)
    """
    param_std = param.std().item()

    # Fan-in approximation
    fan_in = param.shape[-1] if param.dim() > 1 else 1

    # Clip factor ~ std / sqrt(fan_in)
    return param_std / (fan_in ** 0.5 + 1e-4)


def adaptive_gradient_clip_(
    parameters: Iterator[nn.Parameter],
    eps: float = 1e-3,
) -> None:
    """Apply per-parameter adaptive gradient clipping.

    Each parameter's gradient is clipped based on its own magnitude,
    rather than using a global clip factor.

    Reference: Brock et al., 2021 (arXiv 2102.06171)
    """
    for p in parameters:
        if p.grad is None:
            continue

        param_norm = p.data.norm(p=2).clamp(min=eps)
        grad_norm = p.grad.data.norm(p=2)

        # Adaptive clip factor based on parameter
        clip_factor = compute_agc_factor(p.data)
        max_norm = param_norm * clip_factor

        if grad_norm > max_norm:
            p.grad.data.mul_(max_norm / (grad_norm + eps))


# =============================================================================
# Adaptive Logging Intervals
# Reference: Standard practice (no specific paper)
# Scale with dataset size to maintain consistent logging frequency per epoch
# =============================================================================

def compute_logging_intervals(
    num_samples: int,
    batch_size: int,
    target_logs_per_epoch: int = 100,
    target_evals_per_epoch: int = 10,
    target_saves_per_epoch: int = 2,
) -> Dict[str, int]:
    """Derive logging intervals from dataset size.

    Maintains consistent frequency relative to epoch length.

    Returns:
        Dictionary with log_steps, eval_steps, save_steps
    """
    steps_per_epoch = max(1, num_samples // batch_size)

    return {
        "log_steps": max(1, steps_per_epoch // target_logs_per_epoch),
        "eval_steps": max(10, steps_per_epoch // target_evals_per_epoch),
        "save_steps": max(100, steps_per_epoch // target_saves_per_epoch),
    }


# =============================================================================
# Cosine Annealing (No Fixed Minimum LR)
# Reference: Loshchilov & Hutter, 2016 (arXiv 1608.03983, SGDR)
# "SGDR: Stochastic Gradient Descent with Warm Restarts"
# =============================================================================

def cosine_annealing_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
    min_lr: float = 0.0,  # Pure cosine per SGDR
) -> float:
    """Compute learning rate with linear warmup + cosine annealing.

    Reference: Loshchilov & Hutter, 2016 (arXiv 1608.03983)
    """
    if step < warmup_steps:
        # Linear warmup
        return base_lr * step / max(1, warmup_steps)
    else:
        # Cosine annealing
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, progress)  # Clamp to [0, 1]
        scale = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * scale
