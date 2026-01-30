"""RMSNorm: Required for Mamba stability at scale (per Jamba findings)."""

from typing import Optional

import torch
import torch.nn as nn

from kernels.rmsnorm import fused_rmsnorm


class RMSNorm(nn.Module):
    """
    RMSNorm: x * rsqrt(mean(x^2) + eps) * weight.

    Features:
    - Fused Triton kernel on CUDA for efficiency
    - Adaptive epsilon based on dtype (prevents BF16/FP16 underflow)
    """

    def __init__(
        self,
        d_model: int,
        eps: Optional[float] = None,  # None = adaptive based on dtype
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model

        # Adaptive epsilon: larger for low-precision dtypes to prevent underflow
        if eps is None:
            if dtype in (torch.bfloat16, torch.float16):
                self.eps = 1e-4  # BF16/FP16 have limited precision
            else:
                self.eps = 1e-6  # FP32 can use smaller epsilon
        else:
            self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "RMSNorm requires CUDA tensors"
        return fused_rmsnorm(x, self.weight, self.eps)
