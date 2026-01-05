"""
Mamba-2 Block Wrapper with RMSNorm.

CRITICAL: Do NOT re-implement the SSD algorithm in PyTorch.
The official CUDA kernels are 10-50x faster.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..components.normalization import RMSNorm

# Import Mamba2 conditionally
_mamba2_available = False
Mamba2 = None

try:
    from mamba_ssm import Mamba2 as _Mamba2
    Mamba2 = _Mamba2
    _mamba2_available = True
except ImportError:
    pass


class Mamba2BlockWrapper(nn.Module):
    """
    Wrapper around official Mamba2 with RMSNorm for stability.

    CRITICAL: Do NOT re-implement the SSD algorithm in PyTorch.
    The official CUDA kernels are 10-50x faster.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if not _mamba2_available:
            raise ImportError(
                "mamba-ssm is required for Mamba2BlockWrapper. "
                "Install with: pip install mamba-ssm (requires CUDA on Linux)"
            )

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.layer_idx = layer_idx

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            **factory_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        inference_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        # Ensure contiguous for Mamba CUDA kernels (stride alignment requirement)
        x = x.contiguous()

        if inference_params is not None:
            conv_state, ssm_state = inference_params
            x, conv_state_out, ssm_state_out = self.mamba.step(x, conv_state, ssm_state)
            conv_state.copy_(conv_state_out)
            ssm_state.copy_(ssm_state_out)
        else:
            x = self.mamba(x)

        return residual + x

    def allocate_inference_cache(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocate inference cache for autoregressive decoding."""
        d_inner = self.d_model * self.expand
        dtype = dtype or torch.bfloat16
        device = device or next(self.parameters()).device

        conv_state = torch.zeros(batch_size, d_inner, self.d_conv, dtype=dtype, device=device)
        ssm_state = torch.zeros(batch_size, d_inner, self.d_state, dtype=dtype, device=device)

        return conv_state, ssm_state

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_state={self.d_state}, d_conv={self.d_conv}, expand={self.expand}"
