"""Mamba-2 Block Wrapper. Uses official CUDA kernels (10-50x faster than PyTorch reimpl)."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from mamba_ssm import Mamba2

from .normalization import RMSNorm
from .utils import process_segments_unidirectional


class Mamba2BlockWrapper(nn.Module):
    """Wrapper around official Mamba2 with RMSNorm for stability."""

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
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) or (total_tokens, d_model) when packed
            inference_params: (conv_state, ssm_state) for autoregressive decoding
            cu_seqlens: Cumulative seq lengths for packed mode state reset
        """
        residual = x
        x = self.norm(x)

        # Contiguous required for Mamba CUDA kernels (stride alignment)
        x = x.contiguous()

        if inference_params is not None:
            conv_state, ssm_state = inference_params
            x, conv_state_out, ssm_state_out = self.mamba.step(x, conv_state, ssm_state)
            conv_state.copy_(conv_state_out)
            ssm_state.copy_(ssm_state_out)
        elif cu_seqlens is not None:
            # Process each document separately to reset Mamba state at boundaries
            # Optimized: single .tolist() call instead of per-segment .item()
            x = process_segments_unidirectional(x, self.mamba, cu_seqlens)
        else:
            x = self.mamba(x)

        return residual + x

    def allocate_inference_cache(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_inner = self.d_model * self.expand
        dtype = dtype or torch.bfloat16
        device = device or next(self.parameters()).device

        conv_state = torch.zeros(batch_size, d_inner, self.d_conv, dtype=dtype, device=device)
        ssm_state = torch.zeros(batch_size, d_inner, self.d_state, dtype=dtype, device=device)

        return conv_state, ssm_state
