"""Mamba-2 Block Wrapper. Uses official CUDA kernels (10-50x faster than PyTorch reimpl).

Includes PolarizedMamba2Block for recency bias mitigation.
Reference: arXiv:2501.00658 (ICLR 2025) - "Polarization Technique"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from mamba_ssm import Mamba2

from .normalization import RMSNorm
from .utils import process_segments_unidirectional


# Check for CUDA extension availability
try:
    import align_mamba_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False


class Mamba2BlockWrapper(nn.Module):
    """Wrapper around official Mamba2 with RMSNorm for stability."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

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


class PolarizedMamba2Block(nn.Module):
    """Mamba2 with polarized channels for recency bias mitigation.

    Polarization addresses the fundamental recency bias in SSMs where
    tokens farther away are "under-reaching and forgotten rapidly" due
    to exponential decay (Theorem 3.1 in arXiv:2501.00658).

    Three channels:
    - Learnable (standard Mamba): Adaptive temporal dynamics
    - Zero (A=0): No memory, pure local processing
    - One (A=1): Perfect memory via cumulative sum

    The fusion of channels allows the model to attend to both local
    and global context without the information loss from exponential decay.

    Reference: arXiv:2501.00658 (ICLR 2025)
    "Polarized Attention via Controlled A-eigenvalue Channels"
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        polarized_channels: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.polarized_channels = polarized_channels
        d_inner = d_model * expand

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Learnable channel (standard Mamba)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            **factory_kwargs,
        )

        if polarized_channels > 0:
            # Zero channel (A=0): projection only, no temporal dependency
            # Output = x @ zero_proj (pure feedforward)
            self.zero_proj = nn.Linear(d_model, d_inner, bias=False, **factory_kwargs)

            # One channel (A=1): cumulative sum for perfect memory
            # Output = cumsum(x @ one_proj)
            self.one_proj = nn.Linear(d_model, d_inner, bias=False, **factory_kwargs)

            # Fusion: combine all channels back to d_model
            # Input: [mamba_out || zero_out || one_out] (3 * d_inner)
            self.fusion = nn.Linear(d_inner * 3, d_model, bias=False, **factory_kwargs)
        else:
            self.zero_proj = None
            self.one_proj = None
            self.fusion = None

    def _polarized_forward_reference(
        self,
        x: torch.Tensor,
        mamba_out: torch.Tensor,
    ) -> torch.Tensor:
        """CPU reference implementation for polarized channels.

        Args:
            x: Normalized input (B, T, d_model)
            mamba_out: Output from Mamba learnable channel (B, T, d_inner)

        Returns:
            Fused output (B, T, d_model)
        """
        # Zero channel: A=0 means no memory, pure local projection
        y_zero = self.zero_proj(x)

        # One channel: A=1 means perfect memory via cumsum
        # h_t = h_{t-1} + Δ_t · b_t(x_t) reduces to cumsum when A=1
        y_one = torch.cumsum(self.one_proj(x), dim=1)

        # Fuse all channels: [learnable, zero, one] -> d_model
        fused = torch.cat([mamba_out, y_zero, y_one], dim=-1)
        return self.fusion(fused)

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
        x = x.contiguous()

        # Learnable channel (standard Mamba)
        if inference_params is not None:
            conv_state, ssm_state = inference_params
            mamba_out, conv_state_out, ssm_state_out = self.mamba.step(x, conv_state, ssm_state)
            conv_state.copy_(conv_state_out)
            ssm_state.copy_(ssm_state_out)
        elif cu_seqlens is not None:
            mamba_out = process_segments_unidirectional(x, self.mamba, cu_seqlens)
        else:
            mamba_out = self.mamba(x)

        if self.polarized_channels == 0:
            return residual + mamba_out

        # Polarized channels
        if _CUDA_AVAILABLE and not inference_params:
            # Use fused CUDA kernel for training/eval
            polarized_out = align_mamba_cuda.polarized_fwd(
                x, mamba_out,
                self.zero_proj.weight, self.one_proj.weight,
                self.fusion.weight
            )
        else:
            # CPU reference or inference mode
            polarized_out = self._polarized_forward_reference(x, mamba_out)

        return residual + polarized_out

    def allocate_inference_cache(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Allocate cache for autoregressive decoding.

        Returns:
            Tuple of (conv_state, ssm_state, one_cumsum) for polarized inference
        """
        d_inner = self.d_model * self.expand
        dtype = dtype or torch.bfloat16
        device = device or next(self.parameters()).device

        conv_state = torch.zeros(batch_size, d_inner, self.d_conv, dtype=dtype, device=device)
        ssm_state = torch.zeros(batch_size, d_inner, self.d_state, dtype=dtype, device=device)

        # Additional state for cumsum (one channel)
        if self.polarized_channels > 0:
            one_cumsum = torch.zeros(batch_size, 1, d_inner, dtype=dtype, device=device)
            return conv_state, ssm_state, one_cumsum

        return conv_state, ssm_state
