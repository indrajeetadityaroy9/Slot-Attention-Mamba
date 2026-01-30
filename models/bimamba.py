"""Bidirectional Mamba Block for Encoder."""

import torch
import torch.nn as nn
from typing import Optional

from mamba_ssm import Mamba2

from .normalization import RMSNorm
from .utils import process_segments_bimamba


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba: concatenate OUTPUTS (y), not internal states (h).
    Forward + backward scans on full d_model, concat to 2*d_model, project down.
    """

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
        self.mamba_fwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, **factory_kwargs)
        self.mamba_bwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, **factory_kwargs)
        self.out_proj = nn.Linear(d_model * 2, d_model, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) or (total_tokens, d_model) when packed
            cu_seqlens: Cumulative seq lengths; state resets at document boundaries
        """
        residual = x
        x = self.norm(x)

        if cu_seqlens is not None:
            # Process each document separately to reset Mamba state
            # Optimized: single .tolist() call instead of per-segment .item()
            y_fwd, y_bwd = process_segments_bimamba(
                x, self.mamba_fwd, self.mamba_bwd, cu_seqlens
            )
        else:
            y_fwd = self.mamba_fwd(x)

            x_flipped = torch.flip(x, dims=[1])
            y_bwd_rev = self.mamba_bwd(x_flipped)
            y_bwd = torch.flip(y_bwd_rev, dims=[1])

        out = torch.cat([y_fwd, y_bwd], dim=-1)
        out = self.out_proj(out)

        return residual + out
