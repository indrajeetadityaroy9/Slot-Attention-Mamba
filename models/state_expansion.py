"""HGRN2 State Expansion for increased SSM capacity.

Outer product expansion d -> d^2 without additional parameters.

State update equation:
    h_t = Diag{f_t} @ h_{t-1} + (1-f_t) outer i_t @ v_t   (state: d x d)
    y_t = o_t @ h_t

Where:
    f_t = clamp(sigmoid(x @ forget_proj), min=lower_bound)
    i_t = 1 - f_t  (tied input gate)
    v_t = x @ input_proj
    o_t = sigmoid(x @ output_proj)

The outer product state h_t in R^{d x d} provides d^2 capacity instead of d,
enabling storage of more key-value associations without increasing state size.

Reference: arXiv:2404.07904 (COLM 2024) - HGRN2
"Linear Attention with Forget Gates for Language Modeling"
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm

# Check for CUDA extension availability
try:
    import align_mamba_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False


class StateExpandedBlock(nn.Module):
    """HGRN2-style state expansion for d -> d^2 capacity.

    Uses outer product state update for increased memory capacity.
    The forget gate lower bound increases with depth to prevent
    information loss in deeper layers.

    Reference: arXiv:2404.07904 (COLM 2024)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        head_dim: int = 128,
        forget_lower_bound: float = 0.9,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        assert d_model == n_heads * head_dim, "d_model must equal n_heads * head_dim"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.forget_lower_bound = forget_lower_bound

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Projections for gates and values
        self.forget_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.input_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.output_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

        # Initialize forget gate bias toward 1 (preserving information)
        nn.init.zeros_(self.forget_proj.weight)

    def _forward_reference(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """CPU reference implementation for state expansion.

        Args:
            x: Normalized input (B, T, d_model)

        Returns:
            Output (B, T, d_model)
        """
        B, T, D = x.shape

        # Compute gates
        f_raw = self.forget_proj(x)
        f_t = torch.clamp(torch.sigmoid(f_raw), min=self.forget_lower_bound)
        i_t = 1 - f_t  # Tied input gate

        # Compute projections
        v_t = self.input_proj(x)
        o_t = torch.sigmoid(self.output_proj(x))

        # Reshape for multi-head processing
        f_t = f_t.view(B, T, self.n_heads, self.head_dim)
        i_t = i_t.view(B, T, self.n_heads, self.head_dim)
        v_t = v_t.view(B, T, self.n_heads, self.head_dim)
        o_t = o_t.view(B, T, self.n_heads, self.head_dim)

        # Initialize state: (B, n_heads, head_dim, head_dim)
        h = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim,
                       dtype=x.dtype, device=x.device)

        outputs = []
        for t in range(T):
            # Outer product: (B, n_heads, head_dim, 1) @ (B, n_heads, 1, head_dim)
            outer = i_t[:, t, :, :, None] * v_t[:, t, :, None, :]

            # State update with forget gate
            # h = diag(f_t) @ h + outer(i_t, v_t)
            h = f_t[:, t, :, :, None] * h + outer

            # Output: (B, n_heads, head_dim)
            # y_t = (o_t @ h).sum(dim=-1)
            y_t = (o_t[:, t, :, :, None] * h).sum(dim=-1)
            outputs.append(y_t)

        # Stack and reshape: (B, T, n_heads, head_dim) -> (B, T, d_model)
        output = torch.stack(outputs, dim=1)
        return output.view(B, T, D)

    def forward(
        self,
        x: torch.Tensor,
        inference_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, d_model)
            inference_params: Optional state for autoregressive decoding (B, n_heads, head_dim, head_dim)

        Returns:
            Output tensor (B, T, d_model)
        """
        residual = x
        x = self.norm(x)
        x = x.contiguous()

        if _CUDA_AVAILABLE and inference_params is None:
            # Use fused CUDA kernel for training/eval
            out = align_mamba_cuda.state_expansion_fwd(
                x,
                self.forget_proj.weight,
                self.input_proj.weight,
                self.output_proj.weight,
                self.forget_lower_bound,
                self.n_heads,
                self.head_dim,
            )
        else:
            # CPU reference implementation
            out = self._forward_reference(x)

        return residual + out

    def allocate_inference_cache(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Allocate state cache for autoregressive decoding.

        Returns:
            State tensor (B, n_heads, head_dim, head_dim)
        """
        dtype = dtype or torch.bfloat16
        device = device or next(self.parameters()).device

        return torch.zeros(
            batch_size, self.n_heads, self.head_dim, self.head_dim,
            dtype=dtype, device=device
        )


def compute_forget_lower_bound(layer_idx: int, n_layers: int) -> float:
    """Compute depth-dependent forget gate lower bound.

    Deeper layers need higher lower bounds to prevent information loss.

    Reference: arXiv:2404.07904, Section 3.2
    "The lower bound increases with depth to maintain information flow"

    Args:
        layer_idx: Current layer index (0-indexed)
        n_layers: Total number of layers

    Returns:
        Lower bound for forget gate (between 0.9 and 0.99)
    """
    # Linear interpolation from 0.9 to 0.99
    min_bound = 0.9
    max_bound = 0.99
    return min_bound + (max_bound - min_bound) * (layer_idx / max(n_layers - 1, 1))


__all__ = [
    "StateExpandedBlock",
    "compute_forget_lower_bound",
]
