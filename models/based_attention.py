"""BASED Linear Attention for efficient long-range modeling.

Combines 2nd-order Taylor feature map with sliding window attention
for both global (linear) and local (quadratic) context.

Taylor feature map: phi(x) = [1, x, x_i*x_j/sqrt(2) for i<=j]
Linear attention: y = Q @ (K^T @ V) / (Q @ sum(K))

The linear attention branch provides O(n) global context via
associative scan, while the sliding window provides high-precision
local context with O(w*n) complexity.

Reference: arXiv:2402.18668 (Stanford 2024) - BASED
"Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff"
"""

import math
from typing import Optional

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

# Check for FlashAttention availability
try:
    from flash_attn import flash_attn_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False


def taylor_feature_map_reference(x: torch.Tensor) -> torch.Tensor:
    """CPU reference implementation of Taylor feature map.

    phi(x) = [1, x, x_i*x_j/sqrt(2) for i<=j]

    For d' features, output dimension is 1 + d' + d'(d'+1)/2

    Args:
        x: Input tensor (B, T, n_heads, d')

    Returns:
        Feature-mapped tensor (B, T, n_heads, expanded_dim)
    """
    B, T, H, d = x.shape

    # Constant term: 1
    ones = torch.ones(B, T, H, 1, dtype=x.dtype, device=x.device)

    # Linear terms: x
    linear = x

    # Quadratic terms: x_i * x_j / sqrt(2) for i <= j
    quad_terms = []
    sqrt2_inv = 1.0 / math.sqrt(2)
    for i in range(d):
        for j in range(i, d):
            if i == j:
                # Diagonal: x_i^2 / 2
                quad_terms.append(x[..., i:i+1] * x[..., j:j+1] * 0.5)
            else:
                # Off-diagonal: x_i * x_j / sqrt(2)
                quad_terms.append(x[..., i:i+1] * x[..., j:j+1] * sqrt2_inv)

    quadratic = torch.cat(quad_terms, dim=-1)

    return torch.cat([ones, linear, quadratic], dim=-1)


def linear_attention_reference(
    q_feat: torch.Tensor,
    k_feat: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CPU reference implementation of causal linear attention.

    Uses cumulative sum for O(n) complexity.

    y_t = (q_t @ KV_t) / (q_t @ K_sum_t)
    where KV_t = sum_{i<=t} outer(k_i, v_i)
          K_sum_t = sum_{i<=t} k_i

    Args:
        q_feat: Query after feature map (B, T, H, F)
        k_feat: Key after feature map (B, T, H, F)
        v: Value tensor (B, T, H, D)
        eps: Epsilon for numerical stability

    Returns:
        Attention output (B, T, H, D)
    """
    B, T, H, F = q_feat.shape
    D = v.shape[-1]

    # Cumulative KV-state and K-state
    # kv_state: (B, H, F, D)
    # k_state: (B, H, F)
    kv_state = torch.zeros(B, H, F, D, dtype=q_feat.dtype, device=q_feat.device)
    k_state = torch.zeros(B, H, F, dtype=q_feat.dtype, device=q_feat.device)

    outputs = []
    for t in range(T):
        # Update states with current position
        kv_state = kv_state + torch.einsum('bhf,bhd->bhfd', k_feat[:, t], v[:, t])
        k_state = k_state + k_feat[:, t]

        # Compute output: (q @ KV) / (q @ K_sum)
        y_t = torch.einsum('bhf,bhfd->bhd', q_feat[:, t], kv_state)
        z_t = torch.einsum('bhf,bhf->bh', q_feat[:, t], k_state)
        y_t = y_t / (z_t.unsqueeze(-1) + eps)

        outputs.append(y_t)

    return torch.stack(outputs, dim=1)


class BasedAttention(nn.Module):
    """BASED attention combining linear and sliding window attention.

    Linear branch: Taylor feature map + causal linear attention
    Window branch: FlashAttention2 sliding window (local context)

    The outputs are concatenated and projected back to d_model.

    Reference: arXiv:2402.18668 (Stanford 2024)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        feature_dim: int = 16,
        window_size: int = 64,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.dropout = dropout

        # Expanded dimension after Taylor map: 1 + d' + d'(d'+1)/2
        self.expanded_dim = 1 + feature_dim + feature_dim * (feature_dim + 1) // 2

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

        # Feature projection for linear attention (head_dim -> feature_dim)
        self.q_feature = nn.Linear(self.head_dim, feature_dim, bias=False, **factory_kwargs)
        self.k_feature = nn.Linear(self.head_dim, feature_dim, bias=False, **factory_kwargs)

        # Output projection: combines linear (head_dim) + window (head_dim)
        self.out_proj = nn.Linear(d_model * 2, d_model, bias=False, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, d_model)
            attention_mask: Optional attention mask

        Returns:
            Output tensor (B, T, d_model)
        """
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)

        # === Linear Attention Branch ===
        # Project to feature dimension
        q_feat = self.q_feature(q)  # (B, T, H, feature_dim)
        k_feat = self.k_feature(k)

        # Apply Taylor feature map
        if _CUDA_AVAILABLE:
            q_phi = align_mamba_cuda.taylor_feature_map(q_feat)
            k_phi = align_mamba_cuda.taylor_feature_map(k_feat)
        else:
            q_phi = taylor_feature_map_reference(q_feat)
            k_phi = taylor_feature_map_reference(k_feat)

        # Causal linear attention
        if _CUDA_AVAILABLE:
            linear_out = align_mamba_cuda.linear_attention_causal(q_phi, k_phi, v)
        else:
            linear_out = linear_attention_reference(q_phi, k_phi, v)

        # (B, T, H, D) -> (B, T, d_model)
        linear_out = linear_out.view(B, T, self.d_model)

        # === Sliding Window Branch ===
        if _FLASH_ATTN_AVAILABLE:
            # FlashAttention2 with sliding window
            window_out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                window_size=(self.window_size, 0),  # Left window only (causal)
            )
            window_out = window_out.view(B, T, self.d_model)
        else:
            # Fallback: standard attention with manual windowing
            window_out = self._sliding_window_attention_reference(q, k, v)

        # === Combine branches ===
        combined = torch.cat([linear_out, window_out], dim=-1)
        out = self.out_proj(combined)

        return residual + out

    def _sliding_window_attention_reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """CPU reference for sliding window attention.

        Args:
            q, k, v: (B, T, H, D)

        Returns:
            Output (B, T, d_model)
        """
        B, T, H, D = q.shape
        scale = 1.0 / math.sqrt(D)

        # Create causal + window mask
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device)
        mask = torch.triu(mask, diagonal=1)  # Causal mask
        # Window mask: only attend to last window_size positions
        for i in range(T):
            mask[i, :max(0, i - self.window_size)] = True

        # Compute attention
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)  # (B, T, H, D)

        return out.reshape(B, T, self.d_model)


__all__ = [
    "BasedAttention",
    "taylor_feature_map_reference",
    "linear_attention_reference",
]
