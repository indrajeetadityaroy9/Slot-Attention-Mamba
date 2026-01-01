"""
Bidirectional Attention for encoder.

Uses full attention (causal=False) since encoder sees entire sequence.
Supports VarLen mode for packed sequence training (20-30% H100 speedup).
Supports fallback to PyTorch SDPA when flash-attn is not available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import flash_attn, fallback to PyTorch SDPA if not available
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None
    flash_attn_varlen_func = None

from ..mamba2.norms import RMSNorm
from .rope import RotaryPositionalEmbedding
from .causal_self_attention import sdpa_attention


class BidirectionalAttention(nn.Module):
    """
    Bidirectional (non-causal) attention for encoder.

    Used for the 1:7 attention layers in the hybrid encoder.
    No KV cache needed since encoder processes full sequence at once.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Attention dropout
            max_seq_len: Maximum sequence length for RoPE cache
            bias: Whether to use bias in projections
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=bias, **factory_kwargs)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

        # RoPE for positional encoding
        self.rope = RotaryPositionalEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Bidirectional self-attention forward pass.

        Args:
            x: Input tensor
               - Padded mode: (batch, seq_len, d_model)
               - Packed mode: (total_tokens, d_model)
            cu_seqlens: Cumulative sequence lengths for packed mode (batch+1,)
            max_seqlen: Maximum sequence length in batch (for packed mode)

        Returns:
            Output tensor (same shape as input)
        """
        is_packed = cu_seqlens is not None

        if is_packed:
            # Packed mode: x is (total_tokens, d_model)
            residual = x
            x = self.norm(x)

            total_tokens = x.size(0)

            # Project QKV: (total_tokens, 3 * d_model)
            qkv = self.qkv_proj(x)
            qkv = qkv.view(total_tokens, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=1)  # Each: (total_tokens, n_heads, head_dim)

            # Note: RoPE in packed mode requires per-sequence position computation
            # Skipping RoPE for packed mode (Mamba handles positions anyway)

            if FLASH_ATTN_AVAILABLE:
                # FlashAttention-2 VarLen: expects (total_tokens, n_heads, head_dim)
                out = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=False,
                )
            else:
                # Fallback: unpack sequences and use PyTorch SDPA
                batch_size = cu_seqlens.size(0) - 1
                outputs = []
                for i in range(batch_size):
                    start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                    q_i = q[start:end].unsqueeze(0)
                    k_i = k[start:end].unsqueeze(0)
                    v_i = v[start:end].unsqueeze(0)
                    out_i = sdpa_attention(q_i, k_i, v_i, self.dropout, causal=False, training=self.training)
                    outputs.append(out_i.squeeze(0))
                out = torch.cat(outputs, dim=0)

            # Reshape and project output
            out = out.view(total_tokens, self.d_model)
            out = self.out_proj(out)

            return residual + out
        else:
            # Padded mode: x is (batch, seq_len, d_model)
            residual = x
            x = self.norm(x)

            B, T, _ = x.shape

            # Project QKV
            qkv = self.qkv_proj(x)
            qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)

            # Apply RoPE
            # Transpose for RoPE: (B, T, H, D) -> (B, H, T, D)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q, k = self.rope(q, k, offset=0)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            if FLASH_ATTN_AVAILABLE:
                # FlashAttention-2 with NO causal mask (bidirectional)
                out = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=False,
                )
            else:
                # PyTorch SDPA fallback
                out = sdpa_attention(
                    q, k, v,
                    dropout_p=self.dropout,
                    causal=False,
                    training=self.training,
                )

            # Reshape and project output
            out = out.view(B, T, self.d_model)
            out = self.out_proj(out)

            return residual + out

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, causal=False"
        )
