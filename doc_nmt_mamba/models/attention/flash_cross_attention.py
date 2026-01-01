"""
Cross-Attention with FlashAttention-2 for memory efficiency.

This is used for decoder-to-encoder attention, which is O(L_src) per step
but still faster than Transformer's O(L_src + L_tgt).

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


def sdpa_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """
    PyTorch native scaled dot-product cross-attention fallback.

    Args:
        q: Query tensor (batch, seq_len_q, n_heads, head_dim)
        k: Key tensor (batch, seq_len_k, n_heads, head_dim)
        v: Value tensor (batch, seq_len_k, n_heads, head_dim)
        dropout_p: Dropout probability
        training: Whether in training mode

    Returns:
        Output tensor (batch, seq_len_q, n_heads, head_dim)
    """
    # Transpose to (batch, n_heads, seq_len, head_dim) for SDPA
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Use PyTorch's optimized SDPA
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,  # Cross-attention is never causal
    )

    # Transpose back to (batch, seq_len, n_heads, head_dim)
    return out.transpose(1, 2)


class FlashCrossAttention(nn.Module):
    """
    Cross-attention using FlashAttention-2.

    Critical for fitting batch_size=64 with 8K sequences on H100.
    Uses O(N) memory instead of O(N^2).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        bias: bool = False,
        use_rope: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Attention dropout (only during training)
            max_seq_len: Maximum sequence length for RoPE cache
            bias: Whether to use bias in projections
            use_rope: Whether to apply RoPE (optional for cross-attn)
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.use_rope = use_rope

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Q from decoder hidden states
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

        # K, V from encoder output
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=bias, **factory_kwargs)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

        # RoPE for cross-attention (optional)
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        decoder_offset: int = 0,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.

        Args:
            x: Decoder hidden states
               - Padded mode: (batch, seq_len_dec, d_model)
               - Packed mode: (total_dec_tokens, d_model)
            encoder_out: Encoder output
               - Padded mode: (batch, seq_len_enc, d_model)
               - Packed mode: (total_enc_tokens, d_model)
            decoder_offset: Position offset for incremental decoding
            cu_seqlens_q: Cumulative sequence lengths for decoder (packed mode)
            cu_seqlens_k: Cumulative sequence lengths for encoder (packed mode)
            max_seqlen_q: Maximum decoder sequence length (packed mode)
            max_seqlen_k: Maximum encoder sequence length (packed mode)

        Returns:
            Output tensor (same shape as x)
        """
        is_packed = cu_seqlens_q is not None

        if is_packed:
            # Packed mode: x is (total_dec_tokens, d_model)
            residual = x
            x = self.norm(x)

            total_dec_tokens = x.size(0)
            total_enc_tokens = encoder_out.size(0)

            # Project Q from decoder
            q = self.q_proj(x)
            q = q.view(total_dec_tokens, self.n_heads, self.head_dim)

            # Project K, V from encoder
            kv = self.kv_proj(encoder_out)
            kv = kv.view(total_enc_tokens, 2, self.n_heads, self.head_dim)
            k, v = kv.unbind(dim=1)

            # Note: RoPE skipped in packed mode (Mamba handles positions)

            if FLASH_ATTN_AVAILABLE:
                # FlashAttention-2 VarLen cross-attention
                out = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=False,
                )
            else:
                # Fallback: unpack sequences and use PyTorch SDPA
                batch_size = cu_seqlens_q.size(0) - 1
                outputs = []
                for i in range(batch_size):
                    q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
                    k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
                    q_i = q[q_start:q_end].unsqueeze(0)
                    k_i = k[k_start:k_end].unsqueeze(0)
                    v_i = v[k_start:k_end].unsqueeze(0)
                    out_i = sdpa_cross_attention(q_i, k_i, v_i, self.dropout, training=self.training)
                    outputs.append(out_i.squeeze(0))
                out = torch.cat(outputs, dim=0)

            # Reshape and project output
            out = out.view(total_dec_tokens, self.d_model)
            out = self.out_proj(out)

            return residual + out
        else:
            # Padded mode: x is (batch, seq_len_dec, d_model)
            residual = x
            x = self.norm(x)

            B, T_dec, _ = x.shape
            _, T_enc, _ = encoder_out.shape

            # Project Q from decoder
            q = self.q_proj(x)
            q = q.view(B, T_dec, self.n_heads, self.head_dim)

            # Project K, V from encoder
            kv = self.kv_proj(encoder_out)
            kv = kv.view(B, T_enc, 2, self.n_heads, self.head_dim)
            k, v = kv.unbind(dim=2)

            # Apply RoPE if enabled
            if self.rope is not None:
                # Transpose for RoPE: (B, T, H, D) -> (B, H, T, D)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                q, k = self.rope.apply_to_qk(q, k, q_offset=decoder_offset, k_offset=0)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)

            if FLASH_ATTN_AVAILABLE:
                # FlashAttention-2: expects (B, T, H, D)
                # causal=False for cross-attention
                out = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=False,
                )
            else:
                # PyTorch SDPA fallback
                out = sdpa_cross_attention(
                    q, k, v,
                    dropout_p=self.dropout,
                    training=self.training,
                )

            # Reshape and project output
            out = out.view(B, T_dec, self.d_model)
            out = self.out_proj(out)

            return residual + out

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, use_rope={self.use_rope}"
        )
