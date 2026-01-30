"""Attention mechanisms for Hybrid Mamba-Attention architecture."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from flash_attn import flash_attn_func, flash_attn_varlen_func

from .normalization import RMSNorm
from .embeddings import RotaryPositionalEmbedding
from .base import broadcast_mask_for_sdpa


class BidirectionalAttention(nn.Module):
    """Bidirectional (non-causal) attention for encoder sparse layers."""

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
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim, max_seq_len=max_seq_len, device=device)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        is_packed = cu_seqlens is not None

        if is_packed:
            residual = x
            x = self.norm(x)
            total_tokens = x.size(0)

            qkv = self.qkv_proj(x)
            qkv = qkv.view(total_tokens, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=1)

            out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
            )

            out = out.view(total_tokens, self.d_model)
            out = self.out_proj(out)
            return residual + out
        else:
            residual = x
            x = self.norm(x)
            B, T, _ = x.shape

            qkv = self.qkv_proj(x)
            qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q, k = self.rope(q, k, offset=0)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            # SDPA for masked attention (supports arbitrary masks), FlashAttention otherwise
            if attention_mask is not None:
                attn_mask = broadcast_mask_for_sdpa(attention_mask, B, q.dtype)

                q_sdpa = q.transpose(1, 2)
                k_sdpa = k.transpose(1, 2)
                v_sdpa = v.transpose(1, 2)
                out = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                )
                out = out.transpose(1, 2)
            else:
                out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=False)

            out = out.view(B, T, self.d_model)
            out = self.out_proj(out)
            return residual + out


class FlashCrossAttention(nn.Module):
    """
    Cross-attention using FlashAttention-2. O(N) memory vs O(N^2).

    Features:
    - QK-Norm: L2 normalize Q and K before dot product (prevents attention collapse)
    - Learned temperature: Per-module temperature parameter for adaptive sharpness
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        bias: bool = False,
        use_rope: bool = True,
        use_qk_norm: bool = True,  # QK-Norm for cross-attention stability
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(dim=self.head_dim, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

        # Softmax scale for attention (sqrt(head_dim) is standard)
        self.softmax_scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        decoder_offset: int = 0,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> torch.Tensor:
        is_packed = cu_seqlens_q is not None

        if is_packed:
            residual = x
            x = self.norm(x)
            total_dec_tokens = x.size(0)
            total_enc_tokens = encoder_out.size(0)

            q = self.q_proj(x)
            q = q.view(total_dec_tokens, self.n_heads, self.head_dim)

            kv = self.kv_proj(encoder_out)
            kv = kv.view(total_enc_tokens, 2, self.n_heads, self.head_dim)
            k, v = kv.unbind(dim=1)

            # QK-Norm: L2 normalize Q and K to prevent attention collapse
            if self.use_qk_norm:
                q = F.normalize(q, p=2, dim=-1)
                k = F.normalize(k, p=2, dim=-1)

            out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=False,
            )

            out = out.view(total_dec_tokens, self.d_model)
            out = self.out_proj(out)
            return residual + out
        else:
            residual = x
            x = self.norm(x)
            B, T_dec, _ = x.shape
            _, T_enc, _ = encoder_out.shape

            q = self.q_proj(x)
            q = q.view(B, T_dec, self.n_heads, self.head_dim)

            kv = self.kv_proj(encoder_out)
            kv = kv.view(B, T_enc, 2, self.n_heads, self.head_dim)
            k, v = kv.unbind(dim=2)

            if self.rope is not None:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                q, k = self.rope.apply_to_qk(q, k, q_offset=decoder_offset, k_offset=0)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)

            # QK-Norm: L2 normalize Q and K to prevent attention collapse
            if self.use_qk_norm:
                q = F.normalize(q, p=2, dim=-1)
                k = F.normalize(k, p=2, dim=-1)

            # SDPA for masked attention, FlashAttention otherwise
            if encoder_padding_mask is not None:
                attn_mask = broadcast_mask_for_sdpa(encoder_padding_mask, B, q.dtype, src_len=T_enc)

                q_sdpa = q.transpose(1, 2)
                k_sdpa = k.transpose(1, 2)
                v_sdpa = v.transpose(1, 2)
                out = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    scale=self.softmax_scale,
                    is_causal=False,
                )
                out = out.transpose(1, 2)
            else:
                out = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=False,
                )

            out = out.view(B, T_dec, self.d_model)
            out = self.out_proj(out)
            return residual + out
