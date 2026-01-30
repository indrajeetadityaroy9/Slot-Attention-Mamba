"""
Embeddings for Align-Mamba models.

Contains:
- ScaledEmbedding: Token embedding with learnable scale and dropout (shared by encoder/decoder)
- RotaryPositionalEmbedding (RoPE): Applied ONLY to attention layers, NOT Mamba.
  Mamba encodes position implicitly through recurrence.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class ScaledEmbedding(nn.Module):
    """Token embedding with learnable scale and dropout.

    Consolidates the embedding pattern used by both HybridBiMambaEncoder and
    HybridMambaDecoder to eliminate code duplication.

    The scale is initialized to sqrt(d_model) following the Transformer convention,
    but is learnable to allow the model to adjust embedding magnitudes during training.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = 0,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Embedding dimension
            padding_idx: Index of padding token (default 0)
            dropout: Dropout probability (default 0.1)
            device: Device for parameters
            dtype: Dtype for forward pass (embeddings are always float32, cast in forward)
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx, device=device)
        self.embed_scale = nn.Parameter(torch.tensor(math.sqrt(d_model)))
        self.embed_dropout = nn.Dropout(dropout)
        self.dtype = dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token indices of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, d_model)
        """
        x = self.embed(input_ids) * self.embed_scale
        if self.dtype is not None:
            x = x.to(self.dtype)
        return self.embed_dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding for attention layers.

    Applies rotation to Q and K tensors to encode relative positions.
    Supports dynamic cache resizing for length extrapolation.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            dim: Head dimension (d_model // n_heads)
            max_seq_len: Initial maximum sequence length to cache
            base: Base for frequency computation
            device: Device for cached tensors
        """
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Pre-compute cos/sin caches for efficiency."""
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=device or self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _extend_cache(self, seq_len: int):
        """Dynamically extend cache for longer sequences (length extrapolation)."""
        if seq_len > self.max_seq_len:
            new_max = max(seq_len, self.max_seq_len * 2)
            self._build_cache(new_max, device=self.inv_freq.device)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to Q and K.

        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim)
            k: Key tensor (batch, n_heads, seq_len, head_dim)
            offset: Position offset for incremental decoding

        Returns:
            Tuple of rotated (q, k)
        """
        seq_len = q.size(2)
        required_len = offset + seq_len
        if required_len > self.max_seq_len:
            self._extend_cache(required_len)

        cos = self.cos_cache[offset : offset + seq_len].unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin = self.sin_cache[offset : offset + seq_len].unsqueeze(0).unsqueeze(0).to(q.dtype)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot

    def apply_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_offset: int = 0,
        k_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings with separate offsets for Q and K.
        Useful for cross-attention where Q and K may have different lengths.
        """
        seq_len_q = q.size(2)
        seq_len_k = k.size(2)

        required_len = max(q_offset + seq_len_q, k_offset + seq_len_k)
        if required_len > self.max_seq_len:
            self._extend_cache(required_len)

        cos_q = self.cos_cache[q_offset : q_offset + seq_len_q].unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin_q = self.sin_cache[q_offset : q_offset + seq_len_q].unsqueeze(0).unsqueeze(0).to(q.dtype)
        cos_k = self.cos_cache[k_offset : k_offset + seq_len_k].unsqueeze(0).unsqueeze(0).to(k.dtype)
        sin_k = self.sin_cache[k_offset : k_offset + seq_len_k].unsqueeze(0).unsqueeze(0).to(k.dtype)

        q_rot = (q * cos_q) + (self._rotate_half(q) * sin_q)
        k_rot = (k * cos_k) + (self._rotate_half(k) * sin_k)

        return q_rot, k_rot
