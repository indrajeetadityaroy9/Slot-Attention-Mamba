"""Shared utilities for attention mask handling."""

import torch


def broadcast_mask_for_sdpa(
    mask: torch.Tensor,
    batch_size: int,
    dtype: torch.dtype,
    src_len: int = None,
) -> torch.Tensor:
    """Convert padding mask to SDPA-compatible attention mask.

    Transforms a (batch, seq_len) boolean/float mask into the format expected
    by scaled_dot_product_attention: (batch, 1, 1, seq_len) with -inf for masked positions.
    """
    attn_mask = mask.unsqueeze(1).unsqueeze(2)
    assert attn_mask.dim() == 4
    assert attn_mask.shape[0] == batch_size
    if src_len is not None:
        assert attn_mask.shape[-1] == src_len
    attn_mask = attn_mask.to(dtype=dtype)
    return (1.0 - attn_mask) * torch.finfo(dtype).min
