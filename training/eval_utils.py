"""Evaluation utilities for MQAR experiments."""

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_batch_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[int, int, int]:
    """Compute token and sample accuracy for a batch.

    Returns:
        Tuple of (token_correct, token_total, sample_correct)
    """
    token_correct = ((predictions == labels) & mask).sum().item()
    token_total = mask.sum().item()
    sample_correct = ((predictions == labels) | ~mask).all(dim=-1).sum().item()
    return token_correct, token_total, sample_correct


def compute_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute perplexity (exp of cross-entropy loss)."""
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction='mean')
    return torch.exp(loss).item()
