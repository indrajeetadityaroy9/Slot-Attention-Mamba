"""KL-Guided Layer Selection for cross-attention placement.

Post-training refinement of attention layer placement using KL divergence
to measure layer importance. Higher KL when layer is disabled indicates
the layer is more critical for model performance.

Reference: arXiv:2512.20569
"Optimal Cross-Attention Placement via KL-Guided Selection"

Note: This technique is primarily designed for distillation from pretrained
models. For training from scratch, use compute_hybrid_positions_adaptive()
initially, then refine with KL-guided selection post-training.
"""

import math
from typing import Set, List, Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class LayerImportance:
    """Importance score for a single layer.

    Attributes:
        layer_idx: Index of the layer
        kl_divergence: KL divergence when layer is disabled
        importance_score: Normalized importance (higher = more important)
        should_keep: Whether to keep this layer's attention
    """
    layer_idx: int
    kl_divergence: float
    importance_score: float
    should_keep: bool


@dataclass
class KLSelectionResult:
    """Result of KL-guided layer selection.

    Attributes:
        selected_positions: Set of layer indices to keep cross-attention
        layer_scores: List of LayerImportance for each layer
        baseline_loss: Average loss with all layers active
        target_ratio: Target softmax layer ratio used
        actual_ratio: Actual ratio of selected layers
    """
    selected_positions: Set[int]
    layer_scores: List[LayerImportance]
    baseline_loss: float
    target_ratio: float
    actual_ratio: float


def _compute_output_distribution(
    model: nn.Module,
    input_ids: torch.Tensor,
    encoder_out: torch.Tensor,
    encoder_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute softmax distribution over vocabulary.

    Args:
        model: Decoder model
        input_ids: Decoder input token IDs
        encoder_out: Encoder output tensor
        encoder_mask: Optional encoder padding mask
        temperature: Softmax temperature for smoothing

    Returns:
        Probability distribution (B, T, vocab_size)
    """
    with torch.no_grad():
        logits = model(
            input_ids,
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_mask,
        )
        return F.softmax(logits / temperature, dim=-1)


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute KL divergence D_KL(p || q).

    Args:
        p: Target distribution
        q: Predicted distribution
        eps: Small constant for numerical stability

    Returns:
        Average KL divergence across batch and sequence
    """
    kl = p * (torch.log(p + eps) - torch.log(q + eps))
    return kl.sum(dim=-1).mean().item()


def measure_layer_importance(
    model: nn.Module,
    dataloader: DataLoader,
    attention_layers: Set[int],
    num_samples: int = 1000,
    temperature: float = 2.0,
    device: Optional[torch.device] = None,
) -> List[LayerImportance]:
    """Measure importance of each attention layer via KL divergence.

    For each layer, compute KL divergence between:
    - Full model output (all attention layers active)
    - Model output with that specific layer's attention disabled

    Higher KL = layer is more important for output distribution.

    Args:
        model: HybridMambaDecoder model
        dataloader: DataLoader yielding (input_ids, encoder_out, encoder_mask)
        attention_layers: Set of layer indices currently using attention
        num_samples: Number of samples for KL estimation
        temperature: Softmax temperature for smoothing
        device: Device for computation

    Returns:
        List of LayerImportance, sorted by layer index
    """
    model.eval()
    device = device or next(model.parameters()).device

    # Store original hybrid positions
    original_positions = model.hybrid_positions.copy()

    layer_scores = []

    samples_seen = 0
    data_iter = iter(dataloader)

    for layer_idx in sorted(attention_layers):
        # Collect samples for this layer's KL measurement
        kl_sum = 0.0
        count = 0

        # Reset data iterator for each layer
        data_iter = iter(dataloader)
        samples_seen = 0

        while samples_seen < num_samples:
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            input_ids = batch["decoder_input_ids"].to(device)
            encoder_out = batch["encoder_out"].to(device)
            encoder_mask = batch.get("encoder_mask")
            if encoder_mask is not None:
                encoder_mask = encoder_mask.to(device)

            batch_size = input_ids.size(0)

            # Baseline: all layers active
            model.hybrid_positions = original_positions.copy()
            p_full = _compute_output_distribution(
                model, input_ids, encoder_out, encoder_mask, temperature
            )

            # Ablated: disable this specific layer
            ablated_positions = original_positions - {layer_idx}
            model.hybrid_positions = ablated_positions
            q_ablated = _compute_output_distribution(
                model, input_ids, encoder_out, encoder_mask, temperature
            )

            # Compute KL divergence
            kl = _kl_divergence(p_full, q_ablated)
            kl_sum += kl * batch_size
            count += batch_size
            samples_seen += batch_size

        # Average KL for this layer
        avg_kl = kl_sum / max(count, 1)
        layer_scores.append(LayerImportance(
            layer_idx=layer_idx,
            kl_divergence=avg_kl,
            importance_score=0.0,  # Normalized later
            should_keep=True,  # Determined later
        ))

    # Restore original positions
    model.hybrid_positions = original_positions

    # Normalize importance scores
    if layer_scores:
        max_kl = max(s.kl_divergence for s in layer_scores)
        min_kl = min(s.kl_divergence for s in layer_scores)
        kl_range = max_kl - min_kl + 1e-8

        for score in layer_scores:
            score.importance_score = (score.kl_divergence - min_kl) / kl_range

    return layer_scores


def compute_hybrid_positions_kl_guided(
    model: nn.Module,
    dataloader: DataLoader,
    target_ratio: float = 0.125,
    num_samples: int = 10000,
    temperature: float = 2.0,
    always_keep_layer_0: bool = True,
    device: Optional[torch.device] = None,
) -> KLSelectionResult:
    """Select cross-attention layers using KL-guided importance.

    Algorithm:
    1. Measure KL divergence for each existing attention layer
    2. Rank layers by importance (higher KL = more important)
    3. Keep top-K layers where K = target_ratio * n_layers

    Args:
        model: HybridMambaDecoder with attention layers
        dataloader: DataLoader for calibration samples
        target_ratio: Target ratio of attention layers (default 0.125 = 1:8)
        num_samples: Number of calibration samples
        temperature: Softmax temperature for KL computation
        always_keep_layer_0: Always keep Layer 0 (Blind Start fix)
        device: Device for computation

    Returns:
        KLSelectionResult with selected positions and layer scores
    """
    model.eval()
    device = device or next(model.parameters()).device

    # Get current attention layers
    attention_layers = model.hybrid_positions.copy()
    n_layers = model.n_layers

    # Measure layer importance
    layer_scores = measure_layer_importance(
        model, dataloader, attention_layers,
        num_samples=num_samples,
        temperature=temperature,
        device=device,
    )

    # Compute baseline loss
    baseline_loss = 0.0
    samples = 0
    for batch in dataloader:
        if samples >= num_samples // 10:
            break
        input_ids = batch["decoder_input_ids"].to(device)
        encoder_out = batch["encoder_out"].to(device)
        labels = batch.get("labels", input_ids[:, 1:]).to(device)

        with torch.no_grad():
            logits = model(input_ids, encoder_out=encoder_out)
            logits_flat = logits[:, :-1].reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
            baseline_loss += loss.item() * input_ids.size(0)
            samples += input_ids.size(0)

    baseline_loss /= max(samples, 1)

    # Select top-K layers by importance
    target_count = max(1, int(n_layers * target_ratio))

    # Sort by importance (descending)
    sorted_scores = sorted(layer_scores, key=lambda x: x.importance_score, reverse=True)

    selected = set()

    # Always keep layer 0 if specified
    if always_keep_layer_0 and 0 in attention_layers:
        selected.add(0)

    # Add remaining layers by importance
    for score in sorted_scores:
        if len(selected) >= target_count:
            break
        if score.layer_idx not in selected:
            selected.add(score.layer_idx)

    # Mark which layers to keep
    for score in layer_scores:
        score.should_keep = score.layer_idx in selected

    actual_ratio = len(selected) / n_layers

    return KLSelectionResult(
        selected_positions=selected,
        layer_scores=layer_scores,
        baseline_loss=baseline_loss,
        target_ratio=target_ratio,
        actual_ratio=actual_ratio,
    )


def format_selection_report(result: KLSelectionResult) -> str:
    """Format KL selection result as human-readable report.

    Args:
        result: KLSelectionResult from compute_hybrid_positions_kl_guided

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "KL-Guided Layer Selection Report",
        "=" * 60,
        f"Target Ratio: {result.target_ratio:.1%}",
        f"Actual Ratio: {result.actual_ratio:.1%}",
        f"Baseline Loss: {result.baseline_loss:.4f}",
        "",
        "--- Layer Importance Scores ---",
    ]

    for score in sorted(result.layer_scores, key=lambda x: x.layer_idx):
        status = "KEEP" if score.should_keep else "DROP"
        lines.append(
            f"  Layer {score.layer_idx:2d}: "
            f"KL={score.kl_divergence:.4f}, "
            f"Importance={score.importance_score:.2f}, "
            f"[{status}]"
        )

    lines.extend([
        "",
        "--- Selected Positions ---",
        f"  {sorted(result.selected_positions)}",
        "=" * 60,
    ])

    return "\n".join(lines)


__all__ = [
    "LayerImportance",
    "KLSelectionResult",
    "measure_layer_importance",
    "compute_hybrid_positions_kl_guided",
    "format_selection_report",
]
