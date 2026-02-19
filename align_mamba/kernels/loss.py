"""Fused cross-entropy with label smoothing."""

import torch
import triton
import triton.language as tl


@triton.jit
def _ce_kernel(LOGITS, LABELS, LOSSES, stride, V, smoothing, ignore_idx, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    label = tl.load(LABELS + row)

    if label == ignore_idx:
        tl.store(LOSSES + row, 0.0)
        return

    m = float("-inf")
    d = 0.0
    target_logit = float("-inf")
    logit_sum = 0.0
    ptr = LOGITS + row * stride

    for start in range(0, V, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < V
        logits = tl.load(ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)

        is_target = (cols == label) & mask
        target_logit = tl.where(tl.max(is_target.to(tl.int32), axis=0) > 0,
                                tl.sum(tl.where(is_target, logits, 0.0), axis=0), target_logit)

        logit_sum += tl.sum(tl.where(mask, logits, 0.0), axis=0)

        block_max = tl.max(logits, axis=0)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(logits - m_new), axis=0)
        m = m_new

    log_sum_exp = m + tl.log(d)
    hard_loss = log_sum_exp - target_logit
    smooth_loss = log_sum_exp - logit_sum / tl.cast(V, tl.float32)
    loss = (1.0 - smoothing) * hard_loss + smoothing * smooth_loss

    tl.store(LOSSES + row, loss)


def fused_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    smoothing: float = 0.1,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy with label smoothing, mean-reduced over valid tokens."""
    B, T, V = logits.shape
    logits = logits.view(B * T, V).contiguous()
    labels = labels.view(B * T).contiguous()
    n = logits.shape[0]
    losses = torch.empty(n, device=logits.device, dtype=torch.float32)

    BLOCK = min(triton.next_power_of_2(V), 4096)
    _ce_kernel[(n,)](logits, labels, losses, logits.stride(0), V, smoothing, ignore_index, BLOCK=BLOCK)

    valid = (labels != ignore_index).sum().clamp(min=1)
    return losses.sum() / valid
