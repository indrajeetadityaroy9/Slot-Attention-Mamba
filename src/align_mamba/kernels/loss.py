import torch
import triton
import triton.language as tl

_MAX_CE_BLOCK = 4096


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
        target_logit = tl.where(
            tl.max(is_target, axis=0) > 0,
            tl.sum(tl.where(is_target, logits, 0.0), axis=0),
            target_logit,
        )

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


class _FusedCEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing, ignore_index):
        B, T, V = logits.shape
        logits_flat = logits.view(B * T, V).contiguous()
        labels_flat = labels.view(B * T).contiguous()
        n = logits_flat.shape[0]
        losses = torch.empty(n, device=logits.device, dtype=torch.float32)

        BLOCK = min(triton.next_power_of_2(V), _MAX_CE_BLOCK)
        _ce_kernel[(n,)](
            logits_flat, labels_flat, losses, logits_flat.stride(0),
            V, smoothing, ignore_index, BLOCK=BLOCK,
        )

        valid = (labels_flat != ignore_index).sum()
        loss = losses.sum() / valid

        ctx.save_for_backward(logits, labels)
        ctx.smoothing = smoothing
        ctx.ignore_index = ignore_index
        ctx.valid = valid
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels = ctx.saved_tensors
        B, T, V = logits.shape
        smoothing = ctx.smoothing
        ignore_index = ctx.ignore_index
        valid = ctx.valid

        logits_flat = logits.view(B * T, V).float()
        labels_flat = labels.view(B * T)

        # dL/d(logit_j) = softmax_j - (1-α)*δ_{j=label} - α/V  (per valid row)
        probs = torch.softmax(logits_flat, dim=-1)
        mask = labels_flat != ignore_index
        valid_rows = mask.nonzero(as_tuple=True)[0]

        probs[valid_rows, labels_flat[valid_rows]] -= (1.0 - smoothing)
        probs[mask] -= smoothing / V
        probs[~mask] = 0.0

        grad = probs * (grad_output / valid)
        return grad.view(B, T, V).to(logits.dtype), None, None, None


def fused_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    smoothing: float,
    ignore_index: int,
) -> torch.Tensor:
    return _FusedCEFunc.apply(logits, labels, smoothing, ignore_index)
