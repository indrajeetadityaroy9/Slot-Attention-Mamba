import torch
import triton
import triton.language as tl

_RMSNORM_CHUNK = 8192


@triton.jit
def _rmsnorm_kernel(X, Y, W, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0)

    var = tl.sum(x * x, axis=0) / N
    y = x * tl.rsqrt(var + eps) * w.to(tl.float32)

    tl.store(Y + row * stride + cols, y.to(x.dtype), mask=mask)


@triton.jit
def _var_chunk_kernel(X, VAR, stride, N, start, size, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    actual = start + cols
    mask = (cols < size) & (actual < N)

    x = tl.load(X + row * stride + actual, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_add(VAR + row, tl.sum(x * x, axis=0))


@triton.jit
def _norm_chunk_kernel(X, W, Y, VAR, stride, N, eps, start, size, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    actual = start + cols
    mask = (cols < size) & (actual < N)

    var = tl.load(VAR + row)
    rstd = tl.rsqrt(var / N + eps)

    x = tl.load(X + row * stride + actual, mask=mask, other=0.0)
    w = tl.load(W + actual, mask=mask, other=0.0)
    y = x.to(tl.float32) * rstd * w.to(tl.float32)

    tl.store(Y + row * stride + actual, y.to(x.dtype), mask=mask)


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    N = x.shape[-1]
    x_flat = x.contiguous().view(-1, N)
    rows = x_flat.shape[0]
    y = torch.empty_like(x_flat)

    if N <= _RMSNORM_CHUNK:
        BLOCK = triton.next_power_of_2(N)
        _rmsnorm_kernel[(rows,)](
            x_flat,
            y,
            weight,
            x_flat.stride(0),
            N,
            eps,
            BLOCK=min(BLOCK, _RMSNORM_CHUNK),
        )
    else:
        # Chunked path keeps memory bounded for very wide hidden sizes.
        var = torch.zeros(rows, device=x.device, dtype=torch.float32)
        BLOCK = _RMSNORM_CHUNK
        for start in range(0, N, BLOCK):
            size = min(BLOCK, N - start)
            _var_chunk_kernel[(rows,)](x_flat, var, x_flat.stride(0), N, start, size, BLOCK=BLOCK)
        for start in range(0, N, BLOCK):
            size = min(BLOCK, N - start)
            _norm_chunk_kernel[(rows,)](x_flat, weight, y, var, x_flat.stride(0), N, eps, start, size, BLOCK=BLOCK)

    return y.view_as(x)
