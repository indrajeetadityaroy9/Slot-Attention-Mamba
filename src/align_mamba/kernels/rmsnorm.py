import torch
import triton
import triton.language as tl


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


class _FusedRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        N = x.shape[-1]
        x_flat = x.contiguous().view(-1, N)
        rows = x_flat.shape[0]
        y = torch.empty_like(x_flat)

        BLOCK = triton.next_power_of_2(N)
        _rmsnorm_kernel[(rows,)](
            x_flat, y, weight, x_flat.stride(0), N, eps, BLOCK=BLOCK,
        )

        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return y.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        N = x.shape[-1]

        x_f = x.float()
        go_f = grad_output.float()
        w_f = weight.float()

        var = (x_f ** 2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)

        go_w = go_f * w_f
        c = (go_w * x_f).sum(dim=-1, keepdim=True) / (N * (var + eps))
        dx = rstd * (go_w - c * x_f)

        dw = (go_f * x_f * rstd).sum(dim=tuple(range(grad_output.ndim - 1)))

        return dx.to(x.dtype), dw.to(weight.dtype), None


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _FusedRMSNormFunc.apply(x, weight, eps)
