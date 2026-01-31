/**
 * HGRN2 State Expansion CUDA Kernels
 *
 * Outer product state expansion for increased capacity: d -> d^2
 *
 * State update equation:
 *   h_t = Diag{f_t} @ h_{t-1} + (1-f_t) outer i_t @ v_t
 *   y_t = o_t @ h_t
 *
 * Where:
 *   f_t = clamp(sigmoid(x @ forget_proj), min=lower_bound)
 *   i_t = 1 - f_t  (tied input gate)
 *   v_t = x @ input_proj
 *   o_t = sigmoid(x @ output_proj)
 *
 * Reference: arXiv:2404.07904 (COLM 2024) - HGRN2
 * "Linear Attention with Forget Gates for Language Modeling"
 */

#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace align_mamba {

/**
 * State expansion forward pass
 *
 * @param x Input tensor (B, T, d_model)
 * @param forget_proj Forget gate projection (d_model, d_model)
 * @param input_proj Input projection (d_model, d_model)
 * @param output_proj Output projection (d_model, d_model)
 * @param forget_lower_bound Minimum value for forget gate (prevents vanishing)
 * @param n_heads Number of attention heads
 * @param head_dim Dimension per head
 * @return Output tensor (B, T, d_model)
 */
torch::Tensor state_expansion_fwd(
    const torch::Tensor& x,
    const torch::Tensor& forget_proj,
    const torch::Tensor& input_proj,
    const torch::Tensor& output_proj,
    float forget_lower_bound,
    int n_heads,
    int head_dim
);

/**
 * State expansion backward pass
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
state_expansion_bwd(
    const torch::Tensor& grad_out,
    const torch::Tensor& x,
    const torch::Tensor& forget_proj,
    const torch::Tensor& input_proj,
    const torch::Tensor& output_proj,
    float forget_lower_bound,
    int n_heads,
    int head_dim
);

}  // namespace align_mamba
