/**
 * Polarized Mamba CUDA Kernels
 *
 * Fused forward pass for polarized channels:
 * - Zero channel (A=0): Pure local projection, no temporal dependency
 * - One channel (A=1): Cumulative sum for perfect memory
 * - Fusion: Combine learnable + zero + one outputs
 *
 * Reference: arXiv:2501.00658 (ICLR 2025)
 * "Polarized Attention via Controlled A-eigenvalue Channels"
 */

#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace align_mamba {

/**
 * Fused polarized channels forward pass
 *
 * Computes:
 *   y_zero = x @ zero_weight          (A=0: no memory)
 *   y_one = cumsum(x @ one_weight)    (A=1: perfect memory)
 *   out = [mamba_out || y_zero || y_one] @ fusion_weight
 *
 * @param x Normalized input (B, T, d_model)
 * @param mamba_out Output from Mamba learnable channel (B, T, d_inner)
 * @param zero_weight Zero channel projection (d_model, d_inner)
 * @param one_weight One channel projection (d_model, d_inner)
 * @param fusion_weight Fusion projection (d_inner * 3, d_model)
 * @return Fused output (B, T, d_model)
 */
torch::Tensor polarized_fwd(
    const torch::Tensor& x,
    const torch::Tensor& mamba_out,
    const torch::Tensor& zero_weight,
    const torch::Tensor& one_weight,
    const torch::Tensor& fusion_weight
);

/**
 * Backward pass for polarized channels
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
polarized_bwd(
    const torch::Tensor& grad_out,
    const torch::Tensor& x,
    const torch::Tensor& mamba_out,
    const torch::Tensor& zero_weight,
    const torch::Tensor& one_weight,
    const torch::Tensor& fusion_weight
);

}  // namespace align_mamba
