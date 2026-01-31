/**
 * BASED Linear Attention CUDA Kernels
 *
 * 2nd-order Taylor feature map + sliding window attention
 *
 * Taylor feature map: phi(x) = [1, x, x_i*x_j/sqrt(2) for i<=j]
 * Linear attention: y = Q @ (K^T @ V) / (Q @ sum(K))
 *
 * Combined with FlashAttention2 sliding window for local context.
 *
 * Reference: arXiv:2402.18668 (Stanford 2024) - BASED
 * "Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff"
 */

#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace align_mamba {

/**
 * Compute Taylor feature map
 *
 * phi(x) = [1, x, x^2 / sqrt(2)]
 * where x^2 terms are x_i * x_j / sqrt(2) for i <= j
 *
 * @param x Input tensor (B, T, n_heads, d')
 * @return Feature-mapped tensor (B, T, n_heads, 1 + d' + d'(d'+1)/2)
 */
torch::Tensor taylor_feature_map(const torch::Tensor& x);

/**
 * Causal linear attention forward pass
 *
 * Uses associative scan for efficient cumulative sum of K^T @ V
 *
 * @param q_feat Query after Taylor map (B, T, n_heads, expanded_dim)
 * @param k_feat Key after Taylor map (B, T, n_heads, expanded_dim)
 * @param v Value tensor (B, T, n_heads, head_dim)
 * @return Attention output (B, T, n_heads, head_dim)
 */
torch::Tensor linear_attention_causal(
    const torch::Tensor& q_feat,
    const torch::Tensor& k_feat,
    const torch::Tensor& v
);

/**
 * Combined BASED attention forward pass
 *
 * Combines:
 * - Taylor linear attention (global context)
 * - Sliding window attention (local context, uses FlashAttention2)
 *
 * @param q Query tensor (B, T, n_heads, head_dim)
 * @param k Key tensor (B, T, n_heads, head_dim)
 * @param v Value tensor (B, T, n_heads, head_dim)
 * @param window_size Sliding window size
 * @param feature_dim Taylor feature dimension
 * @return Combined attention output (B, T, n_heads, head_dim * 2)
 */
torch::Tensor based_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int window_size,
    int feature_dim
);

}  // namespace align_mamba
