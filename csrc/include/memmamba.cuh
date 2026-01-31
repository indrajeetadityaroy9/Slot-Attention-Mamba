/**
 * MemMamba CUDA Kernels
 *
 * Cross-layer memory pool for long-range retrieval
 *
 * Components:
 * - Token importance scoring: MLP-based scorer
 * - Memory pool update: Priority-based replacement
 * - Cross-token retrieval: Attention-based memory access
 *
 * Reference: arXiv:2510.03279 - MemMamba
 * "Memory-Augmented Mamba for Ultra-Long Sequences"
 */

#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace align_mamba {

/**
 * Token importance scoring
 *
 * Fused MLP: x -> Linear -> ReLU -> Linear -> Sigmoid
 *
 * @param x Input tensor (B, T, d_model)
 * @param w1 First layer weights (d_model, d_model/4)
 * @param w2 Second layer weights (d_model/4, 1)
 * @return Importance scores (B, T)
 */
torch::Tensor importance_scoring(
    const torch::Tensor& x,
    const torch::Tensor& w1,
    const torch::Tensor& w2
);

/**
 * Memory pool update with priority-based replacement
 *
 * @param tokens High-importance tokens to potentially add (N, d_model)
 * @param scores Importance scores (N,)
 * @param summarizer_weight Compression weights (d_model, summary_dim)
 * @param pool Current memory pool (pool_size, summary_dim) [in/out]
 * @param priorities Priority values for each slot (pool_size,) [in/out]
 * @param threshold Minimum score to consider for memory
 */
void memory_pool_update(
    const torch::Tensor& tokens,
    const torch::Tensor& scores,
    const torch::Tensor& summarizer_weight,
    torch::Tensor& pool,
    torch::Tensor& priorities,
    float threshold
);

/**
 * Cross-token retrieval from memory pool
 *
 * Attention-based retrieval:
 *   query = x @ query_proj
 *   key = pool @ key_proj
 *   value = pool @ value_proj
 *   retrieved = softmax(query @ key^T) @ value
 *
 * @param query Query tensor (B, T, d_model)
 * @param memory_pool Memory pool (pool_size, summary_dim)
 * @param key_proj Key projection (summary_dim, d_model)
 * @param value_proj Value projection (summary_dim, d_model)
 * @return Retrieved features (B, T, d_model)
 */
torch::Tensor cross_token_retrieval(
    const torch::Tensor& query,
    const torch::Tensor& memory_pool,
    const torch::Tensor& key_proj,
    const torch::Tensor& value_proj
);

/**
 * Fused MemMamba forward pass
 *
 * Combines scoring, memory update, and retrieval
 *
 * @param x Input tensor (B, T, d_model)
 * @param mamba_out Mamba layer output (B, T, d_model)
 * @param memory_pool Memory pool (pool_size, summary_dim) [in/out]
 * @param scorer_w1 Scorer first layer (d_model, d_model/4)
 * @param scorer_w2 Scorer second layer (d_model/4, 1)
 * @param summarizer Summarizer weights (d_model, summary_dim)
 * @param key_proj Key projection (summary_dim, d_model)
 * @param value_proj Value projection (summary_dim, d_model)
 * @param priorities Pool priorities (pool_size,) [in/out]
 * @param tau1 Threshold for memory insertion
 * @param tau2 Threshold for memory retrieval
 * @return Enhanced output (B, T, d_model)
 */
torch::Tensor memmamba_fwd(
    const torch::Tensor& x,
    const torch::Tensor& mamba_out,
    torch::Tensor& memory_pool,
    const torch::Tensor& scorer_w1,
    const torch::Tensor& scorer_w2,
    const torch::Tensor& summarizer,
    const torch::Tensor& key_proj,
    const torch::Tensor& value_proj,
    torch::Tensor& priorities,
    float tau1,
    float tau2
);

}  // namespace align_mamba
