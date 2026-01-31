/**
 * MemMamba CUDA Kernels
 *
 * Cross-layer memory pool for long-range retrieval
 *
 * Reference: arXiv:2510.03279 - MemMamba
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "../include/memmamba.cuh"

namespace align_mamba {

/**
 * Fused importance scoring kernel
 *
 * x -> Linear(w1) -> ReLU -> Linear(w2) -> Sigmoid
 */
__global__ void importance_scoring_kernel(
    const float* __restrict__ x,           // (B, T, D)
    float* __restrict__ scores,            // (B, T)
    const float* __restrict__ w1,          // (D, D/4)
    const float* __restrict__ w2,          // (D/4, 1)
    int B, int T, int D, int hidden_dim
) {
    int b = blockIdx.x;
    int t = blockIdx.y;

    if (b >= B || t >= T) return;

    extern __shared__ float shared[];
    float* hidden = shared;

    int x_offset = (b * T + t) * D;

    // Linear1 + ReLU
    for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < D; d++) {
            sum += x[x_offset + d] * w1[d * hidden_dim + h];
        }
        hidden[h] = fmaxf(sum, 0.0f);  // ReLU
    }
    __syncthreads();

    // Linear2 + Sigmoid (single output)
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int h = 0; h < hidden_dim; h++) {
            sum += hidden[h] * w2[h];
        }
        scores[b * T + t] = 1.0f / (1.0f + expf(-sum));  // Sigmoid
    }
}

torch::Tensor importance_scoring(
    const torch::Tensor& x,
    const torch::Tensor& w1,
    const torch::Tensor& w2
) {
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");

    auto B = x.size(0);
    auto T = x.size(1);
    auto D = x.size(2);
    auto hidden_dim = w1.size(1);

    auto options = torch::TensorOptions()
        .dtype(x.dtype())
        .device(x.device());

    auto scores = torch::empty({B, T}, options);

    dim3 blocks(B, T);
    dim3 threads(min((int)hidden_dim, 256));
    size_t shared_mem = hidden_dim * sizeof(float);

    importance_scoring_kernel<<<blocks, threads, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        scores.data_ptr<float>(),
        w1.data_ptr<float>(),
        w2.data_ptr<float>(),
        B, T, D, hidden_dim
    );

    return scores;
}

/**
 * Memory pool update kernel
 *
 * Priority-based replacement: if new score > min priority, replace
 */
__global__ void memory_pool_update_kernel(
    const float* __restrict__ tokens,      // (N, D)
    const float* __restrict__ scores,      // (N,)
    const float* __restrict__ summarizer,  // (D, summary_dim)
    float* __restrict__ pool,              // (pool_size, summary_dim)
    float* __restrict__ priorities,        // (pool_size,)
    int* __restrict__ pool_count,
    int N, int D, int summary_dim, int pool_size, float threshold
) {
    // TODO: Implement parallel top-k for priority replacement
    // For now, simple sequential update

    for (int n = 0; n < N; n++) {
        if (scores[n] < threshold) continue;

        // Find minimum priority slot
        int min_slot = 0;
        float min_priority = priorities[0];
        for (int p = 1; p < pool_size; p++) {
            if (priorities[p] < min_priority) {
                min_priority = priorities[p];
                min_slot = p;
            }
        }

        // Replace if new score is higher
        if (scores[n] > min_priority) {
            priorities[min_slot] = scores[n];

            // Compress token to summary: pool[slot] = token @ summarizer
            for (int s = threadIdx.x; s < summary_dim; s += blockDim.x) {
                float sum = 0.0f;
                for (int d = 0; d < D; d++) {
                    sum += tokens[n * D + d] * summarizer[d * summary_dim + s];
                }
                pool[min_slot * summary_dim + s] = sum;
            }
        }
    }
}

void memory_pool_update(
    const torch::Tensor& tokens,
    const torch::Tensor& scores,
    const torch::Tensor& summarizer_weight,
    torch::Tensor& pool,
    torch::Tensor& priorities,
    float threshold
) {
    TORCH_CHECK(tokens.is_cuda(), "Tokens must be on CUDA");

    auto N = tokens.size(0);
    auto D = tokens.size(1);
    auto summary_dim = pool.size(1);
    auto pool_size = pool.size(0);

    auto pool_count = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));

    dim3 blocks(1);
    dim3 threads(min((int)summary_dim, 256));

    memory_pool_update_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        tokens.data_ptr<float>(),
        scores.data_ptr<float>(),
        summarizer_weight.data_ptr<float>(),
        pool.data_ptr<float>(),
        priorities.data_ptr<float>(),
        pool_count.data_ptr<int>(),
        N, D, summary_dim, pool_size, threshold
    );
}

/**
 * Cross-token retrieval kernel
 *
 * Attention-based retrieval from memory pool
 */
__global__ void cross_token_retrieval_kernel(
    const float* __restrict__ query,       // (B, T, D)
    const float* __restrict__ memory_pool, // (pool_size, summary_dim)
    const float* __restrict__ key_proj,    // (summary_dim, D)
    const float* __restrict__ value_proj,  // (summary_dim, D)
    float* __restrict__ retrieved,         // (B, T, D)
    int B, int T, int pool_size, int D, int summary_dim
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;

    if (b >= B || t >= T || d >= D) return;

    // TODO: Implement batched attention retrieval
    // For now, zero output
    retrieved[(b * T + t) * D + d] = 0.0f;
}

torch::Tensor cross_token_retrieval(
    const torch::Tensor& query,
    const torch::Tensor& memory_pool,
    const torch::Tensor& key_proj,
    const torch::Tensor& value_proj
) {
    TORCH_CHECK(query.is_cuda(), "Query must be on CUDA");

    auto B = query.size(0);
    auto T = query.size(1);
    auto D = query.size(2);
    auto pool_size = memory_pool.size(0);
    auto summary_dim = memory_pool.size(1);

    auto options = torch::TensorOptions()
        .dtype(query.dtype())
        .device(query.device());

    auto retrieved = torch::zeros({B, T, D}, options);

    dim3 blocks(B, T);
    dim3 threads(min((int)D, 256));

    cross_token_retrieval_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        query.data_ptr<float>(),
        memory_pool.data_ptr<float>(),
        key_proj.data_ptr<float>(),
        value_proj.data_ptr<float>(),
        retrieved.data_ptr<float>(),
        B, T, pool_size, D, summary_dim
    );

    return retrieved;
}

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
) {
    // Score importance
    auto scores = importance_scoring(mamba_out, scorer_w1, scorer_w2);

    // Flatten for memory update
    auto flat_tokens = mamba_out.reshape({-1, mamba_out.size(-1)});
    auto flat_scores = scores.reshape({-1});

    // Update memory pool
    memory_pool_update(flat_tokens, flat_scores, summarizer, memory_pool, priorities, tau1);

    // Retrieve from memory
    auto retrieved = cross_token_retrieval(mamba_out, memory_pool, key_proj, value_proj);

    // Combine: mamba_out + retrieved
    return mamba_out + retrieved;
}

}  // namespace align_mamba
