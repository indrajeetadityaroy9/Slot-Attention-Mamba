/**
 * BASED Linear Attention CUDA Kernels
 *
 * 2nd-order Taylor feature map + causal linear attention
 *
 * Reference: arXiv:2402.18668 (Stanford 2024) - BASED
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "../include/based_attention.cuh"

namespace align_mamba {

/**
 * Taylor feature map kernel
 *
 * phi(x) = [1, x, x_i*x_j/sqrt(2) for i<=j]
 *
 * For d' features, output dimension is 1 + d' + d'(d'+1)/2
 */
__global__ void taylor_feature_map_kernel(
    const float* __restrict__ x,    // (B, T, n_heads, d')
    float* __restrict__ phi_x,       // (B, T, n_heads, expanded_dim)
    int B, int T, int n_heads, int feature_dim, int expanded_dim
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int h = threadIdx.x;

    if (b >= B || t >= T || h >= n_heads) return;

    int x_offset = ((b * T + t) * n_heads + h) * feature_dim;
    int phi_offset = ((b * T + t) * n_heads + h) * expanded_dim;

    // Constant term: 1
    phi_x[phi_offset] = 1.0f;

    // Linear terms: x
    for (int i = 0; i < feature_dim; i++) {
        phi_x[phi_offset + 1 + i] = x[x_offset + i];
    }

    // Quadratic terms: x_i * x_j / sqrt(2) for i <= j
    int quad_idx = 1 + feature_dim;
    float scale = 0.7071067811865476f;  // 1/sqrt(2)
    for (int i = 0; i < feature_dim; i++) {
        for (int j = i; j < feature_dim; j++) {
            float scale_ij = (i == j) ? 0.5f : scale;
            phi_x[phi_offset + quad_idx] = x[x_offset + i] * x[x_offset + j] * scale_ij;
            quad_idx++;
        }
    }
}

torch::Tensor taylor_feature_map(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (B, T, n_heads, d')");

    auto B = x.size(0);
    auto T = x.size(1);
    auto n_heads = x.size(2);
    auto feature_dim = x.size(3);

    // Output dimension: 1 + d' + d'(d'+1)/2
    int expanded_dim = 1 + feature_dim + feature_dim * (feature_dim + 1) / 2;

    auto options = torch::TensorOptions()
        .dtype(x.dtype())
        .device(x.device());

    auto phi_x = torch::empty({B, T, n_heads, expanded_dim}, options);

    dim3 blocks(B, T);
    dim3 threads(min((int)n_heads, 1024));

    taylor_feature_map_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        phi_x.data_ptr<float>(),
        B, T, n_heads, feature_dim, expanded_dim
    );

    return phi_x;
}

/**
 * Causal linear attention kernel
 *
 * Uses associative scan for efficient cumulative sum of K^T @ V
 */
__global__ void linear_attention_causal_kernel(
    const float* __restrict__ q_feat,  // (B, T, n_heads, expanded_dim)
    const float* __restrict__ k_feat,  // (B, T, n_heads, expanded_dim)
    const float* __restrict__ v,       // (B, T, n_heads, head_dim)
    float* __restrict__ kv_state,      // (B, n_heads, expanded_dim, head_dim)
    float* __restrict__ k_state,       // (B, n_heads, expanded_dim)
    float* __restrict__ out,           // (B, T, n_heads, head_dim)
    int B, int T, int n_heads, int expanded_dim, int head_dim
) {
    // TODO: Implement with warp-level associative scan
    // kv_state[t] = kv_state[t-1] + outer(k[t], v[t])
    // out[t] = (q[t] @ kv_state[t]) / (q[t] @ k_state[t])

    int b = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;

    if (b >= B || h >= n_heads || d >= head_dim) return;

    // Placeholder: copy v to out
    for (int t = 0; t < T; t++) {
        int out_idx = ((b * T + t) * n_heads + h) * head_dim + d;
        int v_idx = ((b * T + t) * n_heads + h) * head_dim + d;
        out[out_idx] = v[v_idx];
    }
}

torch::Tensor linear_attention_causal(
    const torch::Tensor& q_feat,
    const torch::Tensor& k_feat,
    const torch::Tensor& v
) {
    TORCH_CHECK(q_feat.is_cuda(), "q_feat must be on CUDA");

    auto B = q_feat.size(0);
    auto T = q_feat.size(1);
    auto n_heads = q_feat.size(2);
    auto expanded_dim = q_feat.size(3);
    auto head_dim = v.size(3);

    auto options = torch::TensorOptions()
        .dtype(q_feat.dtype())
        .device(q_feat.device());

    // Allocate state buffers
    auto kv_state = torch::zeros({B, n_heads, expanded_dim, head_dim}, options);
    auto k_state = torch::zeros({B, n_heads, expanded_dim}, options);
    auto out = torch::empty({B, T, n_heads, head_dim}, options);

    dim3 blocks(B, n_heads);
    dim3 threads(min((int)head_dim, 1024));

    linear_attention_causal_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        q_feat.data_ptr<float>(),
        k_feat.data_ptr<float>(),
        v.data_ptr<float>(),
        kv_state.data_ptr<float>(),
        k_state.data_ptr<float>(),
        out.data_ptr<float>(),
        B, T, n_heads, expanded_dim, head_dim
    );

    return out;
}

torch::Tensor based_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int window_size,
    int feature_dim
) {
    // TODO: Combine linear + sliding window
    // For now, just return linear attention output
    auto q_feat = taylor_feature_map(q);
    auto k_feat = taylor_feature_map(k);
    return linear_attention_causal(q_feat, k_feat, v);
}

}  // namespace align_mamba
