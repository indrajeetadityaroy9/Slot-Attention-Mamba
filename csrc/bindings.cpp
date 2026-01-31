/**
 * PyTorch C++ Extension Bindings for Align-Mamba CUDA Kernels
 *
 * Exposes the following modules:
 * - polarized_fwd/bwd: Polarized Mamba channels
 * - state_expansion_fwd/bwd: HGRN2 state expansion
 * - taylor_feature_map, linear_attention_causal: BASED attention
 * - importance_scoring, memory_pool_update, cross_token_retrieval: MemMamba
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "include/polarized_mamba.cuh"
#include "include/state_expansion.cuh"
#include "include/based_attention.cuh"
#include "include/memmamba.cuh"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Align-Mamba CUDA Kernels";

    // Polarization (arXiv:2501.00658)
    m.def("polarized_fwd", &align_mamba::polarized_fwd,
          "Polarized Mamba forward pass",
          py::arg("x"), py::arg("mamba_out"),
          py::arg("zero_weight"), py::arg("one_weight"),
          py::arg("fusion_weight"));

    m.def("polarized_bwd", &align_mamba::polarized_bwd,
          "Polarized Mamba backward pass",
          py::arg("grad_out"), py::arg("x"), py::arg("mamba_out"),
          py::arg("zero_weight"), py::arg("one_weight"),
          py::arg("fusion_weight"));

    // State Expansion (arXiv:2404.07904)
    m.def("state_expansion_fwd", &align_mamba::state_expansion_fwd,
          "HGRN2 state expansion forward pass",
          py::arg("x"), py::arg("forget_proj"),
          py::arg("input_proj"), py::arg("output_proj"),
          py::arg("forget_lower_bound"),
          py::arg("n_heads"), py::arg("head_dim"));

    m.def("state_expansion_bwd", &align_mamba::state_expansion_bwd,
          "HGRN2 state expansion backward pass",
          py::arg("grad_out"), py::arg("x"),
          py::arg("forget_proj"), py::arg("input_proj"), py::arg("output_proj"),
          py::arg("forget_lower_bound"),
          py::arg("n_heads"), py::arg("head_dim"));

    // BASED Attention (arXiv:2402.18668)
    m.def("taylor_feature_map", &align_mamba::taylor_feature_map,
          "2nd-order Taylor feature map",
          py::arg("x"));

    m.def("linear_attention_causal", &align_mamba::linear_attention_causal,
          "Causal linear attention with associative scan",
          py::arg("q_feat"), py::arg("k_feat"), py::arg("v"));

    m.def("based_attention_fwd", &align_mamba::based_attention_fwd,
          "Combined BASED attention forward pass",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("window_size"), py::arg("feature_dim"));

    // MemMamba (arXiv:2510.03279)
    m.def("importance_scoring", &align_mamba::importance_scoring,
          "Token importance scoring",
          py::arg("x"), py::arg("w1"), py::arg("w2"));

    m.def("memory_pool_update", &align_mamba::memory_pool_update,
          "Memory pool update with priority replacement",
          py::arg("tokens"), py::arg("scores"),
          py::arg("summarizer_weight"),
          py::arg("pool"), py::arg("priorities"),
          py::arg("threshold"));

    m.def("cross_token_retrieval", &align_mamba::cross_token_retrieval,
          "Cross-token retrieval from memory pool",
          py::arg("query"), py::arg("memory_pool"),
          py::arg("key_proj"), py::arg("value_proj"));

    m.def("memmamba_fwd", &align_mamba::memmamba_fwd,
          "Fused MemMamba forward pass",
          py::arg("x"), py::arg("mamba_out"),
          py::arg("memory_pool"),
          py::arg("scorer_w1"), py::arg("scorer_w2"),
          py::arg("summarizer"),
          py::arg("key_proj"), py::arg("value_proj"),
          py::arg("priorities"),
          py::arg("tau1"), py::arg("tau2"));
}
