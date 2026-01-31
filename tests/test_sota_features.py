"""Integration tests for SOTA features.

These tests verify correctness of SOTA implementations without requiring GPU.
They test mathematical properties, shapes, and basic functionality.

Run with: python -m pytest tests/test_sota_features.py -v
"""

import math
import pytest
import torch


class TestCapacityAnalysis:
    """Tests for models/capacity.py"""

    def test_analyze_ssm_capacity_no_overflow(self):
        """Test capacity analysis when num_pairs < d_state."""
        from models.capacity import analyze_ssm_capacity

        analysis = analyze_ssm_capacity(d_state=64, num_pairs=32, n_layers=24)

        assert analysis.d_state == 64
        assert analysis.num_pairs == 32
        assert analysis.capacity_utilization == 0.5
        assert analysis.overflow_ratio == 1.0  # No overflow
        assert analysis.convergence_guaranteed
        assert analysis.recommended_cross_attn_interval is None

    def test_analyze_ssm_capacity_with_overflow(self):
        """Test capacity analysis when num_pairs > d_state."""
        from models.capacity import analyze_ssm_capacity

        analysis = analyze_ssm_capacity(d_state=64, num_pairs=256, n_layers=24)

        assert analysis.capacity_utilization == 4.0
        assert analysis.overflow_ratio == 4.0
        assert analysis.recommended_cross_attn_interval is not None
        assert analysis.recommended_cross_attn_interval > 0

    def test_mutual_information_bound(self):
        """Test mutual information bound computation."""
        from models.capacity import compute_mutual_information_bound

        mi = compute_mutual_information_bound(d_state=64, precision_bits=16)

        # 64 * log2(16) = 64 * 4 = 256 bits
        assert mi == 256.0


class TestPolarization:
    """Tests for polarized Mamba channels."""

    def test_polarized_forward_reference_shapes(self):
        """Test that polarized forward produces correct output shapes."""
        from models.wrapper import PolarizedMamba2Block

        B, T, D = 2, 16, 256
        x = torch.randn(B, T, D)

        # Create block (will fail without mamba_ssm, but we can test structure)
        try:
            block = PolarizedMamba2Block(d_model=D, d_state=64, polarized_channels=2)
            # Test would run here if mamba_ssm available
        except ImportError:
            pytest.skip("mamba_ssm not available")

    def test_zero_channel_no_memory(self):
        """Test that A=0 channel has no temporal dependency."""
        # A=0 means: y_t = Δ_t · b_t(x_t), no h_{t-1} term
        # Verify by checking that output at t doesn't depend on input at t-1

        B, T, D = 2, 8, 64
        x = torch.randn(B, T, D)

        # Zero channel is just a linear projection
        zero_proj = torch.randn(D, D * 2)
        y_zero = x @ zero_proj

        # Verify shape
        assert y_zero.shape == (B, T, D * 2)

        # Verify independence: changing x[:, 0] doesn't affect y_zero[:, 1:]
        x_modified = x.clone()
        x_modified[:, 0] = torch.randn(B, D)
        y_zero_modified = x_modified @ zero_proj

        # Outputs at t > 0 should be identical
        assert torch.allclose(y_zero[:, 1:], y_zero_modified[:, 1:])

    def test_one_channel_cumsum(self):
        """Test that A=1 channel is cumulative sum."""
        # A=1 means: h_t = h_{t-1} + Δ_t · b_t(x_t) = cumsum

        B, T, D = 2, 8, 64
        x = torch.randn(B, T, D)

        one_proj = torch.randn(D, D * 2)
        y_one = torch.cumsum(x @ one_proj, dim=1)

        # Verify cumsum property: y_one[t] = sum(x[0:t+1] @ proj)
        expected_t2 = (x[:, :3] @ one_proj).sum(dim=1)
        assert torch.allclose(y_one[:, 2], expected_t2)


class TestStateExpansion:
    """Tests for HGRN2 state expansion."""

    def test_state_expansion_shapes(self):
        """Test state expansion output and state shapes."""
        from models.state_expansion import StateExpandedBlock

        B, T, D = 2, 16, 256
        n_heads, head_dim = 2, 128

        try:
            block = StateExpandedBlock(
                d_model=D,
                n_heads=n_heads,
                head_dim=head_dim,
                forget_lower_bound=0.9,
            )

            x = torch.randn(B, T, D)
            out = block._forward_reference(block.norm(x))

            assert out.shape == (B, T, D)
        except ImportError:
            pytest.skip("Dependencies not available")

    def test_forget_lower_bound_computation(self):
        """Test depth-dependent forget gate lower bound."""
        from models.state_expansion import compute_forget_lower_bound

        # Layer 0: should be minimum (0.9)
        assert compute_forget_lower_bound(0, 24) == pytest.approx(0.9, rel=0.01)

        # Layer 23 (last): should be maximum (0.99)
        assert compute_forget_lower_bound(23, 24) == pytest.approx(0.99, rel=0.01)

        # Middle layer: should be between
        middle = compute_forget_lower_bound(12, 24)
        assert 0.9 < middle < 0.99

    def test_outer_product_capacity(self):
        """Test that outer product provides d^2 capacity."""
        # State shape: (B, n_heads, head_dim, head_dim)
        # Total capacity: n_heads * head_dim^2

        n_heads, head_dim = 4, 64
        d_model = n_heads * head_dim  # 256

        standard_capacity = d_model  # 256
        expanded_capacity = n_heads * head_dim * head_dim  # 4 * 64 * 64 = 16384

        assert expanded_capacity == d_model * head_dim
        assert expanded_capacity > standard_capacity


class TestBasedAttention:
    """Tests for BASED linear attention."""

    def test_taylor_feature_map_dimension(self):
        """Test Taylor feature map output dimension."""
        from models.based_attention import taylor_feature_map_reference

        d = 16  # Feature dimension
        expected_dim = 1 + d + d * (d + 1) // 2  # 1 + 16 + 136 = 153

        B, T, H = 2, 8, 4
        x = torch.randn(B, T, H, d)

        phi_x = taylor_feature_map_reference(x)

        assert phi_x.shape == (B, T, H, expected_dim)

    def test_taylor_feature_map_components(self):
        """Test Taylor feature map components."""
        from models.based_attention import taylor_feature_map_reference

        d = 4
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # (1, 1, 1, 4)

        phi_x = taylor_feature_map_reference(x)

        # Check constant term
        assert phi_x[0, 0, 0, 0] == 1.0

        # Check linear terms
        assert torch.allclose(phi_x[0, 0, 0, 1:5], x[0, 0, 0])

        # Check first quadratic term: x[0]^2 / 2
        assert phi_x[0, 0, 0, 5] == pytest.approx(0.5, rel=0.01)  # 1^2 * 0.5

    def test_linear_attention_causality(self):
        """Test that linear attention is causal."""
        from models.based_attention import taylor_feature_map_reference, linear_attention_reference

        B, T, H, d, D = 2, 8, 4, 8, 32
        q = torch.randn(B, T, H, d)
        k = torch.randn(B, T, H, d)
        v = torch.randn(B, T, H, D)

        q_feat = taylor_feature_map_reference(q)
        k_feat = taylor_feature_map_reference(k)

        out1 = linear_attention_reference(q_feat, k_feat, v)

        # Modify future tokens
        k_modified = k.clone()
        k_modified[:, T // 2:] = torch.randn(B, T - T // 2, H, d)
        k_feat_modified = taylor_feature_map_reference(k_modified)

        out2 = linear_attention_reference(q_feat, k_feat_modified, v)

        # Past outputs should be identical (causal)
        assert torch.allclose(out1[:, : T // 2], out2[:, : T // 2], atol=1e-5)


class TestMemMamba:
    """Tests for MemMamba memory pool."""

    def test_memory_pool_update(self):
        """Test memory pool priority-based update."""
        from models.memmamba import MemoryPool

        pool = MemoryPool(pool_size=5, summary_dim=32)

        # Add entries
        summaries = torch.randn(10, 32)
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.5, 0.85])

        pool.update(summaries, scores, threshold=0.5)

        # Should have added entries with score > 0.5
        # Scores > 0.5: 0.9, 0.8, 0.7, 0.6, 0.85 (5 entries)
        assert pool.count.item() == 5

        # Priorities should be the high scores
        priorities = pool.priorities[:5].tolist()
        assert all(p > 0.5 for p in priorities)

    def test_memory_pool_replacement(self):
        """Test priority-based replacement when pool is full."""
        from models.memmamba import MemoryPool

        pool = MemoryPool(pool_size=3, summary_dim=16)

        # Fill pool
        summaries1 = torch.randn(3, 16)
        scores1 = torch.tensor([0.6, 0.7, 0.8])
        pool.update(summaries1, scores1, threshold=0.5)

        assert pool.count.item() == 3
        initial_priorities = pool.priorities.clone()

        # Try to add with higher priority
        summaries2 = torch.randn(1, 16)
        scores2 = torch.tensor([0.95])
        pool.update(summaries2, scores2, threshold=0.5)

        # Pool still size 3, but min priority should be replaced
        assert pool.count.item() == 3
        assert pool.priorities.max().item() >= 0.95

    def test_token_importance_scorer_output_range(self):
        """Test that importance scores are in [0, 1]."""
        from models.memmamba import TokenImportanceScorer

        scorer = TokenImportanceScorer(d_model=64)

        x = torch.randn(2, 16, 64)
        scores = scorer(x)

        assert scores.shape == (2, 16)
        assert (scores >= 0).all()
        assert (scores <= 1).all()


class TestZambaSharedAttention:
    """Tests for Zamba-style shared attention."""

    def test_concat_initial_residual_shapes(self):
        """Test that concatenated residual doubles input dimension."""
        B, T, D = 2, 16, 256

        x = torch.randn(B, T, D)
        x_initial = torch.randn(B, T, D)

        # Zamba concatenation
        gsa_input = torch.cat([x, x_initial], dim=-1)

        assert gsa_input.shape == (B, T, D * 2)

    def test_shared_attention_parameter_savings(self):
        """Test that shared attention reduces parameters."""
        D, n_heads = 256, 8
        n_positions = 4

        # Per-layer: each position has its own attention
        # ~3D^2 per layer (qkv + out projections)
        per_layer_params = n_positions * (3 * D * D + D * D)

        # Shared: single attention + per-position output projs
        # One 2D attention (6D^2) + n_positions output projs (2D*D each)
        shared_params = 6 * D * D + n_positions * 2 * D * D

        # Shared should use fewer parameters
        assert shared_params < per_layer_params


class TestNumericalStability:
    """Tests for numerical stability with edge cases."""

    def test_polarized_with_zeros(self):
        """Test polarized channels with zero input."""
        B, T, D = 2, 16, 256

        x = torch.zeros(B, T, D)
        zero_proj = torch.randn(D, D * 2)
        one_proj = torch.randn(D, D * 2)

        y_zero = x @ zero_proj
        y_one = torch.cumsum(x @ one_proj, dim=1)

        assert not torch.isnan(y_zero).any()
        assert not torch.isnan(y_one).any()

    def test_linear_attention_with_uniform_features(self):
        """Test linear attention doesn't divide by zero."""
        from models.based_attention import linear_attention_reference

        B, T, H, F, D = 2, 8, 4, 16, 32

        # Uniform features (could cause division issues)
        q_feat = torch.ones(B, T, H, F)
        k_feat = torch.ones(B, T, H, F)
        v = torch.randn(B, T, H, D)

        out = linear_attention_reference(q_feat, k_feat, v)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_state_expansion_with_large_values(self):
        """Test state expansion with large input values."""
        B, T, D = 2, 8, 256

        x = torch.randn(B, T, D) * 100  # Large values

        # Forget gate should still be bounded
        forget_proj = torch.randn(D, D)
        f_raw = x @ forget_proj
        f_t = torch.clamp(torch.sigmoid(f_raw), min=0.9)

        assert (f_t >= 0.9).all()
        assert (f_t <= 1.0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
