"""MemMamba: Cross-layer memory pool for long-range retrieval.

Addresses information loss in long sequences by maintaining a
priority-based memory pool across layers.

Components:
1. Token Importance Scoring: MLP-based scorer identifies salient tokens
2. Memory Summarizer: Compresses tokens to compact summaries
3. Memory Pool: Fixed-size buffer with priority-based replacement
4. Cross-token Retrieval: Attention-based memory access

Reference: arXiv:2510.03279 - MemMamba
"Memory-Augmented Mamba for Ultra-Long Sequences"
"""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm

# Check for CUDA extension availability
try:
    import align_mamba_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False


class TokenImportanceScorer(nn.Module):
    """MLP-based token importance scorer.

    Scores each token's importance for memory storage.
    High-scoring tokens are candidates for memory pool insertion.
    """

    def __init__(
        self,
        d_model: int,
        hidden_ratio: float = 0.25,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        hidden_dim = int(d_model * hidden_ratio)
        factory_kwargs = {"device": device, "dtype": dtype}

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(hidden_dim, 1, bias=False, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, d_model)

        Returns:
            Importance scores (B, T)
        """
        h = F.relu(self.w1(x))
        scores = torch.sigmoid(self.w2(h)).squeeze(-1)
        return scores


class MemorySummarizer(nn.Module):
    """Compresses tokens to compact memory summaries."""

    def __init__(
        self,
        d_model: int,
        summary_dim: int = 64,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.proj = nn.Linear(d_model, summary_dim, bias=False, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (..., d_model)

        Returns:
            Summary tensor (..., summary_dim)
        """
        return self.proj(x)


class MemoryPool(nn.Module):
    """Fixed-size memory pool with priority-based replacement.

    Stores compressed token summaries with associated priority scores.
    When full, lowest-priority entries are replaced by higher-scoring candidates.
    """

    def __init__(
        self,
        pool_size: int = 50,
        summary_dim: int = 64,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.pool_size = pool_size
        self.summary_dim = summary_dim

        # Register as buffers (not parameters)
        self.register_buffer(
            "pool",
            torch.zeros(pool_size, summary_dim, device=device, dtype=dtype or torch.float32)
        )
        self.register_buffer(
            "priorities",
            torch.zeros(pool_size, device=device, dtype=dtype or torch.float32)
        )
        self.register_buffer(
            "count",
            torch.tensor(0, device=device, dtype=torch.long)
        )

    def update(
        self,
        summaries: torch.Tensor,
        scores: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        """Update memory pool with new candidates.

        Args:
            summaries: Candidate summaries (N, summary_dim)
            scores: Importance scores (N,)
            threshold: Minimum score for memory insertion
        """
        # Filter by threshold
        mask = scores > threshold
        if not mask.any():
            return

        candidates = summaries[mask]
        candidate_scores = scores[mask]

        for i in range(candidates.size(0)):
            if self.count < self.pool_size:
                # Pool not full: add directly
                idx = self.count.item()
                self.pool[idx] = candidates[i]
                self.priorities[idx] = candidate_scores[i]
                self.count += 1
            else:
                # Pool full: replace lowest priority if higher
                min_idx = self.priorities.argmin()
                if candidate_scores[i] > self.priorities[min_idx]:
                    self.pool[min_idx] = candidates[i]
                    self.priorities[min_idx] = candidate_scores[i]

    def get_memories(self) -> Tuple[torch.Tensor, int]:
        """Get current memory pool contents.

        Returns:
            Tuple of (memories, count) where memories is (pool_size, summary_dim)
            and count is the number of valid entries
        """
        return self.pool, self.count.item()

    def reset(self) -> None:
        """Reset memory pool to empty state."""
        self.pool.zero_()
        self.priorities.zero_()
        self.count.zero_()


class CrossTokenRetrieval(nn.Module):
    """Attention-based retrieval from memory pool."""

    def __init__(
        self,
        d_model: int,
        summary_dim: int = 64,
        n_heads: int = 4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.summary_dim = summary_dim
        self.n_heads = n_heads
        self.head_dim = summary_dim // n_heads

        factory_kwargs = {"device": device, "dtype": dtype}

        # Query projection (from current features)
        self.q_proj = nn.Linear(d_model, summary_dim, bias=False, **factory_kwargs)

        # Key/Value projections (from memory summaries)
        self.k_proj = nn.Linear(summary_dim, summary_dim, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(summary_dim, d_model, bias=False, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        memory_pool: torch.Tensor,
        memory_count: int,
    ) -> torch.Tensor:
        """
        Args:
            x: Query features (B, T, d_model)
            memory_pool: Memory pool (pool_size, summary_dim)
            memory_count: Number of valid memory entries

        Returns:
            Retrieved features (B, T, d_model)
        """
        if memory_count == 0:
            return torch.zeros_like(x)

        B, T, _ = x.shape

        # Get valid memories
        memories = memory_pool[:memory_count]  # (M, summary_dim)

        # Compute attention
        q = self.q_proj(x)  # (B, T, summary_dim)
        k = self.k_proj(memories)  # (M, summary_dim)
        v = self.v_proj(memories)  # (M, d_model)

        # Attention scores
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.matmul(q, k.t()) * scale  # (B, T, M)
        attn = F.softmax(attn, dim=-1)

        # Retrieve
        out = torch.matmul(attn, v)  # (B, T, d_model)

        return out


class MemMambaBlock(nn.Module):
    """Mamba block with cross-layer memory augmentation.

    Combines standard Mamba processing with memory pool for
    long-range information retrieval.

    Reference: arXiv:2510.03279
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        pool_size: int = 50,
        summary_dim: int = 64,
        tau1: float = 0.5,
        tau2: float = 0.3,
        layer_idx: int = 0,
        cross_layer_frequency: int = 4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.tau1 = tau1  # Threshold for memory insertion
        self.tau2 = tau2  # Threshold for memory retrieval
        self.layer_idx = layer_idx
        self.cross_layer_frequency = cross_layer_frequency

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Import Mamba2 here to avoid circular imports
        from mamba_ssm import Mamba2
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, **factory_kwargs)

        # Memory components
        self.scorer = TokenImportanceScorer(d_model, **factory_kwargs)
        self.summarizer = MemorySummarizer(d_model, summary_dim, **factory_kwargs)
        self.memory_pool = MemoryPool(pool_size, summary_dim, **factory_kwargs)
        self.retrieval = CrossTokenRetrieval(d_model, summary_dim, **factory_kwargs)

        # Fusion gate for combining Mamba output with retrieved memory
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False, **factory_kwargs),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, d_model)
            update_memory: Whether to update memory pool

        Returns:
            Output tensor (B, T, d_model)
        """
        residual = x
        x = self.norm(x)
        x = x.contiguous()

        # Mamba processing
        y = self.mamba(x)

        # Score token importance
        scores = self.scorer(y)

        # Update memory pool at cross-layer positions
        if update_memory and self.layer_idx % self.cross_layer_frequency == 0:
            # Flatten batch and time for memory update
            flat_y = y.reshape(-1, self.d_model)
            flat_scores = scores.reshape(-1)

            # Summarize and update
            summaries = self.summarizer(flat_y)
            self.memory_pool.update(summaries, flat_scores, self.tau1)

        # Retrieve from memory if scores indicate need
        mean_score = scores.mean()
        if mean_score > self.tau2 and self.memory_pool.count > 0:
            pool, count = self.memory_pool.get_memories()
            retrieved = self.retrieval(y, pool, count)

            # Gated fusion
            gate_input = torch.cat([y, retrieved], dim=-1)
            gate = self.fusion_gate(gate_input)
            y = y + gate * retrieved

        return residual + y

    def reset_memory(self) -> None:
        """Reset memory pool for new sequence."""
        self.memory_pool.reset()


__all__ = [
    "MemMambaBlock",
    "TokenImportanceScorer",
    "MemorySummarizer",
    "MemoryPool",
    "CrossTokenRetrieval",
]
