"""
Align-Mamba: Hybrid Mamba-Attention Architecture for Document-Level NMT.

This file contains the NOVEL contributions:
- LayerType enum and layer counting utilities
- HybridBlock: Combined Mamba + Cross-Attention block (key innovation)
- HybridBiMambaEncoder: Bidirectional encoder with sparse attention
- HybridMambaDecoder: Causal decoder with HYBRID blocks at [0, 8, 16]

CRITICAL ARCHITECTURE DECISIONS:
1. HYBRID blocks at decoder layers [0, 8, 16] (explicit, not computed)
2. Each HYBRID block: Mamba first (creates contextualized query), then Cross-Attention
3. Layer 0 HYBRID fixes "Blind Start" problem
4. BiMamba encoder with sparse bidirectional attention at strategic positions
"""

import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union, Set

import torch
import torch.nn as nn

from .components.normalization import RMSNorm
from .components.attention import BidirectionalAttention, FlashCrossAttention
from .mamba.wrapper import Mamba2BlockWrapper
from .mamba.bimamba import BiMambaBlock


# =============================================================================
# Layer Type Enum
# =============================================================================

class LayerType(Enum):
    """Types of layers in the hybrid architecture."""
    MAMBA = "mamba"
    BIMAMBA = "bimamba"
    ATTENTION = "attention"
    CROSS_ATTENTION = "cross_attention"
    HYBRID = "hybrid"  # Mamba + Cross-Attention in same block


def count_layer_types(layer_types: List[LayerType]) -> dict:
    """Count the number of each layer type."""
    counts = {}
    for lt in LayerType:
        counts[lt.value] = sum(1 for t in layer_types if t == lt)
    return counts


# =============================================================================
# Inference State Dataclasses
# =============================================================================

@dataclass
class MambaState:
    """State for a Mamba layer during inference."""
    conv_state: torch.Tensor  # (batch, d_inner, d_conv)
    ssm_state: torch.Tensor   # (batch, d_inner, d_state)


@dataclass
class AttentionKVCache:
    """KV cache for an attention layer during inference."""
    key_cache: Optional[torch.Tensor]    # (batch, seq_len, n_heads, head_dim)
    value_cache: Optional[torch.Tensor]  # (batch, seq_len, n_heads, head_dim)


@dataclass
class HybridCacheParams:
    """
    Hybrid cache for Mamba + Attention layers during autoregressive generation.

    CRITICAL: This structure must be correct from Day 1!
    - Mamba: Fixed-size state (B, d_model*expand, d_state)
    - Attention: Growing KV cache (B, current_len, n_heads, head_dim)
    """
    ssm_states: Dict[int, MambaState] = field(default_factory=dict)
    kv_caches: Dict[int, AttentionKVCache] = field(default_factory=dict)
    encoder_output: Optional[torch.Tensor] = None
    seqlen_offset: int = 0


# =============================================================================
# Hybrid Block (Mamba + Cross-Attention)
# =============================================================================

class HybridBlock(nn.Module):
    """
    HYBRID Block: Mamba + Cross-Attention.

    From plan - this is CRITICAL for the "Blind Start" fix:
    Layer 0 must be a HYBRID BLOCK (Mamba -> Cross-Attention), not just Cross-Attention.
    The Mamba sub-layer creates a "Contextualized Query" so Cross-Attention knows *what* to seek.

    Architecture:
        x = x + Mamba(RMSNorm(x))           # Position-aware, contextualized queries
        x = x + CrossAttn(RMSNorm(x), enc)  # Source-aligned output

    Why Layer 0 HYBRID Block is Essential:
    1. First decoder token sees source immediately
    2. Correct initial alignment -> correct state trajectory
    3. Mamba layers 1-7 now have source-informed hidden state
    4. Fits thesis: "Alignment at start + periodic refresh"
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        layer_idx: int = 0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.layer_idx = layer_idx

        # Mamba component (comes first to create contextualized queries)
        self.mamba = Mamba2BlockWrapper(
            d_model=d_model,
            d_state=d_state,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )

        # Cross-attention component (uses Mamba output as query)
        self.cross_attn = FlashCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        decoder_offset: int = 0,
        inference_params=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
    ):
        """
        Forward pass through hybrid block.

        Args:
            x: Decoder hidden states (batch, seq_len, d_model)
            encoder_out: Encoder output (batch, src_len, d_model) or None for decoder-only
            decoder_offset: Position offset for incremental decoding
            inference_params: Mamba inference state (for generation)
            cu_seqlens_*: For packed sequence mode

        Returns:
            Updated hidden states
        """
        # Step 1: Mamba for position-aware contextualization
        if inference_params is not None:
            x = self.mamba(x, inference_params=inference_params)
        else:
            x = self.mamba(x)

        # Step 2: Cross-attention to encoder (skip if no encoder output)
        # For decoder-only mode (e.g., MQAR), HYBRID blocks act as pure Mamba
        if encoder_out is not None:
            x = self.cross_attn(
                x,
                encoder_out,
                decoder_offset=decoder_offset,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )

        return x


# =============================================================================
# Hybrid BiMamba Encoder
# =============================================================================

class HybridBiMambaEncoder(nn.Module):
    """
    Hybrid encoder with BiMamba + sparse bidirectional attention.

    BiMamba provides bidirectional context with O(L) complexity.
    Strategic attention layers (1:7 ratio) enable in-context learning.

    Attention layers are placed at:
    - Middle layer (N/2): captures bidirectional context
    - Final layer (N-1): output refinement
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 16,
        d_state: int = 128,
        n_heads: int = 12,
        attention_ratio: float = 0.125,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        pad_token_id: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.pad_token_id = pad_token_id
        self.dtype = dtype

        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id, device=device)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Compute attention positions explicitly
        # For encoder: place at middle and final positions
        self.attention_positions = self._compute_attention_positions(n_layers, attention_ratio)

        # Build layers explicitly (no factory function)
        self.layers = nn.ModuleList()
        self.layer_types = []

        factory_kwargs = {"device": device, "dtype": dtype}

        for i in range(n_layers):
            if i in self.attention_positions:
                layer = BidirectionalAttention(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    **factory_kwargs,
                )
                self.layers.append(layer)
                self.layer_types.append(LayerType.ATTENTION)
            else:
                layer = BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    **factory_kwargs,
                )
                self.layers.append(layer)
                self.layer_types.append(LayerType.BIMAMBA)

        # Final normalization
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self._gradient_checkpointing = False

    def _compute_attention_positions(self, n_layers: int, attention_ratio: float) -> Set[int]:
        """Compute which layer indices should have attention.

        Strategy: middle layer (N/2) and final layer (N-1).
        """
        if attention_ratio >= 1.0:
            return set(range(n_layers))

        n_attention = max(2, int(n_layers * attention_ratio))
        positions = {n_layers // 2, n_layers - 1}

        if n_attention > 2:
            remaining = n_attention - 2
            step = n_layers // (remaining + 1)
            for i in range(1, remaining + 1):
                pos = i * step
                if pos not in positions:
                    positions.add(pos)

        while len(positions) > n_attention:
            for p in list(positions):
                if p not in {n_layers // 2, n_layers - 1}:
                    positions.remove(p)
                    break

        return positions

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input sequence.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional attention mask (batch, seq_len)

        Returns:
            Encoder output (batch, seq_len, d_model)
        """
        x = self.embed(input_ids) * self.embed_scale
        if self.dtype is not None:
            x = x.to(self.dtype)
        x = self.embed_dropout(x)

        for layer, layer_type in zip(self.layers, self.layer_types):
            if self._gradient_checkpointing and self.training:
                if layer_type == LayerType.ATTENTION:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, attention_mask, use_reentrant=False
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    )
            else:
                if layer_type == LayerType.ATTENTION:
                    x = layer(x, attention_mask=attention_mask)
                else:
                    x = layer(x)

        x = self.final_norm(x)
        return x

    def get_layer_counts(self) -> dict:
        """Get count of each layer type."""
        return count_layer_types(self.layer_types)

    def extra_repr(self) -> str:
        counts = self.get_layer_counts()
        return (
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"bimamba={counts['bimamba']}, attention={counts['attention']}"
        )


# =============================================================================
# Hybrid Mamba Decoder
# =============================================================================

class HybridMambaDecoder(nn.Module):
    """
    Hybrid decoder with HYBRID blocks at strategic positions.

    ARCHITECTURE (explicit, not computed):
    - Layer 0: HYBRID (Mamba + Cross-Attn) - Contextualized Preamble
    - Layers 1-7: Mamba only
    - Layer 8: HYBRID - Refresh 1
    - Layers 9-15: Mamba only
    - Layer 16: HYBRID - Refresh 2
    - Layers 17-23: Mamba only

    Total HYBRID Layers: 3 (at indices [0, 8, 16])
    Ratio: 3/24 = 1:8 = 12.5%

    Complexity per generation step:
    - Mamba: O(1) via state caching
    - Cross-attention: O(L_src) at HYBRID layers only
    """

    # EXPLICIT hybrid positions - not computed!
    DEFAULT_HYBRID_POSITIONS = {0, 8, 16}

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 24,
        d_state: int = 128,
        n_heads: int = 12,
        hybrid_interval: int = 8,
        custom_hybrid_positions: Optional[List[int]] = None,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        pad_token_id: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pad_token_id = pad_token_id
        self.hybrid_interval = hybrid_interval
        self.dtype = dtype

        # For state management during inference
        self.expand = 2
        self.d_conv = 4

        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id, device=device)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # EXPLICIT hybrid positions: [0, 8, 16, ...] or custom
        if custom_hybrid_positions is not None:
            # Use custom positions for ablation experiments
            self.hybrid_positions = set(custom_hybrid_positions)
        else:
            # Default: layer 0 + every hybrid_interval layers
            self.hybrid_positions = {0}
            for i in range(hybrid_interval, n_layers, hybrid_interval):
                self.hybrid_positions.add(i)

        # Build layers EXPLICITLY
        self.layers = nn.ModuleList()
        self.layer_types = []

        factory_kwargs = {"device": device, "dtype": dtype}

        for i in range(n_layers):
            if i in self.hybrid_positions:
                # HYBRID: Mamba + Cross-Attention in same block
                layer = HybridBlock(
                    d_model=d_model,
                    d_state=d_state,
                    n_heads=n_heads,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    layer_idx=i,
                    **factory_kwargs,
                )
                self.layers.append(layer)
                self.layer_types.append(LayerType.HYBRID)
            else:
                # Pure Mamba layer
                layer = Mamba2BlockWrapper(
                    d_model=d_model,
                    d_state=d_state,
                    layer_idx=i,
                    **factory_kwargs,
                )
                self.layers.append(layer)
                self.layer_types.append(LayerType.MAMBA)

        # Final normalization
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Output projection (language model head)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decoder forward pass (training mode).

        Args:
            input_ids: Target token IDs (batch, seq_len)
            encoder_out: Encoder output (batch, src_len, d_model) or None for decoder-only
            attention_mask: Optional attention mask

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        x = self.embed(input_ids) * self.embed_scale
        if self.dtype is not None:
            x = x.to(self.dtype)
        x = self.embed_dropout(x)

        for layer, layer_type in zip(self.layers, self.layer_types):
            if self._gradient_checkpointing and self.training:
                if layer_type == LayerType.HYBRID:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, encoder_out, use_reentrant=False
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    )
            else:
                if layer_type == LayerType.HYBRID:
                    x = layer(x, encoder_out)
                else:
                    x = layer(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits

    def init_cache(
        self,
        batch_size: int,
        encoder_out: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Union[Dict[int, MambaState], Dict[int, AttentionKVCache], torch.Tensor, int]]:
        """
        Initialize inference cache for autoregressive generation.

        Args:
            batch_size: Batch size
            encoder_out: Encoder output (cached for cross-attention)
            device: Device for cache tensors
            dtype: Data type for cache tensors

        Returns:
            Cache dictionary
        """
        device = device or next(self.parameters()).device
        dtype = dtype or torch.bfloat16

        ssm_states = {}
        kv_caches = {}

        for layer_idx, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if layer_type == LayerType.MAMBA:
                conv_state, ssm_state = layer.allocate_inference_cache(
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                )
                ssm_states[layer_idx] = MambaState(conv_state, ssm_state)
            elif layer_type == LayerType.HYBRID:
                conv_state, ssm_state = layer.mamba.allocate_inference_cache(
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                )
                ssm_states[layer_idx] = MambaState(conv_state, ssm_state)

        return {
            "ssm_states": ssm_states,
            "kv_caches": kv_caches,
            "encoder_output": encoder_out,
            "seqlen_offset": 0,
        }

    def step(
        self,
        input_ids: torch.Tensor,
        cache: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Single token generation step.

        Args:
            input_ids: Current token IDs (batch, 1)
            cache: Inference cache from init_cache or previous step

        Returns:
            Tuple of (logits, updated_cache)
        """
        x = self.embed(input_ids) * self.embed_scale
        if self.dtype is not None:
            x = x.to(self.dtype)

        offset = cache["seqlen_offset"]

        for layer_idx, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if layer_type == LayerType.MAMBA:
                state = cache["ssm_states"].get(layer_idx)
                if state is not None:
                    x = layer(x, inference_params=(state.conv_state, state.ssm_state))
                else:
                    x = layer(x)
            elif layer_type == LayerType.HYBRID:
                state = cache["ssm_states"].get(layer_idx)
                inference_params = (state.conv_state, state.ssm_state) if state else None
                x = layer(
                    x,
                    cache["encoder_output"],
                    decoder_offset=offset,
                    inference_params=inference_params,
                )

        cache["seqlen_offset"] = offset + 1

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, cache

    def get_layer_counts(self) -> dict:
        """Get count of each layer type."""
        return count_layer_types(self.layer_types)

    def extra_repr(self) -> str:
        counts = self.get_layer_counts()
        return (
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"mamba={counts['mamba']}, hybrid={counts['hybrid']}"
        )
