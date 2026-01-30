"""
Align-Mamba: Hybrid Mamba-Attention blocks for Document-Level NMT.

HYBRID blocks at decoder [0, 8, 16] fix "Blind Start": Mamba creates contextualized
query before cross-attention, ensuring correct initial alignment.
"""

import math
from typing import Optional, List, Tuple, Dict, Union, Set

import torch
import torch.nn as nn

from .normalization import RMSNorm
from .embeddings import ScaledEmbedding
from .attention import BidirectionalAttention, FlashCrossAttention
from .wrapper import Mamba2BlockWrapper
from .bimamba import BiMambaBlock


def compute_hybrid_positions_adaptive(
    n_layers: int,
    d_state: int,
    num_pairs: int,
) -> Set[int]:
    """
    Derive cross-attention placement from capacity theorem.

    Reference: Huang et al., 2025 (arXiv 2506.11891, Theorem 2)
    "1-layer Mamba solves MQAR with state size N=κ"

    When num_pairs > d_state, information overflows and requires
    periodic cross-attention to retrieve from encoder.

    Spacing derived from exponential decay rate of SSM memory.
    Reference: arXiv 2510.03279 (MemMamba), Section 3.1
    "contribution of x_{t-k} decays as |A^k|"
    """
    # Layer 0 always required (Blind Start fix)
    positions = {0}

    if num_pairs <= d_state:
        return positions

    overflow_ratio = num_pairs / d_state
    interval = max(1, int(d_state / math.log(max(overflow_ratio, math.e))))

    for pos in range(interval, n_layers, interval):
        positions.add(pos)

    return positions


class HybridBiMambaEncoder(nn.Module):
    """BiMamba encoder with sparse attention at N/2 and N-1 layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 16,
        d_state: int = 128,
        n_heads: int = 12,
        attention_ratio: Optional[float] = None,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        pad_token_id: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = ScaledEmbedding(
            vocab_size=vocab_size, d_model=d_model, padding_idx=pad_token_id,
            dropout=dropout, device=device, dtype=dtype,
        )

        # Attention at N/2 and N-1 (2 layers for any n_layers >= 2)
        if attention_ratio is None:
            attention_ratio = 2.0 / n_layers
        n_attention = max(2, int(n_layers * attention_ratio))
        self.attention_positions = {n_layers // 2, n_layers - 1}

        # Add more positions if n_attention > 2
        if n_attention > 2:
            step = n_layers // (n_attention - 1)
            for i in range(1, n_attention - 1):
                self.attention_positions.add(i * step)

        factory_kwargs = {"device": device, "dtype": dtype}
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i in self.attention_positions:
                self.layers.append(BidirectionalAttention(
                    d_model=d_model, n_heads=n_heads, dropout=dropout,
                    max_seq_len=max_seq_len, **factory_kwargs,
                ))
            else:
                self.layers.append(BiMambaBlock(
                    d_model=d_model, d_state=d_state, **factory_kwargs,
                ))

        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(input_ids)

        for i, layer in enumerate(self.layers):
            is_attention = i in self.attention_positions
            if self._gradient_checkpointing and self.training:
                if is_attention:
                    x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask, use_reentrant=False)
                else:
                    x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x, attention_mask=attention_mask) if is_attention else layer(x)

        return self.final_norm(x)


class HybridMambaDecoder(nn.Module):
    """Decoder with cross-attention at adaptive positions (arXiv 2506.11891).

    Layer 0 cross-attention fixes "Blind Start". Additional layers at intervals
    derived from capacity overflow ratio when num_pairs > d_state.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 24,
        d_state: int = 128,
        n_heads: int = 12,
        hybrid_positions: Optional[List[int]] = None,
        num_pairs: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        pad_token_id: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = ScaledEmbedding(
            vocab_size=vocab_size, d_model=d_model, padding_idx=pad_token_id,
            dropout=dropout, device=device, dtype=dtype,
        )

        # Compute hybrid positions: explicit > adaptive > default {0}
        if hybrid_positions is not None:
            self.hybrid_positions = set(hybrid_positions)
        elif num_pairs is not None:
            self.hybrid_positions = compute_hybrid_positions_adaptive(n_layers, d_state, num_pairs)
        else:
            self.hybrid_positions = {0}  # Minimum: Layer 0 for Blind Start fix

        factory_kwargs = {"device": device, "dtype": dtype}
        self.layers = nn.ModuleList([
            Mamba2BlockWrapper(d_model=d_model, d_state=d_state, layer_idx=i, **factory_kwargs)
            for i in range(n_layers)
        ])

        self.cross_attn = nn.ModuleDict({
            str(i): FlashCrossAttention(
                d_model=d_model, n_heads=n_heads, dropout=dropout,
                max_seq_len=max_seq_len, **factory_kwargs
            )
            for i in self.hybrid_positions
        })

        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(input_ids)

        for i, layer in enumerate(self.layers):
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

            if str(i) in self.cross_attn and encoder_out is not None:
                if self._gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        self.cross_attn[str(i)], x, encoder_out, encoder_padding_mask, use_reentrant=False
                    )
                else:
                    x = self.cross_attn[str(i)](x, encoder_out, encoder_padding_mask=encoder_padding_mask)

        return self.lm_head(self.final_norm(x))

    def init_cache(
        self,
        batch_size: int,
        encoder_out: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict:
        device = device or next(self.parameters()).device
        dtype = dtype or torch.bfloat16

        # Store SSM states as tuples: (conv_state, ssm_state)
        ssm_states = {}
        for i, layer in enumerate(self.layers):
            ssm_states[i] = layer.allocate_inference_cache(batch_size=batch_size, dtype=dtype, device=device)

        return {"ssm_states": ssm_states, "encoder_output": encoder_out, "seqlen_offset": 0}

    def step(
        self,
        input_ids: torch.Tensor,
        cache: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        x = self.embedding.embed(input_ids) * self.embedding.embed_scale
        if self.embedding.dtype is not None:
            x = x.to(self.embedding.dtype)

        offset = cache["seqlen_offset"]

        for i, layer in enumerate(self.layers):
            state = cache["ssm_states"].get(i)
            x = layer(x, inference_params=state) if state else layer(x)

            if str(i) in self.cross_attn and cache["encoder_output"] is not None:
                x = self.cross_attn[str(i)](x, cache["encoder_output"], decoder_offset=offset)

        cache["seqlen_offset"] = offset + 1
        return self.lm_head(self.final_norm(x)), cache
