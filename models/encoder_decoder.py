"""Encoder-Decoder wrapper and ModelConfig."""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn

from .align_mamba import (
    HybridBiMambaEncoder,
    HybridMambaDecoder,
)
from ..constants import (
    PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID,
    MAX_SEQ_LEN, MQAR_VOCAB_SIZE,
)
from ..training.adaptive import compute_adaptive_dropout


@dataclass
class ModelConfig:
    """
    Configuration for the Hybrid Mamba-Attention model.

    Adaptive parameters (computed from data/capacity):
    - dropout: Derived from num_params / num_samples ratio
    - hybrid_positions: Derived from capacity theorem when num_pairs provided

    Reference: arXiv 2506.11891 (capacity theorem), Srivastava 2014 (dropout)
    """
    vocab_size: int = MQAR_VOCAB_SIZE
    d_model: int = 256
    encoder_layers: int = 2
    decoder_layers: int = 4
    d_state: int = 64
    n_heads: int = 8
    hybrid_positions: Optional[List[int]] = None  # None = adaptive from capacity
    num_pairs: Optional[int] = None  # For adaptive hybrid position computation
    num_samples: Optional[int] = None  # For adaptive dropout computation

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class HybridMambaEncoderDecoder(nn.Module):
    """
    Full Encoder-Decoder with Hybrid Mamba-Attention architecture.

    Adaptive parameters (computed at construction):
    - dropout: Derived from model capacity / data samples ratio
      Reference: Srivastava et al., 2014 (JMLR 15(56):1929-1958)
    - hybrid_positions: Derived from capacity theorem (arXiv 2506.11891)
    """

    def __init__(
        self,
        config: ModelConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}

        # Compute adaptive dropout from capacity/data ratio
        # Estimate num_params from architecture (rough upper bound)
        estimated_params = (
            config.vocab_size * config.d_model * 2 +  # Embeddings
            config.encoder_layers * config.d_model * config.d_model * 4 +  # Encoder
            config.decoder_layers * config.d_model * config.d_model * 4  # Decoder
        )
        num_samples = config.num_samples if config.num_samples else 100000  # Default
        dropout = compute_adaptive_dropout(estimated_params, num_samples)

        self.encoder = HybridBiMambaEncoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.encoder_layers,
            d_state=config.d_state,
            n_heads=config.n_heads,
            dropout=dropout,
            max_seq_len=MAX_SEQ_LEN,
            pad_token_id=PAD_TOKEN_ID,
            **factory_kwargs,
        )

        self.decoder = HybridMambaDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.decoder_layers,
            d_state=config.d_state,
            n_heads=config.n_heads,
            hybrid_positions=config.hybrid_positions,
            num_pairs=config.num_pairs,  # For adaptive position computation
            dropout=dropout,
            max_seq_len=MAX_SEQ_LEN,
            pad_token_id=PAD_TOKEN_ID,
            **factory_kwargs,
        )

    def forward(
        self,
        src_ids: Optional[torch.Tensor],
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass. Returns logits (batch, tgt_len, vocab_size)."""
        if src_ids is not None and src_ids.size(1) > MAX_SEQ_LEN:
            raise ValueError(
                f"Source sequence length ({src_ids.size(1)}) exceeds max_seq_len ({MAX_SEQ_LEN})."
            )
        if tgt_ids.size(1) > MAX_SEQ_LEN:
            raise ValueError(
                f"Target sequence length ({tgt_ids.size(1)}) exceeds max_seq_len ({MAX_SEQ_LEN})."
            )

        # Decoder-only mode when src_ids is None or encoder has 0 layers
        if src_ids is None or self.config.encoder_layers == 0:
            encoder_out = None
            encoder_padding_mask = None
        else:
            encoder_out = self.encoder(src_ids, attention_mask=src_mask)
            encoder_padding_mask = src_mask

        logits = self.decoder(
            tgt_ids,
            encoder_out,
            attention_mask=tgt_mask,
            encoder_padding_mask=encoder_padding_mask,
        )

        # Shape invariants
        assert logits.shape[:2] == tgt_ids.shape[:2], (
            f"Logits shape {logits.shape[:2]} != tgt shape {tgt_ids.shape[:2]}"
        )
        assert logits.shape[2] == self.config.vocab_size, (
            f"Logits vocab dim {logits.shape[2]} != config {self.config.vocab_size}"
        )

        return logits

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive generation. Returns (batch, gen_len)."""
        batch_size = src_ids.size(0)
        device = src_ids.device

        # Encode source directly
        encoder_out = self.encoder(src_ids, attention_mask=src_mask)
        cache = self.decoder.init_cache(batch_size, encoder_out, device=device)

        generated = torch.full((batch_size, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            logits, cache = self.decoder.step(generated[:, -1:], cache)
            next_logits = logits[:, -1, :] / temperature if temperature != 1.0 else logits[:, -1, :]

            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            if top_k is not None or top_p is not None:
                next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            finished = finished | (next_token.squeeze(-1) == EOS_TOKEN_ID)
            if finished.all():
                break

        return generated

    def gradient_checkpointing_enable(self):
        self.encoder.gradient_checkpointing_enable()
        self.decoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()
        self.decoder.gradient_checkpointing_disable()
