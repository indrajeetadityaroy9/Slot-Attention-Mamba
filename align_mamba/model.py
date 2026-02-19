"""Polar-Mem-Mamba: Unified Hybrid Architecture."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba2
from flash_attn import flash_attn_func

from align_mamba.config import Config, MAX_SEQ_LEN
from align_mamba.kernels.rmsnorm import fused_rmsnorm


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm(x, self.weight, self.eps)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0, device=device)
        self.scale = nn.Parameter(torch.tensor(math.sqrt(d_model)))
        self.dropout = nn.Dropout(dropout)
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embed(x).to(self.dtype) * self.scale)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, device: Optional[torch.device] = None):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(MAX_SEQ_LEN, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        def rotate(x):
            T = x.size(1)
            cos = self.cos[:T].unsqueeze(0).unsqueeze(2).to(x.dtype)
            sin = self.sin[:T].unsqueeze(0).unsqueeze(2).to(x.dtype)
            x1, x2 = x.chunk(2, dim=-1)
            return x * cos + torch.cat([-x2, x1], dim=-1) * sin
        return rotate(q), rotate(k)


class CrossAttention(nn.Module):
    """Cross-attention with separate Q and KV dimensions for GSA."""

    def __init__(self, d_q: int, d_kv: int, n_heads: int, dropout: float, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_q // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.dropout = dropout

        self.norm = RMSNorm(d_q, device=device, dtype=dtype)
        self.q_proj = nn.Linear(d_q, d_q, bias=False, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(d_kv, d_q * 2, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_q, d_q, bias=False, device=device, dtype=dtype)
        self.rope = RotaryEmbedding(self.head_dim, device=device)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, T_dec, _ = x.shape
        T_enc = encoder_out.size(1)

        q = self.q_proj(x).view(B, T_dec, self.n_heads, self.head_dim)
        kv = self.kv_proj(encoder_out).view(B, T_enc, 2, self.n_heads, self.head_dim)
        k, v = kv.unbind(dim=2)

        q, k = self.rope.apply(q, k)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0,
                              softmax_scale=self.scale, causal=False)
        return residual + self.out_proj(out.view(B, T_dec, -1))


class MemoryPool(nn.Module):
    """Differentiable memory pool with learned gating."""

    def __init__(self, d_model: int, pool_size: int, summary_dim: int, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.pool_size = pool_size
        self.summary_dim = summary_dim
        self.scale = summary_dim ** -0.5

        self.score_proj = nn.Linear(d_model, 1, bias=True, device=device, dtype=dtype)
        self.summarizer = nn.Linear(d_model, summary_dim, bias=False, device=device, dtype=dtype)
        self.q_proj = nn.Linear(d_model, summary_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(summary_dim, summary_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(summary_dim, d_model, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model * 2, d_model, bias=False, device=device, dtype=dtype)
        self.out_gate = nn.Linear(d_model, d_model, bias=True, device=device, dtype=dtype)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self.score_proj(x).squeeze(-1)

    def retrieve(self, x: torch.Tensor, pool: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(pool)
        v = self.v_proj(pool)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = attn.masked_fill(~pool_mask.unsqueeze(1).expand(-1, T, -1), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        return torch.bmm(attn, v)

    def update(self, tokens: torch.Tensor, scores: torch.Tensor,
               pool: torch.Tensor, priorities: torch.Tensor, counts: torch.Tensor):
        B, T, _ = tokens.shape
        summaries = self.summarizer(tokens)

        k = min(self.pool_size, T)
        top_scores, top_idx = torch.topk(scores, k, dim=1)
        top_summaries = torch.gather(summaries, 1, top_idx.unsqueeze(-1).expand(-1, -1, self.summary_dim))

        for b in range(B):
            n_new = min(k, self.pool_size - counts[b].item())
            if n_new > 0:
                pool[b, counts[b]:counts[b]+n_new] = top_summaries[b, :n_new]
                priorities[b, counts[b]:counts[b]+n_new] = top_scores[b, :n_new]
                counts[b] += n_new

            if counts[b] >= self.pool_size and k > n_new:
                for i in range(n_new, k):
                    min_idx = priorities[b].argmin()
                    if top_scores[b, i] > priorities[b, min_idx]:
                        pool[b, min_idx] = top_summaries[b, i]
                        priorities[b, min_idx] = top_scores[b, i]

        return pool, priorities, counts

    def forward(self, x: torch.Tensor, pool: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        retrieved = self.retrieve(x, pool, pool_mask)
        combined = torch.cat([x, retrieved], dim=-1)
        gate = torch.sigmoid(self.out_gate(x))
        return x + gate * self.out_proj(combined)


class PolarizedMemBlock(nn.Module):
    """SOTA block: Polarized Mamba2 + Differentiable Memory Pool."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        pool_size: int,
        summary_dim: int,
        layer_idx: int,
        update_freq: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.update_freq = update_freq
        self.pool_size = pool_size

        d_inner = d_model * 2
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2,
                           headdim=64, device=device, dtype=dtype)
        self.zero_proj = nn.Linear(d_model, d_inner, bias=False, device=device, dtype=dtype)
        self.one_proj = nn.Linear(d_model, d_inner, bias=False, device=device, dtype=dtype)
        self.fusion = nn.Linear(d_inner * 3, d_model, bias=False, device=device, dtype=dtype)
        self.memory = MemoryPool(d_model, pool_size, summary_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, pool: torch.Tensor,
                priorities: torch.Tensor, counts: torch.Tensor):
        h = self.norm(x).contiguous()
        y = x + self.fusion(torch.cat([
            self.mamba(h),
            self.zero_proj(h),
            torch.cumsum(self.one_proj(h), dim=1)
        ], dim=-1))

        if self.layer_idx % self.update_freq == 0:
            scores = self.memory.score(y)
            pool, priorities, counts = self.memory.update(y, scores, pool, priorities, counts)

        pool_mask = torch.arange(self.pool_size, device=x.device).unsqueeze(0) < counts.unsqueeze(1)
        y = self.memory(y, pool, pool_mask)

        return y, pool, priorities, counts


class BiMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.fwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2,
                         headdim=64, device=device, dtype=dtype)
        self.bwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2,
                         headdim=64, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model * 2, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).contiguous()
        fwd_out = self.fwd(h)
        bwd_out = torch.flip(self.bwd(torch.flip(h, [1])), [1])
        return x + self.out_proj(torch.cat([fwd_out, bwd_out], dim=-1))


class BiAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False, device=device, dtype=dtype)
        self.out = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.rope = RotaryEmbedding(self.head_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k = self.rope.apply(q, k)
        out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=False)
        return residual + self.out(out.view(B, T, -1))


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        d_state: int,
        n_heads: int,
        dropout: float,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model, dropout, device=device, dtype=dtype)
        self.attn_pos = {n_layers // 2, n_layers - 1}
        self.layers = nn.ModuleList([
            BiAttention(d_model, n_heads, dropout, device=device, dtype=dtype) if i in self.attn_pos
            else BiMambaBlock(d_model, d_state, device=device, dtype=dtype)
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        d_state: int,
        n_heads: int,
        cross_attn_layers: tuple,
        dropout: float,
        pool_size: int,
        summary_dim: int,
        update_freq: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.summary_dim = summary_dim
        self.embed = Embedding(vocab_size, d_model, dropout, device=device, dtype=dtype)

        self.cross_attn_positions = set(cross_attn_layers)
        self.layers = nn.ModuleList([
            PolarizedMemBlock(d_model, d_state, pool_size, summary_dim,
                             i, update_freq, device=device, dtype=dtype)
            for i in range(n_layers)
        ])

        # CrossAttention: Q from GSA (d_model*2), KV from encoder (d_model)
        self.cross_attn = CrossAttention(d_model * 2, d_model, n_heads, dropout, device=device, dtype=dtype)
        self.cross_projs = nn.ModuleDict({
            str(i): nn.Linear(d_model * 2, d_model, bias=False, device=device, dtype=dtype)
            for i in self.cross_attn_positions
        })

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.embed(x)
        x_init = x

        pool = torch.zeros(B, self.pool_size, self.summary_dim, device=x.device, dtype=x.dtype)
        priorities = torch.zeros(B, self.pool_size, device=x.device, dtype=x.dtype)
        counts = torch.zeros(B, device=x.device, dtype=torch.long)

        for i, layer in enumerate(self.layers):
            x, pool, priorities, counts = layer(x, pool, priorities, counts)

            if i in self.cross_attn_positions:
                gsa = torch.cat([x, x_init], dim=-1)
                x = x + self.cross_projs[str(i)](self.cross_attn(gsa, encoder_out))

        return self.head(self.norm(x))


class HybridMambaEncoderDecoder(nn.Module):
    def __init__(
        self,
        config: Config,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.config = config

        self.encoder = Encoder(
            self.config.vocab_size, self.config.d_model, self.config.encoder_layers,
            self.config.d_state, self.config.n_heads, self.config.dropout,
            device=device, dtype=dtype,
        )

        self.decoder = Decoder(
            self.config.vocab_size, self.config.d_model, self.config.decoder_layers,
            self.config.d_state, self.config.n_heads, self.config.cross_attn_layers,
            self.config.dropout, self.config.mem_pool_size, self.config.mem_summary_dim,
            self.config.mem_update_freq, device=device, dtype=dtype,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.decoder(tgt, self.encoder(src))


def get_unwrapped_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP to get the compiled model (which delegates to original)."""
    return model.module if hasattr(model, "module") else model


def load_checkpoint(
    path: str,
    config: Config,
    *,
    device: str,
    dtype: torch.dtype,
) -> "HybridMambaEncoderDecoder":
    model = HybridMambaEncoderDecoder(config, device=device, dtype=dtype)
    ckpt = torch.load(f"{path}/checkpoint.pt", map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model
