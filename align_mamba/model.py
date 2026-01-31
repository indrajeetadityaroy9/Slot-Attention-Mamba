"""Polar-Mem-Mamba: Unified Hybrid Architecture."""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba2
from flash_attn import flash_attn_func

from align_mamba.config import Config, MAX_SEQ_LEN, VOCAB_SIZE
from align_mamba.kernels.rmsnorm import fused_rmsnorm


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        self.eps = 1e-4
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm(x, self.weight, self.eps)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float, device=None, dtype=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0, device=device)
        self.scale = nn.Parameter(torch.tensor(math.sqrt(d_model)))
        self.dropout = nn.Dropout(dropout)
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embed(x).to(self.dtype) * self.scale)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, device=None):
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
    def __init__(self, d_model: int, n_heads: int, dropout: float, device=None, dtype=None, **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.dropout = dropout

        self.norm = RMSNorm(d_model, device, dtype)
        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.rope = RotaryEmbedding(self.head_dim, device)

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


class Mamba2Block(nn.Module):
    def __init__(self, d_model: int, d_state: int, device=None, dtype=None, **kwargs):
        super().__init__()
        self.norm = RMSNorm(d_model, device, dtype)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2,
                           headdim=64, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x).contiguous())


class PolarizedMamba2Block(nn.Module):
    def __init__(self, d_model: int, d_state: int, device=None, dtype=None, **kwargs):
        super().__init__()
        d_inner = d_model * 2
        self.norm = RMSNorm(d_model, device, dtype)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2,
                           headdim=64, device=device, dtype=dtype)
        self.zero_proj = nn.Linear(d_model, d_inner, bias=False, device=device, dtype=dtype)
        self.one_proj = nn.Linear(d_model, d_inner, bias=False, device=device, dtype=dtype)
        self.fusion = nn.Linear(d_inner * 3, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).contiguous()
        return x + self.fusion(torch.cat([self.mamba(h), self.zero_proj(h),
                                          torch.cumsum(self.one_proj(h), dim=1)], dim=-1))


class MemoryPool(nn.Module):
    def __init__(self, d_model: int, pool_size: int, summary_dim: int,
                 tau1: float, tau2: float, device=None, dtype=None):
        super().__init__()
        self.pool_size = pool_size
        self.summary_dim = summary_dim
        self.tau1 = tau1
        self.tau2 = tau2

        hidden = d_model // 4
        self.score_w1 = nn.Linear(d_model, hidden, bias=False, device=device, dtype=dtype)
        self.score_w2 = nn.Linear(hidden, 1, bias=False, device=device, dtype=dtype)
        self.summarizer = nn.Linear(d_model, summary_dim, bias=False, device=device, dtype=dtype)
        self.q_proj = nn.Linear(d_model, summary_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(summary_dim, summary_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(summary_dim, d_model, bias=False, device=device, dtype=dtype)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False, device=device, dtype=dtype),
            nn.Sigmoid(),
        )

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.score_w2(F.relu(self.score_w1(x)))).squeeze(-1)

    def retrieve(self, x: torch.Tensor, pool: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(pool), self.v_proj(pool)
        attn = torch.bmm(q, k.transpose(1, 2)) / (self.summary_dim ** 0.5)
        attn = attn.masked_fill(~mask.unsqueeze(1).expand(-1, T, -1), float('-inf'))
        attn = torch.nan_to_num(F.softmax(attn, dim=-1), nan=0.0)
        return torch.bmm(attn, v)

    def update(self, tokens: torch.Tensor, scores: torch.Tensor, pool: torch.Tensor,
               priorities: torch.Tensor, counts: torch.Tensor):
        B, T, D = tokens.shape
        mask = scores > self.tau1
        sorted_scores, indices = torch.sort(scores, dim=1, descending=True)
        sorted_tokens = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        sorted_mask = torch.gather(mask, 1, indices)
        summaries = self.summarizer(sorted_tokens.view(B * T, D)).view(B, T, -1)

        for slot in range(min(T, self.pool_size)):
            has_imp = sorted_mask[:, slot]
            imp = sorted_scores[:, slot]
            summ = summaries[:, slot]

            add = has_imp & (counts < self.pool_size)
            for b in range(B):
                if add[b]:
                    pool[b, counts[b]] = summ[b]
                    priorities[b, counts[b]] = imp[b]
                    counts[b] += 1

            replace = has_imp & (counts >= self.pool_size)
            min_p, min_idx = priorities.min(dim=1)
            should = replace & (imp > min_p)
            for b in range(B):
                if should[b]:
                    pool[b, min_idx[b]] = summ[b]
                    priorities[b, min_idx[b]] = imp[b]

        return pool, priorities, counts


class MemMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, pool_size: int, summary_dim: int,
                 tau1: float, tau2: float, layer_idx: int, cross_layer_frequency: int,
                 device=None, dtype=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.freq = cross_layer_frequency
        self.tau2 = tau2
        self.pool_size = pool_size
        self.summary_dim = summary_dim

        self.norm = RMSNorm(d_model, device, dtype)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2,
                           headdim=64, device=device, dtype=dtype)
        self.memory = MemoryPool(d_model, pool_size, summary_dim, tau1, tau2, device, dtype)

    def forward(self, x: torch.Tensor, pool: torch.Tensor, priorities: torch.Tensor, counts: torch.Tensor):
        B = x.size(0)
        y = x + self.mamba(self.norm(x).contiguous())
        scores = self.memory.score(y)

        if self.layer_idx % self.freq == 0:
            pool, priorities, counts = self.memory.update(y, scores, pool, priorities, counts)

        mask = torch.arange(self.pool_size, device=x.device).unsqueeze(0) < counts.unsqueeze(1)
        retrieved = self.memory.retrieve(y, pool, mask)
        gate = self.memory.gate(torch.cat([y, retrieved], dim=-1))
        retrieve_mask = ((scores.mean(dim=1) > self.tau2) & (counts > 0)).view(B, 1, 1)
        y = y + gate * retrieved * retrieve_mask

        return y, pool, priorities, counts


class PolarizedMemMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, pool_size: int, summary_dim: int,
                 tau1: float, tau2: float, layer_idx: int, cross_layer_frequency: int,
                 device=None, dtype=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.freq = cross_layer_frequency
        self.tau2 = tau2
        self.pool_size = pool_size
        self.summary_dim = summary_dim
        d_inner = d_model * 2

        self.norm = RMSNorm(d_model, device, dtype)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2,
                           headdim=64, device=device, dtype=dtype)
        self.zero_proj = nn.Linear(d_model, d_inner, bias=False, device=device, dtype=dtype)
        self.one_proj = nn.Linear(d_model, d_inner, bias=False, device=device, dtype=dtype)
        self.fusion = nn.Linear(d_inner * 3, d_model, bias=False, device=device, dtype=dtype)
        self.memory = MemoryPool(d_model, pool_size, summary_dim, tau1, tau2, device, dtype)

    def forward(self, x: torch.Tensor, pool: torch.Tensor, priorities: torch.Tensor, counts: torch.Tensor):
        B = x.size(0)
        h = self.norm(x).contiguous()
        y = x + self.fusion(torch.cat([self.mamba(h), self.zero_proj(h),
                                        torch.cumsum(self.one_proj(h), dim=1)], dim=-1))
        scores = self.memory.score(y)

        if self.layer_idx % self.freq == 0:
            pool, priorities, counts = self.memory.update(y, scores, pool, priorities, counts)

        mask = torch.arange(self.pool_size, device=x.device).unsqueeze(0) < counts.unsqueeze(1)
        retrieved = self.memory.retrieve(y, pool, mask)
        gate = self.memory.gate(torch.cat([y, retrieved], dim=-1))
        retrieve_mask = ((scores.mean(dim=1) > self.tau2) & (counts > 0)).view(B, 1, 1)
        y = y + gate * retrieved * retrieve_mask

        return y, pool, priorities, counts


BLOCKS = {
    "mamba2": Mamba2Block,
    "polarized": PolarizedMamba2Block,
    "memmamba": MemMambaBlock,
    "polarized_mem": PolarizedMemMambaBlock,
}


class BiMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, device=None, dtype=None):
        super().__init__()
        self.norm = RMSNorm(d_model, device, dtype)
        self.fwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2,
                         headdim=64, device=device, dtype=dtype)
        self.bwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2,
                         headdim=64, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model * 2, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.out_proj(torch.cat([self.fwd(h), torch.flip(self.bwd(torch.flip(h, [1])), [1])], dim=-1))


class BiAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, device=None, dtype=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.norm = RMSNorm(d_model, device, dtype)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False, device=device, dtype=dtype)
        self.out = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.rope = RotaryEmbedding(self.head_dim, device)

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
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, d_state: int,
                 n_heads: int, dropout: float, device=None, dtype=None):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model, dropout, device, dtype)
        self.attn_pos = {n_layers // 2, n_layers - 1}
        self.layers = nn.ModuleList([
            BiAttention(d_model, n_heads, dropout, device, dtype) if i in self.attn_pos
            else BiMambaBlock(d_model, d_state, device, dtype)
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, d_state: int,
                 n_heads: int, num_pairs: int, hybrid_positions: List[int],
                 dropout: float, block_type: str, pool_size: int, summary_dim: int,
                 tau1: float, tau2: float, cross_freq: int, device=None, dtype=None):
        super().__init__()
        self.pool_size = pool_size
        self.summary_dim = summary_dim
        self.embed = Embedding(vocab_size, d_model, dropout, device, dtype)

        self.cross_attn_positions = set(hybrid_positions)
        block_cls = BLOCKS[block_type]
        self.has_memory = block_type in ("memmamba", "polarized_mem")

        self.layers = nn.ModuleList([
            block_cls(d_model=d_model, d_state=d_state, pool_size=pool_size,
                      summary_dim=summary_dim, tau1=tau1, tau2=tau2,
                      layer_idx=i, cross_layer_frequency=cross_freq, device=device, dtype=dtype)
            for i in range(n_layers)
        ])

        self.cross_attn = CrossAttention(d_model * 2, n_heads, dropout, device, dtype)
        self.cross_projs = nn.ModuleDict({
            str(i): nn.Linear(d_model * 2, d_model, bias=False, device=device, dtype=dtype)
            for i in self.cross_attn_positions
        })

        self.norm = RMSNorm(d_model, device, dtype)
        self.head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.embed(x)
        x_init = x

        pool = torch.zeros(B, self.pool_size, self.summary_dim, device=x.device, dtype=x.dtype)
        priorities = torch.zeros(B, self.pool_size, device=x.device, dtype=x.dtype)
        counts = torch.zeros(B, device=x.device, dtype=torch.long)

        for i, layer in enumerate(self.layers):
            if self.has_memory:
                x, pool, priorities, counts = layer(x, pool, priorities, counts)
            else:
                x = layer(x)

            if i in self.cross_attn_positions:
                gsa = torch.cat([x, x_init], dim=-1)
                x = x + self.cross_projs[str(i)](self.cross_attn(gsa, encoder_out))

        return self.head(self.norm(x))


class HybridMambaEncoderDecoder(nn.Module):
    def __init__(self, config: Config, device=None, dtype=None):
        super().__init__()
        self.config = config

        params = config.vocab_size * config.d_model * 2 + \
                 (config.encoder_layers + config.decoder_layers) * config.d_model ** 2 * 4
        dropout = 0.5 * (1 - math.exp(-params / config.num_samples * 100))
        dropout = max(0.0, min(0.5, dropout))

        self.encoder = Encoder(config.vocab_size, config.d_model, config.encoder_layers,
                               config.d_state, config.n_heads, dropout, device, dtype)

        self.decoder = Decoder(config.vocab_size, config.d_model, config.decoder_layers,
                               config.d_state, config.n_heads, config.num_pairs,
                               config.hybrid_positions, dropout, config.block_type,
                               config.mem_pool_size, config.mem_summary_dim,
                               config.mem_tau1, config.mem_tau2,
                               config.mem_update_freq, device, dtype)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.decoder(tgt, self.encoder(src))

    @classmethod
    def from_config(cls, config: Config, device: str, dtype: torch.dtype):
        return cls(config, device, dtype)


def get_unwrapped_model(model: nn.Module) -> nn.Module:
    return model._orig_mod.module if hasattr(model, "_orig_mod") else model.module if hasattr(model, "module") else model


def load_checkpoint(path: str, device: str, dtype: torch.dtype):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    config = Config.from_dict(ckpt['config'])
    model = HybridMambaEncoderDecoder(config, device, dtype)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, config
