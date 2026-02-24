import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from flash_attn import flash_attn_func

from align_mamba.config import Config, PAD_TOKEN_ID
from align_mamba.kernels.rmsnorm import fused_rmsnorm

_ROPE_BASE = 10000.0
_RMSNORM_EPS = 1e-5
_HOUSEHOLDER_BETA_MAX = 2.0  # Keep reflector amplitude bounded.
_DECAY_GAMMA_INIT = 1.0


def _retention_bias(top_k: int) -> float:
    #Set sigmoid(bias) to the default retention (K-1)/K.
    return math.log(max(top_k - 1, 1))


def encoder_attn_positions(n_layers: int) -> frozenset[int]:
    return frozenset(range(n_layers // 2, n_layers, 2))


def decoder_inject_positions(n_layers: int) -> frozenset[int]:
    return frozenset(range(0, n_layers // 2, 2))


def decoder_attn_position(n_layers: int) -> int:
    return n_layers - 2


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, *, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = _RMSNORM_EPS
        self.weight = nn.Parameter(torch.ones(d_model, device="cuda", dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm(x, self.weight, self.eps)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float, *,
                 padding_idx: int | None = PAD_TOKEN_ID, dtype: torch.dtype | None = None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx, device="cuda")
        self.scale = nn.Parameter(torch.tensor(math.sqrt(d_model), device="cuda", dtype=dtype))
        self.dropout = nn.Dropout(dropout)
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embed(x).to(self.dtype) * self.scale)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        inv_freq = 1.0 / (_ROPE_BASE ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
        t = torch.arange(max_seq_len, device="cuda", dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        def rotate(x):
            T = x.size(1)
            cos = self.cos[:T].unsqueeze(0).unsqueeze(2).to(x.dtype)
            sin = self.sin[:T].unsqueeze(0).unsqueeze(2).to(x.dtype)
            x1, x2 = x.chunk(2, dim=-1)
            return x * cos + torch.cat([-x2, x1], dim=-1) * sin
        return rotate(q), rotate(k)


class BiMambaBlock(nn.Module):
    def __init__(self, config: Config, *, dtype: torch.dtype | None = None):
        super().__init__()
        d = config.d_model
        self.norm = RMSNorm(d, dtype=dtype)
        headdim = d // config.n_heads
        self.fwd = Mamba2(d_model=d, d_state=config.d_state, d_conv=config.mamba_d_conv,
                         expand=config.mamba_expand, headdim=headdim, device="cuda", dtype=dtype)
        self.bwd = Mamba2(d_model=d, d_state=config.d_state, d_conv=config.mamba_d_conv,
                         expand=config.mamba_expand, headdim=headdim, device="cuda", dtype=dtype)
        self.out_proj = nn.Linear(d * 2, d, bias=False, device="cuda", dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).contiguous()
        fwd_out = self.fwd(h)
        bwd_out = torch.flip(self.bwd(torch.flip(h, [1])), [1])
        return x + self.out_proj(torch.cat([fwd_out, bwd_out], dim=-1))


class Attention(nn.Module):
    def __init__(self, config: Config, *, causal: bool = False, dtype: torch.dtype | None = None):
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = d // config.n_heads
        self.dropout = config.dropout
        self.causal = causal

        self.norm = RMSNorm(d, dtype=dtype)
        self.qkv = nn.Linear(d, d * 3, bias=False, device="cuda", dtype=dtype)
        self.out = nn.Linear(d, d, bias=False, device="cuda", dtype=dtype)
        # RoPE cache must cover both train and eval sequence lengths.
        n_registers = 2 * config.n_heads
        if config.task == "lm":
            max_seq_len = (config.lm_seq_length - 1) + n_registers
        else:
            effective_pairs = max(config.num_pairs, config.eval_max_num_pairs)
            src_len = 3 * effective_pairs + 2
            tgt_len = effective_pairs + 2 + n_registers
            max_seq_len = max(src_len, tgt_len)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k = self.rope.apply_rotary(q, k)
        out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=self.causal)
        return residual + self.out(out.view(B, T, -1))


class BlockDiagonalLRU(nn.Module):
    def __init__(self, config: Config, *, dtype: torch.dtype | None = None):
        super().__init__()
        d = config.d_model
        m = config.block_size
        H = d // m
        self.m = m
        self.H = H

        self.norm = RMSNorm(d, dtype=dtype)
        self.W_v = nn.Linear(d, d, bias=False, device="cuda", dtype=dtype)
        self.W_a = nn.Linear(d, H * m * (m + 1), bias=False, device="cuda", dtype=dtype)
        self.out_proj = nn.Linear(d, d, bias=False, device="cuda", dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        m, H = self.m, self.H

        h = self.norm(x)
        v = self.W_v(h).view(B, T, H, m)
        gate_logits = self.W_a(h).view(B, T, H, m, m + 1)
        gates = F.softmax(gate_logits, dim=-1)

        # Split gate between input injection and recurrent transition.
        a0 = gates[..., 0]
        A = gates[..., 1:]

        state = torch.zeros(B, H, m, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            state = torch.matmul(A[:, t], state.unsqueeze(-1)).squeeze(-1) + a0[:, t] * v[:, t]
            outputs.append(state)

        h_out = torch.stack(outputs, dim=1).reshape(B, T, D)
        return x + self.out_proj(h_out)


class KroneckerAddress(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.U = config.kronecker_partitions
        self.d_p = config.kronecker_subdim
        self.K = config.top_k_slots
        self.log_tau = nn.Parameter(torch.zeros(1, device="cuda"))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = z.size(0)
        parts = z.view(B, self.U, self.d_p)
        tau = self.log_tau.exp()
        probs = F.softmax(parts / tau, dim=-1)

        addr = probs[:, 0]
        for u in range(1, self.U):
            addr = (addr.unsqueeze(-1) * probs[:, u].unsqueeze(-2)).reshape(B, -1)

        # Keep raw top-k weights; normalization is tracked separately in z_K/z_V.
        weights, indices = addr.topk(self.K, dim=-1)
        return indices, weights


class HKSU(nn.Module):
    def __init__(self, config: Config, *, dtype: torch.dtype | None = None):
        super().__init__()
        d = config.d_model
        d_k = config.kronecker_partitions * config.kronecker_subdim
        n_h = config.n_householder_steps
        self.M = config.kronecker_subdim ** config.kronecker_partitions
        self.n_h = n_h
        self.use_pdma = config.use_pdma
        self.scale = d ** -0.5

        self.W_k = nn.Linear(d, d_k, bias=False, device="cuda", dtype=dtype)
        self.W_q = nn.Linear(d, d_k, bias=False, device="cuda", dtype=dtype)
        self.write_addr = KroneckerAddress(config)
        self.read_addr = KroneckerAddress(config)

        self.hh_k_projs = nn.ModuleList([nn.Linear(d, d, bias=False, device="cuda", dtype=dtype) for _ in range(n_h)])
        self.hh_v_projs_k = nn.ModuleList([nn.Linear(d, d, bias=False, device="cuda", dtype=dtype) for _ in range(n_h)])
        self.hh_v_projs_v = nn.ModuleList([nn.Linear(d, d, bias=False, device="cuda", dtype=dtype) for _ in range(n_h)])
        self.hh_beta_projs = nn.ModuleList([nn.Linear(d, 1, bias=True, device="cuda", dtype=dtype) for _ in range(n_h)])

        self.gamma = nn.Parameter(torch.tensor(_DECAY_GAMMA_INIT, device="cuda", dtype=dtype))
        self.W_out = nn.Linear(d, d, bias=False, device="cuda", dtype=dtype)
        self.W_gate = nn.Linear(d, d, bias=False, device="cuda", dtype=dtype)

    def forward(
        self,
        h: torch.Tensor,
        K_slots: torch.Tensor,
        V_slots: torch.Tensor,
        z_K: torch.Tensor,
        z_V: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        B, T, D = h.shape
        M, n_h = self.M, self.n_h
        outputs = []

        for t in range(T):
            h_t = h[:, t]

            write_idx, write_w = self.write_addr(self.W_k(h_t))
            read_idx, read_w = self.read_addr(self.W_q(h_t))

            w_full = torch.zeros(B, M, device=h.device, dtype=h.dtype)
            w_full.scatter_(1, write_idx, write_w)
            if self.use_pdma:
                decay = (1.0 - w_full) ** F.softplus(self.gamma)
                K_slots = decay.unsqueeze(-1) * K_slots
                V_slots = decay.unsqueeze(-1) * V_slots
                z_K = decay * z_K + w_full
                z_V = decay * z_V + w_full
            else:
                z_K = z_K + w_full
                z_V = z_V + w_full

            idx_exp = write_idx.unsqueeze(-1).expand(-1, -1, D)
            K_active = torch.gather(K_slots, 1, idx_exp)
            V_active = torch.gather(V_slots, 1, idx_exp)

            for j in range(n_h):
                k_j = F.silu(self.hh_k_projs[j](h_t))
                k_hat = F.normalize(k_j, dim=-1)
                v_kj = self.hh_v_projs_k[j](h_t)
                v_vj = self.hh_v_projs_v[j](h_t)
                beta_j = _HOUSEHOLDER_BETA_MAX * torch.sigmoid(self.hh_beta_projs[j](h_t))

                k_hat_e = k_hat.unsqueeze(1)
                beta_e = beta_j.unsqueeze(1)

                dot_k = (k_hat_e * K_active).sum(-1, keepdim=True)
                K_active = K_active - beta_e * k_hat_e * dot_k + beta_e * k_j.unsqueeze(1) * v_kj.unsqueeze(1)

                dot_v = (k_hat_e * V_active).sum(-1, keepdim=True)
                V_active = V_active - beta_e * k_hat_e * dot_v + beta_e * k_j.unsqueeze(1) * v_vj.unsqueeze(1)

            K_slots = K_slots.scatter(1, idx_exp, K_active)
            V_slots = V_slots.scatter(1, idx_exp, V_active)

            read_exp = read_idx.unsqueeze(-1).expand(-1, -1, D)
            K_read = torch.gather(K_slots, 1, read_exp)
            V_read = torch.gather(V_slots, 1, read_exp)

            z_K_read = torch.gather(z_K, 1, read_idx).unsqueeze(-1)
            z_V_read = torch.gather(z_V, 1, read_idx).unsqueeze(-1)
            K_normed = K_read / z_K_read
            V_normed = V_read / z_V_read

            relevance = (K_normed * h_t.unsqueeze(1)).sum(-1)
            attn_w = F.softmax(relevance * self.scale, dim=-1)
            readout = (attn_w.unsqueeze(-1) * V_normed).sum(1)

            gate = torch.sigmoid(self.W_gate(h_t))
            outputs.append(gate * self.W_out(readout))

        return torch.stack(outputs, dim=1), K_slots, V_slots, z_K, z_V, write_idx, read_idx


class SurpriseGate(nn.Module):
    def __init__(self, config: Config, *, n_gate_slots: int, dtype: torch.dtype | None = None):
        super().__init__()
        d = config.d_model
        self.scale = d ** -0.5
        self.k_gate_proj = nn.Linear(2, n_gate_slots, bias=True, device="cuda", dtype=dtype)
        self.v_gate_proj = nn.Linear(2, n_gate_slots, bias=True, device="cuda", dtype=dtype)
        retention_bias = _retention_bias(config.top_k_slots)
        nn.init.constant_(self.k_gate_proj.bias, retention_bias)
        nn.init.constant_(self.v_gate_proj.bias, retention_bias)
        # Initialize EMA to roughly a sequence-length window.
        seq_len = config.lm_seq_length if config.task == "lm" else config.num_queries + 2
        eta_init = 1.0 - 1.0 / seq_len
        self.logit_eta = nn.Parameter(torch.tensor(
            math.log(eta_init / (1.0 - eta_init)), device="cuda", dtype=dtype))
        self.surprise_logit_alpha = nn.Parameter(torch.zeros(1, device="cuda", dtype=dtype))

    def forward(
        self,
        K_curr: torch.Tensor, V_curr: torch.Tensor,
        K_prev: torch.Tensor, V_prev: torch.Tensor,
        h: torch.Tensor, momentum: torch.Tensor,
        active_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D = K_curr.size(-1)
        idx_exp = active_idx.unsqueeze(-1).expand(-1, -1, D)

        K_active = torch.gather(K_curr, 1, idx_exp)
        K_prev_a = torch.gather(K_prev, 1, idx_exp)
        V_active = torch.gather(V_curr, 1, idx_exp)
        V_prev_a = torch.gather(V_prev, 1, idx_exp)

        q_probe = h.mean(dim=1).float()

        k_attn = F.softmax(torch.bmm(q_probe.unsqueeze(1), K_active.float().transpose(1, 2)) * self.scale, dim=-1)
        k_predicted = torch.bmm(k_attn, K_active.float()).squeeze(1)
        k_surprise = ((k_predicted - q_probe) ** 2).mean(dim=-1, keepdim=True)

        v_attn = F.softmax(torch.bmm(q_probe.unsqueeze(1), V_active.float().transpose(1, 2)) * self.scale, dim=-1)
        v_predicted = torch.bmm(v_attn, V_active.float()).squeeze(1)
        v_surprise = ((v_predicted - q_probe) ** 2).mean(dim=-1, keepdim=True)

        alpha = torch.sigmoid(self.surprise_logit_alpha)
        combined_surprise = alpha * k_surprise + (1.0 - alpha) * v_surprise
        eta = torch.sigmoid(self.logit_eta.float())
        new_momentum = eta * momentum + (1.0 - eta) * combined_surprise

        k_gate_input = torch.cat([k_surprise, new_momentum], dim=-1).to(self.k_gate_proj.weight.dtype)
        v_gate_input = torch.cat([v_surprise, new_momentum], dim=-1).to(self.v_gate_proj.weight.dtype)
        k_gate = torch.sigmoid(self.k_gate_proj(k_gate_input)).unsqueeze(-1)
        v_gate = torch.sigmoid(self.v_gate_proj(v_gate_input)).unsqueeze(-1)

        K_gated_a = k_gate * K_active + (1.0 - k_gate) * K_prev_a
        V_gated_a = v_gate * V_active + (1.0 - v_gate) * V_prev_a

        K_gated = K_curr.scatter(1, idx_exp, K_gated_a)
        V_gated = V_curr.scatter(1, idx_exp, V_gated_a)
        return K_gated, V_gated, new_momentum


class DecoupledInjection(nn.Module):
    def __init__(self, config: Config, *, dtype: torch.dtype | None = None):
        super().__init__()
        d = config.d_model
        self.scale = d ** -0.5
        self.W_cross = nn.Linear(d, d, bias=False, device="cuda", dtype=dtype)
        self.W_ek = nn.Linear(d, d, bias=False, device="cuda", dtype=dtype)
        self.W_ev = nn.Linear(d, d, bias=False, device="cuda", dtype=dtype)
        self.W_lambda_k = nn.Linear(d * 2, 1, bias=True, device="cuda", dtype=dtype)
        self.W_lambda_v = nn.Linear(d * 2, 1, bias=True, device="cuda", dtype=dtype)

    def forward(
        self,
        K_slots: torch.Tensor,
        V_slots: torch.Tensor,
        encoder_out: torch.Tensor,
        active_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        D = K_slots.size(-1)
        idx_exp = active_idx.unsqueeze(-1).expand(-1, -1, D)

        K_active = torch.gather(K_slots, 1, idx_exp)
        V_active = torch.gather(V_slots, 1, idx_exp)

        attn = torch.bmm(self.W_cross(K_active), encoder_out.transpose(1, 2)) * self.scale
        attn = F.softmax(attn.float(), dim=-1).to(K_active.dtype)
        K_enc = torch.bmm(attn, self.W_ek(encoder_out))
        V_enc = torch.bmm(attn, self.W_ev(encoder_out))

        lam_k = torch.sigmoid(self.W_lambda_k(torch.cat([K_active, K_enc], dim=-1)))
        lam_v = torch.sigmoid(self.W_lambda_v(torch.cat([V_active, V_enc], dim=-1)))

        K_new_a = (1.0 - lam_k) * K_active + lam_k * K_enc
        V_new_a = (1.0 - lam_v) * V_active + lam_v * V_enc

        K_new = K_slots.scatter(1, idx_exp, K_new_a)
        V_new = V_slots.scatter(1, idx_exp, V_new_a)
        return K_new, V_new


class HKSABlock(nn.Module):
    def __init__(
        self,
        config: Config,
        *,
        has_injection: bool,
        has_surprise_gate: bool,
        has_causal_attn: bool = False,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.has_injection = has_injection
        self.has_surprise_gate = has_surprise_gate
        self.has_causal_attn = has_causal_attn
        K = config.top_k_slots

        if has_causal_attn:
            self.causal_attn = Attention(config, causal=True, dtype=dtype)

        self.bdlru = BlockDiagonalLRU(config, dtype=dtype)
        self.hksu = HKSU(config, dtype=dtype)
        self.dropout = nn.Dropout(config.dropout)

        if has_injection:
            self.injection = DecoupledInjection(config, dtype=dtype)

        if has_surprise_gate:
            self.surprise_gate = SurpriseGate(config, n_gate_slots=K, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        K_slots: torch.Tensor,
        V_slots: torch.Tensor,
        z_K: torch.Tensor,
        z_V: torch.Tensor,
        K_prev: torch.Tensor,
        V_prev: torch.Tensor,
        momentum: torch.Tensor,
        encoder_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.has_causal_attn:
            x = self.causal_attn(x)

        h = self.bdlru(x)

        slot_out, K_slots, V_slots, z_K, z_V, write_idx, read_idx = self.hksu(
            h, K_slots, V_slots, z_K, z_V,
        )

        if self.has_injection:
            K_slots, V_slots = self.injection(K_slots, V_slots, encoder_out, write_idx)

        y = h + self.dropout(slot_out)

        if self.has_surprise_gate:
            K_slots, V_slots, momentum = self.surprise_gate(
                K_slots, V_slots, K_prev, V_prev, h, momentum, read_idx,
            )

        return y, K_slots, V_slots, z_K, z_V, momentum


class Encoder(nn.Module):
    def __init__(self, config: Config, *, dtype: torch.dtype | None = None):
        super().__init__()
        d = config.d_model
        n_layers = config.encoder_layers
        pad = None if config.task == "lm" else PAD_TOKEN_ID
        self.embed = Embedding(config.vocab_size, d, config.dropout, padding_idx=pad, dtype=dtype)
        attn_pos = encoder_attn_positions(n_layers)
        self.layers = nn.ModuleList([
            Attention(config, dtype=dtype) if i in attn_pos
            else BiMambaBlock(config, dtype=dtype)
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(d, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, config: Config, *, dtype: torch.dtype | None = None):
        super().__init__()
        d = config.d_model
        n_layers = config.decoder_layers
        attn_layer = decoder_attn_position(n_layers)
        inject_set = decoder_inject_positions(n_layers)

        self.M = config.kronecker_subdim ** config.kronecker_partitions
        self.d_model = d

        pad = None if config.task == "lm" else PAD_TOKEN_ID
        self.embed = Embedding(config.vocab_size, d, config.dropout, padding_idx=pad, dtype=dtype)
        self.norm = RMSNorm(d, dtype=dtype)
        self.head = nn.Linear(d, config.vocab_size, bias=False, device="cuda", dtype=dtype)
        self.n_registers = 2 * config.n_heads
        self.registers = nn.Parameter(torch.randn(1, self.n_registers, d, device="cuda", dtype=dtype) * (d ** -0.5))

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(HKSABlock(
                config,
                has_injection=(i in inject_set) and config.use_injection,
                has_surprise_gate=(i > 0) and config.use_surprise_gate,
                has_causal_attn=(i == attn_layer),
                dtype=dtype,
            ))

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor | None) -> torch.Tensor:
        B = x.size(0)
        x = self.embed(x)
        x = torch.cat([self.registers.expand(B, -1, -1), x], dim=1)

        K_slots = torch.zeros(B, self.M, self.d_model, device=x.device, dtype=x.dtype)
        V_slots = torch.zeros(B, self.M, self.d_model, device=x.device, dtype=x.dtype)
        z_K = torch.full((B, self.M), 1.0 / self.M, device=x.device, dtype=x.dtype)
        z_V = torch.full((B, self.M), 1.0 / self.M, device=x.device, dtype=x.dtype)
        momentum = torch.zeros(B, 1, device=x.device, dtype=torch.float32)

        for layer in self.layers:
            K_prev = K_slots.clone()
            V_prev = V_slots.clone()
            x, K_slots, V_slots, z_K, z_V, momentum = layer(
                x, K_slots, V_slots, z_K, z_V, K_prev, V_prev, momentum, encoder_out,
            )

        x = x[:, self.n_registers:]
        x = self.norm(x)
        return self.head(x)


class HybridMambaEncoderDecoder(nn.Module):
    def __init__(self, config: Config, *, dtype: torch.dtype | None = None):
        super().__init__()
        self.config = config
        self.has_encoder = config.encoder_layers > 0
        if self.has_encoder:
            self.encoder = Encoder(config, dtype=dtype)
        self.decoder = Decoder(config, dtype=dtype)

    def forward(self, src: torch.Tensor | None, tgt: torch.Tensor) -> torch.Tensor:
        encoder_out = self.encoder(src) if self.has_encoder else None
        return self.decoder(tgt, encoder_out)


def load_checkpoint(
    path: str,
    config: Config,
    *,
    dtype: torch.dtype,
) -> HybridMambaEncoderDecoder:
    model = HybridMambaEncoderDecoder(config, dtype=dtype)
    ckpt = torch.load(f"{path}/checkpoint.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model
