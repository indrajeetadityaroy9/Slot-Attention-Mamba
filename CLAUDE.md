# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polar-Mem-Mamba investigates state capacity limits in Selective SSMs (Mamba). When num_pairs exceeds d_state, pure Mamba accuracy collapses (the "capacity cliff"). This project addresses three orthogonal forgetting pathways:

1. **Polarized channels** (A=0/A=1 fusion) - mitigates intra-layer recency bias
2. **Memory pool** with learned gating - retains information across layers
3. **Cross-attention (GSA)** - enables retrieval beyond fixed state capacity

## Commands

```bash
# Install (editable, requires CUDA GPU)
pip install -e .

# Training (distributed, outputs to outputs/)
torchrun --nproc_per_node=N -m align_mamba.train --config configs/main.yaml

# Evaluation (standard)
align-eval --config configs/main.yaml

# Capacity cliff analysis
align-eval --config configs/capacity_cliff.yaml
```

All execution is config-driven via `--config path/to/experiment.yaml`. Canonical configs live in `configs/`. No test suite exists. Validation is done via the MQAR evaluation task.

## Architecture

### Single SOTA Mode

One architecture, no flags. All defaults are hardcoded in `Config`. The entry point model is `HybridMambaEncoderDecoder(config, device=device, dtype=dtype)`.

### Data Flow

```
MQAR input: encoder receives [BOS k1:v1 ... k128:v128 EOS], decoder receives [BOS k3 k7 ...]

Encoder (BiMamba + Attention)
  └─► bidirectional Mamba2 blocks, attention injected at layers n//2 and n-1
  └─► produces encoder_out

Decoder (PolarizedMemBlock + GSA)
  └─► each block: Mamba2 + polarized fusion + memory pool read/write
  └─► GSA cross-attention at layers (0, 2) attends to encoder_out
  └─► logits predict target values [v3 v7 v42 ...]
```

### Key Components

**PolarizedMemBlock** (decoder layer in `model.py`):
- RMSNorm → Mamba2 → three-way fusion: `[mamba_out, zero_proj(h), cumsum(one_proj(h))]`
- Memory pool integration: `score_proj` for importance, `out_gate` (sigmoid) for retrieval

**MemoryPool** (`model.py`):
- Learned `score_proj` replaces fixed tau1 threshold; learned `out_gate` replaces tau2
- Top-k priority queue for pool updates; attention-based retrieval
- Batch-wise state: pool, priorities, counts tracked per sample

**CrossAttention / GSA** (`model.py`):
- Gated Shortcut Attention: concatenates decoder hidden with initial embeddings before cross-attending to encoder output
- Uses Flash Attention with RoPE

**Encoder** (`model.py`): BiMambaBlock layers (forward+backward Mamba2 concatenated), BiAttention at layers n//2 and n-1

### Triton Kernels (`kernels/`)
- `rmsnorm.py` - Fused RMSNorm supporting d_model > 8192
- `loss.py` - Fused cross-entropy with label smoothing and ignore_index

## Key Defaults (config.py)

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 256 | Hidden dimension |
| d_state | 64 | SSM state size (capacity threshold) |
| encoder/decoder_layers | 6 each | |
| cross_attn_layers | (0, 2) | GSA injection points |
| mem_pool_size | 50 | Cross-layer memory entries |
| mem_update_freq | 4 | Layers between pool updates |
| num_pairs | 128 | KV pairs (intentionally > d_state) |
| num_queries | 16 | Queries per MQAR sample |
| batch_size | 256 | |
| learning_rate | 3e-4 | AdamW with weight_decay=0.01 |
| label_smoothing | 0.1 | Applied in fused cross-entropy kernel |

## Token Constants

- PAD=0, BOS=1, EOS=2, SEP=3
- Keys: 10-4095, Values: 4096-8191
- Vocab size: 8192

## Design Principles

1. **No manual tuning** - All thresholds replaced with learned gates
2. **SOTA as default** - Single optimal configuration, no experiment flags
3. **Principled training** - AdamW (fused), global grad clip=1.0, cosine schedule with warmup, bfloat16 mixed precision, torch.compile("reduce-overhead")

## Dependencies

Requires PyTorch >= 2.3, mamba-ssm >= 2, flash-attn >= 2.5, triton >= 2.1, numpy < 2, pyyaml >= 6.0. All except pyyaml require CUDA.
