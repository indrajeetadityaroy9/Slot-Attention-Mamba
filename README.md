# Polar-Mem-Mamba

Overcoming state capacity limits in Selective State Space Models through a unified hybrid architecture.

## Research Objectives

Pure Mamba decoders have limited state capacity (~d_state tokens). When the number of key-value pairs exceeds d_state, accuracy collapses—a phenomenon we term the **capacity cliff**. This project investigates and addresses three orthogonal forgetting pathways through a unified architecture that combines insights from recent SSM research.

### Core Thesis

The capacity cliff in Selective SSMs stems from three distinct mechanisms:
1. **Intra-layer recency bias**: Exponential decay in state transitions overwrites distant tokens
2. **Inter-layer information loss**: Useful information fails to propagate across layer boundaries
3. **State overflow**: Fixed d_state creates a hard ceiling on retrievable associations

## Architecture

The **Polar-Mem-Mamba** architecture addresses each forgetting pathway with a targeted mechanism:

| Forgetting Pathway | Root Cause | Mechanism | Innovation |
|--------------------|------------|-----------|------------|
| Intra-layer recency | Exponential decay A<1 | Polarized channels | Split into A=0 (memory) and A=1 (cumsum) pathways |
| Inter-layer loss | No cross-layer state sharing | Memory pool | Learned gating for cross-layer information persistence |
| State overflow | Fixed d_state capacity | Strategic cross-attention | GSA at early decoder layers for unlimited retrieval |

### PolarizedMemBlock (SOTA Decoder Block)

Each decoder block combines three components:

```
Input x
    │
    ├──► Mamba2(norm(x))           # Standard SSM path
    ├──► zero_proj(norm(x))        # A=0: Perfect memory (no decay)
    └──► cumsum(one_proj(norm(x))) # A=1: Running sum accumulator
    │
    └──► fusion([mamba, zero, cumsum]) ──► y
                                          │
                                          ├──► MemoryPool.score(y)
                                          ├──► MemoryPool.update(y) [periodic]
                                          └──► MemoryPool.retrieve(y) ──► output
```

### Learned Memory Gating

The memory pool uses fully learned gating mechanisms—no fixed thresholds:

- **Write gate**: `score_proj(x)` projects tokens to importance scores; top-k selection determines pool updates
- **Read gate**: `out_gate(x)` produces input-dependent sigmoid gates for retrieval modulation
- **Priority queue**: Pool entries maintain learned priority scores for replacement decisions

### Gated Shortcut Attention (GSA)

Cross-attention at strategic positions (layers 0, 2 by default) provides unlimited-capacity retrieval:

```
decoder_hidden ──┬──► x
                 │
initial_embed ───┴──► x_init
                 │
                 └──► concat([x, x_init]) ──► GSA ──► CrossAttention(encoder_out)
```

The shortcut connection from initial embeddings preserves position information that would otherwise be lost through the recurrent layers.

## Parameter-Free Design

All fixed thresholds and heuristics are replaced with learned mechanisms:

| Original Approach | Polar-Mem-Mamba |
|-------------------|-----------------|
| tau1=0.5 write threshold | Learned `score_proj` + top-k selection |
| tau2=0.3 read threshold | Learned `out_gate` sigmoid |
| Adaptive dropout formulas | Fixed dropout=0.1 |
| Per-parameter gradient clipping | Global norm clip=1.0 |
| Data-dependent weight decay | Fixed weight_decay=0.01 |

## Usage

```bash
# Install
pip install -e .

# Train (distributed)
torchrun --nproc_per_node=N -m align_mamba.train --config configs/main.yaml

# Evaluate
align-eval --config configs/main.yaml

# Capacity cliff analysis
align-eval --config configs/capacity_cliff.yaml
```

### Python API

```python
from align_mamba import Config, HybridMambaEncoderDecoder

config = Config()  # SOTA defaults
model = HybridMambaEncoderDecoder(config, device="cuda", dtype=torch.bfloat16)

# Forward pass
logits = model(src_ids, tgt_ids[:, :-1])
```

## Configuration

The system uses principled defaults requiring no configuration:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| d_model | 256 | Hidden dimension |
| d_state | 64 | SSM state size (capacity threshold) |
| encoder_layers | 6 | Bidirectional Mamba encoder |
| decoder_layers | 6 | Polarized memory decoder |
| cross_attn_layers | (0, 2) | GSA injection points |
| mem_pool_size | 50 | Cross-layer memory capacity |
| mem_summary_dim | 64 | Memory summary dimension |
| mem_update_freq | 4 | Layers between pool updates |
| num_pairs | 128 | KV pairs (exceeds d_state) |
| num_queries | 16 | Query count per sample |

## MQAR Benchmark

Multi-Query Associative Recall tests state capacity by requiring retrieval beyond d_state:

```
Encoder input:  [BOS k1:v1 k2:v2 ... k128:v128 EOS]
Decoder input:  [BOS k3 k7 k42 ... EOS]
Target output:  [v3 v7 v42 ...]
```

Token vocabulary: keys [10, 4095], values [4096, 8191].

When num_pairs > d_state, pure Mamba accuracy drops sharply. Polar-Mem-Mamba maintains accuracy through its three complementary mechanisms.

## Project Structure

```
align_mamba/
├── config.py      # SOTA configuration dataclass
├── model.py       # Encoder, Decoder, PolarizedMemBlock, MemoryPool
├── train.py       # Distributed training with DDP
├── evaluate.py    # Evaluation and capacity cliff analysis
├── data.py        # MQAR dataset generation
└── kernels/       # Fused Triton kernels (RMSNorm, CrossEntropy)
```

## Requirements

- PyTorch >= 2.3
- mamba-ssm >= 2.0
- flash-attn >= 2.5
- causal-conv1d >= 1.2
- triton >= 2.1

## References

### Primary Innovations

- **Polarized Mamba** - A=0/A=1 channel splitting for recency bias mitigation
  [Gated Slot Attention for Efficient Linear-Time Sequence Modeling](https://arxiv.org/abs/2501.00658) (ICLR 2025)

- **MemMamba** - Cross-layer memory pools with learned gating
  [MemMamba: Memory-Efficient Long Sequence Modeling](https://arxiv.org/abs/2510.03279)

- **Samba** - Strategic attention placement for capacity overflow
  [Samba: Simple Hybrid State Space Models](https://arxiv.org/abs/2506.11891)

### Foundational Work

- **Mamba** - Selective State Space Models
  [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (NeurIPS 2024)

- **Mamba-2** - State Space Duality and improved hardware efficiency
  [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) (ICML 2024)

- **Jamba** - Hybrid Mamba-attention architecture at scale
  [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)

### Attention Mechanisms

- **Flash Attention** - IO-aware exact attention
  [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) (NeurIPS 2022)

- **Flash Attention 2** - Improved parallelism and work partitioning
  [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

- **Rotary Position Embedding** - Relative position encoding via rotation
  [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

## License

MIT
