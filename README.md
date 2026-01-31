# Polar-Mem-Mamba

State capacity limits in Selective SSMs and how the **Polar-Mem-Mamba** architecture overcomes them.

## Core Thesis

Pure Mamba decoders have limited state capacity (~d_state tokens). When the number of key-value pairs exceeds d_state, accuracy collapses (the "capacity cliff"). Polar-Mem-Mamba combines three orthogonal innovations to overcome this limitation.

## Architecture

Three mechanisms addressing distinct forgetting pathways:

| Forgetting Pathway | Mechanism | Solution | Reference |
|--------------------|-----------|----------|-----------|
| Intra-layer recency | Exponential decay overwrites old tokens | Polarized channels (A=0/A=1) | [Polarized Mamba](https://arxiv.org/abs/2501.00658) |
| Inter-layer loss | Information lost across layer transitions | Cross-layer memory pool | [MemMamba](https://arxiv.org/abs/2510.03279) |
| State overflow | Fixed d_state capacity | Strategic cross-attention | [Samba](https://arxiv.org/abs/2506.11891) |

### Block Types

| Type | Components | Use Case |
|------|------------|----------|
| `mamba2` | Pure Mamba2 | Baseline |
| `polarized` | Mamba2 + A=0/A=1 fusion | Recency mitigation |
| `memmamba` | Mamba2 + memory pool | Long-range retrieval |
| `polarized_mem` | All components (default) | Full SOTA |

## Usage

```bash
# Install
pip install -e .

# Train with defaults
align-mamba --num_pairs 128 --seed 42

# Train with ablation config
align-mamba --config align_mamba/configs/ablation/a7_full.yaml

# Evaluate
align-eval --checkpoint outputs/best --num_pairs 128

# Capacity cliff analysis
align-eval --checkpoint outputs/best --capacity_cliff
```

## Configuration

Minimal config surface (17 parameters) for publication-grade experiments:

```python
from align_mamba import Config, HybridMambaEncoderDecoder

config = Config(
    # Core research parameters
    d_state=64,                  # Capacity cliff at this value
    num_pairs=128,               # Exceeds d_state to test capacity
    block_type="polarized_mem",  # mamba2 | polarized | memmamba | polarized_mem

    # Architecture
    encoder_layers=6,            # 0 for decoder-only
    hybrid_positions=[0, 2],     # Cross-attention layers
)

model = HybridMambaEncoderDecoder.from_config(config, "cuda", torch.bfloat16)
```

**Fixed values** (not configurable): `vocab_size=8192`, `label_smoothing` (auto-computed), `num_workers=8`.

## Ablation Study (A0-A7)

| Config | Polarized | Memory | Cross-Attn | Expected Result |
|--------|:---------:|:------:|:----------:|-----------------|
| A0 | - | - | - | Capacity cliff at d_state |
| A1 | ✓ | - | - | Better recency, still capacity-limited |
| A2 | - | ✓ | - | Better long-range retention |
| A3 | - | - | ✓ | Retrieval beyond d_state |
| A4 | ✓ | ✓ | - | Best decoder-only |
| A5 | ✓ | - | ✓ | Recency + retrieval |
| A6 | - | ✓ | ✓ | Long-range + retrieval |
| A7 | ✓ | ✓ | ✓ | **Full SOTA** |

Run ablations:
```bash
for i in {0..7}; do
  align-mamba --config align_mamba/configs/ablation/a${i}_*.yaml
done
```

## MQAR Task

Multi-Query Associative Recall benchmark for state capacity:

```
Encoder: [BOS k1:v1 k2:v2 ... kN:vN EOS]
Decoder: [BOS k3 k7 ... EOS]
Target:  [v3 v7 ...]
```

Token vocabulary: keys 10-4095, values 4096-8191.

## Structure

```
align_mamba/
├── config.py           # Configuration (17 params)
├── model.py            # Encoder, decoder, all block types
├── train.py            # Distributed training
├── evaluate.py         # Evaluation + capacity cliff
├── data.py             # MQAR dataset
├── configs/ablation/   # A0-A7 ablation configs
└── kernels/            # Fused Triton kernels
```

## Requirements

- PyTorch >= 2.3
- mamba-ssm >= 2.0
- flash-attn >= 2.5
- causal-conv1d >= 1.2
- triton >= 2.1

## References

- [Polarized Mamba](https://arxiv.org/abs/2501.00658) - A=0/A=1 channels for recency bias (ICLR 2025)
- [MemMamba](https://arxiv.org/abs/2510.03279) - Cross-layer memory pool
- [Samba](https://arxiv.org/abs/2506.11891) - Strategic attention placement for capacity
- [Mamba-2](https://arxiv.org/abs/2405.21060) - State space duality (ICML 2024)
- [Jamba](https://arxiv.org/abs/2403.19887) - Hybrid Mamba-attention architecture
