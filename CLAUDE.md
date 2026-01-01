# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Document-level Neural Machine Translation system using a **hybrid Mamba-2/Attention architecture**. Implements a 1:7 attention ratio (Jamba-inspired) for efficient sequence-to-sequence translation on H100 GPUs with up to 200M parameters.

## Common Commands

### Training (Single GPU)
```bash
# Default training (medium model, H100 optimized)
python doc_nmt_mamba/scripts/train.py

# Override config options
python doc_nmt_mamba/scripts/train.py model=small training=fast training.batch_size=32

# Resume from checkpoint
python doc_nmt_mamba/scripts/train.py training.resume_from=outputs/checkpoint-50000

# Debug mode (small model, minimal data)
python doc_nmt_mamba/scripts/train.py training=debug model=small data=debug
```

### Training (Multi-GPU with DDP)
```bash
# Use both H100 GPUs (2x throughput via NVLink)
torchrun --nproc_per_node=2 doc_nmt_mamba/scripts/train.py

# Explicit DDP strategy
torchrun --nproc_per_node=2 doc_nmt_mamba/scripts/train.py training.distributed_strategy=ddp

# FSDP for very large models (memory-efficient sharding)
torchrun --nproc_per_node=2 doc_nmt_mamba/scripts/train.py training.distributed_strategy=fsdp
```

### Evaluation (Publication-Ready)
```bash
# Full evaluation pipeline (Quality + Efficiency + ContraPro)
python doc_nmt_mamba/scripts/evaluation/run_full_evaluation.py --checkpoint outputs/best_model.pt

# Quick evaluation (skip COMET, fewer samples)
python doc_nmt_mamba/scripts/evaluation/run_full_evaluation.py --checkpoint outputs/best_model.pt --quick

# Translation quality only (BLEU, COMET, ChrF++)
python doc_nmt_mamba/scripts/evaluation/evaluate_quality.py --checkpoint outputs/best_model.pt

# Inference efficiency benchmarks (throughput, memory, TTFT/ITL)
python doc_nmt_mamba/scripts/evaluation/benchmark_inference.py --checkpoint outputs/best_model.pt

# ContraPro pronoun disambiguation (context utilization)
python doc_nmt_mamba/scripts/evaluation/evaluate_contrapro.py --checkpoint outputs/best_model.pt

# Ablation studies
python doc_nmt_mamba/scripts/evaluation/run_ablations.py --ablation hybrid-ratio
python doc_nmt_mamba/scripts/evaluation/run_ablations.py --ablation bidirectional
python doc_nmt_mamba/scripts/evaluation/run_ablations.py --ablation data-strategy
```

### Testing
```bash
# Run all model tests
python -m pytest doc_nmt_mamba/tests/test_models.py -v

# Specific test class
python -m pytest doc_nmt_mamba/tests/test_models.py::TestRMSNorm -v
```

## Architecture

### Code Structure
- `doc_nmt_mamba/configs/` - Hydra YAML configs (model, training, data)
- `doc_nmt_mamba/models/` - Model architecture (encoder_decoder, mamba2/, attention/, hybrid/)
- `doc_nmt_mamba/training/` - Trainer, loss functions, LR schedulers
- `doc_nmt_mamba/data/` - Dataset, tokenization, CAT-N augmentation, collation
- `doc_nmt_mamba/evaluation/` - Metrics (BLEU, COMET, ContraPro, entity recall)
- `doc_nmt_mamba/scripts/` - Entry points (train.py, evaluate.py)
- `resources/` - Reference arXiv papers

### Key Model Components
- **HybridMambaEncoderDecoder** (`models/encoder_decoder.py`) - Main model class
- **HybridBiMambaEncoder** (`models/hybrid/encoder.py`) - Bidirectional Mamba encoder (forward+backward scans)
- **HybridMambaDecoder** (`models/hybrid/decoder.py`) - Causal Mamba decoder with inference state management
- **layer_builder.py** - `compute_attention_positions()` determines which layers get attention (1:7 ratio)
- **Mamba2BlockWrapper** (`models/mamba2/mamba2_wrapper.py`) - Wrapper around official mamba-ssm library

### Model Configurations
| Config | Params | Layers | d_model | Use Case |
|--------|--------|--------|---------|----------|
| small | ~25M | 6 | 384 | Debugging |
| base | ~77M | 12 | 512 | Experiments |
| medium | ~200M | 16 | 768 | **Primary target** |
| large | ~400M | 24 | 1024 | Scaling tests |
| base_transformer | ~77M | 12 | 512 | Baseline (pure attention) |

### Inference State Management
```python
@dataclass
class HybridInferenceState:
    layer_states: List[Union[MambaState, AttentionKVCache, None]]
    encoder_output: Optional[torch.Tensor]  # Cached for cross-attention
    seqlen_offset: int  # For RoPE incremental updates
```

## Critical Technical Notes

1. **Use mamba-ssm official library, NOT custom PyTorch SSM** - Official CUDA kernels are 10-50x faster

2. **BiMamba for encoder** - Simply removing causal masking breaks Mamba. Encoder uses forward+backward parallel scans

3. **RoPE only for attention layers** - Mamba encodes position implicitly via recurrence

4. **BF16 native on H100** - No gradient scaling needed; use `torch.bfloat16` directly

5. **CAT-N augmentation** - Critical for document-level coherence. 50% single sentences + 50% concatenated N sentences

## Tech Stack

- PyTorch 2.9+ (torch.compile, BF16, native Flash SDP)
- **mamba-ssm** + **causal-conv1d** (official Mamba-2 CUDA kernels)
- **flash-attn** (FlashAttention-2) - with PyTorch SDPA fallback
- **Hydra** (config management)
- **unbabel-comet**, **sacrebleu** (evaluation)
- **datasets**, **tokenizers** (HuggingFace)

## Hardware Optimizations

The codebase is optimized for the available hardware (2x H100 80GB with NVLink):

| Optimization | Status | Impact |
|--------------|--------|--------|
| Multi-GPU DDP | ✅ Enabled | 2x throughput via NVLink |
| FSDP support | ✅ Enabled | Memory-efficient large models |
| BF16 native | ✅ Enabled | No scaling, faster compute |
| torch.compile | ✅ Enabled | 15-25% speedup |
| FlashAttention-2 | ✅ Enabled | O(N) memory for attention |
| PyTorch SDPA fallback | ✅ Enabled | Works without flash-attn |
| Packed sequences | ✅ Enabled | 20-30% faster via cu_seqlens |
| Gradient checkpointing | ✅ Enabled | 8K sequence support |
| TF32 matmul | ✅ Enabled | Faster matmul |
| Optimized data loading | ✅ Enabled | 16 workers, persistent, pinned |
| NVLink P2P | ✅ Enabled | ~478 GB/s GPU-to-GPU |

## Evaluation Framework (Publication-Ready)

The evaluation framework generates all artifacts for NeurIPS/ICML/MLSys publication:

### Paper Artifacts Generated

| Artifact | File | Description |
|----------|------|-------------|
| Table 1 | `table1_quality.tex` | BLEU/COMET/ChrF++ vs baselines |
| Figure 2 | `throughput_vs_seqlen.pdf` | Inference throughput (log-log) |
| Figure 3 | `contrapro_accuracy_vs_distance.pdf` | Pronoun accuracy vs context distance |
| Table 2 | `ablation_*.csv` | Ablation study results |

### Evaluation Metrics

**Part 1: Translation Quality (Table 1)**
- BLEU (SacreBLEU) - legacy n-gram metric
- COMET (wmt22-comet-da) - neural metric, correlates with human judgment
- ChrF++ - character-level F-score
- Paired Bootstrap Resampling (p < 0.05) for significance

**Part 2: Efficiency & Scaling (Money Charts)**
- Tokens per Second (TPS) vs sequence length
- Peak GPU memory vs sequence length
- Time-to-First-Token (TTFT) vs Inter-Token Latency (ITL)

**Part 3: Scientific Novelty (ContraPro)**
- Pronoun disambiguation accuracy
- Accuracy vs antecedent distance (proves context utilization)

**Part 4: Ablations**
- Bidirectional: BiMamba vs UniMamba encoder
- Hybrid Ratio: 1:7 vs Pure Mamba vs Pure Attention
- Data Strategy: CAT-N vs Sentence-Level
