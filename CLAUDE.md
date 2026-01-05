# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Document-Level Neural Machine Translation using Hybrid Mamba-2/Attention architecture. Combines causal Mamba blocks with sparse cross-attention layers (1:8 ratio with HYBRID blocks at layers [0, 8, 16]) for O(L) complexity on H100 hardware. Targets coherent document translation with pronoun/entity consistency.

## Environment Setup

### H100 Production Environment (ICML/NeurIPS Reproducibility)

**IMPORTANT:** For H100 hardware, use the install script instead of pip install.
This ensures correct installation order: PyTorch → CUDA kernels (with --no-build-isolation) → dependencies.

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Run the H100 installation script (handles everything)
chmod +x install_env.sh
./install_env.sh
```

The script:
1. Installs PyTorch 2.3.0 + CUDA 12.1 (H100 native)
2. Compiles Mamba-SSM, causal-conv1d, FlashAttention-2 with --no-build-isolation
3. Installs all dependencies from requirements.txt
4. Downloads SpaCy models for Entity Recall evaluation

### CPU Development Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Install base dependencies (CPU-compatible)
pip install -e ".[dev]"

# Optional: Install CPU-compatible extras
pip install -e ".[cpu]"  # comet, spacy, viz, etc.
```

Virtual environment: `venv/` (Python 3.10.11)
Requires CUDA 12.1+ for Mamba-2 kernels. FlashAttention-2 has PyTorch SDPA fallback.

### Verify Installation
```bash
python verify_env.py  # Check all dependencies and CUDA availability
```

## Commands

### Testing
```bash
# All tests (from project root, with venv activated)
python -m pytest doc_nmt_mamba/tests/ -v

# Individual test files
python -m pytest doc_nmt_mamba/tests/test_models.py -v
python -m pytest doc_nmt_mamba/tests/test_synthetic.py -v
python -m pytest doc_nmt_mamba/tests/test_verification_checklist.py -v
python -m pytest doc_nmt_mamba/tests/test_data_pipeline.py -v

# Run single test by name
python -m pytest doc_nmt_mamba/tests/test_verification_checklist.py::TestSegmentAwareFlip::test_segment_aware_flip_basic -v
```

### Linting & Formatting
```bash
black doc_nmt_mamba/          # Format code
isort doc_nmt_mamba/          # Sort imports
mypy doc_nmt_mamba/           # Type checking
```

### Training
```bash
# Single GPU (defaults to medium model - 200M params)
python doc_nmt_mamba/scripts/train.py

# With specific model size (model=hybrid_<size>)
python doc_nmt_mamba/scripts/train.py model=hybrid_small   # 25M params (debugging)
python doc_nmt_mamba/scripts/train.py model=hybrid_base    # 77M params
python doc_nmt_mamba/scripts/train.py model=hybrid_medium  # 200M params (primary)
python doc_nmt_mamba/scripts/train.py model=hybrid_large   # 400M params (scaling)

# Multi-GPU DDP
torchrun --nproc_per_node=2 doc_nmt_mamba/scripts/train.py

# Debug mode
python doc_nmt_mamba/scripts/train.py model=small training=debug

# Hydra overrides (any config value can be overridden)
python doc_nmt_mamba/scripts/train.py training.batch_size=32 training.learning_rate=1e-4
python doc_nmt_mamba/scripts/train.py data=opus_books model.dropout=0.2

# CLI entry points (after pip install -e .)
nmt-train                     # Same as python doc_nmt_mamba/scripts/train.py
nmt-evaluate                  # Run evaluation
nmt-build-tokenizer           # Build tokenizer

# ICML/NeurIPS Experiment Configs (use experiment= override)
python doc_nmt_mamba/scripts/train.py experiment=main_iwslt_hybrid    # Table 1: Main results
python doc_nmt_mamba/scripts/train.py experiment=main_opus_hybrid     # OPUS-Books evaluation
python doc_nmt_mamba/scripts/train.py experiment=ablation_no_hybrid_layer0  # Ablation: no layer-0 HYBRID
python doc_nmt_mamba/scripts/train.py experiment=mqar_state_sweep     # MQAR synthetic benchmark
```

### Evaluation
```bash
# Full publication evaluation
python doc_nmt_mamba/scripts/evaluation/run_full_evaluation.py --checkpoint path/model.pt

# Quick evaluation (skip COMET)
python doc_nmt_mamba/scripts/evaluation/run_full_evaluation.py --checkpoint path/model.pt --quick
```

### Utilities
```bash
# Build 32K BPE tokenizer
python doc_nmt_mamba/scripts/build_tokenizer.py

# Benchmark hardware performance
python doc_nmt_mamba/scripts/benchmark_hardware.py
```

## Architecture

### Model Sizes

| Model   | Params | Enc/Dec Layers | d_model | d_state | HYBRID layers          |
|---------|--------|----------------|---------|---------|------------------------|
| small   | 25M    | 6/6            | 384     | 64      | [0, 4] (interval=4)    |
| base    | 77M    | 12/12          | 512     | 64      | [0, 4, 8] (interval=4) |
| medium  | 200M   | 16/16          | 768     | 128     | [0, 8] (interval=8)    |
| large   | 400M   | 24/24          | 1024    | 128     | [0, 8, 16] (interval=8)|

### Hybrid Design (1:8 Ratio with HYBRID Blocks)

**Encoder**: BiMamba (forward+backward scan) + bidirectional attention at layers N/2 and N-1

**Decoder** (medium model: 16 layers with `hybrid_interval=8`):
- Layer 0: **HYBRID BLOCK** (Mamba + Cross-Attn) - Contextualized Preamble
- Layers 1-7: Mamba only (causal)
- Layer 8: **HYBRID BLOCK** (Mamba + Cross-Attn) - Refresh
- Layers 9-15: Mamba only (causal)

HYBRID blocks placed at layer 0 and every `hybrid_interval` layers (indices [0, 8] for 16 layers).

Each HYBRID block:
```
x = x + Mamba(RMSNorm(x))           # Position-aware queries
x = x + CrossAttn(RMSNorm(x), enc)  # Source-aligned output
```

### Key Modules
```
doc_nmt_mamba/
├── models/
│   ├── align_mamba.py      # NOVEL contributions (import from here):
│   │                       #   LayerType, HybridBlock, HybridBiMambaEncoder,
│   │                       #   HybridMambaDecoder, MambaState, AttentionKVCache
│   ├── encoder_decoder.py  # Full model + config:
│   │                       #   ModelConfig, HybridMambaEncoderDecoder
│   ├── modeling_hybrid.py  # Proxy for backward compat (re-exports above)
│   ├── components/         # Shared components:
│   │   ├── attention.py    #   BidirectionalAttention, FlashCrossAttention
│   │   └── normalization.py#   RMSNorm
│   └── mamba/              # Mamba wrappers:
│       ├── wrapper.py      #   Mamba2BlockWrapper
│       └── bimamba.py      #   BiMambaBlock, segment_aware_flip
├── data/
│   ├── dataset.py          # All datasets: DocumentNMTDataset, IWSLT14Dataset,
│   │                       #   OPUSBooksDataset, MQARDataset, MQARCurriculumGenerator
│   ├── collator.py         # PackedSequenceCollator, ConcatenationAugmenter
│   └── tokenization.py     # CustomBPETokenizer (32K), NMTTokenizer
├── evaluation/
│   ├── metrics.py          # BLEUScorer, CHRFScorer, COMETScorer
│   ├── alignment.py        # SubwordToWordMapper, AlignmentEvaluator
│   ├── contrapro.py        # Contrastive pronoun evaluation
│   └── entity_recall.py    # Entity consistency analysis
├── training/
│   ├── trainer.py          # Trainer, TrainerConfig
│   ├── hardware.py         # H100 optimization, GPU detection
│   ├── distributed.py      # DDP/FSDP setup
│   └── schedulers.py       # Learning rate schedulers
├── tests/
│   ├── test_verification_checklist.py  # Critical mechanism tests
│   ├── test_models.py
│   ├── test_synthetic.py
│   ├── test_reproducibility.py
│   └── test_data_pipeline.py
└── configs/                # Hydra configuration
    ├── config.yaml         # Main entry point
    ├── model/              # hybrid_{small,base,medium,large}, baseline_transformer
    ├── training/           # default, debug
    ├── data/               # iwslt14_de_en, opus_books, mqar, debug
    └── experiment/         # ICML/NeurIPS reproducibility configs (17 experiments)
```

## Critical Technical Decisions

1. **HYBRID Blocks at Layer 0**: First decoder layer MUST be HYBRID (Mamba + Cross-Attn) to fix "Blind Start" problem. Pure cross-attention at layer 0 lacks positional context.

2. **Conditional CUDA Imports**: All `mamba_ssm` imports are wrapped in try/except. Code is importable on CPU for development/testing.

3. **BiMamba for Encoder**: Forward+backward scans with `segment_aware_flip` that respects document boundaries (cu_seqlens).

4. **Use mamba-ssm library**: Do NOT reimplement SSD algorithm - CUDA kernels are 10-50x faster than PyTorch.

5. **Custom 32K BPE Tokenizer**: Not mBART's 250K vocab (wastes 95% of embedding table).

6. **RMSNorm in Mamba blocks**: Required for training stability at 200M+ parameters.

7. **CAT-N Augmentation**: Critical for length generalization. 50% single sentences, 50% CAT-5 concatenated with `<doc>` separator.

8. **SubwordToWordMapper**: Required for AER computation - maps BPE tokens back to word indices.

## Verification Checklist (Pre-Training)

Run before H100 training:
```bash
python -m pytest doc_nmt_mamba/tests/test_verification_checklist.py -v
```

Tests verify:
- segment_aware_flip respects document boundaries
- HYBRID blocks at correct positions ([0, 8] for 16-layer decoder)
- MQAR dataset has no leakage
- SubwordToWordMapper for AER
- CAT-N concatenation with separators

## Optional Dependencies

```bash
pip install -e ".[cuda]"   # mamba-ssm, causal-conv1d (CUDA only)
pip install -e ".[flash]"  # flash-attn (H100 recommended)
pip install -e ".[comet]"  # COMET neural evaluation
pip install -e ".[nlp]"    # spaCy for entity analysis
pip install -e ".[viz]"    # matplotlib/seaborn plotting
pip install -e ".[dev]"    # pytest, black, mypy
pip install -e ".[all]"    # Everything (GPU environment)
```

## Package Justifications (For Reviewers)

| Package | Purpose in Align-Mamba |
|---------|------------------------|
| `mamba-ssm` | Provides Mamba-2 CUDA kernels. Crucial for O(L) efficiency claim. |
| `flash-attn` | FlashAttention-2 for sparse attention layers. Essential for H100 speedups. |
| `sacrebleu` | Scientific standard for BLEU. Ensures scores are comparable to prior work. |
| `unbabel-comet` | BLEU correlates poorly with document coherence. COMET proves context handling. |
| `awesome-align` | Calculates AER (Alignment Error Rate) using mBERT alignments as gold standard. |
| `spacy` | Entity Recall evaluation. NER to verify "Mr. Smith" consistency across source/target. |
| `hydra-core` | Enables `configs/experiment/` pattern for exact hyperparameter reproducibility. |
