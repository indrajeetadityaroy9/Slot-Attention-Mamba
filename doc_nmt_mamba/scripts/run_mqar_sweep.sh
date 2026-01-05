#!/bin/bash
# MQAR State Capacity Sweep - Paper Figure 1
# ============================================
#
# Runs TWO experiments to demonstrate the TC0 vs NC1 hypothesis:
#
# 1. PURE MAMBA (TC0 Baseline) - Decoder-Only Mode
#    - Full concatenated sequence [pairs...queries] to decoder
#    - No encoder, no cross-attention
#    - Tests Mamba's recurrent state capacity
#    - EXPECTED: Accuracy cliff at num_pairs > d_state (64)
#
# 2. ALIGN-MAMBA (NC1 Hybrid) - Seq2Seq Mode
#    - Encoder processes pairs, decoder processes queries
#    - Cross-attention enables retrieval from encoder
#    - Tests whether cross-attn bypasses state limits
#    - EXPECTED: >98% accuracy regardless of num_pairs
#
# Usage:
#   ./scripts/run_mqar_sweep.sh                     # Run in foreground
#   nohup ./scripts/run_mqar_sweep.sh > logs/mqar_sweep.log 2>&1 &  # Background
#
# Monitor:
#   tail -f logs/mqar_sweep.log
#   nvidia-smi dmon -s u -d 1

set -e

cd "$(dirname "$0")/.."

# Create log directory
mkdir -p logs

echo "========================================"
echo "MQAR State Capacity Sweep (Figure 1)"
echo "Started: $(date)"
echo "========================================"

# Activate virtual environment
source ../venv/bin/activate

# Create output directories
mkdir -p outputs/mqar_hybrid
mkdir -p outputs/mqar_pure_mamba

echo ""
echo "[1/2] Running PURE MAMBA (TC0 Baseline)..."
echo "  Mode: DECODER-ONLY (full concatenated sequence)"
echo "  Architecture: No encoder, no cross-attention"
echo "  Expected: Accuracy cliff at num_pairs > 64"
echo ""

python scripts/train.py experiment=mqar_pure_mamba \
    training.output_dir=outputs/mqar_pure_mamba \
    2>&1 | tee outputs/mqar_pure_mamba/train.log

echo ""
echo "[2/2] Running ALIGN-MAMBA (NC1 Hybrid)..."
echo "  Mode: SEQ2SEQ (encoder-decoder split)"
echo "  Architecture: BiMamba encoder + Hybrid decoder with cross-attn"
echo "  Expected: >98% accuracy at all num_pairs levels"
echo ""

python scripts/train.py experiment=mqar_state_sweep \
    training.output_dir=outputs/mqar_hybrid \
    2>&1 | tee outputs/mqar_hybrid/train.log

echo ""
echo "========================================"
echo "MQAR Sweep Complete!"
echo "Finished: $(date)"
echo "========================================"
echo ""
echo "Results for Figure 1:"
echo "  - Pure Mamba (TC0): outputs/mqar_pure_mamba/"
echo "  - Align-Mamba (NC1): outputs/mqar_hybrid/"
