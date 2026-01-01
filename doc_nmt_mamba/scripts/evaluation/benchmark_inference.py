#!/usr/bin/env python3
"""
Inference Benchmark for Hybrid Mamba-Attention NMT.

Generates the "Money Charts" for MLSys/NeurIPS Efficiency Track:
- Experiment A: Throughput vs. Sequence Length
- Experiment B: Memory Consumption vs. Sequence Length
- Experiment C: Pre-fill (TTFT) vs. Decoding (ITL) Split

Usage:
    python scripts/evaluation/benchmark_inference.py --checkpoint outputs/best_model.pt
    python scripts/evaluation/benchmark_inference.py --checkpoint outputs/best_model.pt --compare-transformer
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

from models import ModelConfig, HybridMambaEncoderDecoder
from data import create_tokenizer


@dataclass
class BenchmarkConfig:
    """Configuration for inference benchmarks."""
    # Sequence lengths to test
    seq_lengths: List[int] = field(default_factory=lambda: [512, 1024, 2048, 4096, 8192, 16384])

    # Batch sizes to test
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])

    # Generation settings
    max_new_tokens: int = 128
    warmup_runs: int = 3
    benchmark_runs: int = 20

    # Output
    output_dir: str = "experiments/results"


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    model_name: str
    seq_len: int
    batch_size: int

    # Throughput metrics
    tokens_per_second: float
    samples_per_second: float

    # Latency metrics (ms)
    total_latency_ms: float
    time_to_first_token_ms: float  # TTFT (Pre-fill)
    inter_token_latency_ms: float  # ITL (Decoding)

    # Memory metrics (GB)
    peak_memory_gb: float
    allocated_memory_gb: float

    # Additional info
    total_tokens_generated: int
    flops_estimate: Optional[float] = None


def get_gpu_info() -> Dict:
    """Get GPU information for logging."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": props.total_memory / 1024**3,
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
    }


def create_fake_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int = 32768,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create fake input batch for benchmarking."""
    # Source sequence
    src = torch.randint(1, vocab_size - 1, (batch_size, seq_len), device=device)

    # Target start token (BOS)
    tgt_start = torch.ones((batch_size, 1), dtype=torch.long, device=device)

    return src, tgt_start


def benchmark_prefill(
    model: nn.Module,
    src: torch.Tensor,
    warmup_runs: int = 3,
    benchmark_runs: int = 20,
) -> Tuple[float, float]:
    """
    Benchmark encoder pre-fill phase.

    Returns:
        Tuple of (mean_latency_ms, peak_memory_gb)
    """
    device = src.device

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            if hasattr(model, 'encode'):
                _ = model.encode(src)
            else:
                # For models without separate encode method
                _ = model.encoder(src)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    for _ in range(benchmark_runs):
        start_event.record()
        with torch.no_grad():
            if hasattr(model, 'encode'):
                _ = model.encode(src)
            else:
                _ = model.encoder(src)
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    mean_latency = np.mean(latencies)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    return mean_latency, peak_memory


def benchmark_generation(
    model: nn.Module,
    src: torch.Tensor,
    max_new_tokens: int = 128,
    warmup_runs: int = 3,
    benchmark_runs: int = 20,
) -> Dict:
    """
    Benchmark full generation (pre-fill + decoding).

    Returns:
        Dict with TTFT, ITL, total latency, memory, etc.
    """
    batch_size = src.shape[0]
    device = src.device

    # Check if model has generate method
    if not hasattr(model, 'generate'):
        # Fallback: manual generation loop
        return benchmark_generation_manual(
            model, src, max_new_tokens, warmup_runs, benchmark_runs
        )

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(src, max_new_tokens=10)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark: Full generation
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_latencies = []
    ttft_latencies = []

    for _ in range(benchmark_runs):
        # Measure TTFT (time to first token)
        start_event.record()
        with torch.no_grad():
            _ = model.generate(src, max_new_tokens=1)
        end_event.record()
        torch.cuda.synchronize()
        ttft = start_event.elapsed_time(end_event)
        ttft_latencies.append(ttft)

        # Measure full generation
        start_event.record()
        with torch.no_grad():
            output = model.generate(src, max_new_tokens=max_new_tokens)
        end_event.record()
        torch.cuda.synchronize()
        total_latencies.append(start_event.elapsed_time(end_event))

    mean_ttft = np.mean(ttft_latencies)
    mean_total = np.mean(total_latencies)

    # ITL = (Total - TTFT) / (num_tokens - 1)
    # Approximate since we don't know exact tokens generated
    mean_itl = (mean_total - mean_ttft) / max(1, max_new_tokens - 1)

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    allocated_memory = torch.cuda.memory_allocated() / 1024**3

    total_tokens = batch_size * max_new_tokens * benchmark_runs
    tokens_per_second = total_tokens / (sum(total_latencies) / 1000)

    return {
        "ttft_ms": mean_ttft,
        "itl_ms": mean_itl,
        "total_latency_ms": mean_total,
        "peak_memory_gb": peak_memory,
        "allocated_memory_gb": allocated_memory,
        "tokens_per_second": tokens_per_second,
        "samples_per_second": batch_size * benchmark_runs / (sum(total_latencies) / 1000),
        "total_tokens": total_tokens,
    }


def benchmark_generation_manual(
    model: nn.Module,
    src: torch.Tensor,
    max_new_tokens: int = 128,
    warmup_runs: int = 3,
    benchmark_runs: int = 20,
) -> Dict:
    """
    Manual generation benchmark for models without generate() method.
    Uses autoregressive loop with forward passes.
    """
    batch_size, src_len = src.shape
    device = src.device

    # Warmup with short sequence
    with torch.no_grad():
        for _ in range(warmup_runs):
            tgt = torch.ones((batch_size, 1), dtype=torch.long, device=device)
            for _ in range(5):
                logits = model(src, tgt)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                tgt = torch.cat([tgt, next_token], dim=1)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Measure TTFT (first forward pass)
    ttft_latencies = []
    for _ in range(benchmark_runs):
        tgt = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        start_event.record()
        with torch.no_grad():
            logits = model(src, tgt)
        end_event.record()
        torch.cuda.synchronize()
        ttft_latencies.append(start_event.elapsed_time(end_event))

    # Measure full generation
    total_latencies = []
    for _ in range(benchmark_runs):
        tgt = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        start_event.record()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(src, tgt)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                tgt = torch.cat([tgt, next_token], dim=1)
        end_event.record()
        torch.cuda.synchronize()
        total_latencies.append(start_event.elapsed_time(end_event))

    mean_ttft = np.mean(ttft_latencies)
    mean_total = np.mean(total_latencies)
    mean_itl = (mean_total - mean_ttft) / max(1, max_new_tokens - 1)

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    allocated_memory = torch.cuda.memory_allocated() / 1024**3

    total_tokens = batch_size * max_new_tokens * benchmark_runs
    tokens_per_second = total_tokens / (sum(total_latencies) / 1000)

    return {
        "ttft_ms": mean_ttft,
        "itl_ms": mean_itl,
        "total_latency_ms": mean_total,
        "peak_memory_gb": peak_memory,
        "allocated_memory_gb": allocated_memory,
        "tokens_per_second": tokens_per_second,
        "samples_per_second": batch_size * benchmark_runs / (sum(total_latencies) / 1000),
        "total_tokens": total_tokens,
    }


def run_benchmark_sweep(
    model: nn.Module,
    model_name: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """Run full benchmark sweep across sequence lengths and batch sizes."""
    results = []

    model.eval()
    device = next(model.parameters()).device
    vocab_size = model.config.vocab_size if hasattr(model, 'config') else 32768

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"{'='*60}\n")

    for batch_size in config.batch_sizes:
        for seq_len in config.seq_lengths:
            print(f"  Batch={batch_size}, SeqLen={seq_len}...", end=" ", flush=True)

            try:
                # Create input
                src, _ = create_fake_batch(batch_size, seq_len, vocab_size, device)

                # Run benchmark
                gen_results = benchmark_generation(
                    model, src,
                    max_new_tokens=config.max_new_tokens,
                    warmup_runs=config.warmup_runs,
                    benchmark_runs=config.benchmark_runs,
                )

                result = BenchmarkResult(
                    model_name=model_name,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    tokens_per_second=gen_results["tokens_per_second"],
                    samples_per_second=gen_results["samples_per_second"],
                    total_latency_ms=gen_results["total_latency_ms"],
                    time_to_first_token_ms=gen_results["ttft_ms"],
                    inter_token_latency_ms=gen_results["itl_ms"],
                    peak_memory_gb=gen_results["peak_memory_gb"],
                    allocated_memory_gb=gen_results["allocated_memory_gb"],
                    total_tokens_generated=gen_results["total_tokens"],
                )

                results.append(result)
                print(f"TPS={result.tokens_per_second:.1f}, Mem={result.peak_memory_gb:.2f}GB")

                # Clear cache between runs
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print("OOM!")
                torch.cuda.empty_cache()
                # Record OOM as special result
                results.append(BenchmarkResult(
                    model_name=model_name,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    tokens_per_second=0,
                    samples_per_second=0,
                    total_latency_ms=float('inf'),
                    time_to_first_token_ms=float('inf'),
                    inter_token_latency_ms=float('inf'),
                    peak_memory_gb=80.0,  # Max H100 memory
                    allocated_memory_gb=80.0,
                    total_tokens_generated=0,
                ))
            except Exception as e:
                print(f"Error: {e}")
                continue

    return results


def save_results(
    results: List[BenchmarkResult],
    output_dir: str,
    experiment_name: str = "efficiency",
) -> None:
    """Save benchmark results to CSV and JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame([asdict(r) for r in results])

    # Save CSV
    csv_path = output_path / f"{experiment_name}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Save JSON with metadata
    json_path = output_path / f"{experiment_name}_results.json"
    output_data = {
        "gpu_info": get_gpu_info(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [asdict(r) for r in results],
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved JSON: {json_path}")


def generate_plots(results: List[BenchmarkResult], output_dir: str) -> None:
    """Generate publication-quality plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed, skipping plots")
        return

    output_path = Path(output_dir)
    df = pd.DataFrame([asdict(r) for r in results])

    # Filter out OOM results for plotting
    df_valid = df[df['tokens_per_second'] > 0]

    if df_valid.empty:
        print("No valid results to plot")
        return

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Plot 1: Throughput vs Sequence Length (Log-Log)
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in df_valid['model_name'].unique():
        model_df = df_valid[df_valid['model_name'] == model]
        # Use batch_size=1 for clarity
        bs1_df = model_df[model_df['batch_size'] == 1]
        if not bs1_df.empty:
            ax.loglog(bs1_df['seq_len'], bs1_df['tokens_per_second'],
                     marker='o', linewidth=2, markersize=8, label=model)

    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Tokens per Second', fontsize=12)
    ax.set_title('Inference Throughput vs. Sequence Length\n(Batch Size = 1)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'throughput_vs_seqlen.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'throughput_vs_seqlen.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Memory vs Sequence Length
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in df_valid['model_name'].unique():
        model_df = df_valid[df_valid['model_name'] == model]
        bs1_df = model_df[model_df['batch_size'] == 1]
        if not bs1_df.empty:
            ax.plot(bs1_df['seq_len'], bs1_df['peak_memory_gb'],
                   marker='s', linewidth=2, markersize=8, label=model)

    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Peak GPU Memory (GB)', fontsize=12)
    ax.set_title('Memory Consumption vs. Sequence Length\n(Batch Size = 1)', fontsize=14)
    ax.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='H100 80GB Limit')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'memory_vs_seqlen.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'memory_vs_seqlen.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: TTFT vs ITL breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model in df_valid['model_name'].unique():
        model_df = df_valid[df_valid['model_name'] == model]
        bs1_df = model_df[model_df['batch_size'] == 1]
        if not bs1_df.empty:
            axes[0].plot(bs1_df['seq_len'], bs1_df['time_to_first_token_ms'],
                        marker='o', linewidth=2, markersize=8, label=model)
            axes[1].plot(bs1_df['seq_len'], bs1_df['inter_token_latency_ms'],
                        marker='o', linewidth=2, markersize=8, label=model)

    axes[0].set_xlabel('Sequence Length', fontsize=12)
    axes[0].set_ylabel('Time to First Token (ms)', fontsize=12)
    axes[0].set_title('Pre-fill Latency (TTFT)', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Sequence Length', fontsize=12)
    axes[1].set_ylabel('Inter-Token Latency (ms)', fontsize=12)
    axes[1].set_title('Decoding Latency (ITL)', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'ttft_itl_breakdown.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'ttft_itl_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plots to {output_path}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[nn.Module, str]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            config = ModelConfig(**config)
    else:
        # Default config
        config = ModelConfig()

    # Create model
    model = HybridMambaEncoderDecoder(config=config, device=device, dtype=torch.bfloat16)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model_name = Path(checkpoint_path).stem
    return model, model_name


def main():
    parser = argparse.ArgumentParser(description="Benchmark Inference Efficiency")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                       help="Output directory for results")
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                       default=[512, 1024, 2048, 4096, 8192, 16384],
                       help="Sequence lengths to benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                       default=[1, 8, 32],
                       help="Batch sizes to benchmark")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                       help="Max tokens to generate")
    parser.add_argument("--warmup-runs", type=int, default=3,
                       help="Warmup runs before timing")
    parser.add_argument("--benchmark-runs", type=int, default=20,
                       help="Number of timed runs")
    parser.add_argument("--compare-transformer", action="store_true",
                       help="Also benchmark a Transformer baseline")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")

    args = parser.parse_args()

    # Create config
    config = BenchmarkConfig(
        seq_lengths=args.seq_lengths,
        batch_sizes=args.batch_sizes,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        output_dir=args.output_dir,
    )

    # Load model
    print("Loading model...")
    model, model_name = load_model_from_checkpoint(args.checkpoint)
    model = model.cuda()

    # Run benchmarks
    all_results = []

    results = run_benchmark_sweep(model, f"Hybrid-{model_name}", config)
    all_results.extend(results)

    # Save results
    save_results(all_results, args.output_dir, "efficiency")

    # Generate plots
    if not args.no_plots:
        generate_plots(all_results, args.output_dir)

    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    df = pd.DataFrame([asdict(r) for r in all_results])
    summary = df.groupby(['model_name', 'batch_size']).agg({
        'tokens_per_second': ['mean', 'max'],
        'peak_memory_gb': 'max',
        'time_to_first_token_ms': 'mean',
        'inter_token_latency_ms': 'mean',
    }).round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()
