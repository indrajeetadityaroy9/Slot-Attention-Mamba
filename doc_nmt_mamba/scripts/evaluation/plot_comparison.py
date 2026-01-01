#!/usr/bin/env python3
"""
Comparison Plotting for Baseline vs. Thesis Model.

Generates publication-quality overlay plots comparing:
- Transformer Baseline vs. Hybrid Mamba-Attention

This script addresses "Risk B: Single Point Evaluation" by producing
comparative plots required for scientific papers.

Generates:
- Figure 2: Throughput vs. Sequence Length (Log-Log overlay)
- Figure 3: Memory vs. Sequence Length (overlay)
- Figure 4: ContraPro Accuracy vs. Distance (overlay)
- Table 1: Combined quality metrics (LaTeX)

Usage:
    python scripts/evaluation/plot_comparison.py \
        --baseline results/baseline/evaluation_summary.json \
        --thesis results/thesis/evaluation_summary.json \
        --output-dir paper/figures

    # With custom labels
    python scripts/evaluation/plot_comparison.py \
        --baseline results/baseline/evaluation_summary.json \
        --thesis results/thesis/evaluation_summary.json \
        --baseline-label "Transformer (77M)" \
        --thesis-label "Hybrid Mamba (77M)"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not installed")


# Publication-quality plot settings
PLOT_STYLE = {
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
}

# Color scheme for models
COLORS = {
    'baseline': '#E24A33',  # Red/Orange for Transformer
    'thesis': '#348ABD',    # Blue for Mamba
    'random': '#988ED5',    # Purple for random baseline
}

MARKERS = {
    'baseline': 's',  # Square
    'thesis': 'o',    # Circle
}


@dataclass
class ModelResults:
    """Aggregated results from a model evaluation."""
    name: str
    label: str

    # Quality metrics
    bleu: Optional[float] = None
    bleu_ci: Optional[Tuple[float, float]] = None
    comet: Optional[float] = None
    comet_ci: Optional[Tuple[float, float]] = None
    chrf: Optional[float] = None

    # Efficiency data (lists indexed by seq_len)
    seq_lengths: Optional[List[int]] = None
    throughput: Optional[List[float]] = None  # Tokens per second
    memory_gb: Optional[List[float]] = None
    ttft_ms: Optional[List[float]] = None
    itl_ms: Optional[List[float]] = None

    # ContraPro data
    contrapro_accuracy: Optional[float] = None
    contrapro_by_distance: Optional[Dict[str, float]] = None


def load_results(json_path: str, label: str) -> ModelResults:
    """Load evaluation results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = ModelResults(
        name=Path(json_path).stem,
        label=label,
    )

    # Quality metrics
    if 'quality' in data.get('results', data):
        q = data.get('results', data).get('quality', {})
        results.bleu = q.get('bleu')
        results.comet = q.get('comet')
        results.chrf = q.get('chrf')
        if q.get('bleu_ci_low') and q.get('bleu_ci_high'):
            results.bleu_ci = (q['bleu_ci_low'], q['bleu_ci_high'])
        if q.get('comet_ci_low') and q.get('comet_ci_high'):
            results.comet_ci = (q['comet_ci_low'], q['comet_ci_high'])

    # Efficiency metrics
    if 'efficiency' in data.get('results', data):
        e = data.get('results', data).get('efficiency', {})
        # Try to load from detailed results
        if 'seq_lengths' in e:
            results.seq_lengths = e['seq_lengths']
            results.throughput = e.get('throughput', e.get('tokens_per_second'))
            results.memory_gb = e.get('memory_gb', e.get('peak_memory_gb'))
            results.ttft_ms = e.get('ttft_ms', e.get('time_to_first_token_ms'))
            results.itl_ms = e.get('itl_ms', e.get('inter_token_latency_ms'))

    # ContraPro metrics
    if 'contrapro' in data.get('results', data):
        c = data.get('results', data).get('contrapro', {})
        results.contrapro_accuracy = c.get('accuracy')
        results.contrapro_by_distance = c.get('accuracy_by_distance')

    return results


def load_efficiency_csv(csv_path: str, label: str) -> ModelResults:
    """Load efficiency results from CSV file."""
    df = pd.read_csv(csv_path)

    # Filter for batch_size=1 for cleaner plots
    df_bs1 = df[df['batch_size'] == 1].sort_values('seq_len')

    results = ModelResults(
        name=Path(csv_path).stem,
        label=label,
        seq_lengths=df_bs1['seq_len'].tolist(),
        throughput=df_bs1['tokens_per_second'].tolist(),
        memory_gb=df_bs1['peak_memory_gb'].tolist(),
        ttft_ms=df_bs1['time_to_first_token_ms'].tolist(),
        itl_ms=df_bs1['inter_token_latency_ms'].tolist(),
    )

    return results


def plot_throughput_comparison(
    baseline: ModelResults,
    thesis: ModelResults,
    output_path: Path,
) -> None:
    """
    Generate Figure 2: Throughput vs. Sequence Length (Log-Log).

    This is the "Money Chart" showing Mamba's linear scaling advantage.
    """
    if not PLOTTING_AVAILABLE:
        return

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot baseline (Transformer)
    if baseline.seq_lengths and baseline.throughput:
        ax.loglog(
            baseline.seq_lengths, baseline.throughput,
            marker=MARKERS['baseline'], color=COLORS['baseline'],
            linewidth=2.5, markersize=10,
            label=baseline.label,
        )

    # Plot thesis (Mamba)
    if thesis.seq_lengths and thesis.throughput:
        ax.loglog(
            thesis.seq_lengths, thesis.throughput,
            marker=MARKERS['thesis'], color=COLORS['thesis'],
            linewidth=2.5, markersize=10,
            label=thesis.label,
        )

    # Add theoretical scaling lines
    if thesis.seq_lengths:
        x = np.array(thesis.seq_lengths)
        # O(L^2) reference line
        y_quad = thesis.throughput[0] * (thesis.seq_lengths[0] / x) ** 2
        ax.loglog(x, y_quad, '--', color='gray', alpha=0.5, label=r'$O(L^2)$ scaling')
        # O(L) reference line
        y_lin = thesis.throughput[0] * (thesis.seq_lengths[0] / x)
        ax.loglog(x, y_lin, ':', color='gray', alpha=0.5, label=r'$O(L)$ scaling')

    ax.set_xlabel('Sequence Length (tokens)', fontsize=14)
    ax.set_ylabel('Throughput (tokens/second)', fontsize=14)
    ax.set_title('Inference Throughput vs. Sequence Length\n(Batch Size = 1, H100 80GB)', fontsize=14)

    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Set nice log ticks
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()
    plt.savefig(output_path / 'figure2_throughput.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure2_throughput.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: figure2_throughput.pdf")


def plot_memory_comparison(
    baseline: ModelResults,
    thesis: ModelResults,
    output_path: Path,
) -> None:
    """Generate memory consumption comparison plot."""
    if not PLOTTING_AVAILABLE:
        return

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot baseline
    if baseline.seq_lengths and baseline.memory_gb:
        ax.plot(
            baseline.seq_lengths, baseline.memory_gb,
            marker=MARKERS['baseline'], color=COLORS['baseline'],
            linewidth=2.5, markersize=10,
            label=baseline.label,
        )

    # Plot thesis
    if thesis.seq_lengths and thesis.memory_gb:
        ax.plot(
            thesis.seq_lengths, thesis.memory_gb,
            marker=MARKERS['thesis'], color=COLORS['thesis'],
            linewidth=2.5, markersize=10,
            label=thesis.label,
        )

    # H100 memory limit
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
               label='H100 80GB Limit')

    ax.set_xlabel('Sequence Length (tokens)', fontsize=14)
    ax.set_ylabel('Peak GPU Memory (GB)', fontsize=14)
    ax.set_title('Memory Consumption vs. Sequence Length\n(Batch Size = 1)', fontsize=14)

    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 85)

    plt.tight_layout()
    plt.savefig(output_path / 'figure3_memory.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure3_memory.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: figure3_memory.pdf")


def plot_contrapro_comparison(
    baseline: ModelResults,
    thesis: ModelResults,
    output_path: Path,
) -> None:
    """
    Generate Figure 4: ContraPro Accuracy vs. Distance.

    Shows context utilization capability of each model.
    """
    if not PLOTTING_AVAILABLE:
        return

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Distance bucket centers
    bucket_centers = {
        "0-50": 25, "51-100": 75, "101-200": 150,
        "201-500": 350, "501-1000": 750, "1000+": 1500
    }

    def extract_xy(by_distance: Dict) -> Tuple[List, List]:
        x, y = [], []
        for bucket, acc in sorted(by_distance.items(), key=lambda kv: bucket_centers.get(kv[0], 0)):
            if acc is not None and bucket in bucket_centers:
                x.append(bucket_centers[bucket])
                y.append(acc * 100)
        return x, y

    # Plot baseline
    if baseline.contrapro_by_distance:
        x, y = extract_xy(baseline.contrapro_by_distance)
        ax.plot(x, y, marker=MARKERS['baseline'], color=COLORS['baseline'],
               linewidth=2.5, markersize=10, label=baseline.label)

    # Plot thesis
    if thesis.contrapro_by_distance:
        x, y = extract_xy(thesis.contrapro_by_distance)
        ax.plot(x, y, marker=MARKERS['thesis'], color=COLORS['thesis'],
               linewidth=2.5, markersize=10, label=thesis.label)

    # Random baseline
    ax.axhline(y=50, color=COLORS['random'], linestyle='--', alpha=0.7,
               linewidth=1.5, label='Random Baseline (50%)')

    ax.set_xlabel('Antecedent Distance (tokens)', fontsize=14)
    ax.set_ylabel('Pronoun Disambiguation Accuracy (%)', fontsize=14)
    ax.set_title('Context Utilization: Accuracy vs. Antecedent Distance', fontsize=14)

    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 100)
    ax.set_xlim(0, 1600)

    plt.tight_layout()
    plt.savefig(output_path / 'figure4_contrapro.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure4_contrapro.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: figure4_contrapro.pdf")


def plot_latency_breakdown(
    baseline: ModelResults,
    thesis: ModelResults,
    output_path: Path,
) -> None:
    """Generate TTFT vs ITL breakdown comparison."""
    if not PLOTTING_AVAILABLE:
        return

    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TTFT (Pre-fill)
    if baseline.seq_lengths and baseline.ttft_ms:
        axes[0].plot(baseline.seq_lengths, baseline.ttft_ms,
                    marker=MARKERS['baseline'], color=COLORS['baseline'],
                    linewidth=2.5, markersize=8, label=baseline.label)
    if thesis.seq_lengths and thesis.ttft_ms:
        axes[0].plot(thesis.seq_lengths, thesis.ttft_ms,
                    marker=MARKERS['thesis'], color=COLORS['thesis'],
                    linewidth=2.5, markersize=8, label=thesis.label)

    axes[0].set_xlabel('Sequence Length', fontsize=12)
    axes[0].set_ylabel('Time to First Token (ms)', fontsize=12)
    axes[0].set_title('Pre-fill Latency (TTFT)', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # ITL (Decoding)
    if baseline.seq_lengths and baseline.itl_ms:
        axes[1].plot(baseline.seq_lengths, baseline.itl_ms,
                    marker=MARKERS['baseline'], color=COLORS['baseline'],
                    linewidth=2.5, markersize=8, label=baseline.label)
    if thesis.seq_lengths and thesis.itl_ms:
        axes[1].plot(thesis.seq_lengths, thesis.itl_ms,
                    marker=MARKERS['thesis'], color=COLORS['thesis'],
                    linewidth=2.5, markersize=8, label=thesis.label)

    axes[1].set_xlabel('Sequence Length', fontsize=12)
    axes[1].set_ylabel('Inter-Token Latency (ms)', fontsize=12)
    axes[1].set_title('Decoding Latency (ITL)', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'figure5_latency.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure5_latency.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: figure5_latency.pdf")


def generate_quality_table(
    baseline: ModelResults,
    thesis: ModelResults,
    output_path: Path,
) -> None:
    """Generate Table 1: Translation Quality Comparison (LaTeX)."""

    def fmt_metric(val, ci=None):
        if val is None:
            return "N/A"
        if ci:
            return f"{val:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
        return f"{val:.2f}"

    def fmt_comet(val, ci=None):
        if val is None:
            return "N/A"
        if ci:
            return f"{val:.4f}"
        return f"{val:.4f}"

    # Determine winners
    bleu_winner = None
    if baseline.bleu and thesis.bleu:
        bleu_winner = 'thesis' if thesis.bleu > baseline.bleu else 'baseline'

    comet_winner = None
    if baseline.comet and thesis.comet:
        comet_winner = 'thesis' if thesis.comet > baseline.comet else 'baseline'

    # Generate LaTeX
    latex = r"""
\begin{table}[t]
\centering
\caption{Translation Quality on IWSLT14 De$\to$En Test Set.
Best scores in \textbf{bold}. 95\% confidence intervals from paired bootstrap resampling.}
\label{tab:quality}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Params} & \textbf{BLEU} ($\uparrow$) & \textbf{COMET} ($\uparrow$) & \textbf{ChrF++} ($\uparrow$) \\
\midrule
"""

    # Baseline row
    bleu_base = fmt_metric(baseline.bleu, baseline.bleu_ci)
    comet_base = fmt_comet(baseline.comet, baseline.comet_ci)
    chrf_base = fmt_metric(baseline.chrf)

    if bleu_winner == 'baseline':
        bleu_base = f"\\textbf{{{bleu_base}}}"
    if comet_winner == 'baseline':
        comet_base = f"\\textbf{{{comet_base}}}"

    latex += f"{baseline.label} & 77M & {bleu_base} & {comet_base} & {chrf_base} \\\\\n"

    # Thesis row
    bleu_thesis = fmt_metric(thesis.bleu, thesis.bleu_ci)
    comet_thesis = fmt_comet(thesis.comet, thesis.comet_ci)
    chrf_thesis = fmt_metric(thesis.chrf)

    if bleu_winner == 'thesis':
        bleu_thesis = f"\\textbf{{{bleu_thesis}}}"
    if comet_winner == 'thesis':
        comet_thesis = f"\\textbf{{{comet_thesis}}}"

    latex += f"{thesis.label} & 77M & {bleu_thesis} & {comet_thesis} & {chrf_thesis} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # Save
    table_path = output_path / 'table1_quality.tex'
    with open(table_path, 'w') as f:
        f.write(latex)

    print(f"  Saved: table1_quality.tex")

    # Also print for quick viewing
    print("\n" + "="*70)
    print("TABLE 1: TRANSLATION QUALITY COMPARISON")
    print("="*70)
    print(f"{'Model':<30} {'BLEU':>12} {'COMET':>12} {'ChrF++':>12}")
    print("-"*70)
    print(f"{baseline.label:<30} {fmt_metric(baseline.bleu):>12} {fmt_comet(baseline.comet):>12} {fmt_metric(baseline.chrf):>12}")
    print(f"{thesis.label:<30} {fmt_metric(thesis.bleu):>12} {fmt_comet(thesis.comet):>12} {fmt_metric(thesis.chrf):>12}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Generate Comparison Plots")
    parser.add_argument("--baseline", type=str, required=True,
                       help="Path to baseline evaluation_summary.json")
    parser.add_argument("--thesis", type=str, required=True,
                       help="Path to thesis evaluation_summary.json")
    parser.add_argument("--output-dir", type=str, default="paper/figures",
                       help="Output directory for figures")
    parser.add_argument("--baseline-label", type=str, default="Transformer Baseline",
                       help="Label for baseline model in plots")
    parser.add_argument("--thesis-label", type=str, default="Hybrid Mamba-Attention",
                       help="Label for thesis model in plots")
    parser.add_argument("--efficiency-csv-baseline", type=str,
                       help="Optional: efficiency_results.csv for baseline (more detailed)")
    parser.add_argument("--efficiency-csv-thesis", type=str,
                       help="Optional: efficiency_results.csv for thesis (more detailed)")

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("GENERATING COMPARISON PLOTS")
    print("="*60)
    print(f"Baseline: {args.baseline}")
    print(f"Thesis: {args.thesis}")
    print(f"Output: {args.output_dir}")
    print("="*60 + "\n")

    # Load results
    baseline = load_results(args.baseline, args.baseline_label)
    thesis = load_results(args.thesis, args.thesis_label)

    # Optionally load more detailed efficiency CSVs
    if args.efficiency_csv_baseline:
        eff_baseline = load_efficiency_csv(args.efficiency_csv_baseline, args.baseline_label)
        baseline.seq_lengths = eff_baseline.seq_lengths
        baseline.throughput = eff_baseline.throughput
        baseline.memory_gb = eff_baseline.memory_gb
        baseline.ttft_ms = eff_baseline.ttft_ms
        baseline.itl_ms = eff_baseline.itl_ms

    if args.efficiency_csv_thesis:
        eff_thesis = load_efficiency_csv(args.efficiency_csv_thesis, args.thesis_label)
        thesis.seq_lengths = eff_thesis.seq_lengths
        thesis.throughput = eff_thesis.throughput
        thesis.memory_gb = eff_thesis.memory_gb
        thesis.ttft_ms = eff_thesis.ttft_ms
        thesis.itl_ms = eff_thesis.itl_ms

    # Generate all plots
    print("Generating plots...")

    if baseline.seq_lengths or thesis.seq_lengths:
        plot_throughput_comparison(baseline, thesis, output_path)
        plot_memory_comparison(baseline, thesis, output_path)
        plot_latency_breakdown(baseline, thesis, output_path)

    if baseline.contrapro_by_distance or thesis.contrapro_by_distance:
        plot_contrapro_comparison(baseline, thesis, output_path)

    # Generate quality table
    print("\nGenerating tables...")
    generate_quality_table(baseline, thesis, output_path)

    print(f"\nAll artifacts saved to: {output_path}")


if __name__ == "__main__":
    main()
