#!/usr/bin/env python3
"""Aggregate multi-seed experiment results per literature methodology.

Reports statistics as specified in key papers:
- Mean accuracy across seeds (Revisiting AR, arXiv 2508.19029)
- Standard error (Understanding Input Selectivity, arXiv 2506.11891)
- Max accuracy (Zoology, arXiv 2312.04927)
- Max-min range (Revisiting AR)

Usage:
    python scripts/aggregate_results.py --output_dir outputs/
    python scripts/aggregate_results.py --output_dir outputs/ --format latex
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np


def load_results(output_dir: Path) -> Dict[tuple, List[Dict[str, Any]]]:
    """Load and group results by (d_state, num_pairs, hybrid_positions)."""
    results = defaultdict(list)

    for results_file in output_dir.rglob("mqar_results.json"):
        try:
            with open(results_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read {results_file}: {e}")
            continue

        # Create grouping key
        hybrid = data.get("hybrid_positions")
        if hybrid is None:
            hybrid_str = "[]"
        else:
            hybrid_str = str(sorted(hybrid))

        key = (
            data.get("d_state"),
            data.get("num_pairs"),
            hybrid_str,
        )

        results[key].append({
            "token_accuracy": data.get("token_accuracy", 0),
            "sample_accuracy": data.get("sample_accuracy", 0),
            "perplexity": data.get("perplexity"),
            "seed": data.get("seed"),
            "path": str(results_file),
        })

    return results


def compute_statistics(accuracies: List[float]) -> Dict[str, float]:
    """Compute literature-standard statistics.

    Per Revisiting AR: "report mean and relative max-min errors"
    Per Input Selectivity: "standard error across 3 seeds"
    Per Zoology: "present the maximum accuracy across runs"
    """
    n = len(accuracies)
    if n == 0:
        return {"mean": 0, "sem": 0, "max": 0, "range": 0, "n": 0}

    mean = np.mean(accuracies)
    std = np.std(accuracies, ddof=1) if n > 1 else 0
    sem = std / np.sqrt(n) if n > 1 else 0
    max_acc = np.max(accuracies)
    acc_range = np.max(accuracies) - np.min(accuracies)

    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "max": max_acc,
        "min": np.min(accuracies),
        "range": acc_range,
        "n": n,
    }


def print_table(results: Dict[tuple, List[Dict[str, Any]]], metric: str = "token_accuracy"):
    """Print results table in console format."""
    print("\n" + "=" * 90)
    print(f"AGGREGATED RESULTS - {metric.upper()}")
    print("=" * 90)
    print(f"{'d_state':<8} {'pairs':<6} {'hybrid':<20} {'mean':<10} {'SEM':<8} {'max':<8} {'range':<8} {'n':<4}")
    print("-" * 90)

    for (d_state, num_pairs, hybrid), runs in sorted(results.items()):
        accs = [r[metric] for r in runs if r.get(metric) is not None]
        if not accs:
            continue

        stats = compute_statistics(accs)

        # Detect capacity cliff (accuracy < 50% when num_pairs > d_state)
        cliff_marker = ""
        if d_state is not None and num_pairs is not None:
            if num_pairs > d_state and stats["mean"] < 0.5 and hybrid == "[]":
                cliff_marker = " <- CLIFF"
            elif num_pairs > d_state and stats["mean"] > 0.9 and hybrid != "[]":
                cliff_marker = " <- RESCUED"

        print(f"{d_state:<8} {num_pairs:<6} {hybrid:<20} "
              f"{stats['mean']*100:>7.2f}% {stats['sem']*100:>6.2f}% "
              f"{stats['max']*100:>6.2f}% {stats['range']*100:>6.2f}% "
              f"{stats['n']:<4}{cliff_marker}")

    print("=" * 90)
    print("Columns: mean (Revisiting AR), SEM (Input Selectivity), max (Zoology), range (Revisiting AR)")
    print("Markers: CLIFF = capacity exceeded, RESCUED = hybrid fixed capacity limit")


def print_latex(results: Dict[tuple, List[Dict[str, Any]]], metric: str = "token_accuracy"):
    """Print results in LaTeX table format."""
    print("\n% LaTeX table for MQAR results")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{cclcc}")
    print("\\toprule")
    print("$d_{state}$ & $\\kappa$ & Hybrid & Accuracy (\\%) & $n$ \\\\")
    print("\\midrule")

    for (d_state, num_pairs, hybrid), runs in sorted(results.items()):
        accs = [r[metric] for r in runs if r.get(metric) is not None]
        if not accs:
            continue

        stats = compute_statistics(accs)
        hybrid_fmt = hybrid.replace("[", "\\{").replace("]", "\\}")

        print(f"{d_state} & {num_pairs} & {hybrid_fmt} & "
              f"${stats['mean']*100:.1f} \\pm {stats['sem']*100:.1f}$ & {stats['n']} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{MQAR accuracy (mean $\\pm$ SEM) across seeds per literature methodology.}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed experiment results per literature methodology"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing experiment outputs",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="token_accuracy",
        choices=["token_accuracy", "sample_accuracy"],
        help="Metric to aggregate",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="console",
        choices=["console", "latex", "both"],
        help="Output format",
    )
    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"Error: Output directory not found: {args.output_dir}")
        return

    results = load_results(args.output_dir)

    if not results:
        print(f"No results found in {args.output_dir}")
        return

    print(f"\nFound {sum(len(v) for v in results.values())} result files "
          f"across {len(results)} configurations")

    if args.format in ("console", "both"):
        print_table(results, args.metric)

    if args.format in ("latex", "both"):
        print_latex(results, args.metric)


if __name__ == "__main__":
    main()
