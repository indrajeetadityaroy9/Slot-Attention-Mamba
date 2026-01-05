#!/usr/bin/env python3
"""
Master Experiment Runner for NeurIPS/ICML Evaluation Protocol.

Optimized for single H100 80GB with 26 CPU cores and 221GB RAM.
Runs experiments sequentially with maximum GPU utilization.

Usage:
    # Run all phases
    python scripts/run_experiments.py --all

    # Run specific phase
    python scripts/run_experiments.py --phase 1  # MQAR
    python scripts/run_experiments.py --phase 2  # IWSLT
    python scripts/run_experiments.py --phase 3  # OPUS + ContraPro
    python scripts/run_experiments.py --phase 5  # Ablations

    # Dry run (show commands without executing)
    python scripts/run_experiments.py --all --dry-run
"""

import argparse
import subprocess
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional


# =============================================================================
# Experiment Definitions
# =============================================================================

EXPERIMENTS = {
    # Phase 1: MQAR State Capacity (Figure 1)
    "phase1": [
        {
            "name": "MQAR Hybrid",
            "cmd": "python scripts/train.py experiment=mqar_state_sweep",
            "description": "Tests hybrid Mamba-Attention on MQAR (should maintain >98% accuracy)",
            "estimated_time": "~2 hours",
        },
        {
            "name": "MQAR Pure Mamba",
            "cmd": "python scripts/train.py experiment=mqar_pure_mamba",
            "description": "Tests pure Mamba baseline (should show cliff at num_pairs=128)",
            "estimated_time": "~2 hours",
        },
    ],

    # Phase 2: IWSLT De-En Quality (Table 1)
    "phase2": [
        {
            "name": "IWSLT Transformer Baseline",
            "cmd": "python scripts/train.py experiment=main_iwslt_baseline",
            "description": "Pure Transformer baseline on IWSLT14 De-En",
            "estimated_time": "~8 hours",
        },
        {
            "name": "IWSLT Hybrid",
            "cmd": "python scripts/train.py experiment=main_iwslt_hybrid",
            "description": "Hybrid Mamba-Attention on IWSLT14 De-En",
            "estimated_time": "~8 hours",
        },
    ],

    # Phase 3: OPUS En-De with ContraPro (Table 2)
    "phase3": [
        {
            "name": "Setup ContraPro",
            "cmd": "python scripts/setup_contrapro.py",
            "description": "Download and prepare ContraPro dataset",
            "estimated_time": "~1 minute",
        },
        {
            "name": "OPUS En-De Hybrid",
            "cmd": "python scripts/train.py experiment=main_opus_hybrid data=opus_books_en_de",
            "description": "Hybrid Mamba-Attention on OPUS Books En-De (for ContraPro)",
            "estimated_time": "~12 hours",
        },
        {
            "name": "OPUS En-De Pure Mamba",
            "cmd": "python scripts/train.py experiment=main_opus_hybrid data=opus_books_en_de model.custom_hybrid_positions=[]",
            "description": "Pure Mamba on OPUS Books En-De (baseline for ContraPro)",
            "estimated_time": "~12 hours",
        },
    ],

    # Phase 4: Efficiency Benchmarks (Figure 3)
    "phase4": [
        {
            "name": "Inference Benchmarks",
            "cmd": "python scripts/benchmark.py --inference",
            "description": "Run throughput/latency benchmarks",
            "estimated_time": "~30 minutes",
        },
    ],

    # Phase 5: Ablation Studies (Table 3)
    "phase5": [
        {
            "name": "Ablation: Blind Start (No Layer 0)",
            "cmd": "python scripts/train.py experiment=ablation_no_hybrid_layer0",
            "description": "Tests importance of layer 0 HYBRID block",
            "estimated_time": "~8 hours",
        },
        {
            "name": "Ablation: Ratio 1:4",
            "cmd": "python scripts/train.py experiment=ablation_hybrid_ratio_4",
            "description": "Tests denser HYBRID ratio (every 4 layers)",
            "estimated_time": "~8 hours",
        },
        {
            "name": "Ablation: Pure Mamba (IWSLT)",
            "cmd": "python scripts/train.py experiment=ablation_no_cross_attn",
            "description": "Tests pure Mamba decoder (no cross-attention)",
            "estimated_time": "~8 hours",
        },
    ],
}


# =============================================================================
# Runner Functions
# =============================================================================

def print_header(text: str):
    """Print formatted header."""
    width = 70
    print("\n" + "=" * width)
    print(f" {text}")
    print("=" * width)


def print_status(text: str, status: str = "INFO"):
    """Print status message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    colors = {
        "INFO": "\033[94m",      # Blue
        "SUCCESS": "\033[92m",   # Green
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "RESET": "\033[0m",
    }
    color = colors.get(status, colors["INFO"])
    reset = colors["RESET"]
    print(f"[{timestamp}] {color}{status}{reset}: {text}")


def run_command(cmd: str, dry_run: bool = False) -> Tuple[bool, float]:
    """
    Run a shell command.

    Returns:
        Tuple of (success, elapsed_time_seconds)
    """
    if dry_run:
        print(f"  [DRY RUN] Would execute: {cmd}")
        return True, 0.0

    start_time = time.time()
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Stream output
        for line in process.stdout:
            print(f"  {line}", end="")

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode == 0:
            return True, elapsed
        else:
            return False, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print_status(f"Command failed with exception: {e}", "ERROR")
        return False, elapsed


def run_phase(phase_name: str, experiments: List[dict], dry_run: bool = False) -> List[dict]:
    """
    Run all experiments in a phase.

    Returns:
        List of result dicts with success status and timing
    """
    results = []

    for i, exp in enumerate(experiments, 1):
        print_header(f"{phase_name} - Experiment {i}/{len(experiments)}: {exp['name']}")
        print(f"  Description: {exp['description']}")
        print(f"  Estimated time: {exp['estimated_time']}")
        print(f"  Command: {exp['cmd']}")
        print()

        success, elapsed = run_command(exp['cmd'], dry_run)

        result = {
            "name": exp['name'],
            "success": success,
            "elapsed_seconds": elapsed,
            "elapsed_human": format_time(elapsed),
        }
        results.append(result)

        if success:
            print_status(f"Completed in {result['elapsed_human']}", "SUCCESS")
        else:
            print_status(f"Failed after {result['elapsed_human']}", "ERROR")

    return results


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def check_gpu_available() -> bool:
    """Check if GPU is available and not in use."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        memory_used = int(result.stdout.strip())
        return memory_used < 1000  # Less than 1GB used
    except Exception:
        return False


def print_summary(all_results: dict):
    """Print summary of all experiments."""
    print_header("EXPERIMENT SUMMARY")

    total_time = 0
    total_success = 0
    total_failed = 0

    for phase_name, results in all_results.items():
        print(f"\n{phase_name}:")
        for result in results:
            status = "PASS" if result['success'] else "FAIL"
            status_color = "\033[92m" if result['success'] else "\033[91m"
            reset = "\033[0m"
            print(f"  [{status_color}{status}{reset}] {result['name']} ({result['elapsed_human']})")

            total_time += result['elapsed_seconds']
            if result['success']:
                total_success += 1
            else:
                total_failed += 1

    print(f"\nTotal: {total_success} passed, {total_failed} failed")
    print(f"Total time: {format_time(total_time)}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Master Experiment Runner for NeurIPS/ICML Evaluation Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_experiments.py --all              # Run everything
    python scripts/run_experiments.py --phase 1          # MQAR only
    python scripts/run_experiments.py --phase 1 2        # MQAR + IWSLT
    python scripts/run_experiments.py --all --dry-run    # Show commands only
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all phases (1-5)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4, 5],
        help="Run specific phase(s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing",
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip GPU availability check",
    )

    args = parser.parse_args()

    # Determine which phases to run
    if args.all:
        phases_to_run = [1, 2, 3, 4, 5]
    elif args.phase:
        phases_to_run = args.phase
    else:
        parser.print_help()
        sys.exit(1)

    # Check GPU availability
    if not args.dry_run and not args.skip_gpu_check:
        if not check_gpu_available():
            print_status("GPU appears to be in use. Use --skip-gpu-check to override.", "WARNING")
            response = input("Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                sys.exit(1)

    # Print plan
    print_header("EXPERIMENT PLAN")
    for phase_num in phases_to_run:
        phase_key = f"phase{phase_num}"
        experiments = EXPERIMENTS.get(phase_key, [])
        print(f"\nPhase {phase_num} ({len(experiments)} experiments):")
        for exp in experiments:
            print(f"  - {exp['name']} ({exp['estimated_time']})")

    if not args.dry_run:
        response = input("\nProceed with experiments? [Y/n]: ")
        if response.lower() == 'n':
            print("Aborted.")
            sys.exit(0)

    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    print_status(f"Working directory: {project_dir}", "INFO")

    # Run experiments
    all_results = {}
    start_time = time.time()

    for phase_num in phases_to_run:
        phase_key = f"phase{phase_num}"
        phase_name = f"Phase {phase_num}"
        experiments = EXPERIMENTS.get(phase_key, [])

        if not experiments:
            print_status(f"No experiments defined for {phase_name}", "WARNING")
            continue

        print_header(f"STARTING {phase_name.upper()}")
        results = run_phase(phase_name, experiments, args.dry_run)
        all_results[phase_name] = results

    # Print summary
    total_time = time.time() - start_time
    print_summary(all_results)
    print(f"\nWall clock time: {format_time(total_time)}")


if __name__ == "__main__":
    main()
