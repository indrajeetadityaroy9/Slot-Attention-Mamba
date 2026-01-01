#!/usr/bin/env python3
"""
Ablation Study Runner for Hybrid Mamba-Attention NMT.

Systematic ablation studies for ICML-level rigor:
1. Bi-Directionality Check: BiMamba Encoder vs. UniMamba Encoder
2. Hybrid Ratio: 1:7 (Hybrid) vs. Pure Mamba vs. Pure Attention
3. Data Strategy: CAT-N Augmentation vs. Sentence-Level training

Usage:
    python scripts/evaluation/run_ablations.py --ablation bidirectional
    python scripts/evaluation/run_ablations.py --ablation hybrid-ratio
    python scripts/evaluation/run_ablations.py --ablation data-strategy
    python scripts/evaluation/run_ablations.py --ablation all
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str

    # Model config overrides
    model_overrides: Dict = field(default_factory=dict)

    # Training config overrides
    training_overrides: Dict = field(default_factory=dict)

    # Data config overrides
    data_overrides: Dict = field(default_factory=dict)


@dataclass
class AblationResult:
    """Result from an ablation experiment."""
    name: str
    config: AblationConfig

    # Training metrics
    final_train_loss: float
    final_val_loss: float
    training_time_hours: float

    # Quality metrics (from evaluation)
    bleu: Optional[float] = None
    comet: Optional[float] = None
    chrf: Optional[float] = None

    # Efficiency metrics
    tokens_per_second: Optional[float] = None
    peak_memory_gb: Optional[float] = None


# ============================================================================
# Ablation 1: Bidirectionality Check
# ============================================================================

BIDIRECTIONAL_ABLATIONS = [
    AblationConfig(
        name="bimamba_encoder",
        description="BiMamba encoder (bidirectional, standard)",
        model_overrides={
            "encoder_bidirectional": True,
        },
    ),
    AblationConfig(
        name="unimamba_encoder",
        description="UniMamba encoder (unidirectional, GPT-style)",
        model_overrides={
            "encoder_bidirectional": False,
        },
    ),
]


# ============================================================================
# Ablation 2: Hybrid Ratio
# ============================================================================

HYBRID_RATIO_ABLATIONS = [
    AblationConfig(
        name="pure_mamba",
        description="Pure Mamba (no attention layers)",
        model_overrides={
            "attention_ratio": 0.0,
        },
    ),
    AblationConfig(
        name="hybrid_1_7",
        description="1:7 Hybrid (Jamba-style, 12.5% attention)",
        model_overrides={
            "attention_ratio": 0.125,
        },
    ),
    AblationConfig(
        name="hybrid_1_3",
        description="1:3 Hybrid (25% attention)",
        model_overrides={
            "attention_ratio": 0.25,
        },
    ),
    AblationConfig(
        name="hybrid_1_1",
        description="1:1 Hybrid (50% attention)",
        model_overrides={
            "attention_ratio": 0.5,
        },
    ),
    AblationConfig(
        name="pure_attention",
        description="Pure Attention (Transformer baseline)",
        model_overrides={
            "attention_ratio": 1.0,
        },
    ),
]


# ============================================================================
# Ablation 3: Data Strategy
# ============================================================================

DATA_STRATEGY_ABLATIONS = [
    AblationConfig(
        name="sentence_level",
        description="Sentence-level training (no concatenation)",
        data_overrides={
            "cat_n": 1,
            "p_concat": 0.0,
        },
    ),
    AblationConfig(
        name="cat_3",
        description="CAT-3 augmentation",
        data_overrides={
            "cat_n": 3,
            "p_concat": 0.5,
        },
    ),
    AblationConfig(
        name="cat_5",
        description="CAT-5 augmentation (standard)",
        data_overrides={
            "cat_n": 5,
            "p_concat": 0.5,
        },
    ),
    AblationConfig(
        name="cat_10",
        description="CAT-10 augmentation (aggressive)",
        data_overrides={
            "cat_n": 10,
            "p_concat": 0.7,
        },
    ),
]


# ============================================================================
# Ablation 4: Cross-Attention Frequency
# ============================================================================

CROSS_ATTN_ABLATIONS = [
    AblationConfig(
        name="cross_every_2",
        description="Cross-attention every 2 layers",
        model_overrides={
            "cross_attn_every": 2,
        },
    ),
    AblationConfig(
        name="cross_every_4",
        description="Cross-attention every 4 layers (standard)",
        model_overrides={
            "cross_attn_every": 4,
        },
    ),
    AblationConfig(
        name="cross_every_8",
        description="Cross-attention every 8 layers",
        model_overrides={
            "cross_attn_every": 8,
        },
    ),
    AblationConfig(
        name="cross_final_only",
        description="Cross-attention only in final layer",
        model_overrides={
            "cross_attn_every": 999,  # Effectively only final layer
        },
    ),
]


def build_hydra_overrides(ablation: AblationConfig) -> List[str]:
    """Convert ablation config to Hydra command-line overrides."""
    overrides = []

    for key, value in ablation.model_overrides.items():
        overrides.append(f"model.{key}={value}")

    for key, value in ablation.training_overrides.items():
        overrides.append(f"training.{key}={value}")

    for key, value in ablation.data_overrides.items():
        overrides.append(f"data.{key}={value}")

    return overrides


def run_training(
    ablation: AblationConfig,
    base_output_dir: str = "experiments/ablations",
    max_steps: int = 10000,
    eval_steps: int = 500,
    model_size: str = "small",
    num_gpus: int = 2,
    dry_run: bool = False,
) -> Optional[str]:
    """
    Run training for an ablation experiment.

    Returns:
        Path to the best checkpoint, or None if failed.
    """
    output_dir = Path(base_output_dir) / ablation.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    overrides = build_hydra_overrides(ablation)
    overrides.extend([
        f"training.output_dir={output_dir}",
        f"training.max_steps={max_steps}",
        f"training.eval_steps={eval_steps}",
        f"model={model_size}",
    ])

    if num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "scripts/train.py",
        ] + overrides
    else:
        cmd = ["python", "scripts/train.py"] + overrides

    print(f"\n{'='*60}")
    print(f"Running Ablation: {ablation.name}")
    print(f"Description: {ablation.description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] Would execute the above command")
        return None

    # Save ablation config
    config_path = output_dir / "ablation_config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(ablation), f, indent=2)

    # Run training
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=False,
            text=True,
        )

        training_time = (time.time() - start_time) / 3600  # hours

        # Find best checkpoint
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            best_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
            return str(best_checkpoint)
        else:
            # Look for final model
            final_model = output_dir / "model_final.pt"
            if final_model.exists():
                return str(final_model)

    except Exception as e:
        print(f"Training failed: {e}")

    return None


def run_evaluation(
    checkpoint_path: str,
    ablation: AblationConfig,
    output_dir: str,
) -> Dict:
    """Run quality and efficiency evaluation on a checkpoint."""
    from scripts.evaluation.evaluate_quality import (
        TranslationEvaluator,
        load_test_data,
        generate_translations,
    )
    from scripts.evaluation.benchmark_inference import (
        run_benchmark_sweep,
        BenchmarkConfig,
    )

    results = {}

    # Load model for evaluation
    import torch
    from models import ModelConfig, HybridMambaEncoderDecoder
    from data import create_tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = create_tokenizer(
        tokenizer_type="custom",
        tokenizer_path="data/tokenizer/tokenizer.json",
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    if isinstance(config, dict):
        config = ModelConfig(**config)

    model = HybridMambaEncoderDecoder(config=config, device=device, dtype=torch.bfloat16)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Quality evaluation
    try:
        sources, references = load_test_data("iwslt14", "test", "de", "en")
        hypotheses = generate_translations(model, tokenizer, sources[:500])

        evaluator = TranslationEvaluator(device=device)
        quality_result = evaluator.evaluate(
            sources[:500], hypotheses, references[:500],
            model_name=ablation.name,
        )

        results["bleu"] = quality_result.bleu
        results["comet"] = quality_result.comet
        results["chrf"] = quality_result.chrf_plus_plus
    except Exception as e:
        print(f"Quality evaluation failed: {e}")

    # Efficiency evaluation (quick)
    try:
        bench_config = BenchmarkConfig(
            seq_lengths=[512, 1024, 2048],
            batch_sizes=[1],
            benchmark_runs=5,
        )
        bench_results = run_benchmark_sweep(model, ablation.name, bench_config)

        if bench_results:
            results["tokens_per_second"] = max(r.tokens_per_second for r in bench_results)
            results["peak_memory_gb"] = max(r.peak_memory_gb for r in bench_results)
    except Exception as e:
        print(f"Efficiency evaluation failed: {e}")

    return results


def generate_ablation_table(
    results: List[AblationResult],
    output_dir: str,
    ablation_type: str,
) -> None:
    """Generate LaTeX table for paper."""
    import pandas as pd

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    data = []
    for r in results:
        data.append({
            "Configuration": r.name,
            "Description": r.config.description,
            "BLEU": f"{r.bleu:.2f}" if r.bleu else "N/A",
            "COMET": f"{r.comet:.4f}" if r.comet else "N/A",
            "TPS": f"{r.tokens_per_second:.1f}" if r.tokens_per_second else "N/A",
            "Memory (GB)": f"{r.peak_memory_gb:.2f}" if r.peak_memory_gb else "N/A",
        })

    df = pd.DataFrame(data)

    # Save CSV
    csv_path = output_path / f"ablation_{ablation_type}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Generate LaTeX
    latex_path = output_path / f"ablation_{ablation_type}.tex"
    latex_content = df.to_latex(index=False, escape=True)

    with open(latex_path, 'w') as f:
        f.write("% Auto-generated ablation table\n")
        f.write(f"% Ablation type: {ablation_type}\n")
        f.write(latex_content)
    print(f"Saved: {latex_path}")

    # Print table
    print(f"\n{'='*80}")
    print(f"ABLATION RESULTS: {ablation_type.upper()}")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Studies")
    parser.add_argument("--ablation", type=str, required=True,
                       choices=["bidirectional", "hybrid-ratio", "data-strategy",
                               "cross-attention", "all"],
                       help="Which ablation study to run")
    parser.add_argument("--output-dir", type=str, default="experiments/ablations",
                       help="Output directory for results")
    parser.add_argument("--max-steps", type=int, default=10000,
                       help="Max training steps per ablation")
    parser.add_argument("--eval-steps", type=int, default=500,
                       help="Evaluation frequency")
    parser.add_argument("--model-size", type=str, default="small",
                       choices=["small", "medium", "large"],
                       help="Model size for ablations")
    parser.add_argument("--num-gpus", type=int, default=2,
                       help="Number of GPUs to use")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training, only run evaluation on existing checkpoints")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation, only run training")

    args = parser.parse_args()

    # Select ablation configurations
    ablation_map = {
        "bidirectional": BIDIRECTIONAL_ABLATIONS,
        "hybrid-ratio": HYBRID_RATIO_ABLATIONS,
        "data-strategy": DATA_STRATEGY_ABLATIONS,
        "cross-attention": CROSS_ATTN_ABLATIONS,
    }

    if args.ablation == "all":
        ablations_to_run = []
        for ablations in ablation_map.values():
            ablations_to_run.extend(ablations)
    else:
        ablations_to_run = ablation_map[args.ablation]

    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: {args.ablation.upper()}")
    print(f"Number of experiments: {len(ablations_to_run)}")
    print(f"Model size: {args.model_size}")
    print(f"Max steps: {args.max_steps}")
    print(f"{'='*60}\n")

    results = []

    for ablation in ablations_to_run:
        checkpoint_path = None

        # Training phase
        if not args.skip_training:
            checkpoint_path = run_training(
                ablation,
                base_output_dir=args.output_dir,
                max_steps=args.max_steps,
                eval_steps=args.eval_steps,
                model_size=args.model_size,
                num_gpus=args.num_gpus,
                dry_run=args.dry_run,
            )
        else:
            # Look for existing checkpoint
            ablation_dir = Path(args.output_dir) / ablation.name
            checkpoints = list(ablation_dir.glob("checkpoint-*"))
            if checkpoints:
                checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))

        # Evaluation phase
        if checkpoint_path and not args.skip_evaluation and not args.dry_run:
            eval_results = run_evaluation(
                checkpoint_path,
                ablation,
                args.output_dir,
            )

            results.append(AblationResult(
                name=ablation.name,
                config=ablation,
                final_train_loss=0.0,  # TODO: Extract from logs
                final_val_loss=0.0,
                training_time_hours=0.0,
                **eval_results,
            ))

    # Generate summary table
    if results and not args.dry_run:
        generate_ablation_table(results, args.output_dir, args.ablation)

        # Save all results
        results_path = Path(args.output_dir) / f"ablation_{args.ablation}_results.json"
        with open(results_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nSaved all results: {results_path}")


if __name__ == "__main__":
    main()
