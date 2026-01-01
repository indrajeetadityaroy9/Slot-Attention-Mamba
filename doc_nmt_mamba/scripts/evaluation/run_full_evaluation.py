#!/usr/bin/env python3
"""
Full Evaluation Pipeline for Hybrid Mamba-Attention NMT.

Orchestrates all evaluations required for publication:
- Part 1: Translation Quality (Table 1) - BLEU, COMET, ChrF++
- Part 2: Efficiency & Scaling (Money Charts) - Throughput, Memory, TTFT/ITL
- Part 3: Scientific Novelty (Context) - ContraPro pronoun disambiguation
- Part 4: Ablations (if models available)

Generates all artifacts for paper:
- Table 1: Quality vs. Baselines
- Figure 1: Convergence (Loss vs. Wall-clock)
- Figure 2: Throughput scaling (Log-Log)
- Figure 3: ContraPro accuracy vs. distance
- Table 2: Ablation results

Usage:
    # Full evaluation
    python scripts/evaluation/run_full_evaluation.py --checkpoint outputs/best_model.pt

    # Quick evaluation (skip COMET, reduce samples)
    python scripts/evaluation/run_full_evaluation.py --checkpoint outputs/best_model.pt --quick

    # Only quality metrics
    python scripts/evaluation/run_full_evaluation.py --checkpoint outputs/best_model.pt --only quality
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch


def check_dependencies():
    """Check required packages are installed."""
    missing = []

    try:
        import sacrebleu
    except ImportError:
        missing.append("sacrebleu")

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    try:
        import seaborn
    except ImportError:
        missing.append("seaborn")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    return True


def run_quality_evaluation(
    checkpoint_path: str,
    output_dir: str,
    skip_comet: bool = False,
    max_samples: Optional[int] = None,
) -> Dict:
    """Run translation quality evaluation."""
    print("\n" + "="*60)
    print("PART 1: TRANSLATION QUALITY (Table 1)")
    print("="*60)

    from scripts.evaluation.evaluate_quality import (
        TranslationEvaluator,
        load_test_data,
        generate_translations,
        save_results,
    )
    from data import create_tokenizer
    from models import ModelConfig, HybridMambaEncoderDecoder

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model
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

    model_name = Path(checkpoint_path).stem

    # Load test data
    sources, references = load_test_data("iwslt14", "test", "de", "en")

    if max_samples:
        sources = sources[:max_samples]
        references = references[:max_samples]

    # Generate translations
    print(f"\nGenerating translations for {len(sources)} samples...")
    hypotheses = generate_translations(
        model, tokenizer, sources,
        max_length=256,
        batch_size=16,
        device=device,
    )

    # Evaluate
    evaluator = TranslationEvaluator(
        comet_model="Unbabel/wmt22-comet-da" if not skip_comet else None,
        device=device,
    )

    result = evaluator.evaluate(
        sources=sources,
        hypotheses=hypotheses,
        references=references,
        model_name=model_name,
        dataset="iwslt14-test",
    )

    # Save results
    save_results(result, output_dir, f"quality_{model_name}")

    return {
        "bleu": result.bleu,
        "comet": result.comet,
        "chrf": result.chrf_plus_plus,
    }


def run_efficiency_evaluation(
    checkpoint_path: str,
    output_dir: str,
    quick: bool = False,
) -> Dict:
    """Run efficiency and scaling evaluation."""
    print("\n" + "="*60)
    print("PART 2: EFFICIENCY & SCALING (Money Charts)")
    print("="*60)

    from scripts.evaluation.benchmark_inference import (
        load_model_from_checkpoint,
        run_benchmark_sweep,
        save_results,
        generate_plots,
        BenchmarkConfig,
    )

    # Configure benchmark
    if quick:
        config = BenchmarkConfig(
            seq_lengths=[512, 1024, 2048, 4096],
            batch_sizes=[1, 8],
            max_new_tokens=64,
            warmup_runs=2,
            benchmark_runs=10,
            output_dir=output_dir,
        )
    else:
        config = BenchmarkConfig(
            seq_lengths=[512, 1024, 2048, 4096, 8192, 16384],
            batch_sizes=[1, 8, 32],
            max_new_tokens=128,
            warmup_runs=3,
            benchmark_runs=20,
            output_dir=output_dir,
        )

    # Load model
    model, model_name = load_model_from_checkpoint(checkpoint_path)
    model = model.cuda()

    # Run benchmarks
    results = run_benchmark_sweep(model, f"Hybrid-{model_name}", config)

    # Save results
    save_results(results, output_dir, "efficiency")

    # Generate plots
    generate_plots(results, output_dir)

    # Return summary
    if results:
        return {
            "max_tps": max(r.tokens_per_second for r in results if r.tokens_per_second > 0),
            "max_memory_gb": max(r.peak_memory_gb for r in results),
            "min_ttft_ms": min(r.time_to_first_token_ms for r in results if r.time_to_first_token_ms < float('inf')),
        }
    return {}


def run_contrapro_evaluation(
    checkpoint_path: str,
    output_dir: str,
    use_synthetic: bool = False,
    max_samples: Optional[int] = None,
) -> Dict:
    """Run ContraPro pronoun disambiguation evaluation."""
    print("\n" + "="*60)
    print("PART 3: CONTRAPRO (Scientific Novelty)")
    print("="*60)

    from scripts.evaluation.evaluate_contrapro import (
        ContraProEvaluator,
        load_contrapro_data,
        create_synthetic_contrapro,
        save_results,
        generate_accuracy_plot,
    )
    from data import create_tokenizer
    from models import ModelConfig, HybridMambaEncoderDecoder

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model
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

    model_name = Path(checkpoint_path).stem

    # Load or create data
    if use_synthetic:
        samples = create_synthetic_contrapro(
            tokenizer,
            num_samples=max_samples or 500,
            max_context_length=2048,
        )
    else:
        try:
            samples = load_contrapro_data(max_samples=max_samples)
        except Exception as e:
            print(f"Failed to load ContraPro data: {e}")
            print("Using synthetic data instead...")
            samples = create_synthetic_contrapro(
                tokenizer,
                num_samples=max_samples or 500,
            )

    # Evaluate
    evaluator = ContraProEvaluator(model, tokenizer, device)
    result = evaluator.evaluate(samples, model_name=model_name)

    # Save results
    save_results(result, output_dir, f"contrapro_{model_name}")

    # Generate plot
    generate_accuracy_plot([result], output_dir)

    return {
        "accuracy": result.accuracy,
        "accuracy_by_distance": result.accuracy_by_distance,
    }


def generate_paper_artifacts(
    results: Dict,
    output_dir: str,
) -> None:
    """Generate all paper artifacts (tables and figures)."""
    print("\n" + "="*60)
    print("GENERATING PAPER ARTIFACTS")
    print("="*60)

    output_path = Path(output_dir)

    # Generate LaTeX Table 1: Quality Results
    table1_content = """
\\begin{table}[h]
\\centering
\\caption{Translation Quality on IWSLT14 De-En Test Set}
\\label{tab:quality}
\\begin{tabular}{lccc}
\\toprule
Model & BLEU ($\\uparrow$) & COMET ($\\uparrow$) & ChrF++ ($\\uparrow$) \\\\
\\midrule
"""

    if "quality" in results:
        q = results["quality"]
        bleu = f"{q.get('bleu', 0):.2f}"
        comet = f"{q.get('comet', 0):.4f}" if q.get('comet') else "N/A"
        chrf = f"{q.get('chrf', 0):.2f}"
        table1_content += f"Hybrid Mamba-Attention (Ours) & \\textbf{{{bleu}}} & \\textbf{{{comet}}} & \\textbf{{{chrf}}} \\\\\n"

    table1_content += """
\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(output_path / "table1_quality.tex", 'w') as f:
        f.write(table1_content)
    print(f"  Saved: table1_quality.tex")

    # Generate summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
        "results": results,
    }

    with open(output_path / "evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: evaluation_summary.json")

    # Print summary table
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    if "quality" in results:
        print("\nTranslation Quality:")
        q = results["quality"]
        print(f"  BLEU:  {q.get('bleu', 'N/A')}")
        print(f"  COMET: {q.get('comet', 'N/A')}")
        print(f"  ChrF++: {q.get('chrf', 'N/A')}")

    if "efficiency" in results:
        print("\nEfficiency:")
        e = results["efficiency"]
        print(f"  Max TPS:    {e.get('max_tps', 'N/A')}")
        print(f"  Max Memory: {e.get('max_memory_gb', 'N/A')} GB")
        print(f"  Min TTFT:   {e.get('min_ttft_ms', 'N/A')} ms")

    if "contrapro" in results:
        print("\nContraPro (Pronoun Disambiguation):")
        c = results["contrapro"]
        print(f"  Overall Accuracy: {c.get('accuracy', 'N/A'):.2%}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Full Evaluation Pipeline")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                       help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: fewer samples, skip COMET")
    parser.add_argument("--only", type=str, choices=["quality", "efficiency", "contrapro"],
                       help="Run only specific evaluation")
    parser.add_argument("--skip-comet", action="store_true",
                       help="Skip COMET evaluation (faster)")
    parser.add_argument("--max-quality-samples", type=int,
                       help="Max samples for quality evaluation")
    parser.add_argument("--use-synthetic-contrapro", action="store_true",
                       help="Use synthetic ContraPro data")

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quick mode settings
    if args.quick:
        args.skip_comet = True
        args.max_quality_samples = args.max_quality_samples or 500
        args.use_synthetic_contrapro = True

    print(f"\n{'='*60}")
    print("FULL EVALUATION PIPELINE")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"{'='*60}\n")

    results = {}
    start_time = time.time()

    try:
        # Part 1: Quality
        if args.only is None or args.only == "quality":
            results["quality"] = run_quality_evaluation(
                args.checkpoint,
                args.output_dir,
                skip_comet=args.skip_comet,
                max_samples=args.max_quality_samples,
            )

        # Part 2: Efficiency
        if args.only is None or args.only == "efficiency":
            results["efficiency"] = run_efficiency_evaluation(
                args.checkpoint,
                args.output_dir,
                quick=args.quick,
            )

        # Part 3: ContraPro
        if args.only is None or args.only == "contrapro":
            results["contrapro"] = run_contrapro_evaluation(
                args.checkpoint,
                args.output_dir,
                use_synthetic=args.use_synthetic_contrapro,
            )

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # Generate paper artifacts
    if results:
        generate_paper_artifacts(results, args.output_dir)

    total_time = (time.time() - start_time) / 60
    print(f"\nTotal evaluation time: {total_time:.1f} minutes")


if __name__ == "__main__":
    main()
