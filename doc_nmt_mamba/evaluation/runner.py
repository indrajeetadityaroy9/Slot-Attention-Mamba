"""
Evaluation Pipeline Orchestration for Document-Level NMT.

Provides unified evaluation runner that orchestrates:
- Part 1: Translation Quality (BLEU, COMET, ChrF++)
- Part 2: Efficiency & Scaling (Throughput, Memory, Latency)
- Part 3: Scientific Novelty (ContraPro pronoun disambiguation)

Generates publication artifacts:
- LaTeX tables for quality metrics
- Log-log throughput figures
- JSON summary reports

Usage:
    from evaluation import EvaluationRunner, RunnerConfig

    runner = EvaluationRunner(config)
    results = runner.run_full_evaluation(model, tokenizer)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import time
import warnings

import torch
import torch.nn as nn

from .metrics import (
    EvaluationResult,
    EvaluationSuite,
    BLEUScorer,
    CHRFScorer,
    COMETScorer,
    COMET_AVAILABLE,
)
from .analysis import (
    ContrastiveResult,
    ContrastivePronounEvaluator,
    ContraProDataset,
    EntityRecallAnalyzer,
    LengthSensitivityAnalyzer,
)


@dataclass
class RunnerConfig:
    """Configuration for evaluation runner."""

    # Output settings
    output_dir: str = "outputs/evaluation"

    # Quality evaluation
    skip_comet: bool = False
    comet_model: str = "Unbabel/wmt22-comet-da"
    max_quality_samples: Optional[int] = None

    # Efficiency evaluation
    efficiency_seq_lengths: List[int] = field(
        default_factory=lambda: [512, 1024, 2048, 4096, 8192]
    )
    efficiency_batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])
    warmup_runs: int = 3
    benchmark_runs: int = 20

    # ContraPro evaluation
    use_synthetic_contrapro: bool = False
    max_contrapro_samples: Optional[int] = None
    contrapro_data_path: str = "data/contrapro/contrapro.json"  # Path to ContraPro data
    contrapro_distances: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 10, 20]
    )

    # Quick mode
    quick: bool = False

    def __post_init__(self):
        """Apply quick mode settings."""
        if self.quick:
            self.skip_comet = True
            self.max_quality_samples = self.max_quality_samples or 500
            self.use_synthetic_contrapro = True
            self.max_contrapro_samples = 200
            self.efficiency_seq_lengths = [512, 1024, 2048]
            self.efficiency_batch_sizes = [1, 8]
            self.warmup_runs = 2
            self.benchmark_runs = 5


@dataclass
class QualityResult:
    """Results from translation quality evaluation."""

    bleu: float = 0.0
    comet: Optional[float] = None
    chrf: float = 0.0
    ter: float = 0.0
    num_samples: int = 0
    model_name: str = ""
    dataset: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bleu": self.bleu,
            "comet": self.comet,
            "chrf": self.chrf,
            "ter": self.ter,
            "num_samples": self.num_samples,
            "model_name": self.model_name,
            "dataset": self.dataset,
        }


@dataclass
class EfficiencyResult:
    """Results from efficiency evaluation."""

    seq_length: int = 0
    batch_size: int = 1
    tokens_per_second: float = 0.0
    peak_memory_gb: float = 0.0
    time_to_first_token_ms: float = 0.0
    inter_token_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq_length": self.seq_length,
            "batch_size": self.batch_size,
            "tokens_per_second": self.tokens_per_second,
            "peak_memory_gb": self.peak_memory_gb,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "inter_token_latency_ms": self.inter_token_latency_ms,
        }


@dataclass
class ContraProResult:
    """Results from ContraPro evaluation."""

    accuracy: float = 0.0
    accuracy_by_distance: Dict[int, float] = field(default_factory=dict)
    num_samples: int = 0
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "accuracy_by_distance": self.accuracy_by_distance,
            "num_samples": self.num_samples,
            "model_name": self.model_name,
        }


@dataclass
class FullEvaluationResult:
    """Aggregated results from full evaluation pipeline."""

    quality: Optional[QualityResult] = None
    efficiency: List[EfficiencyResult] = field(default_factory=list)
    contrapro: Optional[ContraProResult] = None
    timestamp: str = ""
    gpu_name: str = ""
    total_time_minutes: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "timestamp": self.timestamp,
            "gpu_name": self.gpu_name,
            "total_time_minutes": self.total_time_minutes,
        }
        if self.quality:
            result["quality"] = self.quality.to_dict()
        if self.efficiency:
            result["efficiency"] = [e.to_dict() for e in self.efficiency]
        if self.contrapro:
            result["contrapro"] = self.contrapro.to_dict()
        return result


def check_dependencies() -> Tuple[bool, List[str]]:
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

    return len(missing) == 0, missing


class EvaluationRunner:
    """
    Unified evaluation pipeline orchestrator.

    Coordinates quality, efficiency, and scientific evaluations
    for publication-ready results.
    """

    def __init__(
        self,
        config: Optional[RunnerConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize evaluation runner.

        Args:
            config: Runner configuration
            device: Device for evaluation (cuda/cpu)
        """
        self.config = config or RunnerConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directory
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize scorers
        self.eval_suite = EvaluationSuite(
            use_comet=not self.config.skip_comet,
            comet_model=self.config.comet_model,
        )

    def run_quality_evaluation(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        model_name: str = "hybrid",
        dataset: str = "test",
    ) -> QualityResult:
        """
        Run translation quality evaluation.

        Args:
            sources: Source sentences
            hypotheses: Model outputs
            references: Reference translations
            model_name: Name for results
            dataset: Dataset name

        Returns:
            QualityResult with BLEU, COMET, chrF
        """
        print("\n" + "=" * 60)
        print("PART 1: TRANSLATION QUALITY")
        print("=" * 60)

        # Apply sample limit
        if self.config.max_quality_samples:
            sources = sources[: self.config.max_quality_samples]
            hypotheses = hypotheses[: self.config.max_quality_samples]
            references = references[: self.config.max_quality_samples]

        print(f"Evaluating {len(hypotheses)} samples...")

        # Run evaluation suite
        result = self.eval_suite.evaluate(sources, hypotheses, references)

        quality_result = QualityResult(
            bleu=result.bleu,
            comet=result.comet if result.comet > 0 else None,
            chrf=result.chrf,
            ter=result.ter,
            num_samples=len(hypotheses),
            model_name=model_name,
            dataset=dataset,
        )

        print(f"\nResults:")
        print(f"  BLEU:  {quality_result.bleu:.2f}")
        print(f"  COMET: {quality_result.comet:.4f}" if quality_result.comet else "  COMET: Skipped")
        print(f"  chrF:  {quality_result.chrf:.2f}")
        print(f"  TER:   {quality_result.ter:.2f}")

        return quality_result

    def run_efficiency_evaluation(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_name: str = "hybrid",
    ) -> List[EfficiencyResult]:
        """
        Run efficiency and scaling evaluation.

        Args:
            model: Model to benchmark
            tokenizer: Tokenizer for input generation
            model_name: Name for results

        Returns:
            List of EfficiencyResult for each configuration
        """
        print("\n" + "=" * 60)
        print("PART 2: EFFICIENCY & SCALING")
        print("=" * 60)

        if not torch.cuda.is_available():
            print("CUDA not available. Skipping efficiency benchmarks.")
            return []

        results = []
        model.eval()

        for seq_len in self.config.efficiency_seq_lengths:
            for batch_size in self.config.efficiency_batch_sizes:
                print(f"\nBenchmarking seq_len={seq_len}, batch_size={batch_size}...")

                try:
                    result = self._benchmark_config(
                        model, tokenizer, seq_len, batch_size
                    )
                    results.append(result)
                    print(
                        f"  TPS: {result.tokens_per_second:.1f}, "
                        f"Memory: {result.peak_memory_gb:.2f}GB, "
                        f"TTFT: {result.time_to_first_token_ms:.1f}ms"
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  OOM - skipping")
                        torch.cuda.empty_cache()
                    else:
                        raise

        return results

    def _benchmark_config(
        self,
        model: nn.Module,
        tokenizer: Any,
        seq_length: int,
        batch_size: int,
    ) -> EfficiencyResult:
        """Benchmark a single configuration."""
        # Generate dummy input
        src_ids = torch.randint(
            100, 1000, (batch_size, seq_length), device=self.device
        )

        # Warmup
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                _ = model.encode(src_ids)
            torch.cuda.synchronize()

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Benchmark encoding
        times = []
        for _ in range(self.config.benchmark_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                encoder_out = model.encode(src_ids)

            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        tokens_per_second = (batch_size * seq_length) / avg_time
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

        # Measure TTFT (time to first token)
        cache = model.init_generation_cache(encoder_out, device=self.device)
        bos_ids = torch.full((batch_size, 1), 1, device=self.device)  # BOS token

        torch.cuda.synchronize()
        ttft_start = time.perf_counter()

        with torch.no_grad():
            _, _ = model.generate_step(bos_ids, cache)

        torch.cuda.synchronize()
        ttft_end = time.perf_counter()

        ttft_ms = (ttft_end - ttft_start) * 1000

        return EfficiencyResult(
            seq_length=seq_length,
            batch_size=batch_size,
            tokens_per_second=tokens_per_second,
            peak_memory_gb=peak_memory_gb,
            time_to_first_token_ms=ttft_ms,
            inter_token_latency_ms=ttft_ms,  # Approximation
        )

    def run_contrapro_evaluation(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_name: str = "hybrid",
    ) -> ContraProResult:
        """
        Run ContraPro pronoun disambiguation evaluation.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            model_name: Name for results

        Returns:
            ContraProResult with accuracy by distance
        """
        print("\n" + "=" * 60)
        print("PART 3: CONTRAPRO (Pronoun Disambiguation)")
        print("=" * 60)

        # Load or create data
        if self.config.use_synthetic_contrapro:
            print("Using synthetic ContraPro data...")
            samples = self._create_synthetic_contrapro(
                tokenizer,
                num_samples=self.config.max_contrapro_samples or 500,
            )
        else:
            try:
                # FIX: Pass data_path from config
                dataset = ContraProDataset(data_path=self.config.contrapro_data_path)
                # FIX: ContraProDataset has no .load() method - use list slicing
                all_samples = list(dataset)
                max_samples = self.config.max_contrapro_samples
                samples = all_samples[:max_samples] if max_samples else all_samples
            except Exception as e:
                print(f"Failed to load ContraPro: {e}")
                print("Using synthetic data instead...")
                samples = self._create_synthetic_contrapro(
                    tokenizer,
                    num_samples=self.config.max_contrapro_samples or 500,
                )

        print(f"Evaluating {len(samples)} contrastive examples...")

        # Evaluate
        evaluator = ContrastivePronounEvaluator(model, tokenizer, self.device)
        result = evaluator.evaluate(samples)

        contrapro_result = ContraProResult(
            accuracy=result.accuracy,
            accuracy_by_distance=result.accuracy_by_distance,
            num_samples=len(samples),
            model_name=model_name,
        )

        print(f"\nResults:")
        print(f"  Overall Accuracy: {contrapro_result.accuracy:.2%}")
        if contrapro_result.accuracy_by_distance:
            print("  Accuracy by distance:")
            for dist, acc in sorted(contrapro_result.accuracy_by_distance.items()):
                print(f"    Distance {dist}: {acc:.2%}")

        return contrapro_result

    def _create_synthetic_contrapro(
        self,
        tokenizer: Any,
        num_samples: int = 500,
    ) -> List[Dict]:
        """Create synthetic ContraPro-style examples."""
        # Simple synthetic examples for testing
        samples = []
        pronouns = ["er", "sie", "es"]
        translations = ["he", "she", "it"]

        for i in range(num_samples):
            pronoun_idx = i % 3
            distance = (i % 20) + 1

            context = " ".join([f"Sentence {j}." for j in range(distance)])
            source = f"{context} Der Mann sagt, {pronouns[pronoun_idx]} ist hier."

            correct = translations[pronoun_idx]
            distractors = [t for t in translations if t != correct]

            samples.append({
                "source": source,
                "correct": f"The man says {correct} is here.",
                "distractors": [f"The man says {d} is here." for d in distractors],
                "distance": distance,
                "pronoun": pronouns[pronoun_idx],
            })

        return samples

    def run_full_evaluation(
        self,
        model: nn.Module,
        tokenizer: Any,
        sources: Optional[List[str]] = None,
        hypotheses: Optional[List[str]] = None,
        references: Optional[List[str]] = None,
        model_name: str = "hybrid",
        run_quality: bool = True,
        run_efficiency: bool = True,
        run_contrapro: bool = True,
    ) -> FullEvaluationResult:
        """
        Run full evaluation pipeline.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            sources: Source sentences (for quality eval)
            hypotheses: Translations (for quality eval)
            references: References (for quality eval)
            model_name: Name for results
            run_quality: Whether to run quality evaluation
            run_efficiency: Whether to run efficiency evaluation
            run_contrapro: Whether to run ContraPro evaluation

        Returns:
            FullEvaluationResult with all results
        """
        start_time = time.time()

        print("\n" + "=" * 60)
        print("FULL EVALUATION PIPELINE")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Mode: {'Quick' if self.config.quick else 'Full'}")
        print("=" * 60)

        result = FullEvaluationResult(
            timestamp=datetime.now().isoformat(),
            gpu_name=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        )

        try:
            # Part 1: Quality
            if run_quality and sources and hypotheses and references:
                result.quality = self.run_quality_evaluation(
                    sources, hypotheses, references, model_name
                )

            # Part 2: Efficiency
            if run_efficiency:
                result.efficiency = self.run_efficiency_evaluation(
                    model, tokenizer, model_name
                )

            # Part 3: ContraPro
            if run_contrapro:
                result.contrapro = self.run_contrapro_evaluation(
                    model, tokenizer, model_name
                )

        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user")
        except Exception as e:
            print(f"\nEvaluation failed: {e}")
            import traceback
            traceback.print_exc()

        result.total_time_minutes = (time.time() - start_time) / 60

        # Generate artifacts
        self._generate_artifacts(result)

        return result

    def _generate_artifacts(self, result: FullEvaluationResult) -> None:
        """Generate paper artifacts (tables, figures, JSON)."""
        print("\n" + "=" * 60)
        print("GENERATING PAPER ARTIFACTS")
        print("=" * 60)

        # Save JSON summary
        with open(self.output_path / "evaluation_summary.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"  Saved: evaluation_summary.json")

        # Generate LaTeX table if quality results available
        if result.quality:
            self._generate_quality_table(result.quality)

        # Print summary
        self._print_summary(result)

    def _generate_quality_table(self, quality: QualityResult) -> None:
        """Generate LaTeX table for quality results."""
        table_content = r"""
\begin{table}[h]
\centering
\caption{Translation Quality on """ + quality.dataset + r"""}
\label{tab:quality}
\begin{tabular}{lccc}
\toprule
Model & BLEU ($\uparrow$) & COMET ($\uparrow$) & ChrF++ ($\uparrow$) \\
\midrule
"""
        bleu = f"{quality.bleu:.2f}"
        comet = f"{quality.comet:.4f}" if quality.comet else "N/A"
        chrf = f"{quality.chrf:.2f}"

        table_content += f"Hybrid Mamba-Attention (Ours) & \\textbf{{{bleu}}} & \\textbf{{{comet}}} & \\textbf{{{chrf}}} \\\\\n"

        table_content += r"""
\bottomrule
\end{tabular}
\end{table}
"""

        with open(self.output_path / "table_quality.tex", "w") as f:
            f.write(table_content)
        print(f"  Saved: table_quality.tex")

    def _print_summary(self, result: FullEvaluationResult) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        if result.quality:
            print("\nTranslation Quality:")
            print(f"  BLEU:   {result.quality.bleu:.2f}")
            if result.quality.comet:
                print(f"  COMET:  {result.quality.comet:.4f}")
            print(f"  chrF:   {result.quality.chrf:.2f}")

        if result.efficiency:
            print("\nEfficiency (best):")
            max_tps = max(e.tokens_per_second for e in result.efficiency)
            max_mem = max(e.peak_memory_gb for e in result.efficiency)
            min_ttft = min(e.time_to_first_token_ms for e in result.efficiency)
            print(f"  Max TPS:    {max_tps:.1f}")
            print(f"  Max Memory: {max_mem:.2f} GB")
            print(f"  Min TTFT:   {min_ttft:.1f} ms")

        if result.contrapro:
            print("\nContraPro (Pronoun Disambiguation):")
            print(f"  Overall Accuracy: {result.contrapro.accuracy:.2%}")

        print(f"\nTotal time: {result.total_time_minutes:.1f} minutes")
        print("=" * 60)
