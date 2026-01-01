#!/usr/bin/env python3
"""
Translation Quality Evaluation for NMT Models.

Generates "Table 1" metrics for NeurIPS/ICML/ACL:
- BLEU (SacreBLEU): Legacy standard, n-gram overlap
- COMET (wmt22-comet-da): Neural metric, correlates with human judgment
- ChrF++: Character-level F-score, critical for morphological richness

Includes Paired Bootstrap Resampling for significance testing (p < 0.05).

Usage:
    python scripts/evaluation/evaluate_quality.py --checkpoint outputs/best_model.pt
    python scripts/evaluation/evaluate_quality.py --checkpoint outputs/best_model.pt --dataset iwslt14
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Evaluation libraries
try:
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    warnings.warn("sacrebleu not installed. Run: pip install sacrebleu")

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    warnings.warn("comet-ml not installed. Run: pip install unbabel-comet")

from models import ModelConfig, HybridMambaEncoderDecoder
from data import create_tokenizer, create_dataset


@dataclass
class QualityResult:
    """Translation quality evaluation result."""
    model_name: str
    dataset: str
    num_samples: int

    # Core metrics
    bleu: float
    bleu_signature: str
    comet: Optional[float]
    chrf: float
    chrf_plus_plus: float

    # Detailed BLEU scores
    bleu_1: float
    bleu_2: float
    bleu_3: float
    bleu_4: float
    brevity_penalty: float

    # Bootstrap confidence intervals
    bleu_ci_low: Optional[float] = None
    bleu_ci_high: Optional[float] = None
    comet_ci_low: Optional[float] = None
    comet_ci_high: Optional[float] = None


@dataclass
class SignificanceResult:
    """Result of significance testing between two systems."""
    system_a: str
    system_b: str
    metric: str
    score_a: float
    score_b: float
    p_value: float
    is_significant: bool  # p < 0.05
    winner: Optional[str]  # None if not significant


class TranslationEvaluator:
    """Comprehensive translation quality evaluator."""

    def __init__(
        self,
        comet_model: str = "Unbabel/wmt22-comet-da",
        device: str = "cuda",
    ):
        self.device = device

        # Initialize BLEU
        if SACREBLEU_AVAILABLE:
            self.bleu_scorer = BLEU(effective_order=True)
            self.chrf_scorer = CHRF(word_order=2)  # ChrF++
        else:
            self.bleu_scorer = None
            self.chrf_scorer = None

        # Initialize COMET
        self.comet_model = None
        if COMET_AVAILABLE:
            try:
                model_path = download_model(comet_model)
                self.comet_model = load_from_checkpoint(model_path)
                self.comet_model.to(device)
                self.comet_model.eval()
            except Exception as e:
                warnings.warn(f"Failed to load COMET model: {e}")

    def compute_bleu(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> Dict:
        """Compute SacreBLEU score."""
        if not SACREBLEU_AVAILABLE:
            return {"bleu": 0.0, "signature": "N/A"}

        # SacreBLEU expects list of references for each hypothesis
        refs = [[ref] for ref in references]

        result = self.bleu_scorer.corpus_score(hypotheses, [[r] for r in references])

        return {
            "bleu": result.score,
            "bleu_1": result.precisions[0],
            "bleu_2": result.precisions[1],
            "bleu_3": result.precisions[2],
            "bleu_4": result.precisions[3],
            "brevity_penalty": result.bp,
            "signature": result.format(),
        }

    def compute_chrf(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> Dict:
        """Compute ChrF++ score."""
        if not SACREBLEU_AVAILABLE:
            return {"chrf": 0.0, "chrf_plus_plus": 0.0}

        # Standard ChrF
        chrf_scorer = CHRF(word_order=0)
        chrf_result = chrf_scorer.corpus_score(hypotheses, [[r] for r in references])

        # ChrF++ (with word order)
        chrfpp_result = self.chrf_scorer.corpus_score(hypotheses, [[r] for r in references])

        return {
            "chrf": chrf_result.score,
            "chrf_plus_plus": chrfpp_result.score,
        }

    def compute_comet(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
    ) -> Dict:
        """Compute COMET score."""
        if self.comet_model is None:
            return {"comet": None, "comet_scores": []}

        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]

        with torch.no_grad():
            output = self.comet_model.predict(data, batch_size=32, gpus=1)

        return {
            "comet": output.system_score,
            "comet_scores": output.scores,
        }

    def paired_bootstrap_resampling(
        self,
        scores_a: List[float],
        scores_b: List[float],
        num_samples: int = 1000,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """
        Paired bootstrap resampling for significance testing.

        Returns:
            Tuple of (p_value, mean_diff, std_diff)
        """
        np.random.seed(seed)
        n = len(scores_a)
        assert len(scores_b) == n

        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)

        # Observed difference
        observed_diff = np.mean(scores_a) - np.mean(scores_b)

        # Bootstrap
        wins_a = 0
        wins_b = 0

        for _ in range(num_samples):
            indices = np.random.randint(0, n, size=n)
            sample_a = scores_a[indices]
            sample_b = scores_b[indices]

            if np.mean(sample_a) > np.mean(sample_b):
                wins_a += 1
            else:
                wins_b += 1

        # Two-tailed p-value
        p_value = 2 * min(wins_a, wins_b) / num_samples

        return p_value, observed_diff, np.std(scores_a - scores_b)

    def bootstrap_confidence_interval(
        self,
        scores: List[float],
        num_samples: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for a metric.

        Returns:
            Tuple of (ci_low, ci_high)
        """
        np.random.seed(seed)
        scores = np.array(scores)
        n = len(scores)

        bootstrap_means = []
        for _ in range(num_samples):
            indices = np.random.randint(0, n, size=n)
            bootstrap_means.append(np.mean(scores[indices]))

        alpha = 1 - confidence
        ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
        ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return ci_low, ci_high

    def evaluate(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        model_name: str = "model",
        dataset: str = "test",
    ) -> QualityResult:
        """Run full evaluation suite."""
        print(f"\nEvaluating {len(hypotheses)} translations...")

        # BLEU
        print("  Computing BLEU...")
        bleu_results = self.compute_bleu(hypotheses, references)

        # ChrF++
        print("  Computing ChrF++...")
        chrf_results = self.compute_chrf(hypotheses, references)

        # COMET
        print("  Computing COMET...")
        comet_results = self.compute_comet(sources, hypotheses, references)

        # Bootstrap CI for BLEU (sentence-level BLEU approximation)
        print("  Computing confidence intervals...")
        sentence_bleus = []
        for hyp, ref in zip(hypotheses, references):
            try:
                sent_bleu = sacrebleu.sentence_bleu(hyp, [ref]).score
                sentence_bleus.append(sent_bleu)
            except:
                sentence_bleus.append(0.0)

        bleu_ci_low, bleu_ci_high = self.bootstrap_confidence_interval(sentence_bleus)

        # Bootstrap CI for COMET
        comet_ci_low, comet_ci_high = None, None
        if comet_results["comet"] is not None:
            comet_ci_low, comet_ci_high = self.bootstrap_confidence_interval(
                comet_results["comet_scores"]
            )

        return QualityResult(
            model_name=model_name,
            dataset=dataset,
            num_samples=len(hypotheses),
            bleu=bleu_results["bleu"],
            bleu_signature=bleu_results["signature"],
            bleu_1=bleu_results["bleu_1"],
            bleu_2=bleu_results["bleu_2"],
            bleu_3=bleu_results["bleu_3"],
            bleu_4=bleu_results["bleu_4"],
            brevity_penalty=bleu_results["brevity_penalty"],
            comet=comet_results["comet"],
            chrf=chrf_results["chrf"],
            chrf_plus_plus=chrf_results["chrf_plus_plus"],
            bleu_ci_low=bleu_ci_low,
            bleu_ci_high=bleu_ci_high,
            comet_ci_low=comet_ci_low,
            comet_ci_high=comet_ci_high,
        )

    def compare_systems(
        self,
        sources: List[str],
        hypotheses_a: List[str],
        hypotheses_b: List[str],
        references: List[str],
        system_a_name: str = "System A",
        system_b_name: str = "System B",
    ) -> List[SignificanceResult]:
        """Compare two systems with significance testing."""
        results = []

        # Sentence-level BLEU
        bleu_a = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hypotheses_a, references)]
        bleu_b = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hypotheses_b, references)]

        p_value, diff, _ = self.paired_bootstrap_resampling(bleu_a, bleu_b)
        results.append(SignificanceResult(
            system_a=system_a_name,
            system_b=system_b_name,
            metric="BLEU",
            score_a=np.mean(bleu_a),
            score_b=np.mean(bleu_b),
            p_value=p_value,
            is_significant=p_value < 0.05,
            winner=system_a_name if diff > 0 and p_value < 0.05 else (
                system_b_name if diff < 0 and p_value < 0.05 else None
            ),
        ))

        # COMET (if available)
        if self.comet_model is not None:
            comet_a = self.compute_comet(sources, hypotheses_a, references)["comet_scores"]
            comet_b = self.compute_comet(sources, hypotheses_b, references)["comet_scores"]

            p_value, diff, _ = self.paired_bootstrap_resampling(comet_a, comet_b)
            results.append(SignificanceResult(
                system_a=system_a_name,
                system_b=system_b_name,
                metric="COMET",
                score_a=np.mean(comet_a),
                score_b=np.mean(comet_b),
                p_value=p_value,
                is_significant=p_value < 0.05,
                winner=system_a_name if diff > 0 and p_value < 0.05 else (
                    system_b_name if diff < 0 and p_value < 0.05 else None
                ),
            ))

        return results


def generate_translations(
    model: nn.Module,
    tokenizer,
    sources: List[str],
    max_length: int = 256,
    batch_size: int = 16,
    device: str = "cuda",
) -> List[str]:
    """Generate translations using the model."""
    model.eval()
    translations = []

    for i in tqdm(range(0, len(sources), batch_size), desc="Translating"):
        batch_sources = sources[i:i + batch_size]

        # Tokenize
        encoded = tokenizer(
            batch_sources,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)

        # Generate
        with torch.no_grad():
            if hasattr(model, 'generate'):
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    num_beams=4,
                    early_stopping=True,
                )
            else:
                # Manual greedy generation
                output_ids = greedy_generate(model, input_ids, max_length, tokenizer)

        # Decode
        for ids in output_ids:
            translation = tokenizer.decode(ids, skip_special_tokens=True)
            translations.append(translation)

    return translations


def greedy_generate(
    model: nn.Module,
    src_ids: torch.Tensor,
    max_length: int,
    tokenizer,
) -> torch.Tensor:
    """Simple greedy generation for models without generate() method."""
    batch_size = src_ids.shape[0]
    device = src_ids.device

    # Start with BOS token
    tgt_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * tokenizer.bos_token_id

    eos_id = tokenizer.eos_token_id
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_length):
        logits = model(src_ids, tgt_ids)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Check for EOS
        finished |= (next_token.squeeze(-1) == eos_id)
        if finished.all():
            break

        tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

    return tgt_ids


def load_test_data(
    dataset_name: str = "iwslt14",
    split: str = "test",
    src_lang: str = "de",
    tgt_lang: str = "en",
) -> Tuple[List[str], List[str]]:
    """Load test data for evaluation."""
    from datasets import load_dataset

    print(f"Loading {dataset_name} {split} set...")

    if dataset_name == "iwslt14":
        # IWSLT uses WMT14 as fallback
        try:
            ds = load_dataset("iwslt2017", f"iwslt2017-{src_lang}-{tgt_lang}", split=split)
        except:
            ds = load_dataset("wmt14", f"{src_lang}-{tgt_lang}", split=split)
            ds = ds.select(range(min(3000, len(ds))))  # Limit for testing
    else:
        ds = load_dataset(dataset_name, split=split)

    sources = []
    references = []

    for item in ds:
        if "translation" in item:
            sources.append(item["translation"][src_lang])
            references.append(item["translation"][tgt_lang])
        else:
            sources.append(item.get("source", item.get("src", "")))
            references.append(item.get("target", item.get("tgt", "")))

    print(f"Loaded {len(sources)} samples")
    return sources, references


def save_results(
    result: QualityResult,
    output_dir: str,
    experiment_name: str = "quality",
) -> None:
    """Save evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_path / f"{experiment_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    print(f"Saved: {json_path}")

    # Print formatted table
    print("\n" + "="*60)
    print("TRANSLATION QUALITY RESULTS")
    print("="*60)
    print(f"Model: {result.model_name}")
    print(f"Dataset: {result.dataset} ({result.num_samples} samples)")
    print("-"*60)
    print(f"BLEU:        {result.bleu:.2f} [{result.bleu_ci_low:.2f}, {result.bleu_ci_high:.2f}]")
    print(f"  BLEU-1:    {result.bleu_1:.2f}")
    print(f"  BLEU-2:    {result.bleu_2:.2f}")
    print(f"  BLEU-3:    {result.bleu_3:.2f}")
    print(f"  BLEU-4:    {result.bleu_4:.2f}")
    print(f"  BP:        {result.brevity_penalty:.4f}")
    print(f"ChrF:        {result.chrf:.2f}")
    print(f"ChrF++:      {result.chrf_plus_plus:.2f}")
    if result.comet is not None:
        print(f"COMET:       {result.comet:.4f} [{result.comet_ci_low:.4f}, {result.comet_ci_high:.4f}]")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Translation Quality")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                       help="Output directory for results")
    parser.add_argument("--dataset", type=str, default="iwslt14",
                       choices=["iwslt14", "wmt14", "opus_books"],
                       help="Test dataset to use")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to evaluate")
    parser.add_argument("--src-lang", type=str, default="de",
                       help="Source language code")
    parser.add_argument("--tgt-lang", type=str, default="en",
                       help="Target language code")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for generation")
    parser.add_argument("--max-length", type=int, default=256,
                       help="Maximum generation length")
    parser.add_argument("--tokenizer-path", type=str,
                       default="data/tokenizer/tokenizer.json",
                       help="Path to tokenizer")
    parser.add_argument("--comet-model", type=str,
                       default="Unbabel/wmt22-comet-da",
                       help="COMET model to use")
    parser.add_argument("--skip-comet", action="store_true",
                       help="Skip COMET evaluation (faster)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = create_tokenizer(
        tokenizer_type="custom",
        tokenizer_path=args.tokenizer_path,
    )

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            config = ModelConfig(**config)
    else:
        config = ModelConfig()

    model = HybridMambaEncoderDecoder(config=config, device=device, dtype=torch.bfloat16)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    model_name = Path(args.checkpoint).stem

    # Load test data
    sources, references = load_test_data(
        dataset_name=args.dataset,
        split=args.split,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )

    # Generate translations
    print("\nGenerating translations...")
    hypotheses = generate_translations(
        model, tokenizer, sources,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )

    # Evaluate
    evaluator = TranslationEvaluator(
        comet_model=args.comet_model if not args.skip_comet else None,
        device=device,
    )

    result = evaluator.evaluate(
        sources=sources,
        hypotheses=hypotheses,
        references=references,
        model_name=model_name,
        dataset=args.dataset,
    )

    # Save results
    save_results(result, args.output_dir, f"quality_{model_name}")

    # Save translations for analysis
    translations_path = Path(args.output_dir) / f"translations_{model_name}.json"
    with open(translations_path, 'w', encoding='utf-8') as f:
        json.dump({
            "sources": sources[:100],  # Save subset for inspection
            "hypotheses": hypotheses[:100],
            "references": references[:100],
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved sample translations: {translations_path}")


if __name__ == "__main__":
    main()
