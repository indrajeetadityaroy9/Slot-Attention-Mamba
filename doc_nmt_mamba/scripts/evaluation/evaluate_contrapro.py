#!/usr/bin/env python3
"""
ContraPro Evaluation: Contrastive Pronoun Disambiguation for Document-Level NMT.

This script evaluates the model's ability to use document context for pronoun resolution.
Key metric for proving document-level understanding (NeurIPS/ACL/EMNLP Scientific Novelty).

The ContraPro test:
- Model scores correct translation vs. incorrect pronoun translation
- Measures accuracy across different antecedent distances (how far back is the context?)
- Hypothesis: Transformer accuracy drops at distance > 512 tokens; Mamba maintains accuracy

Datasets:
- ContraPro (En-De): Müller et al., 2018
- Bawden et al. (En-Fr): For French evaluation

Usage:
    python scripts/evaluation/evaluate_contrapro.py --checkpoint outputs/best_model.pt
    python scripts/evaluation/evaluate_contrapro.py --checkpoint outputs/best_model.pt --max-distance 2048
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from models import ModelConfig, HybridMambaEncoderDecoder
from data import create_tokenizer


@dataclass
class ContraProSample:
    """Single ContraPro evaluation sample."""
    source_context: str
    source_sentence: str
    correct_translation: str
    contrastive_translation: str  # Wrong pronoun
    antecedent_distance: int  # Distance in tokens
    pronoun_type: str  # "he/she", "it/they", etc.
    source_pronoun: str
    correct_pronoun: str
    contrastive_pronoun: str


@dataclass
class ContraProResult:
    """ContraPro evaluation results."""
    model_name: str
    total_samples: int
    accuracy: float

    # Accuracy by distance buckets
    accuracy_by_distance: Dict[str, float]

    # Accuracy by pronoun type
    accuracy_by_pronoun: Dict[str, float]

    # Detailed per-sample results (optional)
    sample_results: Optional[List[Dict]] = None


def download_contrapro_data(output_dir: str = "data/contrapro") -> Path:
    """
    Download and prepare ContraPro dataset.

    ContraPro: https://github.com/ZurichNLP/ContraPro

    IMPORTANT: Standard ContraPro is En->De (English source, German target).
    For De->En models, use create_reversed_contrapro() or Bawden dataset.
    """
    import urllib.request

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    contrapro_file = output_path / "contrapro.json"

    if contrapro_file.exists():
        print(f"ContraPro data already exists at {contrapro_file}")
        return contrapro_file

    # Download from GitHub
    url = "https://raw.githubusercontent.com/ZurichNLP/ContraPro/master/contrapro.json"
    print(f"Downloading ContraPro dataset from {url}...")

    try:
        urllib.request.urlretrieve(url, contrapro_file)
        print(f"Saved to {contrapro_file}")
    except Exception as e:
        print(f"Failed to download: {e}")
        print("Please manually download from: https://github.com/ZurichNLP/ContraPro")
        raise

    return contrapro_file


def download_bawden_data(output_dir: str = "data/bawden") -> Path:
    """
    Download Bawden et al. (2018) discourse test set for Fr->En.

    Paper: "Evaluating Discourse Phenomena in Neural Machine Translation"
    GitHub: https://github.com/rbawden/discourse-mt-test-sets
    """
    import urllib.request

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bawden_file = output_path / "anaphora.json"

    if bawden_file.exists():
        print(f"Bawden data already exists at {bawden_file}")
        return bawden_file

    # Download anaphora test set
    url = "https://raw.githubusercontent.com/rbawden/discourse-mt-test-sets/master/test-sets/anaphora.json"
    print(f"Downloading Bawden anaphora dataset from {url}...")

    try:
        urllib.request.urlretrieve(url, bawden_file)
        print(f"Saved to {bawden_file}")
    except Exception as e:
        print(f"Failed to download Bawden data: {e}")
        raise

    return bawden_file


def load_bawden_data(
    data_path: str = "data/bawden/anaphora.json",
    max_samples: Optional[int] = None,
) -> List[ContraProSample]:
    """
    Load Bawden et al. (2018) anaphora test set for Fr->En evaluation.

    This is the standard discourse test set for French-English.
    """
    data_file = Path(data_path)

    if not data_file.exists():
        data_file = download_bawden_data(str(data_file.parent))

    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        try:
            # Bawden format: src_segment, ref (correct), contrast (wrong)
            context = item.get('src_prefix', '')
            source = item.get('src_segment', '')
            correct = item.get('ref', '')
            contrastive = item.get('contrast', '')

            if not all([source, correct, contrastive]):
                continue

            # Calculate distance
            distance = len(context.split()) if context else 0

            samples.append(ContraProSample(
                source_context=context,
                source_sentence=source,
                correct_translation=correct,
                contrastive_translation=contrastive,
                antecedent_distance=distance,
                pronoun_type=item.get('type', 'anaphora'),
                source_pronoun=item.get('src_pronoun', ''),
                correct_pronoun=item.get('ref_pronoun', ''),
                contrastive_pronoun=item.get('contrast_pronoun', ''),
            ))
        except Exception:
            continue

    if max_samples:
        samples = samples[:max_samples]

    print(f"Loaded {len(samples)} Bawden anaphora samples")
    return samples


def load_contrapro_data(
    data_path: str = "data/contrapro/contrapro.json",
    max_samples: Optional[int] = None,
) -> List[ContraProSample]:
    """Load ContraPro dataset."""
    data_file = Path(data_path)

    if not data_file.exists():
        # Try to download
        data_file = download_contrapro_data(str(data_file.parent))

    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        try:
            # ContraPro format varies, handle different structures
            if isinstance(item, dict):
                # Extract context (previous sentences)
                context = item.get('src_context', item.get('context', ''))
                if isinstance(context, list):
                    context = ' '.join(context)

                source = item.get('src', item.get('source', ''))
                correct = item.get('ref', item.get('correct', ''))
                contrastive = item.get('contrast', item.get('contrastive', ''))

                # Get pronoun info
                pronoun_info = item.get('pronoun', {})
                if isinstance(pronoun_info, dict):
                    src_pronoun = pronoun_info.get('src', '')
                    correct_pronoun = pronoun_info.get('ref', '')
                    contrastive_pronoun = pronoun_info.get('contrast', '')
                else:
                    src_pronoun = correct_pronoun = contrastive_pronoun = ''

                # Calculate distance (approximate in characters, convert to tokens later)
                distance = len(context.split()) if context else 0

                samples.append(ContraProSample(
                    source_context=context,
                    source_sentence=source,
                    correct_translation=correct,
                    contrastive_translation=contrastive,
                    antecedent_distance=distance,
                    pronoun_type=item.get('ante_type', 'unknown'),
                    source_pronoun=src_pronoun,
                    correct_pronoun=correct_pronoun,
                    contrastive_pronoun=contrastive_pronoun,
                ))
        except Exception as e:
            continue

    if max_samples:
        samples = samples[:max_samples]

    print(f"Loaded {len(samples)} ContraPro samples")
    return samples


def create_synthetic_contrapro(
    tokenizer,
    num_samples: int = 500,
    max_context_length: int = 2048,
) -> List[ContraProSample]:
    """
    Create synthetic ContraPro-style test data.

    Useful when ContraPro is unavailable or for testing specific distance ranges.
    """
    import random

    # Templates for pronoun disambiguation
    templates = [
        {
            "context": "The {noun} was on the table. ",
            "source": "It was very {adjective}.",
            "correct": "{pronoun_correct} war sehr {adj_de}.",
            "contrastive": "{pronoun_wrong} war sehr {adj_de}.",
            "pronouns": ("Es", "Er", "Sie"),
            "nouns": [("book", "Buch", "Es"), ("lamp", "Lampe", "Sie"), ("phone", "Telefon", "Es")],
        },
        {
            "context": "I saw {name} yesterday. ",
            "source": "{pronoun} was happy.",
            "correct": "{pronoun_correct} war glücklich.",
            "contrastive": "{pronoun_wrong} war glücklich.",
            "pronouns": ("He", "She"),
            "names": [("John", "Er", "Sie"), ("Mary", "Sie", "Er"), ("Alex", "Er", "Sie")],
        },
    ]

    samples = []
    distances = [10, 50, 100, 200, 500, 1000, 1500, 2000]

    for _ in range(num_samples):
        template = random.choice(templates)
        distance = random.choice(distances)

        # Pad context to desired distance
        base_context = random.choice(template.get("nouns", template.get("names", [("item", "X", "Y")])))
        filler = "This is filler text. " * (distance // 5)

        if "nouns" in template:
            noun, _, correct_pronoun = base_context
            wrong_pronouns = [p for p in template["pronouns"] if p != correct_pronoun]
            wrong_pronoun = random.choice(wrong_pronouns)

            context = template["context"].format(noun=noun) + filler
            source = template["source"].format(adjective="nice")
            correct = template["correct"].format(pronoun_correct=correct_pronoun, adj_de="schön")
            contrastive = template["contrastive"].format(pronoun_wrong=wrong_pronoun, adj_de="schön")
        else:
            name, correct_pronoun, wrong_pronoun = base_context
            context = template["context"].format(name=name) + filler
            source = template["source"].format(pronoun="He" if correct_pronoun == "Er" else "She")
            correct = template["correct"].format(pronoun_correct=correct_pronoun)
            contrastive = template["contrastive"].format(pronoun_wrong=wrong_pronoun)

        samples.append(ContraProSample(
            source_context=context,
            source_sentence=source,
            correct_translation=correct,
            contrastive_translation=contrastive,
            antecedent_distance=len(context.split()),
            pronoun_type="synthetic",
            source_pronoun="it/he/she",
            correct_pronoun=correct_pronoun,
            contrastive_pronoun=wrong_pronoun,
        ))

    return samples


class ContraProEvaluator:
    """Evaluator for ContraPro pronoun disambiguation task."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def score_translation(
        self,
        source: str,
        target: str,
    ) -> float:
        """
        Compute log-probability of target given source.

        This is the core scoring function for contrastive evaluation.
        """
        # Tokenize
        src_encoded = self.tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        tgt_encoded = self.tokenizer(
            target,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        src_ids = src_encoded["input_ids"].to(self.device)
        tgt_ids = tgt_encoded["input_ids"].to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(src_ids, tgt_ids)

        # Compute log probability
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tgt_ids[:, 1:].contiguous()

        # Log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Sum log probabilities (or mean for length normalization)
        total_log_prob = token_log_probs.sum().item()
        mean_log_prob = token_log_probs.mean().item()

        return mean_log_prob  # Use mean for length normalization

    def evaluate_sample(
        self,
        sample: ContraProSample,
    ) -> Tuple[bool, float, float]:
        """
        Evaluate single ContraPro sample.

        Returns:
            Tuple of (is_correct, correct_score, contrastive_score)
        """
        # Combine context with source sentence
        full_source = sample.source_context + " " + sample.source_sentence

        # Score correct translation
        correct_score = self.score_translation(full_source, sample.correct_translation)

        # Score contrastive (wrong pronoun) translation
        contrastive_score = self.score_translation(full_source, sample.contrastive_translation)

        # Model is correct if it assigns higher score to correct translation
        is_correct = correct_score > contrastive_score

        return is_correct, correct_score, contrastive_score

    def evaluate(
        self,
        samples: List[ContraProSample],
        model_name: str = "model",
    ) -> ContraProResult:
        """Run full ContraPro evaluation."""
        print(f"\nEvaluating {len(samples)} ContraPro samples...")

        results = []
        correct_count = 0

        # Distance buckets
        distance_buckets = {
            "0-50": [], "51-100": [], "101-200": [],
            "201-500": [], "501-1000": [], "1000+": []
        }

        # Pronoun type results
        pronoun_results = {}

        for sample in tqdm(samples, desc="Evaluating"):
            is_correct, correct_score, contrastive_score = self.evaluate_sample(sample)

            results.append({
                "correct": is_correct,
                "correct_score": correct_score,
                "contrastive_score": contrastive_score,
                "distance": sample.antecedent_distance,
                "pronoun_type": sample.pronoun_type,
            })

            if is_correct:
                correct_count += 1

            # Bucket by distance
            dist = sample.antecedent_distance
            if dist <= 50:
                distance_buckets["0-50"].append(is_correct)
            elif dist <= 100:
                distance_buckets["51-100"].append(is_correct)
            elif dist <= 200:
                distance_buckets["101-200"].append(is_correct)
            elif dist <= 500:
                distance_buckets["201-500"].append(is_correct)
            elif dist <= 1000:
                distance_buckets["501-1000"].append(is_correct)
            else:
                distance_buckets["1000+"].append(is_correct)

            # Bucket by pronoun type
            ptype = sample.pronoun_type
            if ptype not in pronoun_results:
                pronoun_results[ptype] = []
            pronoun_results[ptype].append(is_correct)

        # Compute accuracies
        overall_accuracy = correct_count / len(samples) if samples else 0

        accuracy_by_distance = {}
        for bucket, vals in distance_buckets.items():
            if vals:
                accuracy_by_distance[bucket] = sum(vals) / len(vals)
            else:
                accuracy_by_distance[bucket] = None

        accuracy_by_pronoun = {}
        for ptype, vals in pronoun_results.items():
            if vals:
                accuracy_by_pronoun[ptype] = sum(vals) / len(vals)

        return ContraProResult(
            model_name=model_name,
            total_samples=len(samples),
            accuracy=overall_accuracy,
            accuracy_by_distance=accuracy_by_distance,
            accuracy_by_pronoun=accuracy_by_pronoun,
            sample_results=results,
        )


def save_results(
    result: ContraProResult,
    output_dir: str,
    experiment_name: str = "contrapro",
) -> None:
    """Save ContraPro evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON (without detailed sample results for brevity)
    result_dict = asdict(result)
    result_dict.pop('sample_results', None)  # Remove detailed results from main file

    json_path = output_path / f"{experiment_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"Saved: {json_path}")

    # Save detailed results separately
    if result.sample_results:
        details_path = output_path / f"{experiment_name}_detailed.json"
        with open(details_path, 'w') as f:
            json.dump(result.sample_results, f, indent=2)
        print(f"Saved detailed results: {details_path}")

    # Print formatted results
    print("\n" + "="*60)
    print("CONTRAPRO EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {result.model_name}")
    print(f"Total Samples: {result.total_samples}")
    print(f"Overall Accuracy: {result.accuracy:.2%}")
    print("-"*60)
    print("Accuracy by Antecedent Distance:")
    for bucket, acc in result.accuracy_by_distance.items():
        if acc is not None:
            print(f"  {bucket:>12}: {acc:.2%}")
        else:
            print(f"  {bucket:>12}: N/A")
    print("-"*60)
    print("Accuracy by Pronoun Type:")
    for ptype, acc in sorted(result.accuracy_by_pronoun.items()):
        print(f"  {ptype:>20}: {acc:.2%}")
    print("="*60)


def generate_accuracy_plot(
    results: List[ContraProResult],
    output_dir: str,
) -> None:
    """Generate accuracy vs. distance plot for paper Figure 3."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed, skipping plots")
        return

    output_path = Path(output_dir)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Distance bucket centers for x-axis
    bucket_centers = {
        "0-50": 25, "51-100": 75, "101-200": 150,
        "201-500": 350, "501-1000": 750, "1000+": 1500
    }

    for result in results:
        x_vals = []
        y_vals = []
        for bucket, acc in result.accuracy_by_distance.items():
            if acc is not None and bucket in bucket_centers:
                x_vals.append(bucket_centers[bucket])
                y_vals.append(acc * 100)  # Convert to percentage

        ax.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=8,
               label=result.model_name)

    ax.set_xlabel('Antecedent Distance (tokens)', fontsize=12)
    ax.set_ylabel('Pronoun Disambiguation Accuracy (%)', fontsize=12)
    ax.set_title('ContraPro: Accuracy vs. Context Distance', fontsize=14)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path / 'contrapro_accuracy_vs_distance.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'contrapro_accuracy_vs_distance.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved accuracy plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ContraPro Pronoun Disambiguation Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                       help="Output directory for results")
    parser.add_argument("--data-path", type=str, default="data/contrapro/contrapro.json",
                       help="Path to ContraPro data")
    parser.add_argument("--tokenizer-path", type=str,
                       default="data/tokenizer/tokenizer.json",
                       help="Path to tokenizer")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to evaluate")
    parser.add_argument("--use-synthetic", action="store_true",
                       help="Use synthetic ContraPro data")
    parser.add_argument("--max-distance", type=int, default=2048,
                       help="Maximum context distance for synthetic data")

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

    # Load or create data
    if args.use_synthetic:
        print("Creating synthetic ContraPro data...")
        samples = create_synthetic_contrapro(
            tokenizer,
            num_samples=args.max_samples or 500,
            max_context_length=args.max_distance,
        )
    else:
        samples = load_contrapro_data(
            data_path=args.data_path,
            max_samples=args.max_samples,
        )

    # Evaluate
    evaluator = ContraProEvaluator(model, tokenizer, device)
    result = evaluator.evaluate(samples, model_name=model_name)

    # Save results
    save_results(result, args.output_dir, f"contrapro_{model_name}")

    # Generate plot
    generate_accuracy_plot([result], args.output_dir)


if __name__ == "__main__":
    main()
