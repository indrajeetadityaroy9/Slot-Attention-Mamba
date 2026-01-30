#!/usr/bin/env python3
"""Evaluation script for State Capacity experiments.

USAGE:
    python evaluate.py checkpoint=outputs/01_mqar_cliff/model.pt
    python evaluate.py checkpoint=<path> data.num_pairs=128
"""

import json
from pathlib import Path
from typing import Dict

import torch
import hydra
from omegaconf import DictConfig

from models.checkpoints import load_model_from_checkpoint
from data.mqar import MQARDataset, MQARConfig
from training.eval_utils import compute_batch_accuracy, compute_perplexity


def evaluate_mqar(
    model,
    config: MQARConfig,
    num_samples: int = 1000,
    batch_size: int = 32,
    device: str = "cuda",
    mode: str = "seq2seq",
) -> Dict[str, float]:
    """Evaluate model on MQAR task."""
    model.eval()

    dataset = MQARDataset(
        config=config,
        num_samples=num_samples,
        split="test",
        mode=mode,
    )

    total_token_correct = 0
    total_tokens = 0
    total_sample_correct = 0
    total_samples = 0
    total_perplexity_sum = 0.0
    perplexity_batches = 0

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]

            # Single forward pass for both accuracy and perplexity
            if mode == "decoder_only":
                input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
                labels = torch.stack([b["labels"] for b in batch]).to(device)
                logits = model(None, input_ids)
            else:
                src_ids = torch.stack([b["src_ids"] for b in batch]).to(device)
                tgt_ids = torch.stack([b["tgt_ids"] for b in batch]).to(device)
                labels = torch.stack([b["labels"] for b in batch]).to(device)
                logits = model(src_ids, tgt_ids[:, :-1])

            predictions = logits.argmax(dim=-1)
            mask = labels != -100

            tc, tt, sc = compute_batch_accuracy(predictions, labels, mask)
            total_token_correct += tc
            total_tokens += tt
            total_sample_correct += sc
            total_samples += len(batch)

            total_perplexity_sum += compute_perplexity(logits, labels)
            perplexity_batches += 1

    # Get hybrid positions from model config
    hybrid_positions = getattr(model.config, "hybrid_positions", None)
    if hybrid_positions is not None:
        hybrid_positions = sorted(list(hybrid_positions))

    return {
        "token_accuracy": total_token_correct / max(total_tokens, 1),
        "sample_accuracy": total_sample_correct / max(total_samples, 1),
        "perplexity": total_perplexity_sum / max(perplexity_batches, 1),
        "num_samples": total_samples,
        "num_pairs": config.num_pairs,
        "d_state": getattr(model.config, "d_state", None),
        "hybrid_positions": hybrid_positions,
    }


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Run evaluation pipeline."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation. No CUDA devices found.")

    print("=" * 60)
    print("State Capacity Evaluation")
    print("=" * 60)

    if not cfg.get("checkpoint"):
        raise ValueError("No checkpoint specified. Use: python evaluate.py checkpoint=<path>")

    device = "cuda"
    dtype = torch.bfloat16

    print(f"\nCheckpoint: {cfg.checkpoint}")

    # Load model - errors propagate directly
    print("\nLoading model...")
    model, model_config = load_model_from_checkpoint(cfg.checkpoint, device=device, dtype=dtype)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {param_count / 1e6:.1f}M parameters")

    # Get dataset config - MQARConfig only has num_pairs and num_queries
    # (vocab_size and seq_length are defined in constants.py)
    data_cfg = cfg.get("data", {})

    config = MQARConfig(
        num_pairs=data_cfg.get("num_pairs", 64),
        num_queries=data_cfg.get("num_queries", 16),
    )
    mqar_mode = data_cfg.get("mode", "seq2seq")

    print(f"\nMQAR Config: num_pairs={config.num_pairs}, num_queries={config.num_queries}, mode={mqar_mode}")

    num_samples = cfg.get("eval_samples", 1000)
    batch_size = cfg.training.get("batch_size", 32)

    print(f"Evaluating on {num_samples} samples...")
    results = evaluate_mqar(
        model=model,
        config=config,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
        mode=mqar_mode,
    )

    # Add seed to results for multi-seed aggregation
    results["seed"] = cfg.project.get("seed") if cfg.get("project") else None

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Token Accuracy:  {results['token_accuracy']*100:.2f}%")
    print(f"Sample Accuracy: {results['sample_accuracy']*100:.2f}%")
    print(f"Perplexity:      {results['perplexity']:.2f}")

    # Save results
    output_dir = Path(cfg.get("output_dir", "outputs/evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "mqar_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
