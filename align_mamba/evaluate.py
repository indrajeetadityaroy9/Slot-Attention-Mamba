"""Evaluation for Align-Mamba."""

import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from align_mamba.config import Config, load_yaml
from align_mamba.model import load_checkpoint
from align_mamba.data import MQARDataset


def evaluate(
    model: nn.Module,
    config: Config,
    *,
    num_samples: int = 1000,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    """Evaluate on MQAR task."""
    model.eval()
    dataset = MQARDataset(config.num_pairs, config.num_queries, num_samples, "test")

    correct, total = 0, 0
    ppl_sum, ppl_n = 0.0, 0

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]

            src = torch.stack([b["src_ids"] for b in batch]).to(device)
            tgt = torch.stack([b["tgt_ids"] for b in batch]).to(device)
            labels = torch.stack([b["labels"] for b in batch]).to(device)
            logits = model(src, tgt[:, :-1])
            labels = labels[:, :logits.size(1)]

            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                  ignore_index=-100, reduction='mean')
            ppl_sum += torch.exp(loss).item()
            ppl_n += 1

    return {
        "token_accuracy": correct / total if total > 0 else 0,
        "perplexity": ppl_sum / ppl_n if ppl_n > 0 else float('inf'),
        "num_pairs": config.num_pairs,
        "d_state": config.d_state,
    }


def capacity_cliff(
    model: nn.Module,
    config: Config,
    *,
    num_samples: int = 500,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    """Find capacity cliff where accuracy drops."""
    d_state = config.d_state
    results = []

    for num_pairs in [32, 48, 64, 80, 96, 112, 128, 160, 192, 256]:
        dataset = MQARDataset(num_pairs, min(16, num_pairs), num_samples, "test")
        correct, total = 0, 0

        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
                src = torch.stack([b["src_ids"] for b in batch]).to(device)
                tgt = torch.stack([b["tgt_ids"] for b in batch]).to(device)
                labels = torch.stack([b["labels"] for b in batch]).to(device)
                logits = model(src, tgt[:, :-1])
                labels = labels[:, :logits.size(1)]
                preds = logits.argmax(dim=-1)
                mask = labels != -100
                correct += ((preds == labels) & mask).sum().item()
                total += mask.sum().item()

        acc = correct / total if total > 0 else 0
        above = num_pairs > d_state
        results.append({"num_pairs": num_pairs, "token_accuracy": acc, "above_capacity": above})
        print(f"pairs={num_pairs:3d} acc={acc:.4f} {'ABOVE' if above else 'below'}")

    cliff = None
    for r in results:
        if r["above_capacity"] and r["token_accuracy"] < 0.9:
            cliff = r["num_pairs"]
            break

    return {"results": results, "cliff_point": cliff, "d_state": d_state}


def main():
    import argparse
    parser = argparse.ArgumentParser(prog="align-eval")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    config, eval_cfg = load_yaml(args.config)
    model = load_checkpoint(
        eval_cfg["checkpoint"], config, device="cuda", dtype=torch.bfloat16,
    )

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if eval_cfg["mode"] == "capacity_cliff":
        print(f"\nCapacity cliff eval (d_state={config.d_state})")
        results = capacity_cliff(
            model, config,
            num_samples=eval_cfg.get("num_samples", 500),
            batch_size=eval_cfg.get("batch_size", 32),
        )
        print(f"Cliff at pairs={results['cliff_point']}")
    else:
        results = evaluate(
            model, config,
            num_samples=eval_cfg.get("num_samples", 1000),
            batch_size=eval_cfg.get("batch_size", 32),
        )
        print(f"acc={results['token_accuracy']:.4f} ppl={results['perplexity']:.2f}")

    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    shutil.copy2(args.config, out / "config.yaml")
    print(f"Results written to {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
