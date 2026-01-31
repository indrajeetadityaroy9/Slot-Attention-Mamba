"""Evaluation for Align-Mamba."""

import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from align_mamba.model import load_checkpoint
from align_mamba.data import MQARDataset


def evaluate(model, num_pairs: int, num_queries: int, num_samples: int,
             batch_size: int, device: str) -> dict:
    """Evaluate on MQAR task."""
    model.eval()
    dataset = MQARDataset(num_pairs, num_queries, num_samples, "test")

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
        "token_accuracy": correct / total,
        "perplexity": ppl_sum / ppl_n,
        "num_pairs": num_pairs,
        "d_state": model.config.d_state,
    }


def capacity_cliff(model, num_samples: int, batch_size: int, device: str) -> dict:
    """Find capacity cliff where accuracy drops."""
    d_state = model.config.d_state
    results = []

    for num_pairs in [32, 48, 64, 80, 96, 112, 128, 160, 192, 256]:
        r = evaluate(model, num_pairs, min(16, num_pairs), num_samples, batch_size, device)
        above = num_pairs > d_state
        results.append({"num_pairs": num_pairs, **r, "above_capacity": above})
        print(f"pairs={num_pairs:3d} acc={r['token_accuracy']:.4f} {'ABOVE' if above else 'below'}")

    cliff = None
    for r in results:
        if r["num_pairs"] > d_state and r["token_accuracy"] < 0.9:
            cliff = r["num_pairs"]
            break

    return {"results": results, "cliff_point": cliff, "d_state": d_state}


def main():
    parser = argparse.ArgumentParser(prog="align-eval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_pairs", type=int, default=64)
    parser.add_argument("--num_queries", type=int, default=16)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--capacity_cliff", action="store_true")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) / "checkpoint.pt"
    model, config = load_checkpoint(str(ckpt_path), args.device, torch.bfloat16)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.capacity_cliff:
        print(f"\nCapacity cliff eval (d_state={config.d_state})")
        results = capacity_cliff(model, args.num_samples, args.batch_size, args.device)
        with open(out / "capacity_cliff.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Cliff at pairs={results['cliff_point']}")
    else:
        results = evaluate(model, args.num_pairs, args.num_queries,
                          args.num_samples, args.batch_size, args.device)
        print(f"acc={results['token_accuracy']:.4f} ppl={results['perplexity']:.2f}")
        with open(out / "results.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
