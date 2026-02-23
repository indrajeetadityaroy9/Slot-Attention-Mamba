import argparse
import json
from pathlib import Path

import torch

from align_mamba.config import load_yaml
from align_mamba.model import load_checkpoint
from align_mamba.evaluation import run_evaluation
from align_mamba.training import emit_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    config = load_yaml(args.config)
    model = load_checkpoint(config.eval_checkpoint, config, dtype=torch.bfloat16)

    results = run_evaluation(model, config)
    results["params"] = sum(p.numel() for p in model.parameters())
    emit_log(event="eval_complete", **results)

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
