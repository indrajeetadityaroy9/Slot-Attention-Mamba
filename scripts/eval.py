import argparse
import json
from pathlib import Path

from config import load_yaml, DTYPE_MAP
from model import load_checkpoint
from evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    config = load_yaml(args.config)
    compute_dtype = DTYPE_MAP[config.compute_dtype]
    model = load_checkpoint(config.eval_checkpoint, config, dtype=compute_dtype)

    results = run_evaluation(model, config)
    results["params"] = sum(p.numel() for p in model.parameters())
    print(f"[eval_complete] {results}")

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
