"""Configuration for Align-Mamba.

Minimal configuration surface for publication-grade experiments.
Fixed values (not configurable): vocab_size=8192, num_workers=8, gradient_accum=1.
"""

from dataclasses import dataclass, fields, asdict
from typing import List
import argparse
import math
import yaml

# Token constants (fixed for MQAR)
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
QUERY_TOKEN_ID = 4
KEY_TOKEN_START = 10
KEY_TOKEN_END = 4096
VALUE_TOKEN_START = 4096
VALUE_TOKEN_END = 8192
VOCAB_SIZE = 8192
MAX_SEQ_LEN = 8192


@dataclass
class Config:
    """Polar-Mem-Mamba configuration.

    Core research parameters:
        d_state: Mamba state dimension (capacity cliff at this value)
        num_pairs: Key-value pairs in MQAR (main capacity test)
        block_type: mamba2 | polarized | memmamba | polarized_mem
        hybrid_positions: Cross-attention layer indices (None = auto)
        encoder_layers: 0 for decoder-only, >0 for encoder-decoder
    """

    # Architecture (essential for ablations)
    d_model: int = 256
    encoder_layers: int = 6
    decoder_layers: int = 6
    d_state: int = 64
    n_heads: int = 4
    hybrid_positions: List[int] = None
    block_type: str = "polarized_mem"

    # MemMamba parameters (for ablation completeness)
    mem_pool_size: int = 50
    mem_summary_dim: int = 64
    mem_tau1: float = 0.5
    mem_tau2: float = 0.3
    mem_update_freq: int = 4

    # Training
    batch_size: int = 256
    max_steps: int = 100000
    learning_rate: float = 3e-4
    output_dir: str = "outputs"

    # Data (capacity testing)
    num_pairs: int = 64
    num_queries: int = 16
    num_samples: int = 100000

    # Runtime
    seed: int = 42

    @property
    def vocab_size(self) -> int:
        """Fixed vocabulary size for MQAR."""
        return VOCAB_SIZE

    @property
    def label_smoothing(self) -> float:
        """Auto-computed label smoothing based on vocab size."""
        return 0.05 * math.log(VOCAB_SIZE) / (math.log(VOCAB_SIZE) + 1)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        data.update(data.pop("sota", {}))
        # Handle legacy parameter names
        renames = {
            "memmamba_pool_size": "mem_pool_size",
            "memmamba_summary_dim": "mem_summary_dim",
            "memmamba_tau1": "mem_tau1",
            "memmamba_tau2": "mem_tau2",
            "memmamba_cross_layer_freq": "mem_update_freq",
        }
        for old, new in renames.items():
            if old in data:
                data[new] = data.pop(old)
        # Remove deprecated fields
        for deprecated in ("vocab_size", "mode", "resume_from", "label_smoothing",
                          "gradient_accumulation_steps", "num_workers"):
            data.pop(deprecated, None)
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

    def to_dict(self) -> dict:
        """Convert to dict including computed properties."""
        d = asdict(self)
        d["vocab_size"] = self.vocab_size
        d["label_smoothing"] = self.label_smoothing
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create Config from dict (e.g., checkpoint config)."""
        # Handle legacy parameter names
        renames = {
            "memmamba_pool_size": "mem_pool_size",
            "memmamba_summary_dim": "mem_summary_dim",
            "memmamba_tau1": "mem_tau1",
            "memmamba_tau2": "mem_tau2",
            "memmamba_cross_layer_freq": "mem_update_freq",
        }
        data = dict(data)  # Copy to avoid mutating input
        for old, new in renames.items():
            if old in data:
                data[new] = data.pop(old)
        # Remove computed/deprecated fields
        for deprecated in ("vocab_size", "mode", "resume_from", "label_smoothing",
                          "gradient_accumulation_steps", "num_workers"):
            data.pop(deprecated, None)
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def from_args(cls, args: List[str] = None) -> "Config":
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str)
        for f in fields(cls):
            if f.type == List[int]:
                parser.add_argument(f"--{f.name}", type=int, nargs="+", default=None)
            elif f.type == int:
                parser.add_argument(f"--{f.name}", type=int, default=None)
            elif f.type == float:
                parser.add_argument(f"--{f.name}", type=float, default=None)
            elif f.type == str:
                parser.add_argument(f"--{f.name}", type=str, default=None)
        parsed = parser.parse_args(args)

        base = cls.from_yaml(parsed.config) if parsed.config else cls()
        updates = {f.name: getattr(parsed, f.name) for f in fields(cls)
                  if getattr(parsed, f.name) is not None}
        return cls(**{**asdict(base), **updates})
