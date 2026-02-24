from dataclasses import dataclass
from pathlib import Path

import yaml

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
_RESERVED_TOKENS = 4  # MQAR reserves 0-3 for PAD/BOS/EOS/SEP.

IGNORE_INDEX = -100


def key_range(vocab_size: int) -> tuple[int, int]:
    return (_RESERVED_TOKENS, vocab_size // 2)


def value_range(vocab_size: int) -> tuple[int, int]:
    return (vocab_size // 2, vocab_size)


@dataclass
class Config:
    d_model: int = 256
    encoder_layers: int = 6
    decoder_layers: int = 6
    d_state: int = 64
    n_heads: int = 4
    vocab_size: int = 8192
    block_size: int = 4
    n_householder_steps: int = 2
    kronecker_partitions: int = 5
    kronecker_subdim: int = 4
    top_k_slots: int = 8
    use_pdma: bool = True
    use_surprise_gate: bool = True
    use_injection: bool = True
    mamba_expand: int = 2
    mamba_d_conv: int = 4
    dropout: float = 0.1

    batch_size: int = 256
    max_steps: int = 100000

    num_pairs: int = 128
    num_queries: int = 16
    num_samples: int = 100000
    val_ratio: float = 0.1

    seed: int = 42
    output_dir: str = "results"
    resume_from: str = ""

    task: str = "mqar"
    lm_seq_length: int = 1024

    lm_dataset: str = "Salesforce/wikitext"
    lm_dataset_config: str = "wikitext-103-raw-v1"
    lm_dataset_streaming: bool = False
    lm_tokenizer: str = "gpt2"

    eval_mode: str = "standard"
    eval_num_samples: int = 1000
    eval_batch_size: int = 32
    eval_checkpoint: str = "results/best"
    eval_max_num_pairs: int = 0
    eval_grid: str = ""
    eval_harness_tasks: str = "arc_easy,hellaswag,piqa,winogrande,copa"


def load_yaml(path: str):
    with open(path) as f:
        raw = yaml.safe_load(f)

    if "base" in raw:
        base_path = str(Path(path).parent / raw.pop("base"))
        with open(base_path) as bf:
            base_raw = yaml.safe_load(bf)
        merged = base_raw.copy()
        for k, v in raw.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        raw = merged

    flat = {}
    for section in ("run", "model", "data", "training", "evaluation"):
        if section in raw:
            flat.update(raw[section])

    config = Config(**flat)

    # Size RoPE buffers to cover evaluation sweeps.
    if config.task == "mqar" and config.eval_max_num_pairs == 0:
        sweep_max = int(config.d_state * 4.0)
        grid_max = max((int(x) for x in config.eval_grid.split(",") if x.strip()), default=256)
        config.eval_max_num_pairs = max(sweep_max, grid_max)

    if config.task == "lm":
        from transformers import AutoTokenizer
        config.vocab_size = AutoTokenizer.from_pretrained(config.lm_tokenizer).vocab_size

    assert not (config.encoder_layers == 0 and config.use_injection), \
        "use_injection requires encoder_layers > 0"

    return config
