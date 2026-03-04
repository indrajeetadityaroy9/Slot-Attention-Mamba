import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
_RESERVED_TOKENS = 4  # MQAR reserves 0-3 for PAD/BOS/EOS/SEP.

IGNORE_INDEX = -100

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
MIXED_PRECISION_MAP = {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}


def key_range(vocab_size: int) -> tuple[int, int]:
    return (_RESERVED_TOKENS, vocab_size // 2)


def value_range(vocab_size: int) -> tuple[int, int]:
    return (vocab_size // 2, vocab_size)


def split_seed_offset(split: str) -> int:
    """Deterministic per-split seed offset derived from the split name."""
    return int.from_bytes(hashlib.sha256(split.encode()).digest()[:4], "big")


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
    val_ratio: float = 0.0

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

    # Self-calibrating fields (0 = derive automatically).
    rope_base: float = 0.0
    decay_gamma_init: float = 0.0
    label_smoothing: float = 0.0
    adam_beta1: float = 0.0
    adam_beta2: float = 0.0
    n_registers: int = 0
    max_gen_toks: int = 0
    compute_dtype: str = "bfloat16"
    num_eval_seeds: int = 0

    # Layer composition (0 = derive automatically).
    encoder_attn_start: int = 0
    encoder_attn_stride: int = 0
    decoder_inject_end: int = 0
    decoder_inject_stride: int = 0
    decoder_causal_attn_pos: int = -1


def derive_defaults(config: Config) -> Config:
    """Fill every zero-valued self-calibrating field from first principles."""
    head_dim = config.d_model // config.n_heads

    # Effective sequence length for this task.
    if config.task == "lm":
        eff_seq = config.lm_seq_length
    else:
        eff_seq = config.num_queries + 2  # tgt length

    # RoPE base: NTK-aware frequency scaling.
    if config.rope_base == 0.0:
        config.rope_base = eff_seq ** (head_dim / (head_dim - 2))

    # Decay init: softplus_inverse(1/eff_seq) so initial per-step decay ~ 1/T.
    if config.decay_gamma_init == 0.0:
        target = 1.0 / eff_seq
        config.decay_gamma_init = math.log(math.expm1(target))

    # Label smoothing: entropy-normalized, dimensionless ratio.
    if config.label_smoothing == 0.0:
        config.label_smoothing = 1.0 / math.log(config.vocab_size)

    # Adam beta1: first-moment window ~ sqrt(epoch_steps).
    epoch_steps = config.num_samples // config.batch_size
    if config.adam_beta1 == 0.0:
        config.adam_beta1 = 1.0 - 1.0 / math.sqrt(epoch_steps)

    # Adam beta2: second-moment window ~ sqrt(max_steps).
    if config.adam_beta2 == 0.0:
        config.adam_beta2 = 1.0 - 1.0 / math.sqrt(config.max_steps)

    # Validation ratio: inversely proportional to log of epoch length.
    if config.val_ratio == 0.0:
        config.val_ratio = 1.0 / math.log(epoch_steps)

    # Registers: one per attention head.
    if config.n_registers == 0:
        config.n_registers = config.n_heads

    # Max generation tokens: bounded by model context window.
    if config.max_gen_toks == 0:
        config.max_gen_toks = config.lm_seq_length

    # Eval seeds: scales with sqrt of eval batches.
    if config.num_eval_seeds == 0:
        config.num_eval_seeds = round(math.sqrt(config.eval_num_samples / config.eval_batch_size))

    # --- Layer composition ---
    # Attention density proportional to d_state/d_model.
    attn_ratio = config.d_state / config.d_model

    # Encoder: n_attn layers placed evenly.
    if config.encoder_layers > 0 and config.encoder_attn_start == 0 and config.encoder_attn_stride == 0:
        n_attn_enc = round(config.encoder_layers * attn_ratio)
        config.encoder_attn_stride = config.encoder_layers // n_attn_enc
        config.encoder_attn_start = config.encoder_layers - n_attn_enc * config.encoder_attn_stride

    # Decoder injection: same ratio.
    if config.decoder_layers > 0 and config.decoder_inject_end == 0 and config.decoder_inject_stride == 0:
        n_inject = round(config.decoder_layers * attn_ratio)
        config.decoder_inject_stride = config.decoder_layers // n_inject
        config.decoder_inject_end = n_inject * config.decoder_inject_stride

    # Decoder causal attention: penultimate layer.
    if config.decoder_layers > 0 and config.decoder_causal_attn_pos == -1:
        config.decoder_causal_attn_pos = config.decoder_layers - 2

    return config


def load_yaml(path: str) -> Config:
    from evaluation import _sweep_ratios, _eval_grid

    with open(path) as f:
        raw = yaml.safe_load(f)

    if "base" in raw:
        base_path = str(Path(path).parent / raw.pop("base"))
        with open(base_path) as bf:
            base_raw = yaml.safe_load(bf)
        base_raw.update(raw)
        raw = base_raw

    flat = {}
    for v in raw.values():
        if isinstance(v, dict):
            flat.update(v)

    config = Config(**flat)
    derive_defaults(config)

    # Size RoPE buffers to cover evaluation sweeps.
    if config.task == "mqar" and config.eval_max_num_pairs == 0:
        sweep_max = int(config.d_state * max(_sweep_ratios(config.d_state)))
        grid_max = max(int(x) for x in config.eval_grid.split(",")) if config.eval_grid else max(_eval_grid(config.d_state))
        config.eval_max_num_pairs = max(sweep_max, grid_max)

    if config.task == "lm":
        config.vocab_size = AutoTokenizer.from_pretrained(config.lm_tokenizer).vocab_size
        config.label_smoothing = 1.0 / math.log(config.vocab_size)

    return config
