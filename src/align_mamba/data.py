import random
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader

from align_mamba.config import (
    Config,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    SEP_TOKEN_ID,
    IGNORE_INDEX,
    key_range,
    value_range,
)

_SPLIT_OFFSETS: dict[str, int] = {"train": 0, "validation": 1000, "test": 2000}
_WIKITEXT_SPLITS: dict[str, str] = {
    "train": "wiki.train.raw",
    "validation": "wiki.valid.raw",
    "test": "wiki.test.raw",
}
_NUM_WORKERS = 4

_FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
_FINEWEB_SUBSET = "sample-10BT"
_FINEWEB_TOKEN_TARGETS: dict[str, int] = {
    "train": 100_000_000,
    "validation": 1_000_000,
    "test": 1_000_000,
}


def _get_tokenizer(name: str) -> Callable[[str], list[int]]:
    if name == "gpt2":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return enc.encode_ordinary

    if name == "llama2":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        return lambda text: tok.encode(text, add_special_tokens=False)

    raise KeyError(name)


class MQARDataset(Dataset):
    def __init__(
        self,
        num_pairs: int,
        num_queries: int,
        num_samples: int,
        split: str,
        *,
        vocab_size: int,
        seed: int = 42,
    ):
        self.num_pairs = num_pairs
        self.num_queries = num_queries
        self.key_start, self.key_end = key_range(vocab_size)
        self.val_start, self.val_end = value_range(vocab_size)

        self._rng = random.Random(seed + _SPLIT_OFFSETS[split])
        self._samples = [self._gen() for _ in range(num_samples)]

    def _gen(self) -> dict[str, torch.Tensor]:
        keys = self._rng.sample(range(self.key_start, self.key_end), self.num_pairs)
        values = [self._rng.choice(range(self.val_start, self.val_end)) for _ in range(self.num_pairs)]
        kv = dict(zip(keys, values))
        qkeys = self._rng.sample(keys, self.num_queries)

        src = [BOS_TOKEN_ID]
        for k, v in zip(keys, values):
            src.extend([k, SEP_TOKEN_ID, v])
        src.append(EOS_TOKEN_ID)

        tgt = [BOS_TOKEN_ID] + qkeys + [EOS_TOKEN_ID]
        labels = [IGNORE_INDEX] + [kv[k] for k in qkeys]

        return {
            "src_ids": torch.tensor(src, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


class LMDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_length: int):
        n_chunks = len(tokens) // seq_length
        self.chunks = tokens[: n_chunks * seq_length].view(n_chunks, seq_length)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return {"input_ids": chunk, "labels": chunk}


def _load_wikitext(data_dir: str, split: str, seq_length: int, tokenizer_name: str) -> LMDataset:
    path = Path(data_dir) / _WIKITEXT_SPLITS[split]
    encode = _get_tokenizer(tokenizer_name)
    tokens = torch.tensor(encode(path.read_text(encoding="utf-8")), dtype=torch.long)
    return LMDataset(tokens, seq_length)


def _load_fineweb(split: str, seq_length: int, tokenizer_name: str) -> LMDataset:
    from datasets import load_dataset

    encode = _get_tokenizer(tokenizer_name)
    # FineWeb-Edu exposes only a train split; use streaming and cap by token target.
    ds = load_dataset(_FINEWEB_DATASET, _FINEWEB_SUBSET, split="train", streaming=True)
    target = _FINEWEB_TOKEN_TARGETS[split]

    all_tokens: list[int] = []
    for example in ds:
        all_tokens.extend(encode(example["text"]))
        if len(all_tokens) >= target:
            break

    tokens = torch.tensor(all_tokens[:target], dtype=torch.long)
    return LMDataset(tokens, seq_length)


def create_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": _NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": True,
    }

    if config.task == "lm":
        if config.lm_dataset == "fineweb":
            train = _load_fineweb("train", config.lm_seq_length, config.lm_tokenizer)
            val = _load_fineweb("validation", config.lm_seq_length, config.lm_tokenizer)
        else:
            train = _load_wikitext(config.lm_data_dir, "train", config.lm_seq_length, config.lm_tokenizer)
            val = _load_wikitext(config.lm_data_dir, "validation", config.lm_seq_length, config.lm_tokenizer)
        return (
            DataLoader(train, shuffle=True, **loader_kwargs),
            DataLoader(val, shuffle=False, **loader_kwargs),
        )

    ds_kwargs = {"vocab_size": config.vocab_size, "seed": config.seed}
    train = MQARDataset(config.num_pairs, config.num_queries, config.num_samples, "train", **ds_kwargs)
    val_samples = int(config.num_samples * config.val_ratio)
    val = MQARDataset(config.num_pairs, config.num_queries, val_samples, "validation", **ds_kwargs)

    return (
        DataLoader(train, shuffle=True, **loader_kwargs),
        DataLoader(val, shuffle=False, **loader_kwargs),
    )
