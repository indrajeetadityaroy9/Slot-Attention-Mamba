import itertools
import math
import os
import random

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config import (
    Config,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    SEP_TOKEN_ID,
    IGNORE_INDEX,
    key_range,
    value_range,
    split_seed_offset,
)

_NUM_WORKERS = os.cpu_count() // 2


def _streaming_targets(config: Config) -> dict[str, int]:
    """Derive token counts from training scale.

    Train: enough tokens for max_steps at full batch utilization.
    Val/test: train / log(max_steps), scaling with training duration.
    """
    train_tokens = config.batch_size * config.lm_seq_length * config.max_steps
    val_test = train_tokens // round(math.log(config.max_steps))
    return {"train": train_tokens, "validation": val_test, "test": val_test}


def _streaming_skips(targets: dict[str, int]) -> dict[str, int]:
    """Cumulative offsets so splits don't overlap in the stream."""
    return {
        "train": 0,
        "validation": targets["train"],
        "test": targets["train"] + targets["validation"],
    }


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

        self._rng = random.Random(seed + split_seed_offset(split))
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


def _load_hf_dataset(
    dataset_path: str,
    config_name: str,
    split: str,
    seq_length: int,
    tokenizer: AutoTokenizer,
):
    ds = load_dataset(dataset_path, config_name, split=split)

    def tokenize_fn(batch):
        return {"input_ids": tokenizer(batch["text"], add_special_tokens=False)["input_ids"]}

    ds = ds.map(tokenize_fn, batched=True, num_proc=_NUM_WORKERS, remove_columns=ds.column_names)

    def group_texts(batch):
        concatenated = list(itertools.chain.from_iterable(batch["input_ids"]))
        total_length = (len(concatenated) // seq_length) * seq_length
        result = [concatenated[i : i + seq_length] for i in range(0, total_length, seq_length)]
        return {"input_ids": result, "labels": result}

    ds = ds.map(group_texts, batched=True, num_proc=_NUM_WORKERS, remove_columns=["input_ids"])
    ds.set_format("torch")
    return ds


def _load_streaming_dataset(
    config: Config,
    split: str,
    tokenizer: AutoTokenizer,
) -> LMDataset:
    ds = load_dataset(config.lm_dataset, config.lm_dataset_config, split="train", streaming=True)
    targets = _streaming_targets(config)
    skips = _streaming_skips(targets)
    skip = skips[split]
    target = targets[split]

    all_tokens: list[int] = []
    skipped = 0
    for example in ds:
        toks = tokenizer.encode(example["text"], add_special_tokens=False)
        if skipped < skip:
            skipped += len(toks)
            continue
        all_tokens.extend(toks)
        if len(all_tokens) >= target:
            break

    tokens = torch.tensor(all_tokens[:target], dtype=torch.long)
    return LMDataset(tokens, config.lm_seq_length)


def load_lm_dataset(config: Config, split: str):
    tokenizer = AutoTokenizer.from_pretrained(config.lm_tokenizer)
    if config.lm_dataset_streaming:
        return _load_streaming_dataset(config, split, tokenizer)
    return _load_hf_dataset(
        config.lm_dataset, config.lm_dataset_config,
        split, config.lm_seq_length, tokenizer,
    )


def create_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": _NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": True,
    }

    if config.task == "lm":
        train = load_lm_dataset(config, "train")
        val = load_lm_dataset(config, "validation")
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
