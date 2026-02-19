"""MQAR dataset for state capacity testing."""

import random
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from align_mamba.config import (
    Config, PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, SEP_TOKEN_ID,
    KEY_TOKEN_START, KEY_TOKEN_END, VALUE_TOKEN_START, VALUE_TOKEN_END,
)

_NUM_WORKERS = 8


Split = Literal["train", "validation", "test"]


class MQARDataset(Dataset):
    """MQAR: Multi-Query Associative Recall."""

    SPLIT_SEEDS: Dict[str, int] = {"train": 42, "validation": 1042, "test": 2042}

    def __init__(self, num_pairs: int, num_queries: int, num_samples: int, split: Split):
        self.num_pairs = num_pairs
        self.num_queries = min(num_queries, num_pairs)

        self._rng = random.Random(self.SPLIT_SEEDS[split])
        self._samples = [self._gen() for _ in range(num_samples)]

    def _gen(self) -> Dict[str, torch.Tensor]:
        keys = self._rng.sample(range(KEY_TOKEN_START, KEY_TOKEN_END), self.num_pairs)
        values = [self._rng.choice(range(VALUE_TOKEN_START, VALUE_TOKEN_END)) for _ in range(self.num_pairs)]
        kv = dict(zip(keys, values))
        qkeys = self._rng.sample(keys, self.num_queries)

        src = [BOS_TOKEN_ID]
        for k, v in zip(keys, values):
            src.extend([k, SEP_TOKEN_ID, v])
        src.append(EOS_TOKEN_ID)

        tgt = [BOS_TOKEN_ID] + qkeys + [EOS_TOKEN_ID]
        labels = [-100] + [kv[k] for k in qkeys]

        max_src = 1 + self.num_pairs * 3 + 1
        max_tgt = 1 + self.num_queries + 1
        src += [PAD_TOKEN_ID] * (max_src - len(src))
        tgt += [PAD_TOKEN_ID] * (max_tgt - len(tgt))
        labels += [-100] * (max_tgt - 1 - len(labels))

        return {
            'src_ids': torch.tensor(src, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch for seq2seq."""
    def pad(seqs, val):
        max_len = max(len(s) for s in seqs)
        return torch.stack([F.pad(s, (0, max_len - len(s)), value=val) for s in seqs])

    src = pad([b['src_ids'] for b in batch], PAD_TOKEN_ID)
    tgt = pad([b['tgt_ids'] for b in batch], PAD_TOKEN_ID)
    labels = pad([b['labels'] for b in batch], -100)
    return {'src_ids': src, 'tgt_ids': tgt, 'labels': labels[:, :tgt.size(1) - 1]}


def create_dataloaders(config: Config, *, world_size: int, rank: int) -> Tuple[DataLoader, DataLoader]:
    """Create distributed train/val loaders."""
    train = MQARDataset(config.num_pairs, config.num_queries, config.num_samples, "train")
    val = MQARDataset(config.num_pairs, config.num_queries, config.num_samples // 10, "validation")

    train_sampler = DistributedSampler(train, world_size, rank, shuffle=True)
    val_sampler = DistributedSampler(val, world_size, rank, shuffle=False)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": _NUM_WORKERS,
        "collate_fn": collate,
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": True,
        "worker_init_fn": lambda w: (np.random.seed(torch.initial_seed() % 2**32 + w),
                                      random.seed(torch.initial_seed() % 2**32 + w)),
    }

    return (
        DataLoader(train, sampler=train_sampler, **loader_kwargs),
        DataLoader(val, sampler=val_sampler, **loader_kwargs),
    )
