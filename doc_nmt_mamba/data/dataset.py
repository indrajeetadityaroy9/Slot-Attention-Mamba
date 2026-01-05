"""
Datasets for Document-Level NMT and MQAR Synthetic Task.

Contains:
- MQAR (Multi-Query Associative Recall):
  - MQARConfig: Configuration for MQAR task
  - MQARDataset: Synthetic dataset for state capacity testing
  - MQARCurriculumGenerator: Curriculum learning for MQAR
  - compute_mqar_accuracy: Evaluation metrics

- NMT Datasets:
  - DocumentNMTDataset: Base class for document-level NMT
  - IWSLT14Dataset: IWSLT14 De-En dataset
  - OPUSBooksDataset: OPUS Books with document structure
  - NewsCommentaryDataset: News Commentary dataset
  - StreamingDocumentDataset: For large corpora
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any, Iterator
from pathlib import Path
import hashlib
import random
import logging

import torch
from torch.utils.data import Dataset, IterableDataset

from .collator import DocumentSample, ConcatenationAugmenter

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def get_split_hash(
    text: str,
    num_buckets: int = 100,
    seed: int = 42,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> str:
    """
    Get deterministic hash-based split for a text.

    This ensures consistent train/val/test splits across runs
    without needing to store split assignments.

    Args:
        text: Text to hash (typically document ID or first sentence)
        num_buckets: Number of buckets for splitting
        seed: Random seed (mixed with hash for reproducibility)
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set

    Returns:
        Split name: "train", "validation", or "test"
    """
    # Mix seed into the hash for reproducibility
    hash_input = f"{seed}:{text}"
    hash_bytes = hashlib.md5(hash_input.encode('utf-8')).digest()
    hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
    bucket = hash_int % num_buckets

    # Assign split based on bucket thresholds
    test_threshold = int(test_ratio * num_buckets)
    val_threshold = test_threshold + int(val_ratio * num_buckets)

    if bucket < test_threshold:
        return "test"
    elif bucket < val_threshold:
        return "validation"
    else:
        return "train"


# =============================================================================
# MQAR (Multi-Query Associative Recall)
# =============================================================================

@dataclass
class MQARConfig:
    """
    Configuration for MQAR synthetic task.

    CRITICAL: d_state=64 forces state capacity cliff.
    This tests whether the model can compress information
    when num_pairs exceeds d_state.

    The MQAR task structure is:
    [BOS, key1, :, val1, key2, :, val2, ..., QUERY, k1, k2, ..., EOS]

    Where queries are a SUBSET of the keys, and the model must
    output the corresponding values.
    """
    d_state: int = 64  # FORCES state capacity cliff
    num_pairs: int = 64  # Number of key-value pairs
    num_queries: int = 16  # Number of queries
    vocab_size: int = 8192  # Total vocabulary size
    seq_length: int = 512  # Maximum sequence length

    # Special token IDs
    pad_token_id: int = 0  # Padding token
    bos_token_id: int = 1  # Beginning of sequence
    eos_token_id: int = 2  # End of sequence
    sep_token_id: int = 3  # Separator token (generic)
    kv_sep_token_id: int = 3  # Separator between key and value (:)
    query_token_id: int = 4  # QUERY marker token

    # Token ranges
    key_token_start: int = 10  # Key range start
    key_token_end: int = 4096  # Key range end (exclusive)
    value_token_start: int = 4096  # Value range start
    value_token_end: int = 8192  # Value range end (exclusive)

    seed: Optional[int] = None

    # Backward compatibility properties
    @property
    def key_range(self) -> Tuple[int, int]:
        return (self.key_token_start, self.key_token_end)

    @property
    def value_range(self) -> Tuple[int, int]:
        return (self.value_token_start, self.value_token_end)

    @property
    def sep_token(self) -> int:
        return self.kv_sep_token_id

    @property
    def pad_token(self) -> int:
        return self.pad_token_id


class MQARDataset(Dataset):
    """
    Synthetic MQAR Dataset for state capacity testing.

    CRITICAL REQUIREMENTS:
    1. Keys are UNIQUE per sample (no duplicate keys)
    2. Queries are a SUBSET of keys (all queries have valid answers)
    3. Structure: [key1 : val1 key2 : val2 ... QUERY k1 k2 ... EOS] - NO INTERLEAVING
    4. d_state=64 to force state capacity cliff

    MODES:
    - "decoder_only": Concatenated sequence for pure Mamba baseline (TC0)
      Returns: input_ids=[BOS pairs... QUERY queries... EOS], labels=[values at query positions]
    - "seq2seq": Split sequence for Hybrid model (NC1) to test cross-attention retrieval
      Returns: src_ids=[BOS pairs... EOS], tgt_ids=[BOS queries... EOS], labels=[expected values]

    Reference: Based on "Hungry Hungry Hippos" and Mamba papers.
    """

    def __init__(
        self,
        config: MQARConfig,
        num_samples: int = 10000,
        split: str = "train",
        seed: Optional[int] = None,
        mode: str = "decoder_only",
    ):
        """
        Args:
            config: MQAR configuration
            num_samples: Number of samples to generate
            split: "train", "validation", or "test"
            seed: Random seed for reproducibility (overrides config.seed)
            mode: "decoder_only" (Pure Mamba) or "seq2seq" (Hybrid with cross-attention)
        """
        if mode not in ("decoder_only", "seq2seq"):
            raise ValueError(f"mode must be 'decoder_only' or 'seq2seq', got '{mode}'")

        self.config = config
        self.num_samples = num_samples
        self.split = split
        self.mode = mode

        # ===== Input Validation (crash prevention) =====
        # Calculate minimum sequence length needed:
        # BOS + (num_pairs * 3 tokens each: key, sep, value) + QUERY + num_queries + EOS
        num_queries = min(config.num_queries, config.num_pairs)
        min_seq_len = 1 + (config.num_pairs * 3) + 1 + num_queries + 1

        if config.seq_length < min_seq_len:
            old_seq_len = config.seq_length
            # Auto-extend with 10% buffer for safety
            config.seq_length = int(min_seq_len * 1.1)
            logger.warning(
                f"seq_length={old_seq_len} too short for num_pairs={config.num_pairs}. "
                f"Auto-extended to {config.seq_length} (min required: {min_seq_len})"
            )

        # Validate vocab_size is sufficient for unique keys
        key_range_size = config.key_token_end - config.key_token_start
        if key_range_size < config.num_pairs:
            raise ValueError(
                f"Key range [{config.key_token_start}, {config.key_token_end}) "
                f"has only {key_range_size} unique keys, but num_pairs={config.num_pairs} required. "
                f"Increase key_token_end or decrease num_pairs."
            )

        # Set seed for reproducibility (seed param takes priority)
        self._seed = seed if seed is not None else (config.seed if config.seed else 42)
        self._rng = random.Random(self._seed)

        # Adjust seed by split for different data
        if split == "validation":
            self._rng.seed(self._seed + 1000)
        elif split == "test":
            self._rng.seed(self._seed + 2000)

        # Pre-generate all samples for consistency
        self._samples = [self._generate_sample(i) for i in range(num_samples)]

    def _generate_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a single MQAR sample.

        For decoder_only mode:
            Structure: [BOS key1 : val1 key2 : val2 ... QUERY k1 k2 ... EOS PAD...]
            Returns: input_ids, labels (with values at query positions)

        For seq2seq mode:
            Source: [BOS key1 : val1 key2 : val2 ... EOS]  (Context for encoder)
            Target: [BOS k1 k2 ... EOS]  (Queries for decoder)
            Labels: [val1 val2 ... EOS]  (Expected values, aligned with queries)
        """
        # Generate unique keys
        key_range = range(self.config.key_token_start, self.config.key_token_end)
        keys = self._rng.sample(list(key_range), self.config.num_pairs)

        # Generate values (can have duplicates)
        value_range = range(self.config.value_token_start, self.config.value_token_end)
        values = [self._rng.choice(list(value_range)) for _ in range(self.config.num_pairs)]

        # Build key-value mapping
        kv_map = dict(zip(keys, values))

        # Select queries (subset of keys)
        query_keys = self._rng.sample(keys, min(self.config.num_queries, len(keys)))

        if self.mode == "seq2seq":
            # ===== SEQ2SEQ MODE (for Hybrid model with cross-attention) =====
            # This tests the NC1 hypothesis: cross-attention can retrieve from encoder.
            #
            # Source (Encoder): [BOS key1 : val1 key2 : val2 ... EOS]
            # Target (Decoder): [BOS q1 q2 ... qM EOS]
            #
            # Label alignment for teacher forcing:
            # - Decoder input: [BOS, q1, q2, ..., qM] (truncated, no EOS)
            # - At position 0 (sees BOS): no prediction -> label = -100
            # - At position 1 (sees BOS, q1): predict v1
            # - At position i (sees BOS, q1..qi): predict vi
            # - Labels: [-100, v1, v2, ..., vM] aligned with decoder positions

            # Build source sequence (context with key-value pairs)
            src_ids = [self.config.bos_token_id]
            for k, v in zip(keys, values):
                src_ids.append(k)
                src_ids.append(self.config.kv_sep_token_id)  # :
                src_ids.append(v)
            src_ids.append(self.config.eos_token_id)

            # Build target sequence (queries) - decoder input
            tgt_ids = [self.config.bos_token_id]
            for qk in query_keys:
                tgt_ids.append(qk)
            tgt_ids.append(self.config.eos_token_id)

            # Build labels with correct alignment:
            # - Position 0 (BOS): -100 (ignore)
            # - Positions 1..M: values v1..vM
            # Length = M+1 to match decoder_input = tgt_ids[:, :-1]
            labels = [-100]  # Ignore BOS position
            for qk in query_keys:
                labels.append(kv_map[qk])

            # Pad sequences
            src_len = len(src_ids)
            tgt_len = len(tgt_ids)
            labels_len = len(labels)

            # Pad source to max source length
            max_src_len = 1 + (self.config.num_pairs * 3) + 1  # BOS + pairs + EOS
            if src_len < max_src_len:
                src_ids.extend([self.config.pad_token_id] * (max_src_len - src_len))

            # Pad target to include EOS (trainer will truncate)
            max_tgt_len = 1 + self.config.num_queries + 1  # BOS + queries + EOS
            if tgt_len < max_tgt_len:
                tgt_ids.extend([self.config.pad_token_id] * (max_tgt_len - tgt_len))

            # Labels length = tgt_len - 1 (no label for EOS position)
            max_labels_len = max_tgt_len - 1
            if labels_len < max_labels_len:
                labels.extend([-100] * (max_labels_len - labels_len))

            return {
                'src_ids': torch.tensor(src_ids, dtype=torch.long),
                'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'query_keys': torch.tensor(query_keys, dtype=torch.long),
                'expected_values': torch.tensor([kv_map[k] for k in query_keys], dtype=torch.long),
                'num_pairs': torch.tensor(self.config.num_pairs, dtype=torch.long),
                'num_queries': torch.tensor(len(query_keys), dtype=torch.long),
            }

        else:
            # ===== DECODER-ONLY MODE (for Pure Mamba baseline) =====
            # Build input sequence: [BOS key1 : val1 key2 : val2 ... QUERY k1 k2 ... EOS]
            input_ids = []

            # Start with BOS token
            input_ids.append(self.config.bos_token_id)

            # Add key-value pairs with separators: key : value
            for k, v in zip(keys, values):
                input_ids.append(k)
                input_ids.append(self.config.kv_sep_token_id)  # :
                input_ids.append(v)

            # Add QUERY marker
            query_marker_position = len(input_ids)
            input_ids.append(self.config.query_token_id)

            # Add queries
            query_positions = []
            for qk in query_keys:
                query_positions.append(len(input_ids))
                input_ids.append(qk)

            # Add EOS
            input_ids.append(self.config.eos_token_id)

            # Build labels: -100 everywhere except answer positions (after query tokens)
            # Using -100 (ignore_index) for positions not contributing to loss
            labels = [-100] * len(input_ids)
            for pos, qk in zip(query_positions, query_keys):
                labels[pos] = kv_map[qk]  # The expected value

            # Pad to seq_length if specified
            seq_length = self.config.seq_length
            if len(input_ids) < seq_length:
                padding_len = seq_length - len(input_ids)
                input_ids.extend([self.config.pad_token_id] * padding_len)
                labels.extend([-100] * padding_len)
            elif len(input_ids) > seq_length:
                # Truncate (should not happen with proper config)
                input_ids = input_ids[:seq_length]
                labels = labels[:seq_length]

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'query_positions': torch.tensor(query_positions, dtype=torch.long),
                'query_keys': torch.tensor(query_keys, dtype=torch.long),
                'expected_values': torch.tensor([kv_map[k] for k in query_keys], dtype=torch.long),
                'num_pairs': torch.tensor(self.config.num_pairs, dtype=torch.long),
                'num_queries': torch.tensor(len(query_keys), dtype=torch.long),
            }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._samples[idx]


class MQARCurriculumGenerator:
    """
    Curriculum generator for MQAR task.

    Starts with easy examples (few pairs) and gradually increases
    difficulty (more pairs) to test state capacity limits.

    Curriculum stages: num_pairs in [16, 32, 64, 128, 256, 512]
    """

    def __init__(
        self,
        stages: List[int] = None,
        num_pairs_range: List[int] = None,
        d_state: int = 64,
        samples_per_stage: int = 5000,
        base_config: Optional[MQARConfig] = None,
        vocab_size: int = 8192,
        seq_lengths: Optional[List[int]] = None,
    ):
        """
        Args:
            stages: List of num_pairs values for curriculum (deprecated, use num_pairs_range)
            num_pairs_range: List of num_pairs values for curriculum
            d_state: State dimension for bottleneck testing
            samples_per_stage: Samples to generate per stage
            base_config: Base MQAR configuration
            vocab_size: Vocabulary size for MQAR
            seq_lengths: List of sequence lengths for state capacity sweep
        """
        # Support both 'stages' and 'num_pairs_range' for compatibility
        self.num_pairs_range = num_pairs_range or stages or [16, 32, 64, 128, 256, 512]
        self.stages = self.num_pairs_range  # Backward compat
        self.d_state = d_state
        self.samples_per_stage = samples_per_stage
        self.vocab_size = vocab_size
        self.seq_lengths = seq_lengths or [256, 512, 1024]
        self.base_config = base_config or MQARConfig(d_state=d_state, vocab_size=vocab_size)

    def get_dataset(self, stage_idx: int, split: str = "train") -> MQARDataset:
        """
        Get dataset for a specific curriculum stage.

        Args:
            stage_idx: Index into stages list
            split: "train", "validation", or "test"

        Returns:
            MQARDataset configured for the stage
        """
        if stage_idx >= len(self.num_pairs_range):
            stage_idx = len(self.num_pairs_range) - 1

        num_pairs = self.num_pairs_range[stage_idx]

        config = MQARConfig(
            d_state=self.d_state,
            num_pairs=num_pairs,
            num_queries=min(16, max(4, num_pairs // 4)),  # Scale queries with pairs
            vocab_size=self.base_config.vocab_size,
            seq_length=self.base_config.seq_length,
            key_token_start=self.base_config.key_token_start,
            key_token_end=self.base_config.key_token_end,
            value_token_start=self.base_config.value_token_start,
            value_token_end=self.base_config.value_token_end,
            seed=self.base_config.seed,
        )

        return MQARDataset(
            config=config,
            num_samples=self.samples_per_stage,
            split=split,
        )

    def get_all_stages(self, split: str = "train") -> List[MQARDataset]:
        """Get datasets for all curriculum stages."""
        return [self.get_dataset(i, split) for i in range(len(self.num_pairs_range))]

    def generate_stage(
        self,
        num_pairs: int,
        seq_length: int = 512,
        num_samples: int = 5000,
        seed: Optional[int] = None,
    ) -> MQARDataset:
        """
        Generate a single curriculum stage.

        Args:
            num_pairs: Number of key-value pairs
            seq_length: Sequence length
            num_samples: Number of samples
            seed: Random seed

        Returns:
            MQARDataset for this stage
        """
        config = MQARConfig(
            d_state=self.d_state,
            num_pairs=num_pairs,
            num_queries=min(16, max(4, num_pairs // 4)),
            vocab_size=self.vocab_size,
            seq_length=seq_length,
            seed=seed,
        )

        return MQARDataset(
            config=config,
            num_samples=num_samples,
            seed=seed,
        )

    def generate_full_curriculum(
        self,
        seq_length: int = 512,
        seed: Optional[int] = None,
    ) -> Dict[int, MQARDataset]:
        """
        Generate datasets for all curriculum stages.

        Args:
            seq_length: Sequence length for all stages
            seed: Base random seed

        Returns:
            Dictionary mapping num_pairs -> MQARDataset
        """
        stages = {}
        for num_pairs in self.num_pairs_range:
            stage_seed = seed + num_pairs if seed else None
            stages[num_pairs] = self.generate_stage(
                num_pairs=num_pairs,
                seq_length=seq_length,
                num_samples=self.samples_per_stage,
                seed=stage_seed,
            )
        return stages

    def generate_state_capacity_sweep(
        self,
        num_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int, MQARDataset]]:
        """
        Generate sweep across num_pairs and seq_lengths for state capacity analysis.

        Skips configurations where num_pairs * 4 > seq_length (not enough space).

        Args:
            num_samples: Samples per configuration
            seed: Base random seed

        Returns:
            List of (num_pairs, seq_length, dataset) tuples
        """
        sweep = []
        for seq_length in self.seq_lengths:
            for num_pairs in self.num_pairs_range:
                # Skip if sequence can't fit the pairs
                # Each pair takes ~3 tokens (key, sep, value) + query section
                if num_pairs * 4 > seq_length:
                    continue

                config = MQARConfig(
                    d_state=self.d_state,
                    num_pairs=num_pairs,
                    num_queries=min(16, max(4, num_pairs // 4)),
                    vocab_size=self.vocab_size,
                    seq_length=seq_length,
                    seed=seed,
                )

                dataset = MQARDataset(
                    config=config,
                    num_samples=num_samples,
                    seed=seed,
                )

                sweep.append((num_pairs, seq_length, dataset))

        return sweep


def compute_mqar_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    label_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute MQAR accuracy metrics.

    Args:
        predictions: Predicted token IDs (batch, seq_len) or (batch, seq_len, vocab_size)
        labels: Ground truth labels (batch, seq_len)
        label_mask: Mask indicating positions to score (1 = score, 0 = ignore)

    Returns:
        Dict with:
        - token_accuracy: Accuracy at masked positions
        - sample_accuracy: Fraction of samples with all positions correct
    """
    # If predictions are logits, convert to IDs
    if predictions.dim() == 3:
        predictions = predictions.argmax(dim=-1)

    # Use provided mask or infer from non-PAD positions
    if label_mask is not None:
        valid_mask = label_mask.bool()
    else:
        # Fall back to non-PAD labels (but prefer explicit mask)
        valid_mask = labels != 0

    if valid_mask.sum() == 0:
        return {'token_accuracy': 0.0, 'sample_accuracy': 0.0}

    # Token-level accuracy
    correct = (predictions == labels) & valid_mask
    token_accuracy = correct.sum().float() / valid_mask.sum().float()

    # Sample-level accuracy (all masked positions correct)
    # Count correct positions per sample
    correct_per_sample = correct.sum(dim=-1)
    total_per_sample = valid_mask.sum(dim=-1)

    # Avoid division by zero for samples with no masked positions
    has_labels = total_per_sample > 0
    perfect_samples = torch.zeros_like(total_per_sample, dtype=torch.float)
    if has_labels.any():
        perfect_samples[has_labels] = (
            correct_per_sample[has_labels] == total_per_sample[has_labels]
        ).float()
    sample_accuracy = perfect_samples.mean() if has_labels.any() else 0.0

    return {
        'token_accuracy': token_accuracy.item(),
        'sample_accuracy': sample_accuracy.item() if isinstance(sample_accuracy, torch.Tensor) else sample_accuracy,
    }


# =============================================================================
# NMT Datasets
# =============================================================================

class DocumentNMTDataset(Dataset):
    """
    Base class for document-level NMT datasets.

    Can be used directly with src_texts/tgt_texts for testing,
    or subclassed with _load_data() for real datasets.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer: Any = None,
        augmenter: Optional[ConcatenationAugmenter] = None,
        max_src_length: int = 512,
        max_tgt_length: int = 512,
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
    ):
        """
        Args:
            split: "train", "validation", or "test"
            tokenizer: Tokenizer instance
            augmenter: Optional augmenter for training
            max_src_length: Maximum source sequence length
            max_tgt_length: Maximum target sequence length
            src_texts: Optional source texts (for testing/in-memory datasets)
            tgt_texts: Optional target texts (for testing/in-memory datasets)
        """
        self.split = split
        self.tokenizer = tokenizer
        self.augmenter = augmenter
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self._epoch = 0

        # If texts provided directly, use them
        if src_texts is not None and tgt_texts is not None:
            self._samples = [
                DocumentSample(
                    src_sentences=[src],
                    tgt_sentences=[tgt],
                    doc_id=f"doc_{i}",
                )
                for i, (src, tgt) in enumerate(zip(src_texts, tgt_texts))
            ]
        else:
            # Load from subclass implementation
            self._samples = self._load_data()

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible augmentation."""
        self._epoch = epoch
        if self.augmenter:
            self.augmenter.set_epoch(epoch)

    def _load_data(self) -> List[DocumentSample]:
        """Load dataset. Override in subclasses."""
        return []  # Return empty list by default for direct instantiation

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        if self.augmenter is not None and self.split == "train":
            # Apply augmentation
            src_text, tgt_text = self.augmenter(self._samples, idx)
        else:
            # No augmentation - use first sentence
            sample = self._samples[idx]
            if len(sample) > 0:
                src_text, tgt_text = sample.get_sentence_pair(0)
            else:
                src_text, tgt_text = "", ""

        # Tokenize
        if self.tokenizer is not None:
            src_ids, tgt_ids = self.tokenizer.encode_pair(
                src_text, tgt_text,
                max_src_length=self.max_src_length,
                max_tgt_length=self.max_tgt_length,
            )
        else:
            # Return raw text if no tokenizer
            return {'src_text': src_text, 'tgt_text': tgt_text}

        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
        }


class IWSLT14Dataset(DocumentNMTDataset):
    """
    IWSLT14 De-En dataset.

    Standard NMT benchmark dataset. Sentences are treated as
    individual documents (no multi-sentence documents).

    Uses HuggingFace datasets for loading.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer: Any = None,
        augmenter: Optional[ConcatenationAugmenter] = None,
        max_src_length: int = 512,
        max_tgt_length: int = 512,
        src_lang: str = "de",
        tgt_lang: str = "en",
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            augmenter=augmenter,
            max_src_length=max_src_length,
            max_tgt_length=max_tgt_length,
        )

    def _load_data(self) -> List[DocumentSample]:
        """Load IWSLT14 dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        # Map split names
        hf_split = self.split
        if self.split == "validation":
            hf_split = "validation"

        # Load dataset
        dataset = load_dataset(
            "iwslt2017",
            f"iwslt2017-{self.src_lang}-{self.tgt_lang}",
            split=hf_split,
            trust_remote_code=True,
        )

        # Convert to DocumentSample (each sentence is a document)
        samples = []
        for item in dataset:
            translation = item['translation']
            samples.append(DocumentSample(
                src_sentences=[translation[self.src_lang]],
                tgt_sentences=[translation[self.tgt_lang]],
            ))

        return samples


class OPUSBooksDataset(DocumentNMTDataset):
    """
    OPUS Books dataset with document structure.

    This dataset preserves document (book chapter) boundaries,
    making it ideal for document-level NMT research.

    Uses HuggingFace datasets for loading.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer: Any = None,
        augmenter: Optional[ConcatenationAugmenter] = None,
        max_src_length: int = 512,
        max_tgt_length: int = 512,
        src_lang: str = "de",
        tgt_lang: str = "en",
        sentences_per_doc: int = 20,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sentences_per_doc = sentences_per_doc
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            augmenter=augmenter,
            max_src_length=max_src_length,
            max_tgt_length=max_tgt_length,
        )

    def _load_data(self) -> List[DocumentSample]:
        """Load OPUS Books dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        # Load dataset
        dataset = load_dataset(
            "opus_books",
            f"{self.src_lang}-{self.tgt_lang}",
            split="train",  # OPUS Books only has train split
            trust_remote_code=True,
        )

        # Split into train/val/test using hash
        all_samples = []
        for item in dataset:
            translation = item['translation']
            all_samples.append((
                translation[self.src_lang],
                translation[self.tgt_lang],
            ))

        # Group into pseudo-documents
        documents = []
        current_src = []
        current_tgt = []

        for src, tgt in all_samples:
            current_src.append(src)
            current_tgt.append(tgt)

            if len(current_src) >= self.sentences_per_doc:
                documents.append(DocumentSample(
                    src_sentences=current_src.copy(),
                    tgt_sentences=current_tgt.copy(),
                ))
                current_src = []
                current_tgt = []

        # Handle remaining sentences
        if current_src:
            documents.append(DocumentSample(
                src_sentences=current_src,
                tgt_sentences=current_tgt,
            ))

        # Split by hash
        train_docs = []
        val_docs = []
        test_docs = []

        for i, doc in enumerate(documents):
            bucket = get_split_hash(doc.src_sentences[0], 100)
            if bucket < 80:  # 80% train
                train_docs.append(doc)
            elif bucket < 90:  # 10% val
                val_docs.append(doc)
            else:  # 10% test
                test_docs.append(doc)

        if self.split == "train":
            return train_docs
        elif self.split == "validation":
            return val_docs
        else:
            return test_docs


class NewsCommentaryDataset(DocumentNMTDataset):
    """
    News Commentary dataset.

    News articles with natural document structure.
    Uses HuggingFace datasets for loading.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer: Any = None,
        augmenter: Optional[ConcatenationAugmenter] = None,
        max_src_length: int = 512,
        max_tgt_length: int = 512,
        src_lang: str = "de",
        tgt_lang: str = "en",
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            augmenter=augmenter,
            max_src_length=max_src_length,
            max_tgt_length=max_tgt_length,
        )

    def _load_data(self) -> List[DocumentSample]:
        """Load News Commentary dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        # Load dataset
        lang_pair = f"{self.src_lang}-{self.tgt_lang}"
        dataset = load_dataset(
            "news_commentary",
            lang_pair,
            split="train",  # Only has train split
            trust_remote_code=True,
        )

        # Convert and split
        samples = []
        for item in dataset:
            translation = item['translation']
            samples.append(DocumentSample(
                src_sentences=[translation[self.src_lang]],
                tgt_sentences=[translation[self.tgt_lang]],
            ))

        # Split by hash
        train_samples = []
        val_samples = []
        test_samples = []

        for sample in samples:
            bucket = get_split_hash(sample.src_sentences[0], 100)
            if bucket < 80:
                train_samples.append(sample)
            elif bucket < 90:
                val_samples.append(sample)
            else:
                test_samples.append(sample)

        if self.split == "train":
            return train_samples
        elif self.split == "validation":
            return val_samples
        else:
            return test_samples


class StreamingDocumentDataset(IterableDataset):
    """
    Streaming dataset for large document corpora.

    Uses lazy loading to handle datasets that don't fit in memory.
    Suitable for very large training runs.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Any = None,
        augmenter: Optional[ConcatenationAugmenter] = None,
        max_src_length: int = 512,
        max_tgt_length: int = 512,
        buffer_size: int = 10000,
    ):
        """
        Args:
            data_path: Path to data file (JSONL format)
            tokenizer: Tokenizer instance
            augmenter: Optional augmenter
            max_src_length: Maximum source length
            max_tgt_length: Maximum target length
            buffer_size: Size of shuffle buffer
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.augmenter = augmenter
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset."""
        import json

        buffer = []

        with open(self.data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                sample = DocumentSample(
                    src_sentences=[item['src']],
                    tgt_sentences=[item['tgt']],
                    doc_id=item.get('doc_id'),
                )

                buffer.append(sample)

                if len(buffer) >= self.buffer_size:
                    # Shuffle and yield
                    random.shuffle(buffer)
                    for s in buffer:
                        yield self._process_sample(s)
                    buffer = []

        # Yield remaining
        random.shuffle(buffer)
        for s in buffer:
            yield self._process_sample(s)

    def _process_sample(self, sample: DocumentSample) -> Dict[str, torch.Tensor]:
        """Process a single sample."""
        src_text = sample.src_sentences[0]
        tgt_text = sample.tgt_sentences[0]

        if self.tokenizer is not None:
            src_ids, tgt_ids = self.tokenizer.encode_pair(
                src_text, tgt_text,
                max_src_length=self.max_src_length,
                max_tgt_length=self.max_tgt_length,
            )
            return {'src_ids': src_ids, 'tgt_ids': tgt_ids}
        else:
            return {'src_text': src_text, 'tgt_text': tgt_text}


# =============================================================================
# Factory Function
# =============================================================================

def create_dataset(
    dataset_name: str = "iwslt14",
    split: str = "train",
    tokenizer: Any = None,
    augmenter: Optional[ConcatenationAugmenter] = None,
    max_src_length: int = 512,
    max_tgt_length: int = 512,
    **kwargs,
) -> Union[DocumentNMTDataset, MQARDataset]:
    """
    Factory function to create datasets.

    Args:
        dataset_name: "iwslt14", "opus_books", "news_commentary", or "mqar"
        split: "train", "validation", or "test"
        tokenizer: Tokenizer instance
        augmenter: Optional augmenter
        max_src_length: Maximum source length
        max_tgt_length: Maximum target length
        **kwargs: Additional dataset-specific arguments

    Returns:
        Dataset instance

    Recommendations:
    - "iwslt14": Standard NMT benchmark
    - "opus_books": Document-level experiments (has document structure)
    - "mqar": State capacity testing
    """
    if dataset_name == "iwslt14":
        return IWSLT14Dataset(
            split=split,
            tokenizer=tokenizer,
            augmenter=augmenter,
            max_src_length=max_src_length,
            max_tgt_length=max_tgt_length,
            **kwargs,
        )
    elif dataset_name == "opus_books":
        return OPUSBooksDataset(
            split=split,
            tokenizer=tokenizer,
            augmenter=augmenter,
            max_src_length=max_src_length,
            max_tgt_length=max_tgt_length,
            **kwargs,
        )
    elif dataset_name == "news_commentary":
        return NewsCommentaryDataset(
            split=split,
            tokenizer=tokenizer,
            augmenter=augmenter,
            max_src_length=max_src_length,
            max_tgt_length=max_tgt_length,
            **kwargs,
        )
    elif dataset_name == "mqar":
        config = kwargs.get('config', MQARConfig())
        num_samples = kwargs.get('num_samples', 10000)
        return MQARDataset(
            config=config,
            num_samples=num_samples,
            split=split,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
