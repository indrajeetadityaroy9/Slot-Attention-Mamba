#!/usr/bin/env python3
"""
Training script for Document-Level NMT with Hybrid Mamba-Attention.

Single GPU:
    python scripts/train.py                           # Default config
    python scripts/train.py model=medium              # Medium model (200M params)
    python scripts/train.py training.batch_size=32    # Custom batch size

Multi-GPU (DDP):
    torchrun --nproc_per_node=2 scripts/train.py      # Use both H100 GPUs
    torchrun --nproc_per_node=2 scripts/train.py training.distributed_strategy=ddp

Multi-GPU (FSDP for large models):
    torchrun --nproc_per_node=2 scripts/train.py training.distributed_strategy=fsdp

Hydra configuration from configs/ directory.
"""

import os
import sys
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import hydra
from omegaconf import DictConfig, OmegaConf

from models import ModelConfig, HybridMambaEncoderDecoder
from data import (
    CustomBPETokenizer,
    NMTTokenizer,
    create_tokenizer,
    ConcatenationAugmenter,
    IWSLT14Dataset,
    OPUSBooksDataset,
    create_dataset,
    create_collator,
)
from training import Trainer, TrainerConfig, setup_distributed


def setup_environment():
    """Setup environment for H100 training."""
    # Enable TF32 for faster matmul
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Set memory allocator for better efficiency
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # NCCL optimizations for NVLink
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")  # Use NVLink for P2P


def create_model(cfg: DictConfig, device: str, dtype: torch.dtype) -> HybridMambaEncoderDecoder:
    """Create model from config."""
    # Handle custom_hybrid_positions from Hydra config (critical for ablations)
    custom_positions = cfg.model.get("custom_hybrid_positions")
    if custom_positions is not None:
        custom_positions = list(custom_positions)  # Convert OmegaConf list to Python list

    model_cfg = ModelConfig(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        encoder_layers=cfg.model.encoder_layers,
        decoder_layers=cfg.model.decoder_layers,
        d_state=cfg.model.d_state,
        n_heads=cfg.model.n_heads,
        attention_ratio=cfg.model.attention_ratio,
        cross_attn_every=cfg.model.cross_attn_every,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
        custom_hybrid_positions=custom_positions,  # FIX: Enable ablation experiments
    )

    model = HybridMambaEncoderDecoder(
        config=model_cfg,
        device=device,
        dtype=dtype,
    )

    print(f"Created model with {model.num_parameters() / 1e6:.1f}M parameters")
    print(f"Encoder: {model.encoder.get_layer_counts()}")
    print(f"Decoder: {model.decoder.get_layer_counts()}")

    return model


def worker_init_fn(worker_id: int):
    """
    Initialize worker with unique seed for reproducibility.

    Each worker gets a unique seed derived from the base seed + worker_id.
    This ensures reproducible data loading across runs.
    """
    # Get current state and derive unique seed per worker
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(cfg: DictConfig, tokenizer, dist_info: dict):
    """Create training and validation dataloaders with distributed support."""
    # Get dataset name from config (default: opus_books for document-level)
    dataset_name = cfg.data.get("dataset_name", "opus_books")
    dataset_type = cfg.data.get("dataset_type", "nmt")

    # Handle MQAR synthetic dataset specially
    if dataset_type == "synthetic" or dataset_name == "mqar":
        from data import MQARDataset, MQARConfig

        # Build MQARConfig from Hydra config
        mqar_cfg = cfg.data.get("mqar", {})
        mqar_config = MQARConfig(
            d_state=mqar_cfg.get("d_state", 64),
            num_pairs=mqar_cfg.get("num_pairs", 64),
            num_queries=mqar_cfg.get("num_queries", 16),
            vocab_size=mqar_cfg.get("vocab_size", 8192),
            seq_length=mqar_cfg.get("seq_length", 512),
            pad_token_id=mqar_cfg.get("pad_token_id", 0),
            bos_token_id=mqar_cfg.get("bos_token_id", 1),
            eos_token_id=mqar_cfg.get("eos_token_id", 2),
            kv_sep_token_id=mqar_cfg.get("kv_sep_token_id", 3),
            query_token_id=mqar_cfg.get("query_token_id", 4),
            key_token_start=mqar_cfg.get("key_token_start", 10),
            key_token_end=mqar_cfg.get("key_token_end", 4096),
            value_token_start=mqar_cfg.get("value_token_start", 4096),
            value_token_end=mqar_cfg.get("value_token_end", 8192),
        )

        # MQAR mode: "decoder_only" for Pure Mamba, "seq2seq" for Hybrid
        # seq2seq mode enables cross-attention retrieval for state capacity test
        mqar_mode = mqar_cfg.get("mode", "decoder_only")

        # Create MQAR datasets with mode
        curriculum_cfg = cfg.data.get("curriculum", {})
        num_samples = curriculum_cfg.get("samples_per_stage", 10000)

        train_dataset = MQARDataset(
            config=mqar_config,
            num_samples=num_samples,
            split="train",
            mode=mqar_mode,
        )
        val_dataset = MQARDataset(
            config=mqar_config,
            num_samples=num_samples // 10,  # Smaller validation set
            split="validation",
            mode=mqar_mode,
        )

        # MQAR collator with matching mode
        from data import MQARCollator
        collator = MQARCollator(pad_token_id=mqar_config.pad_token_id, mode=mqar_mode)

        if dist_info.get("is_main", True):
            print(f"MQAR mode: {mqar_mode}")
            if mqar_mode == "seq2seq":
                print("  -> Encoder-Decoder with cross-attention (tests NC1 retrieval)")
            else:
                print("  -> Decoder-only (tests TC0 state capacity)")

    else:
        # Standard NMT datasets
        # Create augmenter for training
        augmenter = ConcatenationAugmenter(
            n_sentences=cfg.data.cat_n,
            p_concat=cfg.data.p_concat,
        )

        # Training dataset - use factory function for proper dataset selection
        train_dataset = create_dataset(
            dataset_name=dataset_name,
            split="train",
            tokenizer=tokenizer,
            augmenter=augmenter,
            max_src_length=cfg.data.max_src_length,
            max_tgt_length=cfg.data.max_tgt_length,
        )

        # Validation dataset (no augmentation)
        val_dataset = create_dataset(
            dataset_name=dataset_name,
            split="validation",
            tokenizer=tokenizer,
            augmenter=None,
            max_src_length=cfg.data.max_src_length,
            max_tgt_length=cfg.data.max_tgt_length,
        )

        # Create collator
        collator = create_collator(
            mode=cfg.data.collator_mode,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Create samplers for distributed training
    world_size = dist_info.get("world_size", 1)
    rank = dist_info.get("rank", 0)

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        shuffle_train = False  # Sampler handles shuffling
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    # Create dataloaders with optimized settings
    num_workers = cfg.data.get("num_workers", 8)
    pin_memory = cfg.data.get("pin_memory", True) and num_workers > 0
    persistent_workers = cfg.data.get("persistent_workers", True) and num_workers > 0

    loader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": cfg.data.get("drop_last", False),
        "worker_init_fn": worker_init_fn if num_workers > 0 else None,
    }

    if num_workers > 0 and cfg.data.get("prefetch_factor"):
        loader_kwargs["prefetch_factor"] = cfg.data.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=shuffle_train if train_sampler is None else False,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=False,
        **loader_kwargs,
    )

    if dist_info.get("is_main", True):
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        if world_size > 1:
            print(f"Distributed: {world_size} GPUs, {len(train_dataset) // world_size} samples/GPU")

    return train_loader, val_loader


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main training function with distributed support."""
    # Setup environment first
    setup_environment()

    # Setup distributed training (handles both single and multi-GPU)
    dist_info = setup_distributed()
    is_main = dist_info.get("is_main", True)

    if is_main:
        print("=" * 60)
        print("Document-Level NMT with Hybrid Mamba-Attention")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))

    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")

    device = dist_info["device"]
    dtype = torch.bfloat16 if cfg.training.use_bf16 else torch.float32

    if is_main:
        print(f"\nDevice: {torch.cuda.get_device_name(device.index if device.index else 0)}")
        print(f"Dtype: {dtype}")
        print(f"World Size: {dist_info['world_size']}")

    # Check if MQAR synthetic task (no tokenizer needed)
    dataset_type = cfg.data.get("dataset_type", "nmt")
    is_mqar = dataset_type == "synthetic" or cfg.data.get("dataset_name") == "mqar"

    if is_mqar:
        # MQAR uses its own vocabulary (8192 tokens)
        if is_main:
            print("\nMQAR synthetic task - using internal vocabulary")
        mqar_vocab_size = cfg.data.get("mqar", {}).get("vocab_size", 8192)
        cfg.model.vocab_size = mqar_vocab_size
        tokenizer = None  # MQAR doesn't need tokenizer
        if is_main:
            print(f"MQAR vocab size: {mqar_vocab_size}")
    else:
        # Create tokenizer for NMT tasks
        if is_main:
            print("\nLoading tokenizer...")
        tokenizer_type = cfg.data.get("tokenizer_type", "custom")
        tokenizer_path = cfg.data.get("tokenizer_path", "data/tokenizer/tokenizer.json")

        if tokenizer_type == "custom":
            # RECOMMENDED: 32K BPE tokenizer for proper parameter allocation
            tokenizer = create_tokenizer(
                tokenizer_type="custom",
                tokenizer_path=tokenizer_path,
                max_length=cfg.data.max_src_length,
            )
            if is_main:
                print(f"Using Custom 32K BPE tokenizer (RECOMMENDED)")
        else:
            # NOT RECOMMENDED: mBART 250K vocab makes model 95% embedding table
            import warnings
            if is_main:
                warnings.warn(
                    "Using mBART tokenizer (250K vocab). "
                    "This makes the model 95% embedding table. "
                    "Use tokenizer_type='custom' for thesis work."
                )
            tokenizer = NMTTokenizer(
                src_lang=cfg.data.src_lang,
                tgt_lang=cfg.data.tgt_lang,
            )
        if is_main:
            print(f"Vocab size: {tokenizer.vocab_size}")

        # Override model vocab size from tokenizer
        cfg.model.vocab_size = tokenizer.vocab_size

    # Create model
    if is_main:
        print("\nCreating model...")
    model = create_model(cfg, str(device), dtype)

    # Create dataloaders with distributed support
    if is_main:
        print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(cfg, tokenizer, dist_info)

    # Create trainer config with distributed settings
    trainer_config = TrainerConfig(
        max_steps=cfg.training.max_steps,
        batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        scheduler_type=cfg.training.scheduler_type,
        min_lr=cfg.training.min_lr,
        use_bf16=cfg.training.use_bf16,
        use_compile=cfg.training.use_compile,
        compile_mode=cfg.training.compile_mode,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        output_dir=cfg.training.output_dir,
        log_steps=cfg.training.log_steps,
        eval_steps=cfg.training.eval_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        channels_last=cfg.training.get("channels_last", True),
        # Distributed settings
        distributed_strategy=cfg.training.get("distributed_strategy", "ddp"),
        find_unused_parameters=cfg.training.get("find_unused_parameters", False),
        static_graph=cfg.training.get("static_graph", True),
        fsdp_sharding=cfg.training.get("fsdp_sharding", "full_shard"),
        fsdp_cpu_offload=cfg.training.get("fsdp_cpu_offload", False),
    )

    # Create trainer
    if is_main:
        print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        config=trainer_config,
        eval_dataloader=val_loader,
    )

    # Resume from checkpoint if specified
    if cfg.training.get("resume_from"):
        trainer.load_checkpoint(cfg.training.resume_from)

    # Train
    if is_main:
        print("\nStarting training...")
    trainer.train()

    if is_main:
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
