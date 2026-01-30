#!/usr/bin/env python3
"""Training script for Hybrid Mamba-Attention State Capacity experiments."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import hydra
from omegaconf import DictConfig, OmegaConf

from models.encoder_decoder import ModelConfig, HybridMambaEncoderDecoder
from data.mqar import MQARDataset, MQARConfig
from data.collator import MQARCollator
from training.trainer import NMTTrainer, setup_distributed, logger
from constants import PAD_TOKEN_ID, MQAR_VOCAB_SIZE


def create_model(cfg: DictConfig, device: str, dtype: torch.dtype) -> HybridMambaEncoderDecoder:
    """Instantiate model from Hydra config.

    Adaptive parameters derived at runtime:
    - hybrid_positions: From capacity theorem when num_pairs provided
    - dropout: From capacity/data ratio when num_samples provided
    """
    hybrid_positions = cfg.model.get("hybrid_positions")
    if hybrid_positions is not None:
        hybrid_positions = list(hybrid_positions)

    # Extract data statistics for adaptive computation
    num_pairs = cfg.data.get("num_pairs")  # For adaptive hybrid positions
    num_samples = cfg.data.get("num_samples", 100000)  # For adaptive dropout

    model_cfg = ModelConfig(
        vocab_size=cfg.model.get("vocab_size", MQAR_VOCAB_SIZE),
        d_model=cfg.model.d_model,
        encoder_layers=cfg.model.encoder_layers,
        decoder_layers=cfg.model.decoder_layers,
        d_state=cfg.model.d_state,
        n_heads=cfg.model.n_heads,
        hybrid_positions=hybrid_positions,
        num_pairs=num_pairs,  # For adaptive hybrid position computation
        num_samples=num_samples,  # For adaptive dropout computation
    )

    model = HybridMambaEncoderDecoder(config=model_cfg, device=device, dtype=dtype)

    # Log model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {num_params / 1e6:.1f}M params")
    logger.info(f"Encoder: {model_cfg.encoder_layers} layers, {len(model.encoder.attention_positions)} attention")
    logger.info(f"Decoder: {model_cfg.decoder_layers} layers, {len(model.decoder.hybrid_positions)} cross-attn")
    logger.info(f"Hybrid positions: {sorted(model.decoder.hybrid_positions)}")

    return model


def worker_init_fn(worker_id: int):
    """Derive unique per-worker seed for reproducible data loading."""
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(cfg: DictConfig, world_size: int, rank: int):
    """Create train/val dataloaders with distributed sampling if multi-GPU."""
    mqar_config = MQARConfig(
        num_pairs=cfg.data.get("num_pairs", 64),
        num_queries=cfg.data.get("num_queries", 16),
    )

    mqar_mode = cfg.data.get("mode", "seq2seq")
    num_samples = cfg.data.get("num_samples", 10000)

    train_dataset = MQARDataset(config=mqar_config, num_samples=num_samples, split="train", mode=mqar_mode)
    val_dataset = MQARDataset(config=mqar_config, num_samples=num_samples // 10, split="validation", mode=mqar_mode)
    collator = MQARCollator(pad_token_id=PAD_TOKEN_ID, mode=mqar_mode)

    logger.info(f"MQAR: num_pairs={mqar_config.num_pairs}, mode={mqar_mode}")

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    num_workers = cfg.data.get("num_workers", 8)

    loader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": num_workers > 0,
        "persistent_workers": num_workers > 0,
        "drop_last": False,
        "worker_init_fn": worker_init_fn if num_workers > 0 else None,
    }

    train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=shuffle_train if train_sampler is None else False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, **loader_kwargs)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    return train_loader, val_loader


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Entry point: setup, model creation, training loop."""
    dist_info = setup_distributed()

    logger.info("=" * 60)
    logger.info("Align-Mamba: State Capacity Experiments")
    logger.info("=" * 60)
    logger.info(OmegaConf.to_yaml(cfg))

    device = dist_info["device"]
    dtype = torch.bfloat16

    if torch.cuda.is_available():
        logger.info(f"Device: {torch.cuda.get_device_name(device.index or 0)}")
    logger.info(f"Dtype: {dtype}, World Size: {dist_info['world_size']}")

    model = create_model(cfg, str(device), dtype)
    train_loader, val_loader = create_dataloaders(cfg, dist_info["world_size"], dist_info["rank"])

    # Pass num_samples for adaptive logging intervals computation
    num_samples = cfg.data.get("num_samples", 100000)

    trainer = NMTTrainer(
        model=model,
        train_dataloader=train_loader,
        seed=cfg.project.get("seed", 42),
        max_steps=cfg.training.max_steps,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        label_smoothing=cfg.training.get("label_smoothing"),
        output_dir=cfg.training.output_dir,
        eval_dataloader=val_loader,
        dist_info=dist_info,
        num_samples=num_samples,  # For adaptive logging intervals
    )

    if cfg.training.get("resume_from"):
        trainer.load_checkpoint(cfg.training.resume_from)

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
