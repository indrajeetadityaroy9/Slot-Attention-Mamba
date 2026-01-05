"""
Trainer for Document-Level NMT.

H100-optimized training with:
- BF16 mixed precision (native, no scaling)
- torch.compile for kernel fusion
- Gradient checkpointing for long sequences
- Efficient logging and checkpointing
- Multi-GPU support (DDP/FSDP)
- NVLink-optimized NCCL backend
"""

import os
import time
import random
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast

from .objectives import create_loss_fn, create_scheduler
from .distributed import (
    DistributedConfig,
    setup_distributed,
    cleanup_distributed,
    wrap_model_distributed,
    print_distributed_info,
    all_reduce_mean,
    barrier,
)
from .hardware import (
    detect_hardware,
    print_hardware_info,
    setup_h100_optimizations,
    setup_nccl_optimizations,
    CUDAMemoryManager,
    get_optimal_worker_count,
    print_h100_optimization_status,
)


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""

    # Reproducibility (NeurIPS/ICML standard)
    seed: int = 42
    deterministic: bool = False  # Enable for exact reproducibility (may reduce performance)

    # Training
    max_steps: int = 100000
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-8

    # Scheduler
    warmup_steps: int = 4000
    scheduler_type: str = "cosine"
    min_lr: float = 1e-6

    # Mixed precision
    use_bf16: bool = True
    use_compile: bool = True
    compile_mode: str = "max-autotune"

    # Checkpointing
    save_steps: int = 5000
    save_total_limit: int = 3
    output_dir: str = "outputs"

    # Logging
    log_steps: int = 100
    eval_steps: int = 1000

    # Memory
    gradient_checkpointing: bool = True

    # H100 optimizations
    tf32_matmul: bool = True
    cudnn_benchmark: bool = True
    channels_last: bool = True  # Use channels-last memory format
    matmul_precision: str = "high"  # 'high' = BF16 accumulation for H100

    # Fused optimizer (critical for H100 performance)
    use_fused_optimizer: bool = True

    # DataLoader optimizations for high-CPU systems
    dataloader_num_workers: int = 16
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 4
    bucket_cap_mb: int = 512  # H100 NVLink optimized (larger buckets = fewer NCCL calls)

    # Distributed training
    distributed_strategy: str = "ddp"  # "none", "ddp", "fsdp", "fsdp_full"
    find_unused_parameters: bool = False
    static_graph: bool = True
    fsdp_sharding: str = "full_shard"
    fsdp_cpu_offload: bool = False


class Trainer:
    """
    H100-optimized trainer for Document-Level NMT.

    Features:
    - BF16 training (native H100 support)
    - torch.compile for kernel optimization
    - Gradient checkpointing for 8K sequences
    - Efficient checkpoint management
    - Multi-GPU DDP/FSDP support
    - NVLink-optimized communication
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: Optional[TrainerConfig] = None,
        eval_dataloader: Optional[DataLoader] = None,
        eval_fn: Optional[Callable] = None,
    ):
        """
        Args:
            model: Model to train
            train_dataloader: Training data loader
            config: Trainer configuration
            eval_dataloader: Optional evaluation data loader
            eval_fn: Optional evaluation function
        """
        self.config = config or TrainerConfig()
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_fn = eval_fn

        # Setup distributed training
        self.dist_info = setup_distributed()
        self.device = self.dist_info["device"]
        self.is_main = self.dist_info["is_main"]
        self.world_size = self.dist_info["world_size"]

        # Print distributed info (main process only)
        print_distributed_info(self.dist_info)

        # Setup reproducibility (seeds)
        self._setup_reproducibility()

        # Apply H100 optimizations
        self._setup_h100_optimizations()

        # Enable gradient checkpointing before wrapping
        if self.config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Wrap model for distributed training
        if self.world_size > 1:
            dist_config = DistributedConfig(
                strategy=self.config.distributed_strategy,
                find_unused_parameters=self.config.find_unused_parameters,
                static_graph=self.config.static_graph,
                sharding_strategy=self.config.fsdp_sharding,
                cpu_offload=self.config.fsdp_cpu_offload,
            )
            self.model = wrap_model_distributed(
                self.model,
                dist_config,
                self.device,
                use_bf16=self.config.use_bf16,
            )
        else:
            self.model = self.model.to(self.device)

        # Compile model (after DDP wrapping for compatibility)
        if self.config.use_compile:
            self.model = torch.compile(self.model, mode=self.config.compile_mode)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = create_scheduler(
            scheduler_type=self.config.scheduler_type,
            optimizer=self.optimizer,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            min_lr=self.config.min_lr,
        )

        # Setup loss function
        self.loss_fn = create_loss_fn(
            loss_type="label_smoothing",
            smoothing=0.1,
            ignore_index=-100,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")

        # Output directory (main process only creates)
        self.output_dir = Path(self.config.output_dir)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        barrier()  # Wait for directory creation

    def _setup_h100_optimizations(self):
        """Setup H100-specific optimizations using hardware detection."""
        # Detect hardware and print info (main process only)
        if self.is_main:
            self.hardware_info = detect_hardware()
            print_hardware_info(self.hardware_info)
        else:
            self.hardware_info = detect_hardware()

        # Apply H100/Ampere optimizations
        has_ampere = any(
            gpu.compute_capability >= (8, 0) for gpu in self.hardware_info.gpus
        ) if self.hardware_info.gpus else False

        setup_h100_optimizations(
            enable_tf32=self.config.tf32_matmul and has_ampere,
            enable_cudnn_benchmark=self.config.cudnn_benchmark,
            enable_flash_sdp=has_ampere,
            enable_mem_efficient_sdp=has_ampere,
            cuda_alloc_conf="expandable_segments:True",
        )

        # Setup NCCL optimizations for multi-GPU
        if self.world_size > 1:
            setup_nccl_optimizations(
                nvlink_available=self.hardware_info.nvlink_available,
                use_infiniband=False,
            )

        # Print H100 optimization status (main process only)
        if self.is_main:
            print_h100_optimization_status()

    def _setup_reproducibility(self):
        """
        Set seeds for reproducibility per NeurIPS/ICML guidelines.

        This ensures deterministic behavior across runs when using the same seed.
        Note: Full determinism may slightly reduce performance.
        """
        import random
        import numpy as np

        seed = self.config.seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Enable full determinism if requested (may reduce performance by 10-20%)
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            if self.is_main:
                print("Deterministic mode enabled (may reduce performance)")

        if self.is_main:
            print(f"Set random seed: {seed}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with weight decay.

        Uses fused=True for H100 optimization:
        - Runs entire optimizer step on GPU without CPU round-trips
        - ~5-10% training speedup on H100
        """
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
        params_with_wd = []
        params_without_wd = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in no_decay):
                    params_without_wd.append(param)
                else:
                    params_with_wd.append(param)

        optimizer_groups = [
            {"params": params_with_wd, "weight_decay": self.config.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

        # Use fused optimizer for H100 (requires CUDA and PyTorch 2.0+)
        use_fused = (
            self.config.use_fused_optimizer
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        )

        if use_fused:
            if self.is_main:
                print("Using fused AdamW optimizer (H100 optimized)")

        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
            fused=use_fused,  # CRITICAL: Fused kernels for H100
        )

    @classmethod
    def create_dataloader(
        cls,
        dataset,
        batch_size: int,
        is_train: bool = True,
        world_size: int = 1,
        rank: int = 0,
        collate_fn=None,
        drop_last: bool = True,
    ) -> DataLoader:
        """
        Create an optimized DataLoader for H100 training.

        Uses psutil-based worker calculation for optimal CPU utilization
        on high-CPU systems (e.g., 52 vCPUs).

        Args:
            dataset: Dataset to load from
            batch_size: Batch size per device
            is_train: Whether this is for training (affects shuffling)
            world_size: Number of GPUs for distributed training
            rank: Current process rank
            collate_fn: Optional collation function
            drop_last: Whether to drop last incomplete batch

        Returns:
            Optimized DataLoader
        """
        # Calculate optimal workers using hardware detection
        num_workers = get_optimal_worker_count(world_size)

        # Setup distributed sampler if needed
        sampler = None
        shuffle = is_train
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
            )
            shuffle = False  # Sampler handles shuffling

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,  # H100 optimized
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
            prefetch_factor=4 if num_workers > 0 else None,  # Aggressive prefetching
            drop_last=drop_last,
        )

    def train(self):
        """Main training loop with distributed support."""
        self.model.train()

        data_iter = iter(self.train_dataloader)
        accumulated_loss = 0.0
        step_start_time = time.time()

        # Set epoch for distributed sampler if applicable
        if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.epoch)

        # Set epoch for dataset augmentation (reproducible CAT-N across runs)
        if hasattr(self.train_dataloader, 'dataset') and hasattr(self.train_dataloader.dataset, 'set_epoch'):
            self.train_dataloader.dataset.set_epoch(self.epoch)

        while self.global_step < self.config.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                # Update sampler epoch for proper shuffling
                if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(self.epoch)
                # Update dataset epoch for reproducible augmentation
                if hasattr(self.train_dataloader, 'dataset') and hasattr(self.train_dataloader.dataset, 'set_epoch'):
                    self.train_dataloader.dataset.set_epoch(self.epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Move to device
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass with autocast
            with autocast(dtype=torch.bfloat16, enabled=self.config.use_bf16):
                loss = self._training_step(batch)
                loss = loss / self.config.gradient_accumulation_steps

            # NaN/Inf loss guard (publication-grade stability)
            # Check BEFORE backward to avoid corrupting gradients
            if torch.isnan(loss) or torch.isinf(loss):
                if self.is_main:
                    print(f"WARNING: NaN/Inf loss detected at step {self.global_step}, skipping batch")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()

            # Optimizer step
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    # Check for NaN/Inf gradients (publication-grade stability)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        if self.is_main:
                            print(f"WARNING: NaN/Inf gradient detected at step {self.global_step}, skipping update")
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1

            # Logging (main process only)
            if self.global_step % self.config.log_steps == 0:
                # Average loss across processes
                if self.world_size > 1:
                    loss_tensor = torch.tensor(accumulated_loss, device=self.device)
                    accumulated_loss = all_reduce_mean(loss_tensor).item()

                if self.is_main:
                    elapsed = time.time() - step_start_time
                    steps_per_sec = self.config.log_steps / elapsed
                    samples_per_sec = steps_per_sec * self.config.batch_size * self.world_size
                    lr = self.scheduler.get_last_lr()[0]

                    # Get memory stats
                    mem_stats = CUDAMemoryManager.get_memory_stats()

                    print(
                        f"Step {self.global_step}/{self.config.max_steps} | "
                        f"Loss: {accumulated_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Steps/s: {steps_per_sec:.2f} | "
                        f"Samples/s: {samples_per_sec:.1f} | "
                        f"Mem: {mem_stats['allocated']:.1f}/{mem_stats['reserved']:.1f}GB"
                    )

                accumulated_loss = 0.0
                step_start_time = time.time()

            # Evaluation
            if self.global_step % self.config.eval_steps == 0 and self.eval_fn:
                self._evaluate()

            # Checkpointing (main process only)
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()

        if self.is_main:
            print(f"Training complete! Final step: {self.global_step}")

        # Cleanup distributed
        cleanup_distributed()

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step."""
        # Handle packed vs padded batches
        if "cu_seqlens_src" in batch:
            # Packed sequences - convert to padded for model forward
            # This hybrid approach eliminates storage/loading waste while keeping
            # the model's padded interface. For full O(L^2) attention savings,
            # the encoder/decoder would need native VarLen support.
            src_ids, tgt_ids, labels, src_mask, tgt_mask = self._unpack_batch(batch)

            # Forward pass with padded tensors
            # Input to decoder: tgt_ids[:, :-1] (all but last)
            # Labels come from LabelShiftCollator (already shifted with -100 at boundaries)
            decoder_input = tgt_ids[:, :-1]

            # Use pre-computed labels from collator, truncate to match decoder output
            # labels[i] = tgt_ids[i+1], with labels[-1] = -100 (from collator)
            # We need labels of shape (batch, tgt_len-1) to match decoder output
            labels_for_loss = labels[:, :-1].clone()

            # Apply mask to handle padding positions (in case collator didn't mask them)
            labels_for_loss[~tgt_mask[:, :-1]] = -100

            logits = self.model(src_ids, decoder_input)

            # Flatten and compute loss
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels_for_loss.reshape(-1)

            loss = self.loss_fn(logits_flat, labels_flat)
        elif "input_ids" in batch:
            # MQAR synthetic task (decoder-only, no encoder)
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            # For MQAR, we use decoder-only mode: pass None for encoder input
            # The model should handle this case for decoder-only architectures
            logits = self.model(None, input_ids)

            # Compute loss (labels already have -100 for non-query positions)
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)
            loss = self.loss_fn(logits_flat, labels_flat)
        else:
            # Padded sequences (standard NMT)
            src_ids = batch["src_ids"]
            tgt_ids = batch["tgt_ids"]
            labels = batch.get("labels", tgt_ids[:, 1:])

            # Forward pass
            logits = self.model(src_ids, tgt_ids[:, :-1])

            # Compute loss
            loss = self.loss_fn(logits, labels)

        return loss

    def _unpack_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple:
        """
        Convert packed batch to padded format.

        Args:
            batch: Packed batch with cu_seqlens

        Returns:
            Tuple of (src_ids, tgt_ids, labels, src_mask, tgt_mask)
        """
        cu_seqlens_src = batch["cu_seqlens_src"]
        cu_seqlens_tgt = batch["cu_seqlens_tgt"]
        max_src_len = batch["max_seqlen_src"]
        max_tgt_len = batch["max_seqlen_tgt"]
        batch_size = batch["batch_size"]

        src_flat = batch["src_ids"]
        tgt_flat = batch["tgt_ids"]
        labels_flat = batch.get("labels", tgt_flat)

        # Create padded tensors
        pad_id = 0  # Assuming pad_token_id = 0
        src_ids = torch.full(
            (batch_size, max_src_len), pad_id,
            dtype=src_flat.dtype, device=src_flat.device
        )
        tgt_ids = torch.full(
            (batch_size, max_tgt_len), pad_id,
            dtype=tgt_flat.dtype, device=tgt_flat.device
        )
        labels = torch.full(
            (batch_size, max_tgt_len), -100,  # ignore_index
            dtype=labels_flat.dtype, device=labels_flat.device
        )

        # Create masks
        src_mask = torch.zeros(batch_size, max_src_len, dtype=torch.bool, device=src_flat.device)
        tgt_mask = torch.zeros(batch_size, max_tgt_len, dtype=torch.bool, device=tgt_flat.device)

        # Unpack each sequence
        for i in range(batch_size):
            src_start, src_end = cu_seqlens_src[i].item(), cu_seqlens_src[i + 1].item()
            tgt_start, tgt_end = cu_seqlens_tgt[i].item(), cu_seqlens_tgt[i + 1].item()

            src_len = src_end - src_start
            tgt_len = tgt_end - tgt_start

            src_ids[i, :src_len] = src_flat[src_start:src_end]
            tgt_ids[i, :tgt_len] = tgt_flat[tgt_start:tgt_end]
            labels[i, :tgt_len] = labels_flat[tgt_start:tgt_end]

            src_mask[i, :src_len] = True
            tgt_mask[i, :tgt_len] = True

        return src_ids, tgt_ids, labels, src_mask, tgt_mask

    def _compute_masked_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with masking for padded tokens.

        Args:
            logits: Model output (batch, seq_len, vocab_size)
            labels: Target labels (batch, seq_len)
            mask: Boolean mask for valid positions (batch, seq_len)

        Returns:
            Scalar loss
        """
        # Flatten for loss computation
        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = labels.reshape(-1)

        # The loss_fn already handles ignore_index=-100
        loss = self.loss_fn(logits_flat, labels_flat)

        return loss

    def _evaluate(self):
        """Run evaluation (main process only for logging)."""
        if self.eval_fn is None or self.eval_dataloader is None:
            return

        self.model.eval()
        metrics = self.eval_fn(self.model, self.eval_dataloader, self.device)
        self.model.train()

        if self.is_main:
            print(f"Eval @ step {self.global_step}: {metrics}")

            # Track best metric
            if "loss" in metrics and metrics["loss"] < self.best_metric:
                self.best_metric = metrics["loss"]
                self._save_checkpoint("best")

        barrier()  # Sync after evaluation

    def _save_checkpoint(self, name: Optional[str] = None):
        """
        Save training checkpoint with embedded config (NeurIPS/ICML standard).

        Saves a unified checkpoint.pt with:
        - model_state_dict: Model weights
        - config: ModelConfig for architecture reproduction
        - optimizer_state_dict: Optimizer state for resumption
        - scheduler_state_dict: Scheduler state
        - Training metadata (step, epoch, metrics)
        - Environment metadata (PyTorch version, CUDA, timestamp)
        """
        if not self.is_main:
            barrier()  # Wait for main process
            return

        from dataclasses import asdict
        from datetime import datetime

        if name is None:
            name = f"checkpoint-{self.global_step}"

        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get underlying model from wrappers (DDP/FSDP/compile)
        model_to_save = self.model
        # Unwrap torch.compile
        if hasattr(model_to_save, "_orig_mod"):
            model_to_save = model_to_save._orig_mod
        # Unwrap DDP
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        # Get config from model
        config_dict = None
        if hasattr(model_to_save, 'config'):
            config_dict = asdict(model_to_save.config)

        # Build unified checkpoint (NeurIPS/ICML standard format)
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'config': config_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            # RNG states for reproducible resume (critical for publication)
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            'metadata': {
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'timestamp': datetime.now().isoformat(),
                'world_size': self.world_size,
                'seed': self.config.seed,
            }
        }

        # Save unified checkpoint
        torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")

        # Also save model-only checkpoint for inference (backward compatibility)
        torch.save(model_to_save.state_dict(), checkpoint_dir / "model.pt")

        print(f"Saved checkpoint to {checkpoint_dir}")
        if config_dict:
            print(f"  Config: d_model={config_dict.get('d_model')}, "
                  f"layers={config_dict.get('encoder_layers')}/{config_dict.get('decoder_layers')}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        barrier()  # Sync after saving

    def _cleanup_checkpoints(self):
        """Remove old checkpoints to respect save_total_limit."""
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1]),
        )

        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)
            print(f"Removed old checkpoint: {oldest}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load from checkpoint (supports both new and legacy formats).

        New format: checkpoint.pt with embedded config and all state
        Legacy format: model.pt + training_state.pt
        """
        checkpoint_dir = Path(checkpoint_path)

        # Get underlying model
        model_to_load = self.model
        if hasattr(model_to_load, "_orig_mod"):
            model_to_load = model_to_load._orig_mod
        if hasattr(model_to_load, "module"):
            model_to_load = model_to_load.module

        # Try new unified format first
        unified_path = checkpoint_dir / "checkpoint.pt"
        if unified_path.exists():
            checkpoint = torch.load(unified_path, map_location=self.device, weights_only=False)
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.global_step = checkpoint['global_step']
            self.epoch = checkpoint['epoch']
            self.best_metric = checkpoint['best_metric']

            # Restore RNG states for reproducible resume
            if 'rng_state' in checkpoint:
                rng_state = checkpoint['rng_state']
                random.setstate(rng_state['python'])
                np.random.set_state(rng_state['numpy'])
                torch.set_rng_state(rng_state['torch'])
                if rng_state['cuda'] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng_state['cuda'])
                if self.is_main:
                    print("  Restored RNG states for reproducible resume")

            if 'metadata' in checkpoint:
                print(f"Loaded checkpoint from {checkpoint_dir}")
                print(f"  Step: {self.global_step}, Epoch: {self.epoch}")
                print(f"  Saved: {checkpoint['metadata'].get('timestamp', 'unknown')}")
        else:
            # Fallback to legacy format
            model_to_load.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=self.device))

            state = torch.load(checkpoint_dir / "training_state.pt", map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_metric = state["best_metric"]

            print(f"Loaded legacy checkpoint from {checkpoint_dir} (step {self.global_step})")
