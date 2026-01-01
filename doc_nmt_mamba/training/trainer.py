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
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast

from .objectives import create_loss_fn
from .schedulers import create_scheduler
from .distributed import (
    DistributedConfig,
    setup_distributed,
    cleanup_distributed,
    wrap_model_distributed,
    print_distributed_info,
    all_reduce_mean,
    barrier,
)


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""

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
        """Setup H100-specific optimizations."""
        if self.config.tf32_matmul:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if self.config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
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

        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
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

        while self.global_step < self.config.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                # Update sampler epoch for proper shuffling
                if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(self.epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Move to device
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass with autocast
            with autocast(dtype=torch.bfloat16, enabled=self.config.use_bf16):
                loss = self._training_step(batch)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()

            # Optimizer step
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

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

                    print(
                        f"Step {self.global_step}/{self.config.max_steps} | "
                        f"Loss: {accumulated_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Steps/s: {steps_per_sec:.2f} | "
                        f"Samples/s: {samples_per_sec:.1f}"
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
            # Labels: tgt_ids[:, 1:] shifted left (all but first)
            decoder_input = tgt_ids[:, :-1]
            labels_shifted = tgt_ids[:, 1:].clone()

            # Replace padding positions with ignore_index in labels
            labels_shifted[~tgt_mask[:, 1:]] = -100

            logits = self.model(src_ids, decoder_input)

            # Flatten and compute loss
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels_shifted.reshape(-1)

            loss = self.loss_fn(logits_flat, labels_flat)
        else:
            # Padded sequences
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
        """Save training checkpoint (main process only)."""
        if not self.is_main:
            barrier()  # Wait for main process
            return

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

        torch.save(model_to_save.state_dict(), checkpoint_dir / "model.pt")

        # Save optimizer and scheduler
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_metric": self.best_metric,
                "world_size": self.world_size,
            },
            checkpoint_dir / "training_state.pt",
        )

        print(f"Saved checkpoint to {checkpoint_dir}")

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
        """Load from checkpoint."""
        checkpoint_dir = Path(checkpoint_path)

        # Load model
        model_to_load = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        model_to_load.load_state_dict(torch.load(checkpoint_dir / "model.pt"))

        # Load training state
        state = torch.load(checkpoint_dir / "training_state.pt")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_metric = state["best_metric"]

        print(f"Loaded checkpoint from {checkpoint_dir} (step {self.global_step})")
