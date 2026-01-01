"""
Distributed Training Utilities for Multi-GPU Training.

Supports:
- DDP (DistributedDataParallel) for data parallelism
- FSDP (FullyShardedDataParallel) for memory-efficient large model training
- Automatic NVLink detection and optimization
"""

import os
import socket
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from functools import partial


class DistributedStrategy(Enum):
    """Distributed training strategies."""
    NONE = "none"           # Single GPU
    DDP = "ddp"             # DistributedDataParallel
    FSDP = "fsdp"           # FullyShardedDataParallel
    FSDP_FULL = "fsdp_full" # FSDP with full sharding


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    # Strategy
    strategy: str = "ddp"  # "none", "ddp", "fsdp", "fsdp_full"

    # DDP settings
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = True  # Enable for torch.compile compatibility

    # FSDP settings
    sharding_strategy: str = "full_shard"  # "full_shard", "shard_grad_op", "no_shard"
    cpu_offload: bool = False
    backward_prefetch: str = "backward_pre"  # "backward_pre", "backward_post", None
    min_num_params: int = 1_000_000  # Min params per FSDP unit

    # Mixed precision for FSDP
    fsdp_mixed_precision: bool = True

    # Process group
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU

    # Environment
    master_addr: str = "localhost"
    master_port: str = "29500"


def setup_distributed(config: Optional[DistributedConfig] = None) -> Dict[str, Any]:
    """
    Setup distributed training environment.

    Returns:
        Dict with rank, local_rank, world_size, device
    """
    config = config or DistributedConfig()

    # Check if already initialized
    if dist.is_initialized():
        return {
            "rank": dist.get_rank(),
            "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
            "world_size": dist.get_world_size(),
            "device": torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"),
            "is_main": dist.get_rank() == 0,
        }

    # Check for distributed environment variables
    if "RANK" in os.environ:
        # Launched with torchrun or similar
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif torch.cuda.device_count() > 1:
        # Multi-GPU but not launched distributed - setup for single process multi-GPU
        rank = 0
        local_rank = 0
        world_size = 1
        print(f"Detected {torch.cuda.device_count()} GPUs. Use torchrun for multi-GPU training.")
    else:
        # Single GPU
        return {
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "is_main": True,
        }

    # Set environment variables
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", config.master_addr)
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", config.master_port)

    # Initialize process group
    if not dist.is_initialized() and world_size > 1:
        dist.init_process_group(
            backend=config.backend,
            rank=rank,
            world_size=world_size,
        )

    # Set device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "is_main": rank == 0,
    }


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_fsdp_mixed_precision(use_bf16: bool = True) -> MixedPrecision:
    """Get FSDP mixed precision policy for H100."""
    if use_bf16:
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        return MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )


def get_fsdp_sharding_strategy(strategy: str) -> ShardingStrategy:
    """Get FSDP sharding strategy."""
    strategies = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    return strategies.get(strategy, ShardingStrategy.FULL_SHARD)


def get_backward_prefetch(prefetch: Optional[str]) -> Optional[BackwardPrefetch]:
    """Get FSDP backward prefetch policy."""
    if prefetch is None:
        return None
    policies = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
    }
    return policies.get(prefetch, BackwardPrefetch.BACKWARD_PRE)


def wrap_model_distributed(
    model: nn.Module,
    config: DistributedConfig,
    device: torch.device,
    use_bf16: bool = True,
) -> nn.Module:
    """
    Wrap model for distributed training.

    Args:
        model: Model to wrap
        config: Distributed configuration
        device: Device to use
        use_bf16: Whether to use BF16 precision

    Returns:
        Wrapped model (DDP, FSDP, or original)
    """
    strategy = DistributedStrategy(config.strategy)

    if strategy == DistributedStrategy.NONE:
        return model.to(device)

    if not dist.is_initialized():
        print("Warning: Distributed not initialized, using single GPU")
        return model.to(device)

    if strategy == DistributedStrategy.DDP:
        # Move model to device first
        model = model.to(device)

        # Wrap with DDP
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=config.find_unused_parameters,
            gradient_as_bucket_view=config.gradient_as_bucket_view,
            static_graph=config.static_graph,
        )

    elif strategy in (DistributedStrategy.FSDP, DistributedStrategy.FSDP_FULL):
        # Get FSDP wrapping policy
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=config.min_num_params,
        )

        # Get mixed precision policy
        mixed_precision = None
        if config.fsdp_mixed_precision:
            mixed_precision = get_fsdp_mixed_precision(use_bf16)

        # Get sharding strategy
        sharding_strategy = get_fsdp_sharding_strategy(config.sharding_strategy)

        # Get backward prefetch
        backward_prefetch = get_backward_prefetch(config.backward_prefetch)

        # CPU offload
        cpu_offload = CPUOffload(offload_params=True) if config.cpu_offload else None

        # Wrap with FSDP
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=sharding_strategy,
            backward_prefetch=backward_prefetch,
            cpu_offload=cpu_offload,
            device_id=device,
            use_orig_params=True,  # Required for torch.compile
        )

    return model


def get_nvlink_info() -> Dict[str, Any]:
    """Get NVLink topology information."""
    info = {
        "gpu_count": torch.cuda.device_count(),
        "nvlink_available": False,
        "p2p_available": [],
    }

    if info["gpu_count"] < 2:
        return info

    # Check P2P access between GPUs
    for i in range(info["gpu_count"]):
        for j in range(info["gpu_count"]):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                info["p2p_available"].append((i, j, can_access))
                if can_access:
                    info["nvlink_available"] = True

    return info


def print_distributed_info(dist_info: Dict[str, Any]):
    """Print distributed training information."""
    if not dist_info.get("is_main", True):
        return

    print("\n" + "=" * 60)
    print("DISTRIBUTED TRAINING INFO")
    print("=" * 60)
    print(f"World Size: {dist_info['world_size']}")
    print(f"Rank: {dist_info['rank']}")
    print(f"Local Rank: {dist_info['local_rank']}")
    print(f"Device: {dist_info['device']}")

    # NVLink info
    nvlink_info = get_nvlink_info()
    print(f"\nGPU Count: {nvlink_info['gpu_count']}")
    print(f"NVLink Available: {nvlink_info['nvlink_available']}")

    if nvlink_info['p2p_available']:
        print("P2P Access:")
        for i, j, can_access in nvlink_info['p2p_available']:
            status = "✓" if can_access else "✗"
            print(f"  GPU {i} -> GPU {j}: {status}")

    print("=" * 60 + "\n")


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce tensor and compute mean across processes."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast object from source rank to all ranks."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return obj

    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def barrier():
    """Synchronization barrier."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
