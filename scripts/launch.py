#!/usr/bin/env python3
"""Launch script for distributed training with proper NCCL configuration."""

import os
import subprocess
import sys

import torch


def main():
    # NCCL tuning for H100 - MUST be set before process group init
    print("Configuring NCCL environment for H100 training...")
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")  # Use NVLink for peer-to-peer
    os.environ.setdefault("NCCL_IB_DISABLE", "0")   # Enable InfiniBand if available
    os.environ.setdefault("NCCL_DEBUG", "WARN")     # Reduce log spam
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker")  # Exclude loopback/docker

    # PyTorch CUDA settings
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Detect GPU count - CUDA is required
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected. CUDA is required for training.")

    # Build torchrun command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        "train.py",
    ] + sys.argv[1:]

    print(f"Launching distributed training on {num_gpus} GPU(s)")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
