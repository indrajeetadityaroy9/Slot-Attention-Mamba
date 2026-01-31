"""Setup script for Align-Mamba with CUDA extensions.

Build CUDA kernels:
    python setup.py build_ext --inplace

Install with CUDA:
    pip install -e .
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

# Check if CUDA is available
def get_cuda_version():
    """Get CUDA version from nvcc."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            # Parse version from output like "release 12.1"
            for line in result.stdout.split("\n"):
                if "release" in line:
                    parts = line.split("release")[-1].strip().split(",")[0]
                    return parts
        return None
    except FileNotFoundError:
        return None

CUDA_VERSION = get_cuda_version()
CUDA_AVAILABLE = CUDA_VERSION is not None

# Get compute capability for current GPU
def get_compute_capability():
    """Detect GPU compute capability."""
    try:
        import torch
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            return f"sm_{capability[0]}{capability[1]}"
    except ImportError:
        pass
    # Default to sm_80 (A100) or sm_90 (H100)
    return "sm_90"

# CUDA extension configuration
def get_cuda_extensions():
    """Create CUDA extension if CUDA is available."""
    if not CUDA_AVAILABLE:
        print("CUDA not found. Skipping CUDA extension build.")
        return []

    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError:
        print("PyTorch not found. Skipping CUDA extension build.")
        return []

    compute_cap = get_compute_capability()

    # CUDA source files
    cuda_sources = [
        "csrc/bindings.cpp",
        "csrc/kernels/polarized_mamba.cu",
        "csrc/kernels/state_expansion.cu",
        "csrc/kernels/based_linear.cu",
        "csrc/kernels/memmamba.cu",
    ]

    # Check if all source files exist
    missing = [f for f in cuda_sources if not Path(f).exists()]
    if missing:
        print(f"Warning: Missing CUDA source files: {missing}")
        return []

    return [
        CUDAExtension(
            name="align_mamba_cuda",
            sources=cuda_sources,
            include_dirs=["csrc/include"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    f"-arch={compute_cap}",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ]


class BuildExtension(build_ext):
    """Custom build extension that handles CUDA compilation."""

    def build_extensions(self):
        if not self.extensions:
            print("No extensions to build.")
            return
        super().build_extensions()


# Only use cmdclass if we have extensions
def get_cmdclass():
    """Get command class for build."""
    if CUDA_AVAILABLE:
        try:
            from torch.utils.cpp_extension import BuildExtension as TorchBuildExt
            return {"build_ext": TorchBuildExt}
        except ImportError:
            pass
    return {"build_ext": BuildExtension}


if __name__ == "__main__":
    setup(
        name="align-mamba",
        version="0.1.0",
        description="State Capacity Limits in Selective SSMs with CUDA Kernels",
        packages=find_packages(include=["models*", "training*", "data*", "kernels*"]),
        ext_modules=get_cuda_extensions(),
        cmdclass=get_cmdclass(),
        python_requires=">=3.10",
        install_requires=[
            "torch>=2.3.0",
            "numpy<2",
            "hydra-core>=1.3",
            "omegaconf>=2.3",
            "tokenizers>=0.19",
            "datasets>=2.19",
            "mamba-ssm>=2.0",
            "causal-conv1d>=1.2",
            "flash-attn>=2.5",
        ],
        entry_points={
            "console_scripts": [
                "align-train=train:main",
                "align-eval=evaluate:main",
            ],
        },
    )
