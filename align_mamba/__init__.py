"""Align-Mamba: State Capacity Limits in Selective SSMs."""

__version__ = "0.1.0"

from align_mamba.config import Config
from align_mamba.model import HybridMambaEncoderDecoder, load_checkpoint
from align_mamba.train import main as train
from align_mamba.evaluate import main as evaluate

__all__ = ["Config", "HybridMambaEncoderDecoder", "load_checkpoint", "train", "evaluate"]
