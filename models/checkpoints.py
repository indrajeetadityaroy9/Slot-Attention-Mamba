"""Checkpoint utilities with embedded config for reproducibility."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .encoder_decoder import ModelConfig, HybridMambaEncoderDecoder
from .utils import get_unwrapped_model


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = 'cuda',
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
) -> Tuple[nn.Module, ModelConfig]:
    """Load model from checkpoint for evaluation."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(
            "Invalid checkpoint format. Expected dict with 'model_state_dict' and 'config' keys."
        )

    state_dict = checkpoint['model_state_dict']

    config_dict = checkpoint.get('config')
    if config_dict is None:
        raise ValueError(
            "Checkpoint missing required 'config' field. "
            "Checkpoints must include embedded ModelConfig for reproducibility."
        )
    config = ModelConfig(**config_dict)
    metadata = checkpoint.get('metadata')

    if metadata:
        print(f"Checkpoint metadata:")
        print(f"  PyTorch: {metadata.get('pytorch_version', 'unknown')}")
        print(f"  CUDA: {metadata.get('cuda_version', 'unknown')}")
        print(f"  Timestamp: {metadata.get('timestamp', 'unknown')}")

    model = HybridMambaEncoderDecoder(
        config=config,
        device=device,
        dtype=dtype,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing or unexpected:
        if missing:
            print(f"Warning: Missing {len(missing)} keys in state_dict")
        if unexpected:
            print(f"Warning: Unexpected {len(unexpected)} keys in state_dict")

    model = model.to(device)
    model.eval()

    return model, config


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    global_step: int = 0,
    epoch: int = 0,
    best_metric: Optional[float] = None,
    output_path: Union[str, Path] = "checkpoint.pt",
    world_size: int = 1,
) -> None:
    """Save checkpoint with embedded config for reproducibility."""
    unwrapped = get_unwrapped_model(model)
    config = asdict(unwrapped.config)
    state_dict = unwrapped.state_dict()

    checkpoint = {
        'config': config,
        'model_state_dict': state_dict,
        'global_step': global_step,
        'epoch': epoch,
        'best_metric': best_metric,
        'metadata': {
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'timestamp': datetime.now().isoformat(),
            'world_size': world_size,
        }
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    print(f"Saved checkpoint to {output_path}")
    print(f"  Config: d_model={config['d_model']}, layers={config['encoder_layers']}/{config['decoder_layers']}")
    print(f"  Step: {global_step}, Epoch: {epoch}")
