import argparse

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from align_mamba.config import load_yaml
from align_mamba.model import HybridMambaEncoderDecoder
from align_mamba.data import create_dataloaders
from align_mamba.training import Trainer, emit_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    config = load_yaml(args.config)

    set_seed(config.seed)
    accelerator = Accelerator(mixed_precision="bf16")

    model = HybridMambaEncoderDecoder(config, dtype=torch.bfloat16)
    train_loader, val_loader = create_dataloaders(config)

    if accelerator.is_main_process:
        params = sum(p.numel() for p in model.parameters())
        log_fields = dict(
            event="train_start",
            config_path=args.config,
            params_m=round(params / 1e6, 3),
            d_model=config.d_model,
            task=config.task,
            max_steps=config.max_steps,
        )
        if config.task == "lm":
            log_fields["seq_length"] = config.lm_seq_length
            log_fields["vocab_size"] = config.vocab_size
        else:
            log_fields["d_state"] = config.d_state
            log_fields["num_pairs"] = config.num_pairs
        emit_log(**log_fields)

    trainer = Trainer(model, train_loader, config, accelerator, val_loader)
    if config.resume_from:
        trainer.resume_from(config.resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
