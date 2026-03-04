import argparse

from accelerate import Accelerator
from accelerate.utils import set_seed

from config import load_yaml, DTYPE_MAP, MIXED_PRECISION_MAP
from model import HybridMambaEncoderDecoder
from data import create_dataloaders
from training import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    config = load_yaml(args.config)

    set_seed(config.seed)
    compute_dtype = DTYPE_MAP[config.compute_dtype]
    accelerator = Accelerator(mixed_precision=MIXED_PRECISION_MAP[config.compute_dtype])

    model = HybridMambaEncoderDecoder(config, dtype=compute_dtype)
    train_loader, val_loader = create_dataloaders(config)

    if accelerator.is_main_process:
        params = sum(p.numel() for p in model.parameters())
        print(f"[start] config={args.config} params={params / 1e6:.3f}M d_model={config.d_model} task={config.task} max_steps={config.max_steps}")
        print(f"[derived] rope_base={config.rope_base:.1f} decay_gamma_init={config.decay_gamma_init:.4f} "
              f"label_smoothing={config.label_smoothing:.4f} n_registers={config.n_registers} "
              f"compute_dtype={config.compute_dtype}")
        if config.task == "lm":
            print(f"  seq_length={config.lm_seq_length} vocab_size={config.vocab_size}")
        else:
            print(f"  d_state={config.d_state} num_pairs={config.num_pairs}")

    trainer = Trainer(model, train_loader, config, accelerator, val_loader)
    if config.resume_from:
        trainer.resume_from(config.resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
