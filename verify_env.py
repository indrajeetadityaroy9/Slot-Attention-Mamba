#!/usr/bin/env python3
"""
Verify Align-Mamba environment is correctly configured for H100 training.

Run this after install_env.sh to confirm ICML/NeurIPS reproducibility requirements.

Usage:
    python verify_env.py
"""

import sys


def check_package(name: str, import_name: str = None) -> tuple[bool, str]:
    """Check if a package is installed and get its version."""
    import_name = import_name or name
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "OK")
        return True, version
    except ImportError as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("Align-Mamba Environment Verification")
    print("=" * 60)
    print()

    errors = []

    # --- PyTorch & CUDA ---
    print("[1/5] PyTorch & CUDA")
    try:
        import torch

        print(f"      PyTorch: {torch.__version__}")
        print(f"      CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"      CUDA version: {torch.version.cuda}")
            print(f"      GPU: {torch.cuda.get_device_name(0)}")

            # H100 TF32 check
            if torch.backends.cuda.matmul.allow_tf32:
                print("      TF32 matmul: Enabled (H100 optimized)")
            else:
                print("      TF32 matmul: Disabled")
        else:
            errors.append("CUDA not available - GPU training will fail")
    except ImportError as e:
        errors.append(f"PyTorch not installed: {e}")
    print()

    # --- Mamba-SSM ---
    print("[2/5] Mamba-SSM (O(L) efficiency)")
    ok, version = check_package("mamba_ssm", "mamba_ssm")
    if ok:
        print(f"      mamba-ssm: {version}")
    else:
        errors.append(f"mamba-ssm not installed: {version}")
        print(f"      mamba-ssm: FAILED")

    ok, version = check_package("causal_conv1d", "causal_conv1d")
    if ok:
        print(f"      causal-conv1d: {version}")
    else:
        errors.append(f"causal-conv1d not installed: {version}")
        print(f"      causal-conv1d: FAILED")
    print()

    # --- FlashAttention ---
    print("[3/5] FlashAttention-2 (H100 optimized)")
    ok, version = check_package("flash_attn", "flash_attn")
    if ok:
        print(f"      flash-attn: {version}")
    else:
        # FlashAttention is optional - has SDPA fallback
        print(f"      flash-attn: Not installed (will use PyTorch SDPA fallback)")
    print()

    # --- Evaluation Stack ---
    print("[4/5] Evaluation Stack")
    eval_packages = [
        ("sacrebleu", "sacrebleu"),
        ("unbabel-comet", "comet"),
        ("spacy", "spacy"),
        ("awesome-align", "awesome_align"),
    ]

    for pkg_name, import_name in eval_packages:
        ok, version = check_package(pkg_name, import_name)
        if ok:
            print(f"      {pkg_name}: {version}")
        else:
            errors.append(f"{pkg_name} not installed")
            print(f"      {pkg_name}: FAILED")

    # Check SpaCy models
    try:
        import spacy

        try:
            spacy.load("en_core_web_sm")
            print("      spacy en_core_web_sm: OK")
        except OSError:
            errors.append("SpaCy model en_core_web_sm not downloaded")
            print("      spacy en_core_web_sm: MISSING")

        try:
            spacy.load("de_core_news_sm")
            print("      spacy de_core_news_sm: OK")
        except OSError:
            errors.append("SpaCy model de_core_news_sm not downloaded")
            print("      spacy de_core_news_sm: MISSING")
    except ImportError:
        pass
    print()

    # --- Project Installation ---
    print("[5/5] Project Installation")
    try:
        import doc_nmt_mamba

        print("      doc_nmt_mamba: OK")
    except ImportError:
        errors.append("doc_nmt_mamba not installed (run: pip install -e .)")
        print("      doc_nmt_mamba: FAILED")
    print()

    # --- Summary ---
    print("=" * 60)
    if errors:
        print("VERIFICATION FAILED")
        print("=" * 60)
        print()
        print("Errors:")
        for err in errors:
            print(f"  - {err}")
        print()
        print("To fix, run: ./install_env.sh")
        return 1
    else:
        print("VERIFICATION PASSED")
        print("=" * 60)
        print()
        print("Environment is Reproducible and H100 Ready")
        print()
        print("Next steps:")
        print("  1. Run tests: python -m pytest doc_nmt_mamba/tests/ -v")
        print("  2. Train: python doc_nmt_mamba/scripts/train.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
