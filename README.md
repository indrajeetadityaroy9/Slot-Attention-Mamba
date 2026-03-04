# SlotMamba

A hybrid encoder-decoder architecture for associative recall and language modeling that addresses the fixed-size memory bottleneck in state space models (SSMs). When the number of stored associations exceeds the state dimension, standard SSMs suffer sharp retrieval degradation. SlotMamba mitigates this with a structured external memory: a decoder reads and writes to a bank of key-value slots using factored addressing, orthogonal slot updates, adaptive decay, and gated memory refinement, while a bidirectional SSM encoder provides context representations for cross-attention injection into the slot memory.

## Architecture

**Encoder** — Interleaved bidirectional Mamba (forward + reverse SSM) and multi-head attention layers with rotary position embeddings. Attention density is proportional to `d_state / d_model`. Used for sequence-to-sequence tasks; omitted for decoder-only language modeling.

**Decoder** — Each layer contains:
- **Block-diagonal linear recurrence** for local sequential mixing
- **Slot attention with factored addressing** — queries and keys are projected into a factored (Kronecker product) address space, selecting top-k slots from an external memory bank. Slot contents are updated via orthogonal reflector transformations with learned step sizes
- **Adaptive memory decay** — slot activations decay at a learned rate, preventing unbounded accumulation
- **Surprise gating** — an EMA-based novelty signal modulates retention of updated vs. previous slot contents
- **Cross-attention injection** — encoder outputs are injected into active slots via gated cross-attention (encoder-decoder mode only)
- **Causal self-attention** at a single decoder layer for global token mixing

All attention uses FlashAttention with NTK-aware rotary embeddings. Learnable register tokens prepend the decoder input.

## Self-Calibrating Configuration

All behavioral hyperparameters derive from model structure and dataset metadata — no manual tuning constants. Key derivations:

| Parameter | Derivation |
|---|---|
| Learning rate | Gradient-norm probe: `1 / (grad_norm * sqrt(d_model))` |
| Weight decay | `= learning_rate` (scale-invariant equilibrium) |
| Adam betas | `beta1 = 1 - 1/sqrt(epoch_steps)`, `beta2 = 1 - 1/sqrt(max_steps)` |
| LR warmup | `ceil(1 / (1 - beta2))` steps |
| RoPE base | `seq_len ^ (head_dim / (head_dim - 2))` |
| Label smoothing | `1 / log(vocab_size)` |
| Decay init | `softplus_inverse(1 / seq_len)` |
| Gradient clip | EMA-tracked `grad_norm * (1 + 1/sqrt(step))` |
