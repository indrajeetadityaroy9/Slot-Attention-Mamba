"""Constants with explicit theoretical citations.

All values in this file are either:
1. Derived from input statistics at runtime (see training/adaptive.py)
2. Standard values from cited papers
3. Hardware-specific (H100)

REMOVED (Now Computed Adaptively):
- WARMUP_RATIO → Derived from gradient norm stability
- WEIGHT_DECAY → Derived from parameter magnitude
- DROPOUT → Derived from capacity/data ratio
- AGC_CLIP_FACTOR → Derived from initialization scale
- LOG_STEPS, EVAL_STEPS, SAVE_STEPS → Derived from dataset size
- MIN_LR → Set to 0 per SGDR (pure cosine annealing)
"""

# =============================================================================
# Mamba Block Parameters
# Reference: Gu & Dao, 2023 (arXiv 2312.00752, Section 3.4)
# "We use a local convolution of width 4"
# "expansion factor E = 2" (standard SSM practice)
# =============================================================================
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2

# =============================================================================
# Token IDs (NLP Convention)
# Standard practice, no citation needed
# =============================================================================
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SEP_TOKEN_ID = 3  # Key-value separator ':'
QUERY_TOKEN_ID = 4

# =============================================================================
# MQAR Vocabulary Ranges
# Task-specific: prevent key/value collision by partitioning vocabulary
# Keys: [10, 4096), Values: [4096, 8192)
# =============================================================================
KEY_TOKEN_START = 10
KEY_TOKEN_END = 4096
VALUE_TOKEN_START = 4096
VALUE_TOKEN_END = 8192

# =============================================================================
# Optimizer Hyperparameters
# Reference: Hoffmann et al., 2022 (arXiv 2203.15556, Appendix A)
# β1=0.9 is standard momentum, β2=0.95 per Chinchilla training recipe
# ε=1e-8 is IEEE 754 standard for numerical stability
# =============================================================================
ADAM_BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8

# =============================================================================
# H100 Infrastructure (Hardware-Specific)
# These are hardware-dependent choices, not hyperparameters
# =============================================================================
USE_BF16 = True  # H100 native BF16 support
USE_COMPILE = True  # torch.compile for H100 optimization
COMPILE_MODE = "max-autotune"  # Maximum autotuning for H100
GRADIENT_CHECKPOINTING = True  # Memory optimization for large models
MAX_SEQ_LEN = 8192  # H100 memory-bound maximum

# =============================================================================
# MQAR Task Defaults
# =============================================================================
MQAR_VOCAB_SIZE = 8192
MQAR_SEQ_LENGTH = 512

# =============================================================================
# Minimum warmup steps (safety floor)
# Reference: Goyal et al., 2017 (arXiv 1706.02677)
# "Gradual warmup helps alleviate early training instability"
# 100 steps is empirical minimum for gradient statistics to stabilize
# =============================================================================
MIN_WARMUP_STEPS = 100
