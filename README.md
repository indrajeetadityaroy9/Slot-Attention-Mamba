# SlotMamba

Selective state space models (SSMs) provide linear-time sequence processing, but their fixed-size latent state imposes a hard memory bottleneck: when the number of key-value associations exceeds d_state, retrieval accuracy degrades sharply due to state saturation.

SlotMamba is a hybrid recurrent model for associative recall and language modeling. It combines a BiMamba encoder with an HKSA decoder (Kronecker addressing, Householder slot updates, optional PDMA decay, surprise gating, and optional
encoder-slot injection).
