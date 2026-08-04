# Projection-coalesced attention/FFN — matched LM screen

This document freezes the capability screen after the deterministic Stage-0
gate passed and before any arm was trained.

## Immutable inputs

- Data and tokenizer: the existing block-algebra scratch token files and their
  manifest.
- Architecture: 12 layers, hidden size 384, six query heads, two KV heads,
  head size 64, RoPE and SDPA.
- Baseline SwiGLU width: 1,024.
- Coalesced matched width: 1,344, derived before training from exact equality
  of the combined input-projection rows.
- Arms, in order: `sequential_baseline`, `parallel_baseline`,
  `coalesced_1024`, `coalesced_1344`.
- Seed: 1601.
- Optimizer: fused AdamW, beta `(0.9, 0.95)`, epsilon `1e-8`, weight decay
  0.1, peak LR `3e-4`, 100 linear warmup steps, cosine decay, gradient clip 1.
- BF16 autocast, TF32 enabled, H100 SXM, PyTorch 2.5.1+cu124,
  Transformers 4.57.6.
- Sequence 512, microbatch 32, accumulation 2, 1,525 optimizer steps:
  49,971,200 prediction tokens per arm.
- Validation: 128 fixed batches of 32 sequences at steps 0, 305, and 1,525.

## Architecture controls

`sequential_baseline` is the ordinary Llama ordering with two pre-norms and
two residual additions.

`parallel_baseline` retains independent Q/K/V/O and SwiGLU matrices but both
branches read one shared pre-norm state and are added through one residual.
It isolates lost within-layer serialization.

Both coalesced arms use the exact Stage-0 slice map and one shared pre-norm.
The 1,024 arm measures the raw price of tying without budget reinvestment.  The
1,344 arm is the candidate.

All full-strength projection matrices use the same initializer standard
deviation `1/sqrt(384)`.  No special endpoint initialization, branch scaling,
router loss, distillation, or candidate-specific optimizer group is allowed.

## Decision

Promotion requires every Stage-0 preregistered gate.  Operationally:

1. Candidate terminal NLL is at least 0.05% below the sequential baseline and
   its paired per-validation-batch 95% interval is wholly below zero.
2. Candidate terminal NLL is at least 0.025% below both the parallel baseline
   and the width-1,024 coalesced control.
3. Candidate at 10M tokens is no more than 0.05% worse than sequential.
4. Training and sampled activation diagnostics remain finite and bounded.
5. Zeroing local products on all 640 attention-bearing channels worsens
   terminal NLL by at least 0.01%, with a paired interval favoring full.

This one-seed screen can close the branch or justify another seed.  It cannot
prove an A-E claim.
