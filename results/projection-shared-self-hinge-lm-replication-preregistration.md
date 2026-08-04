# Projection-shared self-hinge — frozen 50M-token replication

Status: **frozen before execution**  
Date: 2026-07-28

## Retained candidate

The 10M-token development screen rejected two-way cross-channel coupling in
favor of the simpler matched control.  This replication tests only the
surviving mechanism:

\[
z=\operatorname{RMSNorm}_{g=1}(x),\qquad
z'_i=z_i+\gamma_i|z_i|,
\]

with `gamma = 0.25*tanh(theta/0.25)`.  Each theta occupies the slot of the
removed RMSNorm gain for that channel.  The transform is identity at theta
zero and invertible because `|gamma|<1`; it supplies separate slopes
`1+gamma` and `1-gamma` for positive and negative normalized activations.

The chart is computed once per norm output and shared by all immediate
projections: Q/K/V for attention and gate/up for the FFN.  Final RMSNorm is
unchanged.

## Frozen arms

1. `raw_baseline`: ordinary learned RMSNorm gains.
2. `folded_null`: gain-free RMSNorm with the same theta parameter tensor but
   identity forward, controlling optimization after gain removal.
3. `self_hinge`: candidate sign-dependent slope.

All arms have identical raw/trainable parameters, optimizer slots, matrix
shapes, hidden/FFN widths, attention heads, KV cache, and initialization seed.

## Frozen protocol

- same 12-layer hidden-384, SwiGLU-1,024 Qwen-style scratch model and immutable
  FineWeb-Edu token files as development;
- independent seed 811 on the retained H100;
- sequence 512, microbatch 32, accumulation 2;
- 1,525 steps = 49,971,200 prediction tokens per arm;
- 128 validation batches of 32 at steps 0, 305, and 1,525;
- AdamW LR 3e-4, betas (0.9,0.95), weight decay 0.1, 100-step warmup,
  cosine decay, gradient clip 1.0;
- Torch 2.5.1+cu124, CUDA 12.4, Transformers 4.57.6;
- no checkpoint selection, coefficient-specific LR, or post-result tuning.

## Frozen promotion gates

1. Protocol, data ledger, parameter equality, initial-loss equality within
   `2e-5`, finite training, and stability all pass.
2. At 10M tokens self-hinge is no worse than raw baseline or folded null.
3. At 50M tokens self-hinge NLL is at least 0.005% below both controls, and
   paired per-batch 95% intervals wholly favor self-hinge against both.
4. Mean absolute gamma is at least 0.002; max absolute gamma is below 0.25;
   worst scalar-chart condition `(1+|gamma|)/(1-|gamma|)` is below 1.2.
5. Turning gamma off in the trained candidate worsens NLL by at least 0.005%,
   with a paired interval wholly favoring the full candidate.
6. Normalized coordinate positive fraction remains in `[0.1,0.9]`, and output
   RMS remains within 5% of input RMS.

Passing establishes a replicated BF16 architecture effect at this scale—not a
general scaling law and not quantized-serving equivalence.  The next gate would
be multi-seed scaling plus an exact quantized chart.

