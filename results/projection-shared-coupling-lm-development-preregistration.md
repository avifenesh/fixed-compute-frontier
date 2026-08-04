# Projection-shared coupling — frozen 10M-token LM development screen

Status: **frozen before execution**  
Date: 2026-07-28

## Question

Does converting each sublayer RMSNorm's `d` gain parameters into `d` nonlinear
coupling parameters improve causal language modeling at identical raw and
trainable parameter count, projection shapes, hidden width, training tokens,
and optimizer schedule?

This is a development screen, not a replication claim.  Its purpose is to
reject a locally richer but useless coordinate change quickly, or authorize a
preregistered 50M-token replication.

## Architecture

For each attention input norm and each FFN input norm independently, normalize
without learned gain, split `z=(a,b)`, and bound coefficients by

```text
alpha = 0.25 * tanh(theta_alpha / 0.25)
beta  = 0.25 * tanh(theta_beta  / 0.25)
b' = b + alpha * abs(a)
a' = a + beta * abs(b')
```

The two `d/2` theta vectors exactly replace the removed `d` RMS gain values.
All coefficients initialize to zero, so the candidate begins at the ordinary
gain-one model.  Attention Q/K/V share one transformed norm output; FFN gate/up
share the other.  Final RMSNorm is untouched because its gain cannot be freely
folded into a tied embedding/LM head.

## Frozen arms

1. `raw_baseline`: ordinary learned RMSNorm gains.
2. `folded_null`: gain-free normalization with the same alpha/beta parameter
   tensors, ignored at runtime.  This controls the gain-removal optimizer chart.
3. `self_hinge`: same `d` parameter slots implement
   `z + gamma*abs(z)`, with `|gamma|<0.25`.
4. `one_way`: only `b'=b+alpha*abs(a)` is active; beta is a null slot.
5. `two_way`: the complete candidate.

All arms receive identical seeded matrix initialization.  No arm changes a
matrix dimension, attention head, KV group, FFN width, vocabulary, or cache.

## Frozen protocol

- scratch Qwen-style causal LM: 12 layers, hidden 384, SwiGLU width 1,024,
  six query heads, two KV heads, head dimension 64;
- existing immutable FineWeb-Edu token stream and validation file;
- seed 809; H100; Torch 2.5.1+cu124; CUDA 12.4; Transformers 4.57.6;
- sequence 512; microbatch 32; gradient accumulation 2;
- 305 steps = 9,994,240 prediction tokens per arm;
- evaluation: 128 batches of 32 at steps 0 and 305;
- AdamW, LR 3e-4, betas (0.9, 0.95), weight decay 0.1, 30-step warmup,
  cosine decay, gradient clip 1.0;
- one seed, no checkpoint selection, arm-specific LR, or post-result bound.

## Frozen decision

The two-way arm advances only if:

1. all arms have identical raw/trainable parameter counts and finite training;
2. all initial validation losses match within `2e-5`;
3. two-way terminal NLL is at least 0.025% below raw baseline and at least
   0.01% below the best of folded-null, self-hinge, and one-way;
4. paired per-validation-batch 95% intervals favor two-way against raw and the
   best nonlinear control;
5. mean absolute alpha+beta is at least 0.002, maximum absolute coefficient is
   below 0.25, and sampled chart condition number is below 2;
6. zeroing the learned coupling at terminal worsens NLL with a paired interval
   favoring the full candidate;
7. terminal normalized coordinate signs are not collapsed: positive fraction
   remains in `[0.1,0.9]`.

One-way, self-hinge, or null winning does not validate two-way coupling.  It
does identify which exact subclaim failed and determines whether a narrower
successor is algebraically justified.  Quantized weight serving remains out of
scope even if this BF16 screen passes.

