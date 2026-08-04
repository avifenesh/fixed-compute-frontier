# Static heterogeneous SwiGLU: 50M-token five-seed preregistration

## Status

Frozen before any run at the five seeds below.  The earlier 10M-token ratio
sweep is development evidence and is not part of the confirmatory sample.

## Claim under test

At identical learned-scalar count, optimizer-state size, dense projection
shapes, and dense projection MACs, replace exactly half of each SwiGLU layer's
1,024 gate activations with

\[
  0.6854475404927584\,g\tanh(g),
\]

leaving the other half as `SiLU(g)`.  The multiplier matches the two gate
families' Gaussian-reference RMS.  The candidate changes scalar nonlinear
features but adds no matrix, routing, learned coefficient, or hidden width.

This screen tests language-model quality only.  Equal H100 served latency is a
separate mandatory gate; the eager reference computes both nonlinearities and
is not a serving kernel.

## Frozen model and data

- scratch decoder from the repository's pinned SmolLM2 configuration;
- hidden width 384, 12 layers, 6 query heads, 2 KV heads, head dimension 64;
- SwiGLU intermediate width 1,024;
- sequence length 512;
- existing `block-algebra-scratch` train/validation token files and manifest;
- prediction tokens per arm and seed: 49,971,200.

## Frozen arms

Run in this order for every seed:

1. `raw_baseline`: ordinary SwiGLU, with the same 32 harmless chart-pivot
   initial values used by all arms;
2. `canonical_null`: the packed/fixed-pivot parameterization with every gate
   forced to ordinary SiLU;
3. `mixed50_rms`: first 16 groups of 32 channels use the RMS-matched even gate,
   remaining 16 groups use SiLU.

Channel permutation symmetry makes the identity of a fixed 16-group subset
irrelevant to the represented function class.  The null arm isolates the
fixed-chart optimizer geometry.

## Frozen seeds and optimizer

- seeds: `2741, 2753, 2767, 2789, 2801`;
- 1,525 optimizer steps;
- microbatch 32, gradient accumulation 2;
- AdamW, learning rate `3e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`;
- weight decay `0.1`, gradient clipping `1.0`;
- 100 warmup steps and the repository's cosine schedule;
- evaluations at steps 0, 305, and 1,525;
- 128 validation batches of 32 sequences per evaluation.

## Integrity and no-shortcut gates

- exact source, preregistration, imported ratio-source, algebra-source,
  training-loop source, evaluation/data source, scratch-config source, test,
  and data-manifest hashes must match the integrity manifest;
- runtime must be H100, PyTorch `2.5.1+cu124`, CUDA `12.4`, and Transformers
  `4.57.6`;
- all arms must have identical learned-parameter counts, parameter-tensor
  counts, and post-step optimizer-state bytes;
- initial raw and canonical-null NLL must match within `1e-7` for every seed;
- candidate initial FFN activation RMS must match raw within `1%` for every
  seed (candidate logits need not match because its feature family is live);
- all losses, gradients, and diagnostics must remain finite;
- canonical-null mean terminal NLL may not be worse than raw by more than
  `0.05%`.

## Primary quality gates

All are required:

1. `mixed50_rms` beats the better of raw and canonical-null terminal NLL in at
   least four of five seeds;
2. its mean relative terminal improvement over that per-seed better control is
   at least `0.10%`;
3. the two-sided 95% Student-t interval over the five seed-level relative
   improvements has a lower bound above zero;
4. it beats the better control at the 10M-token checkpoint in at least four of
   five seeds;
5. within each winning terminal seed, the paired validation-batch interval
   favors the candidate; at least four of five seeds must pass this check.

Failure closes this exact 50/50 RMS-matched recipe.  No ratio, multiplier,
seed, learning-rate, or threshold will be changed after observing these runs.

## What a pass does and does not establish

A pass establishes a replicated capability-per-matrix-compute improvement for
this tiny scratch LM.  It does not establish novelty, scale transfer, equal
wall-clock training cost, or equal served latency.  Those require a current
prior-art audit, a larger-model replication, and a fused static split-activation
H100 benchmark.
