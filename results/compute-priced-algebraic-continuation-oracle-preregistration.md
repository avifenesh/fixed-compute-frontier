# CPAC gradient oracle: frozen preregistration

Date frozen: 2026-07-30

## Scope

This is a fatal upper-bound screen for compute-priced algebraic continuation
(CPAC). It asks whether exact from-zero language-model gradients contain enough
mixed algebraic structure to justify implementing a training method. It is not
a quality experiment and cannot establish a breakthrough.

No candidate gradient measurement was run before this protocol and its hashes
were frozen. Unit/algebra tests and a model-shape smoke test were run first.

Frozen artifacts:

- source: `experiments/compute_priced_algebraic_continuation_oracle.py`
  SHA-256 `19c8c2ca53fa0f4359d3d06193ebf45ce76034a8214e2c2383340183a4530f71`
- tests: `tests/test_compute_priced_algebraic_continuation_oracle.py`
  SHA-256 `e187bf2168324bbd5745c28379f14ba102b323ddcb8e3ddf5ba9600fb5fca2bf`
- hypothesis: `results/compute-priced-algebraic-continuation-hypothesis.md`
  SHA-256 `93fbbcdfee69ce0543f663ddd97bffe21c5fdb1c0d0db3c730fdaec605d59dd5`

Pre-run checks: six focused tests passed; the width-192 model exposed exactly
21 expected sampled matrices and 11,798,976 parameters.

## Fixed from-zero models

- Llama causal architecture, vocabulary 49,152, tied embedding/unembedding.
- Widths 192, 384, and 768.
- Six layers; head dimension 64; GQA with one KV head per three Q heads.
- SwiGLU intermediate width `8d/3`.
- Sequence length 256; training batch size 4.
- AdamW, learning rate 0.0003, betas (0.9, 0.95), weight decay 0.1.
- 32-step warmup, cosine decay, gradient norm cap 1.
- 512 steps from random initialization.
- Gradient checkpoints at steps 0, 64, and 512.
- Exact measurement batch: 16 sequences = 4,096 prediction positions, chosen
  so the widest sampled gradient is not rank-capped by token count.
- Seed 260730 plus width.
- Fixed uint16 FineWeb-Edu token streams already present on the dedicated G7.

At each checkpoint, measure Q/K/V/O and gate/up/down matrices in layers 0, 3,
and 5: 21 exact dense gradients per cell, 189 learned-gradient matrices total.

## Frozen oracle families

For each matrix, evaluate:

- dense fallback;
- truncated SVD ranks 1, 2, 4, 8, 16, 32, 64, 128, 256, 384 where valid;
- top-energy 16x16 block supports at scalar fractions 1%, 2.5%, 5%, 10%,
  20%, 35%, 50%, 75%, and 90%;
- low-rank plus top-block residual for ranks up to 128 and residual fractions
  1%, 2.5%, 5%, 10%, 20%, and 35%;
- balanced Kronecker ranks 1, 2, 4, and 8.

For every family, charge ideal factor-application multiplications. No kernel
speedup is assumed. Select the least-cost direction with squared cosine to the
exact gradient of at least 0.90; use dense fallback when needed.

Size-matched isotropic Gaussian gradients are analyzed for square, KV, FFN-up,
and FFN-down shapes at all three widths.

## Frozen decision

Advance to an executable optimizer only if every gate passes:

1. Mixed structured cost is at most 35% of dense matrix cost in at least seven
   of nine width/checkpoint cells.
2. Mixed structured cost is at least 25% below low-rank-only cost in at least
   six cells. Otherwise the method reduces to prior incremental-rank work.
3. Mixed cost is non-increasing from width 192 to 384 to 768, with a strict
   192-to-768 decrease, at every checkpoint.
4. The aggregate isotropic control needs at least 85% of dense matrix cost.
5. At width 768, every checkpoint projects to at least 2.0x useful descent per
   total frontier step after charging fixed attention/logit work and one exact
   dense refresh every 64 steps.
6. All three dense models reduce their measurement loss from step 0 to 512.

The 2.0x oracle threshold deliberately leaves room for factor discovery,
staleness, memory traffic, imperfect optimization, and non-ideal kernels before
the final 1.5x project bar.

## Interpretation fixed in advance

- Failure closes this CPAC formulation; do not rescue it with more atom types.
- A low-rank-only pass is prior-art confirmation, not a new candidate.
- A pass only licenses implementation of the actual adaptive optimizer and
  structured training forward. It does not license a novelty or quality claim.
- Memory savings, a single easy matrix, or initialization-only concentration do
  not count.
