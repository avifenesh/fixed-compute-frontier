# PAEFB exact-error oracle: frozen preregistration

Date frozen: 2026-07-30

## Scope

This is an impossible-oracle upper screen for phase-adaptive error-factored
backpropagation (PAEFB). It measures whether exact token-by-channel error
matrices could make both linear backward products cheap while the forward and
deployed model remain exactly dense. It is not an executable algorithm or a
model-quality test.

No candidate error matrix was captured before freezing this protocol. Four
focused algebra tests passed.

Frozen artifacts:

- source: `experiments/phase_adaptive_error_factored_backprop_oracle.py`
  SHA-256 `d90170fe9d2a5c0c6f1536037ca18cb1bbd08143ae4bb721ead672ca279ef7eb`
- tests: `tests/test_phase_adaptive_error_factored_backprop_oracle.py`
  SHA-256 `6c0216389cc1b04b966ef6e6a79c0bc67d7cdf8faa826d118bb03b44e787c216`
- shared model/data dependency:
  `experiments/compute_priced_algebraic_continuation_oracle.py`
  SHA-256 `19c8c2ca53fa0f4359d3d06193ebf45ce76034a8214e2c2383340183a4530f71`
- hypothesis: `results/phase-adaptive-error-factored-backprop-hypothesis.md`
  SHA-256 `e4af3decbebc253fa6570bae046c7ae9c7163c474fb56a30b22861f16dc717e6`

## Fixed from-zero protocol

- Dense Llama models at widths 192, 384, and 768.
- Six layers; head dimension 64; one KV head per three Q heads; SwiGLU width
  `8d/3`; vocabulary 49,152; tied embedding/unembedding.
- 512 dense AdamW steps from random initialization.
- Sequence length 256; training batch 4; exact measurement batch 16 = 4,096
  prediction positions.
- AdamW learning rate 0.0003, betas (0.9, 0.95), weight decay 0.1, 32-step
  warmup, cosine decay, gradient norm cap 1.
- Checkpoints 0, 16, 64, 256, and 512.
- Seed base 270730 plus width.
- Fixed existing FineWeb-Edu uint16 token streams.

At each checkpoint capture exact linear input `X`, arriving error `E`, and
weight `W` for Q/K/V/O and gate/up/down maps in layers 0, 3, and 5: 21 matrices
per cell, 315 learned-error matrices total.

## Frozen oracle and accounting

For each matrix, compute exact

- `dX = E W`;
- `dW = E^T X`.

Compute the exact SVD of `E`, search ranks 1, 2, 4, ..., 2048 with binary
refinement, and select the smallest rank whose concatenated approximate
`(dX,dW)` direction has squared cosine at least 0.99 to exact `(dX,dW)`.

Charge:

- factored `dX` and `dW`;
- a two-pass randomized range-finder equivalent cost `2 N m r`;
- dense fallback whenever total charged ratio reaches one.

The charged backward ratio is `min(1, r/N + r/m + r/n)`. QR, oversampling,
memory traffic, and nonideal kernels are still excluded, so this remains an
optimistic upper bound.

Analyze size-matched isotropic Gaussian `E`, `X`, and `W` for square, KV,
FFN-up, and FFN-down shapes at all widths.

## Frozen decision

Advance to recursive compressed-backward simulation only if every gate passes:

1. At step 64, aggregate charged backward ratios are at most 20%, 20%, and 8%
   for widths 192, 384, and 768 respectively.
2. At steps 64, 256, and 512, charged cost strictly decreases as width grows.
3. Every width/checkpoint aggregate retains energy-weighted squared cosine at
   least 0.99.
4. Aggregate isotropic control requires at least 85% of dense backward cost.
5. At width 768, projected whole-step useful descent per compute is at least
   2.0x at steps 64, 256, and 512 after charging exact forward and all fixed
   attention/logit work.
6. All dense models lower measurement loss from step 0 to 512 and every exact
   sampled linear gradient has nonzero finite energy.

## Fixed interpretation

- A failed gate closes this formulation without lowering the 0.99 direction
  threshold or deleting range-finder cost.
- A pass does not prove recursive stability, real kernel speed, model quality,
  or novelty.
- Because low-rank backprop and lossy backward have prior art, only a later
  from-zero dense-LLM capability/compute result could establish an edge.
- The discarded 1% tail must later receive rare-token and compositional tests;
  NLL alone cannot certify it harmless.
