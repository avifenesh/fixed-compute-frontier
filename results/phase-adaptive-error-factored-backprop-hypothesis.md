# Phase-adaptive error-factored backpropagation (PAEFB)

Status: pre-candidate; exact-error oracle required  
Date: 2026-07-30

## One mechanism

Keep every forward operation, model weight, objective, and exported checkpoint
ordinary and dense. Change only the algebra used for linear-layer backward.

For a dense linear layer

`Y = X W^T`,

where `X` is `N x n`, `W` is `m x n`, and the arriving error is
`E = dL/dY` with shape `N x m`, exact backprop computes

`dX = E W` and `dW = E^T X`.

If `E` has a rank-`r` approximation `E_r = U V^T`, compute instead

`dX_r = U (V^T W)` and `dW_r = V (U^T X)`.

The dense forward is unchanged. The candidate starts with exact backprop and
switches a layer to factored error only when a causal rank monitor clears a
quality and compute threshold; it returns to dense immediately when the test
fails. No factor or monitor is exported.

## Compute ceiling

The two ordinary backward matmuls cost `2 N m n` multiply-like operations.
Given factors, the two factored gradients cost

`2 r m n + 2 N r n`,

for ideal ratio `r/N + r/m`.

A two-pass randomized range finder for `E` adds approximately `2 N m r`, so a
conservative logical ratio is

`c = r/N + r/m + r/n`.

If learned projections account for fraction `p` of a dense Transformer's
forward work, and only their backward is factored, the whole training-step
ratio is

`R = (1-p) + p(1 + 2c)/3`.

At a frontier-like `p ~= 0.816`, `c=0.08` gives `R ~= 0.500`, leaving a real
2x ceiling while retaining the exact dense forward and inference graph.

## Why the CPAC failure does not kill this

CPAC incorrectly needed a structured current weight to make the forward cheap;
a low-rank current update cannot erase a dense base. PAEFB never makes that
claim. It pays the dense forward and attacks only the two backward products.
The retained CPAC measurement—that exact weight gradients become sharply
low-rank after early learning—motivates measuring the upstream cause, not
assuming it.

## Prior-art boundary

Low-rank backpropagation via Walsh-Hadamard transforms has been studied for
Vision Transformer adaptation, and lossy sparse backward has been used in
sparse LLM pretraining. Activation/gradient compression and low-rank LLM
architectures are also active areas. Therefore “backprop can be low-rank” is
not a novelty claim.

The only potentially open claim is: **from-zero dense LLM pretraining has a
width-improving error-rank phase transition that can be detected online and
used to factor both backward matmuls, with dense escape, at enough retained
direction and charged speed to improve the same deployed dense model.**

## Fatal oracle before implementation

On the same width-192/384/768 six-layer from-zero Llama family, capture exact
`X`, `E`, and `W` for Q/K/V/O and all FFN linear maps in layers 0, 3, and 5 at
steps 0, 16, 64, 256, and 512. Use 4,096 prediction positions per measurement.

For increasing ranks, construct the exact truncated-SVD `E_r`, then directly
compare the concatenated `(dX_r, dW_r)` direction with exact `(dX, dW)`. Select
the cheapest rank reaching squared cosine 0.99 after charging the two-pass
factor finder. Use dense fallback otherwise. Include size-matched isotropic
error controls.

Advance only if:

1. by step 64, aggregate charged backward ratio is at most 8% at width 768 and
   at most 20% at the two smaller widths;
2. at steps 64, 256, and 512, required cost strictly decreases with width;
3. every selected aggregate direction has energy-weighted squared cosine at
   least 0.99;
4. isotropic errors require at least 85% of dense backward cost;
5. the width-768 whole-step projection is at least 2x at steps 64, 256, and
   512 after including factor discovery and all fixed work;
6. all dense models learn and no measured gradient is numerically degenerate.

This is still an impossible-oracle test: exact SVD sees the completed error.
A pass licenses only a recursive compressed-backward simulation and an
executable range finder. It is not a model-quality or novelty result.

## Hard failure modes

- Low-rank `dW` caused by activations rather than `E`.
- Important rare-token directions living in the discarded 1% error tail.
- Rank expansion through elementwise gates, attention, and branch summation.
- Range-finder and QR cost erasing the algebraic saving.
- GPU kernels losing to dense Tensor Core matmul despite fewer operations.
- Recursive approximation changing earlier-layer errors more than independent
  per-layer oracle measurements predict.
