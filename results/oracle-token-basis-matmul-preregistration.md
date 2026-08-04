# Oracle token-basis matmul — fatal gate

Status: **frozen before activation capture**  
Date: 2026-07-30  
GPU: dedicated AWS G7e

## Status-quo assumption under attack

Projection kernels apply the same static matrix to every token independently:

\[
Y=XW,\qquad X\in\mathbb R^{T\times D}.
\]

If the token rows share a rank-`r` basis, `X = C B`, contraction order can be
changed:

\[
XW = C(BW).
\]

The expensive learned matrix is then applied to `r` basis tokens instead of
`T` ordinary tokens.  This attacks QKV and FFN input projections without
structuring `W` and without selecting weight experts.

Ignoring the cost of finding the basis, the dense-MAC ratio is

\[
\rho(r)=\frac{rDM+TrM}{TDM}=\frac rT+\frac rD.
\]

A genuine twofold projection saving therefore requires `rho <= 0.5`, not
merely `r/T <= 0.5`.  Runtime basis discovery, coefficient production,
workspace, traffic, and numerical conversion make the real condition stricter.

## Frozen upper-bound test

Use the same pinned three SmolLM2 widths and WikiText revision as the preceding
locality gate.  Evaluate 16 separate validation sequences of length `T=256`.
At the adjacent layers nearest 25%, 50%, and 75% depth, capture:

1. normalized input to the combined Q/K/V projections;
2. normalized input to the combined gate/up FFN projections.

For each sequence and projection family, compute two rank curves:

- **input-basis:** take the SVD of `X`, then measure the actual projection
  error after keeping the first `r` left singular directions;
- **output oracle:** take the SVD of the already-computed `Y=XW`.  This is
  circular and undeployable, but it is the strongest possible rank-`r` upper
  bound.  If it fails, no basis compiler can rescue the proposal.

Report the minimum rank for 1%, 5%, and 10% relative Frobenius error and its
ideal `rho(r)`.  No causal-loss claim is inferred from Frobenius error.

## Survival gate

At 1% relative output error, every model, depth, projection family, and
validation sequence distribution must satisfy:

1. median **output-oracle** `rho <= 0.5`;
2. p90 output-oracle `rho <= 0.5`;
3. median **input-basis** `rho <= 0.5`;
4. p90 input-basis `rho <= 0.5`;
5. neither rank fraction nor ideal cost ratio worsens with model width at any
   matched depth/family.

These gates demand at least a theoretical 2x saving before basis-discovery
cost.  A sub-percent speed or quality effect is irrelevant.

Failure closes cross-token low-rank contraction as the matmul replacement.
Passing only admits research on a cheap causal basis compiler and an
end-to-end quality test; it is not itself a serving or capability result.

