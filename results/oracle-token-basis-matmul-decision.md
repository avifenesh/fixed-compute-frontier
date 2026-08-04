# Oracle token-basis matmul — fatal-gate decision

Status: **closed before training or kernel work**  
Date: 2026-07-30

## Decision

Do not pursue cross-token basis reuse as a replacement for QKV or FFN-up
matrix multiplication under a low-error language-model claim.  The branch
fails even when given the best rank-`r` basis of the *already computed output*,
an impossible circular oracle stronger than any executable basis finder.

For an input block `X` with `T` tokens and width `D`, a factorization

\[
X = C B, \qquad XW = C(BW)
\]

has optimistic MAC ratio

\[
\rho = \frac{r}{T} + \frac{r}{D}
\]

relative to `XW`, before charging for basis discovery, SVD, reconstruction,
extra reads, or poor kernel shape.  A useful two-times gate therefore requires
`rho <= 0.5` at the target error.

Across three pinned SmolLM2 checkpoints, three layer depths, QKV and FFN-up,
and sixteen held-out WikiText sequences per cell, every cell failed at 1%
relative Frobenius error.  Required ranks were 89%-98% of the 256 tokens.

## Strongest impossible-oracle result

Median ideal cost ratios at 1% error for the SVD basis of the output `Y=XW`
itself were:

| model | QKV, three depths | FFN-up, three depths |
|---|---|---|
| SmolLM2-135M | 1.292, 1.320, 1.337 | 1.382, 1.405, 1.394 |
| SmolLM2-360M | 1.168, 1.188, 1.202 | 1.242, 1.237, 1.232 |
| SmolLM2-1.7B | 1.059, 1.085, 1.090 | 1.103, 1.103, 1.107 |

`1.0` is the original matmul cost and `0.5` is the frozen admission gate.
Thus even the circular oracle is 5.9%-40.5% more expensive than the original
matmul and 2.12x-2.81x over the admission ceiling.  The executable input-SVD
basis was similar or worse.

The apparent improvement with model width is not a route to savings.  It
comes from shrinking the second term `r/D`; the measured first term `r/T`
approaches one because the token axis remains almost full rank at 1% error.
The ideal ratio therefore asymptotes near one, before real overhead.

## What this proves

Low entropy or low effective rank of individual hidden vectors does not imply
low algebraic rank of a token block at the accuracy required to preserve a
projection.  Natural text tokens span different directions, and the exact
QKV/FFN outputs retain almost all of those token directions.

No learned basis finder, cached basis, fused reconstruction kernel, longer
training, or larger model can beat the best rank-`r` approximation of the
finished output for the same `r`.  Recovering this branch would require
changing the representation during training so that token blocks become
low-rank; that is a different architecture and must pay for any lost
token-specific information.

## Integrity

- source SHA-256: `f49a1f84481cfcb6894355f1d945f6426bc62f10506b34e5bdf8c2ce04c4a310`
- preregistration SHA-256: `f5ed05ec0676d9a2d938bc22fef659adbe18a3d71706b5d998a41a8ad021c591`
- result SHA-256: `fe80c99ee09381451b25b53cbbaa58a5409e34d4c8e5c4e48c989d257e644553`
- focused tests: 3 passed on the dedicated G7 instance

Artifacts:

- `experiments/oracle_token_basis_matmul.py`
- `tests/test_oracle_token_basis_matmul.py`
- `results/oracle-token-basis-matmul-preregistration.md`
- `results/oracle-token-basis-matmul.json`
- `results/oracle-token-basis-matmul.log`

## Retained boundary

Do not reopen activation-side token-rank compression for ordinary pretrained
hidden states.  Re-admit only a representation whose *task-relevant output*
is provably constrained to a shared subspace by construction, with the cost
and capability loss of that constraint counted explicitly.
