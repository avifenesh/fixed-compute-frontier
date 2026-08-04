# Peak/confidence-lifted attention — algebra retained, causal mechanism rejected

Status: **closed before language training and production-kernel work**  
Date: 2026-07-27

## Outcome

Peak lift is a real algebraic extension, but it is not a viable fixed-chunk
attention mechanism. The formal Stage-0 screen passed all 13 gates: it exactly
contained ordinary attention, preserved the ordinary merge algebra over
finalized chunks, left within-chunk ratios unchanged, broke cross-chunk IIA,
and raised sampled functional rank from 27 to 28 while global temperature
remained at 27. The independently audited result is
`results/peak-lifted-attention-stage0.json` (SHA-256
`350e76553524891819a527044e0b4de81f9097510db709f1be8d7aa231c7045c`).

The next frozen causal control rejected the mechanism. Its independently
audited result is `results/peak-lifted-attention-causal-control.json` (SHA-256
`8091612564cc9e2e373e0d02021c7005864284becc7adf7adf95ac59a9238b78`).
All six admission gates failed and `live_variant` is null.

## Why it failed

Under neutral IID scores, a chunk with more tokens has more chances to produce
a large maximum. The five-standard-error worst chunk-mass distortions were:

| Arm | Worst neutral distortion | Frozen limit |
|---|---:|---:|
| Raw peak | 54.94% | 5% |
| Normal-calibrated peak oracle | 218.69% | 5%, or 1% on the constant control |
| Existing-state confidence lift | 41.90% | 5% |

The normal oracle shows why subtracting an expected maximum is not a repair:
attention exponentiates the lift, so an unbiased lift is not an unbiased routed
mass. On the all-equal control, raw peak and confidence were exactly neutral,
while the normal correction itself introduced a 218.69% distortion.

Peak lift did create the intended anchor-to-payload signal when both tokens
shared a chunk: payload probability rose by roughly 1.33x to 1.44x. The same
mechanism became harmful at a boundary: the minimum ratio fell to 0.587-0.640.
Confidence produced only 1.0235x best mean uplift and a 0.9719 crossing-boundary
minimum. Query-relative chunks did not repair neutral fairness.

## Retained result

Retain the algebraic fact that discarded online-softmax state can fund a new
function direction. Reject fixed raw-peak, calibrated-peak, and confidence
chunk lifts as architectures. Do not run an LM screen or patch FlashAttention
for these operators.

The closure yields a sharper theorem: universal cardinality-neutral anchor
routing requires a statistic not present in ordinary `(m,l,o,n)` state. That
result and the exact price are recorded in
`results/chunk-router-cardinality-theorem.md`.
