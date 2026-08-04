# TVE remaining-gauge-budget screen

RQ gauge fixing creates 2,016 strict-lower zeros per 64-coordinate
representative KV-head block. The tested block-16 TVE uses only 480 of them;
1,536 cross-block zeros, or 76.19 percent of the available gauge budget, remain
unused. This screen asks whether one full-head triangular flow buys more quality
without changing parameter or cache size.

## Frozen exploratory protocol

- Model/data/training: the same 37.8M architecture, pinned FineWeb-Edu files,
  seed 14321, 305 steps, and 9,994,240 prediction tokens used by the prior TVE
  short screens.
- Arms:
  1. gauge-canonical control, encoding off;
  2. block-16 TVE with tau 0.125 (960 slots/layer);
  3. full-head block-64 TVE with tau 0.25 (4,032 slots/layer).
- Tau changes from 0.125 to 0.25 as a stability heuristic. The exact slot ratio
  is 4.2 and its square root is 2.04939, while per-row fan-in is nonuniform and
  tau also rescales gradients into the shared O weights. This arm therefore
  tests the joint block64/tau0.25 choice, not an isolated fan-in correction.
- All arms use identical initial random weights, parameter/state counts, data
  order, optimizer, and validation batches. No prior seed is pooled.

At step 305, report paired 64-batch normal intervals for both candidates versus
control and full-head versus block-16. Full-head is promoted only if it is
finite, coefficients remain BF16-nonzero and bounded by 0.5, its all-layer
serving bridge is bit-exact, it clears 0.025 percent NLL improvement versus
control, and its paired upper bound versus block-16 is below zero.

This is a development screen, not evidence for generality or served-cost
neutrality. Failure retains block-16 and closes only block64/tau0.25 at seed
14321 and 10M tokens. It does not close all full-head scalings or invalidate the
established block-16 effect.
