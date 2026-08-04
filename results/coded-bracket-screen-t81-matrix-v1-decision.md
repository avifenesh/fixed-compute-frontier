# T81 coded-bracket screen v1 — local matrix decision

Date: 2026-08-02  
Status: **INDEPENDENTLY AUDITED DEVELOPMENT GATE FAILURE; EXECUTION LANE CLOSED**

## Outcome

The fixed-anchor Rademacher construction works as compressed sensing, but it
does not beat the strongest cheap baseline. Across the eight cells in which
both methods reached the frozen global-support criterion, the median ratio was

\[
\frac{\text{adaptive-pair identity loops}}
     {\text{star-code identity loops}}
=0.9053.
\]

A value above one favors the star code. The star code therefore used about
`10.5%` more identity loops at the median. It reached `2x` in zero cells, and
lost by more than 20% in three cells. In the ninth cell (`d=4`, `sigma=0.10`),
adaptive pairwise passed with `419.41` mean loops while star coding never
reached the preregistered recovery criterion within its frozen range.

| degree | noise | adaptive pair loops | star loops | pair/star |
|---:|---:|---:|---:|---:|
| 1 | 0.00 | 262.80 | 360 | 0.730 |
| 1 | 0.05 | 259.73 | 325 | 0.799 |
| 1 | 0.10 | 263.38 | 360 | 0.732 |
| 2 | 0.00 | 348.02 | 360 | 0.967 |
| 2 | 0.05 | 351.38 | 376 | 0.935 |
| 2 | 0.10 | 352.17 | 343 | 1.027 |
| 4 | 0.00 | 422.86 | 468 | 0.904 |
| 4 | 0.05 | 417.22 | 460 | 0.907 |
| 4 | 0.10 | 419.41 | no pass | no pass |

The only cell favoring star coding did so by `2.7%`, which is explicitly a
rejected single-digit gain.

The saved summary incorrectly excluded the ninth no-pass cell from its ratio
denominator. The
[independent audit](coded-bracket-screen-t81-matrix-v1-independent-audit.md)
found that counting it correctly changes the decision to zero of nine cells at
`2x`, three numeric losses over 20% plus one stronger no-phase failure, and a
conservative median near `0.904`. The bug favored the candidate; it cannot
reverse closure.

## Why the weaker comparison looked promising

Against exhaustive all-pairs testing (`496` loops), the selected star code used
between `325` and `468` loops in passing cells. That is up to a `1.53x` identity
count improvement. Exhaustive testing is the wrong production control: once
the supplied degree tells a pairwise learner that it has found every incident
interaction, it can stop. The adaptive known-degree pairwise control used only
about `260` to `423` loops and erased the apparent gain.

This baseline has an analytic noiseless check. If a star has `n_p` possible
partners and `d_p` uniformly placed interactions, the expected position of the
last positive in a random order is

\[
\mathbb E[M_p]=\frac{d_p(n_p+1)}{d_p+1}.
\]

Summing over the frozen `k=32` stars predicts `263.5`, `351.0`, and `420.4`
loops for degrees 1, 2, and 4. The observed noiseless means were `262.80`,
`348.02`, and `422.86`, confirming the implementation and explaining the
finite-size collision.

## Energy and active-channel ledger

Unnormalized star coding spent roughly `5,472` to `6,272` quadratic-energy
units in passing cells, versus `519` to `846` for adaptive pairwise. Its batch
count was not bought at equal actuation cost.

Equal-energy normalization also failed to dominate. Where it passed, it needed
`686` to `936` energy units versus `519` to `834` for adaptive pairwise, and it
failed to pass two degree-4 cells. This matches the paper analysis: spreading a
fixed-energy intervention across `n_p` partners attenuates each bracket by
`1/sqrt(n_p)` and lets endpoint noise consume the nominal coding gain.

## Integrity and scope

- `k=32`, 64 frozen seeds, degrees `{1,2,4}`, noise `{0,0.05,0.10}`;
- exact global support, not average edge accuracy, was the primary event;
- passing required point estimate at least `0.95` and Wilson 95% lower bound at
  least `0.90`;
- selected star cells had 63/64 or 64/64 exact recoveries and pair F1 above
  `0.9994`;
- a pre-run smoke visibly reconstructed the noiseless full-budget cells, but
  this dedicated integrity output was not preserved in the result JSON; and
- the JSON SHA-256 is
  `0f560fbf82359a2cb4d2ac6ec834f314219212f6b7d44b3df2b5317c3ab6a335`.

This was a matrix test of sparse order-sensitivity screening, not a world
model, grounding experiment, or intelligence benchmark.

## Frozen decision

The preregistration said to stop if median identity gain was below 20% before
dense diagnostics. It is negative, not merely below 20%: `0.905x` rather than
the required `>=1.2x`, and far from the preferred `>=2x`.

Therefore:

1. do not run the dense-wedge/generic diagnostic stage;
2. do not run `k=64` confirmation;
3. do not build a neural or raw-observation version;
4. do not use local GPU or rent GPU capacity; and
5. retain Proposition T81.3 only as a correct idealized theorem whose practical
   advantage disappears against adaptive pair testing in this frozen family.

The independent code/result audit verified the Wilson arithmetic, complete
support metric, adaptive baseline, cost ledgers, and seed separation. It found
no bug capable of reversing the negative result. It also notes that swapping
SOMP or changing the code design now would be a new hypothesis, not a repair of
this closed execution lane.
