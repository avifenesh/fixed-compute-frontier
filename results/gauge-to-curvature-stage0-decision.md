# Gauge-to-curvature Stage 0 — GFQV rejected, cyclic mask survives

Status: **retain one mathematical pre-candidate; do not admit candidate 004**  
Date: 2026-07-26  
GPU: **not rented**

## Outcome

The zero-diagonal GFQV design is no longer the candidate.  It preserved useful
second moments, but its missing square terms were a real structural weakness.
An exact-same-ledger square writer tied or beat it, and an axis-aligned control
made the failure decisive.

The broader result survives and is now sharper:

> A value/output basis redundancy can be converted into identifiable,
> pre-cache quadratic features without increasing the logical value-projection
> arithmetic, value-cache width, or attention scan.

The best current specialization replaces the zero-diagonal mask with the
complement of a directed cycle.  That small discrete change preserves every
square and at least one orientation of every cross term while retaining the
same exact ledger and the same one-accumulator Hopper schedule.

## The retained construction

For one value head, use the exact value/output basis chart from a full-rank
minor of `W_V`.  With fixed contiguous pivot coordinates `z`:

\[
u=z+x_{\bar P}B.
\]

Choose an `r`-cycle

\[
\sigma(j)=j+1\pmod r,
\]

and forbid exactly one gate entry per output:

\[
H_{\sigma(j),j}=0.
\]

The new cached value is

\[
\boxed{
v_j=u_j+z_j\sum_{k\ne\sigma(j)}z_kH_{kj}
}.
\]

`H=0` exactly recovers the gauge-fixed linear value map.  This additive form is
intentional.  The seemingly broader multiplicative form
`u * (1 + zH)` needs two live GPU accumulators and introduces tied
non-pivot/pivot quadratic terms when `D>r`.

## Exact resource ledger

For one dense `D -> r` value projection:

| Quantity | Dense value | Cyclic-mask value |
|---|---:|---:|
| Learned value scalars | `Dr` | `(D-r)r+r(r-1)=Dr-r` |
| Scalar multiplications | `Dr` | `(D-r)r+r(r-1)+r=Dr` |
| Scalar additions | `(D-1)r` | `(D-r)r+r(r-2)+r=(D-1)r` |
| Cached value scalars/token | `r` | `r` |
| Attention scan | unchanged | unchanged |

The `r` missing learned scales can be added without extra arithmetic, but they
are absorbable into `W_O`; they add no functional degrees.  Leaving them out
is the cleaner parameterization.

At `D=4096,r=128`, both logical value maps use 524,288 multiplications and
524,160 additions.  The cyclic map contains 16,256 learned quadratic gate
coefficients and still stores 128 cached value scalars.

## What is mathematically gained

A dense full-rank `W_V W_O` factorization stores `2Dr` scalars but has only

\[
2Dr-r^2
\]

functional degrees: `GL(r)` change of basis is pure gauge.  After fixing the
identity chart, the cyclic candidate has

\[
(D-r)r+rD+r(r-1)=2Dr-r
\]

parameters.  For a full-row-rank output block `O`, its linear Jacobian at the
origin identifies `O` and `B`, and its vector-valued quadratic coefficients
identify all `r(r-1)` entries of `H`.  Thus, on the generic chart, it converts

\[
\boxed{r^2-r}
\]

formerly redundant directions into functional nonlinear directions while
using `r` fewer raw parameters than dense V/O.

This is a local attention-sublayer statement, not a proof that an entire
Transformer cannot synthesize the same products in its MLP.

### Why the cycle covers every scalar quadratic monomial

For `r>=3`:

- `z_j^2` is present because an `r`-cycle has no fixed point, so
  `sigma(j) != j` and `H_jj` is allowed.
- A cross term `z_a z_b` would disappear only if both directed coefficients
  were forbidden: `sigma(a)=b` and `sigma(b)=a`.  That is a two-cycle, which a
  single `r`-cycle does not contain.

Therefore an all-nonzero scalar readout can span every symmetric quadratic
form in the pivot coordinates.  The scalar map has only `r(r+1)/2`
identifiable combinations; for `r>3`, its kernel has dimension `r(r-3)/2`.
The full `r(r-1)` identifiability claim requires vector output and independent
rows of `O`.

The construction fails at `r=2`, where the cycle is a two-cycle.  Normal LLM
head widths are far above this boundary.

## Natural covariance-cache gate

The CPU test did not regress a hand-selected monomial.  Each example contained
eight Gaussian samples from a class-dependent covariance, emitted as shuffled
`z,-z` pairs.  The attended linear mean was exactly zero, while the raw second
moment identified the class.  Five independent Haar rotations prevented a
fixed coordinate alignment.

Median test results across five rotated worlds:

| Writer | Accuracy | Binary NLL |
|---|---:|---:|
| Linear mean | 50.00% | 0.69315 |
| Post-mean quadratic | 50.00% | 0.69315 |
| Generic rank-5 bilinear | 81.42% | 0.41768 |
| Zero-diagonal GFQV | 85.50% | 0.34024 |
| Exact-ledger square writer | 86.04% | 0.34008 |
| Full quadratic ceiling | 87.13% | 0.30010 |

The harness passed both validity checks: every full-covariance ceiling exceeded
82%, and the linear/post-mean leakage baselines stayed at 50%.

GFQV won only three of five worlds, below the frozen four-world threshold.  Its
paired relative excess-NLL 95% interval was `[-0.218, 0.678]`, which includes
zero.  It therefore failed the preregistered preference rule.

The structural control was stronger:

| Axis-aligned covariance | Accuracy | Binary NLL |
|---|---:|---:|
| Zero-diagonal GFQV | 49.29% | 0.69629 |
| Square writer | 85.06% | 0.35468 |
| Full quadratic ceiling | 87.28% | 0.30009 |

All paths reached 100% accuracy on the mean-shift control, so the identity
linear path was preserved.

The cyclic mask was derived after this frozen comparison.  Its executable
algebra gate factors an arbitrary scalar quadratic to `1.78e-15` maximum error
and proves support coverage exactly.  On the zero-mean covariance task it can
represent the full quadratic decision statistic algebraically; this has not
yet been trained inside a learned-attention network.

## Hardware truth

Logical equality is not GPU equality.  Hopper tensor cores still execute the
one isolated zero per column inside a dense `r x r` tile, then perform `r`
scalar FMAs.  The viable fused schedule is nevertheless unchanged from the
earlier GFQV design:

1. WGMMA the `zH` slice into one FP32 accumulator.
2. Drain once and apply `Z = fma(z, Z, z)`.
3. Continue WGMMA over `B`, accumulating `x_barP B` into the same `Z`.

This requires one launch, one accumulator, and no HBM intermediate.  Stock
dense GEMM cannot express the mid-mainloop transform.  Cache width and
FlashAttention work remain unchanged.  The diagonal terms can shift cache
means and worsen tails, so RMS, kurtosis, saturation, and KV8/KV4 error are
mandatory gates.

## Prior-art boundary

The ingredients are occupied:

- [Basis Decomposition Attention](https://arxiv.org/abs/2510.01718) already
  supplies the exact contiguous value/output chart and fused projection kernel.
- [Beyond Linearity in Attention Projections](https://arxiv.org/abs/2603.13381)
  already reinvests an algebraically redundant query projection into an
  identity-anchored nonlinear residual at matched budget.
- [Gated Attention](https://arxiv.org/abs/2505.06708) G2 and
  [GLU Attention](https://arxiv.org/abs/2507.00022) already make values
  nonlinear before aggregation.
- [DCN-V2](https://arxiv.org/abs/2008.13535) and polynomial/bilinear networks
  already contain the underlying product algebra.

No exact collision was found through 2026-07-26 for the conjunction of a
gauge-fixed value chart, exact reclaimed arithmetic, a complement-of-cycle
mask, unchanged cache width, and attention placement.  The defensible label is
**possibly novel budget-neutral attention-value specialization**, not new
algebra or first nonlinear values.

## Decision and next gate

Kill zero-diagonal GFQV as the preferred design.  Retain the additive cyclic
mask as a mathematical pre-candidate because it has:

- exact baseline containment on the generic fixed chart;
- `r^2-r` new identifiable vector-valued functional directions;
- every scalar quadratic monomial in its pivot support;
- exact logical value arithmetic and unchanged cache width;
- a credible one-accumulator fused schedule.

Do not call it candidate 004 and do not rent a GPU for the mask alone.  The
next gate is a learned two-layer addressed-bag model with ordinary source MLPs
and learned Q/K.  It must compare cyclic mask, square writer, dense/BDA values,
G2, parameter-matched GLU values, and equal-budget source-MLP reinvestment.
Only a stable win there justifies the H100 kernel/LM round.

## Artifacts

- [Covariance preregistration](gauge-to-curvature-learning-preregistration.md)
- [Covariance experiment](../experiments/gauge_curvature_covariance_gate.py)
- [Covariance result](gauge-curvature-covariance-gate.json)
- [Cyclic-mask algebra gate](../experiments/gauge_curvature_permutation_mask.py)
- [Cyclic-mask result](gauge-curvature-permutation-mask-stage0.json)
- [Focused covariance tests](../tests/test_gauge_curvature_covariance_gate.py)
- [Focused cyclic-mask tests](../tests/test_gauge_curvature_permutation_mask.py)
