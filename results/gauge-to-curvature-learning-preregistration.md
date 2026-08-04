# Gauge-to-curvature covariance-cache gate — preregistration

Status: frozen before observing results  
Date: 2026-07-26  
Stage: oracle-routing CPU mechanism gate

## Question

Can an equal-ledger nonlinear value writer preserve a useful second-order
statistic in an `r`-wide KV cache when linear values provably lose it?

This is a family test, not a monomial-regression demonstration.  Each example
is a bag of zero-mean tokens whose class is encoded only in covariance.

## Data

Use `r=16`.  For a class `y` in `{-1,+1}`:

\[
\Sigma_y=R\,\mathrm{diag}(1+0.15ys)R^\top,
\qquad s=(1^8,-1^8).
\]

`R` is a Haar-random orthogonal matrix, fixed within a world and independently
drawn for five worlds.  Draw eight independent `z ~ N(0, Sigma_y)` and emit
both `z` and `-z`, shuffled.  Uniform oracle attention over the bag therefore
has **exactly zero linear mean for every example**, while its covariance is
class-informative.  Training and test examples are independent.

The primary condition is randomly rotated.  An axis-aligned condition exposes
the missing-square weakness of off-diagonal gates.  A mean-shift condition is
the no-harm control for linear information.

## Equal-ledger writers

All candidate writers retain an `r`-wide cache and include the identity value
path.

1. `gfqv_offdiag`:
   `v = x + x * (x @ G)`, `diag(G)=0`.  Its scalar classifier spans all
   off-diagonal covariance terms.
2. `square_writer`:
   `v = x + (x @ H)^2`, `diag(H)=0`.  It learns `r` signed square features.
3. `bilinear_writer`:
   a rank-5 learned bilinear source writer, the strongest generic low-rank
   quadratic control within the reclaimed budget.

Both GFQV and the square writer use `r(r-1)` nonlinear weights,
`r(r-1)+r=r^2` multiplications, `r(r-1)` additions, and `r` cached scalars.
At zero gate they recover linear values.  The square writer must use small
nonzero initial `H`, because `H=0` has zero gradient through the square.

## Baselines and ceiling

- `linear_mean`: classifier on the attended linear mean.  It must be chance.
- `post_mean_degree2`: arbitrary quadratic features after aggregation.  It
  must also be chance because the exact mean is zero.
- `diagonal_moments`: classifier on raw coordinate variances.
- `full_degree2_oracle`: classifier on every unique covariance entry.  It is
  an unmatched mechanism ceiling.

All linear classifiers and nonlinear writers use the same train/validation
examples within a world.  Nonlinear writers receive the same optimizer steps
and validation selection.  Report accuracy and binary cross-entropy over five
worlds and three writer initializations where initialization matters.

## Frozen rejection rule

The harness is invalid if the full covariance ceiling is below 82% accuracy or
either linear/post-mean leakage baseline exceeds 52%.

Reject **GFQV as the preferred writer** unless it beats the strongest
equal-ledger source writer by at least 5% in relative excess NLL, with a paired
95% confidence interval excluding zero, and wins in at least four of five
rotated worlds.

If another equal-ledger writer wins, retain the broader mathematical result—
**gauge budget can be converted into cacheable curvature**—but kill the GFQV
coordinate-star choice.  No oracle-routing result, positive or negative,
admits candidate 004 or justifies GPU rental.  Promotion requires a learned
Q/K two-layer model with ordinary source MLPs, current G2/GLU controls, a fused
kernel, and low-bit KV-cache checks.
