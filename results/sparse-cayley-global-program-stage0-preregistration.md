# Sparse Cayley global programs — Stage-0 preregistration

Status: **frozen before formal execution**  
Date: 2026-07-30

## Question

Can a bank of linearly sized sparse operators produce stable, dense,
full-direction feature transforms whose ordered compositions are much richer
than additive expert aggregation, without executing a dense matrix multiply?

This is an algebra gate only.  It does not claim language-model quality,
learned routing, GPU speed, or independent information in every composed
route.

## Construction

For even width `D`, construct each real skew-symmetric generator `A_e` from
three random perfect matchings with independent edge weights, then rescale it
to spectral norm `alpha = 0.25`.

Define its Cayley response

\[
Q_e=(I-A_e)(I+A_e)^{-1}.
\]

Because `A_e^T=-A_e`, exact `Q_e` is orthogonal.  To apply it without a dense
inverse, use

\[
z_T=\sum_{t=0}^{T}(-A_e)^t x,\qquad \widehat Q_e x=2z_T-x,
\]

with `T=4`.  The uniform relative output-error bound is

\[
\frac{\|\widehat Q_e x-Q_e x\|}{\|Q_e x\|}
\le \frac{2\alpha^{T+1}}{1-\alpha}=0.00260417.
\]

Five experts at width 16 are composed in every ordered route of length four.
The equal-call additive control is

\[
I+\sum_{e\in route}(Q_e-I).
\]

## Frozen gates

All must pass.

1. Every generator is skew-symmetric to `1e-12`, has spectral norm `0.25`
   within `1e-12`, and is full rank.
2. Every exact Cayley response has orthogonality error at most `1e-12`, at
   least 90% numerically nonzero entries, and `rank(Q_e-I)=D`.
3. Across 64 deterministic random vectors per expert, truncated application
   has relative error no greater than the analytic bound and no greater than
   `0.003`.
4. All `5^4=625` ordered sequential routes are numerically distinct at ten
   decimals.
5. Their flattened operator span is the full `D^2=256` dimensions at relative
   SVD tolerance `1e-9`.
6. The additive control has exactly `C(5+4-1,4)=70` distinct multiset outputs
   and span at most six.
7. Every sequential route remains orthogonal to `1e-11`.
8. At the scaling ledger `D=4096`, degree three, four Neumann multiplies and
   route length four, active sparse edge visits are below 3% of one dense
   `D x D` matvec.  The ledger must disclose route selection and physical
   gather costs as excluded.

## Interpretation boundary

A pass proves a new useful primitive relative to the retained rank-two routed
program: one sparse expert can alter all `D` directions, its exact response is
dense and norm-preserving, and a short word bank spans far beyond additive
aggregation.  It does **not** make `625` routes independent learned memories;
they share their generators.  Promotion requires a non-self-generated
learnability test, then a language-model screen against dense, sparse,
low-rank, additive-MoE, and rank-two-program controls.

## Collision boundary

Cayley-parametrized orthogonal RNNs and Neumann-Cayley training already exist.
The candidate claim is narrower: sparse feature-axis generators used as cheap
dense global experts, then selected in ordered noncommutative programs.  A
direct prior instance of that complete mechanism closes novelty, but not the
measured capability question.
