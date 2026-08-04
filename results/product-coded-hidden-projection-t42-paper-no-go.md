# T42 product-coded hidden projections — paper no-go

Date: 2026-08-01  
Status: **CLOSED BEFORE IMPLEMENTATION; NO CPU, MODEL, OR GPU RUN ADMITTED**

## Proposed exit from matmul

Constrain each hidden vector to a product code, precompute every projection of
every sub-codeword, and replace Q/K/V/O and FFN matrix-vector products by table
reads and vector additions.  One hidden-state encoding could be reused across
all projections in a block.

This is an exact real-arithmetic lookup identity on the quantized reachable
state.  The one-level, fully materialized construction below does not provide a
useful fixed-cost regime for generic language states.  This paper does **not**
prove that every hierarchical, additive, factored, sparse, shared, or generated
LUT organization has the same storage bound.  Closure rests on the absence of a
complete Pareto path plus strong existing LUT and low-bit controls, not on a
universal product-quantization impossibility theorem.

## Exact lookup identity

Partition `x in R^d` into `B=d/s` blocks.  Block `b` chooses one of `K`
codewords `c_(b,k) in R^s`; write its index as `z_b`.  For a projection
`W in R^(m x d)`, split its columns into matching blocks `W_b` and precompute

\[
T_{b,k}=W_b c_{b,k}\in R^m.
\]

Then the product-coded state `q(z)` satisfies exactly

\[
Wq(z)=\sum_{b=1}^{B}T_{b,z_b}.
\]

The construction removes online multiplications after the indices exist.  It
does not remove output-vector reads or accumulations.

## Exact storage law for the proposed materialization

At equal table and weight precision, the dense projection stores `dm` scalars
and a fully materialized, uncompressed, one-codebook-per-block projected table
stores

\[
BKm=\frac{K}{s}dm
\]

scalars.  Its storage ratio is therefore exactly

\[
\rho_{table}=K/s.
\]

Codebooks, indices, scales, alignment, and metadata are additional.  If table
precision is `p_T` and dense-weight precision is `p_W`, the payload ratio is
`(p_T/p_W)K/s` before those additions.  This is not a storage lower bound for
factored, additive, hierarchical, shared, sparse, or generated tables.
Finite-precision bit-exactness also requires charging table rounding and fixing
the accumulation order.

## Rate-distortion obstruction

The product code carries

\[
R=\frac{\log_2K}{s}
\]

bits per hidden dimension.  For an iid Gaussian source with per-coordinate
variance `sigma^2`, the squared-error rate-distortion function gives

\[
\frac{D}{\sigma^2}\ge 2^{-2R}.
\]

To preserve relative mean-square distortion at most `delta`, any such code
therefore needs

\[
R\ge \frac12\log_2(1/\delta),
\qquad
K\ge \delta^{-s/2}.
\]

Combining this with the exact table law gives

\[
\rho_{table}\ge \frac{\delta^{-s/2}}{s}
\]

at equal precision.

For five-percent relative MSE in this one-level construction,
`20^(s/2)/s` is increasing for every integer `s>=1`; the minimizing block is
therefore `s=1`, `K>=5`, hence at least five table scalars for every
dense-weight scalar.  Larger one-level blocks make the exponential codebook
term worse.  If projected tables could use INT4 while dense weights used BF16,
the optimistic payload floor would be `5/4` in that cell, before metadata.
Ten-percent distortion similarly needs `K>=4` at `s=1`, making that optimistic
INT4/BF16 payload comparison tie before overhead.

These byte examples are not exact realizations: INT4 quantization of
`T_(b,k)` adds projection error not included in the hidden-state distortion
`delta`, and a symmetric control must also include quantized dense weights.

The numeric factors are also not universal over compositional codes.  With
`C` additive sub-codebooks per block, the representable count becomes `K^C`,
the rate is `C log2(K)/s`, and the materialized-table ratio is `CK/s`.  Such
layering can reduce the one-level exponent while paying proportionally more
lookups, additions, assignment structure, and metadata.  The nested-lattice
control below explicitly exploits this caveat.

This is a scoped obstruction for isotropic Gaussian approximation.  It is not
a theorem that jointly trained language states are iid Gaussian or that a
different discrete network cannot be useful.

## Assignment and accumulation do not disappear

Brute nearest-code assignment costs on the order of

\[
BKs=dK
\]

distance-coordinate operations.  The exact table sum reads and accumulates
`B` vectors of length `m`, or `dm/s` output scalars.  Hierarchical assignment
can change the search schedule, but its tree, centroids, comparisons, and
errors are charged.

More importantly, attention, residual addition, normalization, and the FFN
produce new continuous states.  Q/K/V may share one encoding of their common
input; FFN gate/up may share another.  But O consumes the continuous attention
mixture, and down consumes the continuous gated FFN intermediate, so each
needs a new encoding.  Assignment is therefore charged at projection groups
inside a block, not just at block boundaries.  Omitting these quantizers makes
the next table identity inapplicable; including them restores assignment work
and repeated distortion.

## Strongest-control collapse

Once activations are restricted to low-rate codes, integer/binary/ternary GEMM
and compact decoded-codeword paths are required controls.  Scalar product
codes in particular turn the table entries into redundant multiples of one
column, so a column-plus-decoder representation is compact but restores online
arithmetic.  For larger blocks, categorical centroid indices are not
automatically numeric low-bit activations: compact storage may lose active
arithmetic or latency.  A complete physical ledger is needed; dominance is not
an algebraic theorem.

For larger sub-codewords this proposal is the established LUT-NN/product-
quantization family.  LUT-NN already learns centroids and precomputes operator
outputs; current lattice-LUT work makes the same rate/table-size trade.  The
mechanism is therefore not a new operation class, and the strongest low-bit
matmul/LUT controls inherit the useful cases.

Relevant primary controls:

- [LUT-NN](https://arxiv.org/abs/2302.03213);
- [high-rate nested-lattice LUT matmul](https://arxiv.org/abs/2505.13164);
- [FLUTE](https://arxiv.org/abs/2407.10960); and
- low-bit activation/weight GEMM under the same finite-state contract.

## Why joint training does not rescue the paper claim

A from-zero model could intentionally learn a very low-rate discrete hidden
state whose distribution evades the Gaussian bound.  That is a different
conditional/discrete architecture, not an exact or safe replacement of a
dense Transformer projection.  It must then explain:

1. why the low-rate state is sufficient for protected language behavior;
2. how its transition is computed without an equally expensive encoder;
3. why a matched low-bit network cannot realize the same function more
   cheaply; and
4. how saved resources fund a breakthrough-sized capability gain.

Those are the complete unknowns, not one local empirical edge.

## Decision

Close the proposed one-level projected product-code table as the next matmul
escape.  Retain the exact table-sum identity, its construction-specific `K/s`
storage law, the Gaussian rate lower bound, and their combined inequality only
for that one-level materialization.  Retain additive/hierarchical codebooks as
a caveat, not as excluded mechanisms.
Do not implement a quantizer, run a local approximation census, or benchmark a
GPU LUT kernel.  A successor must change the discrete transition algebra or
prove a non-isotropic reachable-state law before reopening this family.
