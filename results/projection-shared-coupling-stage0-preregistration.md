# Projection-shared coupling — stage-0 preregistration

Status: **frozen before algebra and H100 timing**  
Date: 2026-07-28

## Candidate

For an even hidden width `d`, RMS-normalize without a learned gain and split
the channels into `a,b in R^(d/2)`.  Apply two elementwise additive couplings:

\[
b' = b + \alpha\odot |a|,\qquad
a' = a + \beta\odot |b'|.
\]

Q/K/V share this transformed attention input.  Gate/up share a separately
transformed FFN input.  The down projection is unchanged.

The `d/2` entries of each of `alpha` and `beta` occupy exactly the `d` parameter
slots removed with the corresponding RMSNorm gain.  Every downstream matrix
can absorb that gain into its columns, so removing it does not reduce the
ordinary baseline function class in exact arithmetic.  `alpha=beta=0` is the
ordinary gain-folded model.

## Claims to falsify

1. **No information bottleneck:** the transform is bijective with a triangular
   Jacobian of determinant one almost everywhere.
2. **Strict local enlargement:** a linear projection after the transform has
   input-orthant-dependent Jacobians that an ordinary linear projection cannot
   represent.  With `p` active pairs it exposes `2^p` distinct but strongly
   tied Jacobians without adding a matrix or hidden channel.  Region count is
   not treated as independent capacity.
3. **Exact budget:** parameter count, normalized-output width, scratch shape,
   QKV/FFN matrix shapes, matrix bytes, and KV-cache shape are unchanged.
4. **Favorable algebraic cost:** the new work is `O(tokens*d)` once per
   projection group, rather than `O(tokens*d*m)` or a pass over every output
   accumulator.

## Frozen stage-0 gates

### Algebra

Advance only if executable tests establish identity at zero, exact inversion,
gain folding for multiple immediate linear consumers, determinant one away
from kinks, and at least `2^p` distinct Jacobians for `p` nonzero channel pairs.

### H100 system screen

Compare a fused BF16 RMSNorm kernel with learned gain against a fused
RMSNorm-plus-two-coupling kernel reading the same `d` coefficient values.  Feed
each output into the same BF16 matrix multiplication.  Frozen shapes:

- hidden width `d=4096`;
- rows `{1, 8, 64, 256, 1024}`;
- attention-like output width `6144` and FFN-like output width `14336`;
- identical output/scratch tensors;
- 50 warmups and 500 timed iterations per cell, interleaved arm order.

Pass if candidate/baseline end-to-end latency is at most `1.01` at rows 256 and
1,024, at most `1.03` at every cell, and the fused normalization kernel uses no
more registers or shared memory.  The matrix instruction family/count must be
identical.

Passing stage 0 authorizes a matched language-model screen.  It does not claim
quality improvement.  The learning screen must compare at least:

- ordinary gain-folded RMSNorm;
- a parameter-matched ordinary RMSNorm control;
- a same-budget invertible self-hinge `x + gamma*abs(x)` control;
- one-way coupling (`beta=0`);
- two-way coupling;
- fixed channel permutation between successive layers.

No threshold, grouping, coefficient bound, or shape is tuned after timing.

The exact gain-folding claim is limited to unrestricted BF16/FP32 matrices.
Per-row or groupwise quantized matrices generally cannot absorb arbitrary
column gains exactly; quantized serving requires a later, separate chart and
quality gate.
