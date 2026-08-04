# Confidence-lifted attention — Stage 0 preregistration

## Question

Can an attention head use a statistic already present inside its softmax tile
to distinguish **one decisive match** from **many mediocre matches**, without
another projection, KV field, or persistent activation?

Keys are partitioned into fixed semantic chunks. For a query/head and chunk
`b`, ordinary stabilized softmax already forms

\[
m_b=\max_j s_j,\qquad
\ell_b=\sum_j e^{s_j-m_b},\qquad
o_b=\sum_j e^{s_j-m_b}v_j.
\]

For `n_b>1`, define the concentration feature

\[
c_b={n_b-\ell_b\over n_b-1}\in[0,1],
\]

and `c_b=0` for a singleton. A learned head coefficient lifts the whole chunk:

\[
\hat m_b=m_b+a_h p_b+b_h c_b,
\]

where `p_b` is a fixed normalized chunk position. The state
`(hat(m_b), ell_b, o_b)` is then merged with the ordinary associative online
softmax operation. Equivalently, every token in chunk `b` has effective logit

\[
\tilde s_j=s_j+a_hp_b+b_hc_b.
\]

`a_h=b_h=0` is ordinary attention exactly. The `a_h` arm is the matched fixed
recency control; `b_h` is the proposed endogenous confidence direction.

## Why this variant was selected

The earlier unexecuted recurrence scaled all prior chunks after each current
chunk. Analysis rejected it before a formal run: for iid-normal scores its
concentration feature is mostly a constant recency decay, cumulative shifts
can saturate with context length, and split-KV reduction needs an extra decay
state.

The lifted form uses confidence once. It is bounded per chunk, merge-order
independent after atomic chunk construction, and exports the ordinary
`(m,l,o)` state. A constant component of `c_b` cancels from the final softmax,
so only differences in within-chunk score concentration affect routing.

## Exact capability witness

Ordinary softmax routes total chunk mass using

\[
L_b=\log\sum_j e^{s_j}=m_b+\log\ell_b.
\]

It therefore assigns the same chunk mass to one score near zero and to `n`
equal scores near `-log(n)`. Confidence lifting separates those cases because
their `c_b` values approach 1 and 0. The full output is a hierarchical mixture

\[
\mu_b=o_b/\ell_b,\qquad
O=\sum_b \operatorname{softmax}_b(L_b+a_hp_b+b_hc_b)\mu_b.
\]

## Frozen numerical screen

- Seed: `20260728`.
- Float64 PyTorch; device is recorded.
- Rank witness: 48 fixed normalized sequences, length 8, semantic chunk size
  2, input width 4, Q/K width 2, value width 3, output width 4.
- Shared ledger: four norm scalars plus dense Q, K, V, and O matrices, exactly
  44 stored scalars in every arm.
- Ordinary applies the four norm scales before Q/K/V.
- The other arms use the exact local chart that folds those scales into Q/K/V
  rows. Carrier 0 sets `a=2*tanh(carrier_0-1)`; carrier 1 sets
  `b=8*tanh(carrier_1-1)`. Carrier values are not applied to the normalized
  hidden vector.
- Arms: ordinary softmax, fixed recency, and recency plus confidence lift.
- Jacobians are evaluated at the common endpoint where carriers equal 1.
  Rank uses `max(shape) * float64_eps * largest_singular_value`.

The run also freezes:

1. source-to-chart conversion for non-unit positive norm scales;
2. left, right, balanced, and permuted ordinary state merges;
3. direct effective-logit equivalence and common-score-shift invariance;
4. a matched-mass decisive-versus-diffuse witness;
5. an IIA witness holding two scores fixed while changing a third;
6. positivity/unit-sum and large-offset finite-value stress;
7. the empirical mean/std/quantiles of `c_b` for iid normal chunks of width 64
   and 128, recorded as a warning rather than a pass gate.

## Pass gates

All gates are fatal unless explicitly described as diagnostic.

1. All arms agree within `1e-12` at `a=b=0`.
2. Gauge-converted ordinary outputs agree within `1e-11`.
3. Every state merge and the direct effective-logit output agree within
   `1e-11`, including a permuted merge order.
4. A common score shift changes output by at most `1e-11`.
5. The matched-mass ordinary chunk-share gap is at most `1e-12`; the
   confidence-lifted share gap is at least `0.20`.
6. Ordinary log target-weight ratio range in the IIA witness is at most
   `1e-12`; confidence lifting's range is at least `1e-3`.
7. Weights are strictly positive and sum to one within `1e-12`.
8. Fixed-recency rank strictly exceeds ordinary rank, and confidence rank
   strictly exceeds fixed-recency rank. Each newly counted singular value is
   at least `1e4` times its SVD tolerance.
9. Carrier derivatives match centered finite differences to relative error
   `1e-6`. The confidence-carrier derivative has a nonzero residual outside
   the fixed-recency Jacobian span.
10. All stress and serialized numeric fields are finite.
11. The static ledger is 44 scalars in every arm, with zero Q/K/V/O MAC change
    and zero per-token cache fields.

## Execution ledger and decision boundary

An implementation aligned to semantic chunks performs the same score-element
exponentials and reductions as ordinary attention. Relative to the most
optimized running-global-max FlashAttention schedule, constructing a local
chunk state before lifting it may add one scalar exponential/rescale and a
vector rescale per query/head/chunk. This cost is explicit, not assumed away.
The candidate adds no split-KV state field because lifted chunks merge with the
ordinary softmax monoid.

Passing proves only a new local function direction, exact endpoint, and stable
parallel algebra. It does not prove language quality, novelty, or an H100
latency pass. Passing authorizes an H100 fatal gate before language training.

## Collision controls

FoX and FiX establish that learned forgetting can improve softmax attention,
but both use additional learned gate machinery. The narrow candidate here is
not a forget recurrence: it is a query-specific chunk router feature derived
from the QK tile's own concentration, with no gate projection or gate cache.
Searches for attention-entropy routing and uncertainty-gated block selection
did not establish this exact dense-softmax algebra; novelty remains unproven.

- <https://arxiv.org/abs/2503.02130>
- <https://openreview.net/forum?id=WsNpCXq6SG>
- <https://arxiv.org/abs/2607.07724>
