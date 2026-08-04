# Peak-lifted attention — Stage 0 preregistration

## Question

Can we split the evidence that ordinary softmax collapses into one scalar and
learn from both parts, while using only state already present in
FlashAttention?

For semantic chunk `b`, ordinary stabilized softmax computes

\[
m_b=\max_j s_j,\quad
\ell_b=\sum_j e^{s_j-m_b},\quad
L_b=\log\sum_j e^{s_j}=m_b+\log\ell_b.
\]

Peak-lifted attention changes only the exported chunk maximum:

\[
(m_b,\ell_b,o_b)\mapsto((1+\alpha_h)m_b,\ell_b,o_b),
\]

then merges chunks with the ordinary online-softmax monoid. Equivalently,

\[
\tilde s_j=s_j+\alpha_hm_{b(j)},\qquad
L'_b=(1+\alpha_h)m_b+\log\ell_b.
\]

The local conditional softmax inside each chunk is unchanged. `alpha=0` is
ordinary attention exactly. The matched control is global temperature,
`tilde(s)=(1+alpha)s`, which can change chunk evidence but also changes every
within-chunk ratio and is reparameterizable by Q/K scale.

## Frozen screen

- Seed `20260730`, float64 PyTorch, runtime/device recorded.
- 48 fixed normalized sequences, length 8, semantic chunk size 2, input width
  4, Q/K width 2, value width 3, output width 4.
- Every arm stores the same 44 scalars: four norm carriers plus dense Q, K, V,
  and O matrices.
- Ordinary applies norm scales before Q/K/V. Temperature and peak arms use the
  exact chart that folds those scales into the Q/K/V rows, and decode
  `alpha=0.5*tanh(carrier_0-1)` from one otherwise redundant norm coordinate.
- Jacobians are evaluated at the common `alpha=0` endpoint with the standard
  float64 SVD tolerance.
- Arms: ordinary, global temperature, peak lift.

The same run tests direct effective logits, left/right/balanced/permuted merge,
common row shifts, local conditional probabilities, a fixed-target IIA
witness, gauge conversion, positive unit-sum weights, large score offsets, and
the score-gradient concentration introduced by the max path.

## Fatal gates

1. All arms agree within `1e-12` at the endpoint.
2. Gauge-converted ordinary output agrees within `1e-11`.
3. Every merge and direct effective-logit output agrees within `1e-11`.
4. A common score shift changes output by at most `1e-11`.
5. Peak lift preserves each chunk's local conditional distribution within
   `1e-12`; nonzero global temperature changes it by at least `1e-3`.
6. Ordinary target log-weight ratio range in the IIA witness is at most
   `1e-12`; peak lift's range is at least `1e-3`.
7. Weights are positive and unit-sum within `1e-12`.
8. Global-temperature rank equals ordinary rank. Peak-lift rank strictly
   exceeds ordinary rank, with the new singular value at least `1e4` times its
   tolerance.
9. Carrier derivatives match centered finite differences within `1e-6`.
   Temperature's derivative residual outside the ordinary Jacobian span is at
   most `1e-8` of its norm; peak lift's residual fraction is at least `1e-3`.
10. Stress output and all serialized numerics are finite.
11. The ledger is exactly 44 scalars in every arm, zero Q/K/V/O MAC change,
    zero KV/cache field, zero split partial field, zero additional exponential,
    and zero additional score reduction. The candidate adds one scalar FMA and
    an adjusted softmax center per query/head/chunk.

The max-gradient concentration record is diagnostic, not fatal at Stage 0; it
becomes a stability gate in any learning screen.

## Decision boundary

Passing proves only exact containment, a new sampled local direction beyond
global temperature, and an ordinary-merge execution algebra. It authorizes a
frozen H100 forward gate. It does not prove language quality, training
stability, novelty, or acceptable latency.

## Collision status

Learned softmax temperatures, Self-Adjust Softmax, max/log-sum-exp pooling,
FoX, and chunk routers are related. The narrow construction here is a
per-query semantic-chunk decomposition of dense attention evidence that
reweights the already-computed local maximum without another gate projection.
The exact collision claim remains open.

- <https://arxiv.org/abs/2502.18277>
- <https://arxiv.org/abs/2503.02130>
- <https://arxiv.org/abs/2607.07724>
