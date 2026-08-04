# Norm-funded triangular chart lift — development decision

## Decision

**Close before formal preregistration.**

The algebra is baseline-containing, invertible, and cheap, but the development
screen found no useful learned mechanism. A 1,200-step run was already enough
to falsify the proposed five-point causal advantage by roughly two orders of
magnitude. Do not spend five formal seeds or a language-model run on it.

## Algebra that remains valid

For a normalized pair `(u0,u1)`, the lift

`v0 = u0`, `v1 = u1 * (1 + alpha * clip(u0 / epsilon, -1, 1))`

is invertible whenever `|alpha| < 1`. At `alpha=0` it is exactly the ordinary
RMSNorm output. In the unsaturated region it supplies the cross term
`alpha/epsilon * u0*u1` to every following dense projection without adding a
matrix multiplication.

One RMSNorm scale coordinate can fund `alpha` without adding a learned value:
that scale is redundant with the corresponding columns of every following
projection. Fixing its effective scale to one and absorbing the ordinary scale
into those columns preserves the complete baseline function class.

At a realistic Transformer width, the lift is `O(D)` scalar work beside
`O(DM)` input projections and can be fused into the existing normalization
kernel. These are correct resource observations, not capability evidence.

## Development result

One unseen development world compared ordinary dense SwiGLU blocks, the exact
carrier-null parameterization, a self-gated chart, and the paired triangular
chart. All arms had 25,792 trainable values and bit-exact initial outputs.

| task | dense | carrier null | self chart | partner chart |
|---|---:|---:|---:|---:|
| ordinary | 92.7002% | 92.7368% | 92.7734% | 92.8101% |
| paired interaction | 79.6265% | 79.5166% | 79.8706% | 79.8584% |
| self interaction | 78.1128% | 78.0762% | 77.3193% | 78.0640% |

The paired chart exceeded dense by only `0.232` percentage points and lost
slightly to the equal-cost self chart. More importantly, its two learned chart
coefficients were only `+0.01076` and `-0.00998` despite a permitted magnitude
of `0.75`.

Causal interventions were negligible:

- setting both coefficients to zero reduced paired accuracy by `0.0366` points;
- shifting the partner relation reduced it by `0.0610` points.

The planned gate required at least five points for each intervention. Longer
training actually drove the coefficients closer to zero than the 300-step
pilot, so the failure is not a delayed-acquisition signal.

## Retained lesson

An invertible nonlinear chart can enlarge a projection's formal function class
at unchanged matrix bytes and MACs. That does not imply the tied cross term is
useful. Here ordinary SwiGLU learned the interaction almost as well and the
optimizer removed the proposed chart.

Future candidates must expose a discrete or continuous feature choice that
uses otherwise redundant checkpoint information **and** replaces an operation
already paid by the serving kernel. Do not retry another fixed coordinate
coupling graph.

Development artifact SHA-256:
`3372b3fc8b93517fbe5848c5146f5c95f079c6255c84563e0a7b2f007ff16817`.
