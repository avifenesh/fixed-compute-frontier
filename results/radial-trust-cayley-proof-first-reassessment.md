# Radial-trust Cayley program tree — proof-first reassessment

Status: **RETAIN AS A PRE-CANDIDATE; WITHDRAW THE LANGUAGE/GPU-SMOKE ORDER**  
Date: 2026-07-31  
Primary edge: B, low-batch decode only until the physical envelope is measured

## Decision

The Cayley program tree remains the strongest current non-matmul FFN proposal
because its possible gain is large.  At equal resident FFN parameters, the
**pre-trust** path is 8.203125% of dense SwiGLU's ideal multiply-like ledger at
width 384 and 0.846% at width 4,096.  Radial trust is not free: even an
optimistic scalar-arithmetic lower count raises those ratios to at least
8.7890625% and 0.906808%, before reductions and reciprocal-square-root cost.
This is still not a sub-percent polish.

The current radial-trust correction proves finite forward states, but it does
not yet justify a language-model run.  It converts the v1 explosion into a
quantifiable saturation/conditioning trade.  The tree also has a fundamental
route-diversity versus weight-traffic trade that the per-token operation ratio
omits.  Finally, its serial sparse critical path has not been mapped to a fused
Hopper block.

Therefore:

- do not execute the frozen one-step GPU smoke or 10M-token pilot in their
  current order;
- retain the abstract operator and its large potential Edge-B effect;
- make a separately frozen CPU conditioning microbench the next experimental
  stage;
- require a target-shape fused physical microbench before language training.

This is a proof-order correction, not a claim that the architecture is dead.

## 1. What is already proved

For width `D`, dense SwiGLU width `M`, and a complete or near-complete balanced
binary layout with `N≈M/2` internal nodes, the tree stores `Theta(D^2)`
diagonal payload scalars but a token visits at most

\[
L=\lceil\log_2(N+1)\rceil
\]

edges.  An arbitrary full binary tree does not have this bound; it can have
depth `N`.  Each selected edge reads three `D`-vectors and applies a
fixed-degree sparse Cayley/Neumann basis operation, so the balanced layout's
logical active work is `O(D log D)`.

With the frozen width-384 construction:

- tree parameters per FFN: 1,179,583 scalars;
- dense SwiGLU parameters per FFN: 1,179,648 scalars;
- active path payload: `27D = 10,368` scalars;
- pre-trust multiply-like path work: 96,768;
- radial-trust lower arithmetic count before reductions/rsqrt: 103,680;
- dense SwiGLU MACs: 1,179,648.

The pre-trust runtime-polynomial edge update has a generically full-rank local
Jacobian, and its toy width-eight path Jacobians span the complete
64-dimensional matrix space.  Radial trust has an invertible Jacobian at every
finite update, so it preserves each edge's local rank.  The complete 64-path
span has not been re-established after inserting the path-dependent trust
Jacobians and is not claimed here.  The retained rank result separates one
selected edge from a width-`r` additive expert whose local update rank is at
most `r`.

It does **not** separate the tree from every structured full-width control.
A routed full-width diagonal expert in an alternating fast basis also has a
full-rank local Jacobian.  Identity itself is full rank.  The real hypothesis
is therefore not rank; it is that language transformations reuse a small bank
of global bases and compositional conditional paths well enough to offset the
loss of independent dense degrees of freedom.

That is one legitimate composition hypothesis, but only after the numerical
and physical blocks below are measured.

## 2. The balance correction moves the wall

For raw edge update `d in R^D`, the correction is

\[
S_\tau(d)=\frac{d}{\sqrt{1+\|d\|_2^2/(\tau^2D)}}.
\]

Let `r=RMS(d)/tau` and `c=(1+r^2)^(-1/2)`.  Its Jacobian has:

- `D-1` tangential eigenvalues `c`;
- one radial eigenvalue `c^3`;
- condition number `kappa=1+r^2`.

For `tau=1`:

| raw update RMS | tangential gain | radial gain | condition number |
|---:|---:|---:|---:|
| 1 | 0.707107 | 0.353553 | 2 |
| 2 | 0.447214 | 0.089443 | 5 |
| 4 | 0.242536 | 0.014267 | 17 |
| 10 | 0.099504 | 0.000985 | 101 |
| 100 | 0.0099995 | 0.00000099985 | 10,001 |

Thus `rank(J_S)=D` for every finite `d`, but the rank statement hides an
arbitrarily ill-conditioned radial direction.  The residual identity preserves
a state-gradient path; it does not prevent parameter gradients through a
saturated raw update from becoming tiny.

This is not an accident of the selected formula.  Any monotone radial map that
sends an unbounded radius into a bounded radius must have radial derivative
arbitrarily close to zero somewhere; otherwise its output radius would keep
growing.  A post-hoc global bound necessarily exchanges forward explosion for
some saturation.

The current correction may still work if trained raw updates remain in the
well-conditioned region.  That is now an isolated measurable primitive, not a
reason to jump directly to language training.

## 3. Route diversity consumes weight traffic

The ideal ledger counts one path per token.  A GPU serves many tokens together,
and different routes touch different payload blocks.  At depth `t`, let
`p_(t,e)` be the probability of edge `e`.  For `B` independently routed tokens,
the expected number of distinct edge payloads touched is exactly

\[
U_t(B)=\sum_e\left[1-(1-p_{t,e})^B\right].
\]

The function `1-(1-p)^B` is concave for `B>=2`.  Uniform, diverse routing
maximizes distinct payload traffic; collapsed routing minimizes it.  The same
route diversity required for conditional capacity therefore erodes the weight
traffic advantage as batch or prefill tokens grow.

For a uniform depth-nine tree, there are 1,022 edge blocks in total:

| tokens routed together `B` | expected distinct edges | fraction of tree payload |
|---:|---:|---:|
| 1 | 9.00 | 0.88% |
| 8 | 55.67 | 5.45% |
| 32 | 163.21 | 15.97% |
| 128 | 412.25 | 40.34% |
| 512 | 796.99 | 77.98% |
| 4,096 | 1,021.83 | 99.98% |

The width-4,096 ledger uses an incomplete balanced tree with 7,166 internal
nodes and 14,332 edge blocks, not a complete 8,191-node tree.  Under uniform
branching, 128 routes touch about 878.00 edges (6.13%), while 4,096 routes touch
about 8,804.23 (61.43%).  Those can still be useful reductions, but they are
about 16.3x and 1.63x payload fractions—not the 118x single-token arithmetic
ratio.  A physical implementation must freeze the exact incomplete-tree
layout; `ceil(log2(N+1))` is only its maximum path depth.

This yields a required joint gate:

> quality must rise with route diversity faster than batch-level unique payload
> traffic and tail latency rise.

Route perplexity cannot be reported only as a quality diagnostic; it is also a
physical traffic variable.

## 4. The missing critical-path ledger

One edge uses a four-term Neumann approximation in the forward basis and its
transpose.  The pre-trust multiply-like expression

\[
2\cdot4\cdot3D+4D=28D
\]

contains dependent sparse propagation stages, an elementwise nonlinear
payload, and the reverse propagation.  Radial trust additionally executes per
edge at least `D` FP32 squares, a `D`-element reduction, one reciprocal square
root/scale formation, and `D` output divisions or scalings.  Counting only the
squares and final per-element operations as scalar arithmetic gives the
optimistic lower bound `30D` per edge:

\[
30\cdot384\cdot9=103{,}680\quad(8.7890625\%),
\]

\[
30\cdot4096\cdot13=1{,}597{,}440\quad(0.906808\%).
\]

The reductions, rsqrt, typed division cost, and synchronization remain
separate and make the complete ledger larger.  Nine tree edges serialize these
stages at width 384; the incomplete width-4,096 tree serializes up to thirteen.
A dense GEMM has more arithmetic but maps to wide tensor-core pipelines with
regular reuse.

Consequently even `103,680 < 1,179,648` does not imply a faster block.  A credible
kernel must freeze and measure:

- number of inter-feature synchronization stages per edge and path;
- formula-index arithmetic and gather coalescing;
- path divergence and unique edge payload bytes at each batch/context cell;
- shared/register/workspace residency and spills;
- p50/p95 block latency against dense SwiGLU, matched top-1 MoE, and a
  hardware-efficient structured matrix control.

## 5. Revised proof-first block order

### B0 — topology and algebra: passed

Retain the existing connected-basis, Neumann-error, invertibility, path
distinctness, Jacobian, and exact-parameter-ledger checks.  Replace the old
"unchanged 8.203125%" radial claim with the typed trust-operation ledger above.

### B1 — conditioning: next primitive, not yet executable

For depth 9 and 13, measure at initialization and after controlled payload
scales:

- raw-update RMS and maxima per depth;
- tangential and radial `J_S` gains;
- complete edge and path singular-value bounds;
- gradients to gate/up/down payloads and route thresholds;
- Gaussian RMS-one and one-coordinate RMS-one adversaries.

Before execution, a separate preregistration must freeze payload-scale grid,
seeds, sample counts, forced and natural routes, singular-value estimator,
definitions of ordinary-scale and material fraction, the near-zero-gradient
threshold, and every pass/kill boundary.  Kill the exact radial correction if
that frozen protocol finds the declared ordinary domain drives a material
fraction of updates into its near-zero parameter-gradient region.  No optimizer
tuning follows a kill.

### B2 — physical operator: later H100/H200 rental

Implement only the served forward block.  Compare width 384 and 4,096 across
decode batch/context cells and prefill token counts.  The kernel must first
beat or be noninferior to the strongest legal block under complete work,
traffic, workspace, and p95 latency accounting.

### B3 — route/traffic coupling: physical microbench

Replay collapsed, Zipf, and uniform frozen route traces.  Verify the expected
unique-edge law and measure whether higher route entropy crosses cache or
latency boundaries.

### B4 — language composition: not yet admitted

Only after B1-B3 pass may the from-zero language experiment ask the sole
remaining question:

> does language have enough reusable basis-and-path structure for this
> conditional program family to match or exceed dense and matched sparse
> controls?

The first language gate must include dense SwiGLU, exact-byte top-1 MoE,
equal-byte alternating-basis diagonal experts, and BTT/Monarch-like structured
controls from the start.  A 0.5% noninferiority result against dense alone is
not a completed breakthrough claim.

## 6. Effect-size boundary

If the candidate preserves protected quality and its complete low-batch decode
latency materially approaches the ideal reduction, it is a genuine Edge-B
result and a credible substrate for reinvesting saved work into more active
paths.  At equal served cost, that reinvestment could then test a smarter-model
claim.

If it is merely nonfinite-safe, full rank, or lower in nominal operations while
slower on Hopper, it is not an advance.  If route diversity needed for quality
materializes most tree payload at the frozen production batch, it is a
workload trade rather than a scaling break.

## Current authorization

Authorized now: paper review and writing the frozen B1 CPU preregistration.  
Not authorized now: executing B1, GPU rental, GPU smoke, language training, or
candidate-004 promotion.

Current literature keeps the broad neighborhood crowded: [Monarch](https://arxiv.org/abs/2204.00595)
already demonstrates hardware-efficient structured matrices; [Mixture of
Universal Experts](https://arxiv.org/abs/2603.04971) develops recursive expert
reuse and virtual width; and [Hi-MoE](https://arxiv.org/abs/2605.08292)
demonstrates hierarchical routing with explicit balance/specialization
controls.  The surviving sliver is the same-byte, full-width diagonal program
path plus its measured capacity/traffic frontier—not hierarchical routing or
compositional route count in general.
