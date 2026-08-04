# Gauge-funded quadratic values — algebra passes, physical and LM gains unproved

Status: **superseded: zero-diagonal GFQV failed the covariance preference gate**  
Date: 2026-07-26  
Candidate number: **not admitted as 004**

See [the follow-up decision](gauge-to-curvature-stage0-decision.md): the broader
gauge-to-curvature result survives, but the retained refinement uses one cyclic
forbidden entry per column so square terms are not discarded.

## What this round gained

The earlier learnability branch is closed.  If a training architecture compiles
to an ordinary served model, the ordinary-model trainer can run the same
training algorithm and compiler.  It then emits the identical served weights
for every dataset and seed.  Developmental over-parameterization, diagonal
lifting, symmetry-restricted training, and train-then-fuse are therefore
training methods or priors, not new served architectures, unless training
compute or the optimizer is explicitly restricted.

The local attention-gauge idea is also not new by itself.  Query/key and
value/output matrix pairs contain exact change-of-basis redundancies.  PIFA
already gives a general pivot-row representation for low-rank matrix products,
and recent attention work explicitly identifies the per-head query/key gauge,
removes a query projection, and reinvests the parameters.  Full RoPE also
blocks an arbitrary query/key basis change: only transformations commuting
with every rotary matrix remain.

The surviving opening is narrower: use the exact value/output redundancy to
fund **new nonlinear value computation**, rather than merely deleting the flat
coordinates.

## Construction

For one value head, let

\[
W_V\in\mathbb R^{D\times r},\qquad W_O\in\mathbb R^{r\times D_o}.
\]

Attention uses this pair only as

\[
A(X)XW_VW_O.
\]

For every invertible `C`, replacing

\[
(W_V,W_O)\mapsto(W_VC,C^{-1}W_O)
\]

preserves the output exactly.  If `r` selected rows `P` of `W_V` form an
invertible minor, choose `C=(W_V[P,:])^{-1}`.  The transformed value matrix has
the chart

\[
W'_V[P,:]=I_r.
\]

With fixed contiguous pivots, write the remaining rows as `B`.  Standard
gauge-fixed linear values are

\[
v_j(x)=x_{p_j}+\sum_{i\notin P}x_iB_{ij}.
\]

Now add an off-diagonal learned matrix `G`, but spend only the operations that
the eliminated dense identity minor used:

\[
\boxed{
v_j^{\mathrm{GFQV}}(x)=
x_{p_j}+\sum_{i\notin P}x_iB_{ij}
+x_{p_j}\sum_{k\ne j}x_{p_k}G_{kj}
}.
\]

`G=0` exactly recovers the gauge-fixed linear circuit.  Consequently the
closure of the candidate contains the ordinary full-rank value/output circuit
for a fixed pivot chart; adaptive pivots cover every full-rank circuit but add
index metadata and irregular gathers.  Training from scratch with one fixed
contiguous chart avoids that metadata, at the price of excluding a measure-zero
chart boundary exactly.

## Exact logical ledger

For one dense `D -> r` value projection:

| Quantity | Dense value | GFQV value |
|---|---:|---:|
| Learned scalars | `Dr` | `(D-r)r + r(r-1) = Dr-r` |
| Scalar multiplications | `Dr` | `(D-r)r + r(r-1) + r = Dr` |
| Scalar additions | `(D-1)r` | `(D-r)r + r(r-2) + r = (D-1)r` |
| Cached value scalars/token | `r` | `r` |
| Attention scan work | unchanged | unchanged |

The final `r` multiplications are the elementwise products between pivot inputs
and the off-diagonal projected pivot inputs.  Thus the candidate adds
second-order interactions with **exactly the same scalar multiply and add
count**, one fewer learned scalar per channel, the same value-cache shape, and
the same attention scan.

At `D=4096, r=128`, both versions use 524,288 scalar multiplications and
524,160 scalar additions for value creation.  GFQV stores 524,160 rather than
524,288 value-projection weights.  This is not a speed claim: two smaller
projections plus an elementwise product may execute slower than one dense GEMM.

### Key-coupled Pareto variant

The same chart has a cheaper, more coupled variant.  Reuse the already computed
pre-RoPE key `k=xW_K` as the gate:

\[
v_j^{\mathrm{KCGV}}(x)=
x_{p_j}+\sum_{i\notin P}x_iB_{ij}
+x_{p_j}\alpha_j k_j.
\]

`alpha=0` again recovers the linear circuit.  The key weights now serve two
algebraically different roles: addressing through QK and nonlinear content
construction through V.  For dense K+V projections the logical ledger is:

| Quantity | Dense K+V | KCGV K+V | Saving |
|---|---:|---:|---:|
| Learned scalars | `2Dr` | `2Dr-r^2+r` | `r(r-1)` |
| Scalar multiplications | `2Dr` | `2Dr-r^2+2r` | `r(r-2)` |
| Scalar additions | `2r(D-1)` | `r(2D-r)` | `r(r-2)` |

At `D=4096,r=128`, that removes 16,256 learned scalars and 16,128 scalar
multiplications/additions per KV head while keeping both cache widths.  With
`W_K=I`, one output can be `x_1+x_1^2`, whose centered second finite difference
is two while every affine value map's is zero.  The trade is parameter coupling:
an independent `G` offers many more value-specific degrees of freedom; KCGV is
cheaper but asks one key basis to serve both search and retrieval.

## Strict layer-local separation

Take `D=r=2`, `B` empty, `G[1,0]=1`, and read the first value channel.  Then

\[
f(x_1,x_2)=x_1+x_1x_2.
\]

Its mixed finite difference on the Boolean square is

\[
f(1,1)-f(1,0)-f(0,1)+f(0,0)=1.
\]

Every affine value projection has mixed finite difference zero.  Therefore
GFQV is strictly more expressive than a linear value projection while retaining
the latter at `G=0`.  This proves a real operator-level gain, not that an entire
fixed-depth Transformer cannot reproduce the same function elsewhere in its
MLP.  That stronger claim is deliberately not made.

## CPU gate

The executable gate:

- transformed a random value/output pair to an identity-pivot chart;
- preserved `W_V W_O` to `1.12e-16` maximum absolute error;
- recovered the original outputs at `G=0` over 128 random inputs to
  `8.89e-16` maximum error;
- produced the nonzero mixed-finite-difference witness exactly;
- recovered the KCGV baseline at `alpha=0` and produced its second-difference
  witness exactly;
- checked the multiply/add identities at multiple shapes.

Artifacts:

- [Executable gate](../experiments/gauge_funded_quadratic_values.py)
- [Machine-readable result](gauge-funded-quadratic-values-stage0.json)
- [Focused tests](../tests/test_gauge_funded_quadratic_values.py)

## Prior-art boundary

The ingredients are close to existing work:

- [Karbevski and Mijoski](https://arxiv.org/abs/2510.23912) prove the exact
  per-head query/key gauge, remove query weights under additional conditions,
  and empirically reinvest saved parameters.
- [PIFA](https://arxiv.org/abs/2501.19090) is a general lossless pivot-row
  factorization of low-rank products and already demonstrates GPU kernels.
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) states the RoPE
  non-commutativity obstruction that prevents ordinary projection absorption.
- [GLU Attention](https://arxiv.org/abs/2507.00022) already introduces
  parameter-matched nonlinear values, while recent multiplicative-gating work
  studies the resulting expressivity/geometry.

No exact match for the specific identity-chart plus off-diagonal quadratic
reinvestment was located in this pass.  That is not proof of novelty.  The
claim worth testing is the exact ledger and the placement of the nonlinear
interactions before value caching, not “nonlinear values” in general.

## Fatal controls and next gate

The first experiment must compare all of these at matched total model
parameters, training tokens, and served dimensions:

1. ordinary dense value projection;
2. gauge-fixed/PIFA linear values, leaving the saved work unused;
3. GFQV;
4. parameter/MAC-matched GLU values;
5. the same nonlinear budget moved into the MLP;
6. a random or permuted pivot-coordinate control.

Before language-model training, a fused-kernel microbenchmark must show that
the split projection and elementwise product stay within the frozen TPOT
tolerance.  Quantization must be checked because products can widen activation
tails.  Only after a kernel plan and an LM preregistration survive those
controls should an H100 be rented.

The independent-G schedule can remain one Hopper launch only with a custom
mid-mainloop transformation: accumulate the pivot `G` slice, drain the WGMMA
pipeline, replace the accumulator by `x_P * accumulator + x_P`, then continue
over the non-pivot `B` slice.  Stock dense GEMM cannot express this.  A separate
small `x_P G` kernel is already disqualifying.  KCGV avoids that small GEMM but
requires paired K/V output tiles or cross-tile synchronization, so it is not
automatically the easier kernel.

## Decision

Retain GFQV as a **pre-candidate**.  It has a proved, strict local expressivity
gain at equal logical value-projection arithmetic and unchanged KV state.  It
does not yet have a physical GPU result, an LM-quality result, or a defensible
novelty claim strong enough for candidate 004.
