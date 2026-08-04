# Radial-trust Cayley program tree B1 conditioning preregistration

Status: **FROZEN FOR INDEPENDENT PAPER AUDIT; DO NOT EXECUTE**  
Date: 2026-08-01  
Parent decision: `results/radial-trust-cayley-proof-first-reassessment.md`

## Binary question

At the exact from-zero initialization and normalized-input scale claimed by the
Cayley program tree, does the radial trust map operate before its
precision-anchored magnitude/gradient-collapse region on a majority of token
paths, or does it merely turn the predecessor's explosion into an
ill-conditioned bounded update?

This is a conditioning preflight.  It does not train a language model, measure
language capability, establish a hardware advantage, or prove the final
raw-prose/digital-plane objective.

## The one unresolved empirical fact

Everything in this test except the distribution of raw update radii is fixed
by algebra or by the candidate's initialization:

> Under the frozen width/depth, initialization, RMS-one input families, and
> route census, what fraction of token paths contains an edge whose raw update
> radius enters the declared collapse region?

No optimization, quality, routing-specialization, kernel, or language claim is
bundled into this quantity.

## The theorem already established

For

\[
S_\tau(d)=\frac{d}{\sqrt{1+\lVert d\rVert_2^2/(\tau^2D)}},
\qquad
r=\operatorname{RMS}(d)/\tau,
\]

put `c(r)=(1+r^2)^(-1/2)`.  The Jacobian is

\[
J_S=cI-\frac{c^3}{\tau^2D}dd^T.
\]

It has `D-1` tangential eigenvalues `c`, one radial eigenvalue `c^3`,
and condition number

\[
\kappa(J_S)=\frac{c}{c^3}=1+r^2.
\]

The trusted radius is

\[
t(r)=\operatorname{RMS}(S_\tau(d))/\tau
=\frac{r}{\sqrt{1+r^2}}<1.
\]

Thus the map is finite and full rank for every finite input, but neither fact
lower-bounds its useful radial sensitivity.

### Precision-anchored surfaces

Let `epsilon_b=2^-7`, the spacing from `1` to the next larger BF16 number.  It
is used as a declared relative-response scale, not as a claim that every pair
below this difference rounds to the same BF16 vector.

The **amplitude-response surface** is the unique positive solution of

\[
\frac{t(2r)}{t(r)}-1=\epsilon_b.
\]

Solving gives

\[
r_{amp}=6.8966100057915725.
\]

At and beyond this surface, doubling a raw update along its ray changes the
trusted RMS by no more than one declared BF16 relative-response quantum.

The **directional-gradient surface** solves

\[
\frac{\lambda_{radial}}{\lambda_{tangential}}
=\frac{c^3}{c}=\frac{1}{1+r^2}=\epsilon_b,
\]

so

\[
r_{grad}=\sqrt{127}=11.269427669584644.
\]

At and beyond this surface, the radial derivative is at most `1/128` of the
tangential derivative.  This names a scale-relative conditioning failure; it
does not claim that FP32 optimizer state literally rounds the gradient to
zero.

At `r_amp`, `kappa=48.56322957198444`, `c^3=0.0029548719066573763`,
and `c^3/c=0.020591711235302367`.  At `r_grad`, `kappa=128`.

### Why these are necessary but not sufficient

The residual identity preserves a state-gradient path.  It does not restore a
parameter's suppressed radial influence through `S`.  Conversely, a small
radial eigenvalue at one rare edge does not prove language-training failure:
most useful parameter changes may be tangential, Adam can rescale coordinate
gradients, and the route may be rare.  The experiment therefore gates on a
predeclared **majority of complete token paths**, not on one maximum or on the
mere existence of an adversarial vector.

## Plain explanation

The trust map keeps the direction of an update and squeezes only its length.
When the raw length is already large, making it twice as large barely changes
the served update.  The same squeeze makes learning “more or less of this same
direction” much slower than learning a sideways change.  The test asks whether
ordinary initialized paths already spend most of their time in that regime.

## Small witness and adversary

- Small witness: with `D=2`, `tau=1`, and `d=(sqrt(2),0)`, tangential and radial
  gains are `1/sqrt(2)` and `1/(2 sqrt(2))`; both are plainly nonzero.
- Small adversary: with `d=(100 sqrt(2),0)`, `r=100`; the trusted radius is
  below one while the radial gain is about `10^-6`.  Bounded output has hidden
  a nearly unusable radial degree of freedom.

## Frozen implementation object

The reference candidate is the current
`experiments/radial_trust_cayley_program_tree.py` with:

- `tau=1`;
- width `384`, depth `9`, three bases, degree `3`, Neumann order `4`, and
  residual scale `1/sqrt(9)`;
- gate/up payload initialized `Normal(1, 0.02)`;
- down payload initialized `Normal(0, 0.15)`;
- threshold initialized to zero;
- basis raw weights initialized `Normal(0,1)` and bounded by the existing
  `alpha=0.25` construction.

The audit occurs before a B1 executable exists.  After audit approval, the
implementation, tests, this preregistration, and every imported candidate file
must be hashed into the result.  A hash mismatch makes the result invalid.

## Frozen domains

### Seeds

Use exactly five independently initialized modules:

`815, 6703, 6997, 7307, 7949`.

For each seed `s`, execute in this exact order:

1. call `random.seed(s)`, `numpy.random.seed(s)`, and `torch.manual_seed(s)`;
2. construct
   `RadialTrustCayleyProgramTreeMLP(width=384, depth=9, trust_rms=1.0, seed=s)`
   before constructing any input tensor;
3. call `module.train()` so natural routing uses the candidate's exact
   hard-forward/straight-through path;
4. construct inputs only with a separate CPU `torch.Generator` seeded
   `s + 1_000_003`.

Enable `torch.use_deterministic_algorithms(True)`.  The constructor's `seed`
argument controls route-coordinate hashing but does not seed its parameters;
the explicit `torch.manual_seed(s)` above is therefore mandatory.  No sample
is rejected or regenerated.

The width-384 input generator is consumed in this exact order: all Gaussian
rows, all Rademacher rows, then sparse rows in ascending `k`, drawing each
row's support before its signs.  The width-4,096 Gaussian and Rademacher
directions use a separate generator seeded `s + 3_000_003` and are generated
in that order.  The `D=8` Rademacher directions use seed `4_000_815`; its
JVP/VJP probe vectors use an independent generator seeded `5_000_815`.

### Input families

Every vector is normalized in FP64 to RMS exactly one before conversion to the
execution dtype.

1. **Gaussian ordinary:** 256 iid `Normal(0,1)` rows per seed.
2. **Rademacher ordinary:** 256 iid `{-1,+1}` rows per seed.
3. **Sparse stress:** 64 rows for each support size
   `k in {1,2,4,16,64,384}`.  Supports are uniform without replacement and
   signs are Rademacher; nonzero magnitude is `sqrt(D/k)`.

The Gaussian and Rademacher families define the non-adversarial ordinary
domain.  Sparse rows expose the known coordinate-concentration boundary and
are reported separately; they do not kill the candidate unless a numerical
correctness gate fails.

### Raw-update multiplier

Use exactly

`lambda in {0.25, 0.5, 1, 2, 4}`.

At each edge, after the unmodified payload and inverse-basis calculation forms
raw `d`, replace it by `lambda*d` immediately before `S`.  The resulting
trusted state is fed to the next edge, so downstream routes and raw updates are
allowed to change.  `lambda=1` is the only ordinary-scale decision cell.  The
other values measure the stability margin and cannot rescue a failure at one.

### Routes at width 384

1. **Natural:** use the candidate's exact hard-forward/straight-through
   routing for every ordinary and sparse row at every `lambda`.
2. **Exhaustive forced:** at `lambda=1`, enumerate all `2^9=512` bit strings on
   the first eight Gaussian and first eight Rademacher rows of every seed.

Forced routes use the existing `force_bits` mechanism.  They are a route
coverage census, not a training proxy; threshold gradients are undefined in
these cells and are not reported.

### Width-4,096 target-shape reference

The radial theorem is dimension independent, but the eventual width-4,096
candidate is not yet a complete executable graph.  B1 must not silently
pretend otherwise.

B1 therefore includes only a **trust-only numerical reference** at `D=4096`
with no tree or path-composition claim: for each seed, 256 Gaussian and 256
Rademacher RMS-one directions are assigned prescribed raw radii

`r in {0, 0.25, 0.5, 1, 2, 4, r_amp, 8, r_grad, 16, 64}`

in independent single applications of `S`.  It verifies formula and dtype
behavior only.  It does not estimate the raw-radius distribution or Jacobian
composition of a nonexistent width-4,096 tree.  A target-shape natural-radius
or depth-13 path claim remains prohibited until the incomplete
7,166-internal-node graph, per-depth directions, and exact initialization are
frozen in B2.

This explicit boundary replaces the earlier loose instruction to “measure
depth 13” without a defined depth-13 candidate.

## Typed measurements

For every width-384 edge/token cell record in FP64 from the observed raw
update:

- `r`, `c`, `c^3`, `1+r^2`;
- whether `r >= r_amp` and whether `r >= r_grad`;
- raw and trusted RMS, maximum absolute coordinate, node, bit, depth, seed,
  input family, route mode, and `lambda`.

For every complete token path record:

- maximum `r`;
- minimum tangential and radial gains;
- maximum condition number;
- whether any edge crosses either surface.

Report counts, medians, p90, p95, p99, maxima, and empirical CDFs separately by
seed, family, route mode, depth, and multiplier.  Do not pool input families or
seeds for a decision.

### Exact small-domain Jacobians

In FP64 at `D=8`, use radii

`{0, 0.25, 0.5, 1, 2, 4, r_amp, 8, r_grad, 16, 64}`

and 16 seeded Rademacher directions.  Form the full autograd Jacobian, compute
all singular values, and compare them with one analytic `c^3` and seven
analytic `c` values.  Also compare autograd JVPs and VJPs against the closed
form on 16 seeded probe vectors per cell.

## Numerical representations

- Analytic and exact-Jacobian work: FP64.
- Width-384 raw-radius census: FP32 candidate execution with FP64 metrics.
- Serving-reference trust calculation: cast raw `d` to BF16, compute
  `mean(d.float()^2)` and the square root in FP32, cast the denominator to
  BF16, divide in BF16, and compare with the FP64 formula.

Record BF16 elementwise equality between `S(d)` and `S(2d)` as a measurement.
It is not inferred from `r_amp`, because equality depends on exponent bins,
coordinates, and rounding.

## Material fraction and confidence

For one seed/family/route census, define

\[
F_{amp}=\frac{\#\{\text{paths with any }r\ge r_{amp}\}}
                 {\#\{\text{paths}\}},
\]

and analogously `F_grad`.

“Material” is frozen as a strict path majority: `F >= 0.5`.  This is the
weakest categorical claim that the collapse regime is typical rather than an
outlier.  It is deliberately lenient because B1 can reject but cannot prove
language learnability.

For natural IID input rows, report Wilson 95% intervals conditional on each
frozen module.  The 512 forced routes sharing an input are not IID Bernoulli
trials: report their fractions as exact conditional censuses separately for
each of the first eight inputs, with no Wilson interval and no pooling across
inputs.  Exact threshold decisions use observed fractions; no optional
stopping or sample extension is allowed.

## Harness-validity gates

These gates determine whether B1 produced interpretable evidence.  Failure of
any one is **inconclusive** and requires a corrected harness, a new artifact
hash, and independent re-audit; it does not scientifically close the
architecture.

1. **Protocol integrity:** exact constants, sample counts, hashes, seeds, and
   no rejected samples.
2. **Algebra:** every `D=8` singular value matches its analytic value with
   absolute error at most `1e-11` and relative error at most `1e-9` when the
   analytic value is nonzero; every JVP/VJP has relative error at most `1e-9`.
3. **Deterministic numeric reference:** the prescribed width-4,096
   single-application cells match the FP64 formula to the same analytic
   tolerance before BF16 casting; the runtime supports deterministic BF16
   conversion and division for the frozen reference path.
4. **Instrumentation accounting:** recorded row counts exactly equal the
   counts implied by every seed/family/route/multiplier cell, route bits and
   node indices match independent integer reconstruction, and every recorded
   path has the required number of edges.

## Scientific pass/kill gates

With all harness-validity gates passed, all following scientific gates must
pass:

1. **Numerical stability:** all candidate outputs are finite, trusted FP64 RMS
   is below one for every nonzero finite raw update, and BF16 trusted RMS is at
   most `1 + 2^-7`.
2. **Ordinary natural paths:** at `lambda=1`, both ordinary families in every
   seed have `F_amp < 0.5` and `F_grad < 0.5`.
3. **Ordinary forced paths:** at `lambda=1`, each of the first eight inputs in
   both ordinary families and every seed has exact conditional 512-route
   census fractions `F_amp < 0.5` and `F_grad < 0.5`.

The two fraction gates deliberately include both surfaces even though
`r_grad > r_amp`; recording both makes the severity visible and guards against
an implementation swapping the predicates.

## Decisions fixed before execution

### Pass

A pass means only:

> The exact radial correction is not already dominated by
> precision-anchored saturation on a majority of initialized RMS-one paths at
> width 384, and its implementation matches the derived local geometry.

It authorizes writing—not yet running—a target-shape B2 physical-operator
preregistration.  It does not authorize language training, a GPU rental, or a
capability claim.

### Fail

With a valid harness, any failed **scientific** gate closes this exact radial
correction, initialization, and tree combination.  Do not tune `tau`, payload
standard deviation, residual scale, or the material threshold after seeing
B1.  A successor must derive a new stability mechanism on paper and receive a
new name.

### Inconclusive

Missing hashes, an unavailable dependency, unsupported deterministic BF16
behavior, a failed harness-validity gate, or a protocol deviation is
inconclusive—not a scientific failure and not a pass.  Fixing the harness
requires a new preregistration hash and independent re-audit before execution.

## Controls and claim boundary

No empirical gradient-alignment control belongs in B1.  The local eigenvalues
are analytic once `r` is observed.  How loss gradients align with radial and
tangential directions would be a second empirical unknown and cannot be hidden
inside this radius census.  A same-forward custom backward with `J=cI` would
isolate the denominator's rank-one term, while a same-forward `J=I` estimator
would be a stronger gradient-through control; neither establishes a legal
training rule or enters this decision.

The current closest public radial-bounding work preserves tangential gradients
while bounding vector norms, but tangential preservation does not establish
radial learnability.  This B1 test is specifically about the missing radial
degree of freedom.  Even a complete pass says nothing about whether the Cayley
tree's compositional basis/path prior matches language, whether its routing is
learnable, or whether serial sparse operations beat tensor-core GEMM.

## Runtime and resource ceiling

- CPU only; no CUDA device may be visible to the executable.
- No network, training data, checkpoint, optimizer step, or model download.
- Expected wall time: under 30 minutes on the local CPU.
- Peak process RSS ceiling: 8 GiB.
- Result JSON and optional compressed diagnostic arrays together: at most
  256 MiB.
- Exceeding a ceiling aborts as inconclusive.

## Current authorization

Authorized now: independent paper audit of this preregistration.  
Not authorized now: B1 implementation or execution, GPU start/rental, B2
kernel work, language training, or promotion of the architecture.

Current closest reference: [Z-Plane Neural Networks](https://arxiv.org/abs/2606.15669)
uses radial bounding and argues from preserved tangential gradients.  Its
existence strengthens the need to type radial and tangential sensitivity
separately; it does not answer this candidate's path-distribution question.
