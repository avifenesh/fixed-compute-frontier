# T30 learnability screen — pre-run matched-control audit

Status: **DRAFT SCREEN WITHDRAWN; ZERO TRAINING RUNS**  
Date: 2026-07-31

## Verdict

Do not run the first T30 learnability implementation.  It can produce a large
number without testing the advertised T30-versus-T29 separation.

This is a successful pre-run falsification.  No training metric, seed, or GPU
result exists, so no result was discarded and no gate was tuned after seeing
data.

## The intended claim

T30 adds a noncommuting permutation frame to T29's diagonal-affine recurrence:

\[
h' = a_x\odot P_{g_x}h+b_x.
\]

The intended causal claim is that the permutation factor learns useful state
routing that a byte-matched diagonal-affine update

\[
h'=\widetilde a_x\odot h+\widetilde b_x
\]

cannot realize at the same width and depth.

## Why the withdrawn screen does not test it

### 1. The control is only an additive subcase

The implementation fixes every recurrence scale to one.  Its control is

\[
h'=h+b_x,
\]

not the full T29 class.  Equal parameter counts do not repair a functionally
disabled baseline.

### 2. The document arm uses one generator

Every value token is followed by four copies of `R`; `F` never occurs in a
document.  Powers of one rotation commute.  Therefore success on the document
task cannot be attributed to learning a noncommutative algebra.

### 3. Zero initial state removes the proved Jacobian witness

The T30 separation concerns how an existing arbitrary state is transformed:
the Jacobian of a diagonal-affine word is diagonal, whereas a nontrivial
permutation has off-diagonal entries.  The document arm always starts from
zero.  It observes only accumulated offsets, so it never asks the model to
route arbitrary incoming features.

### 4. A legal diagonal-affine model can encode position

With a learned scale for `R`, a value written at position `k` is multiplied by
a different power of that scale than a value written at position `j`:

\[
b_v\mapsto a_R^{n-k}\odot b_v.
\]

Across coordinates, distinct scale values form a moment or Vandermonde-like
positional code.  The exact additive-alias proof therefore applies only to
the deliberately fixed `a_R=1` subcase; it is not a no-go theorem for T29.

### 5. Logical operation counts are not measured serving cost

The candidate performs address generation or gather-like access for its
logical permutation.  The lazy-frame theorem avoids a physical state copy,
but coalescing, fusion, and latency remain empirical.  The screen's equal
multiply/add count cannot establish equal serving cost.

## What remains valid

- The exhaustive Stage-0 `D_109` algebra, codec, title substitution, and lazy
  frame results remain valid.
- The group-generator subtask is a legitimate route-learning diagnostic, but
  permutation-diagonal state tracking is already strongly covered by PD-SSM
  prior art and is not a breakthrough claim by itself.
- The unused draft is retained for provenance and made non-executable from its
  command-line entry point.

## Replacement witness

The correct local comparison supplies arbitrary incoming states and gives the
control its full diagonal-affine map.

For a word `w`, let its target be `P_w h`.  Any diagonal-affine control,
regardless of how its token parameters compose, yields

\[
D_w h+c_w
\]

for some diagonal `D_w` and vector `c_w`.

Evaluate on the centered basis distribution

\[
\mathcal H=\{+\sqrt m e_j,-\sqrt m e_j:j=0,\ldots,m-1\}.
\]

It has zero mean and identity covariance.  Hence the normalized mean-squared
error for a permutation `P` is

\[
\frac1m\mathbb E_{h\in\mathcal H}
\|Ph-Dh-c\|_2^2
=\frac{\|P-D\|_F^2+\|c\|_2^2}{m}.
\]

The optimal affine offset is zero and the optimal diagonal is `diag(P)`.  If
`P` has `f(P)` fixed points, the exact minimum is

\[
1-\frac{f(P)}m.
\]

For odd `m=109`, the identity has 109 fixed points, each of the other 108
rotations has zero, and each of the 109 reflections has one.  Uniformly over
all 218 group elements, even an independently optimized diagonal map for each
word has normalized MSE at least

\[
\frac{108+108}{218}=\frac{108}{109}\approx0.9908257.
\]

A single shared token recurrence is no stronger than this word-specific
relaxation.  T30's constructive setting

```text
route(R) = rho
route(F) = tau
a_R = a_F = 1
b_R = b_F = 0
```

has exactly zero error for every word and every incoming state.

This is the right deterministic separation microbenchmark.  It tests a known
operator advantage, not natural-language usefulness.  Because the operator
advantage overlaps PD-SSM, passing it can only validate our compact compiled
specialization; it cannot by itself admit a production or novelty claim.

## Next admission question

Before any new training run, write a separate natural-information argument:
name the raw-prose statistic that noncommutative routing preserves, show why a
full diagonal-affine or matched hybrid control loses it at the frozen width,
and give an overpowered reader oracle a breakthrough-size predicted margin.
If that bridge cannot be explained and falsified, T30 remains a useful exact
primitive but is closed as the active breakthrough direction.
