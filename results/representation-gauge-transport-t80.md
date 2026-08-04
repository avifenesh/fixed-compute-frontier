# T80 representation gauge transport — exact preservation, anchor impossibility, and direct collision

Date: 2026-08-02  
Status: **METHOD CLOSED AS NOVEL CANDIDATE; DIAGNOSTIC RETAINED; NO RUN**

## 1. Candidate

Continual updates can change an internal representation even when its old
task-relevant information remains decodable. A downstream reasoning or policy
module bound to the old coordinates can then fail.

The candidate was to estimate the coordinate change and transport the old
downstream computation through it:

```text
new prefix -> learned transport -> old protected continuation
```

A plastic residual path would learn genuinely new features while the
transported path preserved earlier behavior. The hoped-for gain was substantial
continual learning without freezing the whole representation or replaying every
old example.

## 2. Exact conditional preservation lemma

Write the old model as

\[
f_0(x)=g_0(h_0(x)),
\]

where `h_0` is a prefix representation and `g_0` is the protected continuation.
After learning, let the new prefix be `h_1`.

### Lemma T80.1

If there is a map `T` such that

\[
h_0(x)=T(h_1(x))
\]

for every protected input `x`, then the stitched model

\[
\tilde f(x)=g_0(T(h_1(x)))
\]

preserves the old function exactly on that set.

This is immediate by substitution. It proves preservation **if** such a map
exists on the protected set. It does not prove that the chosen transport family
contains the map, that finite anchors identify it, or that it generalizes to
uncovered inputs. In the special linear-coordinate case
`h_1(x)=G h_0(x)` with invertible `G` and old linear head `W_0`, transporting
the head as

\[
W_1=W_0G^{-1}
\]

gives `W_1 h_1(x)=W_0 h_0(x)` exactly. Full ambient invertibility is stronger
than necessary: injectivity on the protected activation subspace and a left
inverse there suffice. A translated representation requires an affine map or
homogeneous coordinates rather than this purely linear formula.

### Approximate bound

Fix a protected-input distribution `mu`, a representation norm, and an output
norm. If `X ~ mu`, `g_0` is `L`-Lipschitz between those norms, and

\[
\mathbb E\|h_0(X)-T(h_1(X))\|^2\le \epsilon^2,
\]

then

\[
\mathbb E\|f_0(X)-\tilde f(X)\|\le L\epsilon
\]

by Lipschitzness and Cauchy--Schwarz. This is a mean bound, not a pointwise or
distribution-shift guarantee.

For a classifier, write `z_0=g_0(h_0(x))`,
`tilde z=g_0(T(h_1(x)))`, let `c=argmax_j z_{0j}`, and define the old top-two
margin

\[
m(x)=z_{0c}-\max_{j\ne c}z_{0j}.
\]

The old class is certainly unchanged at `x` if

\[
\|\tilde z-z_0\|_\infty < m(x)/2.
\]

For any `tau>0`, the distributional change rate obeys

\[
\Pr[\tilde c(X)\ne c(X)]
\le \Pr[m(X)\le 2\tau]
+\Pr[\|\tilde z(X)-z_0(X)\|_\infty\ge\tau].
\]

If `g_0` is `L_inf`-Lipschitz into logit `ell_inf`, Markov's inequality bounds
the second term by `L_inf^2 epsilon^2/tau^2`. Classification preservation thus
also depends on the margin distribution and reconstruction-error tails.

These statements explain why interface alignment can recover order-one
behavior when the information survived but its coordinates moved.

## 3. Anchor-rank impossibility

Suppose a linear transport `T in R^{d x d}` is fit from paired anchor matrices

\[
H_0=T H_1,
\qquad H_0,H_1\in\mathbb R^{d\times n}.
\]

### Proposition T80.2

If `rank(H_1)<d`, the anchors do not identify an unrestricted ambient `T`
outside their span.

Choose any nonzero matrix `A` whose nullspace contains the columns of `H_1`.
Then

\[
(T+A)H_1=TH_1=H_0,
\]

yet there exists a vector `v` outside the anchor span for which
`(T+A)v != Tv`. Thus perfect anchor fit supplies no worst-case guarantee on
uncovered feature directions.

This does not say that ambient uniqueness is always operationally necessary.
Only the restriction of `T` to the protected activation support, modulo
directions ignored by `g_0`, affects the protected function. Consequences:

- an unrestricted exact `d`-dimensional transport needs anchors whose
  activations actually span every protected feature direction; `d` anchors are
  necessary for ambient full rank but are not sufficient unless their matrix
  has rank `d`;
- noisy fitting also depends on coverage and the smallest relevant singular
  value, so an ill-conditioned full-rank anchor matrix can be unstable;
- an affine transport needs the corresponding rank after augmenting activations
  with a constant coordinate;
- fewer anchors require a declared low-rank, orthogonal, sparse, or
  task-subspace assumption, and such a structured family may be identifiable
  from fewer anchors; and
- one successful two-task stitch does not establish lifetime consolidation.

## 4. Complete cost ledger

- A generic dense transport stores and applies `O(d^2)` scalars/operations.
- A factorized rank-`r` map `UV^T` costs `O(dr)` but is rank deficient for
  `r<d`; a full-rank near-identity drift model is instead `I+UV^T`.
- Retaining `n` paired anchor activations costs `O(nd)` if they are stored.
  Producing them requires both checkpoints and two prefix passes, followed by
  fitting whose conditioning and compute must be charged.
- The stitch retains and executes the old continuation `g_0`. That tail can
  dominate the small transport. A plastic residual path adds parameters and
  serving work, while multiple task-specific keys or tails can grow with the
  number of protected snapshots.
- Approximate maps can compound error when composed through large operator
  norms or poor conditioning, but this is conditional rather than inevitable;
  isometric or freshly re-estimated maps can avoid that failure.
- An expressive nonlinear stitcher may relearn protected computation or absorb
  substantial downstream work. Its capacity, labels, anchors, optimization,
  and inference cost therefore belong in the comparison; nonlinearity alone
  does not mean it contains a second model.

## 5. Why the method is closed

The central operator is directly occupied.

- [Forgetting is Not Erasure: Recovering Latent Knowledge via Transport
  Keys](https://arxiv.org/abs/2606.02860) attributes a significant portion of
  apparent forgetting to interface drift and estimates compact task-specific
  alignment operators from paired anchor activations. It evaluates exactly the
  post-update-prefix / pre-update-continuation stitch and reports recovering
  most old Task-A performance on Split CIFAR-100, with a similar result on a
  compact vision transformer.
- [Learnable Drift Compensation](https://arxiv.org/abs/2407.08536) learns
  mappings that move old prototypes into a continually changing feature space
  and reports that apparent forgetting in its tested setting need not reflect
  lost discriminative power.
- [Query Drift Compensation](https://proceedings.mlr.press/v330/goswami26a.html)
  projects queries from an updated text embedding model into the old indexed
  document space, preserving compatibility without re-indexing.
- [Model stitching](https://arxiv.org/abs/2106.07682) and later
  [invariance-aware stitching](https://arxiv.org/abs/2505.20142) already provide
  the broader learned-interface framework and its diagnostic caveats.
  [Functional Alignment Can Mislead](https://openreview.net/forum?id=glLqTK9En3)
  further shows that a successful stitch does not by itself prove that two
  representations encode the same information or differ only by a benign
  coordinate gauge.

Adding a separate residual path for new features reduces to familiar expansion,
adapter, progressive-network, or mixture-of-experts controls. It spends growing
parameters unless a later consolidation step solves the original interference
problem.

## 6. Intelligence verdict

The algebra explains one real part of the knowing--using and continual-learning
gap, and prior work suggests that part can be large. It does not create new
variables, acquire external causal evidence, improve reasoning algorithms, or
solve lifetime consolidation. More importantly, the proposed operator is not
new.

Therefore:

- retain exact transport as a control and diagnostic;
- require anchor-rank and full-cost accounting whenever a future method relies
  on interface alignment;
- do not claim a new architecture, a 20% intelligence gain, or broad continual
  learning from the theorem;
- do not run a CPU, neural, local-GPU, or rental-GPU experiment on T80.

This closure is narrow: it closes the presented anchor-fitted interface map as
a novel research candidate. It does not close all in-place online alignment,
transport-free gauge-invariant computation, or bounded consolidation.

The next candidate must alter the learning process beyond fitting an interface
map between model snapshots.
