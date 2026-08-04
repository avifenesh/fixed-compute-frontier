# T52 hidden-wiring causal recombination — alignment by intervention, not names

Date: 2026-08-01  
Status: **AUDITED FINITE-CLASS CONTROL; CONSERVATIVE BOUND SHARPENED; NO ARCHITECTURE OR EXPERIMENT**

## 0. Removed privilege

T51 supplies the composition graph and asks only which reusable mechanism sits
in each slot. T52 hides both the parent wiring and every environment's variable
names. It retains observable state variables and perfect interventions; raw
perception and latent-object discovery are not solved here.

## 1. Permuted hidden-wiring world

At each round the learner can set the full binary current state

\[
x=(x_1,\ldots,x_r)\in\{0,1\}^r
\]

and observe the next-state vector `y`. Each next-state coordinate is generated
by one library mechanism on an unknown ordered tuple of `k` distinct parents:

\[
y_{t,j}=f_{m_j}(x_{t,P_j})\oplus E_{t,j},
\qquad m_j\in\{1,\ldots,M\},
\]

where `P_j` is an ordered `k`-tuple of distinct coordinates from
`{1,...,r}` and `E_{t,j} ~ Bernoulli(eta)` with `eta<1/2`. Noise is
independent of the intervention and candidate hypothesis and iid across
rounds. Cross-node independence is needed only for the simple noisy-channel
interpretation below, not for the ERM union bound. All mechanisms in this
statement have the same binary `k`-input type; typed or self-parent constraints
must instead be reflected in node-specific legal hypothesis classes.

Variable labels may be permuted independently in every environment. The
learner does not transfer those labels; it searches over parent tuples and
aligns mechanism identities through their interventional behavior.

For one output coordinate, define the finite hypothesis class

\[
\mathcal H=
\{h_{m,P}(x)=f_m(x_P):m\in[M],P\in(r)_k\},
\]

so

\[
H=|\mathcal H|\le M(r)_k\le Mr^k.
\]

Because `H` counts functions rather than parameterizations, duplicate
mechanisms and port symmetries are already quotiented. Distinguish global
behavioral equivalence,

\[
h\equiv h' \Longleftrightarrow h(x)=h'(x)\quad\text{for every legal }x,
\]

from sampled equivalence,

\[
h\equiv_Q h' \Longleftrightarrow h(x)=h'(x)\quad Q\text{-almost surely}.
\]

Data from `Q` can identify only the latter. The separation assumption below
makes the two coincide on this finite candidate class.

## 2. Separating intervention distribution

Let interventions be drawn from a declared distribution `Q` over full states.
Assume every pair of inequivalent hypotheses is separated by

\[
d_Q(h,h')=
P_{x\sim Q}[h(x)\ne h'(x)]\ge\alpha>0.
\]

This is the exact identifiability resource. If `alpha=0`, the available
interventions cannot distinguish the pair. A uniform `Q` works only for
libraries and parent structures whose behavioral differences occupy a
nonnegligible fraction of the Boolean cube.

## 3. T52.1 — constructive recovery theorem

Draw `n` independent interventions `x_t ~ Q`, observe every `y_{t,j}`, and for
each node choose the empirical-risk minimizer

\[
\widehat h_j\in\arg\min_{h\in\mathcal H}
\widehat L_j(h),
\qquad
\widehat L_j(h)=\frac1n\sum_{t=1}^n
\mathbf 1\{h(x_t)\ne y_{t,j}\}.
\]

Let

\[
g=(1-2\eta)\alpha.
\]

Then all `r` node hypotheses are recovered up to behavioral equivalence with
probability at least `1-delta` whenever

\[
\boxed{
n\ge
\left\lceil
\frac{9}{2g^2}
\ln\frac{2rH}{\delta}
\right\rceil.
}
\]

This valid Hoeffding bound is conservative in `alpha`. Comparing each wrong
hypothesis's loss directly with the true hypothesis gives a sharper result.
For

\[
Z_t=\mathbf 1\{h(x_t)\ne y_{t,j}\}
-\mathbf 1\{h_j^*(x_t)\ne y_{t,j}\},
\]

we have `E[Z_t]=(1-2eta)d` and `E[Z_t^2]=d` when the two hypotheses disagree
on `Q`-mass `d`. Bernstein or conditional Chernoff concentration therefore
gives the natural finite-class scaling

\[
\boxed{
n=O\!\left(
\frac{\ln(rH/\delta)}
{\alpha(1-2\eta)^2}
\right).
}
\]

Even without noise, order `1/alpha` random interventions can be necessary just
to hit the disagreement region. The displayed Hoeffding constant remains the
elementary fully derived sufficient condition; it must not be reported as
separation-optimal.

One intervention round supplies all `r` node outputs, so `n` is also the
parallel environment-round count; the scalar observation count is `rn`.

**Proof.** The true hypothesis has population 0-1 loss `eta`. A wrong
hypothesis that disagrees with it on fraction `d` has loss

\[
\eta(1-d)+(1-\eta)d
=\eta+(1-2\eta)d
\ge\eta+g.
\]

Hoeffding gives, for each node-hypothesis pair,

\[
P(|\widehat L-L|\ge g/3)
\le 2e^{-2ng^2/9}.
\]

Union-bound over at most `rH` pairs. On the resulting event, the true loss is
at most `eta+g/3`, while every inequivalent candidate is at least
`eta+2g/3`; empirical risk therefore selects the true behavioral class. QED.

## 4. T52.2 — information floor

Let the complete environment class, after all declared graph constraints and
behavioral symmetries, be

\[
\mathcal G\subseteq\mathcal H^r/\!\sim.
\]

In the noiseless case, each intervention round returns `r` bits. Exact
identification therefore needs at least

\[
\boxed{
n\ge\left\lceil\frac{\log_2|\mathcal G|}{r}\right\rceil.
}
\]

rounds in the worst case, even with adaptive interventions. For one node this
reduces to `ceil(log2 H)`; when `G=H^r`, the environment-level scaling does too.

**Proof.** A depth-`n`, `r`-bit transcript tree has at most `2^(rn)` leaves.
Exact identification of `|G|` targets requires at least `|G|` leaves. QED.

With independent BSC noise and a uniform prior on `G`, error probability at
most `delta` requires

\[
nr[1-h_2(\eta)]\ge
\log_2|\mathcal G|-h_2(\delta)
-\delta\log_2(|\mathcal G|-1).
\]

Feedback through adaptive interventions does not increase BSC capacity. This
is only a floor: legal interventions may split the candidate set badly enough
to require many more rounds.

## 5. The hidden bill: sample efficiency is not computational efficiency

Direct ERM evaluates up to

\[
rH\le rMr^k
\]

node hypotheses on every sample. Its naive comparison work is the upper bound

\[
\boxed{F_{search}=O(nrMr^k),}
\]

plus storage or executable access for all `M` mechanism definitions.

This is not a computational lower bound. Cached or bit-parallel evaluation,
symmetry reduction, sparse Boolean learning, group testing, constraint solving,
or adaptive design can improve it. The full ledger must also charge library
storage and access, intervention optimization, quotienting, meta-training,
routing, and active compute. A neural amortizer must beat the best structured
inference algorithm, not merely this loop.

## 6. What the theorem genuinely gains

Given the mechanism library, T52 can identify a new local wiring diagram
without seeing all global input configurations. It then
executes the recovered composition on unseen interventions. The adaptation
rounds scale with the logarithm of the mechanism/wiring hypothesis count rather
than the `2^r` truth-table size of an unrestricted global map.

This is systematic recombination under an explicit invariant-mechanism prior.
It is stronger than retrieving a similar prose episode. It is not yet a model
claim because a finite exact causal learner already achieves it under the same
interface. Independent per-environment permutations establish equivariance to
renaming, not canonical variable alignment across worlds. Repeated laws and
graph automorphisms can make that alignment unidentifiable.

## 7. Remaining privileges and no-go boundary

T52 still assumes:

- stable observable variables rather than pixels or token descriptions;
- a known arity `k` and typed mechanism library;
- perfect full-state interventions drawn from `Q`;
- one-step observation of every node;
- independent stationary bit-flip noise;
- no latent confounders, omitted interaction mechanisms, or state aliasing;
- invariant local laws across compositions; and
- a constant or at least controlled separation `alpha`.

Without these, the same observed interventional distribution may admit multiple
slot encoders, graphs, or mechanism libraries. The appropriate target is then
an equivalence class, or additional assumptions must be paid for.

## 8. Architecture disposition

T52 supports the *possibility* of a causal mechanism bank recognized through
behavior under renaming. It does not yet prefer a Transformer, GNN, causal MoE,
program synthesizer, or exact table. Mandatory controls receive the identical
library, interventions, and raw observations.

The fair contrast is a known bounded-arity reusable-mechanism class versus an
unrestricted vector transition class under the same full-state query and
full-vector response oracle. It is not evidence that one neural architecture
learns better than another.

No architecture experiment is admitted until one candidate removes a remaining
privilege and proves a resource edge over:

1. exact finite-hypothesis ERM/search;
2. sparse causal graph discovery with the same intervention design;
3. modular graph neural world models;
4. causal representation learning with intervention targets; and
5. recurrent/meta-learned system identifiers.

## 9. Next proof obligation

The next paper decision is whether removing observable state slots is
identifiable at all:

> characterize the smallest multi-environment intervention/equivariance
> assumptions under which a raw observation encoder and reusable causal
> mechanisms are jointly identifiable, and state the unavoidable equivalence
> class when they are not.

The corrected research target and joint-discovery proof object are specified in
[T53](intelligence-gap-reset-t53.md). No CPU, local GPU, or rented GPU run is
admitted by T52.
