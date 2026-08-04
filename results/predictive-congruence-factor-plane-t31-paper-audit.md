# Predictive-congruence factor plane T31 — paper audit

Status: **CLOSED ON PAPER; NO IMPLEMENTATION OR GPU RUN**  
Date: 2026-07-31

## Verdict

Predictive equivalence gives a canonical, raw-behavioral meaning for a
language state. It cleanly solves one problem that defeated earlier semantic
planes: the state is defined by observable future behavior rather than by a
post-hoc name assigned to a hidden vector.

It does **not** by itself give a smaller or smarter language model. A product
factorization can rename exponentially many predictive states using a linear
number of bits, but a modern finite-precision hidden vector is already a
distributed bit carrier. Unless both token transitions and emissions have
bounded interaction order, their tables recover the full joint-state cost.
Raw next-token observations also do not identify the desired product factors.

The broad T31 direction is therefore closed before code. No synthetic
automaton run, natural-model run, or H100 microbenchmark is justified.

## P0: proposed edge

The proposed edge was A/C/D: factor an exact or approximate predictive state
into many independently updated digital coordinates, so a fixed-width model
could carry combinatorially many useful configurations while keeping served
weights, state bytes, active work, and latency fixed.

The intended chain was

```text
raw history -> predictive equivalence class -> product factors
            -> sparse factor updates -> next-token distribution
```

The first arrow is canonical. The second and third arrows contain the missing
assumption.

## P1: exact predictive quotient

Let `Sigma` be a finite token alphabet and let `P` be a stochastic language.
For histories with positive probability, define

\[
u\sim_P v
\quad\Longleftrightarrow\quad
P(w\mid u)=P(w\mid v)\quad\text{for every }w\in\Sigma^*.
\]

### Theorem 1: right congruence

If `u ~P v` and `P(a | u)>0`, then `ua ~P va`.

**Proof.** Equality for one-token suffix `a` gives
`P(a|u)=P(a|v)>0`. For every suffix `w`,

\[
P(w\mid ua)
=\frac{P(aw\mid u)}{P(a\mid u)}
=\frac{P(aw\mid v)}{P(a\mid v)}
=P(w\mid va).
\]

Zero-probability transitions may be omitted from the reachable quotient. A
token therefore induces a deterministic transition on predictive classes.

### Theorem 2: minimal predictive sufficiency

Let `T(u)` be any statistic sufficient for the complete future: equal `T`
values imply equal conditional distributions over all suffixes. Then

\[
T(u)=T(v)\Longrightarrow u\sim_P v.
\]

Hence the predictive class `[u]` is a function of every sufficient statistic;
it is the coarsest exact predictive partition. This is a semantic contract,
not a compression result.

### Theorem 3: phrase operators compose

For a string `w`, define `tau_w([u])=[uw]`. Right congruence makes this map
well defined, and

\[
\tau_{xy}=\tau_y\circ\tau_x.
\]

Thus phrases can be represented as composable state operators. Weighted
automata, predictive-state representations, and recurrent linear models
already occupy this algebraic family; the composition law is retained as a
useful specification, not claimed as a discovery.

## P2: explainable epsilon guarantee

Suppose an approximate model `Q` satisfies, for every reachable true history
`h`,

\[
D_{KL}\!\left(P(\cdot\mid h)\,\|\,Q(\cdot\mid h)\right)\le\epsilon.
\]

The KL chain rule gives, for every horizon `H`,

\[
D_{KL}(P(X_{1:H})\|Q(X_{1:H}))
=\sum_{t=1}^{H}\mathbb E_P
 D_{KL}(P(X_t\mid X_{<t})\|Q(X_t\mid X_{<t}))
\le H\epsilon.
\]

This is an understandable error budget: local predictive error accumulates at
most linearly in KL. It does not construct the approximate states.

A naive tolerance quotient is invalid. For Bernoulli distributions with
parameters `0`, `delta`, and `2 delta`, adjacent total-variation distances are
`delta`, while the endpoints are `2 delta` apart. Therefore “distance at most
`delta`” is not transitive and does not define equivalence classes. A valid
approximation needs an explicit cover, congruence closure, or behavioral
pseudometric, each with its own state/error cost.

## P3: fatal matched-baseline lower bound

### Independent-fact witness

Let a history reveal `n` independent bits `b_1,...,b_n`. Give every index `j`
positive probability of a future query whose answer is `b_j`. Two histories
that differ in bit `j` induce different probabilities for that query-answer
suffix. They are therefore in different predictive classes.

The exact quotient has `2^n` states. Yet the obvious factor representation is
only the tuple of `n` bits. This sounds exponential until compared with the
right baseline: any exact finite state with `B` bits has at most `2^B`
configurations, so `B >= n`. The tuple meets, but does not beat, the
information lower bound. A `d`-coordinate state with `b` physical bits per
coordinate already has up to `2^(bd)` configurations and needs only
`d >= ceil(n/b)` on this witness.

Modern Transformer and recurrent hidden states are distributed numerical
states, not one-hot automaton states. Replacing a one-hot vector of length
`2^n` by `n` factors therefore defeats a straw baseline.

### Interaction-cost theorem

Let the proposed factors have sizes `n_1,...,n_k` and joint state count

\[
N=\prod_i n_i.
\]

An unrestricted categorical emission over vocabulary size `V` has
`N(V-1)` independent real degrees of freedom: every joint state can choose an
interior point of a `(V-1)`-dimensional simplex. A regular continuous
parameterization with fewer than `N(V-1)` scalars cannot cover an open set of
these emission tables. Arbitrary joint-state transitions have the same
problem.

Consequently, product coordinates reduce the state description only when the
language also has a declared restriction such as

\[
\operatorname{logit}P(x\mid s_1,...,s_k)
=c_x+\sum_i f_{i,x}(s_i)
+\sum_{(i,j)\in E} f_{ij,x}(s_i,s_j),
\]

with bounded factor sizes and bounded-degree interaction graph `E`, plus
similarly local token transitions. This low-interaction law is precisely what
T31 needed to discover from raw prose; assuming it would assume the result.

## P4: identifiability obstruction

The exact predictive quotient is unique up to state renaming. A decomposition
of one quotient state into coordinates is not. Invertible recodings can mix
the coordinates while preserving every observed continuation probability.
Even a compact exact quotient therefore supplies no observational rule saying
which coordinate is an entity, relation, fact, or independently writable
factor.

This is the same missing bridge in a cleaner form:

```text
observable future behavior -> unique joint predictive state        yes
unique joint predictive state -> useful independent factorization  no
```

General causal-representation identifiability results likewise require extra
conditions such as interventions or paired counterfactual structure. Raw
prose may contain natural variation that acts like an intervention, but T31
provides no raw-computable detector or recovery theorem for it.

## P5: prior-art boundary

- [Congruence-based learning of PDFA](https://arxiv.org/abs/2412.09760)
  explicitly constructs language-model quotients from future-distribution
  congruences and proves their minimality properties.
- [Connecting weighted automata and RNNs](https://proceedings.mlr.press/v89/rabusseau19a.html)
  proves an expressivity equivalence between weighted finite automata and
  linear second-order RNNs.
- [Quantum tensor networks, stochastic processes, and weighted automata](https://proceedings.mlr.press/v130/adhikary21a.html)
  connects predictive-state representations, HMM-like models, tensor networks,
  and weighted automata.
- [Future Summary Prediction](https://openreview.net/forum?id=aeYIFVn4vb) and
  [ContextLM](https://openreview.net/forum?id=ARA2xneUDG) already train causal
  LMs against longer-horizon future summaries while removing or amortizing
  auxiliary machinery at ordinary autoregressive serving.

Predictive quotients, tensor/product sequence models, and future-prediction
training are therefore established blocks. A contribution would need a new
raw-identifiable low-interaction factor law and a fixed-cost realization, not
their recombination by name.

## P6: why no microbenchmark is run

The right-congruence, composition, KL, state-count, and interaction claims are
closed algebra. A toy product automaton would only confirm the implementation
of known constructions. It cannot test the two failed premises:

1. that natural-language predictive states admit a bounded-interaction product
   decomposition; and
2. that the factors are recoverable from raw observational prose.

Running a synthetic factor world would manufacture both premises in the data
generator. Running an LM would leave two important empirical miracles at
once. Both violate the proof-first admission contract.

## Retained result and reopening condition

Retain predictive equivalence as a **semantic specification** for future
states and phrase operators. Do not retain the factor-plane architecture.

Reopening requires one new object, defined before training:

```text
raw corpus -> observable factor witnesses -> bounded interaction graph
```

It must include a recovery theorem, an ambiguity counterexample, a natural
census showing that interaction degree does not grow with model width or
corpus size, and a matched distributed-state baseline. Only then would small
exhaustive worlds and a natural factor microbenchmark be informative.

## Key shift

Do not ask whether language has fewer predictive states than histories. It
does. Ask whether its **predictive interaction order stays bounded as useful
knowledge grows**. State count is not the scarce resource; independently
addressable interactions, their identification, and their served execution
are.
