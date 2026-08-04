# T53 intelligence-gap reset — from prediction to self-maintaining models

Date: 2026-08-01  
Status: **INDEPENDENTLY REVISED RESEARCH RESET; FINITE NEXT PROOF OBJECT; NO ARCHITECTURE OR RUN**

## 0. Corrected target

The target is not a cheaper Transformer and not a better prose compressor. It
is a model that becomes materially more capable through experience: it forms
new abstractions, tests uncertain beliefs, transfers learned mechanisms into
new compositions, repairs the part that failed, and retains unaffected
knowledge.

Efficiency remains a constraint, not the research object. A successful model
may trade dense compute for bounded persistent state, sparse mechanism storage,
interaction, or consolidation work, but the complete bill must be measured.

## 1. What raw prediction cannot guarantee

### T53.1 — passive causal non-identifiability

Let two environments `E0,E1` induce the same distribution over every passive
observation history,

\[
P_{E_0}(O_{1:T})=P_{E_1}(O_{1:T})
\quad\text{for all }T,
\]

but disagree after some legal intervention `a`,

\[
P_{E_0}(O'\mid do(a))\ne P_{E_1}(O'\mid do(a)).
\]

For an equal prior over the two environments and no environment-correlated side
information, every learner trained only on passive histories has posterior
error `1/2` when asked which interventional law is present.

**Proof.** The passive-data likelihood ratio is one for every history, so the
posterior remains the prior and `I(E;O_{1:T})=0`. Therefore no binary
environment-identification rule can succeed with probability greater than
`1/2`. QED.

**Witness.** Let `U ~ Bernoulli(1/2)`. In `E0`, set `X=U` and `Y=X`. In `E1`,
set `X=U` and `Y=U`. Passive observations of `(X,Y)` are identical. Under
`do(X=0)`, however, `E0` gives `Y=0`, while `E1` leaves `Y=U`.

The statement is deliberately architecture-independent. Infinite parameters
or context cannot recover information absent from the declared observation
channel. Interaction, experimental variation, environment-correlated side
information, or a causal assumption must enter somewhere. Passive text can
contain reports of other people's experiments, so “passive to the model” is
not the same thing as “observational-only” in this theorem.

### What this does not prove

It does not show that Transformers cannot represent causal models, that every
task needs physical action, or that active learning is novel. It rules out a
universal guarantee of identifying intervention laws from an observation
channel that assigns identical distributions to causally different worlds.

## 2. What present systems already contain

As of August 2026, the ingredients cannot individually be called the missing
breakthrough:

| ingredient | present control | remaining failure |
|---|---|---|
| persistent experience | retrieval, episodic/program memory, modular memory | stored episodes are not necessarily an executable abstraction |
| online adaptation | context learning, fast weights, TTT, meta-RL | updates can be transient, globally interfering, or uncalibrated |
| world prediction | latent and video world models | predictive state can preserve correlation without identifiable mechanisms |
| causal structure | causal representation/discovery and modular causal models | usually assumes interventions, variables, targets, or environment changes that the learner did not discover |
| active reasoning | search, planners, verifiers, epistemic policies | the model may search inside an incorrect ontology and cannot localize durable repair |

The gap is therefore not absence of one component. The defensible claim is
that the full **knowledge lifecycle** remains unclosed: no demonstrated general
system reliably joins these operations under raw partial observations, weak
assumptions, matched controls, and complete-cost accounting.

## 3. Working missing object

The current hypothesis is a **self-maintaining executable model**. Its durable
knowledge is organized as uncertain, reusable mechanisms and their bindings,
not only as diffuse parameter correlations or retrieved prose. It must learn
five operations:

1. **Induce:** propose variables and mechanisms from raw, partial observations.
2. **Bind:** assemble known mechanisms into an executable model of the current
   situation without relying on surface names.
3. **Interrogate:** choose actions that distinguish consequential hypotheses.
4. **Revise locally:** attribute surprise to a variable, binding, mechanism, or
   change point and update only the responsible knowledge.
5. **Consolidate:** merge repeated discoveries into reusable abstractions while
   preserving provenance, uncertainty, and old competence.

This is not yet an architecture. A graph, program library, causal MoE, memory
tree, or neural compiler earns admission only if its learned interface solves
these operations better than a generic recurrent/meta-learned system at the
same complete cost.

## 4. Why this object could produce a large gain

Assume an environment is generated by `d` reusable mechanisms and a new world
changes or recombines only `s << d` of them. If the correct mechanism boundary
and binding are available, the learner can preserve the `d-s` unchanged
components exactly and spend update samples/work on the affected subset. It
can also reuse a learned mechanism in exponentially many global compositions.

An unconstrained vector transition table over `r` binary variables has `2^r`
entries containing `r 2^r` output bits. A bounded-arity mechanism graph with
`r` nodes, library size `M`, and arity `k` has the conditional code length

\[
DL(\mathcal L)
+\sum_{j=1}^r
\left[\log_2 M+\log_2(r)_k\right],
\]

before charging types, ports, observation maps, variable encoders, action
semantics, bindings, noise parameters, and the interpreter. An arbitrary
Boolean mechanism library can itself cost `M 2^k` bits, and its cost can be
amortized only across a declared number of environments. T52 shows that, when
the variables and intervention interface are supplied and hypotheses are
separated, a new wiring can be recovered in rounds logarithmic in `Mr^k`.

That is the possible source of a substantial edge: **reuse and local repair**,
not a fractional kernel improvement. But this is a hypothesis-class advantage,
not yet an architecture advantage. If learning the variables, mechanisms,
bindings, compiler, or router costs as much as learning the global map, the
advantage disappears. Exact preservation of unchanged mechanisms additionally
requires stable encoders, interfaces, bindings, routing, and planning—or
explicitly versioned changes to each.

## 5. The real unresolved seam

Recent work already proposes mechanism-centric world models, modular causal
components, continual modular memory, causal representation learning, and
active world-model discovery. Their own boundary exposes the unresolved seam:

> jointly discover useful variables, reusable mechanisms, and their bindings
> from partial raw observations; actively remove consequential ambiguity; then
> revise this structure continually without global forgetting.

This conjunction is not an empty part of the literature. Mechanistic World
Models states it as a program; Explicit Symbolic Behavioral Models combines
typed mechanisms, active probes, and local edits; causal-representation work
jointly learns latent state and controlled dynamics under declared excitation;
and online world models address continual planning. The open claim is the full
conjunction under weak assumptions and complete cost, not any ingredient or
the slogan “self-maintaining model.”

This is harder than adding memory to an LLM. The variable ontology can change,
multiple latent explanations can be observationally equivalent, and any
front-end "compiler" can hide the full intelligence problem. A weaker compiler
damages the mechanism layer; a stronger one should be charged as the model.

## 6. Exact next proof object

Do not implement the full system. First solve the smallest joint-discovery
problem that T52 deliberately avoided.

### Finite partial-observation recombination world

Use finite alphabets and the declared family

\[
z_{t+1,j}\sim
p_{\theta_{e,j}}(\cdot\mid z_{t,Pa_{e,j}},a_t),
\qquad
o_t\sim g_e(\cdot\mid z_t).
\]

Freeze the number of latent variables, bounded indegree, shared finite
mechanism library, at most `s` changed mechanisms per environment, observation-
map class, reset semantics, training action policy, deployment query/loss,
noise, confounding assumptions, and action coverage. The learner cannot
directly inspect or set latent state slots.

### Required theorem pair

1. **Impossibility:** construct two worlds indistinguishable under the frozen
   training policy but different on the declared deployment action or query.
   State the unavoidable target—action-observation bisimulation, latent
   permutation, or a stricter declared quotient. Worlds equivalent under every
   legal policy cannot require observably different plans in that interface.
2. **Sufficiency:** choose one quantified addition, initially an action-
   excitation/separation margin, that breaks the pair. Give an identification
   algorithm and sample, computation, interaction, and persistent-memory bounds
   against an equally informed Bayesian, PSR, or POMDP learner. Do not call an
   assumption “weakest” without defining an ordered assumption family.

### Admission test

A candidate survives only if the reusable representation yields a strict
separation over the strongest generic learner, not merely over a truth table:

- at least `2x` fewer interactions or `30%` lower cumulative regret on unseen
  recombinations—empirical substantiality gates, not theorem-derived constants;
- local changes pass a simultaneous noninferiority test across protected tasks
  and repeated update times;
- disabling reuse or local attribution causally destroys the gain;
- perception, compiler, proposal search, interventions, verifier/oracle,
  attribution, replay, consolidation, routing, and storage costs are charged;
- the same learned interface transfers across several world families.

## 7. Closed shortcuts

- More context or retrieval alone stores evidence but does not resolve causal
  non-identifiability.
- A learned test ranker without information is prior amortization (T50).
- A supplied mechanism library and observable state slots reduce to structured
  identification (T51/T52).
- A residual memory tree is not novel by itself, and the old shared-weight
  delta-MoE premise is empirically closed in this workspace.
- A hand-authored symbolic ontology simply moves the intelligence into the
  designer.

## 8. Decision

No breakthrough architecture has been found yet. The surviving research claim
is narrower and stronger than the previous transformer-compute search:

> the high-leverage target is a learned, executable, self-revising knowledge
> organization that turns interaction into transferable mechanisms while
> preserving unaffected competence.

The next result must establish when that organization is identifiable and
cheaper to acquire than an equally informed generic learner. Until then there
is no reason to run a local model, GPU, or rented box.
