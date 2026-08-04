# T76 epistemic operator machine — from prose prediction to a developmental learner

Date: 2026-08-02  
Status: **INDEPENDENTLY AUDITED RESEARCH CHARTER; NO DISTINCT MECHANISM, ARCHITECTURE, OR RUN ADMITTED**

## 0. Objective

The target is not a model that merely answers more questions from its frozen
pretraining prior. It is a model whose competence changes substantially and
correctly because of post-training experience. Language is one interface, not
the definition of the system.

The desired system must jointly:

1. construct consequential alternative hypotheses;
2. choose observations or interventions that separate them;
3. update calibrated belief from the result;
4. turn repeated evidence into reusable mechanisms rather than episodic prose;
5. compose those mechanisms in new situations;
6. revise the smallest responsible mechanism after contradiction; and
7. retain and eventually consolidate what remains valid.

Memory, recurrence, search, test-time gradients, world models, causal graphs,
programs, modular networks, and ordinary Transformers are all admissible
implementations. None is the goal by name. The gate is a large improvement to
the complete resulting model under an honest information and cost ledger.

## 1. What is plausibly missing

Current systems demonstrate every component in restricted settings. The
unestablished object is a **general developmental update process** that learns
and reuses its own epistemic operations across unlike interactive worlds.

A standard language model is optimized mainly as

\[
p_\theta(x_{t+1}\mid x_{\le t}).
\]

It can simulate hypotheses, planning, and updating in its activations. But the
objective does not require a persistent, calibrated, causally testable update
state, nor does it require an observation now to improve the model's future
learning process after the transcript disappears. Scaling can make that
simulation better without making the developmental operation reliable.

The proposed missing conjunction is therefore not "a graph" or "memory." It is

> a learned algebra of epistemic operations, executed over bounded persistent
> state and trained by future lifetime competence rather than answer imitation
> alone.

This is a hypothesis to falsify. A generic recurrent model trained on the same
lifetime objective may already learn it. If so, the objective—not a structured
container—is the result.

## 2. Four constraints any real proposal must respect

### 2.1 Information cannot be inferred from prose that does not contain it

If two environments induce the same law for every complete visible passive
history under every legal passive policy but differ under an intervention, no
passive predictor can identify the correct interventional law under an equal
prior. This requires a closed information boundary: task selection, names,
timing, rewards, reset behavior, metadata, pretrained corpus recall, and prior
persistent state cannot correlate with world identity. Intervention is one
remedy; a valid structural assumption or another informative channel can also
identify the law. A better reader cannot recover information absent from all
of its channels.

### 2.2 Fixed capacity cannot absorb arbitrary unrelated knowledge forever

Let the reliably distinguishable complete persistent state be `S`, including
parameters, finite precision, optimizer, external memory, caches, and routing
state, with `|S|<=P`. Then `H(S)<=log2(P)`. If old behavior `b` must be exact,
the residual capacity is at most `log2(|E_b|)`, where `E_b` is the equivalence
class of states that preserve `b`. Arbitrary continual learning at fixed finite
state, exact zero forgetting, and zero external state is impossible. Approximate
retention instead obeys a rate-distortion tradeoff.

Substantial continual gain must therefore come from at least one charged
source: unused capacity, external state, lossy compression, increased compute,
or reusable structure. T76 targets reusable structure; it does not promise
lossless infinite learning.

### 2.3 Unstructured task learning pays for combinations repeatedly

Let there be `k` primitive names and programs of length `L`. In a deliberately
opaque no-sharing family, independently permute the answer of every complete
program and expose no primitive structure. There are `k^L` ordered programs,
an unseen program is unidentifiable, and worst-case exact coverage is
`Omega(k^L)`. This is a no-free-lunch construction, not a lower bound on a
Transformer, recurrent learner, or dense state allowed to discover sharing in
a structured family.

If a learner identifies each primitive's semantics plus one shared associative
composition law and execution is homomorphic,

\[
E(f_{i_L}\circ\cdots\circ f_{i_1})
=E(f_{i_L})\otimes\cdots\otimes E(f_{i_1}),
\]

then `O(k C_primitive+C_grammar)` identification can determine all `k^L`
programs, where the primitive and grammar identification costs are explicit.
This additionally assumes direct or identifiable program routing, stable
typing, closure, local compatibility witnesses, bounded noise, and controlled
approximation error. Every execution still costs `Omega(L)` operator
applications and carries `L log2(k)` bits of program identity. This is a
conditional description/sample separation, not a theorem that T76 learns it.

### 2.4 Revision must follow causal responsibility

Suppose a task program contains `L` independently testable operators and only
operator `j` changes. An opaque table over all length-`L` compositions can need
to change `k^L-(k-1)^L` entries: exactly the sequences containing `j`. A
representation that truly shares one instance of `j` can edit that object once
only if interventions identify causal responsibility, routing and interfaces
remain stable, all affected programs reference the same object, and shared
neural parameters do not entangle protected operators. Edit sparsity and
localized behavioral effect are separate required measurements.

The advantage is localized revision and combinatorial reuse, not a mysterious
increase in information capacity.

## 3. Candidate object: an epistemic operator machine

T76 does not freeze a particular neural architecture. It freezes functional
roles that every candidate and control must expose to intervention.

At lifetime step `t`, the system has:

- `w_t`: disposable working state;
- `b_t`: bounded structured or unstructured belief/developmental state;
- `O_t`: a bounded library of reusable update/query/composition operators; and
- fixed slow parameters `theta` during the first deployment stages.

The generic transition is

\[
(a_t,w_{t+1},b_{t+1},O_{t+1})
=F_\theta(o_t,f_{t-1},w_t,b_t,O_t).
\]

The candidate realization has five separable roles:

1. **Hypothesis constructor** — creates or retrieves competing executable
   explanations of observations, with explicit uncertainty.
2. **Experiment selector** — chooses a legal action by expected reduction in
   future loss, not merely immediate reward or token likelihood.
3. **Update executor** — applies learned typed evidence-update operators to the
   responsible state factors.
4. **Composer** — builds a new update/execution program from previously learned
   operators; it may be neural, symbolic, or hybrid.
5. **Consolidator** — promotes stable operators or state only when replayed
   witnesses show improvement, calibration, and protected-task retention.

The language model can remain the semantic encoder, decoder, and prior. The
new research object is the developmental loop around and possibly inside it.

## 4. Why a factorized belief state is a candidate, not an assumption

For `n` variables of bounded domain size `d`, `O(n)` factors, and a known
junction tree of treewidth `w`, exact inference uses
`O(n d^(w+1))` time and comparable worst-case potential/message storage, versus
`O(d^n)` for an explicit joint table. Discovering the graph/elimination order is
not included, and local evidence may still require global message propagation.
A learned dense vector may discover an equally compact code and must be allowed
to do so. The structured candidate earns its place only if it provides one of
the following under the same byte/work cap:

- calibrated multi-hypothesis tracking that a dense state fails to learn;
- sparse local updates after a mechanism change;
- systematic depth/width extrapolation; or
- causal state interventions with predictable downstream effects.

No claim is based on comparing a factor graph with a deliberately inefficient
enumerated table.

## 5. Training distribution: compounds, not isolated atoms

T73, T74, and T75 are calibration units. They are not the final curriculum.
Current evidence reports stronger module recovery from compound reasoning
traces than from isolated atomic modules. The decisive training object should
therefore be a **composition lattice**:

- multiple non-isomorphic world families: deterministic mechanisms,
  stochastic causal factors, guarded changes, resource constraints, spatial or
  graph dynamics, and tool/API state transitions;
- many compounds containing overlapping subsets of epistemic operations;
- held-out edges, orders, depths, bindings, and cross-family compositions;
- post-training random mechanisms that cannot be recalled from pretraining;
- interventions whose outcomes, not prose descriptions, ground the new world;
  and
- hidden changes that require targeted revision while protecting unrelated
  knowledge.

Training on isolated components remains an ablation. The key causal comparison
is untouched base versus single-family versus diverse-compound versus
counterfactually relabeled compound curricula from the same initialization.

## 6. Lifetime objective

The minimum objective gives credit to the whole epistemic loop:

\[
J=\mathbb E\left[\sum_{t=1}^{T}
  \ell_{pred,t}
 +\lambda c(a_t)
 +\alpha\ell_{cal,t}
 +\beta\ell_{revision,t}
 +\rho\ell_{retention,t}
 +\gamma\ell_{compression,t}
\right],\qquad \text{minimize }J,
\]

where `T` or its distribution, all nonnegative weights, units, protected-task
sampling, state precision, and sources of randomness are frozen. `compression`
is not smaller state for its own sake. It rewards a shorter
state/operator description only when future predictive and interventional
behavior is preserved. `revision` scores whether a contradiction changes the
responsible mechanism and not protected ones. Every term must have an ablation;
otherwise the project name absorbs causal credit.

Full-lifetime BPTT, detached state, test-time gradients, RL, behavior cloning,
and explicit Bayesian/program controls are separate factors. Actual training
and serving work is measured, not assumed equal.

## 7. Breakthrough gate

A synthetic pass is not the goal. Before calling the resulting model more
intelligent, the same frozen developmental mechanism must pass all three
levels.

### Level 1 — mathematical/operator validation

- prove identifiability, sufficient-state, active-query, and composition
  boundaries for each synthetic family;
- show unseen composition or revision gains that grow with program depth or
  number of mechanisms rather than a fixed percentage;
- causally intervene on evidence, state, and operators and observe the
  predicted localized changes; and
- beat generic recurrent, context, retrieval, test-time-update, router, and
  explicit modular controls under a complete ledger.

### Level 2 — broad interactive transfer

Without changing the updater or gates, transfer to at least three unlike
domains such as hidden-rule software tools, scientific/causal environments,
and embodied or visual state transitions. Language-only variants do not satisfy
this level.

Require, with simultaneous intervals:

- at least `30%` lower lifetime regret than the strongest equal-served-cost
  learner;
- at least `2x` fewer interactions to a fixed competence;
- at least `10x` composition-horizon extrapolation over the training depth;
- no protected-domain loss greater than one absolute point; and
- calibrated recovery after hidden mechanism changes.

### Level 3 — resulting-model improvement

Compare complete systems, not components. At a predeclared serving envelope,
the developmental model must beat the strongest static/larger, full-context,
retrieval, agentic-search, and adaptive baselines by an order-one margin on a
fresh one-shot evaluation. Parameter count, persistent bytes, update FLOPs,
traffic, latency, intervention cost, offline consolidation, and tuning trials
are all charged.

Only Level 3 supports the scoped phrase **substantially more capable adaptive
system on the preregistered environment distribution**. If the backbone stays
frozen, the result belongs to the complete system, not to improved foundation-
model parameters or universal intelligence.

## 8. Current-art boundary

This direction overlaps established work and makes no component-novelty claim:

- [ORBIT](https://arxiv.org/abs/2602.04089) already establishes cross-episode
  online meta-RL gains in unseen interactive environments.
- [From Reasoning Traces to Reusable Modules](https://arxiv.org/abs/2606.18089)
  formalizes latent reusable modules and reports that SFT plus RL can recover
  and recombine them; compound traces outperform isolated atoms.
- [Compositional meta-learning through probabilistic task
  inference](https://arxiv.org/abs/2510.01858) explicitly represents tasks as
  compositions of reusable computations and infers new tasks probabilistically.
- [WorldEvolver](https://arxiv.org/abs/2606.30639) revises episodic and semantic
  deployment context while keeping model parameters frozen.
- [ACE](https://arxiv.org/abs/2510.04618),
  [Memp](https://arxiv.org/abs/2508.06433), and
  [Supersede](https://arxiv.org/abs/2606.27472) make evolving playbooks,
  procedural memory correction/deprecation, and bounded self-maintained memory
  direct anti-prose controls.
- [Self-Evolving Cognitive Framework via Causal World
  Modeling](https://arxiv.org/abs/2606.22449) explicitly frames interaction as
  hypothesis generation, intervention, and continual causal refinement.
- Current agentic-automata and mechanistic-world-model work already identifies
  experiment selection, evidence integration, hypothesis construction, active
  inquiry, and localized revision as open integration problems.

The research frontier is not naming these pieces. It is learning one reusable
developmental operator system, proving the source of its gain, and showing a
large complete-model advantage across unlike post-training worlds.

## 9. Immediate disposition

T75 Stage A remains worthwhile only as a cheap integrity test for exact belief
update and active acquisition. Its neural test is not automatically admitted.
A two-primitive threshold world cannot establish T76.

The audited T76 transition is a role signature, not a mechanism: generic
recurrent meta-RL, probabilistic modules, evolving text memory, or an explicit
program system can all instantiate it. The next proposal must first specify:

1. an operator interface shared across at least three non-isomorphic families;
2. a composition lattice rather than two isolated atoms;
3. a theorem predicting a scaling separation as mechanisms or depth grow;
4. a generic-state control capable of discovering the same code;
5. an explicit modular/program control capable of hand-composing it; and
6. a path from synthetic operator validation to non-language interactive
   transfer without changing the learned updater.

It must also freeze exact operator semantics, state bytes/precision, discovery,
update, composition, consolidation/rollback, transcript purge, and the boundary
between slow parameters and deployment state. The primary durability test is a
fresh process: learn a post-freeze mechanism, destroy transcript and undeclared
caches, reload only bounded state, reuse it in new bindings and an independently
generated domain, revise it after a hidden change, and retain unrelated learned
mechanisms.

Anti-prose and anti-generator controls are mandatory: equal-byte evolving text,
canonical rerendering, shuffled cross-domain correspondences, independently
authored held-out generators, operator lesion/swap/rebinding, and a naturalistic
domain not emitted by the training lattice.

The first admissible successor is a theorem-plus-manifest for one distinct
state/update algorithm. No neural training, local GPU use, or rental is
admitted by this charter.
