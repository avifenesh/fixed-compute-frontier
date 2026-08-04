# T71 developmental-intelligence gap — optimize the learner, not only its answers

Date: 2026-08-02  
Status: **INDEPENDENTLY REVISED OBJECTIVE RESET; T70 DEMOTED TO A MEMORY-CHANNEL PROBE; NO MODEL MECHANISM SELECTED; NO RUN**

## 0. Corrected objective

The target is a substantially more intelligent resulting model, not a cheaper
way to process prose and not a new memory mechanism by itself. Language can be
an input/output surface, but the model must improve as a learner:

1. acquire facts, procedures, and environment models from post-training
   experience;
2. choose actions that resolve important uncertainty;
3. form abstractions that transfer beyond the observed instances;
4. reason with those abstractions beyond memorized trace depth;
5. revise the responsible belief or procedure after contradictory outcomes;
6. retain unrelated knowledge; and
7. eventually consolidate reusable experience without retraining from zero.

Any architectural, objective, data, memory, search, recurrent, modular,
symbolic, or hybrid change is admissible. Novelty of a component is not the
gate. A large, reproducible improvement to the complete model-system is.

T70 tested one prerequisite: whether a bounded state can carry new information
through erasure. Its lookup smoke says yes, but also shows that direct
cross-boundary writer credit is unnecessary for finite lookup. T70 therefore
does not own the mission. It is one diagnostic inside this larger program.

## 1. What current systems do and do not establish

The claim cannot be that current models have no memory, no world model, no
reasoning, or no online adaptation. Current primary results already occupy
each component:

- [ORBIT](https://arxiv.org/abs/2602.04089) trains cross-episode online
  learning and reports that Qwen3-14B can match GPT-5.2 on its unseen
  interactive environments.
- [Online Experiential Learning](https://arxiv.org/abs/2603.16856) repeatedly
  extracts deployment experience and consolidates it into model parameters.
- [Self-Consolidating Language Models](https://arxiv.org/abs/2605.07076)
  meta-trains sparse self-selected weight updates over evolving streams and
  reports acquisition/retention gains over prompting, summarization, batch
  test-time training, and sequential fine-tuning.
- [Self-Consolidation for Self-Evolving Agents](https://arxiv.org/abs/2602.01966)
  is a direct reflection-plus-parameter-consolidation control.
- [ALMA](https://arxiv.org/abs/2602.07755) meta-learns executable agent-memory
  designs across sequential-decision domains.
- [T3](https://arxiv.org/abs/2510.12264) reports gains up to 30 points across
  five tasks from controlling belief deviation during active reasoning rather
  than merely extending trajectories.
- [RLAD](https://arxiv.org/abs/2510.02263) directly trains models to propose
  and reuse reasoning abstractions.
- [In-Place TTT](https://arxiv.org/abs/2604.06169) makes test-time weight
  adaptation a concrete control rather than a hypothetical capability.
- [Agentic automata learning](https://arxiv.org/abs/2606.16576), however,
  reports sharp degradation as hidden-world complexity grows and identifies
  failures in experiment selection, evidence integration, and hypothesis
  construction.
- [Mechanistic World Models](https://arxiv.org/abs/2607.12474) identifies
  reusable explanatory mechanisms, active inquiry, and localized revision as
  an integration target, not a completed general system.
- A July 2026 [self-improving-agent survey](https://arxiv.org/abs/2607.13104)
  already formalizes self-induced update operators at the system level. That
  is a terminology and design-space collision, not evidence that the complete
  capability conjunction has been solved.

The evidence therefore supports a narrower, explicitly unestablished target:

> Existing results establish strong instances of the component operations, but
> the cited evidence does not yet establish one system that jointly acquires,
> actively tests, revises, transfers, retains, and consolidates knowledge
> across a declared broad task distribution under matched information and
> full-cost controls.

T71 treats that conjunction as an evaluation target, not as a novel component
or a proven universally missing mechanism. It may be solved by an ordinary
recurrent architecture under the right lifetime objective. If so, that is a
valid result.

## 2. The key distinction: inference state versus developmental state

A usual deployed model implements an answer map

\[
a_t\sim p_\theta(a\mid c_t),
\]

where `theta` is fixed and `c_t` is a bounded current context. Longer context,
retrieval, chain of thought, and search can make this map much stronger. They
do not by themselves train the model to improve its future update process.

A developmental model has at least three timescales:

\[
(a_t,w_{t+1},m_{t+1})=
L_\theta(o_t,f_{t-1},w_t,m_t),
\]

where:

- `w_t` is disposable working state for the present problem;
- `m_t` is bounded developmental state that survives interactions; and
- `theta` is meta-trained general skill and prior knowledge.

Stages A--C below keep `theta` fixed at deployment and test only working state
and bounded developmental state. Slow parameter consolidation is a later stage
and is not represented by this operator. It will require its own typed update,
optimizer/replay state, feedback source, rollback/provenance rules, retention
test, and complete cost ledger before admission.

The important change is the objective, not the symbols. Train over complete
lifetimes:

\[
J(\theta)=\mathbb E_z\left[
\sum_{t=1}^{T}
\ell(a_t,z)+c(a_t)+
\lambda\,\mathrm{CalErr}_t+
\rho\,\mathrm{Forget}_t
\right].
\]

Later competence supplies credit to earlier observation, hypothesis, action,
and update choices. Transcript erasure, surface randomization, interventions,
change points, and protected old tasks prevent the objective from collapsing
to answer imitation or replay.

This objective permits many realizations: latent recurrent state, fast weights,
test-time gradients, an episodic store, a program or factor graph, a learned
optimizer, or a hybrid. The strongest equal-cost realization is the model, not
a baseline to defeat merely because it lacks a new name.

## 3. Three mathematical boundaries

### T71.1 — static prediction cannot learn post-training randomness

Let an environment draw a hidden bit `Z` uniformly after training. Assume the
current input and reset model state are independent of `Z`. Any fixed predictor
has accuracy at most `1/2`. If one interaction reveals `Z` and a single
persistent bit survives, all later answers can be correct.

For `R` repeated queries, expected errors fall from `R/2` to `1/2`, an error
ratio of `1/R`. This is an order-one separation, but proves only the value of
information persistence—not abstraction or intelligence.

### T71.2 — passive equivalence blocks causal identification

Let environments `z_0,z_1` induce the same distribution over every passive
history under every allowed passive collection policy but different outcomes
under some legal action `a*`. No learner using only passive histories can
identify which action law holds: its posterior odds remain its prior odds. If
the post-intervention outcome laws have nonzero KL or total-variation
separation, repeated interventions can identify the world with the ordinary
hypothesis-testing dependence on that separation and the target confidence.

This is an information boundary. More parameters or a better prose predictor
cannot recover an unobserved intervention law. A capable learner must either
act, receive equivalent side information, or state the structural assumption
that makes passive identification possible.

### T71.3 — zero-shot output-label grounding is impossible under arbitrary permutation

Let meta-training data contain families `F_1,...,F_k`, and let held-out family
`F_*` emit one of `K` opaque labels. Construct `K!` coupled test worlds that are
identical on all meta-training data and all unlabeled `F_*` observations but
differ by an arbitrary permutation of the `F_*` output labels. Before any
`F_*` grounding example, reward, semantic side information, or informative
feedback, the learner has the same state distribution in all these worlds.
Averaged over a uniform output permutation, its exact-label accuracy is `1/K`.

This does not rule out learning after held-out-family feedback. It shows only
that a zero-shot opaque-label score cannot identify reusable learning
separately from label grounding or prior knowledge.

The lookup/affine/parity/DFA rotation remains useful as an online stress test,
including acquisition regret after feedback. It is not a decisive transfer
experiment until the transferable object and common meta-distribution are
declared. Hold the present manifest for that reason, then test held-out
**compositions, bindings, surfaces, horizons, and mechanism changes** within a
declared grammar. Treat a foreign natural domain as external validity, not as a
consequence of the permutation theorem.

## 4. The research object: a developmental update operator

The mechanism-neutral evaluation object is an update policy

\[
U_\theta:(m_t,o_t,a_t,f_t)\mapsto m_{t+1}
\]

that must learn four operations under one lifetime objective:

1. **compress:** replace repeated evidence with a predictive rule or sufficient
   state when this improves future performance;
2. **query:** choose an observation or action that separates consequential
   hypotheses;
3. **revise:** change the smallest responsible knowledge needed to explain new
   outcomes; and
4. **retain:** stabilize reusable rules inside the declared bounded state while
   protecting unrelated state. Parameter consolidation is deferred to the
   later typed stage described in Section 2.

The first scientific question is not which container stores `m_t`. It is
whether one trainable update policy can acquire these operations and transfer
them to unseen compositions better than answer-level training, full context,
retrieval, generic recurrence, and test-time gradients under equal information
and complete cost.

## 5. Valid first experimental program

### Stage A — world grammar, before a model

Define a small executable grammar whose programs compose reusable mechanisms:
finite maps, affine transforms, Boolean predicates, counters, guarded state
transitions, sparse causal factors, and change points. Rendering maps latent
variables to freshly permuted symbols or observations. Legal actions can
observe, intervene, or answer, each with a cost.

The grammar is an experimental assumption and must be charged. It creates the
shared structure that makes transfer possible. Program descriptions, latent
variables, family IDs, and factor boundaries remain hidden from every learned
system.

Freeze latent-program/AST and factorial-composition holdouts, post-training
random bindings, and template-fingerprint checks. Bound cross-boundary state
below verbatim experience size. Include byte-matched example/retrieval stores,
full context as an information ceiling rather than a cost-matched winner, and
task-optimal Bayesian/program-induction or classical algorithmic controls where
available. Counterfactual queries must require the inferred rule; targeted
rule/state swaps must change exactly the corresponding predictions.

Before training, prove for each frozen subfamily:

- which hypotheses are observationally equivalent;
- the minimum interventions needed to separate them;
- the sufficient-state bit bound;
- the exact algorithmic oracle and random/memorization ceilings; and
- which held-out compositions are identifiable from the allowed interaction.

### Stage B — same-backbone objective ladder

Use one small backbone and fixed resource envelopes. Compare:

1. ordinary sequence/answer training;
2. full-context in-context learning;
3. byte-matched retrieval or episodic memory;
4. generic recurrent state with only next-step loss;
5. the same state trained on the lifetime objective;
6. test-time gradients/fast weights with all optimizer work and bytes charged;
7. self-consolidation and experiential-distillation controls with their update
   selection, replay/distillation, and offline work charged; and
8. only then, a structured executable hypothesis state if generic state leaves
   a specific compression, extrapolation, or revision failure.

This ladder tests whether the gain comes from the training unit, persistence,
online optimization, or a structured representation. It does not assume the
answer in advance.

### Stage C — substantial gates

On every frozen held-out composition split, require predeclared metric
denominators, absolute competence floors, train/test horizons, confidence
intervals, seed count, and multiple-comparison handling, then require:

- at least `30%` lower cumulative regret than the strongest matched learner;
- at least `2x` fewer interactions to reach a fixed competence;
- at least `10x` reasoning-horizon extrapolation after short-horizon training;
- no protected capability loss above one absolute point outside uncertainty;
- calibrated revision at hidden change points; and
- a causal drop under state/update/hypothesis corruption.

A pass earns a natural interactive pilot using noisy language, software, or
scientific feedback. It establishes reusable learning only within the charged
grammar, not general intelligence. The pilot must compare complete adaptive
Pareto frontiers: the smaller developmental model and the strongest larger
model each receive their best declared context, retrieval, memory, and update
path with equal evidence and complete serving/update costs. A preregistered
large end-to-end natural-domain gain plus retention and transfer is required
before claiming a substantially more intelligent resulting model.

## 6. Decision

Do not freeze or run the current four-family T70 manifest as the next decisive
experiment. Its online rotations remain useful stress tests, but it
overcommits to persistent latent state and underdefines the shared transferable
object and meta-distribution.

The next step is mathematical and executable but CPU-sized: freeze the smallest
world grammar with a nontrivial identifiability boundary and an order-one
advantage for acquiring and using a reusable rule over verbatim replay. An
optimal sufficient-state code remains an information ceiling/control; no
latent representation can claim a fundamental bit advantage over it. Then
compare training objectives on the same tiny backbone. A state-token
architecture is one entrant, not the research direction.

No GPU use or rental is justified by this reset.

### Subsequent theorem status

[T72](finite-mechanism-grammar-t72.md) closes full affine dynamics as an
active-learning test because `d+1` fixed basis probes are already minimax.
[T73](guarded-mechanism-worlds-t73.md) adds one ordered guard and proves a
complete depth-one separation even when both local affine maps are unknown:
adaptive acquisition uses at most
`2(d+1)+ceil(log2(m-1))` probes, while every exact fixed schedule needs
`m(d+1)`. At `m=33,d=1`, this is at most 9 versus exactly 66. The theorem
earns only a future CPU calibration and successor-grammar work; no neural or
natural-pilot admission follows.

Independent review:
[`developmental-intelligence-gap-t71-independent-audit.md`](developmental-intelligence-gap-t71-independent-audit.md).
