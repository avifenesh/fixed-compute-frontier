# T55 lifetime-trained mechanism memory — one candidate integration

Date: 2026-08-01  
Status: **INDEPENDENTLY REJECTED AS DISTINCT ARCHITECTURE; TYPED RUNTIME CONTRACT ONLY; NO RUN**

## 0. Why this is the next object

T45–T54 remove the easy claims. Persistent state, active tests, experiment
ranking, finite mechanism libraries, hidden wiring, and action-grounded
coordinates each reduce to strong classical or current learned controls when
their interfaces are supplied.

The remaining target is the conjunction: a deployed model should convert raw
experience into persistent executable knowledge, actively test uncertainty,
reuse that knowledge in new compositions, and revise it without globally
damaging prior competence.

The proposed integration is **Lifetime-Trained Mechanism Memory (LTMM)**. An
independent audit finds that this is not currently a distinct architecture: it
combines active lines in executable knowledge-base repair, explicit behavioral
models, online causal world models, lifetime meta-learning, modular memory, and
dynamic experts. Retain it only as a typed runtime contract. Its one narrower
research claim is whether a frozen learned update policy can localize sparse
mechanism changes at a better complete-cost frontier than exact and learned
controls.

## 1. Typed state

At deployment step `t`, the complete system state is

\[
S_t=(\theta,W_t,E_t,L_t,B_t,Q_t,P_t),
\]

where:

- `theta` is the frozen slow neural core;
- `W_t` is bounded transient working state;
- `E_t` is episodic evidence under a hard byte quota, with versioned
  compression/eviction decisions;
- `L_t` is a library of addressable executable mechanism versions;
- `B_t` binds variables and mechanism ports in the current world hypothesis;
- `Q_t` is calibrated uncertainty over competing encoders, bindings,
  mechanisms, and regimes; and
- `P_t` records provenance, protected evaluations, and version migrations.

Every byte is part of the model. A perception encoder, symbolic compiler,
retriever, verifier, planner, or external program used to construct or execute
this state is included in `theta` or charged as an external call.

## 2. Bounded edit algebra

The learned update policy cannot silently overwrite arbitrary state. It emits a
typed edit name from the following vocabulary:

\[
u_t\in\{
\operatorname{appendEvidence},
\operatorname{updatePosterior},
\operatorname{instantiateBinding},
\operatorname{forkMechanism},
\operatorname{splitVariable},
\operatorname{proposeMerge}
\}.
\]

Forks and splits preserve their parent versions. A merge is committed only
after a declared equivalence/noninferiority test; otherwise both branches
remain addressable. Routing-policy changes are themselves versioned edits.

This vocabulary is not yet a mathematical algebra. Each payload needs a finite
grammar and complexity, pre/postconditions, failure and rollback semantics,
composition/conflict rules, dependency invalidation, and denotational effects
on future observations and losses. Without those restrictions, candidate and
payload synthesis can contain the entire intelligence problem.

The vocabulary is useful to distinguish evidence accumulation,
belief change, new composition, mechanism change, ontology refinement, and
compression for audit. A generic recurrent/vector state can encode the same
transition, so interpretability alone is not a capability separation.

## 3. Sparse executable read path

Given observation `o_t`, the core proposes a bounded candidate set of variables,
mechanisms, and bindings. The posterior selects or mixes hypotheses. Only the
active bound subgraph is executed:

\[
\hat y_t=
\operatorname{Readout}_\theta
\left(
\operatorname{Execute}(L_t,B_t,E_\theta(o_t),W_t)
\right).
\]

The same executable graph supports prediction, counterfactual rollout, and
planning. Retrieval of prose is allowed as evidence but is not itself a
mechanism. Active execution cost, candidate proposal/search, graph depth,
memory traffic, and readout calls are charged.

## 4. Lifetime objective

Training examples are not independent prompts. One sample is a lifetime
`tau=(z_1,...,z_T)` containing stationary periods, mechanism changes, renamed
or recombined worlds, delayed outcomes, and protected old queries. The update
is a forward operation

\[
S_{t+1}=U_\theta(S_t,o_t,a_t,f_t),
\]

and the outer objective is

\[
\min_\theta\ \mathbb E_\tau
\sum_{t=1}^T
\left[
\ell_t
+\lambda_a c(a_t)
+\lambda_c c(S_t,U_t)
+\lambda_m |S_t|
+\lambda_q \operatorname{CalErr}_t
\right]
+\lambda_R\operatorname{RetentionRisk}(\tau).
\]

Future losses must credit earlier edit decisions during meta-training, but the
estimator is unspecified: discrete variable-size edits require policy gradients,
relaxation, supervised targets, or search, each with variance/bias and compute.
At deployment, `theta` is frozen: learning is the bounded state transition
`U_theta`, not an unreported SGD job. A realizable objective must impose hard
resource constraints or a declared rate-distortion ledger; soft penalties alone
permit retaining everything. Proper calibration labels and the protected-query
distribution must also be declared without leaking task identity.

This is the key training change. Next-token or next-state prediction can still
be an auxiliary loss, but the optimized object is improvement across a
lifetime.

## 5. Exact properties available before learning

### T55.1 — version-addressed retention

Let protected query `q` address a deterministic fully pinned path `v(q)`,
including encoder, schema, bindings, mechanisms, selector/router, readout,
executor/runtime versions, external dependencies, and random state. If an edit
appends new objects without changing any object, global index, normalization,
cache, or route reachable from `v(q)`, then the output on `q` is bit-identical
before and after the edit.

**Proof.** The executed deterministic program and every addressed value are
unchanged. Appended but unreachable state cannot affect its output. QED.

This is intentionally narrow. It does not guarantee that a learned router will
continue selecting `v(q)`, that a changed encoder emits the same variables, or
that new-world queries should use the old version. Those are measured learning
problems, not hidden inside the retention theorem.

### T55.2 — information necessity

T53 applies unchanged. If competing environment/edit hypotheses induce the
same distribution under the system's available history and actions, no update
policy can reliably select the right edit. LTMM must retain the posterior or
abstain; confident local repair is impossible.

### T55.3 — conditional benchmark ceiling

Given a correct shared library and identifiable bindings, T51/T52 show that a
new bounded-arity composition can have description and interaction complexity
logarithmic in the finite mechanism/wiring hypothesis count rather than the
truth-table size of an unrestricted global transition. T54 shows one exact way
actions can recover a coordinate gauge under affine sensing.

These are oracle ceilings, not an LTMM theorem. LTMM earns the gain only if its learned encoder,
library, and update policy approach them after all acquisition and amortization
costs are charged.

## 6. Required new theorem — learned edit localization

The candidate is not admitted until it proves something not supplied by the
previous controls. Freeze a finite piecewise-stationary family in which at most
`s` of `K` mechanisms, bindings, or variables change at each change point. The
learner observes only partial outcomes and can choose bounded legal actions.

Required theorem:

> Under an explicit edit-separation margin `kappa`, an LTMM update policy can
> identify the responsible edit equivalence class, route to a forked version,
> and bound cumulative regret, false edits, state growth, and protected-path
> interference in terms of true changes—not total lifetime length.

No regret form is retained yet. It must first define whether `kappa` is KL,
total variation, or a prediction gap; include stationary estimation, detection
delay times per-step loss, false alarms under optional stopping, exploration,
planning, partial observation, minimum dwell time, and separate representation,
compiler, routing, and approximation errors. A version ledger starts at
`O(K+sC)` only before false branches, schema migrations, evidence, and
provenance are charged. Syntactic immutability does not imply protected-task
retention under changed routing.

Mandatory comparison is an equally informed Bayesian change detector,
factored POMDP/PSR learner, dynamic-program library, recurrent meta-RL learner,
and expandable/frozen-expert continual learner. If one already attains the
same ledger, LTMM has no theoretical edge.

## 7. Conditional intended gain

If the hypothesis holds, the result model gains capabilities absent from a
frozen answer predictor:

1. it learns new facts, rules, and action consequences after deployment without
   gradient updates;
2. it turns repeated episodes into reusable executable mechanisms rather than
   ever-growing prompt text;
3. it adapts to new combinations and sparse world changes using bounded new
   evidence;
4. it preserves old versions and can explain which evidence caused a revision;
5. it spends serving compute only on the active mechanism graph while durable
   capacity grows mainly in external structured state.

The intended production win is not lower latency alone. It is higher capability
after experience: lower learning regret, better unseen recombination, calibrated
self-correction, and negligible protected-task loss at a competitive complete
cost.

## 8. Controls that can kill the candidate

- **ORBIT/meta-RL/recurrent state:** may learn the same update algorithm without
  explicit mechanisms.
- **Modular/experience memory:** may achieve the same adaptation through
  retrieved episodes or programs.
- **TTT/fast weights:** may adapt more compactly once update compute and
  forgetting are matched.
- **Mechanistic/graph world models and dynamic predicate invention:** may
  already supply the executable factorization.
- **Exact Bayesian/program learner:** may dominate every learned edit under the
  finite interface.

LTMM survives only through a large matched result across several lifetime
families, plus a causal ablation showing that executable reuse and localized
versioned edits create the gain.

## 9. Block proof and microbenchmark order

1. Prove the finite learned-edit localization theorem or its impossibility.
2. Prove routing/retention and bounded-growth behavior under hidden regime
   identity; close the lane if metadata is required.
3. CPU-test the edit policy against exact Bayesian and recurrent controls on
   exhaustive finite worlds.
4. Train tiny models locally only if the learned policy closes a substantial
   fraction of the exact ceiling across unseen families.
5. Check the local GPU before any rental; rent only after a replicated large
   local effect and a frozen complete-cost gate.

The independent audit concludes that no microbenchmark is currently earned: a
run would mainly measure a hand-built DSL, compiler, candidate generator, and
verifier. No implementation, CPU benchmark, local GPU, or rented GPU run is
admitted by this specification.
