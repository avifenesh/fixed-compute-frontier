# Independent audit: Lifetime-Trained Mechanism Memory (T55)

Date: 2026-08-01  
Scope: paper and theorem audit only; no experiments run

## Verdict

**LTMM is coherent as a typed runtime contract, but it is not yet a coherent, distinct research candidate.** Its state tuple and edit vocabulary describe a sensible way to implement persistent, inspectable adaptation. They do not yet define a mathematical algebra, a trainable mechanism, or a capability unavailable to a generic recurrent or external-memory learner. The surrounding proposal is primarily an integration of already-active lines: executable knowledge-base repair, explicit behavioral-model revision, online causal world-model repair, provenance-preserving schema revision, and dynamic expert or memory growth.

The one potentially distinct claim is narrower and stronger than the document currently establishes:

> A frozen update policy, trained across complete lifetimes, can learn to localize sparse mechanism changes into typed persistent edits and obtain a cost-adjusted advantage over exact Bayesian/program-search, recurrent-state, online-world-model, and dynamic-expert controls under identical observations, action rights, compiler support, and persistent-state budgets.

No present theorem proves that claim. T55.1 is an immutability lemma, T55.2 is an information-theoretic impossibility result, and T55.3 imports oracle ceilings. The proposed regret expression is not justified as written. **No microbenchmark is earned yet.** Keep the paper-stage no-run gate until the hypothesis family, observation model, edit semantics, separation definition, and matched controls are formalized well enough that a run cannot merely reward a hidden compiler or oracle.

## 1. Typed state and edit algebra

The proposed persistent state

\[
S_t=(\theta,W_t,E_t,L_t,B_t,Q_t,P_t)
\]

is operationally legible: a frozen core, transient workspace, append-only evidence, versioned mechanisms, bindings, uncertainty, and provenance/protected-evaluation metadata. Likewise, `appendEvidence`, `updatePosterior`, `instantiateBinding`, `forkMechanism`, `splitVariable`, and `proposeMerge` are a useful interface for an auditable system.

It is not yet an algebra in the mathematical sense. Missing pieces include:

- state invariants and a typed grammar for every edit payload;
- edit preconditions, postconditions, failure semantics, and commit/rollback behavior;
- rules for composition, conflicts, dependency invalidation, and migrations;
- a finite complexity measure for programs, variables, bindings, and routes;
- semantics connecting an edit to the distribution of future observations and losses.

Several concrete problems follow.

1. A finite set of edit names does not imply a finite action space. The payloads can contain unbounded predicates, variables, programs, or graphs. Most of the intelligence may therefore sit in payload synthesis or candidate generation, not in choosing an edit type.
2. Append-only evidence conflicts with a bounded-lifetime-state claim. A penalty on \(|S_t|\) does not impose a hard bound, and evidence plus provenance can grow as \(O(T)\) even if the number of mechanism versions is small.
3. `proposeMerge` is not a state transition until accept/reject and rollback semantics are specified. Semantic equivalence and non-inferiority can themselves require an oracle, exhaustive evaluator, or undecidable program-equivalence test.
4. `splitVariable` changes interfaces. Preserving old behavior requires versioning the encoder, schema, downstream bindings, mechanisms, routes, and migrations—not merely preserving the parent variable object.
5. `updatePosterior` is underspecified when hypotheses can be born, split, merged, or deleted. Probability transport and calibration across changing hypothesis spaces need explicit semantics.
6. Versioned routing needs a complete address model. A protected path must pin all dependencies, including encoder, schema, binding, mechanism, selector, readout, external runtime, and randomness.

The claim that a generic vector write cannot expose the same operation is an interpretability or interface claim, not a capability separation. An unrestricted recurrent state can encode the same typed transition. Distinctness requires either restrictions that rule this out or a theorem about statistical, computational, or auditability advantages.

## 2. Lifetime objective

The lifetime objective is directionally sensible:

\[
\mathbb E_\tau\sum_t [\ell_t+\lambda_a c(a_t)+\lambda_c c(S_t,U_t)+\lambda_m|S_t|+\lambda_q\operatorname{CalErr}_t]
+\lambda_R\operatorname{RetentionRisk}(\tau).
\]

But it does not yet specify a realizable training problem.

- \(c(S_t,U_t)\), \(|S_t|\), and the units of the scalarized terms are undefined. Bytes, nodes, wall time, and verifier calls are not interchangeable without an explicit accounting rule.
- Calibration error requires a proper score and eventual labels. In partially observed deployment, labels may be delayed or never available.
- Retention risk requires a protected-query distribution and outcomes. If those are supplied during learning, they can leak task identity or create a hidden replay oracle; if they are not supplied, the learner cannot optimize that term directly.
- Backpropagation through discrete edits, variable-size programs, external execution, and long horizons is not ordinary BPTT. The proposal must choose policy gradients, relaxations, supervised edit targets, search, or another estimator, and account for its variance and truncation bias.
- The environment/action policy, intervention rights, horizon generalization, meta-training task distribution, and treatment of delayed outcomes are not specified.
- Soft storage penalties invite degenerate strategies such as retaining all evidence and never merging. Hard resource constraints or an explicit rate-distortion objective are needed.

A fair complete-cost objective must charge at least peak and cumulative stored bytes, read/write traffic, candidate generation, routing/search, verifier evaluations, actions and resets, external model or tool calls, interpreter overhead, and outer-loop meta-training compute. The frozen \(\theta\) is not itself a fairness advantage: the external state is the learned system, so controls need the same persistence and update budget.

## 3. Theorem audit

### T55.1: version-addressed retention

This is correct only as a narrow functional-immutability lemma: if a query executes a deterministic, fully version-pinned path and no reachable object or route is changed, the output is unchanged.

That result is useful for software integrity, but nearly tautological. It does not prove retained competence under ordinary routing, retained performance on a distribution, or immunity to interference in newly selected paths. “Bit-identical” additionally requires a pure deterministic executor, pinned runtime/library versions and numerics, fixed random state, and no shared mutable cache. Appending a new object must not change retrieval normalization, candidate ranking, top-k selection, or any global index reachable from the query.

The theorem should explicitly pin the encoder, variable schema, bindings, mechanisms, selector/router, readout, dependencies, executor version, and randomness. Otherwise an unchanged mechanism version can still produce a changed answer. If the query itself supplies the historical version address, that address is also side information and must be available to matched controls.

### T55.2: indistinguishable edits

The information-theoretic conclusion is sound: edits that induce identical observable distributions under all permitted histories and interventions cannot be reliably distinguished. It supports maintaining a posterior, gathering information, or abstaining.

It is not an LTMM advantage. The statement needs a formal hypothesis class, prior or minimax quantifier, adaptive action policy, observation sigma-field, equivalence relation, and decision loss. The result is inherited by every learner with the same information.

### T55.3: exact conditional reuse

This is not a theorem about LTMM. It imports T51/T52 conditions and the T54 affine-coordinate oracle, including a known library, correct bindings, and correct action coordinates. Those assumptions provide much of the compilation and routing work the candidate is supposed to learn.

Relabel this as a **conditional benchmark ceiling**. It can test whether an implementation preserves already-provided structure, but cannot establish discovery, localization, or learned reuse.

## 4. Proposed regret theorem

The target

\[
\operatorname{Reg}_T\leq \widetilde O\!\left(
\sum_{c=1}^{C}\frac{\log|\mathcal H_c|+\log(1/\delta_c)}{\kappa_c}
\right)+\operatorname{ApproxErr}
\]

is a research aspiration, not a supported theorem. In particular:

- The dependence on \(1/\kappa_c\) is unjustified until \(\kappa_c\) is defined. It is plausible if \(\kappa_c\) is per-sample KL information. If it is a reward, mean-loss, total-variation, or observation gap, concentration often gives \(1/\kappa_c^2\). Regret during detection also normally includes a per-step loss gap multiplying detection delay.
- There is no stationary estimation or control term, often of order \(\sqrt T\) or problem-dependent equivalent, and no cost for exploration, interventions, delayed detection, false alarms between changes, mixing, planning, routing, or partial observability.
- \(|\mathcal H_c|\) may be infinite or data-dependent when variables and programs can be invented. It needs a finite frozen class, covering number, or prefix-code/description-length complexity.
- A simple \(\sum_c\delta_c\) false-edit bound requires a valid predetermined union bound or time-uniform testing argument. Adaptive repeated tests and optional stopping cannot be ignored.
- Change points need minimum dwell time and identifiability under the permitted action policy. A separation that is reachable only through an uncharged oracle intervention is not sufficient.
- `ApproxErr` currently absorbs representation misspecification, compiler failure, planning error, and ontology mismatch. Those terms must be separated or the bound can be vacuous.
- The version count should begin at \(O(K+sC)\), not \(O(C+sC)\), and it must additionally include false edits, abandoned branches, failed merges, and schema migrations. Evidence and provenance can still grow as \(O(T)\).
- “Zero direct modification” is syntactic. It does not imply zero protected-task loss if new routing, retrieval, normalization, or shared dependencies change behavior.
- The objective is multi-resource, while the regret theorem contains no computational or memory cost. Either prove a resource-constrained result or fix the Lagrange multipliers and units.

The first comparison should be an exact Bayesian change detector or finite program enumerator over the same hypothesis family. If that control obtains the same statistical bound, LTMM’s possible contribution is amortized computation or scaling—not a new regret principle.

## 5. Hidden oracle, compiler, and routing bills

The current proposal does not yet charge or remove the following likely sources of advantage:

- perceptual parsing, variable proposal, predicate invention, schema typing, and program synthesis;
- candidate-edit enumeration and the recall guarantee that places the correct edit in a small bounded set;
- graph matching, binding, namespace resolution, and migration after a split;
- change-point detection and selection of the affected mechanism subset;
- protected-task evaluators, semantic-equivalence tests, merge certificates, and multiple-comparison correction;
- intervention selection, resets, environment actions, and delayed-outcome acquisition;
- route selection, retrieval indices, top-k normalization, version-address metadata, and task identifiers;
- sandboxing, interpreter/runtime execution, provenance storage, garbage collection, and consolidation;
- outer-loop lifetime data, curriculum construction, BPTT/search compute, and foundation-model calls.

“Bounded candidate proposal” is especially dangerous. Unless its compute, information, and recall are matched, it can simply be the hidden oracle that has already localized and synthesized the edit. The exact Bayesian/program control must receive the same observations, compiler products, action rights, candidate set, and storage limit.

## 6. 2026 collision set and required controls

The nearest current work makes the broad novelty claim untenable:

- [Kintsugi: Learning Policies by Repairing Executable Knowledge Bases](https://arxiv.org/abs/2605.09487) already combines typed executable knowledge bases, rollout-localized typed edits, deterministic verification, improvement gates, and protected-regression checks.
- [Learning Explicit Behavioral Models with Adaptive Questions and World-Model Probes](https://arxiv.org/abs/2606.07127) uses typed predicates, executable mechanism memory, adaptive probes, local edits, and mechanism-change handling.
- [Continual learning and refinement of causal models through dynamic predicate invention](https://arxiv.org/abs/2602.17217) continually repairs a symbolic causal world model and invents reusable abstractions inside the decision loop.
- [Self-Revising Discovery Systems](https://arxiv.org/abs/2606.01444) covers provenance-preserving schema updates, regime transitions, and MDL-gated revision.
- [Continual Reinforcement Learning by Planning with Online World Models](https://arxiv.org/abs/2507.09177) provides an online-world-model control with no-forgetting-by-construction and a regret result.
- Dynamic continual experts are represented by [D-MoLE](https://proceedings.mlr.press/v267/ge25d.html), [DIMoE-Adapters](https://arxiv.org/abs/2605.07494), [SAME](https://arxiv.org/abs/2602.01990), and [theory for mixture-of-experts in continual learning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17a234c91f746d9625a75cf8a8731ee2-Abstract-Conference.html).
- [Continual Knowledge Updating in LLM Systems: Multi-Timescale Memory Dynamics](https://arxiv.org/abs/2605.05097) and [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](https://arxiv.org/abs/2603.00903) cover adaptive memory consolidation and fast/meta lifetime learning.

Therefore a credible evaluation must include, at minimum:

1. exact Bayesian/change-point inference over the same finite edit family;
2. exact or budget-matched program/edit enumeration;
3. online world-model planning;
4. a recurrent or state-space learner with the same persistent-byte and compute budget;
5. dynamic expert/adaptor growth with matched routing and parameter budget;
6. executable-KB repair and explicit-behavioral-model revision controls;
7. ablations for typed edits, version pinning, active probes, verifier access, and lifetime meta-training.

Every control must share raw versus compiled inputs, interventions, candidate-generator outputs, protected-query metadata, verifier calls, and total training/inference/state costs.

## 7. Is a microbenchmark earned?

**No—not yet.** A benchmark now would mostly measure choices in the hand-built DSL, compiler, candidate generator, and verifier. It could produce a positive result without testing the claimed learned edit localization.

The minimum paper gate before any run is:

1. define a finite, fully generative piecewise-stationary family with raw observations and explicit permitted actions;
2. give denotational semantics and invariants for every state component and edit;
3. define \(\kappa\) as an information quantity or derive the correct gap dependence;
4. state a theorem or lower/upper-bound pair that includes detection delay, false alarms, stationary learning, memory, and action/compute costs;
5. expose and charge the candidate generator, compiler, router, verifier, protected set, and version address;
6. specify an exact matched Bayesian/program baseline before introducing a neural amortizer;
7. identify a falsifier: for example, LTMM loses if its complete-cost frontier is not strictly better than that exact control or if its advantage vanishes when the edit proposal is generated from raw observations under the same budget.

If those conditions are met, the first earned test should be an exhaustive finite-state theorem check or exact enumerated simulation—not a broad neural benchmark. The current no-experiment decision is correct.

## Final classification

- **Coherent typed implementation contract:** yes, after stronger invariants and complete version pinning.
- **Coherent mathematical edit algebra:** no.
- **Distinct architecture relative to 2026 prior art:** no; currently an integration/relabeling.
- **LTMM-specific theorem:** no. T55.1 is immutability, T55.2 is impossibility, T55.3 is an oracle ceiling.
- **Proposed regret bound justified:** no.
- **Hidden oracle/compiler/routing costs closed:** no.
- **Microbenchmark earned:** no.
- **Disposition:** retain only as a specification for a much narrower learned edit-localization claim; do not promote it as a candidate architecture or reopen experiments without the formal gate above.
