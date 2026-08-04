# T76 epistemic operator machine — independent audit

Date: 2026-08-02  
Audited artifact: `results/epistemic-operator-machine-t76.md`  
Decision: **retain as a research charter; do not admit an architecture or run**

## 0. Bottom line

T76 correctly names the user's real target: a complete resulting model whose competence changes substantially and correctly from post-training experience. It does not yet specify a mechanism capable of establishing that result.

The present “epistemic operator machine” is an umbrella of desirable functional roles. A generic recurrent meta-RL agent, a probabilistic modular meta-learner, a frozen LLM with evolving textual memory, or an explicit program system could all instantiate its transition equation. The document explicitly allows this, which is scientifically honest, but it means there is no distinct candidate mechanism to test yet.

The mathematical arguments establish conditional representational separations, not a learning theorem:

- passive observational equivalence implies non-identifiability only under a closed information boundary;
- finite distinguishable state gives a pigeonhole capacity bound, but the current wording hides precision and old-behavior equivalence classes;
- `Omega(k^L)` versus `O(k)` is valid only for a deliberately opaque no-sharing family versus a known/identified homomorphic family with unit-cost primitive identification;
- low-treewidth factor graphs admit compact exact inference under bounded-domain assumptions, but not necessarily local learning or local message propagation; and
- one shared operator can be revised once only if factor identity, routing, interfaces, and all shared parameters remain stable.

The proposal is also very close to current work. In particular, [Compositional meta-learning through probabilistic task inference](https://arxiv.org/abs/2510.01858) already learns modules and a gating grammar, performs posterior hypothesis tracking under sparse feedback, recovers rule and motor primitives, and reports four-times length extrapolation. [From Reasoning Traces to Reusable Modules](https://arxiv.org/html/2606.18089v2) already provides latent-module identification and local-witness composition results plus compound-trace training evidence. [ORBIT](https://arxiv.org/pdf/2602.04089) already trains a cross-episode in-context online learner with a lifetime reward. [WorldEvolver](https://arxiv.org/pdf/2606.30639) already performs frozen-model deployment-time revision using episodic transitions and mismatch-derived semantic rules. The causal-epistemic loop in [Yu and Inamura](https://arxiv.org/pdf/2606.22449) overlaps almost verbatim at the conceptual level.

The residual research claim can still be meaningful, but it must be narrowed to this conjunction:

> one frozen learned updater discovers reusable epistemic operations from interaction, carries them in a strictly bounded non-oracle state after the transcript is removed, actively reuses them across independently generated domains, and locally revises them after hidden mechanism changes while beating the strongest recurrent, probabilistic-modular, full-context, retrieval/procedural-memory, and explicit-program controls under one complete ledger.

Nothing in Levels 1–3 currently guarantees that conjunction. Level 3 could support a substantial **system-level adaptive-competence** result after repair; it cannot establish a universal increase in “intelligence,” and with a frozen backbone it cannot by itself show that the foundation model's parameters became more intelligent.

## 1. Claim-by-claim mathematical audit

### 1.1 Passive information and intervention

Claim: if two environments induce the same law for every passive history but differ under intervention, no passive predictor can identify the interventional law.

Status: **correct with a missing information-boundary condition**.

The exact statement must quantify over every legal passive policy. If environments `e0,e1` induce identical transcript laws

\[
P_{e_0}^{\pi}(H_T)=P_{e_1}^{\pi}(H_T)
\]

for every passive policy `pi`, and the learner has no side information correlated with the environment identity, no estimator based on `H_T` can distinguish them. Under an equal two-world prior its Bayes environment-identification error is at least `1/2`.

The claim fails if the environment identity leaks through task selection, names, action availability, timing, rewards, reset behavior, metadata, pretrained memorization of a finite corpus, or persistent state correlated with the world. “Passive history” must include every channel visible to the model, not only the nominal observation field.

The conclusion is non-identifiability, not that intervention is always the only remedy. A valid structural assumption, an independent proxy, or another informative observation channel can also identify the interventional law. The source text acknowledges structural assumptions but should explicitly close side channels.

### 1.2 Fixed-state information capacity

Claim: a system with `P` distinguishable parameter states cannot encode more than `log2(P)` independent bits while exactly preserving all old behavior.

Status: **a finite-state pigeonhole statement, currently overgeneralized**.

Let the reliably distinguishable persistent state be `S` with `|S| <= P`. Then

\[
H(S)\le \log_2 P,
\]

and no exactly decodable independent message stored solely in `S` can have more entropy. If old behavior `b` must be preserved, the relevant capacity is smaller:

\[
H(S\mid B=b)\le \log_2 |\mathcal E_b|,
\]

where `E_b` is the equivalence class of states producing exactly that old behavior. If the old behavior uniquely fixes the state, the residual exact capacity is zero.

Required repairs:

- replace “parameter states” with **reliably distinguishable complete persistent states**, including optimizer, external memory, caches, routing state, and numeric precision;
- define “saturated” or remove it;
- charge the bits/precision used by real-valued state—mathematical real parameters otherwise have infinitely many distinguishable values;
- distinguish storing arbitrary independent facts from learning a reusable algorithm, which can improve many outputs without storing one independent bit per output; and
- state that approximate retention permits rate-distortion tradeoffs rather than this exact bound.

The qualitative no-free-lunch point is sound: fixed finite state cannot absorb an unbounded stream of unrelated, exactly retained information.

### 1.3 `Omega(k^L)` opaque-task coverage

Claim: with `k` primitives and length-`L` ordered programs, unstructured task learning can require `Omega(k^L)` coverage.

Status: **correct only as a worst-case construction, not a lower bound on generic neural learners**.

There are exactly `k^L` length-`L` sequences when repetition and order are allowed. If each complete sequence is assigned an independent random answer/permutation and the task interface reveals no shared primitive structure, an unseen sequence is unidentifiable. Exact worst-case coverage of all sequences therefore requires observing all `k^L` tasks, hence `Omega(k^L)` task instances.

This lower bound follows by assuming away reusable structure. It does not show that Transformers, recurrent models, or dense states generally pay `k^L`; any of them can discover the compositional rule when the data identify it. Calling the opposing learner “unstructured” risks defining the baseline to lose.

The benchmark must instantiate two separate families:

1. an opaque-permutation no-free-lunch family, used only to validate the theorem; and
2. a structured family in which every capacity-matched learner is allowed to discover sharing.

Only the second is relevant to learned-model intelligence.

### 1.4 `O(k)` compositional identification

Claim: identifying the `k` primitive semantics plus one associative homomorphic composition law can determine all `k^L` programs with `O(k)` primitive identification.

Status: **conditionally correct as representation/sample accounting in primitive units; incomplete as a learning-complexity claim**.

If all of the following are given or identified:

- each primitive can be uniquely identified with `O(1)` actively chosen evidence independent of `k` and `L`;
- the program's primitive sequence and routing are observable or separately identifiable;
- the type system is known, closed, and stable;
- one shared composition law has `O(1)` description/identification cost;
- `E` is homomorphic on every evaluated composition;
- there are no context-dependent primitive interactions, aliasing, hidden state, or accumulated approximation error; and
- execution time is charged separately,

then storing/grounding `k` primitives costs `O(k)` and their descriptions determine `k^L` sequences. This is a valid conditional exponential generalization separation.

What the notation suppresses:

- arbitrary primitive functions may require many samples/bits each, so the actual cost is `Omega(k C_primitive)`;
- passive random exposure to all primitives generally incurs a coupon-collector factor, while exactly `k` requires direct active access;
- learning the composition grammar may scale with types, interfaces, arity, or primitive pairs rather than `O(1)`;
- every length-`L` execution still costs at least `Omega(L)` operator applications and the requested program carries `L log2(k)` bits of identity;
- noisy identification introduces confidence factors and error can compound with depth;
- a non-injective `E` gives only equivalence-class identification; and
- pair-specific compatibility witnesses can require a graph of exposures, as the local-witness conditions in [2606.18089v2](https://arxiv.org/html/2606.18089v2) make explicit.

The document should call this a **conditional description/sample separation**, not a theorem that the proposed learner will achieve `O(k)`.

### 1.5 Localized revision

Claim: when only operator `j` changes, a factorized executable representation can update it once while a monolithic table may relearn every composition containing it.

Status: **correct for a shared-factor representation under strong stability assumptions; not yet a learned-system result**.

The number of length-`L` sequences containing `j` is

\[
k^L-(k-1)^L.
\]

An opaque complete-program table can require changing that many entries in the worst case. A representation that truly shares one instance of `j` can change it once.

The single update is sufficient only if:

- operator identity and causal responsibility are identifiable from the available interventions;
- every affected program references the same operator object rather than copies;
- the input/output type and semantics of all other operators remain stable;
- the router, encoder, decoder, calibration, and downstream state interpretation do not also need revision;
- changing `j` does not change the data distribution in a way that invalidates other learned factors; and
- shared neural parameters do not entangle `j` with protected operators.

“Local state edit” and “localized behavioral effect” are different. Both must be measured. A sparse edit that causes broad downstream regressions fails; a distributed edit that cleanly repairs only relevant behavior may still be functionally localized.

The monolithic table is a theorem foil, not a fair strongest control. A generic recurrent or dense learner may discover the same sharing and must remain in the comparison.

### 1.6 Factor-graph complexity

Claim: for `n` latent variables with treewidth `w`, exact inference uses `O(n 2^w)` work/space instead of `O(2^n)` for a full joint table.

Status: **asymptotically right for bounded-degree binary graphical models, but underspecified**.

For variable domain size `d`, bounded factor count `O(n)`, and a junction tree whose largest bag has `w+1` variables, standard exact inference is more safely stated as

\[
O(n d^{w+1})
\]

time and comparable worst-case message/potential storage, up to graph- and implementation-dependent constants. For binary variables this is `O(n 2^{w+1})`, asymptotically the stated `O(n 2^w)`. An arbitrary binary joint table has `2^n` entries.

Hidden assumptions and costs:

- a valid low-treewidth graph and factors are already known;
- finding optimal treewidth/elimination order is itself generally hard;
- learning the graph and potentials is not included;
- a local evidence change may require messages to propagate across the whole junction tree, so compact inference does not imply `O(1)` localized update work;
- dynamic/nonstationary histories can increase effective state dimension or treewidth; and
- comparing against an explicit full joint table is not enough—dense learned states and alternative structured codes are fair controls.

The source text already recognizes the last point. It should repair the formula and avoid equating graphical factorization with learned causal modularity.

### 1.7 Lifetime objective

Claim: the displayed objective credits prediction, action cost, calibration, revision, retention, and compression.

Status: **conceptually appropriate, mathematically incomplete**.

As written,

\[
J=\mathbb E\left[\sum_t \cdots\right]
\]

does not specify a finite lifetime, discount, or average-cost limit and may diverge. It also does not state that `J` is minimized. Freeze:

- lifetime horizon/distribution or discount;
- nonnegative weights and units/normalization of every term;
- whether losses are per step, per episode, or per world;
- the protected-task sampling distribution;
- the compression code and state precision;
- the counterfactual defining “responsible mechanism”; and
- which randomness the expectation covers.

A scalar weighted sum can hide Pareto regressions. Primary admission should remain a strict Pareto gate; use the scalar objective for training only, with every term ablated as proposed.

## 2. Missing no-free-lunch boundary

The document states several local no-free-lunch facts but omits the global one most relevant to its ambition:

> No learner can acquire useful operations across arbitrary unrelated future worlds without a shared meta-distribution, prior, interface, or structural regularity connecting training experience to those worlds.

“Across unlike worlds” does not eliminate this assumption. The worlds must still share something learnable: causal motifs, an action-observation grammar, object persistence, compositional interfaces, language semantics, or a common generator. The learned updater can exploit only that shared structure.

This creates the central synthetic-trick risk. A composition lattice generated by one codebase can expose one hidden grammar across superficially different wrappers. The model may learn generator recognition rather than a general epistemic operation. Countermeasures:

1. independent generators and authors for held-out domains;
2. no shared latent labels, serialization templates, naming conventions, action ordering, or RNG signatures;
3. post-freeze randomized mechanisms not present in pretraining;
4. counterfactual rebinding of every semantic primitive;
5. naturalistic domains whose transition code is not derived from the training generator;
6. wrapper-only and generator-ID probes;
7. a control trained on the same worlds with shuffled cross-domain correspondences; and
8. explicit reporting of which invariance is expected to transfer.

There is no contradiction in seeking broad meta-learning. The result must be stated relative to a predeclared family of environments, not as structure-free general intelligence.

## 3. Current-art collision audit

The proposal's related-work section is directionally honest but understates how much of the candidate object is already instantiated.

### 3.1 Reusable-module reasoning, arXiv:2606.18089v2

[Kong et al.](https://arxiv.org/html/2606.18089v2) model traces as hierarchical selections of reusable skills and routing mechanisms. They provide sufficient identification conditions, local compatibility witnesses for novel compositions, SFT hidden-support and RL enrichment analyses, and experiments in which compound traces plus RL recover/recombine atoms. The latest checked version is v2 dated 2026-07-05.

Direct collision:

- T76's operator library maps to skills plus routing modules;
- its composer maps to the learned/recovered routing hierarchy;
- its composition lattice and claim that compounds outperform isolated atoms are already the paper's central data-design result; and
- its identifiability/local-witness requirement is prior art, not a new T76 theorem.

Residual T76 scope: interactive active experiment choice, bounded state after transcript purge, deployment-time revision, consolidation, and transfer across non-trace domains. Those additions are not yet a specified mechanism.

### 3.2 Probabilistic compositional meta-learning, arXiv:2510.01858

[Bakermans et al.](https://arxiv.org/html/2510.01858v1) are the closest architectural collision. Their system learns reusable module RNNs and a gating RNN that represents composition statistics, casts new-task learning as probabilistic inference, uses particle filtering to maintain module-sequence hypotheses, works from sparse feedback without test-time parameter updates, recovers ground-truth components in rule and motor tasks, and reports four-times training-length extrapolation.

Direct collision:

- factorized belief over executable hypotheses;
- reusable computations/operators;
- learned composition grammar/router;
- explicit uncertainty and parallel hypotheses;
- sparse-feedback evidence update;
- one-shot new-task acquisition; and
- cross-domain rule/motor instantiation.

Merely raising extrapolation from `4x` to `10x` is a stronger benchmark threshold, not a distinct mechanism. T76 must add active query selection, persistent cross-episode operator formation, localized hidden-change revision, and a single frozen updater that transfers to independently generated domains. It must include this probabilistic model, scaled fairly, as a primary control.

### 3.3 ORBIT, arXiv:2602.04089

[ORBIT](https://arxiv.org/pdf/2602.04089) trains an LLM by multi-task, multi-episode meta-RL to use early interactions for information gathering and later interactions for exploitation. It retains the full cross-episode transcript, evaluates in-context regret, trains on five environments, and tests on unseen Maze and Mastermind. Its own limitations are exactly the opening T76 gap: three episodes and short horizons constrained by a 32k context, no external memory, and limited environment diversity.

Direct collision:

- lifetime/cross-episode reward;
- active exploration-exploitation from interaction;
- no test-time parameter update;
- generic recurrent/in-context state; and
- held-out-environment transfer.

Residual T76 scope: bounded durable state that survives transcript removal, reusable operations rather than whole-history recurrence, longer lifetimes, targeted revision, and compositional transfer. An ORBIT-style recurrent/full-context learner is therefore the most important non-structured control, not just one item in a long baseline list.

### 3.4 WorldEvolver and evolving-context systems

[WorldEvolver](https://arxiv.org/pdf/2606.30639) keeps model and downstream agent parameters frozen while maintaining episodic transition memory, extracting persistent textual heuristics from prediction-observation mismatches, and filtering low-confidence foresight. It evaluates prediction and downstream action success on Word2World, ALFWorld, and ScienceWorld.

This overlaps T76's bounded belief/developmental state, mismatch-driven update, reusable rules, frozen slow parameters, and consolidation/filtering motivation. T76 is distinct only if learned operators do more than retrieved episodes and prose heuristics, and if that difference causally explains a complete-system advantage.

The proposal should also include direct current controls for:

- [ACE](https://arxiv.org/abs/2510.04618), which evolves playbooks by generation, reflection, and curation;
- [Memp](https://arxiv.org/abs/2508.06433), which builds, retrieves, corrects, and deprecates procedural memories from trajectories; and
- [Supersede](https://arxiv.org/abs/2606.27472), which directly studies the bounded self-maintained memory-update gap.

These are essential to rule out “better prose memory” as the entire result.

### 3.5 Causal self-evolving cognition, arXiv:2606.22449

[Yu and Inamura](https://arxiv.org/pdf/2606.22449) frame embodied learning as causal hypothesis construction, intervention-driven experimentation, counterfactual reasoning, and continual structural revision. Their paper is conceptual and explicitly leaves stable continual update, failure-driven learning, consolidation, and safe intervention as open implementation problems.

This means T76's conceptual loop is not novel, but the implementation-and-evidence gap remains open. The residual contribution would be a working, controlled mechanism with a demonstrated complete-model gain—not the phrase “epistemic intelligence” or the five-role diagram.

## 4. Is the proposed mechanism actually distinct?

No, not yet.

The transition

\[
(a_t,w_{t+1},b_{t+1},O_{t+1})=F_\theta(o_t,f_{t-1},w_t,b_t,O_t)
\]

is a type signature, not a mechanism. `F_theta` can implement every system listed above. `b_t` and `O_t` are not operationally distinguishable; an arbitrary string, dense vector, program, retrieved trajectory, or entire hidden transcript can be called an operator library.

A distinct candidate must freeze at least:

1. **Operator semantics:** typed input/output, preconditions, effects, uncertainty representation, and executable behavior.
2. **State boundary:** exact byte/precision caps for `w`, `b`, `O`, caches, retrieval index, and any optimizer state.
3. **Update rule:** how evidence assigns causal responsibility and changes one operator or belief factor.
4. **Composition rule:** how operators are selected, routed, and executed; what closure and interface compatibility mean.
5. **Discovery rule:** how new operators are proposed and distinguished from copies or prose summaries.
6. **Consolidation rule:** witness set, acceptance test, rollback, conflict handling, and retirement.
7. **Transcript purge:** what is destroyed and how fresh-process continuity is verified.
8. **Training rule:** what is learned in `theta` versus deployment state, with all gradients and data charged.

If a generic recurrent control learns the same behavior and matches performance, the result is the lifetime objective/curriculum. If an explicit probabilistic modular control matches it, the neural operator container is unnecessary. If ACE/Memp/WorldEvolver matches it, the result is evolving context or procedural memory. Those are legitimate findings, but they are not evidence for a distinct epistemic-operator mechanism.

## 5. Can Levels 1–3 establish a model-level result?

### 5.1 Level 1: necessary integrity evidence, never sufficient

Level 1 can establish that a synthetic family is identifiable, that a candidate implements the intended update, and that measured scaling matches a conditional theorem. It cannot establish a substantially more intelligent model.

Current problems:

- “gain grows with depth/mechanisms” lacks a frozen functional form, range, and confidence test;
- synthetic families can share generator artifacts;
- operator labels and causal interventions can make the desired factorization privileged;
- “beat generic recurrent, context, retrieval, test-time-update, router, and explicit modular controls” does not freeze comparable implementations or tuning budgets; and
- a hand-coded interpreter can pass composition/revision without any learned developmental improvement.

Required Level 1 additions:

1. preregister scaling exponents/curves for `k`, `L`, noise, change count, and state cap;
2. include a capacity-matched dense recurrent learner trained on the identical lifetime objective;
3. include the 2510.01858-style probabilistic modular learner and an explicit Bayesian/program oracle;
4. include shuffled-factor and opaque-permutation negative families;
5. prove no target/operator label enters the candidate interface;
6. require state-swap, operator-lesion, operator-rebinding, and targeted-change causal effects;
7. distinguish edit sparsity from behavioral locality; and
8. freeze the complete cost ledger and all tuning trials.

Passing Level 1 should only admit Level 2.

### 5.2 Level 2: useful transfer evidence, currently underdefined

Three unlike interactive domains are directionally appropriate, but can still be three wrappers over one synthetic grammar. “Without changing the updater or gates” must also forbid domain-specific update prompts, state schemas, learned adapters, privileged parsers, action heuristics, and post-hoc thresholds unless all are treated as charged fixed parts of the model.

Gate repairs:

- **30% lower lifetime regret:** define the per-domain oracle, horizon, normalization, aggregation weights, and treatment of negative/zero denominators. Use simultaneous intervals over environments and seeds.
- **2x fewer interactions:** freeze competence threshold, censoring rule, maximum budget, and whether one action returns one observation. Report extra inference/update compute.
- **10x composition horizon:** freeze training maximum and test depths before training, charge `Omega(L)` execution work, and include the 2510.01858 prior-art baseline. A recurrent loop or interpreter can extrapolate indefinitely without becoming broadly more intelligent.
- **one-point retention:** freeze protected suite, baseline, absolute metric, noninferiority interval, and multiplicity correction. Average retention cannot hide localized catastrophic loss.
- **calibrated recovery:** define proper score, change distribution, detection delay, false-revision rate, recovery criterion, and worst-factor results.

Most importantly, Level 2 must require a fresh-process test:

1. learn a new post-freeze mechanism through interaction;
2. destroy the interaction transcript and all undeclared caches;
3. reload only the bounded declared developmental state;
4. solve new tasks requiring that mechanism in new bindings and a different domain;
5. revise it after a hidden change; and
6. retain unrelated learned mechanisms.

Without this test, full-context recurrence or prose replay can satisfy the level.

### 5.3 Level 3: can support a scoped system claim after major repair

The intent—compare complete systems at one serving envelope—is correct. The current wording is not operational:

- “strongest” is an open-ended comparator and must become a frozen baseline roster plus equal tuning protocol;
- “order-one margin” has no numeric definition;
- “fresh one-shot evaluation” is ambiguous between one hidden evaluation pass and one statistical sample;
- one scalar aggregate can hide domain regressions;
- a larger static baseline cannot be both cost-matched and arbitrarily larger without an explicit serving frontier; and
- an external memory/scaffold result is a complete-system result, not automatically an intrinsic backbone-model result.

Level 3 should use a one-time sequestered evaluation **suite** with enough independent worlds for simultaneous confidence, not one example. Freeze primary domain-level metrics and require strict Pareto admission on:

- post-transcript fresh-process competence;
- new-mechanism acquisition rate;
- interaction cost;
- served inference/update compute and latency;
- persistent bytes and precision;
- calibration;
- hidden-change recovery;
- protected-domain retention; and
- worst-domain rather than only mean performance.

The baseline roster must include:

1. same backbone, full context;
2. same backbone, retrieval/RAG;
3. ACE/Memp-style evolving procedural text;
4. WorldEvolver-style episodic plus mismatch-rule memory;
5. ORBIT-style generic recurrent/meta-RL updater;
6. 2510.01858-style probabilistic modules/gating;
7. test-time gradients with all optimizer state charged;
8. agentic search/planning at matched served work;
9. explicit Bayesian/program composition; and
10. a larger static model on the same serving Pareto frontier.

A passing result would justify:

> “This complete adaptive system acquires, retains, composes, and revises mechanisms substantially better on the preregistered environment distribution under the stated resource envelope.”

It would not justify universal intelligence, unbounded continual learning, or a claim that the frozen base weights themselves improved.

## 6. Anti-prose and anti-synthetic result tests

To meet the user's actual goal, the following are primary gates, not optional ablations.

1. **Text-memory equivalence:** compare against an equal-byte evolving textual playbook given identical evidence and model calls.
2. **Canonical rendering:** evaluate whether the learned state still works when superficial names/order/serialization are counterfactually changed.
3. **Transcript removal:** prove fresh-process performance from only declared state.
4. **Novel mechanism:** generate the mechanism after all model/control/tuning freeze; ensure it is absent from pretraining-recall prompts.
5. **Cross-task utility:** the learned state must improve new tasks, not replay the task that produced it.
6. **Cross-domain utility:** at least one learned operation must causally improve a different independently generated domain under a predeclared interface mapping.
7. **Operator lesion:** removing one learned operator must selectively remove its predicted competence.
8. **Operator swap:** swapping two operator states must swap the corresponding behavior and leave protected behavior intact.
9. **Localized revision:** a hidden change to one mechanism must update relevant behavior with low false revision elsewhere.
10. **No oracle semantics:** no ground-truth factor, operator name, causal parent, or decomposition label is exposed.
11. **Naturalistic transfer:** at least one Level 2/3 domain must not be emitted by the composition-lattice generator.
12. **Full-system mediation:** the candidate advantage must shrink as predicted under operator-state ablation and not be explainable solely by more tokens, calls, interventions, or training trials.

If equal-byte procedural prose matches the candidate, that is evidence for a useful memory system but falsifies the special operator-machine claim. If only synthetic lattice tasks pass, that is a benchmark mechanics result. If only the explicit program oracle passes, the learned discovery problem remains unsolved.

## 7. Minimum theorem and manifest required before architecture design

The next proposal should supply one concrete family and prove:

1. the hidden operator library is identifiable up to a stated equivalence from legal interaction;
2. a declared active policy has a finite acquisition bound including noise/confidence;
3. the learned state is sufficient for future prediction/action after transcript deletion;
4. the composition rule determines held-out programs under explicit typing/local-witness conditions;
5. a single-factor change is identifiable and admits a localized functional repair;
6. an opaque family has the stated exponential lower bound, while all general controls can access the structured family; and
7. state bits, intervention count, execution work, and update work are all separately bounded.

The corresponding frozen manifest must minimally specify:

- the system boundary and exact persistent-state byte/precision cap;
- environment meta-distribution and every expected transferable invariance;
- operator type/interface and program grammar;
- observation/action channels and all costs;
- training families, independent held-out generators, and contamination checks;
- updater, composer, consolidator, and rollback semantics;
- transcript/cache erasure procedure and fresh-process reconstruction;
- baseline implementations and equal tuning budgets;
- scaling grid over `k`, `L`, noise, changes, and state cap;
- primary metrics, oracle/regret definitions, intervals, multiplicity, and Pareto rule;
- direct prior-art reproductions or justified faithful surrogates; and
- a sequestered Level 3 suite that cannot influence architecture selection.

## 8. Final disposition

The T76 document is valuable as a constraint ledger and correctly refuses premature training. Its strongest sentence is that a generic recurrent model may already learn the desired behavior, in which case the lifetime objective—not the structured container—is the result.

The architecture gate remains closed because:

1. the mechanism is a role decomposition, not an algorithm;
2. its claimed exponential advantage is conditional on structure it has not shown a learner can discover;
3. factorized inference and local revision assumptions are not connected to a trainable representation;
4. the closest prior art already covers most of the proposed object;
5. Levels 1 and 2 can be passed by synthetic grammar learning, a hand-coded interpreter, full-context recurrence, or evolving prose;
6. Level 3 lacks a frozen statistical claim and system boundary; and
7. no gate yet proves durable post-transcript cross-domain competence caused by learned operator state.

Do not choose an architecture or run T76 from the current charter. First freeze a distinct operator/state/update mechanism and the anti-prose, anti-generator, fresh-process proof described above. The first admissible output is a theorem-plus-manifest, not a neural experiment.
