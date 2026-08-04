# T70 Production-Intelligence Audit: Bounded Persistent Meta-Learner

**Audit scope:** adversarial production-intelligence review of `bounded-persistent-meta-learner-t70.md`; complementary to the mathematical audit. This audit does not modify T70.

**Verdict: HOLD.** T70 should not be closed: a positive leave-one-family-out result could establish a genuinely useful bounded online-learning capability. It should not yet run as written: the cheap screen is underspecified, its strongest controls are not operationally defined, and a passing result can still be explained by compressed episodic memory or meta-distribution lookup. The required edits below are the minimum admission contract for a decisive local run.

## 1. What becomes smarter if every current paper gate passes?

The frozen base model does not acquire new static knowledge or become a better zero-shot model. The **stateful learner system** becomes better at:

- identifying a latent task or environment from a short interaction history;
- choosing informative actions and queries;
- compressing evidence into a bounded state;
- reusing that state after the evidence-bearing context has been erased;
- revising the state when the environment changes;
- applying a learned update policy to a held-out task family.

That is a real intelligence claim, but a narrower one than “the model learns continually.” The capability is conditional on the persistent state `m_t`; reset that state and the gain should disappear. It is per learner, tenant, or environment unless T70 separately demonstrates transfer of the acquired state. It is not knowledge absorbed into the frozen weights.

If all existing gates pass on synthetic streams, the strongest defensible conclusion is:

> A bounded-state, frozen-weight system learned a transferable policy for acquiring, retaining, using, and revising task information across controlled task families, under a fixed serving-resource envelope.

It would **not** yet establish that a 4B–14B language model becomes broadly more capable in production. That requires a later natural-domain pilot with noisy language observations, delayed or partial feedback, real tool actions, and a genuinely held-out domain.

The “smaller adapted model matches a larger frozen model” claim is also currently ambiguous. A persistent-state learner is not a smaller standalone model; it is a smaller backbone plus environment-specific state and update compute. The larger baseline must receive the same observations and must be allowed its best matched context, retrieval, and bounded-state mechanisms. Otherwise the comparison measures denied evidence, not learning ability.

## 2. Is this merely memory or lookup?

### What the current witnesses establish

- T70.1 proves only that information can cross erasure through persistent state. It is intentionally a lookup task and is not an intelligence witness.
- The corrected affine task is stronger: because an optimal control can retain two examples in constant memory, coefficient storage has no memory-complexity advantage. A win can test learned sufficient coding or systematic computation.
- The DFA task can test active identification, but a meta-learner trained on DFA generators can still be an amortized L* implementation rather than a general cross-family learner.

### What would separate learning from compressed episodic memory

A positive result is more than lookup only if all of the following hold simultaneously:

1. The test family is absent from updater training, not merely a held-out instance from a familiar generator.
2. Family identifiers, fixed prompt templates, schema fingerprints, task-boundary tokens, and reusable environment IDs are removed or made equally available to every control.
3. The learner beats a byte-matched episodic store with learned retrieval and compression, not only raw context or FIFO memory.
4. The learner beats an optimal or near-optimal sufficient-statistic/model-based control where one is available.
5. The gain transfers across surface randomizations and unseen horizons, so it cannot be a learned address table or fixed-time policy.
6. The same frozen updater works on multiple rotated held-out families without family-specific fine-tuning, tools, verifiers, or adapters.

Even then, the mechanism will necessarily store information. “Not memory” is the wrong bar. The meaningful question is whether the updater has learned a **transferable acquisition and state-update skill** that produces a better task model than the same storage and work budget used as episodic memory.

The plausible high-value outcome is therefore not a universal learner emerging from lookup. It is a learned, bounded system-identification and exploration prior that transfers across several structurally different families. That could be a large improvement for agents that revisit users, tools, codebases, devices, or environments. It would remain weaker than broad continual learning until demonstrated on natural tasks.

## 3. Control and cost fairness

The resource equation is directionally right, but “same bytes, FLOPs, interactions, and data” is not yet an executable fairness contract.

### Controls that must be present

The decisive screen needs, at minimum:

- reset/no-state and shuffled-state controls;
- full-context information oracle, explicitly labeled as a ceiling rather than a cost-matched baseline;
- byte-matched raw example memory and reservoir memory;
- byte-matched learned retrieval/compression memory;
- an optimal sufficient-statistic or symbolic system-identification control for the affine task;
- the strongest equally trained recurrent meta-RL learner with the same state, observations, actions, horizon distribution, and lifetime objective;
- a Replay-MAML/PTW-style replay baseline where applicable;
- test-time training/adapters, charged for gradients, optimizer state, checkpoints, and backward passes;
- a larger frozen backbone given the same evidence through its best viable context or retrieval path for the final “smaller matches larger” comparison.

All controls must receive the same feedback timing, action space, task-boundary information, reset schedule, and change-point signals. If the candidate receives an erasure or boundary event, either every control receives it or the event must be removed from the candidate.

If the proposed learner is itself a GRU, SSM, or recurrent meta-RL policy with the same lifetime objective, “same-state recurrent control” is not a distinct control. T70 must name the concrete intervention being tested—state writer, training objective, commitment rule, architecture, or update schedule—and compare it against an ablation that differs only in that intervention. If no such intervention exists, T70 is an evaluation program for recurrent meta-learning, not yet a runnable model hypothesis.

### Costs that must be reported separately

Report a resource vector and Pareto curves rather than forcing unlike methods into one scalar:

- offline meta-training FLOPs, simulator/verifier calls, number of lifetimes, and wall time;
- serving forward FLOPs and backward/update FLOPs per interaction;
- persistent bytes at declared precision, including optimizer state, metadata, checksums, checkpoints, and redundant copies;
- context/KV-cache bytes, retrieval-index bytes, query traffic, and memory bandwidth;
- p50 and p95 latency, throughput, batching loss, and multi-tenant state movement;
- interactions, active queries, environment actions, and any risk or monetary cost of exploration;
- amortized total cost at stated deployment volume and the break-even number and length of lifetimes.

The active learner cannot buy lower regret with extra queries unless those queries are charged. Report both regret per environment step and performance at a fixed interaction/query budget.

The production ledger must also include per-tenant state isolation, encryption, serialization, migration, eviction, corruption recovery, deletion, and reset. These do not belong in the mathematical capacity bound, but they do belong in a production claim.

## 4. Is the cheap local falsifier strong enough to earn scaling?

**As written, no.** “Large post-erasure benefit,” “most of the oracle gain,” “unseen horizon,” and “beat the same-state recurrent control” leave enough freedom to accept a lookup system, a weakly tuned baseline, or an in-distribution meta-learner after seeing results.

A cheap local test can earn a **natural-domain pilot**, not a large rental, if it is preregistered as follows:

1. Define at least four structurally different families and rotate each one as the completely held-out family. Train on `F \ {j}`, freeze weights and updater, and evaluate on `j` with no family-specific training.
2. Permit only a predeclared generic observation/action interface. No family label, family prompt, family-specific verifier, schema adapter, or hidden task ID may enter persistent state.
3. Randomize symbols, surfaces, horizons, task order, and change points after training. Include unseen horizons materially longer than the training range.
4. Require the full-context oracle to clear a preregistered competence floor on every held-out family; otherwise a negative result says nothing about the updater.
5. Require the candidate to recover a fixed fraction of the oracle's improvement after erasure. Replace “most” with a number chosen before training; **at least 80%** is a defensible starting threshold for the synthetic screen.
6. Require at least **30% lower lifetime regret** than the strongest non-oracle, resource-matched control and **2x sample efficiency** at a fixed target score, on every held-out-family rotation rather than only in aggregate.
7. Require no more than the predeclared forgetting tolerance on retained tasks, successful change-point revision, and no advantage when persistent state is reset, shuffled, or causally corrupted.
8. Compare against the full control set in Section 3 with equal tuning-search budget and report all seeds, confidence intervals, and failures. Do not select the winning control after the run.
9. Pass the affine and lookup tasks only as mechanism unit tests. Admission depends on the rotated cross-family result; neither unit witness is sufficient.
10. Freeze the implementation, data generators, budgets, thresholds, and analysis code before the decisive seeds.

The small from-zero model is useful only if the full-context oracle proves it can solve the held-out tasks. A failed oracle means the backbone is underpowered; a passed oracle plus failed persistent learner is the desired cheap falsification.

## 5. Exact required edits to T70 before RUN

1. **Section 1 / target claim:** State explicitly that the claimed improvement belongs to the frozen-backbone-plus-persistent-state system, is normally environment-specific, and vanishes on state reset. Rewrite the smaller-versus-larger target to give both systems the same observations and their best resource-matched evidence path.
2. **Section 3 / lookup:** Label T70.1 a causal-path and capacity unit test only. State that it cannot admit the candidate or support a cross-family learning claim.
3. **Section 4 / affine:** Specify held-out primes/rule surfaces, symbol randomization, horizons, and the exact byte/work-matched two-example, symbolic-solver, and learned recurrent controls. Treat it as a systematic-update witness, not the production gate.
4. **Section 8 / resources:** Split offline meta-training, online serving, storage/bandwidth, interaction, and operational state costs. Add amortized break-even, latency/throughput, precision/metadata, and active-query accounting.
5. **Section 9 / controls:** Convert the narrative list into an executable control matrix. For each control, predeclare accessible information, state bytes, compute, interaction budget, tuning budget, and whether it is an information ceiling or a cost-matched competitor.
6. **Section 9 or 10 / intervention identity:** Name the concrete candidate implementation and its causal delta from the strongest recurrent meta-RL control. If it has no delta, relabel T70 as a benchmark/evaluation program and do not claim a new learner mechanism.
7. **Gate 1:** Replace “recover most” with a preregistered fraction; use at least 80% for the synthetic admission screen. Require the oracle competence floor separately for every held-out family.
8. **Gates 2–5:** Remove all candidate-only family IDs, task-boundary signals, and change-point hints, or expose them identically to every control. Add learned compressed episodic memory and task-optimal sufficient-statistic controls.
9. **Gate 6:** Define exact rotated leave-one-family-out evaluation: train on `F \ {j}`, freeze everything, test on `j`, rotate all `j`, forbid family-specific adapters/tools/verifiers, and require the threshold on each rotation. Holding out instances from a seen family does not count.
10. **Section 11 / first screen:** Replace qualitative kill language with the preregistered ten-point screen in Section 4 of this audit, including seed count, uncertainty rule, absolute competence floors, and the exact strongest-baseline selection rule.
11. **Section 11 / scaling ladder:** Preserve the current “no rental” boundary. Make the sequence explicit: mechanism unit tests -> rotated synthetic cross-family screen -> 4B–14B natural-domain pilot -> only then a scaling decision.
12. **Natural-domain pilot gate:** Before any broad production claim, require at least one noisy, delayed-feedback language/tool domain held entirely out of updater training; per-tenant state lifecycle tests; a larger frozen baseline with equal evidence access; and the same regret, acquisition, retention, and cost ledger used in the synthetic screen.

## 6. Final decision

**HOLD, with a narrow path to RUN.** T70 is not just a memory proposal in principle: a frozen updater that wins rotated leave-one-family-out tests against learned episodic memory, task-optimal sufficient statistics, and equally trained recurrent meta-RL controls would demonstrate transferable online-learning skill. That is significant enough to justify a cheap controlled experiment.

The current falsifier cannot yet distinguish that outcome from compressed examples, generator recognition, or an amortized task-specific algorithm. Apply the edits above, choose a concrete learner intervention, and preregister the decisive cross-family screen. Passing it earns a natural-domain pilot—not a claim of production intelligence and not a large scaling run.
