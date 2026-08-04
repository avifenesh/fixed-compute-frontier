# T78 prospective counterfactual abstraction credit — train what becomes useful later

Date: 2026-08-02  
Status: **INDEPENDENTLY AUDITED; METHOD CLOSED; THEOREM BOUNDARY RETAINED; NO CODE OR RUN**

## 0. Corrected research target

The target is not a cheaper attention block and not a better memory wrapper. It
is a model that becomes more capable through experience: it forms a reusable
abstraction, carries it beyond the source episode, applies it under new names
and compositions, and retains unrelated competence.

Current systems already contain strong versions of every obvious component:
cross-episode meta-RL, skill libraries, executable skill programs, continual
weight updates, retrieval, abstraction generators, world models, planners, and
verifiers. Therefore none of those labels is a novelty claim.

T78 asks whether the **credit assignment unit** is wrong. Most model training
rewards the answer to the current example. Most library learning compresses
past solved examples. The candidate instead gives an abstraction delayed credit
for the causal improvement it produces on later, independently generated tasks.

The one-sentence mechanism is:

> Propose a bounded executable abstraction now; train its proposal and
> admission by the future loss difference between otherwise matched learners
> that can and cannot use it.

This document derives the exact separation that this objective can produce and
the equally important cases in which it cannot help. It does not yet establish
novelty, a useful neural implementation, or a smarter production model.

## 1. Why current-task success is not abstraction value

Let task `X_t` yield a solution trace `D_t`. A proposer emits a typed executable
candidate

\[
z\sim q_\phi(z\mid H_t,D_t),
\]

where `H_t` is the legal history before the proposal. A candidate may be a
parameterized program, a finite-state procedure, a reusable tool composition,
or a bounded neural adapter. Its syntax, parameters, bindings, state, trigger,
and runtime are all charged.

Three objectives must be kept distinct:

1. **source utility:** does `z` help solve `X_t`?
2. **retrospective compression:** does `z` shorten solutions in
   `X_1,...,X_t`?
3. **prospective causal utility:** does possessing `z` change outcomes on
   tasks after `t`?

A description may solve or compress its source while being useless under new
bindings. Conversely, a candidate may be redundant on the source task but
remove a large search barrier on later compositions. Only the third quantity
directly measures the requested learning behavior.

## 2. The operator

Let `A` be one fixed base solver and let `L_t` be a persistent library with a
predeclared bit and execution budget. For a candidate born at time `t`, define
two learner branches:

\[
L_t^1=L_t\cup\{z\},\qquad L_t^0=L_t.
\]

The branches use the same base parameters, tools, task budget, decoding rule,
and allowed persistent state. On future task `j`, let their losses be
`ell_j^1(z)` and `ell_j^0(z)`. The horizon-`H` net value is

\[
V_H(z\mid H_t)=
\mathbb E\left[
\sum_{j=t+1}^{t+H}
\left(\ell_j^0(z)-\ell_j^1(z)\right)
-C_H(z)
\middle|H_t\right],
\]

where `C_H` includes proposal, serialization, storage, retrieval, routing,
execution, extra tokens, update work, interference, and eviction opportunity
cost. Training rewards the proposer for `V_H`, not for the source answer or
the prose plausibility of `z`.

### 2.1 What makes the comparison causal

For exogenous task streams, the two branches may receive identical future
tasks and common randomness. Their paired difference is an unbiased sample of
the branch effect under that task generator. Common random numbers reduce
variance; they do not create validity.

For interactive environments, the branches generally choose different actions
and therefore create different histories. Replaying the same observation path
is then an invalid counterfactual. The valid routes are independent environment
replicas drawn from the same declared generator, randomized assignment of the
candidate across many streams, or a separately justified off-policy estimator
with overlap. Every extra replica and interaction is training cost.

### 2.2 Training and deployment

During training, delayed branch return `R_t(z)` can train the discrete proposer
with an ordinary score-function estimator

\[
\nabla_\phi J
=\mathbb E\left[
(R_t(z)-b(H_t))\nabla_\phi\log q_\phi(z\mid H_t,D_t)
\right].
\]

The solver must also be trained to recognize and use a valid abstraction;
otherwise the proposal reward confounds abstraction quality with an untrained
reader. The matched controls receive the same reader-training budget.

At deployment, the system does not run the shadow branch for every candidate.
The proposer and a utility critic predict which candidates to write into the
bounded library. Periodic randomized audits may recalibrate that critic, but
their cost is explicit. If shadow evaluation is required continuously for the
gain to survive, the method has moved inference into an auxiliary training
service rather than improved the served learner.

## 3. Exact positive separation

The first theorem deliberately separates objectives, not universal model
classes.

### Construction T78.1 — predictable future motif

There are two equal-cost executable helpers `z_0,z_1`. At a bridge episode,
both occur equally in the present and past corpus, so their source reward and
retrospective compression score are identical. The learner can retain one.

A hidden future regime `Z` is uniform on `{0,1}`. The legal bridge history
contains a cue `C` satisfying

\[
\Pr(C=Z)=q,\qquad q>\frac12.
\]

The cue is irrelevant to the source solution and past compression. The next
`H` independently surfaced tasks share regime `Z`. On each future task, the
fixed solver has expected loss `ell_1` when the retained helper matches `Z` and
`ell_0>ell_1` otherwise. Let `Delta=ell_0-ell_1`.

### Proposition T78.1 — order-one objective separation

Any symmetric selector whose decision is a function only of the tied source or
retrospective scores matches `Z` with probability `1/2`. The prospective rule
`hat Z=C` matches with probability `q`. Therefore its expected cumulative loss
improvement is

\[
H\Delta\left(q-\frac12\right).
\]

Its per-future-task accuracy improvement in the zero-one loss case is
`Delta(q-1/2)`. At `q=1` and `Delta=1`, this is fifty absolute accuracy points,
not a sub-percent effect.

**Proof.** Symmetry and independence of the tie breaker give match probability
`1/2`; using the cue gives match probability `q`. Conditional expected loss is
`ell_0-Delta Pr(match)`. Subtract and sum over `H`. QED.

This result is narrow but useful. It shows that a future-sensitive learning
objective can have an order-one fixed-budget advantage when present/past
utility is deliberately uninformative and the stream contains predictive
meta-structure.

It does **not** separate T78 from a generic recurrent or meta-RL learner. Such a
learner can also use `C` and attain the same `q`. It also does not prove
abstraction invention: `z_0,z_1` are supplied in the construction.

## 4. No-free-lunch and information boundary

### Proposition T78.2 — no predictable future, no prospective advantage

If `Z` is uniform and conditionally independent of the legal history
`H_t`, then every retained-helper decision measurable with respect to `H_t`
matches `Z` with probability exactly `1/2`.

**Proof.** For any history-measurable decision `d(H_t)`,

\[
\Pr(d(H_t)=Z\mid H_t)=\frac12.
\]

Take expectations. QED.

More generally, with equal priors the best possible regime classifier from
history has accuracy

\[
\frac12\left(1+\operatorname{TV}
(P(H_t\mid Z=0),P(H_t\mid Z=1))\right).
\]

Prospective learning therefore spends real predictive information in the task
stream. It cannot manufacture future structure from a stationary symmetric
process. If a learned future-task model is itself as expensive as solving all
future tasks, the advantage also disappears under complete cost accounting.

## 5. The proposal problem remains unsolved

Selection is not invention. If the proposer assigns zero probability to every
useful candidate, no future reward can recover one. If candidates are mined as
subtrees from supplied exact programs, the abstraction interface has already
been partially solved by the parser and DSL.

An admissible learned formation test must therefore hide:

- the future regime label;
- helper identity and human names;
- reusable subprogram boundaries;
- surface bindings and symbol names; and
- the future tasks used for delayed credit.

The same proposer must recover a typed computation from several raw traces,
survive transcript deletion, and transfer under renamed symbols and unseen
compositions. An exact enumerator over the charged DSL and a generic neural
meta-learner are mandatory controls.

## 6. What is and is not occupied by current work

Current primary sources checked on 2026-08-02 establish strong collisions:

- **Prospective Compression in Human Abstraction Learning** already states the
  normative principle of selecting helpers to compress a predicted future
  corpus and supplies human evidence in non-stationary Pattern Builder
  curricula. T78 cannot claim the prospective-abstraction principle.
  https://arxiv.org/abs/2605.09985
- **RLAD** jointly trains an abstraction generator and a solution generator,
  rewarding abstractions through solver success on the same reasoning problem.
  https://arxiv.org/abs/2510.02263
- **ORBIT** trains cross-episode online adaptation with meta-RL, so delayed
  future utility is not a novelty at the generic learner level.
  https://arxiv.org/abs/2602.04089
- **AgentCL** supplies controlled compositional task streams and shows that
  naive or held-out streams often expose little transfer or memory-induced
  degradation. Its stream controls are mandatory.
  https://arxiv.org/abs/2606.02461
- **SCoL** meta-trains sparse layer updates that consolidate context into an
  evolving model state. Persistent learned updates are therefore occupied.
  https://arxiv.org/abs/2605.07076
- **ALMA** searches over open-ended executable memory update, storage, and
  retrieval programs and scores each design by a fixed agent's performance on
  later deployment tasks. It is a direct operational collision with
  future-performance credit for learned executable persistence. ALMA is
  offline, searches whole memory designs rather than per-experience typed
  abstractions, and explicitly leaves dynamic online design learning open, but
  those are scope differences rather than a new principle.
  https://arxiv.org/abs/2602.07755
- **ReSkill** creates, compares, revises, and prunes trigger-based skills inside
  agentic RL using controlled within-group rollouts and policy co-evolution. It
  is a close collision with causal skill utility.
  https://arxiv.org/abs/2606.01619
- **SkillMaster** is the decisive direct collision. It trains one policy to
  create, refine, and select skills; evaluates each candidate skill edit by its
  counterfactual utility on related probe tasks; separates task-solving and
  skill-editing advantages; and reports `8.8` and `9.3` absolute success-rate
  gains on ALFWorld and WebShop. T78's proposed credit signal is therefore an
  existing method, not an unoccupied seam.
  https://arxiv.org/abs/2605.08693
- **HASP** turns skills into executable program functions and reports large
  reasoning, search, and coding gains. Executability is occupied.
  https://arxiv.org/abs/2605.17734
- **ReuseRL** rewards structurally reusable skill dictionaries through an MDL
  objective, and **Notes to Self** extracts and reuses experiential
  abstractions. Retrospective skill abstraction is crowded.
  https://arxiv.org/abs/2605.31509
  https://arxiv.org/abs/2607.20372

No method-level seam remains. Prospective Compression owns future-sensitive
abstraction selection; ALMA scores executable persistence designs on later
deployment tasks; and SkillMaster directly supplies counterfactual skill-edit
utility on related probes. Requiring an independently generated later stream,
a particular type system, or a more complete resource penalty would improve
evaluation discipline, not create a new learning principle.

## 7. Strongest controls

A model experiment would need all of the following at matched complete cost:

1. same-task abstraction reward in the RLAD style;
2. retrospective MDL/library learning;
3. raw episodic memory plus retrieval;
4. ReSkill-style version comparison and skill evolution;
5. a generic recurrent cross-episode meta-RL learner with the same persistent
   bits, parameters, interactions, and update FLOPs;
6. continual weight update/TTT or SCoL-style consolidation;
7. a larger static model or more ordinary RL training bought with T78's extra
   branch and environment cost;
8. exact enumeration or program-library induction when the DSL is finite; and
9. oracle future-regime and oracle-helper ceilings.

The causal ablations are: remove future-task reward, shuffle future order,
destroy cue-regime dependence, swap helpers between streams, erase the source
transcript, rename all bindings, lesion the accepted helper, and replace the
typed helper with an equal-bit opaque memory record.

## 8. Complete cost equation

For `N` proposal points and future horizon `H`, a naive paired training design
can approach twice the future solver work:

\[
C_{train}=C_{base}
+N(C_{propose}+C_{serialize})
+C_{paired\ future\ solves}
+C_{utility\ critic}
+C_{solver\ reader\ training}.
\]

Deployment adds

\[
C_{serve}=C_{base\ serve}
+B_{library}
+C_{retrieve}+C_{route}+C_{execute}+C_{update}.
\]

Model calls, generated tokens, environment interactions, CPU execution,
storage, network traffic, wall time, and energy remain separate ledger rows.
The correct control may spend the additional training budget on more RL data,
larger batches, a better teacher, or a larger base model.

## 9. Run decision

No code, local GPU, or rented GPU is justified. The exact construction is
simple enough that running it would only animate its assumptions, while
SkillMaster already tests the proposed causal credit mechanism in LLM agents.

Any future method that uses this retained control must test candidate formation,
not supplied-helper selection, and should pre-register an order-one gate:

- at least `30%` lower cumulative future regret or `10` absolute points over
  every equal-cost learned control;
- at least `2x` fewer future interactions to criterion;
- no protected old-task loss above one absolute point outside uncertainty;
- transfer after transcript deletion, binding renaming, and unseen
  composition;
- causal collapse under future-signal destruction and helper lesion; and
- a complete training/deployment ledger.

Passing a synthetic learner screen would admit a small open-model test. Only a
cross-domain model result on coding, tool use, and reasoning streams could
support the requested claim that the resulting model is substantially smarter.

## 10. Verdict

T78 identifies an important training signal: **future causal value of learned
computation**. Its positive theorem is real and permits a large effect. Its
boundary is equally decisive: it requires predictable future structure, does
not solve candidate invention, is emulable by generic meta-RL, and may cost
nearly a second training branch.

The proposed method is nevertheless closed by direct current work, especially
SkillMaster. The independent audit further shows that the common-baseline
paired objective has the same argmax as ordinary cost-sensitive held-out risk,
and that singleton admission fails under skill complementarity. Retain
T78.1/T78.2 as controls for future proposals; do not run or relabel the
mechanism as a breakthrough.
