# Research charter: from static prediction to real learning systems

Date: 2026-08-02  
Status: **ACTIVE OBJECTIVE; MECHANISM NOT YET SELECTED**

## Corrected objective

Identify what current AI systems fundamentally lack for real intelligence, then
find an architectural, training, memory, reasoning, or hybrid change that
produces a substantial model-level improvement at a defensible complete cost.

Raw-prose compilation and digital memory are possible mechanisms, not the
goal. Continual learning, reasoning, abstraction, world models, memory,
self-correction, active learning, and learning efficiency are all in scope.

The project still rejects `better = merely more parameters, more sampled
tokens, or more hidden external work`. A legitimate result may spend a new
resource, but it must name, measure, and justify that resource.

**The optimized object is lifetime intelligence, not efficiency.** Compute,
memory, interaction, and latency are accounting variables and controls. They
become primary only when two systems learn equally well. The intended result is
a model or bounded learning system that becomes materially more capable through
experience—not a cheaper implementation of an otherwise unchanged predictor.

## Operational meaning of intelligence

This project does not try to settle a philosophical definition. It asks whether
one system can do all of the following measurably better than another:

1. **Learn from new experience:** improve after observations or outcomes that
   were unavailable during pretraining.
2. **Retain:** acquire a new fact, skill, or model without erasing unrelated
   old capability.
3. **Abstract:** convert several experiences into a reusable rule that helps on
   unseen instances and compositions.
4. **Model interventions:** predict how actions change the environment, not
   only what text or observations usually follow.
5. **Act to learn:** choose informative experiments when uncertainty matters.
6. **Reason:** compose latent facts, rules, and counterfactual transitions over
   horizons longer than a memorized response pattern.
7. **Correct itself:** use prediction error or external outcome evidence to
   localize and repair the responsible belief or procedure.
8. **Know uncertainty:** distinguish absent evidence, conflicting evidence,
   environmental change, and confident knowledge.
9. **Consolidate:** preserve useful experience across episodes while bounding
   memory, update work, and interference.
10. **Express learned knowledge:** reliably turn a retained fact, mechanism, or
    skill into the answer or action it implies. A correct internal world model
    with a policy that cannot exploit it has not closed the learning loop.

A system that starts with a higher static score but cannot improve through
experience is more capable at initialization; it has not demonstrated a
stronger learning process.

## The central hypothesis

The standard foundation-model core is usually optimized as an **amortized
predictor**:

\[
p_\theta(x_{t+1}\mid x_{\le t}),
\]

then deployed with nearly fixed `theta`. Current systems already add prompt
state, retrieval, recurrent memory, world models, search, fast weights,
meta-RL, causal discovery, and continual-learning modules. None of those labels
alone names the missing breakthrough.

The unresolved integration is a **self-maintaining executable knowledge
lifecycle**:

```text
raw partial experience -> variables/hypotheses -> executable prediction
     -> informative action -> outcome -> localized revision
     -> reusable abstraction -> policy/answer expression
     -> retention and consolidation
```

The working hypothesis is therefore not “attention is unintelligent” or
“models need memory.” It is:

> Current systems contain strong fragments of intelligence, but do not yet
> reliably discover their own useful variables and reusable mechanisms, bind
> them into an executable world model, actively resolve consequential
> ambiguity, and revise the responsible knowledge without global forgetting.

This is falsifiable. A generic recurrent, context, TTT, meta-RL, or modular
memory learner that acquires, recombines, and safely revises the same knowledge
at the same complete cost defeats any specialized proposal.

## Current exclusion boundary after T75--T78

The following are valuable capabilities but can no longer be named as the
missing operation or as standalone novelty:

- persistent state or cross-episode meta-RL;
- active selection among a supplied finite hypothesis class;
- an anytime statistical gate for admitting a supplied predictor;
- textual, typed, or executable skill libraries;
- retrospective or prospective library compression;
- future deployment reward for a memory or skill design;
- counterfactual probe-task utility for a skill edit; and
- sparse continual updates to model layers.

Current work directly occupies these families. In particular,
[Prospective Compression](https://arxiv.org/abs/2605.09985) owns the normative
future-sensitive abstraction principle, [ALMA](https://arxiv.org/abs/2602.07755)
selects executable memory programs by later deployment performance, and
[SkillMaster](https://arxiv.org/abs/2605.08693) trains autonomous skill edits by
counterfactual utility on related probe tasks. The project's
[T78 audit](results/prospective-counterfactual-abstraction-credit-t78-independent-audit.md)
also shows that a common-baseline paired score has the same candidate argmax as
ordinary cost-sensitive held-out empirical risk.

The surviving research frontier is not another memory representation or
admission score. It is the coupled raw-learning problem:

1. form candidate variables, mechanisms, or algorithms from unsegmented partial
   observations rather than from supplied slots, programs, helper boundaries,
   task IDs, or human names;
2. acquire external evidence that distinguishes useful structure from a
   plausible compression or self-generated story;
3. assign that evidence to the responsible representation or procedure rather
   than globally reinforcing the whole trace;
4. reuse the result under new surfaces and compositions;
5. make the learned result reliably control answers and actions; and
6. preserve unrelated competence at a fully charged update and serving cost.

Any next candidate must contribute a concrete operator to at least the first,
third, or fifth item. A new router, verifier, memory schema, skill syntax, or
future reward without a candidate-formation, error-localization, or
knowledge-to-behavior theorem is a control, not a new direction.

## Why this hypothesis is worth testing now

Current evidence does not prove the hypothesis, but it localizes high-leverage
slack:

- [ORBIT](https://arxiv.org/abs/2602.04089) reports that cross-episode meta-RL
  lets Qwen3-14B learn online in unseen environments at a level matching
  GPT-5.2, suggesting that learn-to-learn training can substitute for a large
  amount of static scale on that domain.
- [PABU](https://arxiv.org/abs/2602.09138) reports a 23.9-point completion gain
  and 26.9% fewer interaction steps from compact progress-aware belief update,
  while [Unified Memory Agent](https://arxiv.org/abs/2602.18493) jointly trains
  memory operations and answering. These are direct controls against any claim
  that belief or memory management is absent from current systems.
- [T3 belief-deviation control](https://openreview.net/forum?id=r8hzDA3pUY), an
  ICLR 2026 oral, reports gains up to 30 points while cutting token cost up to
  34%, suggesting that preserving a correct belief trajectory matters more
  than simply extending reasoning traces.
- [RLAD](https://openreview.net/forum?id=fvJPjCioeR) explicitly trains models to
  discover and reuse abstractions instead of relying on depth-first reasoning
  traces.
- [Agentic automata learning](https://arxiv.org/abs/2606.16576) finds that
  frontier LLM agents still degrade sharply when they must actively infer a
  hidden world model, with failures in experiment selection, evidence
  integration, and hypothesis construction.
- [In-Place TTT](https://arxiv.org/abs/2604.06169) and the broader fast-weight
  family show that test-time parameter updates are technically viable, while
  still exposing update cost, objective alignment, memory capacity, and
  forgetting as open problems.
- [ContextLM](https://openreview.net/forum?id=ARA2xneUDG) reports a shifted
  parameter/performance frontier from a higher-level predictive objective,
  reinforcing that token likelihood need not be the best finite-budget credit
  path even though it contains the full sequence likelihood.
- [Attractor Models](https://arxiv.org/abs/2605.12466) report parameter-matched
  language-model and hard-reasoning gains from adaptive fixed-point refinement
  with implicit differentiation. Fixed-depth computation is therefore a live
  architectural seam, and equilibrium/recurrent reasoning is a mandatory
  control rather than an unexplored label.
- [Mechanistic World Models](https://arxiv.org/abs/2607.12474) explicitly place
  reusable explanatory mechanisms at the center of representation, learning,
  and computation while identifying joint variable/mechanism/structure
  discovery, partial observability, active inquiry, and mechanism management as
  open integration problems. This makes “use causal mechanisms” a control and
  localizes the harder lifecycle seam.
- [Modular Memory is the Key to Continual Learning
  Agents](https://arxiv.org/abs/2603.01761) argues for combining in-context and
  in-weight learning through modular memory. Persistent or modular memory is
  therefore also a mandatory control, not a novelty claim.
- [The World Model Remembers, the Actor
  Forgets](https://arxiv.org/abs/2607.19749) reports a component-level
  intervention in which the world model retained old reward, value, and
  termination information while the actor's behavior collapsed. Graded dream
  self-imitation restored the behavior in its tested setting while ordinary
  reinforcement learning on the same imagined rollouts did not. This makes the
  knowledge-to-policy channel a distinct causal seam rather than an assumption.
- [Rethinking Continual Experience
  Internalization](https://arxiv.org/abs/2606.04703) reports progressive rather
  than compounding capability under repeated LLM experience internalization,
  with strong dependence on abstraction granularity, step alignment, and the
  teacher distribution. Repeated updating alone is therefore not evidence of a
  stronger learner.
- [Towards Mechanistically Understanding Why Memorized Knowledge Fails to
  Generalize](https://arxiv.org/abs/2607.08393) directly formalizes a
  `Knowing--Using Gap` in LLM fine-tuning and reports that activation
  self-patching recovers 58--75% of oracle generalization headroom. The
  knowledge-to-computation channel is therefore a demonstrated target, but not
  a novelty claim by itself.

These works are controls and evidence, not a recipe to copy.

## Mathematical evaluation object

Let an environment `z` be drawn from a family `Z`. At episode `t`, the system
has slow parameters `theta`, bounded persistent learner state `m_t`, receives
observation `o_t`, chooses action or answer `a_t`, observes feedback `f_t`, and
updates

\[
m_{t+1}=U_\theta(m_t,o_t,a_t,f_t).
\]

The system is evaluated as a **learner trajectory**, not as one frozen function.

### 1. Cumulative learning regret

For a policy `pi`, let

\[
J_T(\pi;z)=\mathbb E_{\tau\sim P_z^\pi}
\left[\sum_{t=1}^{T}\ell_t(\tau_{\le t})\right],
\qquad
\operatorname{Reg}_T=J_T(\pi_{learner};z)
-\inf_{\pi\in\Pi_z}J_T(\pi;z).
\]

The policy-induced trajectory distribution is part of the definition; an
active learner and its comparator need not visit the same states.

Report the complete learning curve and area under it. A late asymptotic win
does not hide catastrophic early cost; a high initial score does not hide zero
adaptation.

### 2. Retention and backward transfer

For earlier task `i`, let `q_i(t)` be quality after later learning. Report

\[
F_i=\max_{s\le t_i}q_i(s)-q_i(T)
\]

and the distribution across tasks. Positive backward transfer is reported
separately from forgetting.

### 3. Forward transfer

Compare episodes or examples needed to reach a fixed competence on a new task
after related prior tasks versus a fresh learner with the same initial model.

### 4. Learning efficiency

Every point includes:

\[
(P_{static},B_{persistent},B_{workspace},F_{forward},F_{update},
T_{traffic},N_{calls},N_{generated},E_{joules},L_{wall}).
\]

Backpropagation, replay, retrieval, simulation, verifier calls, and sleep-time
consolidation are charged. “Test-time learning” is not free inference.

### 5. Calibration and belief revision

Measure log score or Brier score before and after evidence, contradiction, and
change points. Correct answers without calibrated update behavior do not prove
a better learner.

## Capability worlds before natural-language scale

The first test suite must isolate missing operations rather than reward a model
for memorized benchmark conventions.

1. **Persistent fact stream:** arbitrary facts arrive after training, are
   queried after long gaps, conflict at known or hidden change points, and must
   not erase old unrelated facts.
2. **Latent rule families:** infer a reusable rule from few episodes, then
   apply it to renamed symbols, longer compositions, and unseen combinations.
3. **Active automata discovery:** infer a hidden finite-state environment by
   choosing informative queries; compare with exact algorithmic baselines.
4. **Causal intervention worlds:** observational correlations are insufficient;
   the learner must intervene, update a world model, and plan under it.
5. **Delayed-credit worlds:** outcomes arrive after distracting actions;
   successful adaptation requires attributing error to the correct decision.
6. **Self-correction pairs:** the same injected error appears as the model's
   own output and as an external proposal; measure diagnosis and durable repair.
7. **Nonstationary worlds:** mechanisms change; the learner must distinguish a
   change point from ordinary noise and retain unaffected knowledge.
8. **Natural transfer:** only after the controlled worlds pass, test evolving
   factual knowledge, interactive software tasks, scientific hypothesis
   revision, and multi-step reasoning.

Each world has random-label and structure-destroyed controls. The method must
fail where its information assumptions fail.

## What would count as substantial

Before a screen, each world freezes its distribution, horizon, loss, confidence
rule, interaction budget, policy comparator, and the exact normalization of its
primary effect. Oracle Bayes or exact algorithmic learners are ceilings to
approach, not learned controls that a candidate must beat. Report relative
change, absolute change, error/regret reduction, and fraction of available
headroom where meaningful; the pass criterion is chosen before results.

The magnitude rule is:

- **proved success:** at least 20% improvement on a preregistered major
  end-to-end capability or learning-process measure, with the relevant frozen
  confidence bound reaching 20% and the gain covering a substantial part of
  the learning loop rather than one narrow submetric. Twenty percent is the
  acceptance floor; the method should be designed for a larger effect;
- **arguable, not success:** 10% to less than 20%. Broad replication, transfer,
  causal evidence, and no important regression may make it a real finding, but
  it does not satisfy the project goal. It earns further work only when one
  bounded replication can plausibly resolve uncertainty across the 20% floor;
  and
- **reject as the research outcome:** any single-digit improvement, even when
  statistically significant. It may remain an engineering observation but is
  not the result this project seeks, and it closes the candidate as the main
  research lane.

For a sampled comparison, a point estimate above 20% is insufficient when its
confidence bound crosses below 20%. The relative denominator and direction are
frozen before measurement; absolute change and fraction of available headroom
are always reported as diagnostics, not selected afterward as alternate pass
metrics.

A candidate counts as successful only if it also clears all of these against
every matched *learned* control across unseen world families and seeds:

1. the 20% magnitude rule above on the frozen primary end-to-end metric;
2. nonpositive median forgetting and no protected task losing more than one
   absolute point outside its confidence interval;
3. for an acquisition-efficiency claim, at least 2x fewer interactions to
   acquire a new related rule or world model; a durable new capability may
   instead qualify without being acquired faster;
4. a causal drop when the learned belief, abstraction, or update edge is
   shuffled or disabled;
5. the same system—not a task-specific rewrite—wins continual, abstraction,
   and active-world-model families; and
6. an explicit complete-cost point—parameters, state, traffic, update work,
   interactions, generated tokens, energy, and latency—that can plausibly
   deploy. Extra cost is allowed only when the capability gain is substantial
   and remains preferable to spending that cost on the strongest control.

The later model-level target is stronger: after bounded experience, a smaller
learner should match or beat a substantially larger frozen model while
retaining its initial competence. A result confined to one toy algorithm, a
single-digit gain, or a sub-percent NLL change is not the goal.

## Mechanism-neutral candidate map

| candidate object | possible missing ability | decisive control |
|---|---|---|
| persistent belief state with typed uncertainty | evidence integration, contradiction handling, active learning | generic recurrent state and external memory with equal bits/work |
| learned fast weights / online optimizer | rapid adaptation inside the model | In-Place TTT, Titans/ATLAS-style memory, ordinary gradient update |
| reusable abstraction/program library | transfer and compositional reasoning | retrieved demonstrations, MoE, larger static model, RLAD-style abstraction |
| causal world model plus planner | intervention and counterfactual action | model-free meta-RL with equal interactions and total work |
| error-attribution and consolidation loop | self-correction without global forgetting | replay, regularization, model editing, full retraining |
| adaptive recurrent reasoning workspace | depth and revision without long textual traces | equal-token CoT, recurrent/implicit models, search/verifier loops |

No row is selected by this table. A candidate enters only after it has a
positive construction, a counterexample, a matched-control separation, a full
update/serving ledger, and a path to the substantial gate.

## First algebraic question

T45 through T52 establish a boundary, not an architecture. Persistent state,
active interventions, learned experiment ranking, and a reusable mechanism
library all help, but each becomes a classical control once the correct state
variables, tests, or mechanism interface is supplied.

For histories `h,h'`, the controlled predictive relation remains useful:

\[
h\sim_I h'
\Longleftrightarrow
\forall \pi,\quad
P(O_{future},R_{future}\mid h,\operatorname{do}(\pi))
=P(O_{future},R_{future}\mid h',\operatorname{do}(\pi)).
\]

The harder question is now joint discovery and maintenance:

> From raw partial observations and bounded actions, under what weakest
> assumptions can a learner discover a reusable quotient of controlled
> futures, factor it into executable mechanisms, transfer those mechanisms to
> new compositions, and revise only the factor invalidated by later evidence?

T53 proves the elementary information boundary: two worlds with identical
passive observation laws but different interventional laws cannot be
distinguished from passive prose, regardless of model size. It then defines
the required impossibility/sufficiency theorem pair for removing T52's supplied
state slots and mechanism interface.

This is a research question, not an admitted architecture. A hand-coded latent
factor, ontology, causal graph, likelihood, mechanism bank, or compiler is an
oracle/control. The candidate must learn the interface and charge its cost.

## Admission order

1. Construct the smallest raw partial-observation worlds whose latent
   mechanisms are action-observation equivalent, and state the unavoidable
   quotient.
2. Add the weakest declared variation or intervention assumption that restores
   identifiability; derive sample, computation, storage, and update bounds.
3. Prove whether reuse plus localized revision scales with changed mechanisms
   rather than total knowledge after charging representation discovery.
4. Compare with the strongest generic recurrent, context, TTT, meta-RL, causal
   representation, exact-search, and modular-memory controls before naming a
   new module.
5. Run exhaustive CPU theorem worlds only after the theorem pair survives.
6. Run a tiny local model only if the theorem worlds leave one learned-interface
   question.
7. Check the local GPU before any rental; rent only after a large, replicated
   local effect earns the target machine.

No GPU work is admitted by this charter.
