# T71 developmental-intelligence gap — independent audit

Date: 2026-08-02  
Status: **REVISE BEFORE FREEZE; OBJECTIVE RESET IS SOUND; NO RUN IS ADMITTED**

## Verdict

T71 makes the right program-level correction: persistent state is a component,
not the objective, and no T70 scale-up is justified. Its proposed grammar and
same-backbone ladder are a credible *synthetic admission screen*. They cannot by
themselves establish domain-general intelligence or a substantial resulting-model
gain.

Three corrections are required before even the CPU grammar is frozen:

1. T71.3 proves a **zero-shot output-label grounding** no-free-lunch result. It
   does not prove that learning or transfer after held-out-family feedback is
   impossible, and therefore does not by itself invalidate T70's online
   four-family protocol.
2. The central gap is a proposed joint evaluation contract, not an established
   missing capability. Current work already directly trains cross-episode
   learning, consolidation, active belief tracking, abstraction, and update
   selection. T71 may claim that no cited result has established the complete
   conjunction under matched controls; it may not infer a unique, generally
   missing “developmental update policy.”
3. T71 names slow consolidation into `theta` but defines no deployment update to
   `theta`. The formal object and the experimental ladder currently test
   persistent state and online adaptation, not the promised third timescale.

The current no-run decision should stand.

## Required exact corrections

### 1. Narrow T71.3 and its inference for T70

The permutation argument is valid under the stated construction only when the
`F_*` label names have no observable semantics, grounding examples, reward, or
side information before scoring. More precisely, learners have the same **state
distribution**, not necessarily the same realized state, across the coupled
worlds. Averaging exact-label accuracy over a uniform permutation gives `1/K`.

Once informative `F_*` feedback arrives, the theorem no longer applies. A learner
can ground the permutation and can be compared by regret or feedback sample
complexity. The theorem therefore cannot carry the broad heading “arbitrary
cross-family transfer is impossible without shared structure.” Replace it with:

> **T71.3 — zero-shot output-label grounding is impossible under an arbitrary
> permutation.** Before any held-out-family label grounding, and absent semantic
> or reward side information, exact label accuracy averaged over the `K!`
> permutations is `1/K`. This does not rule out learning after held-out-family
> feedback. It shows that a zero-shot label-name score cannot identify reusable
> learning separately from label grounding or prior knowledge.

Replace the T70 inference with:

> The lookup/affine/parity/DFA rotation remains useful as an online stress test,
> including comparison of acquisition regret after feedback. It is not a
> decisive transfer experiment until the transferable object and common
> meta-distribution are declared. Hold the present manifest for that reason,
> then test held-out compositions, bindings, surfaces, horizons, and mechanism
> changes within the declared grammar. Treat a foreign natural domain as
> external validity, not as a consequence of the permutation theorem.

That preserves the correct T71 decision without claiming more than the proof.

### 2. Repair the T70/README status contradiction

The T70 status says “cross-family manifest not yet frozen,” consistent with T71.
The body still says the executable matrix is frozen (T70 lines 393–397), fixes a
scaling ladder (lines 453–457), and says the lane is ready for the preregistered
cheap screen (lines 459–462). README lines 243–250 likewise describe the old
audit/preregistration as a live admission before later reporting the completed
operator smoke. These cannot all be current.

Replace those live-admission statements in T70 and README with:

> The prior preregistration is retained as a historical control inventory, not a
> currently frozen or admitted manifest. Its cheap lookup operator smoke has
> been completed: causal state transport passed and the direct writer-credit
> hypothesis failed. T71 supersedes the remaining cross-family admission. No
> further T70 screen is admitted until a shared mechanism grammar, held-out
> split, and updated comparator matrix are frozen under T71.

The T70 operator result itself is represented correctly: it closes direct
cross-boundary writer credit on finite lookup, not learned writing generally.

### 3. State the gap as an unestablished conjunction

The sentence that training is “still usually an answer or trajectory” is too
broad immediately after citing counterexamples. Primary results now include:

- [ORBIT](https://arxiv.org/abs/2602.04089), which meta-trains cross-episode
  online learning on unseen interactive environments;
- [Online Experiential Learning](https://arxiv.org/abs/2603.16856), which loops
  experience extraction and parameter consolidation;
- [Self-Consolidating Language Models](https://arxiv.org/abs/2605.07076), which
  meta-trains sparse self-selected weight updates across streams;
- [ALMA](https://arxiv.org/abs/2602.07755), which meta-learns executable memory
  designs; and
- [Self-Consolidation for Self-Evolving Agents](https://arxiv.org/abs/2602.01966),
  an omitted direct collision on reflection plus parameter consolidation.

The July 2026 [self-improving-agent survey](https://arxiv.org/abs/2607.13104)
also already formalizes a system-level self-induced “update operator.” It is a
terminology/taxonomy collision, not evidence that the conjunction is solved.

Use this narrower claim:

> Existing results establish strong instances of the component operations, but
> the cited evidence does not yet establish one system that jointly acquires,
> actively tests, revises, transfers, retains, and consolidates knowledge across
> a declared broad task distribution under matched information and full-cost
> controls. T71 treats that conjunction as an evaluation target, not as a novel
> component or a proven universally missing mechanism.

Also replace “domain-general” with “broadly transferable over a declared task
distribution” until natural-domain evidence exists. The T3 citation should use
its primary [paper](https://arxiv.org/abs/2510.12264) and say “up to 30 points
across five tasks,” rather than the vague “order-ten-point gains.” RLAD is
accurately described, but its stable primary link is
[the paper](https://arxiv.org/abs/2510.02263).

### 4. Type the third timescale or defer the claim

The displayed learner updates `w_t` and `m_t` while `theta` is only the
meta-trained parameter of `J(theta)`. It cannot simultaneously be “slowly
consolidated” at deployment without an update rule, state budget, optimizer,
feedback source, retention constraint, and cost.

Either defer consolidation explicitly:

> Stages A–C test working and bounded developmental state with deployment
> parameters fixed. Parameter consolidation is a later stage and is not yet
> represented by this operator.

or add a typed slow event such as
`(theta_{j+1},q_{j+1}) = C_phi(theta_j,q_j,D_j,m_{1:T})`, charge `q`, replay,
backward work, and selection data, and test post-consolidation acquisition,
retention, rollback, and provenance. Without one of these changes, item 7 of the
objective is outside the experiment.

## Boundary checks

- **T71.1: pass with assumptions made explicit.** `R/2` versus `1/2` expected
  errors is correct if each reset query excludes prior feedback, the first
  forced answer is the revealing interaction, and `Z` is post-training uniform.
  It proves a persistence channel only.
- **T71.2: pass after quantitative separation.** “Same distribution over every
  passive history” must quantify over every allowed passive collection policy.
  Identification after `a*` also needs a nonzero separation parameter; for
  example, sample complexity scales with the relevant KL/TV gap and confidence.
  Rename the boundary “passive equivalence blocks causal identification,” not
  causal intelligence.
- **T71.3: valid theorem, overbroad title and application.** Apply the exact
  correction above.

## Can the grammar and ladder distinguish reusable learning?

Not yet from the written controls. A finite grammar can be solved by recognizing
a task family and invoking a memorized solver, while fresh surface symbols only
rule out literal surface replay. Before freezing Stage A/B, add all of:

1. latent-program/AST holdouts, factorial composition holdouts, post-training
   random bindings, and no family identifier or template fingerprint;
2. a state bit budget below verbatim experience, byte-matched retrieval and
   example stores, and a full-context **information ceiling** kept distinct from
   resource-matched baselines;
3. counterfactual queries and interventions whose correct answers require the
   inferred rule, plus targeted rule/state swaps showing corresponding answer
   changes;
4. the same checkpoint, data order, optimization/tuning budget, and declared
   meta-training grammar for every neural entrant; and
5. task-optimal program-induction/Bayesian controls and classical algorithms
   where available. The agentic-automata result itself finds current LLM agents
   far less robust and efficient than classic automata learners
   ([primary paper](https://arxiv.org/abs/2606.16576)).

Even after these controls, the result is reusable learning **within the charged
grammar**. That is legitimate and useful, but not domain-general development.

## Do the gates imply substantial resulting-model gains?

No. They are strong synthetic admission gates, not resulting-model gates. Define
the regret denominator and absolute gain, `10x` train/test horizons, competence
threshold, revision calibration metric, noninferiority confidence interval, and
multiple-comparison rule before freezing. A causal drop under corruption proves
dependence on state, not that the state contains a reusable rule; targeted
counterfactual swaps are also needed.

Most importantly, the natural pilot must compare complete Pareto frontiers, not
only a smaller adaptive model with a larger frozen model. The strongest larger
model must receive its best declared context/retrieval/adaptation path with equal
evidence and all state/update costs charged. Require a preregistered large
end-to-end gain in at least one natural domain plus retention and transfer before
using “substantially more intelligent resulting model.” A Stage C pass earns
that pilot; it does not make the final claim plausible by itself.

## Final disposition

- Keep T71 as the active objective reset.
- Keep T70 demoted and all further runs held.
- Correct T71.3, the T70/README admission language, the gap claim, and the missing
  consolidation type before freezing the CPU grammar.
- After correction, Stage A mathematics/executable specification is the right
  next artifact. No GPU experiment is admitted by this audit.
