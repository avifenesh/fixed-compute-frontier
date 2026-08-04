# T79 intelligence-component localization — find the broken learning channel before proposing a mechanism

Date: 2026-08-02  
Status: **INDEPENDENTLY AUDITED; REVISED; RETAINED AS INTERNAL GATE; NO RUN**

## 1. Objective correction

The project is not trying to make a static next-token predictor slightly
cheaper. It is trying to produce a system that becomes substantially more
capable through experience: it discovers useful structure, seeks evidence,
reasons with what it learned, corrects the responsible error, and retains and
expresses the result.

Compute is an accounting variable and a matched-control constraint. It is not
the optimized object.

The immediate problem is that labels such as `memory`, `world model`,
`reasoning`, `continual learning`, and `self-correction` do not localize the
causal failure. A system may:

- fail to encode the needed information;
- encode it but fail to expose it to the decision process;
- expose it but choose a bad computation or action;
- learn the right behavior and later overwrite it; or
- retain both knowledge and behavior but fail to retrieve the right mode.

Adding a memory module to the second case or a better policy optimizer to the
first case is mechanism drift, not research.

## 2. Exact one-step localization identity

Let:

- `H` be every information source available at the audited decision, including
  history, fixed weights, prior lifetime state, retrieval, tools, and other
  side information;
- `K` be the state actually exposed by the candidate system;
- `Y` be the decision target or outcome-relevant latent;
- `d(K)` be the candidate's answer or action; and
- `ell(d(K),Y)` be a bounded loss.

Define the best achievable risk from an information source `X` as

\[
R^*(X)=\inf_f \mathbb E[\ell(f(X),Y)].
\]

The candidate's excess risk over the best history-based decision decomposes
exactly as

\[
\underbrace{\mathbb E[\ell(d(K),Y)]-R^*(H)}_{\text{total intelligence gap}}
=
\underbrace{R^*(K)-R^*(H)}_{\text{representation / knowledge gap}}
+
\underbrace{\mathbb E[\ell(d(K),Y)]-R^*(K)}_{\text{use / policy gap}}.
\]

This is an add-and-subtract identity. Both terms are nonnegative because `K`
is a Blackwell garbling of `H`: it is deterministic `K=E(H)`, or more generally
is sampled from a kernel with `Y` conditionally independent of `K` given `H`.
The decision classes must also be matched: the class used in `R^*(K)` contains
the deployed rule, and the class used in `R^*(H)` can reproduce decisions made
from `K`. With finite risks, Blackwell monotonicity gives
`R^*(H) <= R^*(K)`, while the definition of the infimum gives
`R^*(K) <= E[ell(d(K),Y)]`.

If target-correlated information enters `K` through an omitted side channel,
or the probe and deployed system use different data, function, or compute
classes, the nonnegativity interpretation is invalid. Empirical probes only
estimate these ideal risks; they require held-out data, uncertainty intervals,
and capacity/control-task checks.

### Consequence T79.1 — a memory-only fix can be causally irrelevant

If `R^*(K)=R^*(H)` but the deployed readout `d` is suboptimal, the entire
measured gap is in knowledge use. Increasing the capacity of `K` does not
address the demonstrated failure unless it also changes the readout channel.

### Consequence T79.2 — policy search cannot reconstruct absent evidence

If `R^*(K)>R^*(H)`, even the best possible decoder from `K` retains that first
term. More search, chain of thought, or policy optimization over the same `K`
cannot remove information discarded before or at the `K` interface.

### Order-one witnesses

Let `Y` be a balanced bit.

1. `K=Y`, while `d` always predicts zero. Then the knowledge gap is zero and
   the use gap is `1/2` under zero-one loss. A perfect memory can coexist with
   50% behavior.
2. `K` is independent of `Y`. Then the knowledge gap is `1/2`; no decoder or
   reasoning budget using only `K` can exceed chance.

The distinction can therefore account for tens of capability points, not a
sub-percent metric effect.

## 3. The one-step identity is not a sequential-return decomposition

The identity is exact only on a fixed reference distribution over `(H,K,Y)`.
For an acting learner, changing the planner or policy changes future actions,
state occupancy, observations, and targets. A return difference then combines
component quality, coverage, compounding error, control stability, and
distribution shift.

Audits must therefore report two distinct objects:

1. **Fixed-distribution audit:** freeze the checkpoint and compare components
   on one exogenous audit distribution `mu` over reset states, histories, and
   queries.
2. **Closed-loop causal audit:** start from matched initial-state distributions
   and paired environment seeds, intervene on one mechanism, and measure total
   return. Call this a total causal effect unless an occupancy or
   policy-performance decomposition justifies a narrower attribution.

At a frozen lifetime checkpoint `t`, use matched interventions:

1. **Knowledge probe:** freeze `K_t`; fit or compute the strongest allowed
   probe from `K_t` to held-out factual, causal, and counterfactual targets.
2. **Use probe:** compare the deployed policy/readout with that probe on the
   same frozen state distribution.
3. **Oracle-knowledge intervention:** replace only the queried knowledge with
   the environment oracle while retaining the learner's planner and policy.
4. **Oracle-policy intervention:** retain the learned knowledge and replace only
   action selection with an exact or search-complete controller where
   available.
5. **Retention intervention:** repeat all probes after unrelated subsequent
   learning, not only immediately after acquisition.
6. **Closed-loop confirmation:** rerun each replacement online and measure
   cumulative regret, while explicitly reporting the induced occupancy shift.

### Retention, retrieval, and use are different interfaces

One `K` conflates several failures. For the next display, condition every risk
on the same current query `Q` (equivalently include `Q` beside each state). Use
the nested information path

\[
H \longrightarrow M \longrightarrow Z \longrightarrow A,
\]

where `M` is retained memory/state, `Z=G(M,Q)` is what retrieval exposes, and
`A=d(Z)` is the deployed output. Under the same information-order
and matched-class assumptions,

\[
\begin{aligned}
\mathbb E[\ell(A,Y)]-R^*(H)
={}&[R^*(M)-R^*(H)]\\
&+[R^*(Z)-R^*(M)]\\
&+[\mathbb E\ell(A,Y)-R^*(Z)].
\end{aligned}
\]

These are formation/retention, retrieval/exposure, and use gaps. An
unconstrained decoder absorbs both planning and readout, so those require
surgical interventions rather than another algebraic label.

| contrast | held fixed | changed | supported inference |
|---|---|---|---|
| learned `M`, forced verified retrieval | planner, readout, memory, native input interface | retrieval/router | retrieval effect conditional on retention |
| same `Z`, oracle planner using only `Z` | representation, retrieval, executor | planning computation | planning effect conditional on exposed information |
| same frozen plan, oracle executor | representation, retrieval, plan | readout/execution | output-channel effect |
| oracle-correct `M` through the native interface | retrieval and downstream computation | stored representation | representation effect, subject to interface matching |
| immediate vs delayed on identical replayed `mu`, with restored-checkpoint control | query/task distribution and audit | intervening learning/time | retention effect rather than ordinary drift |
| fixed-`mu` vs paired closed-loop version of every row | component intervention | occupancy feedback | distribution/compounding interaction |

A probe establishes decodability by that probe, not causal use. Oracle planners
must see only the same `M` or `Z`, oracle injections must use the native
interface with sham controls, and a fresh probe at each checkpoint must
distinguish lost information from coordinate reparameterization. The probes are
diagnostics, not trainable auxiliary losses by default; training on them can
change the representation and destroy the intended localization. Coupled
mechanisms and factorial interactions are allowed and must be reported rather
than forced into a unique component label.

## 4. Why this gate is newly necessary

[The World Model Remembers, the Actor
Forgets](https://arxiv.org/abs/2607.19749) reports exactly the kind of causal
separation this project had omitted: in its DreamerV3 continual-RL setting, the
world model retained old reward discrimination, value, and termination
structure while actor behavior collapsed. With the world model frozen,
supervised self-imitation on graded dreams recovered the old skill in all three
reported seeds, while reinforcement learning on the same imagined rollouts
recovered none. This is evidence for a knowledge-to-policy failure in that
system, not proof that all current models share the same bottleneck.

[Rethinking Continual Experience
Internalization](https://arxiv.org/abs/2606.04703) reports that repeated LLM
experience internalization can progressively collapse rather than compound,
and that durability depends on principle-level abstraction, step-aligned
injection, and an off-policy teacher distribution. This makes an update count
or a post-update training loss an inadequate measure of continual intelligence.

[Mechanistic World Models](https://arxiv.org/abs/2607.12474) argues that reusable
explanatory mechanisms must organize representation, computation, and learning,
while identifying their joint discovery and management as unresolved. That is
a complementary representation-side hypothesis, not evidence that the actor
channel is already solved.

[Towards Mechanistically Understanding Why Memorized Knowledge Fails to
Generalize](https://arxiv.org/abs/2607.08393) directly names and formalizes the
`Knowing--Using Gap` in LLM fine-tuning. Its self-patching intervention reports
recovering 58--75% of oracle generalization headroom, which directly occupies
T79's broad framing. Earlier work on
[representation forgetting](https://arxiv.org/abs/2203.13381), causal probes,
and [causal abstraction](https://www.jmlr.org/papers/v26/23-0058.html) also
occupies the diagnostic methodology. T79 therefore makes no novelty claim.

## 5. Decision after T79

T79 does **not** propose another architecture. The algebra is elementary, the
knowing--using distinction is directly occupied, and the component
interventions have direct precedents. Its retained project contribution is a
hard selection rule:

> No candidate is admitted under the word `intelligence` until it predeclares
> the representation, retention, retrieval, planning, and readout channels it
> may change; demonstrates an order-one end-to-end capability gap; and
> localizes causal mediation as far as matched interventions permit.

The next candidate search therefore has two legitimate branches:

1. **Knowledge formation:** reduce `R^*(K)-R^*(H)` by discovering useful
   variables/mechanisms from raw interaction and revising the responsible
   factor.
2. **Knowledge expression:** reduce the deployed excess over `R^*(K)` by keeping
   policy, reasoning, or answer generation synchronized with what the system
   already knows.

A generic memory, rehearsal, policy-distillation, world-model, or reasoning
module is already occupied prior art. A candidate must provide a new operator
or a new composition theorem and beat those matched controls. Until then there
is no CPU, neural, local-GPU, or rental-GPU experiment to run.

## 6. Substantial model-level gate

Whichever branch survives must improve the same bounded learner across several
unseen families, not one synthetic algorithm:

- at least 20% improvement on a preregistered major end-to-end capability or
  learning-process measure against the strongest equal-complete-cost learned
  control, outside uncertainty;
- report relative and absolute change, error/regret reduction, and available
  headroom so low baselines or ceilings cannot manufacture the percentage;
- treat 10% to less than 20% as provisional and demanding unusually broad
  evidence; reject every single-digit gain as the research outcome;
- either a durable new capability or, for a learning-efficiency claim, at
  least `2x` faster acquisition of a related unseen mechanism;
- no material loss of protected prior capabilities after later learning;
- a causal ablation that removes the gain when the claimed channel is severed;
- persistence from controlled interactive worlds into at least one natural
  tool-use, scientific-revision, or multi-step reasoning setting; and
- a full ledger of model, state, update, interaction, generated-token, latency,
  and energy costs.

This is the corrected target: a substantially stronger learner whose gain has a
localized cause, not an efficiency trick and not an unmeasured agent wrapper.

## 7. Independent-audit decision

The [independent audit](intelligence-component-localization-t79-independent-audit.md)
found the one-step identity valid after the information-order and decision-class
conditions above, but rejected its use as a sequential-return decomposition and
found direct prior-art collisions. Its verdict is **revise, then retain as an
internal gate**. The required revisions are incorporated here. No method,
CPU, neural, local-GPU, or rental-GPU experiment is admitted by T79.
