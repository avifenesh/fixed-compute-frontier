# T82 independent audit: likelihood-state hybrid

Date: 2026-08-02  
Scope: theorem, novelty, family-separation, resource, and experiment-admission audit; no code or experiment run

Verdict: **CLOSE ON PAPER AS A MAIN RESEARCH CANDIDATE; RETAIN THE THEOREMS AND USE T82 AS A STRONG CONTROL/COMPONENT IN A FUTURE OPEN-HYPOTHESIS METHOD**

T82.1--T82.3 are mathematically valid after modest qualification. The total-
variation bound in T82.2 is valid but unnecessarily loose. These corrections
do not rescue the research claim. With the candidate mechanisms supplied, T82
is a finite Bayesian model-selection/filtering layer with learned measurement
and transition models plus an amortized or searched experiment policy. That is
an important engineering prior, but it is not a new route to the missing model
capability. The proposal deliberately moves hypothesis formation, variable
formation, candidate coverage, and semantic mechanism invention outside the
method.

This exact architectural pattern is already occupied by differentiable Bayes
and particle filters, including learned motion/measurement models and filters
embedded in decision policies. Current work also directly occupies structured
learned filtering, predictive belief representations, belief-deviation control,
and amortized/online Bayesian experimental design. A three-family synthetic
comparison could show that a known algorithmic prior helps a small recurrent
learner, but it would not constitute the requested new intelligence method.
The project's own gate correctly says that such a CPU result cannot support a
production-LLM claim. It therefore does not earn implementation as T82's main
lane.

## 1. Decision summary

| Question | Audit result |
|---|---|
| Adaptive-policy cancellation | Correct for a common non-anticipating policy kernel on a declared sufficient history; revise the measure/support wording |
| Cumulative log-ratio error | Correct and tight for arbitrary signed local ratio errors |
| `TV <= tanh(E)` | Correct but loose; `TV <= tanh(E/2)` follows from the stated unnormalized-log-weight premise |
| Decision-margin certificate | Correct as a sufficient condition; use the sharper TV term if retained |
| Factorized noninterference | Correct only for an actually factorized posterior, a valid event-local likelihood, and an isolated update path |
| Learned capability beyond filtering | No; not while all candidate mechanisms are supplied |
| Three-family non-isomorphism | Dynamic transducer versus static identification is defensible; ordered causality versus fault localization is asserted, not proved; all three collapse to the same supplied finite-filter interface |
| Cost isolation | Incomplete: the decisive accumulator-only control and a matched differentiable-filter control are missing |
| Path to the frozen `>=20%` model-process success | Not from this screen; at best it can reproduce a component-level algorithmic-prior gain |
| Execution admission | **No CPU, local GPU, or rented GPU for T82 as written** |

## 2. T82.1: adaptive policy does cancel, under a common policy kernel

The clean statement uses conditional probability kernels. Let

\[
\kappa_t(da_t\mid H_{t-1})
\]

be one common, non-anticipating action kernel in every world, and let the
world-dependent observation kernel be

\[
P_h(do_t\mid H_{t-1},a_t).
\]

On trajectories in the common support, the Radon--Nikodym derivative of two
interactive trajectory laws is

\[
\frac{d\mathbb P_h}{d\mathbb P_{h_0}}(\tau_T)
=\prod_{t=1}^T
\frac{dP_h(\cdot\mid H_{t-1},a_t)}
     {dP_{h_0}(\cdot\mid H_{t-1},a_t)}(o_t),
\]

because the identical `kappa_t` factors cancel. Thus T82.1's algebra and its
central conclusion are correct: adaptive data acquisition does not by itself
require an action-propensity correction when the policy is the same
history-conditioned mechanism in every candidate world.

Required revisions and qualifications:

1. “The policy assigns positive probability to every realized action” is an
   informal discrete-support condition, not the general requirement. For
   continuous actions, individual actions have probability zero. State the
   result with common kernels/densities and absolute continuity of trajectory
   laws on the event being compared.
2. Policy randomness, optimizer state, tool state, clocks, resets, and private
   memory must either be included in the declared conditioning state or induce
   the same conditional action kernel after marginalization. Merely using the
   same source code is insufficient if an undeclared state has a
   world-dependent distribution.
3. Mutual absolute continuity gives finite log ratios but is stronger than
   necessary. One-way domination is enough for a Radon--Nikodym derivative;
   support violations produce extended-real evidence. T82 correctly notes that
   clipping changes the model.
4. The environment must not alter action acceptance, timing, availability, or
   reset behavior in a hidden world-specific channel. T82 already recognizes
   this failure mode.

Corollary T82.1a is also correct only if `q_theta` is the true conditional law
for every candidate on every legal reached history. In the aliased transducer
family, a single `z_{t,i}` must itself be a sufficient representation of the
candidate-conditional hidden-state posterior. A learned deterministic
transition `F_theta` does not receive exactness merely from the outer
log-sum-exp update.

## 3. T82.2: the log-ratio result is right; the TV certificate can be sharper

Let the cumulative log-weight error for candidate `i` be `e_i`. The reference-
hypothesis premise gives

\[
|(e_h-e_{h_0})|\le \sum_t\epsilon_t,
\]

by direct summation. This is correct and tight without assumptions on the sign,
dependence, calibration, or martingale structure of local errors. It exposes
the important failure correctly: exact accumulation removes arithmetic drift,
not persistent semantic bias.

The stated total-variation bound is valid. If `|e_i| <= E`, the approximate
posterior is the exponential tilt

\[
q_i=\frac{p_i e^{e_i}}{\sum_j p_j e^{e_j}}.
\]

T82's interval argument gives the safe bound

\[
\operatorname{TV}(p,q)\le \tanh(E).
\]

It is not the best consequence of the premise. Since the tilt range satisfies

\[
\frac{\max_i e^{e_i}}{\min_i e^{e_i}}\le e^{2E},
\]

the extremum is attained by placing mass on the two endpoint tilts, yielding

\[
\operatorname{TV}(p,q)
\le
\frac{e^{E/2}-e^{-E/2}}{e^{E/2}+e^{-E/2}}
=\tanh(E/2).
\]

Equivalently, the general exponential-tilt bound is `tanh(osc(e)/4)`;
`osc(e) <= 2E`. The manuscript's `tanh(E)` therefore remains true, but is about
twice as loose for small `E`.

For losses in `[0,1]`, the risk of each fixed action changes by at most TV. If
the oracle best action has a unique risk gap `Delta` to every competitor, then

\[
\Delta>2\operatorname{TV}(p,q)
\]

is sufficient to preserve the argmin. The proposed `2 tanh(E)` condition is
therefore correct but conservative; `2 tanh(E/2)` is sufficient under the
stated premise. Neither result guarantees useful lifetime behavior because
`E` can grow linearly and the certificate quickly becomes vacuous.

The experiment would also need to distinguish:

- proper log score versus calibration and discrimination;
- average local likelihood error versus the uniform reached-history error used
  by the theorem;
- errors under the learned adaptive policy versus errors under an offline data
  distribution; and
- finite candidate posterior calibration versus decision calibration.

An average next-observation loss cannot be cited as evidence that the theorem's
uniform premise holds.

## 4. T82.3: exact noninterference is conditional, not learned intelligence

Suppose, for one declared partition, that

\[
b_t(h_S,h_{-S})=b_t(h_S)b_t(h_{-S})
\]

and that the full likelihood is `L(o|h_S)`. Then

\[
b_{t+1}(h_S,h_{-S})
\propto b_t(h_S)L(o\mid h_S)b_t(h_{-S}),
\]

so the posterior remains factored and the complete complementary distribution,
not merely each marginal, is unchanged. T82.3 is correct.

The theorem does not establish any of the difficult premises:

- which partition is valid;
- whether the observation carries likelihood information about another factor;
- whether the routing decision itself exposes an undeclared observation;
- whether shared neural descriptors or states change outside the routed group;
- whether later consolidation/backbone learning preserves the protected state;
  or
- whether preserving the old belief is desirable after a real dependency
  change.

If `S_t` is selected from the just-observed event, the routing mechanism must be
a deterministic/audited function of already modeled data. If it uses a side
channel, that routing likelihood is part of the evidence. If it simply routes
incorrectly, exact noninterference guarantees retention of the wrong belief.
T82 states most of this limitation correctly. The result should be retained as
a testable memory invariant, not presented as a continual-learning theorem.

## 5. The supplied descriptors collapse the method to differentiable filtering

T82 supplies every candidate mechanism and asks the neural model to predict
candidate-conditional outcomes. The exact layer then performs finite Bayesian
model selection. In the dynamic family, `z_{t,i}` additionally performs a
candidate-conditional filter. This is precisely the standard division of labor
in a differentiable Bayes/particle filter:

1. learn a transition/motion model;
2. learn a measurement/observation likelihood or discriminative weight;
3. execute a structured recursive probability update; and
4. let a learned decision policy consume the belief.

That boundary is directly occupied:

- [Differentiable Particle Filters](https://arxiv.org/abs/1805.11122) use
  learnable motion and measurement models inside a recursive distributional
  filter, explicitly motivate the filter as an algorithmic prior, compare to
  LSTMs, and report large error reductions and policy-agnostic generalization.
- [Discriminative Particle Filter RL](https://arxiv.org/abs/2002.09884) embeds
  a differentiable particle filter with a learned discriminative update inside
  a neural policy trained end to end for decisions under partial observation.
- The current revision of [Differentiable Filtering for Learning Hidden Markov
  Models](https://arxiv.org/abs/2511.10571) implements the HMM forward filter as
  a structured neural network, learns it by next-observation prediction, and
  compares against Transformers. The T82 manuscript uses the older “Belief
  Net” title; the April 2026 v2 title should be cited.
- [Predictive Coding Enhances Meta-RL](https://arxiv.org/abs/2510.22039)
  directly studies Bayes-like belief representation, active information
  seeking, and generalization under partial observability. It reports cases
  where predictive modules reach optimal information-seeking policies and
  ordinary meta-RL does not.
- [T3](https://openreview.net/forum?id=r8hzDA3pUY) is not the same operator—it
  controls belief-deviating RL trajectories—but it already establishes belief
  drift and repetitive/uninformative action as a current LLM-agent target and
  reports substantial gains from controlling it.
- [Deep Adaptive Design](https://proceedings.mlr.press/v139/foster21a.html),
  [Step-DAD](https://proceedings.mlr.press/v267/hedman25a.html), and
  [constrained Bayesian experimental design via online planning](https://arxiv.org/abs/2605.26990)
  occupy amortized, test-time-refined, and online-lookahead action selection for
  sequential experimental design.
- The 2026 [Bayesian Linguistic Forecaster](https://openreview.net/forum?id=Iw4jKx6hIx)
  demonstrates a sequential numeric/linguistic belief-state agent in a current
  model application, although its update is not T82's exact finite filter.

The closest conceptual precedent is not T3 but differentiable particle
filtering. T82's description—“learn local evidence, execute belief update
exactly”—is almost a definition of that algorithmic-prior family. Applying the
same prior to multiple supplied model classes can be useful, but the
architecture is not an unoccupied finding.

The descriptor choices make the issue sharper:

- If a descriptor is executable enough to return its outcome distribution,
  `G_theta` is learning an emulator of a supplied simulator; the semantic
  mechanism has already been handed to the model.
- If descriptors are opaque symbolic programs and execution is forbidden,
  learning their outcome semantics may be nontrivial, but candidate invention
  and coverage remain supplied. The descriptor grammar, interpreter, and
  training distribution become the real method and must be identified as such.
- If “behaviorally queryable” candidates receive extra counterfactual calls,
  those calls are privileged interactions and must be charged for every arm.
- A fixed `K` makes the method fail exactly where open-world intelligence is
  most needed: the true mechanism is absent, split across candidates, or cannot
  be expressed in the supplied grammar.

T82 explicitly excludes candidate birth and T77's coverage problem. That is
honest, but it also removes the step that could make the proposal a new
intelligence architecture rather than a structured inference component.

## 6. The three families do not prove the claimed kind of transfer

The aliased controlled transducer is genuinely different from a static
conditionally independent identification problem: there exist histories with
the same action/observation multiset but different order and different future
predictive laws. Noncommutation/order sensitivity is an invariant that rules
out a cost-preserving map to an exchangeable static family.

The separation between ordered guarded causality and sparse fault localization
is not yet proved. At finite size, both are static finite-hypothesis
identification problems with a table

\[
L_{h,a,o}=P(o\mid h,a).
\]

After supplying all candidates and legal tests, both are consumed by the same
generic Bayesian update. “Ordered” versus “graph-valued” is not itself a proof
that no cost-preserving bijection exists. The preregistration would need an
explicit invariant—for example, a laminar/chain query-incidence system for the
ordered family and a generated non-laminar overlapping incidence witness for
the fault family—plus matched cardinalities and costs.

More importantly, environment non-isomorphism does not imply learned update
transfer. The exact update is intentionally identical across every finite
candidate problem. A model can recognize the family from the descriptor
syntax, run three separate likelihood emulators, and share only the hard-coded
sum-and-normalize operation. Holding out a new binding, size, topology, or
cross-family conjunction tests descriptor interpretation and local likelihood
generalization, not discovery of a new inference algebra.

Before any future use of this suite, freeze:

1. formal legal-history/action/observation spaces and a cost measure;
2. a constructive invariant or explicit no-bijection witness for each family
   pair;
3. whether family tags, grammar productions, dimensionality, or descriptor
   lengths leak the solver family;
4. the exact compound-training split—“diverse compounds of two families” is
   currently ambiguous; and
5. a control that receives a canonical likelihood table, separating family
   recognition from accumulation.

The suite can still be useful as a future benchmark. It does not make T82 a
novel method.

## 7. Same-information controls and cost ledger are not yet sufficient

The decisive causal comparison is absent from the ladder as written. To test
the value of the exact accumulator, train or freeze one identical `G_theta` and
`F_theta`, expose the same complete vector of local log likelihoods to both
arms, and compare:

- fixed `s <- normalize(s + ell)`; versus
- a parameter-matched learned accumulator given `(s, ell)` with no transcript
  advantage.

Then separately compare end-to-end training of those two architectures. Without
the frozen-head swap, “same local-likelihood auxiliary loss” does not establish
same likelihood quality; the exact arm can win because its backbone learned a
better `G`, not because accumulation is exact.

The strongest learned control must also include a faithful differentiable
Bayes/particle-filter implementation with the same descriptors, particle/slot
count, dynamics, resampling/no-resampling choice, and policy. A generic GRU,
Transformer, predictive-coding agent, and direct posterior regressor do not
replace this adjacent control.

Cost matching needs a concrete equation rather than a list. T82 performs `K`
candidate-conditioned likelihood/state evaluations per event. One batched call
does not remove the `O(K)` arithmetic, activation traffic, descriptor reads, or
state bytes. Conversely, a generic recurrent baseline may use one compact
state transition rather than `K` slots. Parameter equality alone is not a
resource match. The comparison must freeze one primary boundary such as total
training plus lifetime inference FLOPs and charged interactions, then constrain:

- active parameters, FLOPs, wall time, peak and persistent bytes;
- descriptor storage and execution/query cost;
- `K` and candidate-coverage failure;
- local-likelihood supervision bytes and oracle-generation work;
- BPTT/truncation and action-policy optimization;
- one-step VOI's `K * |A| * |O|` enumeration or rollout cost; and
- reload, versioning, routing, change detection, and consolidation.

The larger ordinary model in the ladder is useful, but it must receive the
same *complete cost*, not merely the same serving budget after T82's descriptor
or likelihood-label cost has been excluded.

## 8. The `>=20%` gate is strong, but a pass would still be component evidence

The frozen magnitude boundary is appropriate: simultaneous confidence bounds
must establish at least 20% lower complete primary cost, while a 10--20% result
is provisional and a single-digit result closes the lane. Requiring `2x` fewer
interactions on active acquisition is stronger still. The preregistration would
need to specify whether both are mandatory in every family or whether the
primary weighted statistic can mask one family, and it must freeze absolute
competence floors, headroom, family weights, multiplicity correction, and the
unit conversion between regret and interaction cost.

However, passing this gate on a small synthetic suite would prove only:

> Given the true finite candidate set and matched local-likelihood supervision,
> a hard-coded Bayesian update prior lets a small learned system use evidence
> more reliably or cheaply than the selected learned controls.

That is a substantial component result if it is genuinely `>=20%`. It is not a
new model-level intelligence result, because the model still cannot form the
hypotheses whose probabilities it updates. Prior differentiable-filter work
already establishes that structured recursive inference can outperform generic
recurrent baselines by large margins. Re-running that proposition on three
synthetic families does not earn a new architecture lane.

The CPU screen also cannot establish the eventual production claim by T82's own
terms. It could only admit a later model experiment. Since the method's novel
claim has already disappeared at the literature/abstraction boundary, a cheap
run is still not warranted merely because it is cheap.

## 9. Closure boundary and retained value

Close these claims:

- T82 as a new architecture for real intelligence;
- exact likelihood accumulation over supplied candidates as a novel method;
- three-family synthetic success as sufficient evidence of hypothesis or
  variable formation; and
- a CPU result as satisfaction of the project's model-process success gate.

Retain:

- T82.1's common-policy likelihood-ratio identity;
- T82.2's lifetime semantic-error warning and the sharper TV certificate;
- T82.3 as a protected-factor test invariant;
- the cost ledger and same-information discipline;
- the three-family suite as a benchmark/control suite; and
- exact finite Bayesian filtering as the strongest structured baseline for any
  future open-world method.

A future direction may reuse T82 only if it attacks the omitted step. A genuine
reopening case would require the model to create, split, merge, reject, and
compress candidate mechanisms from raw partial observations under a bounded
description budget; predict outcomes for unseen mechanisms; retain an explicit
unknown mass when coverage fails; and then use T82 merely as the posterior
backend. The decisive ablation would remove learned candidate formation while
keeping the same filter. Only the full raw-observation-to-new-hypothesis-to-
decision process could claim the requested model-level gain.

**Final admission decision: CLOSE ON PAPER. Do not implement the T82 CPU
screen, do not use the local GPU, and do not rent a GPU for this lane.**
