# T82 likelihood-state hybrid — learn local evidence, execute belief update exactly

Date: 2026-08-02  
Status: **INDEPENDENTLY AUDITED; CLOSED ON PAPER AS MAIN CANDIDATE; RETAINED AS FILTER CONTROL; NO RUN**

## 0. The exact gap left by T75

T75 established a qualifying component result. At at least 95% average and
worst-world exact identification, an active belief-guided reference used `168`
instead of `1,280` probes, an `86.875%` reduction. It did not establish a
better model because the hypothesis space, outcome likelihoods, posterior
update, and query rule were supplied.

T82 targets the missing bridge:

> Can one trained model infer local evidence from raw events while a bounded
> architectural state performs the universal part of belief accumulation,
> enabling active acquisition, revision, and retention to transfer across
> unseen world structures?

The proposed change is a hybrid recurrent layer inside the model, not an
external teacher. A neural backbone predicts local action-conditioned outcome
laws and updates hypothesis-specific dynamical state. A parameter-free
likelihood-state operator accumulates the resulting evidence exactly. A learned
policy and decoder read that state. The first deployment stage freezes the slow
backbone weights; only bounded recurrent state changes.

This is not claimed as a new Bayesian identity. The research claim is empirical
and model-level: separating semantic likelihood estimation from exact evidence
composition may remove belief drift that an equally sized generic recurrent or
Transformer learner fails to remove during finite training.

## 1. Architecture

Maintain at most `K` active hypothesis slots. Slot `i` contains:

- a hypothesis descriptor `c_i`;
- a hypothesis-conditional dynamical state `z_{t,i}`;
- a normalized log belief `s_{t,i}=log b_t(i)`; and
- provenance, version, and protected dependency metadata.

Before observing `o_t`, the shared neural model computes a normalized predictive
law from the prior slot state,

\[
q_{t,i}(\cdot)
=G_\theta(z_{t,i},c_i,a_t,x_t^-),
\qquad
\ell_{t,i}=\log q_{t,i}(o_t),
\]

where `x_t^-` contains only information available before the outcome. After the
outcome has been scored and belief updated, the hypothesis-conditional state
transition is

\[
z_{t+1,i}=F_\theta(z_{t,i},c_i,a_t,o_t,x_t).
\]

This order is mandatory: a network that sees `o_t` before assigning its
likelihood can trivially leak the answer and does not define a predictive
model. For a large or continuous observation space, `G` must still define and
charge a proper density, discrete probability, or explicitly audited ratio
estimator; an arbitrary compatibility score is not a likelihood.

The architectural belief update is

\[
\tilde s_{t+1,i}=s_{t,i}+\ell_{t,i},
\qquad
s_{t+1,i}=\tilde s_{t+1,i}
-\operatorname{LSE}_j\tilde s_{t+1,j}.
\]

The decoder receives posterior-weighted slot features, the uncertainty summary,
and the current query. The action head receives the same bounded state and is
trained on complete-lifetime regret and action cost. It may learn a multi-step
policy; a one-step expected-value-of-information policy is an explicit
algorithmic control, not silently built into the candidate.

The state is part of the model's recurrent architecture. It must survive
evidence-transcript deletion and fresh-process reconstruction. Hypothesis
descriptors and conditional states count toward the served-state budget.

## 2. Why the exact updater remains valid under adaptive action

Let the hidden stationary hypothesis be `h`, and let the agent choose

\[
a_t\sim\pi_\theta(\cdot\mid H_{t-1})
\]

from its observed history. The policy is the same measurable rule in every
candidate world; it cannot condition on the unobserved true `h` except through
history. The probability of an observed interactive trajectory is

\[
P_h(\tau_T)=
\prod_{t=1}^{T}
\pi_\theta(a_t\mid H_{t-1})
p_h(o_t\mid H_{t-1},a_t).
\]

### Proposition T82.1 — adaptive-policy cancellation

Assume the two outcome laws are mutually absolutely continuous on the realized
trajectory and the policy assigns positive probability to every realized
action. For hypotheses `h` and `h_0`,

\[
\log\frac{P_h(\tau_T)}{P_{h_0}(\tau_T)}
=\sum_{t=1}^{T}
\log\frac{p_h(o_t\mid H_{t-1},a_t)}
          {p_{h_0}(o_t\mid H_{t-1},a_t)}.
\]

**Proof.** The action-policy factor at each time is identical in the numerator
and denominator for the realized history, so it cancels. The remaining
conditional outcome factors multiply; taking logs gives the sum. QED.

Zero-probability outcomes instead give extended-real evidence and require an
explicit support convention; numerical clipping changes the model and must be
charged as approximation. Therefore active action choice does not require an extra likelihood correction
when the policy has no hidden world-specific side channel. If world identity
changes the action interface, timing, reset mechanism, tool errors, or policy
state outside the declared history, cancellation can fail and the omitted
channel must be modeled.

### Corollary T82.1a — exact posterior state

If the initial `s_0` is the log prior and `q_\theta=p` for every slot and legal
history, the recurrent log-sum-exp update equals the exact Bayesian posterior
after every adaptive trajectory.

For conditionally independent evidence whose likelihood depends only on the
action and observation, the accumulated state is also invariant to evidence
order. In a dynamical world, order generally changes the conditional likelihood
and must not be erased; the hypothesis-specific states `z_{t,i}` carry that
dependence.

## 3. What approximate neural likelihoods can and cannot preserve

The exact operator cannot repair an incorrect semantic likelihood model.
Suppose that for a reference hypothesis `h_0`, every realized local log-ratio
error satisfies

\[
\left|
\log\frac{q_t(o_t\mid h)}{q_t(o_t\mid h_0)}
-
\log\frac{p_t(o_t\mid h)}{p_t(o_t\mid h_0)}
\right|\le\epsilon_t
\]

for all active `h`. Then direct summation gives:

### Proposition T82.2 — cumulative evidence error

\[
\left|
\log\frac{b_T^{q}(h)}{b_T^{q}(h_0)}
-
\log\frac{b_T^{p}(h)}{b_T^{p}(h_0)}
\right|
\le\sum_{t=1}^{T}\epsilon_t.
\]

This bound is tight at the level of arbitrary signed local errors. Exact
accumulation removes aggregation error, not model error; persistent bias can
grow linearly with lifetime length.

If instead every unnormalized posterior log weight differs from the oracle by
at most `E`, the sharp uniform consequence is

\[
\operatorname{TV}(b_T^q,b_T^p)\le\tanh(E/2).
\]

To see the constant, write the approximate unnormalized weights as the oracle
weights times `exp(e_i)`, with `e_i in [-E,E]`. The extremal distribution uses
only the two endpoint errors. Optimizing the oracle mass placed on those
endpoints gives

\[
\sup\operatorname{TV}(b^q,b^p)
=\frac{e^E-1}{e^E+1}=\tanh(E/2).
\]

The simpler normalized density-ratio range argument gives the valid but looser
`tanh(E)` bound; it is not used for the frozen certificate.

For losses in `[0,1]`, every fixed action's posterior risk changes by at most
that total variation. If the oracle Bayes action has a risk margin greater than
`2 tanh(E/2)` over the runner-up, the approximate state selects the same action.
This is a conditional robustness certificate, not a claim that neural
likelihoods satisfy the premise.

The experiment must therefore score local likelihood calibration separately
from posterior aggregation. A candidate win is inadmissible if it simply
receives a better encoder, more outcome supervision, or oracle likelihoods than
the recurrent control.

## 4. Factorized revision without global overwriting

For a world represented by factors `j=1,...,m`, maintain separate slot groups
`b_{t,j}` only across a declared posterior factorization. Let `S_t` be a group
of factors whose joint state may be represented together. Assume immediately
before the event that

\[
b_t(h)=b_t(h_{S_t})b_t(h_{-S_t})
\]

and that the event likelihood depends only on `h_{S_t}`. Update the joint group
`S_t` and leave the complement untouched. With frozen slow weights and
immutable versioned states:

### Proposition T82.3 — conditional posterior noninterference

For every `j` not in `S_t`, `b_{t+1,j}=b_{t,j}` exactly.

**Proof.** Multiplying the factored prior by a likelihood of `h_{S_t}` changes
only the first factor; the normalizer also sums only over that factor. The
complementary marginal remains `b_t(h_{-S_t})`. QED.

This is an architectural preservation property, not a learned responsibility
theorem. If `h_{S_t}` and `h_{-S_t}` are posterior-correlated, evidence about
one generally changes belief about the other and forced locality is wrong. The
neural model must identify both `S_t` and a valid separation; an incorrect dependency route
can preserve the wrong object and corrupt the right one. Shared hypothesis
descriptors, a changing observation encoder, consolidation, or later backbone
training can also couple the factors. Those paths require versioned snapshots
and end-to-end protected-query tests.

To represent a declared change hazard `rho_j`, a factor may first apply

\[
b^-_{t,j}=(1-\rho_j)b_{t,j}+\rho_j b_{0,j}
\]

before its likelihood update. This is exact only for the corresponding reset
mixture. Unknown or nonstationary hazards require a matched Bayesian
change-point, adaptive-filter, and generic recurrent control. A convenient
forget gate is not evidence of correct self-revision.

Hypothesis birth is deliberately outside the first claim. An `unknown` slot may
signal that all current models predict poorly, but proposing a useful new
hypothesis remains the T77 candidate-coverage problem. No result on a supplied
hypothesis set is relabeled as abstraction invention.

Consequently, the first screen must supply every learned control with the same
complete candidate set, expressed as executable or behaviorally queryable
descriptors under fresh random bindings. Candidate enumeration, code bytes,
execution, and any grammar are charged. The descriptor may specify a candidate
mechanism but never which candidate is true. This deliberately isolates belief
formation and active use; it also means a pass cannot claim raw hypothesis or
variable invention. If the class cannot fit under the frozen `K` cap, coverage
failure is a method failure rather than permission to reveal the answer.

## 5. Why this can be better than the existing model

The candidate does not contain a stronger semantic reasoner. It removes one
operation from the neural approximation burden:

- the backbone learns `what outcome each hypothesis predicts locally`;
- the likelihood layer computes `how independent evidence changes their
  relative support`;
- the policy learns `which evidence is worth buying`; and
- the decoder learns `how to act from the resulting uncertainty`.

The layer can improve the resulting model only if finite-trained generic state
is losing, double-counting, misordering, or failing to expose evidence that its
local outcome model already predicts. The causal test is same local-likelihood
quality plus different accumulation. If a generic recurrent state matches the
candidate, or if local likelihood error dominates, the architectural claim is
closed.

The intended capability gains are concrete:

1. fewer repeated or uninformative queries at matched decision accuracy;
2. better calibration and recovery after contradictory evidence;
3. retention after the raw transcript is removed;
4. transfer of the update algebra beyond trained lifetime length; and
5. localized revision of one factor without changing protected factors.

It does not by itself improve hypothesis invention, semantic grounding,
long-horizon planning, or the correctness of the learned world model.

## 6. Three development generator/action geometries

The first learned falsifier must use one shared serialization and one shared
backbone over at least three different generator/action geometries:

1. **Ordered guarded causality.** T75's hidden threshold and causal orientation;
   actions query ordered selectors and receive noisy interventional outcomes.
2. **Sparse fault localization.** A hidden subset of components in a freshly
   generated dependency DAG is faulty; legal tests cover overlapping ancestor
   sets and return noisy pass/fail outcomes. The sufficient query structure is
   set- and graph-valued rather than ordered binary search.
3. **Aliased controlled transducer.** The hidden mechanism is a finite
   action-conditioned state machine with aliased observations. Action order
   changes hidden state, so hypothesis-conditional dynamic states are necessary
   and evidence is not exchangeable.

Training sees diverse compounds of two families' update demands. The decisive
split holds out bindings, sizes, horizons, graph topologies, machine programs,
and at least one cross-family conjunction. A family counts as distinct only
at the application/generator level: the ordered, overlapping-set, and dynamic
path-dependent action structures have different costs and outcome laws. All
three finite families intentionally reduce to the same controlled Bayesian
filter abstraction. No distinct update algebra or formal non-isomorphism claim
is made.

Every family must have an exact small-instance posterior/filter, a declared
information lower bound or audited oracle frontier, and a blind generator. The
latent world ID, factor names, graph roles, and state-machine IDs never enter
the learned interface as truth labels. Candidate descriptors are supplied to
all controls as declared above, under counterfactual permutation, and contain
no marker correlated with which candidate generated the lifetime.

For every conjunction, the manifest must freeze whether the candidate set is
the full Cartesian product or a proved factorization. The full product charges
its complete `K` slots. A factorized implementation is allowed only when T82.3's
posterior-separation condition holds; shared evidence that couples factors
invalidates locality and forces a joint group or an explicitly approximate
filter.

## 7. Same-information model ladder

All learned cells receive identical raw events, hypothesis descriptors, local
outcome targets, action opportunities, state-byte caps, training lifetimes,
tuning budgets, and slow-backbone initialization.

Required controls are:

1. full transcript Transformer and byte-matched retrieval;
2. generic GRU/SSM persistent state trained on the same complete-lifetime loss;
3. a permutation-equivariant candidate-slot GRU/SSM that receives the identical
   realized candidate-indexed likelihood vector, conditional-state tensors,
   masks, and legal actions as T82;
4. direct neural posterior regression over those same replayed values;
5. T82 exact accumulation over the replayed values with a learned policy;
6. T82 state with a one-step exact value-of-information policy;
7. oracle-likelihood Bayes/filter and Bayes-optimal or audited search controls on
   tractable sizes;
8. predictive-coding meta-RL, a faithful T3-style belief-deviation control, and
   an explicit linguistic/numeric belief-state agent; and
9. a larger ordinary model allowed the same total serving and interaction cost.

The exact-policy and oracle-likelihood rows are ceilings, not competitors that
the learned candidate must beat. The candidate must beat the strongest
resource-matched **learned** control while approaching the algorithmic frontier.
If the same local-likelihood head plus a generic state ties T82, exact
accumulation has no demonstrated model value.

The causal screen first trains one shared `F/G`, freezes it, and materializes a
blind replay stream. Exact addition, learned accumulation, reset accumulation,
shuffled accumulation, and direct posterior regression consume bit-identical
`F/G` outputs. The primary fixed-trajectory phase uses common exogenous actions
and outcomes. Only after that phase is scored does the active phase cross the
same frozen policy heads between exact and learned accumulators. This separates
accumulation from likelihood quality, action-policy quality, and occupancy
shift.

The decisive table is a `2x2`: shared learned versus oracle likelihoods, each
with exact versus learned accumulation. If exact accumulation helps only with
oracle likelihoods, learned semantic error dominates and the model claim fails.

## 8. Frozen success and kill rules before code

The eventual primary statistic is cumulative lifetime decision regret plus
charged interaction cost, measured from raw observation through final action.
Before the first decisive seed, freeze its units, action cost, oracle, horizon,
family weights, and confidence method.

A learned result counts only if simultaneous uncertainty bounds establish:

- at least 20% lower primary cost than the strongest matched learned control,
  with the relative and absolute change and available headroom reported;
- at least 2x fewer interactions to the same competence on the active-
  acquisition phase, or a separately frozen 20% gain on another complete major
  phase;
- no family below its absolute competence floor and no protected old-family
  loss above one absolute point;
- persistence after transcript deletion and fresh-process state reload;
- a causal loss of the gain when likelihood accumulation is shuffled, reset,
  or replaced by direct posterior regression; and
- all model, state, training, action, latency, generated-token, and energy costs
  within their frozen comparison ledger.

`10%` to less than `20%` is provisional. Every single-digit result closes the
architecture as the main finding. Failure to outperform the generic recurrent
state closes the method even if it beats a static or fixed-query baseline.

The preregistration must additionally freeze the precise regret and interaction
units, their scalarization or lexicographic rule, generator parameters, family
weights, competence floors, `K`, precision, number of independent seeds,
simultaneous interval construction, tuning budget, and stopping rule. No CPU
implementation is admitted from this paper alone.

The first executable step, if independently admitted, is CPU-sized and uses a
small from-scratch backbone. It may test arithmetic, calibration, and learned
transfer; it cannot support a production-LLM claim. A GPU run is admitted only
after the local screen clears the complete 20% gate across the three-family
suite and an independent code/result audit finds no reversing defect.

## 9. Complete cost boundary

For `K` slots of state width `d_z`, the recurrent state alone costs at least
`K(d_z+1)` stored scalars plus descriptors, provenance, factor routing, and
normalization workspace. Evaluating every slot costs at least `O(K)` shared
model calls or one batched call with `O(K d_z)` activation traffic. Enumerating
actions and outcomes for one-step experimental design can cost
`O(K|A||O|)` before neural encoding.

The ledger must separately report:

- slow model parameters and active FLOPs;
- hypothesis descriptors and proposal-generation cost;
- persistent slot bytes and numerical precision;
- per-event likelihood/transition work;
- action-search rollouts or outcome enumeration;
- transcript, retrieval, cache, and reload bytes;
- environment interactions and resets;
- complete-lifetime BPTT or truncation cost; and
- change detection, rejected hypotheses, and any consolidation work.

A `K`-slot belief state is not free merely because it replaces a long text
transcript. Conversely, if lifetime length `T` grows while `K` and `d_z` remain
fixed, T82 can have bounded served state where full KV or raw retrieval grows
with `T`; that is a measured secondary edge, not the primary intelligence
claim.

For the aliased transducer, exact dynamic filtering requires `z_{t,i}` to
represent the complete hypothesis-conditional distribution over latent machine
state. A point state is generally insufficient. Particle truncation,
resampling, or a learned finite bottleneck makes only the outer slot
normalization exact; every result must label the inner filter approximate and
charge its state and compute.

## 10. Direct current-art boundary

The blocks are substantially occupied:

- [T3](https://openreview.net/forum?id=r8hzDA3pUY) identifies belief deviation
  in active LLM reasoning and reports gains up to 30 points plus lower token
  cost by truncating corrupted training tails.
- [Belief Net](https://arxiv.org/abs/2511.10571) implements an HMM forward
  filter as a structured neural network trained by next-observation prediction.
- [Predictive Coding Enhances Meta-RL](https://arxiv.org/abs/2510.22039)
  reports that predictive objectives improve Bayes-like belief representation,
  active information seeking, and generalization.
- [Bayesian Linguistic Forecaster](https://openreview.net/forum?id=Iw4jKx6hIx)
  maintains a linguistic/numeric sequential belief state and reports a strong
  forecasting result.
- [Step-DAD](https://arxiv.org/abs/2507.14057) and
  [constrained Bayesian experimental design](https://arxiv.org/abs/2605.26990)
  directly occupy amortized and online-refined active experiment selection.
- [Particle Filter Networks](https://proceedings.mlr.press/v87/karkus18a.html)
  already combine learned transition/observation models with an end-to-end
  differentiable particle filter and test unseen-environment generalization.
- [BeliefTrack](https://arxiv.org/abs/2605.30219) directly evaluates explicit
  belief management on rule-discovery and circuit-diagnosis tasks.
- Classical Bayes filters, differentiable filters, particle filters, POMDP
  belief states, predictive-state methods, and Bayesian continual learning
  occupy the underlying mathematics.

Therefore T82 makes no novelty claim for Bayesian update, explicit belief,
active experimental design, or differentiable filtering. The only admissible
research contribution is a substantial, causal, cross-family model result from
the exact same-information architecture comparison. If current work already
contains this complete comparison, or if the independent audit finds the three
families reducible to a supplied generic solver, the lane closes before code.

## 11. Independent-audit decision

1. Are T82.1--T82.3 correct with all adaptive-policy, support, dynamic-state,
   and factorization conditions stated?
2. Is the approximate-posterior total-variation and decision-margin bound
   correct, and does it expose rather than hide the lifetime error problem?
3. Does the architecture add any learned capability beyond a differentiable
   Bayes/particle filter or current belief-state agent?
4. Do the three generator/action geometries test meaningful transfer despite
   sharing one finite controlled-filter solver?
5. Can the same-information controls isolate exact evidence accumulation from
   better likelihood supervision, policy search, state bytes, or extra compute?
6. Is there a defensible path to a 20% major-process gain and eventually a
   production-model result, or should the lane close on paper?

The [independent audit](likelihood-state-hybrid-t82-independent-audit.md)
verifies T82.1--T82.3 after the current conditions and classifies the method as
established differentiable Bayes-filter territory. Supplying every candidate
mechanism removes hypothesis and variable formation, the part that could make
the system materially more intelligent. Differentiable particle filters,
structured learned filters, predictive-coding meta-RL, and Bayesian
experimental design already occupy the remaining learned-likelihood,
recursive-filter, and policy components.

The final verdict is **close on paper as the main candidate**. Even a positive
three-family CPU result would reproduce a known structured-filter inductive
bias and would not establish the requested model-level process improvement.
Retain the theorems, cost discipline, generator suite, and exact filter as
controls for a future open-hypothesis method. Do not implement T82, do not use
the local GPU, and do not rent a GPU for this lane.
