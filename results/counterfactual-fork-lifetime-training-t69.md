# T69 counterfactual-fork lifetime training — teach state by testing many futures

Date: 2026-08-02  
Status: **INDEPENDENTLY REVISED; BROAD MECHANISM CLOSED; FORK-COVERAGE AND PAIRED-CONTRAST RESULTS RETAINED; NO RUN**

## 0. Question

The corrected research target is a model that becomes more capable through
experience, rather than a static prose predictor with a larger context. A
natural training proposal is to stop rewarding one realized future from each
past. Instead, restore the same past state, execute many legal continuations,
and require one compact persistent representation to support all of them.

The hoped-for causal edge is:

```text
one experienced past
    -> many action-conditioned futures
    -> one state must preserve every distinction that can affect a future
    -> the state becomes an executable interventional world model
```

This note asks whether the fork itself supplies a learning advantage that a
matched action-conditioned world model does not have.

## 1. Controlled predictive equivalence

Let `H` be a history, `E` a legal continuation experiment (an open-loop action
sequence or a closed-loop policy), and `Y_E` a declared future outcome. Define

\[
h\sim_{\mathcal E}h'
\quad\Longleftrightarrow\quad
P(Y_E\in B\mid h,\operatorname{do}(E))
=P(Y_E\in B\mid h',\operatorname{do}(E))
\]

for every \(E\in\mathcal E\) and measurable outcome set `B`. The quotient is a
controlled predictive/causal-state object: histories are different exactly
when some legal future intervention distinguishes them.

A candidate learner contains a bounded state `m=phi(h)` and decoder

\[
q_\theta(Y\mid m,E).
\]

Training samples continuations from `mu(E|h)` and minimizes a strictly proper
scoring rule:

\[
R(\phi,q)=
\mathbb E_{h,E,Y}\left[S(q(\cdot\mid\phi(h),E),Y)\right].
\]

### T69.1 — exact sufficiency at the population optimum

Assume:

1. optimization is over both `phi` and `q`, with no regularizer or auxiliary
   objective trading predictive risk against compression;
2. the representation class has enough states to realize the entire declared
   predictive quotient, and one decoder can jointly realize all of its
   conditional future laws;
3. the learned pair attains the full-history Bayes risk of the integrable,
   strictly proper score—zero excess risk relative to conditioning on `(H,E)`;
4. the compared histories are discrete support points with `P(H=h)>0`, and
   every declared experiment has `mu(E|h)>0` at each compared history; and
5. the same decoder receives only `phi(h)` and `E`.

If `phi(h)=phi(h')`, then \(h\sim_{\mathcal E}h'\).

### Proof

Proper-score regret is an expected nonnegative strict divergence between the
true `P(Y|H,E)` and `q(Y|phi(H),E)`. Equality with the full-history Bayes risk
makes this divergence zero at every positive-mass `(h,E)`. If two histories
share a state, the decoder receives the same `(phi(h),E)` at both and therefore
emits one distribution. It can equal both true laws only if those laws are
equal for every supported experiment. QED.

Without assumption 3 the result is false. An insufficient bottleneck may merge
two histories and the proper-score optimum simply predicts their conditional
mixture. For continuous experiment spaces the conclusion is only
`mu`-almost-everywhere unless extra continuity or domination assumptions extend
it to every legal experiment.

Conversely, mapping each history to its equivalence class is sufficient. If the
quotient is finite, a representation constrained to the minimum possible
number of discrete states recovers that quotient up to relabeling. A generic
continuous bottleneck or finite neural width does not guarantee minimality,
identifiability, interpretability, or successful optimization.

This theorem explains why diverse interventions are a good representation
target. It is not a new theorem about an architecture: predictive-state,
causal-state, belief-state, and action-conditioned world-model objectives aim
at the same sufficient statistic.

## 2. How many forks can expose a bad merge?

Suppose a finite candidate set contains `M` incorrectly merged pairs of
histories. For every such pair, let a fresh random continuation from `mu`
distinguish their outcome laws with probability at least `rho>0`. Give each
pair `K` independent common continuation tests.

### T69.2 — fork coverage bound

The probability that any bad merge receives no truly distinguishing sampled
experiment is at most

\[
M(1-\rho)^K\le M e^{-\rho K}.
\]

For `0<rho<1`, the exact integer condition is

\[
K\ge
\left\lceil\frac{\ln(M/\delta)}{-\ln(1-\rho)}\right\rceil.
\]

The simpler condition

\[
K\ge\frac{\ln(M/\delta)}{\rho}
\]

is conservative. If `rho=1`, one experiment per pair suffices.

### Boundary

This is only a coverage calculation: it says that a truly distinguishing
experiment was sampled, not that one realized stochastic outcome reveals the
difference. Detection additionally needs a declared separation, repeated
rollouts per history/experiment, a calibrated test, and family-wise error and
power control. The tests must be legal from both histories. The bad-pair set,
experiment policy, and `rho` must be fixed independently of the evaluation
sample or covered by a uniform adaptive guarantee. If `rho` is exponentially
small, branch enumeration is exponentially expensive. An intelligent active
experiment selector may increase `rho`, but then its acquisition cost and
comparison with exact active-learning controls are required.

## 3. The fatal matched-data control

Let the empirical loss be separable:

\[
\widehat R=\sum_i
S(q(\cdot\mid\phi(h_i),E_i),Y_i).
\]

A forked dataset is the multiset

\[
D=\{(h_i,E_{ij},Y_{ij}):i=1\ldots n,j=1\ldots K\}.
\]

### T69.3 — grouping invariance

For a fixed dataset, model class, and separable loss, presenting
the `K` sibling samples as a named fork or as ordinary shuffled
action-conditioned examples changes no empirical objective and has the same
set of global minimizers.

Let `G` be sibling membership. Grouping adds no information beyond `D` only
when every tuple retains a stable exact history/snapshot identifier from which
the sibling groups are deterministically recoverable, so `H(G|D)=0`. If
grouping reveals a shared seed or snapshot relation absent from the tuples,
the metadata is additional information and must be supplied to the control.

Different minibatch orderings can change finite-step SGD trajectories. That is
an optimization heuristic, not an information or population-objective edge;
the matched control must receive the same batching schedule when this matters.

### Consequence

Any advantage must come from something beyond the broad proposal:

- a nonseparable cross-branch loss;
- correlated branch generation unavailable in ordinary data;
- a different architecture or optimizer;
- active selection of continuations; or
- additional simulator/reset work.

Each of these is the actual candidate and must beat a control given the same
resource. A universal action-conditioned world model trained on exactly `D`
already receives all of the claimed interventional information.

Even a contrastive cross-branch loss is not protected: a matched control can
use the same sibling identities and loss. The word “counterfactual” does not
create an information advantage.

## 4. The one genuine statistical edge: common exogenous randomness

A simulator may expose a stronger oracle. Let a structural transition be

\[
Y(a)=F(h,a,U),
\]

where `U` is exogenous randomness. Restore one past and one RNG state, then run
two actions `a,b` with the same `U`. For iid seeds `U_i`, estimate the average
action effect by

\[
\widehat\Delta_{paired}
=\frac1n\sum_i[F(h,a,U_i)-F(h,b,U_i)].
\]

It is unbiased for `E[Y(a)-Y(b)]`, with

\[
\operatorname{Var}(\widehat\Delta_{paired})
=\frac{\sigma_a^2+\sigma_b^2-2\operatorname{Cov}(Y(a),Y(b))}{n}.
\]

Independent, unpaired rollouts have variance

\[
\operatorname{Var}(\widehat\Delta_{unpaired})
=\frac{\sigma_a^2+\sigma_b^2}{n}.
\]

Thus positive paired covariance reduces the samples needed for the same
action-contrast precision. Zero covariance gives no variance gain and negative
covariance makes this coupling worse. The paired and unpaired displays both
use two simulator calls per index—`n` under each action. In the additive world

\[
Y(a)=\tau_a+U,
\]

the paired difference is exactly `tau_a-tau_b` and its variance is zero, while
unpaired variance is `2 Var(U)/n`.

This can be a large contrast-estimation advantage, not a sub-percent
optimization. It does not reduce the marginal sampling variance required to
learn each complete future law and therefore does not by itself establish
controlled predictive sufficiency. It also has a hard price: access to paired
potential outcomes under aligned exogenous variables. Restoring one PRNG state
is not automatically a structural counterfactual coupling because different
actions may consume random draws differently. Ordinary physical experience
cannot rewind the world while holding all nuisance causes fixed. A learned
simulator cannot manufacture information it has not identified and must be
validated on held-out real interaction. The correct control is a conventional
world model trained on the same paired rollouts, seed/sibling identifiers, and
paired-effect targets; it can use the same variance reduction.

## 5. Collision audit

- [Recurrent Predictive State Policy Networks](https://proceedings.mlr.press/v80/hefny18a.html)
  explicitly represent future-observation distributions conditioned on future
  actions and train a recurrent state with prediction error.
- [Action-conditional self-predictive RL](https://proceedings.mlr.press/v258/khetarpal25a.html)
  directly analyzes latent prediction conditioned on future actions and its
  low-rank dynamics interpretation.
- [Action-Sufficient State Representations](https://proceedings.mlr.press/v162/huang22f.html)
  learn compact partially observed state for downstream action and imagined
  outcomes.
- [NextLat](https://arxiv.org/abs/2511.05963) adds a next-latent predictive
  objective and reports more compact belief-like world models without changing
  inference.
- [ORBIT](https://arxiv.org/abs/2602.04089) trains cross-episode online learning
  and reports Qwen3-14B matching GPT-5.2 on unseen interactive environments.
- [State commitment learning](https://arxiv.org/abs/2606.05201) already trains a
  counterfactual keep/erase branch under the same prefix so retained state must
  remain sufficient after scratch reasoning is removed.

These do not prove that current systems solve lifelong intelligence. They do
show that forked futures, predictive state, compact latent belief, cross-episode
meta-learning, and persistent-state counterfactual tests are occupied blocks.

## 6. Cost ledger

For `K` branches of horizon `L` from each past:

\[
C_{fork}=C_{snapshot}+K C_{restore}
+KL(C_{env}+C_{model})+C_{update}.
\]

The proposal may keep deployment parameters and per-step inference unchanged,
but multiplies training environment transitions and model tokens roughly by
`K`. Snapshot storage, simulator licensing, RNG capture, rejected branches,
and active-selection calls are charged. When the simulator is itself learned,
its training cost and bias are also charged.

## 7. Decision

The broad proposal does not survive the strongest ordinary control. It
correctly targets controlled predictive sufficiency, but a matched
action-conditioned world model sees the same information and optimizes the same
separable objective. The fork label is not a new intelligence mechanism.

Retain two useful results:

1. continuation coverage must include the interventions that separate aliased
   histories, with experiment-coverage miss probability controlled by T69.2;
   statistical detection is a separate cost; and
2. exact paired rollouts with common exogenous randomness can reduce causal
   action-contrast variance dramatically under positive covariance, but this
   is a simulator-oracle and training-data result shared by the control.

No local or rented run is admitted. A benchmark showing the forked candidate
beats a control denied sibling identities or paired data would only measure an
unfair information difference.

## 8. What the closure changes

The next direction cannot merely give a model more futures, more memory, or a
better arrangement of the same transition tuples. It must change at least one
of the following in a way a matched learner cannot absorb:

1. what evidence the agent can acquire per real interaction;
2. what class of update can be executed and safely retained;
3. what computation can be reused across genuinely new tasks; or
4. the asymptotic resource needed to discover and apply a reusable abstraction.

That is the remaining intelligence frontier.
