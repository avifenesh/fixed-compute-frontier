# T75 guarded causal composition — held-out combination of active localization and Bayesian revision

Date: 2026-08-02  
Status: **MODEL-FREE CPU STAGE A PASSED; ACTIVE OPERATOR VALIDATED; NO NEURAL OR GPU RUN**

## 0. Claim tested

T73 teaches a deterministic guarded-identification primitive. T74 teaches an
unguarded stochastic causal-update primitive. T75 combines them but is absent
from updater training.

The question is no longer whether one network can route between two known task
families. It is:

> Can the same frozen update policy compose guarded active localization with
> noisy interventional belief update in one new lifetime, without a
> family-specific adapter or training example of their conjunction?

A positive result is reusable learning within declared primitives. It is not
general intelligence, but it is materially stronger than joint multitask
success on T73 and T74 separately.

## 1. Hidden guarded causal world

Fix public `epsilon=0.1`, `m>=3`, and a selector

\[
u\in\{0,\ldots,m-1\}.
\]

A world samples a threshold uniformly and independently of every model choice
and surface binding,

\[
t\in\{1,\ldots,m-1\}
\]

and a low-region orientation `Theta_0` independently and uniformly from
`{X->Y,Y->X}`. The high-region orientation is the opposite:

\[
\Theta(u)=
\begin{cases}
\Theta_0,&u<t,\\
1-\Theta_0,&u\ge t.
\end{cases}
\]

At every selector, `(X,Y)` follows the noisy structural equations from T74 for
the local orientation. Selector, pair, and variable names are freshly bound
after training without encoding threshold or causal role. The passive sampler,
stopping mechanism, action interface, and score channel are world-independent;
passive observation exposes no intervention, score, or change side channel.

The unified typed interface is:

```text
OBSERVE selector=u
ACT selector=u, operation=do(X=0) -> FEEDBACK Y
PREDICT selector=u, operation=do(X=0) -> private probability
ORIENT selector=u -> private orientation and confidence
```

T73 and T74 are serialized through the same generic `OBSERVE/ACT/FEEDBACK/
PREDICT` event envelope during meta-training. Their payload schemas remain
typed and visible to every system. T75's combination of selector and
intervention is never present during training.

## 2. T75.1 — passive histories reveal nothing

For every selector and either local orientation, the passive `(X,Y)` law is the
same symmetric distribution proved in T74. Consequently, for any adaptive
passive selector/stopping policy with no correlated side channel,

\[
I((t,\Theta_0);\tau_{passive})=0.
\]

The posterior over the `2(m-1)` guarded causal worlds remains the prior. Neither
more passive data nor a larger static predictor can recover the hidden guard or
orientations.

## 3. T75.2 — constructive noisy adaptive search

Under `do(X=0)`, observing `Y` has mean `epsilon` for local orientation
`X->Y` and `1/2` for `Y->X`. Let

\[
g=1/2-\epsilon.
\]

### Adaptive algorithm

1. At `u=0`, repeat the intervention `r` times and classify `Theta_0`.
2. Treat the high endpoint as structurally opposite once `Theta_0` is known,
   then binary-search the first selector whose orientation is opposite to
   `Theta_0`. At every queried midpoint, repeat the intervention `r` times and
   classify the local orientation.

The number of tested selector locations is at most

\[
L=1+\lceil\log_2(m-1)\rceil.
\]

Each midpoint test has conditional error at most `exp(-r g^2/2)` under fresh
independent intervention noise. The bound holds conditional on every adaptive
history. A union bound therefore gives total world-identification error at most
`delta` when

\[
r\ge \frac{2}{g^2}\log\frac{L}{\delta}.
\]

Thus total interventions are at most `Lr`. Exact sequential likelihood-ratio
tests and a Bayes noisy-binary-search policy are stronger controls and must be
reported.

### Nonadaptive selector lower bound

Even with infinitely many repetitions at selected locations, adjacent
thresholds `t,t+1` differ only at `u=t`. Any selector set fixed before outcomes
must include every interior

\[
u=1,\ldots,m-2,
\]

and the interiors alone are insufficient to identify the low-region orientation:
worlds `(t=1,Theta_0=0)` and `(t=m-1,Theta_0=1)` agree at every interior
selector. At least one endpoint is also necessary. Conversely,
`{0,...,m-2}` suffices. Therefore the exact noiseless location complexity is

\[
q_{nonadapt}=m-1.
\]

At `m=33`, adaptive testing uses at most `6` locations while an exact fixed
design needs `32`.

Under finite noise, total-intervention comparison depends on confidence
allocation and sequential stopping. Stage A computes exact posteriors, frozen
constructive and greedy reference-policy frontiers, and a small-instance exact
dynamic-programming calibration. No exact `m=33` Bayes-optimal or minimax
frontier is claimed; the location bound alone is not an exact sample ratio.

Each charged intervention is also a binary channel from the queried local
orientation to `Y`. At `epsilon=0.1`, its capacity is

\[
C=\max_q\{h_2(0.1+0.4q)-(1-q)h_2(0.1)-q\}
\approx0.147589\text{ bits}.
\]

For 64 uniform worlds and average decoding error at most `0.05`, Fano and the
adaptive chain rule imply a fixed integer budget of at least `37`
interventions; for variable stopping, `E[N]>=36.687...`. This assumes
world-independent initial state and policy randomness, one binary result per
charged probe, actions chosen only from prior transcript, and no metadata,
timing, cache, or transcript side channel. It is a lower bound, not an
achievable policy or an optimum certificate.

## 4. T75.3 — state and scoring

All `2(m-1)` worlds are behaviorally distinct: the orientation at `u=0`
identifies `Theta_0`, and the first opposite orientation identifies `t`. Under
the uniform prior, exact hidden-world identity requires at least

\[
\lceil\log_2(2(m-1))\rceil
\]

bits after exact acquisition, plus any binding association not already supplied
by stable public identifiers. Under prediction error, use the score-equivalence
Fano/rate-distortion floor with only the discrete state and public
protocol/binding information after erasure. The decoder never receives the
erased evidence transcript.

The sufficient Bayesian state is a posterior over `(t,Theta_0)`, not a prose
summary. Posterior entries use a frozen quantization and their complete served
bit cost is charged; no learned state gets credit for using fewer bits than the
best compressed task-equivalent code. Controls may represent state however they
choose within the same served bit/work budget.

Private scoring includes the following frozen conventions:

- posterior log score `-log(max(p_true,10^-6))` over `(t,Theta_0)`;
- scalar binary Brier `(p-Y)^2` and clipped log score for interventional
  outcomes at unseen selectors;
- cumulative intervention cost plus prediction loss;
- calibration before and after each decisive outcome; and
- performance after the evidence transcript is erased.

State reset, independent-world state, evidence-order corruption, selector
permutation, and intervention-result corruption must remove the corresponding
gain.

## 5. Change and protected-region extension

The first T75 screen is stationary. A change extension may move `t` or flip
`Theta_0`, but only after the stationary composition passes.

That extension must define a quotient-invariant change kernel over the complete
piecewise causal function, freeze a finite horizon/hazard, and construct
changed/protected selector regions. A syntax-level “change the guard node” is
not admissible.

## 6. Training split: components seen, conjunction absent

Meta-training includes:

- T73 lifetimes with ordered guards and deterministic unknown local maps; and
- T74 lifetimes with noisy causal orientations but no guard.

It includes no lifetime in which a guard controls a causal orientation. T75
worlds and selector-plus-intervention conjunctions are generated only after the
updater, serialization, prompts, thresholds, and gates are frozen. The decisive
generator is hash-committed and used once. A separately generated T75-trained
ceiling cannot feed tuning information back into the candidate.

This is a factorial composition holdout, not an arbitrary foreign family. The
shared primitives and typed semantics are declared and charged. A model may
recognize the new conjunction; it must still infer how the two known update
operations compose from interaction.

## 7. Factorial causal training comparison

Use one recurrent backbone and fixed off-policy T73/T74 training lifetimes. The
core is a genuine `2x2` design:

| boundary credit | objective |
|---|---|
| full lifetime BPTT | ordinary per-step answer/action losses |
| detached persistent state | ordinary per-step answer/action losses |
| full lifetime BPTT | added future regret/cost/calibration/retention losses |
| detached persistent state | added future regret/cost/calibration/retention losses |

The four cells receive identical target sets. Exact-policy behavior cloning is
a separate oracle factor/control; matched cells either all receive its labels
or none do, and oracle calls are charged.

Match checkpoint size, persistent bytes, tokens, data order, unroll length,
tuning trials, and seeds. Report actual optimizer FLOPs and gradient work.
Because detachment is cheaper, add a compute-matched detached cell allowed to
spend the saving on additional steps. A separate on-policy phase freezes reward
access, intervention budget, initialization, and stopping rule and reports the
complete policy-data feedback loop rather than pretending interaction
distributions stay equal.

The primary causal claim belongs to whichever intervention actually produces
the T75 gain—ordinary BPTT, added lifetime terms, exact-policy supervision, or
architecture—not to the project label.

## 8. Controls against two hidden solvers

The decisive comparison includes:

1. the untouched base checkpoint;
2. T73-only, T74-only, joint T73+T74, and compute/data-matched unrelated-task
   curricula from the same initialization;
3. a counterfactually relabeled joint curriculum in which one primitive's
   evidence-to-update semantics is wrong;
4. separately trained T73 and T74 models plus an explicit learned router, with
   both experts, router, calls, state, and training fully charged;
5. a shared backbone with family-specific heads/router;
6. a sequential modular neural composition that feeds a T74 orientation
   estimator into a T73 threshold searcher;
7. exact-posterior constructive and greedy noisy-search references, plus an
   exact finite-horizon DP only on audited small instances;
8. full context and byte-matched evidence memory/retrieval;
9. test-time gradients and self-consolidation; and
10. a model trained directly on T75 as an in-distribution competence ceiling,
    not a matched zero-shot competitor.

All controls receive the same primitive-training lifetimes, typed schemas,
evidence, intervention budget, and complete tuning/resource ledger where their
role permits. Candidate-state interventions must show that corrupting learned
orientation evidence changes the corresponding threshold decisions without
changing unrelated retained worlds. The exact-posterior references are
algorithmic controls, not certified global optima. A neural model must approach
their competence/action frontier and beat the strongest non-compositional
learned matched control. If the matched sequential modular composer ties it,
the result supports component learning rather than an emergent unified updater.

## 9. Synthetic admission gates

Stage A freezes `m=33`, `epsilon=0.1`, the intervention cap `B`, the action cost
`lambda`, seed count, simultaneous-interval method, and every margin before a
decisive neural seed. Let `A_j` be post-erasure world-identification accuracy,
`A_0` the reset accuracy, and `A_ID` the separately generated directly
T75-trained ceiling. Define recovered gain as

\[
G_j=(A_j-A_0)/(A_{ID}-A_0).
\]

Let

\[
C_j=E[-\log(\max(p_j(W_{true}),10^{-6}))+\lambda N_j]
\]

Let `ctrl` be the lowest-`C` matched non-compositional learned/router control
selected by a frozen rule, `mod` the matched explicit sequential modular
composer, and `ref` the frozen exact-posterior reference policy. A result is
interesting only if simultaneous `95%` confidence intervals establish every
condition:

1. the exact-posterior reference achieves both average and frozen worst-world
   `A>=0.95`, the directly T75-trained ceiling achieves average `A>=0.95`, and
   the candidate itself achieves average `A>=0.80` under cap `B`;
2. the lower confidence bound of `G_candidate` is at least `0.80`;
3. `C_candidate <=0.70 C_ctrl`, `C_ctrl-C_candidate >=0.05` nats per lifetime,
   and `C_candidate <= C_mod+0.02` nats; a zero or negative comparison
   denominator fails rather than rescues the ratio;
4. both expected and `95th`-percentile candidate interventions are at most
   `1.5x` the exact-posterior reference at matched average and worst-world error;
5. candidate scalar Brier is no more than `0.01` and clipped log loss no more
   than `0.02` nats worse than the exact posterior predictor;
6. T73 and T74 retained accuracies are each no more than `0.01` below the same
   arm's pre-joint checkpoint; and
7. relative to `ctrl`, candidate parameter count, served-state bytes, update
   FLOPs, traffic, and p95 latency are each within a frozen `5%` equivalence
   margin, training FLOPs are within `10%`, and the quality inequalities above
   are strict. This is the declared Pareto rule.

Stage A may revise `B` and `lambda` only before its hash-committed feasibility
manifest. The family-wise interval procedure, denominator policy, seed count,
and no-threshold-changing rule are then immutable. A failure on the composed
holdout cannot be rescued by favorable T73/T74 averages.

## 10. What a pass would and would not mean

A pass would show that one trained neural update policy recombined two learned
developmental primitives in an unseen interactive world: active guard
localization and calibrated causal evidence integration. That is a large
synthetic held-out-composition gain inside a controlled grammar, especially if
ordinary full-lifetime BPTT and explicit router controls fail by an order-one
margin.

It would not yet establish a substantial model-level intelligence improvement:
there are only `2(m-1)` hypotheses and the exact answer is a short known
program. It would earn a richer synthetic successor. A natural external-
validity pilot requires independent replication, attribution to the component
curriculum rather than pretrained solver synthesis, and transfer of the same
frozen updater under the complete adaptive Pareto ledger.

### Current-art collision and negative-result limit

T75 does not claim novelty for noisy binary search, Bayesian updating,
compositional meta-learning, or interactive causal discovery. In particular,
[From Reasoning Traces to Reusable Modules](https://arxiv.org/abs/2606.18089)
reports that SFT plus RL can recover and recombine latent reasoning modules and
that training on compound traces generalizes better than training on isolated
atomic modules. [Compositional meta-learning through probabilistic task
inference](https://arxiv.org/abs/2510.01858) already makes structured reusable
computations and probabilistic task inference explicit. Current self-evolving
world-model work also updates deployment context from prediction errors while
holding model parameters fixed
([WorldEvolver](https://arxiv.org/abs/2606.30639)).

Therefore T75's isolated-component curriculum is a deliberately severe test,
not the presumed best recipe. A negative result cannot reject compositional
learning or an architecture; it only rejects the frozen curriculum/mechanism
cell. A successor must train on a lattice of diverse compound mechanisms while
holding out new compositions, and must test a property beyond reconstructing a
short known solver.

## 11. Audited disposition

The independent audit verified the passive impossibility and adaptive noisy-
search upper bound, corrected the exact nonadaptive count, and required the
attribution and gate changes above. It admits only model-free CPU Stage A:

- generator and behavioral-invariant checks;
- exact posterior/Bayes and nonadaptive solver comparisons;
- split, serialization, binding, and conjunction-leak audits; and
- numeric-gate feasibility under a frozen manifest.

The frozen [Stage A decision](guarded-causal-composition-t75-stage-a-decision.md)
reports a `7.62x` analytic adaptive-versus-fixed probe reduction at at least
95% worst-world accuracy. Its entropy-greedy reference reaches 96.03% average
accuracy with 45.40 mean probes against a 36.69 information floor. This
validates the active Bayesian operator, not learned composition.

Stage A cannot train a neural model or tune on a decisive T75 split. A separate
review is required before any tiny neural proposal. No local GPU or rental is
admitted.
