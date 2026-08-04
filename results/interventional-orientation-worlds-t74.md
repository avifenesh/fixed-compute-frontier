# T74 interventional orientation worlds — exact stochastic causal calibration

Date: 2026-08-02  
Status: **INDEPENDENTLY REVISED; DECOMPOSABLE BAYESIAN-UPDATE CALIBRATION ONLY; NO CODE OR RUN**

## 0. Scope

T73 is noiseless active concept learning plus deterministic execution. T74
tests a different operation: calibrated causal belief update from stochastic
interventions when every passive observation is exactly uninformative.

Independent review classifies this paired v0 as a decomposable calibration,
not causal-discovery breadth evidence. Its sufficient statistic is one scalar
log-odds per pair. T74 alone cannot establish a shared developmental policy or
admit a natural pilot.

## 1. Paired causal world

V0 uses public fixed `epsilon=0.1`. A latent or varying noise rate is a new
version requiring joint estimation.

A world contains `n` independent binary pairs

\[
(X_i,Y_i),\qquad
\Theta_i\in\{X_i\to Y_i,\ Y_i\to X_i\},
\]

where orientations are freshly uniform after model training and independent of
the model and surface binding.

For `X_i -> Y_i`,

\[
X_i=U_i,
\qquad
Y_i=X_i\oplus N_i,
\]

and for `Y_i -> X_i`,

\[
Y_i=U_i,
\qquad
X_i=Y_i\oplus N_i,
\]

with independent `U_i~Bernoulli(1/2)` and
`N_i~Bernoulli(epsilon)` on every sample.

Pair and variable names are freshly permuted independently of orientation and
cannot encode causal roles. The typed interface is:

```text
OBSERVE                 -> passive sample of all pairs
INTERVENE variable=bit  -> post-intervention sample, charged
PREDICT intervention    -> private scored scalar probability
ORIENT pair             -> private scored orientation and confidence
```

## 2. T74.1 — passive information is exactly zero

Under either orientation,

\[
P(X=x,Y=y)=
\begin{cases}
\frac12(1-\epsilon),&x=y,\\
\frac12\epsilon,&x\ne y.
\end{cases}
\]

Thus every passive sample has the same orientation-independent product law.
For a policy whose pair selection and stopping depend only on prior passive
history and private policy randomness, the chain rule adds zero conditional
mutual information at every step:

\[
I(\Theta_{1:n};\tau_{passive})=0.
\]

This includes adaptive passive stopping. It requires no score, intervention
result, change announcement, orientation-correlated name, or other side channel
to enter the transcript. Expected exact orientation accuracy remains `1/2` per
pair regardless of passive sample count or model size.

## 3. T74.2 — intervention creates a separated statistical test

Under a hard `do(X_i=0)` with fresh exogenous noise, observing `Y_i` gives

\[
Y_i\sim
\begin{cases}
Bernoulli(\epsilon),&X_i\to Y_i,\\
Bernoulli(1/2),&Y_i\to X_i.
\end{cases}
\]

Let `g=1/2-epsilon`. After `r` repeated interventions on the pair, the midpoint
classifier has, by one-sided Hoeffding,

\[
P(error\mid\Theta_i)\le \exp(-r g^2/2).
\]

A union bound gives simultaneous correctness for all `n` pairs with probability
at least `1-delta` when

\[
r\ge \frac{2}{g^2}\log\frac{n}{\delta}
\]

interventions are used **per pair**. This is one informative intervention type,
not one sample. The exact likelihood-ratio/Bayesian error is the primary
control; the displayed result is a transparent sufficient bound for known
`epsilon`.

## 4. T74.3 — order-one scoring gap

Before intervention, direction accuracy is exactly `50%`; after the bounded
test it approaches `100%`.

For private `P(Y_i=1 | do(X_i=0))`, use scalar binary squared loss
`L(p,y)=(p-y)^2`. The two oracle probabilities are `epsilon` and `1/2`; the
passive equal-prior mixture is their mean. Expected excess passive Brier loss is

\[
\operatorname{Var}(p_{\Theta_i})
=\frac14(1/2-\epsilon)^2
=\frac{g^2}{4}.
\]

At `epsilon=0.1`, this equals `0.04` per query. The two-coordinate Brier
convention would double the number and is not used. Log-score regret is reported
separately.

## 5. T74.4 — state and evidence floors

Let `Z` contain only public protocol and stable binding information available
after erasure; it excludes the intervention transcript being compressed. Then
`H(Theta|Z)=n`.

If a decoder recovers the full orientation vector from a discrete or quantized
`B`-bit state `M` with block error `p_e`, conditional Fano and
`I(Theta;M|Z)<=H(M)<=B` give

\[
B\ge n-h_2(p_e)-p_e\log_2(2^n-1).
\]

For mean Hamming error `D`, use the independent-bit rate-distortion floor

\[
B\ge n[1-h_2(D)].
\]

If later queries do not supply stable pair identities, their association bits
are charged too. The optimal learned-system ceiling stores one posterior
log-odds per unresolved pair at declared precision, not merely the final bit.

## 6. Frozen change/monitoring family

V0 has a finite public monitoring horizon `H_m`. Independently for each pair, a
public prior draws either no change or one orientation-flip time in
`1,...,H_m`. There is at most one flip, no restoration, and `epsilon` remains
fixed. The per-pair priors factorize.

Passive observations remain unchanged after a flip. An intervention on pair
`i` is the only source of direct likelihood about its current orientation;
factorized priors mean evidence about another pair does not update `i`.

Before a manifest runs, freeze numerical `H_m`, the no-change probability and
conditional change-time distribution, intervention cost, declaration
threshold, false-alarm metric, delay definition, and Bayes-average versus
worst-case reporting. Private changed/protected query sets use the stable
behavioral pair identities after surface permutation.

Report the full monitoring-cost/detection-delay frontier, posterior calibration,
revision regret, false changes, and protected loss. An exact per-pair finite-
horizon Bayes dynamic program is the task-optimal control. Restoration requires
a new version with its own prior.

## 7. Exact controls

Mandatory controls are:

1. passive Bayes mixture and chance orientation;
2. exact per-pair likelihood-ratio/Bayesian update;
3. fixed round-robin interventions;
4. exact finite-horizon Bayes monitoring;
5. full transcript as an information ceiling;
6. byte-matched raw/reservoir evidence memory;
7. learned retrieval/compression and generic recurrent state;
8. test-time gradients and self-consolidation with update costs; and
9. state, evidence-order, action, calibration, revision, and retention
   corruptions.

Pair permutation, evidence order, `n`, and change times are disjoint across
development and decisive generators. `epsilon=0.1` is public and common.

## 8. Current-art boundary

T74 is classical sequential hypothesis testing and change detection. It makes
no component-novelty claim.

[CausaLab](https://arxiv.org/abs/2605.26029) already evaluates interactive SCM
discovery and reports a prediction/mechanism/intervention-strategy gap.
[Agentic automata learning](https://arxiv.org/abs/2606.16576) reports query
planning, evidence integration, and hypothesis construction failures against
classical learners. [Causal Discovery in
Action](https://proceedings.mlr.press/v323/panayiotou26a.html) gives a 2026
identifiability and logarithmic-sample control for structured chain-reaction
systems.

T74 is deliberately simpler: it calibrates stochastic evidence integration,
indexing, monitoring, and retention exactly.

## 9. Why T73 plus T74 is not enough

Their schemas reveal the family. One checkpoint can encode two conditional
solvers without learning a reusable update operation. Combined controls must
therefore include:

1. two separately trained family solvers plus an explicit router under the
   same total parameter/state/training ledger;
2. one shared backbone with family-specific heads/router;
3. one generic event/action serialization, while acknowledging that statistics
   can still reveal family;
4. T73's factorial BPTT/credit/objective comparison; and
5. a held-out composed world requiring guarded localization and noisy causal
   updating in one lifetime, with no family-specific retraining.

A positive T74 result means only scalar Bayesian updating. A positive T73/T74
mixture without the composed holdout means only that a multitask network stores
two solvers. Neither earns a natural pilot.

## 10. Decision

The passive equality, intervention separation, Hoeffding bound, scalar-Brier
gap, and corrected state floors are retained. T74 is a valid mathematical
calibration after a numerical monitoring manifest is frozen.

The next paper object is the held-out cross-composition: causal orientation that
changes across a hidden ordered guard. That composition must combine T73's
active localization with T74's noisy evidence update and must be absent from
updater training.

No CPU experiment, neural training, local GPU use, or rental is admitted by
this revision.

Independent review:
[`interventional-orientation-worlds-t74-independent-audit.md`](interventional-orientation-worlds-t74-independent-audit.md).
