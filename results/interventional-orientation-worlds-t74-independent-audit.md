# T74 interventional orientation worlds — independent audit

Date: 2026-08-02  
Status: **CORE PROBABILITY SEPARATION PASSES; CHANGE/FANO/JOINT-TRAINING CONTRACT REQUIRES REVISION; NO COMBINED TRAINING ADMITTED**

## Verdict

T74 is mathematically sound as a small calibration of passive causal
nonidentifiability, noisy evidence accumulation, and set-indexed retention. The
passive-law equality, intervention distributions, Hoeffding/union bound, and
scalar Brier calculation are correct under explicit assumptions below.

It is not a causal-discovery breadth result as written. Each pair is an
independent binary hypothesis solved by one scalar log-likelihood ratio, and the
T73/T74 action schemas reveal the family. A single checkpoint can therefore
route to a threshold/affine solver or a Bernoulli counter without learning a
shared developmental update policy. Before combined training, T74 must fix the
status of `epsilon`, the Fano conditioning, the change hazard/monitoring loss,
and add separate-solver/router and cross-family-composition controls.

Keep the no-code/no-run status.

## 1. Passive equality and adaptive collection

The one-pair joint law is correct in both orientations:

\[
P(X=x,Y=y)=\tfrac12(1-\epsilon)\mathbf 1[x=y]
+\tfrac12\epsilon\mathbf 1[x\ne y].
\]

Because samples and pairs are independent, the product passive law is
orientation-independent. The mutual-information claim also survives adaptive
passive stopping or pair selection: at each step the passive action is a
function of the previous passive history and private policy randomness, while
the next observation has the same conditional law for every `Theta`. The chain
rule therefore adds zero conditional mutual information at every step, including
the stopped transcript.

State the required conditions explicitly:

- orientations are freshly uniform and independent of the pretrained model;
- pair/variable-name permutations are independent of orientations and do not
  encode causal roles;
- a passive policy receives no score, intervention result, change announcement,
  or correlated side channel; and
- the passive sampling/stopping mechanism itself is orientation-independent.

Under these conditions `I(Theta;tau_passive)=0` and expected per-pair exact
accuracy is `1/2`. The claim should be conditional on the public typed protocol,
not on any erased evidence transcript.

## 2. Intervention and finite-sample calculations

For a perfect hard intervention `do(X_i=0)` with fresh independent exogenous
noise on every repetition,

\[
Y_i\sim\operatorname{Bernoulli}(\epsilon)
\quad\text{under }X_i\to Y_i,
\]

and

\[
Y_i\sim\operatorname{Bernoulli}(1/2)
\quad\text{under }Y_i\to X_i.
\]

Thus `g=1/2-epsilon`. At the midpoint threshold, the deviation from either mean
is `g/2`, so one-sided Hoeffding gives

\[
P(error\mid\Theta_i)\le \exp(-r g^2/2).
\]

The union bound over `n` pairs yields the stated sufficient condition

\[
r\ge (2/g^2)\log(n/\delta).
\]

This is correct for `r` interventions **per pair** and known `epsilon`. “One
intervention creates a separated test” must mean one intervention *type*, not
one sample.

T74 later varies allowed `epsilon` values without saying whether the current
value is known. Freeze one of:

1. public fixed `epsilon`, to which the displayed threshold and Bayes control
   apply; or
2. latent `epsilon`, in which case passive equality still holds but epsilon must
   be estimated from passive agreement rates or jointly marginalized. Charge
   those samples/state and recompute the composite-hypothesis test and monitoring
   policy.

The exact likelihood-ratio/Bayes error is the right primary control.

## 3. Brier and Fano corrections

For scalar binary squared loss `L(p,y)=(p-y)^2`, Bayes regret is `(p-q)^2`.
The passive mixture differs from either oracle probability by `g/2`, so expected
excess loss is indeed

\[
g^2/4,
\]

equal to `0.04` at `epsilon=0.1`. The commonly used two-coordinate Brier score
sums the errors for both outcomes and doubles this to `g^2/2`. Declare the scalar
convention in the protocol. Report the log-score analogue separately rather than
sharing the Brier number.

The displayed block-error Fano bound is also correct for uniform
`Theta in {0,1}^n`, a discrete/quantized `B`-bit state, and a decoder that sees
only that state plus public side information after erasure:

\[
B\ge n-h_2(p_e)-p_e\log_2(2^n-1).
\]

Do **not** condition this bound on the complete pre-erasure intervention
transcript: that transcript is the evidence being compressed and is unavailable
to the decoder. Write instead `I(Theta;M | Z)`, where `Z` is only public protocol
and stable binding information and `H(Theta|Z)=n`. If stable surface pair names
are not supplied with later queries, charge the state needed to associate each
posterior with its permuted pair. For mean Hamming error `D`, state the relevant
rate-distortion floor (for independent uniform bits, `n[1-h_2(D)]`) rather than
using block Fano.

## 4. Change detectability and monitoring are underdefined

Passive observations remain exactly unchanged after an orientation flip. A new
intervention on pair `i` is the only source of **direct likelihood** about its
current orientation. The stronger sentence that only interventions on changed
pairs “supply evidence for revision” is false under a coupled change prior: if
exactly one pair flips, evidence that an intervened pair did not flip increases
the posterior probability that another pair did. Either use independent
per-pair flip hazards, which preserve posterior factorization, or model and
charge this coupling in the dynamic program.

A monitoring theorem also needs all of:

- finite horizon or a frozen change-time hazard;
- distribution/cardinality of subset `C` and whether pairs flip independently;
- minimum dwell time and restoration law;
- intervention budget/cost and whether passive observation has a cost;
- detection declaration threshold, false-alarm metric, and delay definition;
- Bayes-average versus worst-case guarantee; and
- the initial post-acquisition posterior supplied at monitoring start.

With an unbounded horizon, positive intervention cost, and adversarial
change time/pair, no policy can promise both bounded total monitoring cost and
uniform bounded detection delay. The proposed cost-delay Pareto frontier is the
right form, but its horizon/hazard and comparator must be frozen first. Define
`Q_changed` and `Q_protected` over the behavioral pair identities after surface
permutation; their construction is otherwise sound.

## 5. T74 is intentionally decomposable—and therefore limited

For known `epsilon` and independent flips, the sufficient statistic is one
log-likelihood ratio per pair. Pair permutation tests equivariance/indexing,
`n` tests bounded state, and changes test monitoring. There is no multi-node
causal graph, intervention-choice ambiguity within a pair, shared mechanism to
discover, or cross-pair abstraction. Either variable/bit intervention supplies
a separated test; the nontrivial action problem is only allocation over pairs
and time.

This makes T74 an excellent unit test for calibrated evidence accumulation but
a weak “non-isomorphic breadth” test. It is substantially simpler than current
interactive causal-discovery work. The cited boundaries are accurate:
[CausaLab](https://arxiv.org/abs/2605.26029) requires recovering graphs and
structural equations and reports a prediction/mechanism gap;
[Causal Discovery in Action](https://proceedings.mlr.press/v323/panayiotou26a.html)
provides finite-sample interventional recovery under a structured chain-reaction
model; and
[agentic automata learning](https://arxiv.org/abs/2606.16576) already tests
interactive hypothesis construction against classical learners.

Classical sequential probability-ratio/Bayesian testing and change-point
detection are mandatory task-optimal controls. A neural entrant should approach,
not be expected to beat, their information frontier.

## 6. Joint T73/T74 training does not yet show one updater

The interfaces expose different action/type tokens (`PROBE/ROLLOUT` versus
`OBSERVE/INTERVENE/ORIENT`) and different observation shapes. Family recognition
is therefore immediate. “No family-specific parameters or adapters” does not
prevent one weight tensor from encoding two conditional solvers. Quotient-disjoint
worlds inside each family test within-family generalization, not transfer of an
update algorithm between families.

Before a combined protocol, add these controls:

1. two separately trained family solvers plus an explicit router, under the same
   total parameter/state/training-cost ledger;
2. one shared backbone with family-specific heads/router, to measure whether
   hidden conditional specialization explains the result;
3. a unified generic action/event serialization, while acknowledging that
   observation statistics can still identify the family;
4. the full T73 factorial credit/objective comparison: detached state, ordinary
   full-lifetime per-step loss with BPTT, added future-competence terms, and exact
   policy imitation; and
5. a held-out **composed** world requiring guarded active localization and noisy
   Bayesian interventional updating in one lifetime, evaluated without
   family-specific retraining.

Use fixed off-policy lifetimes to isolate loss/credit effects, then report a
separate on-policy end-to-end comparison; otherwise objectives change their own
training data. Match action supervision, reward access, tokens, updates, tuning,
and compute.

Without the composed-world test, a positive joint result means only that one
multi-task network stores two solvers. It does not establish a reusable
developmental policy and should not admit a natural domain. Passing T74 alone
can admit a richer stochastic-causal synthetic successor; passing the combined
factorial and composed holdout may then justify a small natural external-validity
pilot under T71's full adaptive Pareto controls.

## Exact corrections before combined training

- Keep the passive, intervention, Hoeffding, and scalar-Brier results, with their
  assumptions and conventions stated.
- Freeze whether `epsilon` is public or latent.
- Rewrite Fano conditioning so the erased evidence transcript is not side
  information.
- Freeze a complete change hazard, subset prior, horizon, and cost-delay loss;
  correct the coupled-prior evidence statement.
- Label T74 a decomposable Bayesian-update calibration, not causal-discovery
  breadth evidence.
- Add matched router/two-solver controls, the factorial objective comparison,
  and a cross-family composed holdout.
- Admit no CPU/neural combined protocol or natural pilot from the current draft.
