# T45 closed-loop belief boundary — theorem packet before architecture

Date: 2026-08-01  
Status: **AUDITED HOLD: RESET-VERSUS-STATE BOUNDARY ONLY; NO ARCHITECTURE OR EXPERIMENT ADMITTED**

## 0. Purpose

The corrected research goal is broader than memory or fixed compute: find a
change that produces a substantial improvement in real learning, continual
adaptation, abstraction, reasoning, world modeling, or self-correction at a
defensible complete cost.

T45 asks the first algebraic question without naming a module:

> What resource must exist before a deployed system can genuinely improve from
> experience, and what can that resource buy that a reset/frozen predictor
> cannot?

The answer is not “a Bayesian layer is intelligence.” The exact Bayes update
below is an oracle and a control. The retained result is the separation between
a reset predictor and a persistent evidence state, plus the fact that generic
recurrence contains the same solution. Any new architecture must therefore
win on acquisition, abstraction, or physical efficiency—not on renaming state.

## 1. Typed learning loop

Let a latent environment `Z` remain fixed for a run. At step `t`, an agent has
persistent state `M_t`, chooses `A_t`, observes outcome `R_t`, and applies

\[
M_{t+1}=U(M_t,A_t,R_t).
\]

There are four different objects:

1. **feedback information:** whether `(A_t,R_t)` says anything about `Z`;
2. **persistent state:** whether that information survives after the prompt or
   episode that contained it;
3. **update algebra:** whether repeated evidence is integrated correctly; and
4. **policy:** whether uncertainty changes the next action or answer.

Memory without feedback cannot learn. Feedback without persistent state cannot
improve after the evidence leaves. State with an arbitrary update can drift.
An accurate belief with a passive policy need not acquire missing evidence.

## 2. L1 — exact positive construction

### Hidden two-world family

Let `Z` be uniform on `{-1,+1}`. At every step choose `A_t in {-1,+1}` and
observe conditionally independent Bernoulli reward

\[
P(R_t=1\mid A_t,Z)=\tfrac12+\Delta A_tZ,
\qquad 0<\Delta<\tfrac12.
\]

Equivalently, conditional on the history `H_{t-1}`, the same law holds and the
new reward noise is independent of earlier reward noise. The likelihood gap
`Delta` is known to the oracle construction; its magnitude is required for a
calibrated posterior, though not for the sign of the MAP decision.

The informed action is `A_t=Z`, with expected reward `1/2+Delta`.

Define signed evidence

\[
Y_t=A_t(2R_t-1)\in\{-1,+1\}.
\]

Then

\[
P(Y_t=Z)=\tfrac12+\Delta,
\]

independently of which action was selected. The complete posterior log-odds
has the additive update

\[
L_{t+1}=L_t+Y_t
\log{\tfrac12+\Delta\over\tfrac12-\Delta},
\qquad L_0=0.
\]

Choose the MAP action `A_{t+1}=sign(L_{t+1})`, breaking a zero tie uniformly.
This is an exact persistent belief learner with one scalar sufficient
statistic.

**Plain explanation.** Every outcome is weak evidence about which world is
active. The sign of the accumulated evidence is the best current guess; its
magnitude is confidence.

**Small witness.** With `Delta=0.25`, choosing `A=+1` and receiving reward one
adds positive evidence. Choosing `A=-1` and receiving reward zero also adds
positive evidence: failure of the negative-world action supports `Z=+1`.

**Boundary.** The likelihood law is known. Learning the observation-to-evidence
map is not solved by this construction.

## 3. L2 — regret separation and state information

### Theorem 1: reset prediction has linear regret

Suppose `A_t` is independent of every earlier outcome after conditioning on
the current empty observation—that is, no prompt, weight, external memory, or
persistent state carries run-specific evidence. Define cumulative reward
`G_T=sum_{t=1}^T R_t`. Then its Bayes expected reward is exactly `1/2` per step
and

\[
\mathbb E[G_T^{reset}]=T/2,
\qquad
\mathbb E[\operatorname{Reg}_T^{reset}]=\Delta T.
\]

**Proof.** Because `Z` is uniform and independent of `A_t`,
`E[A_tZ]=0`. Thus
`E[R_t]=1/2+Delta E[A_tZ]=1/2`. The informed policy obtains `1/2+Delta`,
so per-step regret is `Delta`. Sum over `T`. QED.

This lower bound applies to the reset condition, not to Transformers in
general. A Transformer with persistent context or writable state is no longer
in the reset class.

### Theorem 2: the exact belief learner has bounded regret

After `n` evidence items, majority/MAP error obeys

\[
P(\widehat Z_n\ne Z)\le e^{-2n\Delta^2}.
\]

The wrong action loses `2Delta` expected reward, so

\[
\mathbb E[\operatorname{Reg}_T^{belief}]
\le \Delta+2\Delta\sum_{n=1}^{T-1}e^{-2n\Delta^2}
\le \Delta+{2\Delta\over e^{2\Delta^2}-1}.
\]

For fixed `Delta`, this loose upper bound is bounded as `T` grows, while reset
regret is linear. It is not uniform as `Delta` approaches zero.

**Proof.** Map `Y_t` to a Bernoulli variable that is correct with probability
`1/2+Delta`. Hoeffding bounds the probability that its empirical mean falls on
the wrong side of `1/2` by `exp(-2nDelta^2)`. Multiply by the `2Delta` loss of a
wrong action and sum. The initial tie contributes `Delta`. QED.

### Theorem 3: persistent information lower bound

For `K` equiprobable environments and any state `M` from which a decoder
identifies the current environment with error at most `epsilon`, Fano gives

\[
B_M\ge H(M)\ge I(Z;M)
\ge \log_2K-h_2(\epsilon)-\epsilon\log_2(K-1)
\]

bits when `M` is a discrete code or a physical channel with capacity `B_M`.
An unconstrained mathematical real is not a finite bit ledger. Once the
identifying evidence is absent from the request, those bits
must live in model-readable persistent state, updated parameters, or a charged
external artifact.

This is the price of learning, not an inefficiency. The research question is
whether the state represents shared structure and uncertainty efficiently.

## 4. L3 — identifiability boundary

The exact update is learnable only if experience distinguishes worlds.

### No-feedback counterexample

If

\[
P(A_t,R_t\mid Z=z)=P(A_t,R_t\mid Z=z')
\]

for two worlds under every legal policy, their likelihood ratio is always one.
No learner can distinguish them better than its prior, regardless of memory,
optimizer, model size, or reasoning depth.

### Wrong-factor counterexample

Even informative outcomes can be integrated into the wrong coordinate. A
generic hidden state may correlate with recency, action identity, or reward
frequency while failing to represent `Z`. High state entropy or perfect
reconstruction does not prove a belief update.

### The bundled edge exposed by this witness

The tempting empirical arrow

```text
raw observation + action + outcome
    -> calibrated evidence increment over a reusable latent factor
```

is not one primitive. It bundles representation discovery, factor routing,
causal credit assignment, likelihood calibration, environment segmentation,
and cross-family transfer. T45 cannot attribute a gain to any one of them
because it assumes a known factor and known likelihood. The next paper must
separate these maps before any learned component is admitted.

## 5. L4 — strongest-control containment

An unrestricted recurrent controller with the same scalar state and operations
implements `L_{t+1}=L_t+delta_t` exactly. So can a fast-weight layer, a small
external table, a program, or a Transformer that keeps the evidence in an
untruncated context.

Therefore:

1. persistent belief has a strict capability separation from **reset**
   prediction;
2. it has no function-class separation from a matched recurrent/stateful
   control; and
3. a proposed architecture must demonstrate at least one of:
   - lower sample complexity for learning the evidence coordinates;
   - less cross-task interference or better horizon extrapolation;
   - a strict state/update/traffic advantage;
   - better abstraction reuse across world families; or
   - a better active-information policy at the same interaction and work cost.

The oracle Bayes filter is the upper control, not the candidate.

## 6. L5 — resource vector

The learning ledger extends the ordinary serving vector:

\[
C_{learn}=(P_{static},B_{persistent},B_{workspace},F_{predict},F_{update},
T_{traffic},D_{critical},N_{interactions},N_{calls},N_{tokens},E,L).
\]

For the exact two-world oracle:

- persistent state is one log-odds scalar or one clipped signed count;
- update work is one signed increment;
- action read is one sign test; and
- a signed counter that supports every intermediate time through horizon `T`
  is representable in `ceil(log2(2T+1))` bits. This is a sufficient horizon-wide
  allocation, not a tight lower bound at one known parity-constrained time.

The representation must additionally charge `Delta`, time/parity, calibration,
and update metadata whenever they are not compiled constants.

For a one-time decision, collecting

\[
n_0=\left\lceil{\log(1/\epsilon)\over2\Delta^2}\right\rceil
\]

samples is sufficient for the Hoeffding error target. This does **not** justify
clipping a continually updated signed count. A posterior stopping rule may
instead freeze when

\[
|L|\ge\log{1-\epsilon\over\epsilon},
\]

or, in count units, when

\[
|k|\ge
{\log((1-\epsilon)/\epsilon)\over
 \log((1/2+\Delta)/(1/2-\Delta))}.
\]

Freezing loses change-point adaptation. A saturating counter that keeps
updating is a separate finite Markov chain and needs its own error and
calibration theorem; no such clipping theorem is admitted here.

A neural candidate must additionally charge key formation, slot creation,
uncertainty representation, addressing, update writes, conflict/change-point
logic, replay or consolidation, and any backward pass.

## 7. L6 — substantial end-effect path

The theorem gives a potentially unbounded horizon gap, but only on a known
two-world family. A real candidate must propagate one learned evidence-error
quantity into the end curve.

Let the learned evidence sign be correct independently with probability
`1/2+gamma` on an unseen family while the environment reward gap remains
`Delta`. The corresponding optimistic envelope is

\[
P(\widehat Z_n\ne Z)\le e^{-2n\gamma^2},
\qquad
\mathbb E[\operatorname{Reg}_T]
\le \Delta+{2\Delta\over e^{2\gamma^2}-1}.
\]

A candidate is not admitted by average sign accuracy alone: it must also bound
dependence, variable evidence magnitude, calibration, change-point harm, and
routing to the correct latent factor.

The early substantial gate inherited from the charter is:

- at least 30% lower cumulative regret or ten absolute points against every
  same-size matched learner control;
- at least 2x lower interaction complexity to a fixed competence;
- nonpositive median forgetting with no protected loss beyond one point;
- causal collapse when the learned evidence/update edge is shuffled; and
- success across continual facts, abstraction transfer, and active world
  inference—not only this two-world witness.

## 8. What the result changes

The key shift is precise:

> One necessary resource is model-readable persistent information updated by
> outcome evidence. Adding such state is trivial and fully contained by generic
> recurrence. The earlier frontier is discovering a behaviorally and causally
> sufficient state, attributing outcomes to the right coordinates, and choosing
> interventions that distinguish competing mechanisms; only then does belief
> accumulation become meaningful.

The two-world family cannot test this frontier because every action has the
same evidence quality: it contains no exploration-versus-exploitation choice.
It points toward a multi-timescale learner—slow general knowledge, fast belief
state, and consolidation—but does not yet select its implementation.

## 9. Next admissible paper work

Before code, define an interventional predictive-state quotient, its
identification conditions, exact controlled update, information-seeking policy,
and delayed-credit counterexample. Compare any later learned realization
against:

1. finite-context Transformer with equal state bytes;
2. GRU/SSM/recurrent controller;
3. ordinary gradient TTT and In-Place TTT;
4. external episodic key-value memory;
5. oracle Bayes/filtering and exact algorithmic learner; and
6. model-free cross-episode meta-RL.

The candidate must learn a shared update representation across at least three
world families. A per-world hand-coded sufficient statistic, another prompt
memory, or a one-benchmark gain is closed by this packet.

No CPU, local GPU, or rented GPU run is admitted yet.
