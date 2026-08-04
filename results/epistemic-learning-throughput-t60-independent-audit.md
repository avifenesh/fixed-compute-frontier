# Independent adversarial audit: epistemic learning throughput (T60)

Date: 2026-08-01 — Scope: information-theoretic claims, interactive separation, literature accuracy, and run gate; no experiments run

## Verdict

**RETAIN AS A DIAGNOSTIC DECOMPOSITION; REPAIR THE ADAPTIVE-LIFETIME INEQUALITY; NO DISTINCT UPDATER; NO RUN.**

The one-step data-processing statements, the worst-case storage bound, and the
basic Fano consequence are standard and valid after their Markov, conditioning,
and decision-identification assumptions are stated. The lifetime acquisition
display is not valid as written: conditioning on the entire adaptively chosen
action sequence can condition away information already carried through earlier
outcomes and retained in memory.

The small-interactive-versus-static example is true but trivial and unmatched.
Only the interactive learner receives `n` bits about the world; its questions
need not be adaptive and a static learner given the same answer batch matches it.
It proves an evidence-access separation, not a model-size, updater, or
interaction advantage.

Acquisition, retention, and utilization are useful diagnostic axes, not an
update rule; AgentOdyssey already occupies the framing at systems level.

## T60 section 1: internal computation

If

\[
Z=f(K,R),\qquad R\perp\!\!\!\perp\Theta\mid K,
\]

and `f` invokes no additional `Theta`-correlated source, then

\[
I(\Theta;Z\mid K)=0.
\]

This follows from the conditional Markov chain `Theta -> K -> Z`. The conditional
entropy equality also follows for discrete variables with finite/well-defined
entropy. Mutual information is the safer statement for general variables.

The assumptions are load-bearing. A tool result, environment observation,
human label, nondeterministic verifier correlated with the world, or retrieved
record not already included in `K` is a new channel. Conversely, proof search
can expose consequences computationally hidden from a bounded actor while adding
zero Shannon information conditional on the full `K`; T60 must distinguish
statistical information from computational accessibility.

## T60 section 2: one-step DPI is correct; lifetime display is not

The claimed one-step inequality

\[
I(\Theta;M_{t+1}\mid M_t,A_t)
\le I(\Theta;O_{t+1}\mid M_t,A_t)
\]

requires

\[
\Theta\;--\;(M_t,A_t,O_{t+1})\;--\;M_{t+1},
\]

conditional on `(M_t,A_t)`: updater randomness must be independent of `Theta`,
and the update may receive no other world-correlated input. Under that condition,
conditional data processing proves the inequality.

The next display is false in general:

\[
I(\Theta;M_T\mid M_0)
\not\le I(\Theta;O_{1:T}\mid M_0,A_{0:T-1}).
\]

Counterexample: let the first outcome equal `Theta`, let the next action copy
that outcome, and let final memory retain it. Conditioning on the full action
sequence reveals `Theta`, so the right-hand conditional mutual information can
be zero while the left side is positive. Future actions are descendants of past
outcomes; treating them as ordinary conditioning variables creates this defect.

The exact adaptive statement uses the full transcript. Define

\[
H_t=(M_0,A_{0:t-1},O_{1:t}).
\]

If each policy chooses `A_t` from `H_t` using randomness independent of `Theta`,
and final memory is a stochastic function of the transcript with independent
update randomness, then

\[
I(\Theta;M_T\mid M_0)
\le I(\Theta;A_{0:T-1},O_{1:T}\mid M_0)
=\sum_{t=0}^{T-1}I(\Theta;O_{t+1}\mid H_t,A_t).
\]

The equality follows by the chain rule because
`I(Theta;A_t|H_t)=0`. If the policy has private world-correlated side
information, add it to the transcript; otherwise the theorem silently omits a
channel. This is an adaptive-information/directed-information accounting, not
the outcome-stream conditioning used in T60.

## Retention, utilization, and Fano

If final memory has worst-case support size at most `2^B`, then

\[
I(\Theta;M_T\mid M_0)\le H(M_T\mid M_0)\le B
\]

bits. This requires an actual finite digital state. A continuous vector with
unbounded precision has no such `B`; variable-length storage needs an expected
code-length argument. Only unbounded **independent innovation entropy** forces
growing storage. Repeated or compressible evidence need not.

For common side information `S=(X,G)` and independent decision randomness,

\[
Y\perp\!\!\!\perp\Theta\mid(M_T,S)
\quad\Longrightarrow\quad
I(\Theta;Y\mid S)\le I(\Theta;M_T\mid S).
\]

This fails if the actor also sees a fresh observation, tool result, pretrained
side channel, or other world-correlated input not in `(M_T,S)`.

Fano applies to exact identification. If `Theta` is uniform over `N` hypotheses
conditional on side information `S`, and `hat Theta` is decoded from `(Y,S)`,
then

\[
P_e\ge 1-\frac{I(\Theta;Y\mid S)+\log 2}{\log N}.
\]

Logs must use one base consistently. For decision performance, apply Fano to the
required decision label, not automatically to the complete world. If many worlds
share the same correct action, world-identification information is unnecessary.
Mutual information also does not weight rare but decision-critical distinctions;
regret or value of information remains necessary.

## The three `C` quantities are not valid capacities as written

T60 conditions acquisition, retention, and use on different variables, so

\[
C_{use}\le C_{ret}\le C_{acq}
\]

does not follow. Conditional mutual information can rise or fall when different
side information is introduced, and the acquisition term repeats the invalid
conditioning on all adaptive actions.

Use one common side-information variable `S` and define

\[
C_{acq}=I(\Theta;H_T\mid S),\quad
C_{ret}=I(\Theta;M_T\mid S),\quad
C_{use}=I(\Theta;Y\mid S).
\]

If `Theta -> H_T -> M_T -> Y` is a conditional Markov chain given `S`, data
processing gives the ordering. These are realized mutual informations, not
channel capacities; “capacity” normally requires a supremum over an allowed
input/design distribution. Rename them `J_acq,J_ret,J_use`, or explicitly define
the admissible policies/channels and optimization that makes each a capacity.
Nor does a minimum-of-three capacity law follow automatically from the current
definitions.

## The small-interactive/static separation is not an updater result

The exact success claims hold if `Theta` is independent of the static model's
weights/context, the prior is uniform, exact single-guess identification is
scored, and the interactive interface exposes noiseless truthful bit queries in
a known encoding. Then no-evidence success is `1/N`, while storing `n=log_2 N`
answers permits exact decoding.

But:

- the `n` bit queries are nonadaptive and can be requested as one static batch;
- a static learner given the same `n` answers also succeeds exactly;
- the interactive learner is handed the correct question family and decoder;
- persistent storage grows as `log N`, while neural parameter count is
  irrelevant to the information comparison; and
- noisy answers require channel-capacity/error-correction analysis and more than
  `n` observations.

The correct conclusion is: **a system receiving informative evidence can beat
one receiving none, regardless of parameter count**. A genuine interaction
separation needs matched evidence/action costs and an environment where adaptive
experiments achieve lower sample complexity than every passive/static design.

## Accuracy of the cited 2026 evidence

- [Persistent Computational State](https://arxiv.org/abs/2607.21686) does show
  that snapshot/restore of existing runtime state fixes a class of apparent
  continuity failures. It is a runtime-state result, not lifetime epistemic
  learning.
- [Current World Models Lack a Persistent State Core](https://arxiv.org/abs/2606.20545)
  reports 9,600 videos from 23 models and failure to evolve unseen events. This
  supports a model-state diagnostic, not the three-link theorem.
- [Agentic Test-Time Training](https://arxiv.org/abs/2607.03441) reports up to
  5.0/4.9-point gains, `1.9x` cost, and explicitly says gains concentrate on
  preserving existing competence. T60's summary is accurate.
- [Dream Rehearsal](https://arxiv.org/abs/2607.19749) finds world-model retention
  with actor forgetting only in its DreamerV3, never-clear-replay protocol with
  three seeds. Present it as that scoped result, not a general property.
- [Fast-Slow Training](https://arxiv.org/abs/2605.12484), [Continual
  Harness](https://arxiv.org/abs/2605.09998), [ALMA](https://arxiv.org/abs/2602.07755),
  and [dynamic predicate invention](https://arxiv.org/abs/2602.17217) support the
  broad adaptation/memory-design/online-repair descriptions, but do not validate
  T60's information inequalities.
- [AgentOdyssey](https://arxiv.org/abs/2606.24893), listed but not discussed in
  T60, already evaluates acquisition, episodic memory, exploration, planning,
  and cost separately. It is the closest direct collision with the proposed
  diagnostic framing.

All nine are 2026 preprints; treat them only as evidence for their reported protocols.

## Exact corrections and experiment disposition

1. Replace the lifetime outcome-conditioned inequality with the full-transcript
   chain-rule statement above.
2. State all conditional-Markov and independent-randomness assumptions for
   acquisition and utilization.
3. Use common side information for all three information quantities; rename
   them unless an actual channel-capacity optimization is defined.
4. Scope the `B`-bit theorem to finite digital state and independent innovation
   entropy.
5. State conditional Fano and apply it to the required decision label when exact
   world identification is unnecessary.
6. Reclassify the toy as evidence-versus-no-evidence, not interactive-versus-
   static or small-versus-large architecture separation.
7. Add AgentOdyssey to the prose and qualify Dream Rehearsal and all preprints by
   their actual protocols.

**No experiment is earned.** These are accounting identities and diagnostic
cuts, not an updater. A later candidate must improve a preregistered task-weighted
measure under matched evidence, storage, actions, model calls, and served compute.
