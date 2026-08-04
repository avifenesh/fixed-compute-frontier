# T60 epistemic learning throughput — acquisition, retention, and use are separate channels

Date: 2026-08-01  
Status: **DIAGNOSTIC INFORMATION ACCOUNTING RETAINED; DISTINCT UPDATER REJECTED; NO RUN**

## 0. Result in one sentence

A deployed system becomes more knowledgeable only when information about an
unknown world crosses three links: external evidence must enter, a persistent
state must retain it, and later decisions must depend on that state. The useful
learning rate is bottlenecked by the weakest link. More internal prose or
self-reflection can spend computation on an existing posterior, but cannot add
epistemic information without a new observation, verifier, or consequence.

## 1. Separate computation from learning

Let `Theta` denote an unknown world, rule, user, task, or mechanism. Let `K`
contain everything the model already has: pretrained weights, current context,
retrieved records, and the current goal. Before receiving any new external
outcome, let an internal transcript be

\[
Z = f(K,R),
\]

where `R` is fresh randomness independent of `Theta` conditional on `K`, and
`f` invokes no additional world-correlated source.
Then

\[
I(\Theta;Z\mid K)=0,
\qquad
H(\Theta\mid K,Z)=H(\Theta\mid K)
\]

for discrete variables with well-defined entropy. The mutual-information claim
is the safer general statement.

This does **not** say internal reasoning is useless. Search, sampling,
reflection, or proof execution can find a better decision computable from the
same information by making it computationally accessible. It says they do not
create new Shannon evidence about `Theta`.
Self-generated labels can transfer computation into weights, but any new truth
content must ultimately come from a rule/verifier already in `K` or from an
external observation.

## 2. The acquisition bound

At lifetime step `t`, let the agent choose an action or experiment `A_t` from
history `H_t`, receive outcome `O_{t+1}`, and update persistent state

\[
M_{t+1}=U(M_t,A_t,O_{t+1},R_t).
\]

If updater randomness is world-independent and the updater receives no other
world-correlated side channel, conditional data processing gives

\[
I(\Theta;M_{t+1}\mid M_t,A_t)
\le
I(\Theta;O_{t+1}\mid M_t,A_t).
\]

No updater can retain more new world information than the outcome supplies.
An active learner can improve the right-hand side by choosing a more
informative legal experiment, but must charge its actions, latency, risk, and
reset requirements.

Across an adaptive lifetime, use the complete transcript

\[
H_t=(M_0,A_{0:t-1},O_{1:t}).
\]

If the policy chooses `A_t` from `H_t` with world-independent randomness and
final memory is a stochastic function of the transcript with independent
update randomness, then

\[
I(\Theta;M_T\mid M_0)
\le
I(\Theta;A_{0:T-1},O_{1:T}\mid M_0)
=
\sum_{t=0}^{T-1} I(\Theta;O_{t+1}\mid H_t,A_t).
\]

The equality follows by the chain rule because the policy contributes no fresh
world information conditional on `H_t`. Conditioning only on the complete
adaptive action sequence is invalid: later actions can reveal earlier outcomes
and condition away information. Any private world-correlated policy input must
be added to the transcript.

## 3. The retention bound

If the final finite digital persistent state has worst-case support size at
most `2^B`, then

\[
I(\Theta;M_T\mid M_0)\le B.
\]

Lossy state, forgetting, overwrite, state reset, or a discarded runtime lowers
the achieved value further. This distinguishes four objects often conflated in
LLM discussions:

- a KV or recurrent runtime state that may disappear at a request boundary;
- an episodic record that persists but is not generalized;
- a learned world state that predicts consequences; and
- slow parameters that contain reusable population-level regularities.

Continuous vectors require a precision or code-length contract, and
variable-length states require expected code-length accounting. Unbounded
independent innovation entropy requires growing storage somewhere.
It need not increase active model parameters or accelerator memory, but it
cannot be free.

## 4. The utilization bound

Let common side information be `S=(X,G)`, and let the later answer or action be

\[
Y=D(M_T,S,R').
\]

If the actor has no fresh world-correlated input beyond `(M_T,S)`, data
processing yields

\[
I(\Theta;Y\mid S)
\le
I(\Theta;M_T\mid S).
\]

If routing, retrieval, a stale actor, or an answer policy ignores the retained
distinction, its decision value is zero even though memory probes succeed.
This is why "the world model remembers" and "the agent still performs the
skill" are different claims.

For a decision label `D_Theta` uniform over `N` alternatives conditional on
`S`, decoded as `\widehat D` from `(Y,S)`, Fano's inequality supplies a concrete
consequence:

\[
P(\widehat D\ne D_\Theta)
\ge
1-\frac{I(D_\Theta;Y\mid S)+\log 2}{\log N}.
\]

Logs use one base throughout. Exact world identification is unnecessarily
strong when many worlds share the same correct decision.

## 5. Learning throughput

Define realized lifetime information quantities over a declared distribution
and common side information `S`:

\[
J_{acq}=I(\Theta;H_T\mid S),
\]

\[
J_{ret}=I(\Theta;M_T\mid S),
\qquad
J_{use}=I(\Theta;Y\mid S).
\]

They obey

\[
J_{use}\le J_{ret}\le J_{acq}
\]

when `Theta -> H_T -> M_T -> Y` is a conditional Markov chain given `S`. These
are realized mutual informations, not channel capacities. A
practical intelligence system is not defined by a large memory score or a
strong one-shot reasoner alone; it needs high **end-to-end useful information
throughput per unit lifetime cost**.

This is a diagnostic invariant, not a new model. Mutual information is also not
directly equal to task value; a task-weighted regret or value-of-information
measure is required for decisions.

## 6. Exact evidence-access separation

Let `Theta` be uniform over `N=2^n` possible worlds and independent of a
no-evidence model's weights and context. A model receiving no world-dependent
observation has best exact single-guess identification probability `1/N`.

A learner can request `n` noiseless truthful binary answers in a known encoding,
retain them, and identify the world exactly. Its neural parameter count need not
grow with `N`; it pays `n` evidence bits and `n` persistent bits.

This proves only that a system receiving informative evidence can beat one
receiving none, regardless of parameter count. The queries are nonadaptive, a
static learner given the same answer batch also succeeds, and the question
family and decoder are supplied. A genuine interaction advantage requires
matched evidence and action costs plus an environment where adaptive experiments
beat every passive design.

## 7. What the 2026 evidence says

Different current systems repair different links:

- Persistent Computational State shows that some apparent world-state failure
  is a serving/runtime reset, not absent latent dynamics.
- Current World Models Lack a Persistent State Core shows a separate model
  failure: unseen events often do not continue evolving consistently.
- Agentic Test-Time Training updates live weights, but its reported gains mainly
  preserve existing competence and cost about `1.9x` serving.
- Dream Rehearsal finds this separation in its specific DreamerV3,
  never-clear-replay protocol with three seeds: the utilization channel fails
  after its measured retention probes work.
- Fast-Slow Training, Continual Harness, ALMA, and dynamic predicate invention
  demonstrate useful continual adaptation, memory-design search, or online
  causal repair, but already occupy generic fast/slow, editable-harness, and
  symbolic-world-model designs.
- AgentOdyssey already diagnoses acquisition, episodic memory, exploration,
  planning, and cost separately; it is the closest systems-level collision with
  the three-channel framing.

Primary references:

- https://arxiv.org/abs/2607.21686
- https://arxiv.org/abs/2606.20545
- https://arxiv.org/abs/2607.03441
- https://arxiv.org/abs/2607.19749
- https://arxiv.org/abs/2605.12484
- https://arxiv.org/abs/2605.09998
- https://arxiv.org/abs/2602.07755
- https://arxiv.org/abs/2602.17217
- https://arxiv.org/abs/2606.24893

## 8. Consequence for the research target

The missing object is not simply a bigger context, writable memory, world
model, or actor. It is a trained lifetime loop that simultaneously:

1. chooses affordable actions that expose decision-relevant uncertainty;
2. converts their outcomes into a persistent, revisable state without
   catastrophic interference; and
3. guarantees that retained evidence changes later prediction, planning, and
   behavior when relevant.

The generic typed operator is

```text
(persistent state, goal, uncertainty)
    -> legal experiment / action
    -> external consequence
    -> bounded evidence-grounded update
    -> later policy conditioned on the update
```

This operator is still generic meta-RL / active world-model learning. T60 does
not provide a representation or update rule and therefore has no distinct
updater claim.

## 9. Admission gate

No experiment is earned by the bounds alone. The next candidate must identify
one trainable bottleneck that current systems leave weak and predict a
substantial end-to-end gain, not a memory-probe gain. It must declare:

- the unknown family `Theta` and real source of new evidence;
- the update state and bit/storage budget;
- the action/acquisition cost and legal intervention model;
- the channel by which retained information changes later decisions; and
- a matched static, recurrent/meta-RL, retrieval, fast/slow, and world-model
  control.

Only a theorem-level separation or a concrete operator surviving independent
audit earns a small local falsifier.

The independent audit gives the exact adaptive-information, storage, Fano, and
literature corrections:

- [T60 independent audit](epistemic-learning-throughput-t60-independent-audit.md)
