# T67 evidence-carrying learning — train the update operator, not only the predictor

Date: 2026-08-01  
Status: **BROAD INTELLIGENCE-MECHANISM CLAIM REJECTED; TRANSACTIONAL EVIDENCE-GOVERNANCE CONTRACT RETAINED; NO RUN**

## 0. Corrected target

The research target is not a cheaper transformer block. It is a model that
becomes more capable from experience without requiring a larger serving model:

- it can acquire useful knowledge after deployment;
- it can revise a wrong belief without corrupting unrelated competence;
- it can improve reasoning rather than merely rerank samples; and
- it can distinguish evidence from its own generated prose.

The central hypothesis is:

> Modern models optimize the predictor `f_theta` far more than the state-update
> operator `U`. Real continual intelligence requires a learned, evidence-grounded
> update operator that decides what an experience may change, produces a
> checkable state transition, and preserves the evidence needed to reverse it.

Write the lifetime system as

\[
S_{t+1}=U_\phi(S_t,e_t),\qquad y_t=f_\theta(x_t,S_t).
\]

Here `S_t` is not a flat prompt. It contains raw episodes, tentative hypotheses,
certified beliefs or mechanisms, their dependencies, and version history. Most
deployed LLMs have a strong `f_theta` and either no `U_phi`, a hand-written
memory rule, or another unconstrained LLM call that rewrites text. T67 asks
whether training `U_phi` as a first-class model component can produce large
lifetime capability gains.

This formulation changes the earlier fixed-compute question. Extra computation
while learning is allowed. The required advantage is a substantially smarter
model or model-system at matched normal serving cost, with the complete
learning bill reported.

## 1. The state that prose-only models are missing

Use four logically distinct stores:

\[
S_t=(E_t,H_t,C_t,D_t).
\]

- `E_t`: immutable raw observations, actions, outcomes, and provenance;
- `H_t`: tentative and competing hypotheses;
- `C_t`: beliefs or executable mechanisms admitted for normal use; and
- `D_t`: the dependency and version graph relating evidence, hypotheses,
  derived claims, model modules, and predictions.

An experience should not directly overwrite `C_t`. The update model proposes a
typed transaction

\[
\Delta_t=(\text{edit},\text{witness},\text{affected slice},
\text{predictions},\text{tests},\text{rollback parent}).
\]

A deterministic executor applies the edit to a shadow state. A verification
stack checks executable constraints, private held-out outcomes, provenance,
and protected predictions. Accepted transactions obtain a certificate and
become a new content-addressed version. Rejected or presently unverifiable
claims remain in `H_t`; their raw evidence is not destroyed.

The architecture therefore separates three operations that a free-form LLM
memory rewrite conflates:

1. proposing what an experience might mean;
2. testing whether the proposed change is supported; and
3. making the change durable and available to future reasoning.

## 2. What would make this a model improvement rather than a wrapper

There are three progressively stronger outcomes:

1. **Selection only.** A verifier chooses among answers. This can improve a
   served system but does not improve the generator.
2. **Persistent state learning.** Certified updates improve future tasks without
   replaying the entire history. This is a continual-learning improvement to the
   model-system even if base weights are frozen.
3. **Absorbed capability.** Training inside the evidence/revision loop improves
   the generator or updater when the verifier is removed at inference. This is
   a direct model-level improvement.

T67 is admitted only if it reaches outcome 2 and, on verifiable reasoning
domains, outcome 3. A larger test-time search budget alone does not pass.

## 3. Exact selectivity boundaries

### Proposition T67.1 — revision threshold

Let the current answer or proposed belief be wrong with probability `e`. A
verifier misses a wrong item with probability `beta` and rejects a correct item
with probability `alpha`. Conditional on revising a detected wrong item, the
revision fixes it with probability `q`. Conditional on revising a correct item,
the revision corrupts it with probability `c`. Under these stated outcomes,

\[
e_{new}=e[1-(1-\beta)q]+(1-e)\alpha c.
\]

Therefore verification and revision improve error exactly when

\[
e(1-\beta)q>(1-e)\alpha c.
\]

**Proof.** A wrong item remains wrong unless it is detected and successfully
fixed, giving the first term. A correct item becomes wrong only when falsely
flagged and then corrupted, giving the second. Subtract from `e`. QED.

The posterior form is more useful. Given verifier signal `Z=z`, let
`p_z=P(wrong | Z=z)`. Revision has expected error reduction

\[
p_z q-(1-p_z)c.
\]

Assume `q+c>0` and that the fix/corruption probabilities are constant across
signals. The Bayes-optimal rule revises exactly when

\[
p_z>\frac{c}{q+c}.
\]

If revision quality depends on the signal or proposed revision, use the more
general condition `p_z q_z>(1-p_z)c_z`.

If `Z` contains no information about correctness, `p_z=e` for every signal.
The verifier then cannot selectively improve on the best unconditional
revision decision.

### Proposition T67.2 — admitted-update quality

Let a proposed update be beneficial with prior probability `p`. A gate accepts
a beneficial update with probability `a` and a damaging update with probability
`b`. Assume these classes are exhaustive and `pa+(1-p)b>0`. Among accepted
updates,

\[
P(\text{damage}\mid\text{accept})=
\frac{(1-p)b}{pa+(1-p)b}.
\]

Define the acceptance-conditional magnitudes

\[
G=E[\Delta\mid\text{beneficial},\text{accept}],\qquad
C=-E[\Delta\mid\text{damaging},\text{accept}].
\]

The expected net gain per proposal is

\[
paG-(1-p)bC.
\]

When `p,G,C,b>0`, it is positive exactly when

\[
\frac{a}{b}>\frac{(1-p)C}{pG},
\]

with the natural infinite-ratio interpretation when `b=0`. Persistent damage
raises `C`; it makes verifier false acceptance much more expensive than false
rejection. Accuracy or AUC alone is therefore not the right gate. The verifier
must be evaluated at the operating point induced by the damage/gain ratio.

### Proposition T67.3 — value of an informative gate

For a proposed transition, let integrable random variable `Delta` be its future
gain excluding the cost of obtaining verifier evidence, and let `Z` be that
evidence. A selective policy can accept when `E[Delta | Z]>0`. Its
free-information decision value is

\[
V_Z=E[\max(0,E[\Delta\mid Z])].
\]

Because `max(0,x)` is convex, Jensen's inequality gives

\[
V_Z\geq \max(0,E[\Delta]).
\]

The right side is the value of the best accept-all/reject-all policy. The
inequality is strict exactly when the posterior expected gain is positive and
negative on sets of positive probability. Thus evidence strictly improves the
free-information decision value only when it changes decisions.

If obtaining `Z` costs random amount `kappa(Z)`, its net value is

\[
V_Z^{net}=E[\max(0,E[\Delta\mid Z])]-E[\kappa(Z)].
\]

Paid verification is useful only when this exceeds
`max(0,E[Delta])`. Jensen's inequality alone proves only that free information
can be ignored; it does not establish an economic advantage.

This is a control theorem, not an intelligence theorem. T67 still needs a
mechanism that obtains informative `Z` more cheaply than recomputing the answer
or future itself.

## 4. What a certificate can and cannot guarantee

### Proposition T67.4 — private probe certificate

For `0<gamma<1`, call an update `gamma`-bad on a frozen probe distribution if it fails a random
probe with probability at least `gamma`. Suppose a candidate is fixed before
seeing `k` independent private probes. Its probability of passing all probes is
at most

\[
(1-\gamma)^k\leq e^{-\gamma k}.
\]

Let `N` bound all candidates attempted over the declared lifetime. Candidates
may be proposed adaptively, but each candidate must be fixed before its own
fresh probes and its `gamma`-bad guarantee must hold conditional on the complete
prior history. A union bound makes the probability that any `gamma`-bad
candidate passes at most

\[
N e^{-\gamma k}.
\]

The exact common-`k` requirement is

\[
k\geq\frac{\ln(N/\delta)}{-\ln(1-\gamma)}.
\]

Using `-ln(1-gamma)>=gamma`, a simpler conservative sufficient condition is

\[
k\geq\frac{\ln(N/\delta)}{\gamma}.
\]

This guarantee disappears if the proposal sees or adaptively overfits the
same tests, if probes are dependent, or if the future distribution differs.
It certifies behavior on a declared distribution, not open-world truth.

### Exact dependency certificate

If every result is a referentially transparent function of immutable nodes in
a complete dependency DAG, then changing nodes outside an output's transitive
dependency set cannot change that output. This is a software noninterference
statement. It applies naturally to external executable beliefs or isolated
modules. It does not apply to a dense shared neural-weight update unless the
architecture enforces the claimed isolation.

### Repeated generation

If a candidate is correct with probability `p`, a correct candidate is
accepted with probability `a`, and a wrong candidate is never accepted, the
probability of no accepted-correct result after `K` independent attempts is

\[
(1-pa)^K,
\]

and the expected attempts to acceptance are `1/(pa)`. With a nonzero
false-accept rate, first-accept sampling has the admitted error from T67.2;
repetition alone does not improve it.

## 5. Proposed mechanism: Epistemic Transition Model

The concrete candidate is an **Epistemic Transition Model (ETM)** rather than a
generic critic.

### Runtime learning cycle

1. Append the raw action/observation/outcome to immutable episodic storage.
2. Generate one or more typed state-transition proposals, not free-form memory
   summaries.
3. Execute each proposal in a shadow version of the belief/mechanism graph.
4. Produce discriminating predictions and an affected dependency slice.
5. Check hard invariants, provenance, protected queries, and fresh environment
   outcomes; use a learned verifier only for residual semantic judgments.
6. Admit the best positive-value transition or keep all proposals tentative.
7. Train the proposal/update policy from transition-level outcomes. Optionally
   distill repeatedly certified mechanisms into isolated adapters or the base
   model during a slow offline phase.

### Privileged-hindsight verifier training

At training time the verifier may observe information that the deployed updater
does not yet have: future outcomes, reference solutions, interventions, or the
full raw history. It learns to predict whether a transition preserves relevant
facts and improves future behavior. The deployed updater then learns from the
verifier's localized feedback.

The asymmetry is legitimate only if the privileged signal is external to the
suspect proposal. A model's own confidence, paraphrase, or second sample is not
independent evidence merely because it is placed in another prompt role.
External privileged training information does not itself make a deployed
verifier independently checkable. Runtime certificate claims still require a
sound deterministic checker or an external outcome channel.

### Cross-transition hypothesis

The potentially new research seam is not `use a verifier`. It is this stronger
hypothesis:

> Train one updater on some transition families; freeze its parameters and
> transition schema; then test evidence-preserving revision on a held-out family
> without family-specific updater or verifier training. The frozen updater must
> also improve future first-attempt behavior without a verifier call.

If each domain needs a separate hand-written verifier, transition schema, and
training run, T67 reduces to a useful collection of task-specific systems and
is not a general intelligence mechanism.

## 6. Direct collisions with current work

The broad components are already occupied:

- **TrustMem** already evaluates each memory transition for coverage,
  preservation, and faithfulness, then trains the memory updater with
  transition-ranked GRPO. It reports a 12.14-point HaluMem extraction gain and
  40.1%, 79.1%, and 50.0% reductions in omission, corruption, and hallucination.
  Its verifier is a frozen LLM, not a sound certificate, and its state is text
  memory rather than an executable belief/dependency model.
  <https://arxiv.org/abs/2606.25161>
- **Useful Memories Become Faulty When Continuously Updated by LLMs** shows why
  evidence must remain first-class: repeated consolidation can fall below no
  memory, while episodic-only memory remains competitive.
  <https://arxiv.org/abs/2605.12978>
- **Supersede** isolates a memory-update policy gap. Full context scores 92%
  versus 77% for bounded self-memory on a frontier model; training a 3B updater
  moves held-out accuracy from 9.0% to 16.7%, although only in one run.
  <https://arxiv.org/abs/2606.27472>
- **Models That Prove Their Own Correctness** trains models to interact with a
  manually specified, sound verifier. The paper's experiment is a 6.3M model
  on GCD and its stated limitation is a separate model per capability.
  <https://arxiv.org/abs/2405.15722>
- **Self-Trained Verification** already trains a verifier from privileged
  reference solutions, then trains the generator inside the verifier/revision
  loop. It reports roughly 30% relative standalone pass@1 gain after ordinary
  RLVR plateaued, a 1.7B/4B weak-to-strong verifier result, and much larger
  multi-round gains. Its stated open problems include larger models, code,
  open-ended reasoning, and the compute-optimal verifier/generator split.
  <https://arxiv.org/abs/2605.30290>
- **Verifier-driven test-time training** already selects verifier-approved
  pseudo-labels and updates LoRA parameters, reporting up to 32.29% relative
  gain over base and 6.66% over verifier-only methods.
  <https://arxiv.org/abs/2505.19475>
- **MemoPilot** explicitly treats memory updating as a trainable multi-turn
  decision process and optimizes it end to end with GRPO. It substantially
  improves a frozen player's test-time learning in repeated games, although it
  trains separate memory models for Rock-Paper-Scissors and poker.
  <https://arxiv.org/abs/2606.08656>
- **Kumiho** already combines immutable revisions, typed dependency edges,
  graph-native belief state, safety-guarded consolidation, and formal AGM-style
  belief-revision semantics. Its formal guarantees cover individual graph
  operations, not truth of LLM-generated content or composed consolidations.
  <https://arxiv.org/abs/2603.17244>
- **AgentCL** already evaluates reusable experience and transfer across coding,
  research, and language/reasoning streams, and finds that naive streams often
  hide memory-induced degradation.
  <https://arxiv.org/abs/2606.02461>

Consequently, T67 cannot claim novelty for verifier-guided revision,
proof-carrying answers, gated consolidation, or verifier-trained generators.
Even the trainable update operator and versioned graph state are therefore
occupied. Cross-transition transfer and a genuinely independent certificate
spanning textual memory, executable world-model factors, and parametric edits
are not demonstrated by the cited work. Integration by itself is not a
breakthrough.

## 7. Complete bill and failure modes

The bill includes:

- proposal sampling and shadow execution;
- raw episodic retention and indexes;
- verifier training, inference, and private probe generation;
- environment interactions and resets;
- protected-query replay and dependency maintenance;
- slow consolidation or adapter training; and
- rejected proposals and rollbacks.

The main fatal risks are:

1. **Verifier circularity.** A learned verifier shares the generator's blind
   spots, so accepted errors become more convincing rather than less frequent.
2. **Coverage collapse.** Strict gating rejects novel but correct updates and
   preserves a conservative, stagnant model.
3. **Certificate Goodharting.** The updater learns the declared tests rather
   than the underlying mechanism.
4. **Non-modular weights.** Distillation destroys the dependency and rollback
   guarantees of the external state.
5. **Task-specific occupation.** Success depends on a bespoke answer key or
   checker for every domain and does not transfer as an update skill.

## 8. Substantial-improvement gate

An eventual experiment must compare matched-size, matched-serving-cost systems
on streams containing all of the following:

- changing temporal facts and retractions;
- latent executable rules learned from examples;
- intervention-dependent causal mechanisms; and
- misleading observations or generated false explanations.

Controls must include full context, episodic-only retrieval, forced
consolidation, a TrustMem-style transition scorer, a Supersede-style updater,
verifier-only answer selection, STV/ViL-style diagnostic-feedback training,
verifier-driven test-time training, and extra ordinary training at matched
learning compute.

T67 passes only if replicated runs show:

1. at least 30% relative lifetime-error reduction over the strongest control
   across at least three transition families and two model sizes;
2. at least 2x fewer damaging durable commits at comparable useful-update
   recall;
3. no more than 2% absolute protected-capability regression;
4. at least 20% relative improvement in update-proposal quality on new
   experiences with no verifier call, beyond matched additional ordinary
   training;
5. separately, improved downstream answering with the verifier off and
   previously certified state available—reported as system learning, not
   absorbed updater capability; and
6. no more than 5% normal serving latency/VRAM increase, with all learning-time
   compute, storage, and environment calls reported separately.

These thresholds are research gates, not predictions.

## 9. Local falsifier before a rental

No local run is admitted for the current broad claim. If reopened around
leave-one-transition-family-out transfer, the decisive local falsifier should
use one small open model and a fixed adapter budget on three executable streams:
temporal supersession, induced rule revision, and small causal mechanism
changes. It must not merely rediscover that verifiers can rerank answers.

Before any multi-hour GPU run, the local stage must establish all of:

- the verifier's posterior crosses the T67.1 decision threshold on held-out
  proposals;
- verifier selectivity predicts real future gain rather than its own score;
- transition feedback improves unseen first attempts after the verifier is
  removed;
- one update schema transfers across the three streams; and
- episodic-only and matched extra-training controls are beaten by a large
  margin, not by less than one percentage point.

If any condition fails, close T67 without rental. A GPU is not currently
justified.

## 10. Current verdict

The missing capability is better described as **governed model-state change**
than as memory, reflection, or more chain of thought. Current models can propose
beliefs and revisions; they are weak at deciding which proposal may safely
become part of the machinery that generates future answers.

T67.1 and T67.4 are valid under their stated sampling models. T67.2 requires
acceptance-conditional gain and damage magnitudes; T67.3 is a free-information
bound that does not itself charge verification. The surrounding empirical
results show that verifier-conditioned training can cause substantial
model-level gains, but the broad method is already established. Independent audit
found that T67's only defensible retained object is a transactional protocol in
which evidence stays attached, tentative state is causally quarantined, commits
are atomic and versioned, and an actually independent checker admits or revokes
them. That is useful evidence governance, not yet a distinct intelligence
mechanism. T67 is closed without an experiment.
