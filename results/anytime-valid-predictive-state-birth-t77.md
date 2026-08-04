# T77 anytime-valid predictive state birth — invent structure, then make it earn belief

Date: 2026-08-02  
Status: **INDEPENDENTLY AUDITED; RETAINED AS A PREDICTIVE-ADMISSION PROTOCOL; STRUCTURAL-BIRTH MECHANISM REJECTED; NO CODE OR RUN**

## 0. Research target

T75 validates active belief-guided evidence acquisition in a supplied finite
hypothesis class. T76 shows that naming hypothesis construction, updating,
composition, and consolidation does not create a mechanism. The next missing
operation is narrower and harder:

> When the current representation cannot predict experience, invent a new
> executable state variable or mechanism and admit it without confusing a
> plausible story with evidence.

T77 freezes one concrete **admission protocol**. A neural or language model may
propose candidate programs from raw history. A candidate becomes eligible for
durable state only when prospective, action-conditioned outcomes make an
anytime-valid predictive score large enough to pay for its prefix-code length
and its lifetime testing allocation. An accepted predictor must then survive
transcript deletion, transfer to new bindings, and produce selective behavior
under lesion/revision.

The proposal model supplies search. The environment supplies evidence. The
statistical gate, not the model's prose or confidence, decides predictive
support. It does **not** decide that a candidate is a true causal variable,
minimal ontology, semantic novelty, or evidence that the proposer itself became
more intelligent.

## 1. Executable candidate class

Let `h_t` be the complete legal action-observation history. The deployed system
has a current predictive state/model `M_0` with conditional outcome law

\[
P_0(y_t\mid h_t,a_t).
\]

A candidate structural edit `g` is a finite executable program that declares:

1. a bounded new state `z_t=g(h_t)` or recursively updated `z_{t+1}=g(z_t,o_t,a_t)`;
2. typed inputs, outputs, precision, initialization, and reset semantics;
3. an alternative normalized predictive law
   `Q_g(y_t|h_t,a_t)` that uses the new state;
4. the state/factors it replaces or augments;
5. an execution and persistent-bit cost; and
6. a prefix-free description of length `L(g)` bits.

The programs may represent a latent bit, counter, object binding, finite-state
split, local causal rule, typed arithmetic transformation, or composition of
previously accepted programs. Free-form prose is not an admitted state birth.
An LLM may write the program, but the program and its outcome distribution must
execute without asking the LLM what it meant.

The prefix code assigns prior weight

\[
w_g=2^{-L(g)},\qquad \sum_g w_g\le1.
\]

The serialized description must include executable syntax, constants,
quantized parameters, bindings, state layout and initialization, reset/update
semantics, hyperparameters that determine the law, parent/library versions,
interpreter/DSL/runtime versions, tables, external dependencies, and any
proposal artifact used at prediction time. Continuous point parameters require
a frozen quantization or prior mixture. All codebooks, interpreters, primitives,
numeric precision, runtime limits, and conditional parent references are
charged and frozen before scored outcomes. Choosing among DSLs is itself a
charged choice. A short ID that calls an uncharged model, data table, seed, or
proposal transcript is a side channel, not compression.

## 2. Timing, prediction, and validation

Let `F_t` contain all legal information after outcome `Y_t`, and let

\[
\mathcal G_t=\mathcal F_{t-1}\vee\sigma(A_t)
\]

contain the information available after choosing action `A_t` but before
observing `Y_t`. Randomized actions are allowed only when their fresh coins do
not reveal the current outcome, a future outcome, or hidden environment state.
The action and next predictive distribution must be `G_t`-measurable.

Under an exact simple null, write

\[
p_t(y)=P_0(Y_t=y\mid\mathcal G_t).
\]

For candidate `g`, let `q_{g,t}` be a normalized predictable outcome kernel and
assume conditional absolute continuity `Q_g << P_0`. Define

\[
R_{g,t}=\frac{q_{g,t}(Y_t)}{p_t(Y_t)},\qquad
\Lambda_{g,n}=\prod_{t=1}^{n}R_{g,t}.
\]

Then `E_P0[R_g,t | G_t]=1`, so `Lambda_g,n` is a nonnegative
`P_0`-martingale. Causal adaptive action selection is valid because the action
is conditioned on before the outcome. Outcome leakage, post-outcome action
recording, or an unmodeled policy-dependent simulator mode breaks the claim.

“Prospective” means that the factor used for `Y_t` was fixed before `Y_t`. It
does not require a physically separate environment. A fixed prequential
updater may learn from past outcomes when its next normalized kernel is chosen
before the next outcome.

Post-hoc clipping or replacing a zero-support denominator changes the null and
needs a new proof. A real protocol must instead predeclare a support floor,
contamination neighborhood, or uniformly valid composite-null construction.

## 3. T77.1 — family-wise anytime predictive-admission control

### Theorem T77.1 — fixed coded family on a shared stream

Let `{Q_g : g in C}` be a fixed countable family of fully specified predictable
kernels and let deterministic weights satisfy `w_g >= 0` and
`sum_g w_g <= 1`. Under the timing and support conditions above,

\[
M_n=\sum_{g\in\mathcal C}w_g\Lambda_{g,n}
\]

is a nonnegative supermartingale with `M_0 <= 1`. Therefore

\[
P_0\left(\exists n,\exists g:
w_g\Lambda_{g,n}\ge1/\alpha\right)
\le
P_0\left(\sup_n M_n\ge1/\alpha\right)
\le\alpha.
\]

The first event implies the second because the mixture contains every
nonnegative weighted process; the final inequality is Ville's inequality. For
`w_g=2^-L(g)`, Kraft's inequality supplies the weight bound only when the code
is genuinely fixed and prefix-free.

Candidates may share the same validation outcomes, and data-dependent or
retrospective selection among this already coded family is safe: the family has
already paid for the selection. The theorem controls rejection of the one
exact conditional distribution `P_0`. It does **not** control false declaration
of a causal variable, semantic novelty, minimal structure, or usefulness.

### Adaptively proposed or fitted candidates

If proposal `j` is created at stopping time `tau_j` from `F_tau_j`, it may start
a future-only process

\[
\Lambda^{post}_{j,n}
=\prod_{t=\tau_j+1}^{n}
\frac{q_{j,t}(Y_t)}{p_t(Y_t)}.
\]

It may share the physical stream with other candidates. It may not fit on an
outcome and retroactively score that outcome unless all realized alternatives
and their multiplicity were already paid for by a fixed code or prior mixture.

An open-ended proposal sequence receives predictable wealth `a_j >= 0` with
`sum_j a_j <= alpha` almost surely and accepts `j` only after
`Lambda_post_j,n >= 1/a_j`. If proposal `j` opens its own coded family, use
conditional weights `w_j,g` with `sum_g w_j,g <= 1` and threshold
`1/(a_j w_j,g)`. One hierarchical lifetime ledger must cover candidates,
versions, resets, task families, and every newly committed baseline. Restarting
an unfavorable process at wealth one is a new test, not optional stopping.

| candidate construction | shared future stream | scores fitting outcomes | valid route |
|---|---:|---:|---|
| fixed fully specified coded candidate | yes | yes | Kraft-weighted mixture |
| split-fit candidate, then frozen | yes | no | condition on fit split; score future data |
| candidate fitted at a stopping time | yes | no | start prospectively and spend new wealth |
| updater fixed in advance | yes | prequentially | each next factor is predictable |
| continuous point fit on the validation block | no naive reuse | no | quantized code, prior mixture, split, or separately proved e-process |

## 3.1 Exact-null limitation and operational repair

If the real conditional environment is `R != P_0`, a small persistent defect in
`P_0` can make an ordinary recalibration, nuisance feature, lookup table, or
observationally equivalent factorization cross any finite threshold with
probability approaching one. Exact-null rejection is therefore not a
structural-birth guarantee.

The preferred production-facing null is instead **no material predictive
advantage**. Let a bounded proper-score gain be

\[
X_t=s(P_0,Y_t)-s(Q_g,Y_t)\in[a,b]
\]

and test the conditional null
`E[X_t | G_t] <= delta`, where `delta` is a predeclared practically meaningful
margin. For fixed `lambda >= 0`, Hoeffding's lemma gives the null e-process

\[
E_n=\exp\left(
\lambda\sum_{t=1}^{n}(X_t-\delta)
-\frac{\lambda^2(b-a)^2n}{8}
\right).
\]

This remains an operational predictive-improvement test, not proof of causal
or semantic structure. A composite old-model class instead requires e-factors
uniformly safe over every member, a valid least-favorable construction, or
split universal inference against the supremum null likelihood. A fitted
denominator, generalized likelihood ratio, or Bayesian mixture is not
automatically uniformly valid.

## 4. T77.2 — evidence needed for a predictively better alternative

For one frozen candidate, define

\[
X_t=\log\frac{q_t(Y_t)}{p_t(Y_t)},\qquad
\mu_t=\mathbb E_Q[X_t\mid\mathcal G_t],\qquad
D_t=X_t-\mu_t.
\]

Assume the predictable pathwise drift `sum_t mu_t >= N d` almost surely (the
stronger per-step condition `mu_t >= d` is sufficient), and let the centered
martingale differences satisfy `|D_t| <= c` almost surely. An unconditional
mean-gain statement is not enough. Write

\[
T_g=L(g)\ln2+\ln(1/\alpha).
\]

### Proposition T77.2

After `N` prospective outcomes,

\[
P_Q(\log\Lambda_{g,N}<T_g)
\le
\exp\left(-\frac{(Nd-T_g)^2}{2Nc^2}\right)
\]

whenever `Nd>T_g`.

### Proof

The accumulated predictable conditional mean is at least `Nd`. Apply the
one-sided Azuma bound to `sum_t D_t`. QED.

Thus a sufficient `1-beta` detection condition is

\[
Nd-T_g\ge c\sqrt{2N\ln(1/\beta)}.
\]

Ignoring the confidence term, the evidence scale is

\[
N\approx
\frac{L(g)\ln2+\ln(1/\alpha)}{d}.
\]

An explicit sufficient integer horizon is

\[
N\ge\left\lceil
\left(
\frac{\kappa+\sqrt{\kappa^2+4dT_g}}{2d}
\right)^2
\right\rceil,
\qquad
\kappa=c\sqrt{2\ln(1/\beta)}.
\]

A short reusable mechanism with a large predictive KL advantage can earn
admission quickly; a verbose or weakly predictive story cannot. This is a
conditional detection bound. If the actual environment is `R`, then

\[
\mu_t=D_{KL}(R_t\|P_{0,t})-D_{KL}(R_t\|Q_{g,t}),
\]

so exact truth-matching is unnecessary; a pathwise positive predictive
advantage is enough. Fitting on earlier proposal data is allowed after
conditioning and freezing. Fitting on the current outcome breaks adaptedness;
prequential past-only updating remains allowed when declared in advance.

The displayed constant uses the exact convention `|X_t-mu_t| <= c`. If only
`|X_t| <= c` is known, centering gives the looser `|D_t| <= 2c`; a bounded-range
Hoeffding argument must be stated separately to recover sharper constants. Log
ratios are generally unbounded, so a real manifest must provide support floors,
tail/variance conditions, or a different concentration argument.

## 5. The active experiment rule

For a posterior or weighted set of proposed candidates, choose legal actions by
expected evidence growth per complete cost:

\[
a_t=\arg\max_a
\frac{\mathbb E[\Delta\log\Lambda\mid H_t,a]}
     {c_{world}(a)+c_{compute}(a)}.
\]

This is an implementable objective only when candidates predict action-
conditioned outcomes and the expectation names its distribution: null,
candidate, design mixture, or worst case. Candidate-only expectation can select
self-confirming actions. In T75's frozen model-free synthetic artifact, a
transparent constructive adaptive policy used 168 probes versus 1,280 for its
fixed-design control at the declared worst-world threshold; an exact-posterior
entropy-greedy heuristic measured 96.03% average accuracy at 45.40 mean probes
against a 36.69 expected-information floor. These are one synthetic theorem,
calculation, and CPU measurement—not inherited T77 performance.

T77 does not inherit those numbers. Its proposed structure, action catalog,
and predictions may be wrong. The reference controls are exact Bayesian or
information-gain experiment design over the same candidate class, random/legal
action schedules, and the strongest recurrent/meta-RL query policy.

## 6. Structural commitment transaction

Crossing the e-value threshold moves a candidate from `tentative` to
`predictively supported relative to the declared null`, not directly to
structurally or causally supported and not to unrestricted control. A commit is
a versioned transaction:

1. freeze the accepted code and its validation evidence hash;
2. execute it in a shadow state;
3. test held-out predictive/action competence;
4. test protected behaviors and dependency slices;
5. serialize only the declared program, parameters, state, provenance, and
   version parent;
6. destroy proposal/validation transcript, KV cache, scratch files, and
   undeclared model state;
7. start a fresh process and reconstruct from the bounded serialized state;
8. require competence on new bindings and tasks that need `g`; and
9. commit or roll back atomically.

The exact-simple-null e-process protects against rejecting `P_0` when `P_0` is
the true declared law; the bounded-score version protects against accepting an
advantage below its declared material margin. Neither protects structural or
causal interpretation. Private competence and retention tests protect
different properties. Passing one cannot compensate for failing another.

## 7. Revision, retirement, and local responsibility

An accepted program may later become wrong. Never overwrite it silently.
Create a candidate replacement `g'` with a new version and compare its
action-conditioned predictive law with the current version using fresh
evidence. The old version remains available for historical queries and
rollback.

Local revision is claimed only if:

- operator lesion removes the predicted competence;
- swapping/rebinding operator state swaps the predicted behavior;
- the new evidence raises the replacement's likelihood on the affected slice;
- protected slices outside the complete dependency set remain noninferior; and
- a generic dense or textual update given the same evidence does worse under
  the complete ledger.

This is functional locality, not merely a sparse file or parameter edit.

## 8. Candidate discovery remains the central risk

T77 controls when a proposed predictor earns declared statistical support. It
does not guarantee that the proposal model invents the right variable.

### Proposition T77.3 — proposal coverage boundary

If every candidate ever proposed assigns the same outcome law as `P_0` on all
legal validation actions, then every likelihood ratio is identically one and
no candidate with `alpha<1` can be admitted. More generally, if the true useful
program is outside the closure of the candidate DSL/proposal support, the
statistical gate cannot create it.

The proposal mechanism must therefore be evaluated separately on:

- coverage of known hidden structures under blinded raw interfaces;
- code length and execution cost of its best valid proposal;
- proposals, retries, rejected candidates, and validation interactions;
- invention of a variable not nameable by the starting ontology;
- reuse in an independently generated domain; and
- performance against enumerative, stochastic-program, ILP/predicate-
  invention, neural state-splitting, and unrestricted LLM-code baselines.

An exact MDL/Bayesian search over the same DSL is the algorithmic control. A
learned proposer can win only by amortizing search compute or finding useful
candidates within a practical budget; it cannot beat the exact posterior on
information.

## 9. Anti-prose, anti-generator, and anti-pretraining gates

The first admissible learned test must include:

1. equal-byte evolving text/playbook memory;
2. full context and retrieval of raw episodes;
3. generic recurrent state trained on the identical lifetime objective;
4. probabilistic modular task inference;
5. ADVENT-style LLM predicate invention plus exact logical verification;
6. truth-maintained/transactional belief memory;
7. explicit MDL/Bayesian program search;
8. test-time gradients and self-consolidation; and
9. a larger static/agentic-search system on the same serving frontier.

Equal bytes alone are not equal information or work. Every code/program control
must additionally charge compressed seeds, lookup content, interpreters,
external calls, execution, retrieval, and proposal compute.

The test mechanism is sampled after every model, prompt, codebook, and gate is
frozen. At least one held-out domain is written by an independent generator and
one is naturalistic rather than emitted by the synthetic DSL. Surface names,
ordering, serialization, and bindings are counterfactually rerendered. No
ground-truth latent, predicate name, factor ID, or causal parent enters the
candidate interface.

## 10. Three-family theorem-plus-manifest requirement

If this protocol is ever reopened, define three non-isomorphic candidate
classes that use the same T77 commit rule but different executable semantics:

1. **state birth:** an aliased controlled process needs a new finite predictive
   state variable;
2. **mechanism birth:** a sparse action-conditioned causal/transition factor is
   absent from the current model; and
3. **procedure birth:** a reusable typed program compresses and solves a family
   of tool or graph transitions.

Here “non-isomorphic” is not a visual or naming judgment. Two evaluation
families count as isomorphic when cost-preserving bijections of their legal
histories, actions, outcomes, scores, candidate codes, and artifact execution
reduce one learner problem to the other. Each manifest must name an invariant
that prevents such a reduction and must show that no one latent DSL switch or
family-specific wrapper solves all three.

For each family, freeze:

- legal raw observation/action interface and complete side-channel boundary;
- candidate DSL and prefix code;
- `P_0`, alternative fitting split, e-process, `alpha`, and stopping rule;
- information/identifiability and detection lower/upper bounds;
- state bits, execution work, proposal work, intervention cost, and rollback;
- component, compound, independent-generator, and natural transfer splits;
- fresh-process transcript-purge proof;
- lesion/swap/rebinding and hidden-change tests;
- all current-art controls and equal tuning budget; and
- Level 1 synthetic, Level 2 cross-domain, and Level 3 complete-system gates.

The same learned proposer/updater and codebook must cross all three families.
Family-specific parsers, prompts, adapters, or DSL extensions are architecture
changes and cannot be hidden inside evaluation. Passing these families can
establish scoped breadth for one proposer; it cannot alone establish a
substantial increase in general intelligence.

## 11. Current-art and novelty boundary

The ingredients are established:

- [Safe Testing](https://arxiv.org/abs/1906.07801),
  [e-values](https://arxiv.org/abs/1912.06116), time-uniform concentration,
  universal inference, MDL, and Bayesian structure selection supply the
  statistical mathematics;
- active causal discovery and experimental design supply query policies;
- automata/HMM/PSR state discovery supplies predictive-state splitting;
- online predicate invention, causal representation learning, and executable
  world-model repair supply structural candidates;
- [ADVENT](https://arxiv.org/abs/2607.01585) already combines LLM abductive
  predicate invention, Prolog verification, iterative execution feedback, and a
  reusable knowledge pool, reporting up to 31-point gains;
- [MemTX](https://arxiv.org/abs/2607.23929) already provides transactional
  belief commit and cascading repair;
- [BeliefMem](https://arxiv.org/abs/2605.05583) already retains probabilistic
  alternatives; and
- compositional probabilistic meta-learning already tracks module-program
  hypotheses under sparse feedback;
- [PatchWorld](https://arxiv.org/abs/2605.30880) already induces executable
  symbolic belief-state programs and applies counterexample-guided local code
  repair in partially observed agent environments; and
- [Recursive Concept Evolution](https://arxiv.org/abs/2602.15725) directly
  claims inadequacy-triggered, MDL-gated low-rank concept spawning and reported
  benchmark gains. Those are paper claims, not independently validated facts;
  its stated constant-gate covariance-rank proposition is false because
  `I+gP` is invertible for `g>0`, so conjugation preserves covariance rank.

T77 does not claim new statistical theory, executable belief-state induction,
counterexample repair, or concept spawning. Its residual contribution is only
a possible systems conjunction: executable bounded predictors, prospective
anytime-valid complexity-weighted admission, causal evidence-seeking actions,
fresh-process persistence, and cross-family lesion/local-revision tests. The
current document specifies part of that conjunction; it does not demonstrate
that the proposer invents useful structure or that a complete model improves.

## 12. Audited disposition

The independent audit accepts the fixed-family martingale/Kraft kernel and the
Azuma power constant under the corrected assumptions. It rejects T77 as a
structural-birth mechanism and as evidence of model-level intelligence:

- exact-null rejection is not structural discovery;
- adaptive proposal streams need one explicit lifetime wealth ledger;
- operational code lengths must include every fitted and executable object;
- predictive evidence cannot establish causal ontology, semantic novelty, or
  proposer intelligence; and
- RCE, PatchWorld, program synthesis, predictive-state discovery, MDL, and
  e-processes occupy the individual ingredients.

Retain T77 as a predictive-admission and research-governance protocol. The next
mechanism must improve **proposal coverage and useful abstraction formation**;
it cannot take credit for this verifier. Reopening T77 implementation would
require a second independent theorem audit of a complete manifest with a
composite or material-tolerance null, canonical code, lifetime wealth ledger,
side-channel boundary, same-class refit control, and full system-level causal
mediation tests.

No CPU experiment, neural training, local GPU use, or rental is admitted. See
the [independent audit](anytime-valid-predictive-state-birth-t77-independent-audit.md).
