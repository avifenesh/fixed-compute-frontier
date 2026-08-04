# T65 intelligence-system architecture — learn factors once, execute and repair them repeatedly

Date: 2026-08-01  
Status: **INDEPENDENTLY REVISED RESEARCH HYPOTHESIS; BOUNDED-WALK AND CONDITIONAL NONINTERFERENCE LEMMAS RETAINED; NO RUN**

## 0. Research correction

The target is no longer a distinct function class or a cheaper Transformer. A
method qualifies if it produces a large, reproducible capability gain under a
complete cost accounting, even when a generic recurrent network could emulate
it with ideal weights. Finite training, conditioning, systematic
generalization, continual adaptation, and useful inductive bias are legitimate
sources of improvement.

The operational target is an agent that becomes better through experience. It
must discover a compact executable model, use that model beyond the depth and
surface forms seen in training, actively resolve important uncertainty, repair
contradicted knowledge, and retain unaffected competence.

This supersedes the old rule that containment in a generic recurrent control is
automatically fatal. It does not relax the empirical bar: a method must change
what the resulting model can reliably do, not merely move work into an
uncharged compiler, planner, memory, simulator, or data generator.

## 1. Unclosed production-scale loop, not missing component

Modern systems separately demonstrate language priors, world prediction,
memory, test-time adaptation, planning, causal discovery, program induction,
and verification. VisualPredicator, Pixels-to-Predicates, online predicate
invention, and ESBM additionally close much of the following loop in restricted
robotic, symbolic, or Atari-style settings. What is not yet demonstrated in a
substantial production model, without a privileged ontology or uncharged
components, is the complete loop

\[
\text{experience}
\longrightarrow \text{grounded factors}
\longrightarrow \text{executable consequences}
\longrightarrow \text{informative action}
\longrightarrow \text{localized revision}.
\]

Plain next-token training does not require this loop. Let a training example
reset durable state to `s_0` and optimize only

\[
J(\theta)=\mathbb E[-\log p_\theta(x_{t+1}\mid x_{\le t},s_0)].
\]

An update to a persistent state after the example has zero causal effect on
this loss if no later example reads it. Sequence prediction can still learn
world knowledge, algorithms, and in-context adaptation; the point is narrower:
the reset objective supplies no direct credit for durable, selectively
revisable learning from deployment experience.

Likewise, passive prediction cannot universally identify intervention laws.
T46 and T53 give pairs of worlds with identical passive histories and different
post-intervention outcomes. No architecture can recover information absent
from its channel. Action, environment variation, side information, or a causal
assumption must be paid for.

## 2. Candidate: an Experience-to-Executable Factor Machine

The candidate is a hybrid system with four learned interfaces and one explicit
executor.

### 2.1 Grounder

From raw multimodal history `h_t`, the grounder produces a distribution over
entities, state variables, bindings, and local transition factors:

\[
q_\phi(Z_t,G_t,\Theta_t\mid h_t).
\]

It is trained across interventions and multiple environments with randomized
surfaces. Predicting action-conditioned futures supplies the behavioral
target; cross-environment invariance discourages storing surface nuisance.
Uncertainty is retained rather than collapsed to one prose explanation.

### 2.2 Persistent factor memory

Knowledge is stored as versioned factors

\[
F_i(z_i,z_{Pa(i)},a;\theta_i),\qquad i=1,\ldots,d,
\]

plus bindings, provenance, uncertainty, and dependency metadata. Reusable
factor templates are shared across environments; environment-specific state
contains only bindings and changed parameters where possible.

### 2.3 Query-to-algebra compiler

A small learned controller maps a task into a declared algebra and objective,
not an unrestricted answer. Examples are:

- Boolean `(or,and)` for reachability and logical support;
- min-plus for shortest path and deterministic planning;
- max-product for a most likely explanation;
- separate variable-elimination or sum-product algorithms for probabilistic
  marginalization; and
- separately specified Bellman operators for stochastic control.

The controller may select factors, constraints, semiring, boundary values, and
stopping condition. The actual closure is executed, so a token model does not
need to relearn or imitate every depth of the algorithm.

### 2.4 Active falsifier and local repair

For unresolved hypotheses `H`, choose an action by expected decision value,
not novelty alone:

\[
a^*=\arg\max_a
\Big[\mathbb E_{o\sim p(o\mid a,H)}V(H\mid a,o)-V(H)-c(a)\Big].
\]

After observing the outcome, responsibility is assigned to observation map,
binding, factor, change point, or executor selection. Only implicated objects
are revised; unresolved alternatives remain explicit.

### 2.5 Consolidator

Repeated local factors are compressed into reusable templates when this
improves a held-out lifetime objective. Consolidation is not allowed to erase
provenance or silently change protected predictions. Fast episodic evidence and
slow reusable mechanisms remain distinct until a controlled promotion test.

## 3. What the architecture could improve

The candidate changes the division of labor:

- the neural model learns **what the local mechanisms are**;
- the executor computes **what follows from them**;
- interaction determines **which uncertain mechanism is wrong**;
- persistent modular state records **what was learned for next time**.

If it works, the result should improve four capabilities at once:

1. **Systematic reasoning:** apply a learned relation at horizons and problem
   sizes far beyond training rather than imitate observed chain lengths.
2. **Learning efficiency:** reuse one mechanism across many entities and
   worlds instead of relearning a global input-output map.
3. **Continual adaptation:** update a changed mechanism without overwriting
   unrelated knowledge.
4. **Causal reliability:** seek interventions that distinguish explanations
   whose passive predictions agree.

The predicted gain is not a fraction of a benchmark point. On environments
with reusable sparse mechanisms, the aim is at least `2x` fewer interactions or
`30%` lower cumulative regret on held-out families, reliable `10x` to `100x`
horizon extrapolation after short-horizon training, and at most `2%` loss on
protected old capabilities after repeated changes.

## 4. Theorem block A — factorization can change representation cost

Consider `d` binary next-state variables with indegree at most `k` and a
discrete action set of size `|A|`. An unstructured deterministic transition
table from all current states to all next states requires

\[
d|A|2^d
\]

output bits. If action is not counted among the `k` inputs and the correct
local parents are known, the local tables require at most

\[
d|A|2^k
\]

bits. Unordered parent identities additionally cost approximately
`sum_i log_2 binom(d,|Pa(i)|)`. If factors are chosen from a shared library of
size `M`, a loose ordered-slot wiring bound is

\[
d[\log_2 M+k\log_2 d]
\]

bits. A new environment changing only `s\ll d` factors needs to reacquire
roughly `s` local objects rather than a global table only when changed locations
are known, the observation map and bindings remain valid, the correct templates
already exist, and interactions excite the changed input configurations.
Otherwise even change localization carries a finite idealized identification
burden of at least `log_2 binom(d,s)` bits before noisy sample costs.

This is an exponential representational advantage over a literal global truth
table and only a potential sample advantage in `d-k`. It disappears when the
grounder, parent search, observation map, bindings, change detection, or factor
library costs as much as the global map. It is not an advantage over an equally
informed factored learner unless a separate sample or end-to-end result
establishes one. Supplied predicates or a hand-authored ontology do not solve
the research problem.

## 5. Theorem block B — exact closure separates learning depth from reasoning depth

Let `(S,oplus,otimes)` be an idempotent semiring and `A` a weighted adjacency
matrix. Define

\[
C_0=I\oplus A,
\qquad
C_{r+1}=C_r\oplus(C_r\otimes C_r).
\]

### Proposition T65.1

For a finite matrix over an idempotent semiring with identity, `C_r` aggregates
the semiring values of all walks of length at most `2^r`.

**Proof.** At `r=0`, `I` and `A` contain walks of length zero and one. Assume
`C_r` contains every walk of length at most `2^r`. A walk of length at most
`2^{r+1}` can be split into two walks of length at most `2^r`; semiring matrix
multiplication enumerates their concatenations, and `oplus` keeps both the old
and concatenated alternatives. Conversely, every concatenation represented by
`C_r otimes C_r` has length at most `2^{r+1}`. QED.

Idempotence is essential because the recurrence can generate the same walk
through multiple decompositions. This is a bounded-horizon result, not an
unrestricted Kleene closure when cycles matter. It directly covers Boolean,
min-plus, and max-product in their standard idempotent domains. It does not
cover sum-product or log-sum-exp, where duplicate derivations have different
semantics, nor a generic Bellman operator without a separate algebra.

Consequently, after the edge/factor recognizer is learned, increasing the
reasoning horizon does not require examples containing equally long reasoning
traces. The executor uses `O(log L)` closure stages for walks up to length `L`,
although dense matrix multiplication may cost `O(n^3 log L)` and must be
charged. Sparse or task-specific solvers are controls, not free improvements.

### Error boundary

Suppose every learned additive edge cost has absolute error at most `epsilon`.
Any fixed walk, and the optimal bounded-walk value, over at most `L` edges then
has cost error at most `L epsilon`. This does not imply that the same minimizing
walk is selected without a sufficient value margin. If exact min
is replaced by

\[
\operatorname{softmin}_\tau(x_1,\ldots,x_m)
=-\tau\log\sum_i e^{-x_i/\tau},
\]

then

\[
\min_i x_i-\tau\log m
\le \operatorname{softmin}_\tau(x)
\le \min_i x_i.
\]

The outer merge between the old and newly composed values contributes its own
softmin bias. A one-node example with exact base value zero gives
`softmin_tau(0,0)=-tau log 2`, disproving a recurrence that includes only the
intermediate reduction. If every min in the bounded min-plus recurrence is
intentionally softened, a safe coarse recurrence over `n` intermediates is

\[
E_{r+1}\le 2E_r+\tau\log(2n),
\]

and hence

\[
E_r\le 2^rE_0+(2^r-1)\tau\log(2n).
\]

This is an approximation bound for bounded min-plus values, not a correctness
result for sum-product or log-sum-exp. Long-horizon execution therefore
amplifies bad grounding. Exact algebra does
not rescue wrong factors; it can make a false edge globally consequential.
This makes calibrated acquisition and falsification the central research
problem, not a front-end detail.

## 6. Theorem block C — modular repair gives a precise preservation guarantee

Let an answer for query `q` be

\[
y_q=E_q(F_{D(q)}),
\]

where `D(q)` is a mechanically complete transitive dependency set containing
every learned and procedural input used by a referentially transparent
executor. This includes shared templates, encoders, grounder outputs, priors,
bindings, schema, compiler choice, constraints, caches, normalization,
resource limits, consolidation metadata, and randomness semantics. Replace a
set `R` of immutable, versioned nodes without shared aliases while keeping the
complete old query slice fixed.

### Proposition T65.2

If `D(q) intersect R` is empty, then the snapshot output `y_q` is exactly
unchanged. For stochastic execution the claim is equality under declared
common randomness or equality in distribution, not identical independent
samples.

**Proof.** The arguments supplied to the deterministic function `E_q` are
identical before and after the repair. QED.

This is conditional software noninterference, not yet a continual-learning
guarantee. Snapshot equality also does not guarantee identical future behavior
after policy-environment feedback. Incorrect or incomplete dependency
discovery, mutable shared encoders or templates, changed bindings, planner
changes, and stochastic approximations can alter supposedly protected queries.
An implementation must use a content-addressed dependency DAG, version repairs
as new nodes, replay old slices bit-for-bit, deliberately test aliasing and
incomplete certificates, and measure the full end-to-end effect.

## 7. The unsolved theorem block — grounding

Factorization and exact execution are useful only after raw observations have
been mapped to the right behavioral variables. General causal representation
learning can identify latent structure only under declared conditions such as
sufficient environment changes, faithfulness, injective observation mixing,
noise families, interventions, or paired views. These are paid information
interfaces, not free intelligence.

A tractable first family is

\[
Z_{t+1,i}\sim F_{m_i}(Z_{t,Pa(i)},a_t),
\qquad X_t=g_e(Z_t,U_t),
\]

where factor templates recur across environments, surfaces `g_e` change, and
legal actions provide excitation. The target is not recovery of human-named
latent variables. It is the smallest controlled predictive quotient that
preserves all legal action-conditioned futures and decision values, up to
permutation or other declared equivalence.

The next mathematical task is an end-to-end sufficiency theorem: quantify the
environment diversity and action excitation needed for a learned quotient to
recover reusable factor boundaries, then compare its lifetime sample and
compute cost with an equally informed generic Bayesian or recurrent learner.
No such theorem is claimed here.

## 8. Current-art collision audit

The blocks and most of the architecture diagram are already occupied in narrow
settings:

- Neural algorithmic reasoning already studies algorithm-aligned executors and
  size extrapolation. Nerem et al. prove that a sparsity-regularized GNN can
  recover Bellman-Ford and extrapolate to arbitrary shortest-path instances.
- Online predicate invention already implements predict-verify-repair over
  lifted symbolic dynamics and reports orders-of-magnitude sample advantages
  over PPO in MiniHack, but it receives fully observable ground atoms,
  hand-selected metarules, types, background predicates, deterministic
  dynamics, and an off-the-shelf planner.
- VisualPredicator learns visually grounded neuro-symbolic predicates and an
  abstract planning model online across five simulated robot domains.
  Pixels-to-Predicates uses a pretrained VLM to propose and evaluate visual
  predicates, learns a compact symbolic world model from short demonstrations,
  and plans over much longer horizons in simulation and real-world tasks. Both
  pay for strong perception, supplied skills or goals, and a symbolic planner.
- ESBM is the closest systems collision: failures, uncertainty, QA errors, and
  transition errors trigger active probes and typed local edits to predicates,
  rules, options, and executable mechanism memory under a regression-protecting
  verifier. It already instantiates most of experience-to-falsification-to-local
  repair in an Atari-style protocol.
- Mechanistic World Models gives nearly the same conceptual blueprint of
  jointly discovered variables, reusable mechanisms, bindings, active
  experimental design, compositional reuse, and localized adaptation, while
  noting that no current system integrates the full program at scale.
- Current causal representation learning proves identifiability under
  sufficient cross-environment changes and structural assumptions, but recent
  general-environment evidence remains small simulation rather than an
  end-to-end intelligent agent.
- CoMap already co-evolves textual world models and policies and reports a
  `16.75%` relative gain with Qwen3-4B, but remains text-state, one-step model
  calling, and self-distillation rather than a grounded modular causal model.
- ShadowDancer already uses paired same-dynamics/different-appearance videos to
  learn action-conditioned dynamics. Such shadow pairs are valuable but paid
  simulator data and are not a general raw-grounding solution.
- CausalGame shows the remaining empirical gap: greater agentic engagement did
  not reliably become mechanism discovery; even its fully engaged pattern
  achieved only `7.6%` causal-reasoning attribution, with widespread variable
  lock-in and optimization drift.

Therefore T65 is not a novelty claim for semirings, causal factors, symbolic
repair, world models, or even the high-level loop. The narrower unsolved
interface is to jointly discover uncertain reusable variables, mechanisms, and
bindings from raw changing observations; distinguish factor errors from
grounding and binding errors by intervention; compile diverse queries into
sound executable algebras; and commit local repairs with mechanically complete
dependency certificates under one charged lifetime objective. A mere pipeline
integration is not enough; the learned interface must make later execution and
repair measurably better than equally informed current systems.

Primary sources:

- https://proceedings.mlr.press/v336/nerem26a.html
- https://arxiv.org/abs/2410.23156
- https://arxiv.org/abs/2501.00296
- https://arxiv.org/abs/2602.17217
- https://arxiv.org/abs/2604.23800
- https://arxiv.org/abs/2606.02372
- https://arxiv.org/abs/2606.07127
- https://arxiv.org/abs/2607.04293
- https://arxiv.org/abs/2607.12474
- https://arxiv.org/abs/2607.28362

## 9. Mechanism-validation ladder before any production claim

### Block 1 — executor unit control, CPU only

Supply exact factors and compare a fixed-depth decoder, recurrent GNN, exact
solver, and semiring executor after training on path or rule depths `<=8` and
testing at depths `80` and `800`. This validates implementation and error
propagation only; it cannot establish intelligence or novelty. It must include
the one-node softmin counterexample and exact task solvers.

### Block 2 — factor acquisition, small local model

Randomize entity names, observation skins, graph size, and distractors while
reusing hidden mechanisms. Train on several mechanism families and test on
held-out combinations and at least one held-out family. Require calibrated
action-conditioned prediction, correct intervention selection, and factor
reuse. Compare against a same-size recurrent meta-learner, Transformer with
full history, predictive-state learner, program learner, and oracle-factor
upper bound, plus an equally informed factored learner and
VisualPredicator/Pixels-to-Predicates-style controls with the same tools and
planner budget. Factor IDs, parents, ontology, and change locations must remain
hidden; otherwise this tests table fitting.

### Block 3 — surgical continual revision

Switch `s` of `d` hidden mechanisms, then later restore them. Measure samples
to recover, retained performance on dependency-disjoint queries, repeated
switches, memory growth, and false attribution. Ablate factor boundaries and
local repair. Exact protection is expected only when the learned dependency
sets are correct. Deliberately include shared templates and grounders, compiler
changes, false attribution, rollback, and incomplete dependency certificates.

### Block 4 — combined lifetime test

One frozen learning algorithm faces disjoint interactive families with no
family ID or latent labels. It trains on short horizons and acts at `10x` to
`100x` horizons, while mechanisms recur behind new raw interfaces. A result is
substantial only if it clears the large capability gates in Section 3 across
multiple seeds and families while charging interaction, solver, memory,
grounding, search, verification, and consolidation cost.

Passing these blocks only justifies a larger trial. A substantially smarter
production-model claim additionally requires a production-scale backbone on
consequential held-out tasks; no oracle state or free verifier labels; matched
data, interactions, model and tool calls, latency, memory, solver, compiler,
and consolidation cost; multiple seeds and confidence intervals; and the
downstream Pareto win in Section 3 while retaining protected capabilities.

No rented GPU is justified until Blocks 1–3 produce a large replicated local
effect and the independent audit accepts the comparison.

## 10. Decision

The strongest current direction is not “add memory” or “add a graph.” It is:

> train a model to turn experience into uncertain reusable local mechanisms,
> execute their consequences with an algebra that extrapolates beyond training
> depth, actively falsify consequential alternatives, and revise only the
> responsible mechanisms.

The bounded-walk execution property and the conditional snapshot
noninterference lemma are real. The potential gain—systematic horizon transfer,
exponential representation reuse, and structurally isolated continual
adaptation—is large enough to matter. No sample advantage, learned dependency
certificate, or production capability gain has yet been established. The
decisive unsolved part is raw grounding: learning factor boundaries and
bindings cheaply and reliably enough that exact execution amplifies truth
rather than error.

T65 therefore earns continued theorem and microbenchmark work, but not a GPU
run or a claim that the breakthrough has been found.
