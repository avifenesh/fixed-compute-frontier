# Intelligence Frontier Research Ledger

## Goal

Find a reproducible change that makes an AI system substantially better at
learning, retaining, abstracting, reasoning, modeling interventions, planning,
or self-correcting. More resources are allowed when they buy a qualitatively
important ability or a compelling capability-per-cost gain, but every moved
resource must be disclosed and compared with the strongest way to spend it.

The optimized object is the model's lifetime capability trajectory. Efficiency
is a constraint and comparison axis, not the destination: a cheaper unchanged
predictor does not satisfy this goal, while a bounded system that genuinely
learns, transfers, corrects, and retains may spend justified extra resources.

This is a research ledger, not a queue of mechanisms. A candidate enters only
after its claimed edge, conserved budget, hidden costs, and falsification test
are written down.

The magnitude boundary is explicit: a proved improvement of at least 20% on a
major end-to-end part of the learning process counts as success. Twenty percent
is the acceptance floor, not the design target; candidates should aim materially
higher. Results from 10% to less than 20% are arguable findings, not successes:
they may justify one bounded replication when broad evidence suggests the true
effect can clear the floor. Single-digit gains are engineering observations,
not the research result being sought, and close the candidate as the main lane.
Every experiment reports both relative and absolute change and freezes the
normalization before seeing results.

“Major part” means a complete capability phase with declared inputs and outputs,
such as evidence acquisition to fixed competence, retained learning after a
change, or end-to-end reasoning to a correct decision. It does not mean one
kernel, proxy loss, oracle-only subroutine, or uncharged internal operation.
At matched quality, a cost reduction of at least 20% qualifies; at matched
complete cost, an error or regret reduction of at least 20% qualifies. The
primary percentage is relative to one preregistered denominator; absolute
change and headroom consumed are reported but cannot replace it after results
are visible. A finite proof may establish the boundary analytically; a sampled
result needs a frozen confidence procedure whose relevant lower bound reaches
20%. A point estimate at or above 20% whose bound falls below 20% is not yet a
proved success.

A qualifying component result is recorded as a successful research finding even
before integration. It is called a **smarter model** only after a learned system
causally uses that component and preserves the gain against the strongest
matched learned and algorithmic controls. This distinction keeps genuine large
process improvements while preventing an oracle solver or toy subroutine from
being mislabeled as the final model result.

## Cost-efficiency edges

These five strict-Pareto edges remain especially valuable, but they no longer
exhaust the mission. A candidate may instead make a large capability trade if
the added state, training, interaction, or reasoning cost is explicit and the
same budget does not buy more from an ordinary control.

### A. Smaller model, larger-model capability

Strictly reduce served model bytes and active parameters while matching a
larger reference on a preregistered capability suite. Report FLOPs, latency,
context length, output-token count, and any external state separately.

### B. Faster model, slower-model capability

Strictly reduce end-to-end latency or increase throughput on fixed hardware and
workloads while preserving the reference output distribution or meeting a
predeclared quality non-inferiority margin. Prefill and decode are separate.

### C. Light model, dense knowledge

Increase reliable factual and procedural coverage per resident checkpoint byte,
without retrieval or hidden external memory. Long-tail recall, calibration,
conflict resolution, and common-capability retention are all required.

### D. Less GPU cost, more performance

Improve capability per GPU-dollar or joule under the same service-level
objective. Include batching, p50/p95 latency, throughput, prompt length,
generated tokens, host work, networking, and amortized auxiliary memory.

### E. Other genuine dominance

Strictly improve a declared outcome with no statistically meaningful regression
on the other served resources and capabilities. If another resource increases,
the result is a trade, not a Pareto improvement.

## Budget ledger

Every experiment records:

- resident checkpoint bytes and datatype;
- active weight bytes and FLOPs per token;
- KV-cache bytes per token and maximum context;
- activation/workspace memory;
- prefill and decode latency distributions;
- throughput at fixed batch and concurrency;
- total generated tokens per solved task;
- CPU, network, storage, retrieval, and auxiliary-model work;
- training tokens, training FLOPs, data provenance, and teacher cost;
- the complete quality suite, seeds, uncertainty, and regressions.

Training-only cost may be traded for a serving improvement, but it is never
called free. For total-cost claims it must be amortized over an explicit serving
volume.

## Admission and kill gates

1. **Claim:** Name the missing capability, the proposed causal edge, every
   resource allowed to move, and the substantial end-model threshold.
2. **Mechanism-free baseline:** Establish the strongest ordinary baseline under
   that same budget before adding the candidate.
3. **Accounting:** Write the complete resource equation. Logical feature counts,
   nominal FLOPs, or parameter counts alone are insufficient.
4. **Small falsification:** Run the cheapest test that can disprove the claimed
   causal mechanism, including a random/memorization control when relevant.
5. **Hardware gate:** Test target-shape kernels and end-to-end p50/p95 behavior
   before training at scale.
6. **Scaling gate:** Scale only after the candidate wins across seeds and does
   not rely on one benchmark, one batch size, or one prompt distribution.
7. **Close the branch:** A failed candidate becomes a numbered negative result;
   it is not repaired into a different idea under the same claim.

The active mission is now broader than fixed raw-prose compilation. It is
defined in the [real-intelligence research charter](INTELLIGENCE_RESEARCH_CHARTER.md):
identify what current systems lack as learners, then obtain a substantial
model-level gain in continual learning, reasoning, abstraction, world models,
memory, self-correction, or learning efficiency at a fully disclosed cost.
Raw-prose/digital-plane work remains preserved as one historical mechanism
family, not as the objective.

The current theorem boundary is:

- [T45 closed-loop belief boundary](results/closed-loop-belief-boundary-t45-paper.md):
  persistent evidence separates a reset predictor from a learner, but generic
  stateful controls contain the solution;
- [T45 independent audit](results/closed-loop-belief-boundary-t45-independent-audit.md):
  corrected equations and no-go on selecting a belief module;
- [T46 interventional predictive-state boundary](results/interventional-predictive-state-boundary-t46-paper.md):
  passive causal identification can be impossible while interventions recover
  the hidden mechanism with information-bounded sample complexity; still no
  architecture or experiment is admitted;
- [T47 operator-signature intervention kernel](results/operator-signature-intervention-kernel-t47-paper.md):
  audited no-go as a transfer candidate; deterministic excitation, separation,
  automorphism, epsilon, and legal commutator identities are retained, but no
  run is admitted. See the [independent audit](results/operator-signature-intervention-kernel-t47-independent-audit.md);
- [T48 controlled-Hankel action signature](results/controlled-hankel-action-signature-t48-paper.md):
  the raw-observable correction is valid but collapses exactly into spectral
  predictive-state/system-identification controls, so the operator-signature
  lane is closed on paper with zero runs; the
  [independent audit](results/controlled-hankel-action-signature-t48-independent-audit.md)
  confirms closure after reward-label, core-selection, and rank-scope fixes;
- [T49 counterexample-guided state birth](results/counterexample-guided-state-birth-t49-paper.md):
  corrected finite-machine refinement lemmas retained, but the architecture is
  closed. Once replay, localization, stable routing, and a complete oracle are
  charged, the core is classical; no run was admitted.
- [T50 amortized counterexample search](results/amortized-counterexample-prior-boundary-t50-paper.md):
  exact no-signal and Bayes-ranking boundary. A learned proposer can only
  amortize task information or informative probes; ranking a frozen test
  catalog is therefore closed as a standalone architecture. The
  [independent audit](results/amortized-counterexample-prior-boundary-t50-independent-audit.md)
  validates the algebra with operational-symmetry qualifications and closes
  T49 completely; no run was admitted.
- [T51 compositional causal mechanism recombination](results/compositional-causal-mechanism-recombination-t51-paper.md):
  independently revised to a transparent-interface codebook-decoding control.
  Probe cost is governed by actual separating complexity `s*`, which can range
  from `log2 M` to `M-1`; the atomic contrast changes class and oracle. The
  [audit](results/compositional-causal-mechanism-recombination-t51-independent-audit.md)
  denies architecture admission; no run occurred.
- [T52 hidden-wiring causal recombination](results/hidden-wiring-causal-recombination-t52-paper.md):
  audited finite-class control removing supplied wiring and shared names.
  The valid conservative ERM bound sharpens from `alpha^-2` to natural
  `alpha^-1` scaling under paired-loss concentration; the environment-level
  information floor charges all `r` returned bits. The
  [independent audit](results/hidden-wiring-causal-recombination-t52-independent-audit.md)
  confirms that exact ERM remains the control and no architecture or run is
  admitted.
- [T53 intelligence-gap reset](results/intelligence-gap-reset-t53.md):
  independently revised reset from transformer efficiency to models that
  acquire, interrogate, reuse, and locally revise executable knowledge. It
  proves a narrow passive causal information boundary and freezes a finite
  joint variable-mechanism-binding discovery problem as the next theorem
  object. The [audit](results/intelligence-gap-reset-t53-independent-audit.md)
  constrains the claim to an unclosed lifecycle, not an absent one; no
  architecture or run is admitted.
- [T54 action algebra as latent coordinates](results/action-algebra-latent-coordinate-t54.md):
  audited finite-field control showing that `r+1` calibrated observations
  recover coordinates under invertible affine sensing and full-rank toggle
  actions. Rank deficiency leaves a stabilizer ambiguity, while arbitrary
  path-independent opaque decoding costs `Theta(r 2^r)` bits. The
  [audit](results/action-algebra-latent-coordinate-t54-independent-audit.md)
  confirms the construction but rejects architecture novelty; no run occurred.
- [T55 lifetime-trained mechanism memory](results/lifetime-trained-mechanism-memory-t55.md):
  coherent typed runtime contract combining lifetime training, executable
  memory, active tests, and versioned edits, but independently rejected as a
  distinct architecture. The
  [audit](results/lifetime-trained-mechanism-memory-t55-independent-audit.md)
  finds no LTMM-specific theorem, an unjustified regret target, and hidden
  compiler/router bills; no microbenchmark or run is admitted.
- [T56 lifetime learning signal](results/lifetime-learning-signal-t56.md):
  corrects the target from generic memory to a persistent update process
  trained by future lifetime utility. It proves a conditional-information
  ceiling and a narrow persistence separation. The
  [independent audit](results/cross-family-persistent-meta-learning-t56-independent-audit.md)
  rejects its bounded latent meta-learner as generic recurrent meta-RL and
  retains only the narrower capability target: family-agnostic,
  horizon-extrapolating abstraction transport. No microbenchmark or run is
  admitted.
- [T57 falsification-gated theory memory](results/falsification-gated-theory-memory-t57.md):
  retained only as an acceptance/control protocol: freeze executable edits,
  evaluate future-only randomized interventional evidence, charge a complete
  code, and preserve provenance/rollback. The
  [independent audit](results/falsification-gated-theory-memory-t57-independent-audit.md)
  rejects distinct updater status, corrects the proposer result to stochastic
  domination, rejects the incomplete reuse bound, and finds raw binding still
  hidden. No run is admitted.
- [T58 action Lie algebra](results/action-lie-algebra-variable-formation-t58.md):
  retains Lie brackets and commutator loops as local geometric-control
  diagnostics under full-state diffeomorphic observation, legal inverse flows,
  and strong regularity assumptions. The
  [independent audit](results/action-lie-algebra-variable-formation-t58-independent-audit.md)
  rejects the distinct updater and transfer claims: identical algebra can hide
  arbitrarily difficult passive state, observation, reward, and policy
  learning. No run is admitted.
- [T59 value-relative abstraction boundary](results/value-relative-abstraction-boundary-t59.md):
  proves that a measurable point-separating reward family forbids nontrivial
  exact state merging. This is not a general coding or compute lower bound.
  Identical action dynamics can still induce opposite policies under different
  rewards. The
  [independent audit](results/value-relative-abstraction-boundary-t59-independent-audit.md)
  finds that broad persistent state plus a goal-conditioned working state is an
  occupied design contract, not a distinct updater. No run is admitted.
- [T60 epistemic learning throughput](results/epistemic-learning-throughput-t60.md):
  separates acquisition, retention, and utilization as three information
  channels. Internal reflection can improve computation but cannot add world
  information without new evidence; persistent storage and later policy
  influence impose independent bottlenecks. The
  [independent audit](results/epistemic-learning-throughput-t60-independent-audit.md)
  corrects adaptive acquisition to a full-transcript directed-information
  accounting and reclassifies the toy result as evidence access versus no
  evidence, not an architecture separation. The result is diagnostic
  instrumentation, not a distinct updater; no run is admitted.

Before these experiment gates, every new direction must pass the
[proof-first breakthrough research protocol](results/proof-first-breakthrough-research-protocol.md):
the operator is typed and explained, its positive and negative mathematical
bounds are written, each block has an isolated microbenchmark, and only one
composition hypothesis remains for training.  A plausible architecture sketch
is not an admitted experiment.

The latest closed theory lane is
[T69 counterfactual-fork lifetime training](results/counterfactual-fork-lifetime-training-t69.md)
and its [independent audit](results/counterfactual-fork-lifetime-training-t69-independent-audit.md).
Training one compact state against many action-conditioned futures targets the
right controlled predictive equivalence, but fork grouping adds no information
to the same transition tuples and is absorbed by a matched action-conditioned
world-model control. A common-random-number simulator can sharply reduce
action-effect variance, but that is an explicitly charged data oracle shared by
the control. The broad mechanism is closed on paper and no run is admitted.

The active objective reset is
[T71 developmental intelligence gap](results/developmental-intelligence-gap-t71.md):
optimize the model as a learner over experience, not merely as an answer map.
It defines the open evaluation target as a broadly transferable developmental
update policy that forms, tests, revises, transfers, and retains internal
knowledge across a declared task distribution. Its permutation theorem is
narrowly about zero-shot opaque-label grounding; the practical hold on T70 is
that the transferable object and common meta-distribution were never declared.
No mechanism is selected and no run is admitted; the next object is a tiny
identifiable world grammar and a same-backbone training-objective ladder.
That object is now drafted as the
[T72 finite mechanism grammar](results/finite-mechanism-grammar-t72.md), which
scores only identifiable behavior, proves the sufficient-state bit floor, and
closes pure affine dynamics as an active-learning test because a fixed basis is
already minimax. The successor
[T73 guarded mechanism worlds](results/guarded-mechanism-worlds-t73.md) adds a
threshold decision-tree primitive with a proved adaptive gap while retaining
exact affine execution and protected-change controls. Its
[independent audit](results/guarded-mechanism-worlds-t73-independent-audit.md)
rejects multi-level and neural admission but validates the stronger complete
depth-one bound: at `m=33,d=1`, adaptive acquisition needs at most 9 probes
while every exact fixed schedule needs 66. It remains paper-only.
The proposed non-isomorphic companion is
[T74 interventional orientation worlds](results/interventional-orientation-worlds-t74.md):
opposite causal directions have exactly identical passive distributions but
separate under intervention, creating an exact stochastic test of calibrated
belief update, monitoring, revision, and retention. Its
[independent audit](results/interventional-orientation-worlds-t74-independent-audit.md)
validates the probability separation but confines it to a decomposable
Bayesian-update calibration; no run is admitted.
The held-out cross-composition is
[T75 guarded causal composition](results/guarded-causal-composition-t75.md):
meta-train on guarded deterministic discovery and unguarded causal updating,
then test causal orientation behind an unseen hidden guard. Its
[independent audit](results/guarded-causal-composition-t75-independent-audit.md)
corrects the fixed-design bound to 32 versus at most 6 adaptive selector
locations at `m=33`. The frozen
[model-free Stage A](results/guarded-causal-composition-t75-stage-a-decision.md)
passes: a constructive adaptive reference guarantees at least 95% worst-world
accuracy with 168 probes versus 1,280 for the fixed comparator, while an
entropy-greedy exact-posterior reference reaches 96.03% average accuracy with
45.40 mean probes against a 36.69 information floor. This validates active
evidence selection. Under the project's 20% rule it is a qualifying component
success: the complete reliable acquisition phase costs `86.875%` fewer probes.
It is not a smarter-model result because the posterior and query policy were
supplied rather than learned. No neural or GPU run is admitted.
The broader successor hypothesis is the
[T76 epistemic operator machine](results/epistemic-operator-machine-t76.md):
train a bounded developmental learner to form hypotheses, choose experiments,
update and compose reusable mechanisms, revise them locally, and consolidate
them across a lattice of unlike interactive worlds. It is a mathematical
pre-gate only; its
[independent audit](results/epistemic-operator-machine-t76-independent-audit.md)
keeps it as a research charter and rejects it as a distinct mechanism.
The concrete successor
[T77 anytime-valid predictive state birth](results/anytime-valid-predictive-state-birth-t77.md)
now survives only as a predictive-admission protocol: prospective
action-conditioned evidence can be complexity-weighted with an anytime-valid
e-process, but this does not invent a candidate or prove causal structure. Its
[independent audit](results/anytime-valid-predictive-state-birth-t77-independent-audit.md)
corrects shared-stream and adaptive-proposal accounting, preserves the Azuma
constant under a precise centered-increment assumption, and finds direct RCE
and PatchWorld collisions. Structural-birth, model-intelligence, CPU, neural,
and GPU claims are all rejected.
The next future-utility proposal,
[T78 prospective counterfactual abstraction credit](results/prospective-counterfactual-abstraction-credit-t78.md),
is also closed before a run. Its exact latent-curriculum construction shows
that future-sensitive selection can yield an order-one gain when history
predicts a persistent future phase, while independence removes the advantage.
The [independent audit](results/prospective-counterfactual-abstraction-credit-t78-independent-audit.md)
shows that common-baseline paired scoring has the same argmax as ordinary
cost-sensitive held-out risk and fails under complementary or redundant skill
libraries. Prospective Compression, ALMA, and especially SkillMaster directly
occupy prospective, deployment-scored, and counterfactual skill learning. The
information boundary is retained; the method, CPU, neural, and GPU lanes are
closed.

The post-reset diagnostic is
[T79 intelligence-component localization](results/intelligence-component-localization-t79.md).
On a fixed audit distribution it decomposes decision regret into information
lost in formation/retention, retrieval/exposure, and deployed use, then defines
interface-matched sequential interventions without pretending that the
one-step identity decomposes closed-loop return. Its
[independent audit](results/intelligence-component-localization-t79-independent-audit.md)
finds direct knowing--using and causal-probing precedents and retains T79 only
as a revised internal gate. It admits no run; it prevents a memory fix from
being applied to an actor-channel failure, or policy search from being applied
to information the representation discarded.

The first post-gate candidate,
[T80 representation gauge transport](results/representation-gauge-transport-t80.md),
is closed as a novel candidate after
[independent audit](results/representation-gauge-transport-t80-independent-audit.md).
A transport map conditionally preserves an old continuation when the required
map exists on the protected support; this does not establish finite-anchor
identification or generalization. Ambient rank deficiency identifies only the
map's restriction to the anchor span, modulo downstream invariances, and the
retained old continuation must be charged. Transport Keys, learnable semantic
drift compensation, query drift compensation, and model stitching directly
occupy the operator. The theorem remains a diagnostic; no run is admitted.

The next paper candidate is
[T81 intervention-operator algebra](results/intervention-operator-algebra-t81.md).
Its
[independent audit](results/intervention-operator-algebra-t81-independent-audit.md)
rejects the broad grounding claim: brackets are coordinate-natural signatures
of local order sensitivity, not mechanism-independence or raw-variable
identification, and the main architecture repeats the closed T58 lane plus
current Lie-bracket causal discovery and Lie-action world models. One narrow
paper seam survives. Signed mixtures turn sparse pairwise brackets into coded
linear sketches; a constructive star code reduces each interaction row to
ordinary Rademacher compressed sensing. Fixed-energy normalization, reset
error, and second-order noise were charged in a frozen
[local matrix screen](results/coded-bracket-screen-t81-matrix-v1-decision.md).
The star code's median adaptive-pair/star identity-loop ratio was `0.9053`, so
it was worse rather than at least 20% better; it reached `2x` in zero of nine
cells, had one no-pass cell, and its only win was `2.7%`. The
[independent result audit](results/coded-bracket-screen-t81-matrix-v1-independent-audit.md)
accepted closure and found no bug capable of reversing it. The theorem remains
an idealized paper result; dense diagnostics, `k=64`, neural scaling, local GPU,
and rental GPU work are closed.

The subsequent
[T82 likelihood-state hybrid](results/likelihood-state-hybrid-t82.md) asks a
neural model to learn local action-conditioned likelihoods while an exact
bounded layer accumulates evidence. Its common-policy likelihood-ratio identity,
semantic-error bound, and conditional factor-preservation theorem survive
[independent audit](results/likelihood-state-hybrid-t82-independent-audit.md),
including the sharper `TV <= tanh(E/2)` certificate. The method itself is
closed on paper: once the complete candidate mechanisms are supplied, it is a
standard differentiable finite Bayes/particle filter, directly occupied by
current filtering, predictive-belief, and experimental-design work. A synthetic
CPU result would only reproduce a known filter prior, not solve hypothesis or
variable formation. T82 is retained as the strongest structured filter control
for a future open-hypothesis method; no CPU or GPU run is admitted.

The resulting
[open-hypothesis frontier contract](results/open-hypothesis-frontier-after-t82.md)
forbids supplying that omitted ontology. The next admitted method must create,
split, merge, reject, and compress testable mechanisms from raw partial
observations; retain explicit unknown mass; prove a non-negligible proposal or
coverage path; and causally beat matched state-splitting, program/predicate
invention, open-world mixture, expanding-expert, and exact-search controls by at
least 20% on the complete formation-to-decision process. No method or run is
currently admitted.

The first post-T82 screen,
[T83 explicit operator interfaces and active coverage](results/explicit-interface-curriculum-t83-paper-screen.md),
combines a recurrent discrete operator library with a curriculum that targets
under-identified learned interfaces. Independent review finds the architecture
occupied by NEO, Reusable Modules, recursive/depth-latent reasoning, and current
autocurricula. The only narrow remainder is raw-trace identification and active
coverage of latent interfaces, which is a curriculum mechanism rather than a
demonstrated smarter-model architecture. The
[T83 independent audit](results/explicit-interface-curriculum-t83-independent-audit.md)
therefore applies a `NO-RUN` decision: interface identifiability, adaptive
complete-cost coverage against an oracle/static policy, and a
coverage-to-capability theorem are all required before a CPU experiment. The
exact XOR-fingerprint witness is retained only as a sparse-recovery/group-testing
control because its matched method-level gain is zero. No GPU or rental work is
admitted.

[T70 bounded persistent meta-learner](results/bounded-persistent-meta-learner-t70.md)
is retained as one subordinate mechanism probe: train one fixed-weight model
over complete lifetimes, erase its transcript, and force a declared bounded
latent state to carry post-deployment learning. The minimal witnesses permit
order-one rather than sub-percent gains, but persistent memory and
cross-episode meta-RL are occupied blocks. Its current cross-family manifest is
held rather than frozen because T71 requires the shared mechanism grammar to
be explicit first.
The earlier [mathematical audit](results/bounded-persistent-meta-learner-t70-independent-audit.md),
[production audit](results/bounded-persistent-meta-learner-t70-production-audit.md),
and [local preregistration](results/bounded-persistent-meta-learner-t70-local-preregistration.md)
are retained as the historical admission path and control inventory. The cheap
lookup smoke they permitted is complete; T71 supersedes the remaining
cross-family screen.
The first [operator smoke decision](results/bounded-persistent-meta-learner-t70-operator-smoke-v1-decision.md)
and its [independent audit](results/bounded-persistent-meta-learner-t70-operator-smoke-v1-independent-audit.md)
validated causal state transport but failed its direct-credit hypothesis: full
credit scored 100%, a detached boundary still scored 92.81%, and reset or
within-batch-permuted state returned near chance. Shared Transformer weights
still trained through the query pass, so this closes the need for direct
cross-boundary BPTT on lookup—not learned state writing in general. No decisive
or rented run is admitted from it.

For the historical raw-prose/digital-plane lane, entity renaming has now been
proved insufficient to identify semantic relations by itself.  Candidate
generation is restricted to the
[Stage -2 semantic-bridge search contract](results/stage-minus2-semantic-bridge-search-contract.md),
which requires a raw-computable cross-surface bridge, a recovery theorem, an
explicit counterexample, and a decoded-information oracle before code reaches
the language model or GPU.

T27 proved exact fixed-width token packing and reading primitives, but its
[frozen decoded-prefix oracle failed](results/packed-token-evidence-plane-t27-prefix-oracle-decision.md)
on an exact BF16 yes/no tie.  The 55-token natural payload is closed without a
rerun; only the mathematical digital blocks are retained.

The next paper candidate was the
[T28 extensional incidence quotient](results/extensional-incidence-quotient-t28-paper.md).
It replaces free latent semantics with an observable relation: two raw surface
patterns share a code only when they recur on the same entity--value tuples.
Its exact and noisy recovery bounds remain valid, but the
[frozen CPU census failed](results/extensional-incidence-quotient-t28-stage0-decision.md):
only 3.7398% of eligible tuples had multiple surface witnesses and no recurring
pattern nodes connected.  T28 is closed before any model or GPU work.

The retained base candidate is the
[T29 title-triggered affine prefix operator](results/title-triggered-affine-prefix-t29-paper.md).
It avoids semantic-code induction: a raw document is compiled into the exact
affine transition of the same dedicated recurrent memory layer that reads the
query.  Its
[frozen Stage-0 gate passed](results/title-triggered-affine-prefix-t29-stage0-decision.md):
the exhaustive scalar/vector identities, BF16 codec, quantization bound, and
880-bit fixed ledger all held.  This is not a capability result.  Only a newly
frozen unquantized information oracle is admitted next; quantized, model,
physical-kernel, and production claims remain gated.

The retained structural refinement is the
[T30 dihedral-monomial prefix operator](results/dihedral-monomial-prefix-t30-paper.md).
It replaces the diagonal head's non-routing homogeneous transition with a
rotation/reflection plus elementwise scale, while keeping the recurrent state
at 220 bytes and every document record at 220 four-bit cells.  Its closed
composition, strict noncommutative separation, error bounds, and ledger are
proved on paper.  Only the frozen exact CPU reference is admitted next; no
learning or GPU run was admitted by the paper alone.  The
[frozen CPU reference passed](results/dihedral-monomial-prefix-t30-stage0-decision.md)
all group, operator, lazy-frame, codec, and ledger gates.  The first proposed
learnability screen was then
[withdrawn by its pre-run matched-control audit](results/dihedral-monomial-prefix-t30-learnability-pre-run-audit.md):
it disabled T29's learned scales, used only one generator in its document arm,
and therefore could not attribute a large win to noncommutative routing.  Zero
training runs were made.  A corrected
[arbitrary-state separation preflight](results/dihedral-monomial-prefix-t30-separation-preflight-decision.md)
then matched its theorem exactly: routed error was zero while an independently
optimized diagonal-affine map for every group element still had average NMSE
`108/109`.  This is a local operator result already adjacent to PD-SSM, not a
language breakthrough.  No further GPU run is admitted until a natural-
information argument names a statistic that a full matched diagonal-affine
control provably loses.  The subsequent
[natural-information bridge audit](results/dihedral-monomial-prefix-t30-natural-bridge-audit.md)
failed on paper: factual memory needs semantic addressing and keyed
overwrite/join operations, while T30 only moves an already-existing state by a
global group action.  T30 is retained as an exact primitive and closed as the
active breakthrough direction without a training run.

The next paper candidate,
[T31 predictive-congruence factor plane](results/predictive-congruence-factor-plane-t31-paper-audit.md),
gave the semantic state a canonical raw-behavioral definition: histories are
equal exactly when all their future distributions agree. The right
congruence, phrase composition, and finite-horizon KL bound are valid, but a
product factorization does not beat a matched distributed hidden state.
Unrestricted emissions recover exponential interaction cost, approximate
distance is not an equivalence relation, and raw observation does not identify
useful independent coordinates. T31 is closed on paper with no implementation
or GPU run. The retained search object is bounded predictive interaction order,
not predictive-state count.

The next representation rethink is the
[T32 scale-referenced whole-record plane](results/scale-referenced-whole-record-t32-paper.md).
It avoids both sequential entropy decoding and T27's 55-token truncation by
placing a directly addressable 128-token record into 382 of 384 state cells.
Its [frozen CPU gate passed](results/scale-referenced-whole-record-t32-stage0-decision.md):
all 2,405 corpus records round-tripped, 181,440 registered BF16 boundary
comparisons had zero errors, and the complete prospective write ledger was
4.595043% of the unchanged checkpoint. Its subsequently
[frozen natural-information oracle passed](results/scale-referenced-whole-record-t32-information-oracle-decision.md):
correct full records reached 89.4231% and deterministic 48-token windows
88.4615%, versus 46.1538% question-only and 48.0769% shuffled evidence. This
establishes a breakthrough-sized causal information ceiling, not a smarter
small model. Only a layer-by-layer reader theorem and physical same-budget
microbench are admitted before training.

The first reader refinement is the
[T32R exact-record reader](results/layer-striped-record-reader-t32r-paper-audit.md).
Its paper audit proves that token IDs need a semantic embedding lookup and that
one ordinary value path cannot injectively transfer two full records into one
query state. Broad per-layer/lookup memory is collided by PLE, TIDE, Bank of
Values, MemoryFormer, Parametric RAG, and Engram. T32R retains a narrower
parameter-neutral hypothesis: exact ID expansion through the existing token
embedding followed by a rank-32 query-conditioned scan. Its
[frozen CPU census passed](results/layer-striped-record-reader-t32r-stage0-decision.md):
all routes and records were exact, handles saved 381 request tokens, and the
worst conservative compute margin was +4.90M multiplies while 947,025 allocated
entries fit inside 967,680 removed dense entries. Only a frozen scan algebra
and block microbench are admitted next. The
[frozen rank-32 scan CPU reference passed](results/t32r-rank32-scan-stage0-decision.md)
its independent implementation, role, concentration, and exact ledger gates.
A streaming online-softmax H100 comparison against the actual width-1,024 and
width-940 FFN blocks is the next boundary; materializing expanded records is
not admitted.

The later raw-plane sequence is now recorded under the
[control-closure theorem](results/post-t35-control-closure-theorem.md).  T35's
lossless-weight/raw-context loop was absorbed by a strongest conditional-memory
control at the same artifact and reader, so it closed before measurement.
[T36](results/associative-bitparallel-raw-reader-t36-paper.md) retained an exact
bit-parallel segment monoid but closed because the direct automaton control
contains the complete method.  [T37](results/activation-hash-digital-plane-t37-paper.md)
retained its exact random-hyperplane capacity/robustness bound but closed
because exact-bucket recall vanishes under fixed semantic noise unless probes,
replication, or state restore the cost.

[T38 equivariant template Engram](results/equivariant-template-engram-t38-paper.md)
is different: its renaming quotient and covariant `COPY/LITERAL` interpreter
survived as representation algebra, but the frozen empirical protocol is now
[closed](results/equivariant-template-engram-t38-protocol-closure.md).  Its
477,249-row CodeParrot population is a partial viewer derivative while the
preregistration requires `partial=false`; direct pinned-shard sampling would
change the population and every frozen offset.  Two sealed attempts produced
no corpus statistic, and a proposed local oracle proxy was
[rejected pre-run](results/equivariant-template-engram-t38-local-upper-bound-pre-run-audit.md).
An attempted E3 generalization,
[T41 group-action executable memory](results/group-action-executable-memory-t41-paper.md),
was then closed by independent paper audit.  Its orbit and finite-field
identities are correct, but an identical quotient-aware learner ties it, while
the proposed natural uncertainty bundled the entire end effect.  No model,
GPU, or local census result exists.

[T39 residual decision DAG](results/residual-decision-dag-t39-paper-no-go.md)
then closed on paper.  A reduced ordered DAG over hidden bits is exactly a
hard hierarchical MoE/FFF/conditional table once its terminal payload is
specified; the identical-DAG control reproduces its function and resource
point.  ROBDD minimality is fixed-order and does not provide a neural-circuit,
learnability, or hardware separation.  No implementation was admitted.

[T40 projection-free record cross-read](results/projection-free-record-cross-read-t40-paper.md)
retained valid full-width transport algebra but also closed in independent
paper review.  Direct token IDs expand exactly through the shared embedding,
identity-coordinate values avoid T32R's rank-32 side-output subspace, and the
streaming-softmax/concentration identities are correct.  The proposal still
bundled several unidentified semantic edges, was exactly tied by a same-tape
same-reader control, lacked a complete title/index/code ledger, and had no
repair-minus-harm bridge from T32's explicit-text 9B ceiling to four summaries.
Zero corpus, model, CPU-census, or GPU runs were made.

[T42 product-coded hidden projections](results/product-coded-hidden-projection-t42-paper-no-go.md)
then attacked QKV/FFN matmuls directly by precomputing every projected
sub-codeword.  The table-sum identity is exact.  For the proposed one-level,
fully materialized table, the table-to-weight scalar ratio is `K/s`, while an
iid Gaussian source needs `K >= delta^(-s/2)` for relative MSE `delta`.  At
five-percent distortion that construction needs at least five table scalars
per weight scalar.  Independent audit correctly rejected treating this as a
universal lower bound: additive or hierarchical codebooks can reduce the
one-level exponent, but pay additional lookups, additions, assignment
structure, and metadata.  Re-encoding remains online work and low-bit/LUT
paths remain mandatory controls, so no complete Pareto path was established.
The scoped lane closed on paper with zero runs.

## Current state

The search has been reset to architecture: Transformer attention,
recurrent/SSM state, hybrid memory, MoE capacity, and the topology connecting
them. Corpus selection and training-data optimization are not candidate
mechanisms.

Candidate 003 tested bounded top-written semantic feedback. Top writing produced
a real three-seed state-tracking gain over a matched input-written control, but
the recurrent summary was unnecessary and the exact feedback edge imposes an
`nL` prompt-prefill dependency chain. The architecture is closed rather than
promoted:

- [Architecture frontier and falsification plan](THINKING_TABLE.md)
- [003 — Bounded semantic feedback](results/003-bounded-semantic-feedback.md)

A return-to-zero anatomy pass then measured Qwen3.5's actual hybrid cache.
Unequal recurrent-state rank improved causal KL about 2.5x over equal rank at
the same representable bytes, but every sub-dense point failed the frozen
no-harm gate. Rank 64 reached the passing neighborhood only after factor storage
ceased to be smaller than the dense state. The broad state-rank mechanism also
collides with existing work, so this remains a diagnostic rather than candidate
004:

- [Qwen3.5 hybrid recurrent-state anatomy](results/qwen35-hybrid-state-anatomy.md)

No candidate 004 is admitted. Routed private recurrent memories, routed SSM
parameters, shared KV plus MoE, combinatorial expert paths, and unequal
recurrent-state rank are being treated as closed or collided families rather
than relabeled as discoveries.

A permutation-graph/logical-refinement idea was closed at Stage 0 before GPU
rental. The two independent exact implementations agreed on 3,102 transitions
and 13,440 queries, but that exposed the fatal control: oracle bound commands
let a zero-parameter exact executor achieve 100% trajectory accuracy with 514
logical task-state bytes. The learned refiner cannot exceed 100%, so its frozen
five-point superiority gate is unreachable. Its separate ancestry accounting
also showed that closure compression is not memory superiority:

- [Stage 0 closure result](results/permutation-graph-refinement-stage0.md)
- [Permutation-graph pre-candidate preregistration](results/permutation-graph-refinement-preregistration.md)
- [Machine-readable pre-candidate plan](manifests/permutation-graph-refinement.pre004.json)

A nature-inspired constraint-basin direction was then closed at its algebra
gate. Sparse parity-energy descent is exactly a tied recurrent hypergraph
message-passing update; at binary states it is threshold bit-flipping. One unit
step of the specified log-sum-exp prototype energy is a softmax attention read.
A direct Hamming syndrome decoder also reaches the MAP ceiling on all 128 frozen
clean and single-flip episodes. This leaves a possible decoder state/quality
trade, not a new fixed-cost model capability algebra:

- [Constraint-basin algebra closure](results/constraint-basin-no-go.md)
- [Executable equivalence gate](experiments/constraint_basin_no_go.py)

The next outward pass derived quotient-group memory. Its algebra and CPU checks
are correct, but the architecture claim is closed: it is classical weighted
union-find, and ED-Batch Algorithm 5 already uses the same parent-relative
non-commutative transformations and consistency rejection in a neural-systems
pipeline. It is retained as a typed-RAM primitive, not as a new model method or
candidate 004:

- [Quotient-group Stage-0 proof](results/quotient-group-memory-stage0.md)
- [Executable graph cross-check gate](experiments/quotient_group_memory.py)

The algebraically valid remnants are preserved separately without carrying
forward the failed capability claim. This includes exact discrete commit,
proposal/verification admission conditions, two-involution transport, Beneš
switch accounting, ancestry representations, and mandatory fatal controls:

- [Retained primitive registry](RETAINED_PRIMITIVES.md)

A randomized finite-field pass then exposed the first strict different-currency
trade: at fixed declared error, one-sided error can replace linear deterministic
equality state with a logarithmic fingerprint. At the frozen
million-byte-string point, 30 logical
bytes replace a 1,000,000-byte exact lower bound for error below `5.5e-14`.
This is classical Karp–Rabin fingerprinting and a randomized recurrent cell is
an identical control, so it is retained as an epsilon/checksum primitive rather
than candidate 004:

- [Epsilon fingerprint Stage-0 proof](results/epsilon-fingerprint-register-stage0.md)

A second return-to-zero pass asked the algebraic question directly: can the
full-attention reads in the frozen hybrid be replaced by exact-oracle top-k,
top-k plus one additive tail statistic, or bounded low-order moments? Every
family failed the pre-registered distribution gate before causal selector or
kernel costs were counted. Per-KV-group localization found zero of 12 physical
groups individually passed, so the planned 9-of-12 combination was stopped:

- [Full-attention algebra separability](results/qwen35-attention-algebra-separability.md)
- [Per-KV-group localization](results/qwen35-attention-group-localization.md)

A later CPU-only pass attacked QKV/O and FFN dense projections with a shared
feature straight-line circuit.  The exact prefix witness and five hierarchy
worlds confirmed that structured full-rank transforms can have low circuit
complexity.  The proposed fixed ancestry did not survive even a 25% Haar
component: mixed noninferiority failed in all five worlds, and pure-Haar error
was 58.86 times dense linear at the median.  The branch was rejected before a
GPU rental:

- [Feature-DAG matrix-replacement decision](results/feature-dag-matmul-stage0-decision.md)
- [Feature-DAG preregistration](results/feature-dag-matmul-stage0-preregistration.md)

The next pass preserved arbitrary dense weights and tried to reuse projections
through exact temporal innovations.  The identity and its RMSNorm form are
valid, but a generic dense map turns every nonzero sparse innovation into a
dense output change with probability one.  The frozen SmolLM2-360M probe found
100% exact coordinate changes at every sampled projection.  Even oracle
per-coordinate int4 codes changed about 82% of attention coordinates and 81%
of FFN coordinates while causing 9.36% and 12.74% median projection
distortion.  Top-10% delta truncation caused roughly 60%-63% projected-delta
error.  The exact stack and pretrained-slack route were rejected without GPU
rental:

- [Temporal-innovation Stage-0 decision](results/temporal-innovation-matmul-stage0-decision.md)
- [Temporal-innovation preregistration](results/temporal-innovation-matmul-stage0-preregistration.md)

The resulting “cheap bulk plus capped dense exception” shift was then reduced
to a scaling law.  A rank-`r(D)` exception costs `2Dr(D)`; it breaks quadratic
projection scaling only if `r(D)/D` vanishes with width.  For a 25% isotropic
exception at `D=4096`, preserving 5% total relative error requires rank 3,994
of 4,096 and an ideal correction already costs 1.95x the dense projection.
For power-law singular values, squared exception energy becomes summable above
exponent `1/2`; only then can fixed-error rank approach a width-independent
cap.  The broad
structured-plus-residual family collides with DLoR, FOSL, structured FFNs,
BLAST, SLA, and ELSAA, so no duplicate GPU run is justified:

- [Dense-exception scaling law](results/dense-exception-scaling-law.md)
- [Executable phase table](results/dense-exception-scaling-law.json)

Reference documents:

- [Outward-mathematics reset and matched-machine theorem](results/outward-mathematics-reset.md)
- [Foundations and impossibility controls](FOUNDATIONS.md)
- [Mechanism-neutral opportunity map](OPPORTUNITY_MAP.md)
- [Current evidence boundary](EVIDENCE_BASELINE.md)
- [Capability and scale-gate contract](CAPABILITY_CONTRACT.md)
- [Physical resource measurement contract](RESOURCE_CONTRACT.md)
- [Frozen training ladder, service trace, and baselines](TARGET_WORKLOAD.md)

## Active paper-screened lane

T84 tests one narrow algebraic allocation change: compute a learned QK relation
score once, then reuse it for ordinary signed-softmax feature channels and
channelwise max-plus feature channels inside each head. The local mathematics
and exact fixed-score Bellman primitive survived independent audit. The broad
max-plus mechanism is prior art; the surviving question is whether sharing one
learned relation across heterogeneous reducers provides a large fixed-width
learning advantage over grouped softmax, narrow heads, independent/whole-head
hybrids, all-max, finite-temperature, official Tropical Attention, conditioned
DeepSets, and CPU-time-matched ordinary compute.

The CPU falsifier is now preregistered as the immutable effective-v4 ordered
triple and accepted by two independent audits. It requires simultaneous >=20%
relative and >=1-point absolute final-error gains against all twelve controls,
a non-saturated task, <=20% error on both latent subdecisions, protected soft/ID
behavior, causal lesions, and branch health. Passing is reject-only and earns a
separate causal model stage; it is not itself a smarter-model result. No source
execution, GPU, or rental is currently authorized:

- [T84 revised paper screen](results/product-semiring-attention-t84-paper-screen.md)
- [T84 effective-v4 acceptance](results/shared-score-heterogeneous-attention-t84-effective-v4-acceptance.md)
- [T84 v2 base](results/shared-score-heterogeneous-attention-t84-cpu-preregistration-v2.md)
- [T84 v3 overlay](results/shared-score-heterogeneous-attention-t84-cpu-preregistration-v3.md)
- [T84 v4 overlay](results/shared-score-heterogeneous-attention-t84-cpu-preregistration-v4.md)
- [T84 independent audit A](results/product-semiring-attention-t84-independent-audit-a.md)
- [T84 independent audit B](results/product-semiring-attention-t84-independent-audit-b.md)

Closed results:

- [001 — Block-routed SwiGLU](results/001-block-routed-swiglu.md)
- [002 — Attention evidence scaling](results/002-attention-evidence-scaling.md)
- [003 — Bounded semantic feedback](results/003-bounded-semantic-feedback.md)
- [Hybrid recurrent-state anatomy — no candidate admitted](results/qwen35-hybrid-state-anatomy.md)
- [Full-attention algebra separability — no family admitted](results/qwen35-attention-algebra-separability.md)
- [KV-group localization — stopped at individual-unit gate](results/qwen35-attention-group-localization.md)
- [Permutation-graph logical refinement — fatal exact control at Stage 0](results/permutation-graph-refinement-stage0.md)
- [Constraint-basin completion — ordinary recurrence/message passing](results/constraint-basin-no-go.md)
- [Quotient-group memory — valid classical primitive; architecture claim closed](results/quotient-group-memory-stage0.md)
- [Epsilon fingerprint register — valid error/state trade; architecture claim closed](results/epsilon-fingerprint-register-stage0.md)
- [Feature-DAG matrix replacement — fixed ancestry rejected](results/feature-dag-matmul-stage0-decision.md)
- [Temporal-innovation dense projection — exact composition and frozen slack rejected](results/temporal-innovation-matmul-stage0-decision.md)
- [Dense-exception scaling law — admission criterion, not a candidate](results/dense-exception-scaling-law.md)
- [T35 entropy-reinvested context — control-absorbed on paper](results/entropy-reinvested-inplace-context-t35-paper.md)
- [T36 associative bit-parallel reader — direct control contains the method](results/associative-bitparallel-raw-reader-t36-paper.md)
- [T37 activation hash — fixed-noise capacity/robustness bound closes exact buckets](results/activation-hash-digital-plane-t37-paper.md)
- [T38/T38R empirical census — frozen population/provenance contradiction](results/equivariant-template-engram-t38-protocol-closure.md)
- [T39 residual decision DAG — identical conditional control absorbs the method](results/residual-decision-dag-t39-paper-no-go.md)
- [T40 projection-free cross-read — valid transport blocks, semantic/model claim closed](results/projection-free-record-cross-read-t40-paper.md)
- [T41 group-action executable memory — valid orbit lemmas, strongest learner ties](results/group-action-executable-memory-t41-paper.md)
- [T42 product-coded hidden projections — exact one-level LUT identity and scoped rate/storage obstruction](results/product-coded-hidden-projection-t42-paper-no-go.md)
- [T43 KV-group token-tape substitution — logical resource law retained; useful reader and capability claim closed by independent audit](results/kv-group-token-tape-substitution-t43-paper.md)
- [Retained permutation/switch primitives — systems library, not a candidate](RETAINED_PRIMITIVES.md)
- [T58 action Lie algebra — local control diagnostic retained; distinct updater rejected](results/action-lie-algebra-variable-formation-t58.md)
- [T59 value-relative abstraction — exact no-state-merge boundary retained; design contract occupied](results/value-relative-abstraction-boundary-t59.md)
- [T60 epistemic learning throughput — acquisition/retention/use accounting retained; no updater](results/epistemic-learning-throughput-t60.md)
- [T61 future-value workspace admission — diagnostic intervention surface retained; distinct mechanism rejected](results/future-value-workspace-admission-t61.md)
- [T62 slow-thought automaticity consolidation — valid different-currency control; mechanism occupied](results/slow-thought-automaticity-consolidation-t62.md)
- [T63 function-preserving plasticity homeostasis — exact rank no-go; optimizer control only](results/function-preserving-plasticity-homeostasis-t63.md)
- [T64 same-size fiber-rank lift — exact construction spends representational slack](results/same-size-fiber-rank-lift-t64.md)
- [T65 intelligence-system architecture — learn factors once, execute and repair them repeatedly](results/intelligence-system-architecture-t65.md)
- [T65 independent audit — revise before running; full-loop and softmin corrections](results/intelligence-system-architecture-t65-independent-audit.md)
- [T66 interventional error-syndrome decoding — sparse diagnosis before local repair](results/interventional-error-syndrome-decoding-t66.md)
- [T66 independent audit — fingerprint circularity closes the run](results/interventional-error-syndrome-decoding-t66-independent-audit.md)
- [T67 evidence-carrying learning — train the update operator, not only the predictor](results/evidence-carrying-learning-t67.md)
- [T67 independent audit — evidence governance retained, intelligence claim closed](results/evidence-carrying-learning-t67-independent-audit.md)
- [T68 holonomy-triggered state birth — cycle obstructions as representation alarms](results/holonomy-triggered-state-birth-t68.md)
- [T68 independent audit — algebra retained; intelligence mechanism closed](results/holonomy-triggered-state-birth-t68-independent-audit.md)

## G0 admission validator

`frontier_g0.py` validates the frozen claim, artifact accounting, hardware,
workload cells, protected quality slices, and seed plan before an experiment is
admitted:

```bash
python frontier_g0.py manifests/experiment.json
```

The [SmolLM2-360M instrumentation self-check plan](manifests/smollm2-360m-instrumentation-selfcheck.plan.json)
uses a public, serving-compatible artifact to exercise the harness. It is not a
causal training baseline and deliberately fails G0 until a real rental run pins
the model revision and captures its artifact, hardware, container, server, and
trace hashes. A planning file is not evidence. The first captured run is the
[H100 PCIe manifest](manifests/smollm2-360m-instrumentation-selfcheck.h100-pcie-45568493.json);
its timing output remains an instrumentation smoke check, not benchmark evidence.
