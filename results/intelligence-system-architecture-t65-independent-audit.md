# Independent adversarial audit: intelligence-system architecture (T65)

Date: 2026-08-01  
Scope: theorem correctness, compression/sample scope, current full-loop prior art,
and experiment sufficiency; no experiments or GPU used

## Verdict

**REVISE BEFORE RUNNING. RETAIN THE EXACT IDEMPOTENT-SEMIRING EXECUTOR AS A
BOUNDED-WALK RESULT; REJECT THE PRESENT SOFTMIN BOUND; RETAIN T65.2 ONLY AS A
CONDITIONAL SOFTWARE NONINTERFERENCE LEMMA; DO NOT CLAIM A MISSING LOOP OR A
PRODUCTION-INTELLIGENCE TEST YET.**

The corrected research goal is valid: a structured inductive bias may produce a
substantial intelligence gain even when a generic Transformer can represent the
same function. T65 therefore does not fail merely because its executor is
emulable. It fails its current run gate for narrower reasons:

1. T65.1 is exact only for bounded **walk aggregation** in an idempotent
   semiring. It does not cover two compiler examples, sum-product and
   log-sum-exp, and its differentiable softmin error recurrence is false.
2. T65.2 is true only after assuming a pure executor and a complete, immutable
   dependency slice. Those assumptions exclude the shared learned state most
   likely to cause interference in the proposed system.
3. The factor count is an oracle-parent representational comparison, not a
   sample-complexity result. It also omits action arity and the amortized cost of
   acquiring the library, bindings, and change location.
4. Current work already implements most of the alleged closed loop in narrow
   settings. T65 is an integration hypothesis, not an unoccupied architecture.
5. Blocks 1--3 can validate or kill components, but cannot show that a
   production model became substantially smarter. Block 4 is not yet a
   production-scale protocol.

## T65.1: exact result and failed error bound

For a finite matrix over an idempotent semiring, with semiring identity matrix
`I`, the recurrence

\[
C_0=I\oplus A,\qquad C_{r+1}=C_r\oplus(C_r\otimes C_r)
\]

does aggregate the semiring values of all **walks** of length at most `2^r`.
Idempotence is essential because the recurrence can generate the same walk
through multiple decompositions. “Paths” must be replaced by “walks.” The
result is bounded-horizon; it is not an unrestricted Kleene closure in the
presence of relevant cycles. For min-plus, negative cycles are harmless only
for the stated bounded-walk question, not for an unrestricted shortest path.

The theorem directly covers Boolean, min-plus, and max-product examples under
their usual domains. It does **not** cover sum-product or log-sum-exp because
their additions are not idempotent. Generic Bellman operators also do not
automatically form the stated semiring. Duplicate derivations are semantics in
sum-product, so they cannot be silently collapsed or multiply counted.

The stated softmin recurrence is false. Take one node (`n=1`), identity cost
zero, and no finite edge. With an exact base `C_0=0`, the first outer merge is

\[
\operatorname{softmin}_\tau(0,0)=-\tau\log 2.
\]

Thus `E_1=tau log 2`, while T65's recurrence gives
`E_1 <= 2E_0 + tau log 1 = 0`. The missing term is the bias of the outer
`C_r` versus `C_r tensor C_r` merge; repeated decompositions then accumulate
additional multiplicity bias.

If every merge is intentionally interpreted as a differentiable approximation
to exact bounded min-plus, a safe coarse induction is instead

\[
E_{r+1}\le
\max(E_r,2E_r+\tau\log n)+\tau\log 2
\le 2E_r+\tau\log(2n),
\]

and, for exact base merging,

\[
E_r\le 2^rE_0+(2^r-1)\tau\log(2n).
\]

This is not a sum-product/log-sum-exp correctness result. It also does not
preserve the selected minimizing walk without a margin.

The fixed-path `L epsilon` statement is valid only for additive edge costs with
uniform absolute error. It extends to the optimal value over walks of length at
most `L`, because every candidate has that uniform error, but not to arbitrary
semirings and not to argmin stability. Finally, `O(log L)` counts closure
stages, not work or wall-clock depth of matrix multiplication. T65 correctly
mentions dense work, but all later comparisons must charge it.

## T65.2: true tautology, missing system assumptions

T65.2 is correct as written for a deterministic pure function: identical
arguments to `E_q` produce an identical answer. It is not yet a preservation
theorem for the proposed learned agent. `D(q)` must be transitively closed over
all of the following, not just named factors:

- factor templates and aliased/shared parameters;
- observation encoders, grounder outputs, calibration, and global priors;
- parent bindings, entity identity, schema, and library indices;
- compiler choice, boundary conditions, constraints, planner state, and solver
  tolerances;
- caches, retrieval indices, normalization state, resource limits, and
  consolidation metadata; and
- randomness, action-policy state, and every mutable input that can affect a
  future trajectory.

The executor must be referentially transparent, unchanged nodes must be
immutable/versioned, and a repaired factor must not alias a shared object used
outside `R`. For stochastic execution, the conclusion must be equality under a
declared common-randomness coupling or equality in distribution, not identical
sample output. Snapshot equality also does not imply unchanged future behavior
after policy-environment feedback.

The implementable form is a content-addressed dependency DAG whose query slice
includes every learned and procedural input. A repair creates new nodes; replay
of the old slice must reproduce the old answer bit-for-bit. Until T65 defines
and tests that closure, `D(q) intersect R = empty` assumes away the main
interference problem. The comparison with dense updates should be described as
a conditional structural advantage, not an unconditional stronger guarantee.

## Factor compression is not yet a sample theorem

The truth-table comparison is correct only under oracle factorization and after
fixing the omitted action term. For a discrete action set of size `|A|`, when
action is not counted among the `k` inputs, the dense and local counts are

\[
d|A|2^d\quad\text{and}\quad
|A|\sum_i2^{|Pa(i)|}\le d|A|2^k.
\]

Parent identities cost approximately `sum_i log binom(d,|Pa(i)|)` for unordered
parent sets (or the stated `dk log d` as a loose ordered-slot upper bound).
Library use must additionally charge template definitions/parameters, library
learning, type schemas, binding discovery, and amortization across environments.

“Reacquire roughly `s` objects” requires that the changed locations are known,
the observation map and bindings remain valid, the correct templates already
exist, and interactions excite the changed input configurations. Otherwise the
agent must detect a change, locate it (at least a `log binom(d,s)` identification
burden in the finite idealization), distinguish factor from grounding/binding
error, and learn the affected tables.

The exponential storage advantage is therefore real relative to a literal
unstructured table with known parents. It is not an advantage over an equally
informed factored learner, object-centric world model, program learner, or
generic model that also exploits locality. Representational compression alone
does not imply fewer samples. T65 must either supply a PAC/minimax result under
declared coverage, noise, intervention, and structure-learning assumptions, or
retain only “potential sample advantage” and make the experiment establish it.

## Current full-loop collision (primary sources, checked 2026-08-01)

The statement that the closed loop itself is missing is no longer supportable:

- [VisualPredicator](https://arxiv.org/abs/2410.23156) learns visually grounded
  neuro-symbolic predicates and an abstract planning model online, reporting
  sample-efficiency and OOD gains in five simulated robot domains.
- [From Pixels to Predicates](https://arxiv.org/abs/2501.00296) uses a VLM to
  propose and evaluate visual predicates, learns a compact symbolic world model
  from short demonstrations, and plans at much longer horizons in simulation
  and real-world tasks.
- [Dynamic predicate invention](https://arxiv.org/abs/2602.17217) explicitly
  places online causal-model learning and repair in the decision loop and
  reports large sample gains in its restricted symbolic setting.
- [ESBM](https://arxiv.org/abs/2606.07127) is the decisive systems collision: a
  challenger turns rollout failures, uncertainty, QA errors, and transition
  errors into active checkpoint interventions; an optimizer makes typed local
  edits to predicates, rules, options, and executable mechanism memory; a
  multi-criterion verifier protects regressions; accepted artifacts are
  versioned; and recovery after mechanism changes is measured. This already
  instantiates most of experience -> falsification -> local repair.
- [Mechanistic World Models](https://arxiv.org/abs/2607.12474) supplies almost
  the same architectural blueprint as T65: jointly discovered typed variables,
  reusable mechanism libraries, bindings, compositional reuse, active
  experimental design, localized adaptation, and mechanism management. It also
  accurately says that no existing system fully integrates these pieces at
  scale.

Together, VisualPredicator/Pixels-to-Predicates supply perceptual predicate
acquisition and symbolic planning, ESBM supplies the active falsify-edit-verify
loop, and Mechanistic World Models supplies reusable factors, bindings, and
consolidation as the organizing principle. At the architectural-diagram level,
their combination covers T65. T65 adds a more explicit query-to-semiring
executor and a proposed dependency certificate, but the former is standard and
currently misstated for softmin, while the latter is conditional rather than an
implemented learned interface.

The exact interface that remains plausibly open is narrower and valuable:

> Jointly discover, from raw changing observations, uncertain reusable local
> variables, mechanisms, and bindings; select interventions that distinguish
> factor errors from grounding and binding errors; compile diverse downstream
> queries into sound executable algebras; and commit a local repair with a
> mechanically complete dependency certificate, all under one charged lifetime
> objective.

No cited system demonstrates that full interface in a substantial production
model. T65 may test it, but should not claim the loop or component conjunction
as novel.

## Why the proposed blocks cannot establish production intelligence

Block 1 is an executor unit test with oracle factors. It can validate bounded
algorithmic extrapolation and numerical error; it cannot validate learning or
intelligence. It must include the one-node softmin counterexample and compare
against exact task solvers, not just learned decoders.

Block 2 can test acquisition only if parents, factor IDs, ontology, and change
locations are hidden and observations are raw. Symbolic/oracle inputs test table
fitting. Controls must include an equally informed factored learner and the
VisualPredicator/Pixels-to-Predicates class, plus a generic backbone with the
same planner/tools and total budget.

Block 3 must deliberately include shared templates, shared grounding, compiler
changes, false change attribution, repeated repair/rollback, and incomplete
dependency candidates. Separate hand-isolated tables make T65.2 true by
construction and do not test the proposed system.

Block 4 remains a controlled lifetime benchmark. Passing it may justify a
larger trial, but is not evidence that a production model is substantially
smarter. A production claim additionally requires:

- integration with a production-scale backbone on consequential held-out tasks;
- no oracle factors, family IDs, simulator state, or free verifier labels;
- matched training data, environment interactions, model/tool calls, inference
  work, latency, memory, planner/compiler cost, and consolidation cost;
- strong base-model-plus-planner/memory/program-execution and ESBM-like controls;
- multiple seeds, confidence intervals, preregistered held-out mechanism
  compositions and surface shifts, and component ablations; and
- a downstream Pareto win, not merely better factor accuracy: the stated large
  interaction/regret and horizon gains while retaining protected capabilities.

The current blocks are appropriate falsifiers. They must be labelled
mechanism-validation gates, not a route by themselves to a production
intelligence claim.

## Mandatory corrections before any run

1. Change “paths” to bounded walks and state the finite idempotent-semiring,
   identity, and cycle scope of T65.1.
2. Remove sum-product, log-sum-exp, and generic Bellman operators from the
   theorem's coverage, or give separate correct algorithms and semantics.
3. Replace the softmin recurrence/bound; add the `n=1` counterexample and the
   outer `tau log 2` term. Do not treat softmin multiplicity as exact closure.
4. Scope `L epsilon` to additive costs and separate value error from selected
   path/policy stability.
5. Restate T65.2 with referential transparency, immutable versions, no shared
   aliases, a closed dependency DAG, and explicit stochastic semantics. Limit
   its conclusion to snapshot output noninterference.
6. Add `|A|`, parent/library/binding costs, change detection, excitation, and
   structure-learning costs to Block A. Keep storage compression distinct from
   sample complexity.
7. Replace “missing loop” with the narrower unsolved interface above and add
   VisualPredicator, Pixels-to-Predicates, ESBM, and Mechanistic World Models to
   the collision audit.
8. Relabel Blocks 1--3 as component falsifiers and specify the separate
   production-scale, full-cost downstream Pareto gate before claiming a
   substantially smarter model.

After these corrections, a no-GPU Block 1 remains justified as a mathematical
implementation check. Blocks 2--3 are justified only after their oracle inputs
and prior-art controls are fixed. No production or rented-GPU claim is earned
by the present protocol.
