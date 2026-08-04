# Proof-first breakthrough research protocol

Status: **FROZEN PRE-RUN CONTRACT**  
Date: 2026-07-31

## Purpose

The project is not looking for a small benchmark improvement.  It is looking
for a qualitative capability change, a scaling-law break, or a large
compute-equivalent gain under the fixed served-resource ledger.

No architecture idea advances directly from verbal plausibility to model
training.  It must first become an explainable mathematical construction whose
blocks can be falsified independently.

This contract distinguishes three kinds of statements:

1. **Theorem:** exactness, capacity, invariance, rank, error, or resource claims
   that can be proved before a run.
2. **Measured primitive claim:** a hardware, numerical, or learnability fact
   isolated in a microbenchmark.
3. **Composition hypothesis:** the remaining claim that the proved and measured
   blocks combine into better language-model capability.

A training run is allowed only when the composition hypothesis is the only
important unknown left.

## What successful research programs actually expose

This is a pattern extracted from the public papers, not a claim about the
authors' private creative process.

| Work | Named obstruction | Constructive change | Pre-scale evidence shape |
|---|---|---|---|
| [Transformer](https://arxiv.org/abs/1706.03762) | recurrent sequence computation and long dependency paths | all-pairs self-attention with parallel position processing | per-layer complexity, sequential-operation count, and maximum path length are compared explicitly before task results |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | attention is limited by HBM traffic, not only nominal FLOPs | an exact tiled online-softmax schedule | equality to ordinary attention, an I/O-complexity analysis, an optimality range, kernel measurements, then end-to-end models |
| [Mamba](https://arxiv.org/abs/2312.00752) | fixed SSM dynamics are weak at content-dependent selection | make SSM parameters input-dependent and supply a hardware-aware recurrent algorithm | the selective state update is the semantic primitive; synthetic recall and hardware behavior precede broad language claims |
| [Switch Transformer](https://arxiv.org/abs/2101.03961) | dense parameter capacity and active FLOPs grow together | route each token through a sparse expert subset | routing, capacity, communication, load balance, numerical stability, and quality are separate concerns |
| [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) / [AlphaEvolve](https://arxiv.org/abs/2506.13131) | free-form generation cannot certify discovery | evolve a small executable program under automated evaluators | a problem skeleton, isolated evolvable function, rich score, correctness filter, population diversity, and external verification |

The reusable pattern is:

> Name one obstruction, construct one operator that removes it, preserve the
> required invariants, prove its local claims, measure its physical primitives,
> and leave only one causal composition question for training.

The lesson from program-discovery systems is not “sample many architecture
names.”  It is to make the search object executable and the evaluator more
trustworthy than the generator.

## Candidate object

Every candidate is a tuple

\[
\mathcal C=(G, B, W, \mathcal O, \Delta, \Pi, \Phi),
\]

where:

- `G` is one promised capability gain;
- `B` is the conserved served-resource vector;
- `W` is a concrete witness where the matched baseline fails;
- `O` is the new typed operator or operator composition;
- `Delta` is the named source of slack: statistical structure, learnability,
  representation redundancy, allocation, or physical execution;
- `Pi` is the set of mathematical proofs;
- `Phi` is the one remaining empirical composition hypothesis.

If two important empirical miracles are required, the candidate is too vague
to run.  Split it into blocks or reject it.

## Paper gate: required before code

### P0. One-sentence edge

State one of the five project edges, the protected capability slices, and the
resource that does not increase.  “Potentially more expressive” is not an
edge.

### P1. Baseline failure witness

Give a minimal family of inputs on which the baseline's relevant mechanism
provably aliases, has the wrong rank, uses excess work/traffic, or presents a
measurable optimization obstruction.  Distinguish a limitation of the chosen
baseline from a universal lower bound.

### P2. Typed operator and semantics

Write every block as a map with shapes and declared meaning, for example

\[
E_\theta:\text{raw tokens}\to\text{relation statistic},\qquad
W:\text{statistic}\to\text{served state},\qquad
R:(\text{query},\text{state})\to\text{meaning-bearing code}.
\]

For each intermediate, specify at least one observable semantic contract:
equivariance, invariance, conservation, recoverability, compositional law, or
calibrated error.  “A latent vector the network will learn to use” fails this
gate.

### P3. Positive construction

Prove the exact property that creates the proposed gain.  Examples include:

- `R(q, W(E(x))) = f(x,q)` on the declared domain;
- a state or operation bound smaller than the matched baseline;
- an approximation error bound with the accepted epsilon named;
- a constant-depth or reduced-I/O schedule preserving the same function;
- an identifiability or sample-complexity result for the claimed structured
  family.

An existence proof must include a constructive algorithm.  An oracle that
uses labels or hidden structure unavailable to the real writer is only an
upper bound, not the construction.

### P4. Obstruction and scope

Prove where the construction stops working: rank ceiling, collision bound,
precision limit, adversarial family, dependency depth, or information lower
bound.  A proposal without a failure boundary is not understood well enough to
train.

### P5. Matched control

Show whether an ordinary Transformer, recurrent RAM, dense layer, or other
matched served class can embed the same operator.  If it can, the claim is an
inductive-bias or training-algorithm advantage, not a larger function class.
The experiment must then measure that advantage directly against the embedded
control.

Before freezing a run, perform an adversarial **baseline-construction audit**:
give the control its strongest legal parameters and try to solve the proposed
task on paper.  A witness against a restricted subcase does not justify a
claim against the named baseline class.  The input family must also exercise
the exact algebraic property named by the theorem; for example, a task using
only powers of one generator cannot establish an advantage from
noncommutativity.

### P6. Full resource equation

Count static bytes, active operations, mutable state, workspace, memory
traffic, dependency depth, and latency envelope.  State explicitly which cost
is paid at training/compilation time and which is paid for every served token.
Hardware slack may fund more arithmetic without more latency, but those
operations still belong in the ledger.

### P7. Effect-size prediction

Explain why the mechanism could cross the project's breakthrough bar.  The
prediction must be a qualitative capability, a changed asymptotic regime, or a
large compute-equivalent shift.  If the algebra can plausibly move only a
sub-percent metric, stop before implementation.

## Block microbench contract

After the paper gate, each block receives a reference implementation and its
own kill test.  Passing adjacent blocks does not imply that their composition
works.

| Block | Required explanation | Required microbench |
|---|---|---|
| semantic extractor | what information each output represents and which input transformations preserve/change it | exhaustive small worlds; entity renaming; paraphrase and relation swaps; held-out compositions; label-leak audit |
| writer/compiler | exactly which inputs cause which state changes | compare with a direct oracle; permutation-order tests; collision/capacity sweep; adversarial documents |
| stored representation | bit meaning, precision, and aliasing boundary | exhaustive codebook or state enumeration where feasible; BF16/quantized round-trip; corruption margin |
| reader | why the requested record is causally selected | exact identity test; zero/shuffle/random/additive controls; unseen keys and relations |
| decoder | how the meaning-bearing code selects an answer | code-distance/margin sweep; wrong-code nearest neighbors; calibration |
| physical operator | how the algebra maps to the fixed serving graph | target-shape latency, traffic, occupancy, workspace, and numerical agreement |
| composition | which single interaction is not already guaranteed | stronger decoded-information oracle first; then a small from-zero, multi-seed causal A/B |

Every primitive result must report:

- input and output types;
- invariant being tested;
- exhaustive or sampled domain size;
- oracle and adversarial controls;
- numerical precision;
- asymptotic and concrete resource use;
- exact kill condition;
- what a pass does **not** establish.

## Stage ladder

### Stage -2: mathematical paper

No implementation beyond scratch calculations.  Complete P0-P7.  Search the
literature only after the operator is precise enough to search for exact
collisions, lower bounds, and controls rather than nearby buzzwords.

### Stage -1: executable reference

Use a small, obvious CPU implementation.  Cross-check two independent forms
when the claimed identity is novel or easy to implement incorrectly.  Exhaust
the smallest nontrivial domains and actively search for counterexamples.

### Stage 0: isolated primitives

Run the block microbench matrix.  Hardware microbenchmarks are permitted only
for a mathematically admitted physical block.  No language-model training.

### Stage 0.5: information oracle

Give the downstream reader more information than the deployable mechanism can
provide.  If this upper bound cannot produce a large natural-capability gain,
close the representation before training its interface.

### Stage 1: composition screen

Run the smallest from-zero experiment that leaves only `Phi` unknown.  Use
matched parameters, operations, training data, optimizer opportunity, seeds,
and causal controls.  A synthetic win authorizes a natural-domain test; it is
not itself the project result.

### Stage 2: physical integration

Measure the complete block on target hardware and the frozen service envelope.
Primitive kernel speed does not substitute for end-to-end noninferiority.

### Stage 3: scale

Scale only after the capability effect is large, causal, replicated, natural,
and physically within the complete resource ledger.

## A discovery engine that an LLM can safely help with

The LLM may generate breadth, but it does not decide truth.  The discovery loop
operates on small candidate programs implementing typed `encode`, `write`,
`read`, and `resource_count` functions plus machine-checkable claimed
identities.

1. Freeze a baseline witness family and hard multi-objective evaluator.
2. Isolate one critical function to vary; keep the surrounding correct
   skeleton fixed.
3. Maintain separate populations for statistical compression, computation
   reuse, learnability/topology, and hardware asymmetry so one fashionable
   family does not consume the search.
4. Generate algebra/program mutations broadly.
5. Reject candidates that fail types, exactness, adversarial controls, resource
   accounting, or the matched-control audit.
6. Score survivors on margin to the proof bounds, generalization across small
   worlds, simplicity, and projected effect size—not one noisy LM metric.
7. Preserve diverse survivors and counterexamples.  Convert a survivor into a
   human-readable proof packet before any model run.

This borrows the verifiable search loop from FunSearch/AlphaEvolve while making
the evaluator specific to fixed-compute model research.  It does not automate
the natural-language sufficiency claim; that remains a frozen oracle and then
a causal experiment.

## Stop rules

- A failed stronger-information oracle closes the representation, not merely
  its neural reader.
- A pre-run matched-control counterexample withdraws the experiment before GPU
  use; it is a design correction, not a negative model result.
- A rank or information obstruction cannot be repaired by optimizer tuning.
- A primitive that is exact but solves the wrong statistic is retained as a
  component and removed from the active candidate.
- A hardware failure closes the physical mapping, not necessarily the abstract
  operator.
- A small gain does not authorize scaling under the breakthrough goal.
- A new idea starts again at Stage -2; it does not inherit admission from a
  neighboring failed mechanism.

## Immediate application

T24 passed storage/read exactness and failed the stronger decoded-information
oracle.  Therefore the XOR substrate is retained, while the local
anchor-window relation quotient is closed.

The proposed T25 equivariant coded operator is evaluated separately in
[`equivariant-coded-operator-t25-paper-audit.md`](equivariant-coded-operator-t25-paper-audit.md).
It is not admitted for implementation.

The broader reason that entity-renaming consistency cannot supply the missing
semantic key is proved in
[`entity-renaming-semantic-identifiability-no-go.md`](entity-renaming-semantic-identifiability-no-go.md).
The active Stage -2 search object is consequently the cross-surface evidence
bridge defined in
[`stage-minus2-semantic-bridge-search-contract.md`](stage-minus2-semantic-bridge-search-contract.md),
not another storage or reader implementation.

T29 subsequently supplied an executable rather than latent record: the exact
affine transition of its own input-driven memory stream.  Its frozen Stage-0
algebra and codec gate passed, but current cache literature narrows the novelty
claim and the diagonal homogeneous transition cannot route information between
coordinates.  The direct T30 refinement is therefore evaluated under the same
protocol as a dihedral-monomial operator, beginning with its independent exact
CPU gate rather than a language-model run.

T30's exact CPU gate and corrected arbitrary-state separation preflight passed,
but the first learnability screen was withdrawn before execution by the new
baseline-construction audit.  A subsequent natural-information audit then
closed T30 as the active breakthrough direction: keyed factual overwrite is
already diagonal-affine after the semantic address is known, while T30 does
not construct that address or a relational join.  This is the protocol's
intended outcome for an exact primitive that solves the wrong target
statistic: retain the primitive, spend no GPU, and return to Stage -2.
