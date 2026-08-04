# Proof-before-run research method v3

Date: 2026-07-31  
Status: **ACTIVE; INDEPENDENTLY AUDITED; NO MODEL OR GPU RUN IS ADMITTED BY THIS DOCUMENT**

## Purpose

The target is not an interesting component.  It is a production language model
with materially more capability under the same complete serving-cost vector.
An experiment is allowed to resolve a named uncertainty; it is not allowed to
stand in for an unexplained mechanism.

Mathematics cannot prove in advance that a new architecture will improve
natural-language capability.  It can, however, remove most invalid candidates
before training by proving or falsifying:

1. the baseline obstruction on a minimal witness;
2. the candidate's positive construction;
3. the candidate's information and approximation boundary;
4. separation from the strongest matched control;
5. identifiability from the observations actually available during training;
6. the complete resource exchange; and
7. whether the maximum possible end effect is large enough to matter.

Only the final composition claim should remain empirical.

## What the public record of breakthroughs teaches

The papers expose their validation logic, not every private act of creativity.
They do not establish that the researchers followed a strict proof-before-run
chronology.  The reusable pattern is the evidence structure in the completed
work.

| Work | Concrete obstruction | Change of object | Evidence package in the completed work |
|---|---|---|---|
| [Transformer](https://arxiv.org/abs/1706.03762) | recurrent models have sequential operations and long dependency paths | recurrence becomes an all-pairs dependency graph | complexity/path-length comparison, controlled architecture, ablations, translation |
| [Sparse MoE](https://arxiv.org/abs/1701.06538) | stored capacity and active dense work are coupled | one dense parameter set becomes a conditionally addressed population | capacity/active-work arithmetic, routing and load controls, language models; not complete cost equality |
| [Mamba](https://arxiv.org/abs/2312.00752) | prior structured LTI recurrence cannot choose what content to retain | fixed dynamics become input-conditioned dynamics | empirical selective-copy and induction witnesses, hardware-aware scan, block ablations, scaling |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | FLOPs are not the binding cost; HBM traffic is | an operator graph becomes an exact tiled I/O schedule | real-arithmetic equality, HBM-access bound, floating-point numerical check, kernel and end-to-end timing |
| [AlphaTensor](https://www.nature.com/articles/s41586-022-05172-4) | unconstrained algorithm search is too large and hard to verify | matrix multiplication becomes an exact finite tensor game | rediscovery of known algorithms, exact checking, diverse solutions, then hardware objective |
| [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) and [AlphaEvolve](https://arxiv.org/abs/2506.13131) | language-model proposals are unreliable as proofs | a small mutable program is placed inside an automatic evaluator | executable evaluator or proxy, population search, human-readable survivors, separate validation |

The common move is not “try a different module.”  It is:

> make the obstruction measurable, change the mathematical object, prove the
> local advantage in the correct resource model, and leave one causal question
> for experiment.

## Stage 0: freeze the end claim

Before ideation, write one tuple

\[
E=(M, D, Q, C, \tau, P),
\]

where:

- `M` is the end metric and protected slices;
- `D` is the raw training-data contract;
- `Q` is the inference workload and context distribution;
- `C` is the full serving vector on one named hardware/software stack and
  workload: persistent bytes, active operations by type, HBM/host/PCIe and
  interconnect traffic, mutable state, workspace, critical path, p50/p95/p99
  latency, throughput at frozen concurrency, energy, tokens, external calls,
  and statistical confidence;
- `tau` is the minimum material gain; and
- `P` is the comparison class, including the strongest legal dense, sparse,
  recurrent, memory, and compiled controls.

If a proposal silently changes `D`, `Q`, `C`, or `P`, it is a different claim.
The serving requirement is componentwise Pareto parity,

\[
C_{cand}\preceq C_{base},
\]

under the frozen stack and workload.  Training, search, compilation, and
amortization costs are reported separately; any permitted increase is named as
the alternative currency rather than hidden outside `C`.

## Stage 1: derive the obstruction, not the module

Write a minimal witness family `W_n` and answer four questions:

1. What exact function must be computed?
2. Which restriction of the baseline prevents or wastes resources on it?
3. Is the claim about representation, learnability, data efficiency, or
   physical execution?
4. What is the smallest counterexample to our own obstruction claim?

The result must be a scoped statement such as:

\[
\inf_{f\in\mathcal F_{base}(C)} R_{W_n}(f) \ge r_0
\]

or a resource lower bound for computing the witness.  Failure of one trained
checkpoint is not an obstruction theorem.

## Stage 2: change one algebraic object

The search space is organized by object changes rather than model names:

| Status-quo object | Possible object change | Required proof |
|---|---|---|
| dense linear mixing | addressed, factored, logical, or local transition | same useful function with a strict byte/work/traffic separation |
| recomputed token graph | persistent sufficient state or quotient state | recoverability plus update and alias bounds |
| extensional weight table | executable rule or shared subgraph | description-length and execution separation on a declared source family |
| uniform execution | conditional population or event | active-work cap, routing correctness, load/tail bound |
| nominal FLOP graph | concrete memory/synchronization schedule | operator equality or bounded error plus a physical lower/upper bound |
| continuous hidden vector | discrete/continuous hybrid state | quantization margin, transition semantics, and failure surface |

An object change is not yet a candidate.  It becomes one only after the proof
ladder below survives.

## Stage 3: the proof ladder

Each candidate needs all six lemmas.  A lemma may instead be an explicit
counterexample that closes the candidate.

### L1. Positive construction

Give an explicit algorithm or parameter setting that solves the witness.  The
construction may not use task labels, record handles, teachers, or tools absent
from the end claim.  It must be one uniform construction for the declared
witness family, not a different hand-set solution for every instance.

### L2. Information and approximation

State what information reaches every block.  For lossy state `Z=g(X)`, prove a
recoverability result or an error bound.  For approximate prediction, a useful
form is

\[
\mathbb E_h\!\left[\mathrm{KL}(p(\cdot\mid h)\|q(\cdot\mid g(h)))\right]
\le \epsilon,
\]

because it directly bounds excess next-token log loss.  A claim based only on
coverage, rank, or mutual information is not automatically an accuracy bound.

### L3. Non-circular identifiability

Every learned block must name its legal training algorithm, objective, and
available observations.  If it must recover a hidden object `z*`, state a
source family and derive a finite-sample result:

\[
\Pr(\hat z_n\not\sim z^*)\le \alpha
\quad\text{for}\quad n\ge n_0(\text{separation, noise, complexity},\alpha).
\]

“The true object uniquely minimizes the objective” cannot be assumed as the
separation premise.  The margin must follow from observable source properties.
Before candidate search, freeze the source family, equivalence relation,
estimator, fit test, development audit, and untouched adjudication split.  The
paper must provide:

- an observable lower confidence bound `underline(kappa) > 0` on the required
  separation;
- a goodness-of-fit test that can reject the assumed source family;
- `n0(underline(kappa), ...) <= n_available`; and
- an adaptive-search correction protecting the untouched adjudication set.

If acquisition cannot be proved, it may become the **sole** important
empirical hypothesis only for one named block or one named edge in the block
DAG.  “Joint acquisition,” “the model learns the pipeline,” and any bundle of
extractor, router, reader, decoder, and optimizer success are forbidden as one
hypothesis.  Every other block must already have a proved or independently
bounded contract.

### L4. Strongest-control separation

Construct the best legal control first.  Non-containment somewhere in the
function classes is not enough; the separation must concern the target witness.
Show either a finite-cost matched-risk gap

\[
\inf_{f\in\mathcal F_{cand}(C)}R_W(f)+\delta
\le
\inf_{g\in\mathcal F_{control}(C)}R_W(g),
\]

or a strict resource gap for realizing that same witness function.  If the
control implements the complete candidate function at the same ledger, the
remaining claim is only acquisition or inductive bias and must be named that
way.  Grant the control every shared operation, but do not make the comparison
vacuous by granting it the candidate's defining new primitive without charging
that primitive's full resource graph.

### L5. Complete resource exchange

Give symbolic and finite-shape ledgers for:

\[
C=(B_{persist},\ O_{active},\ T_{HBM},\ T_{net},\ B_{state},\
B_{workspace},\ D_{critical},\ N_{tokens},\ N_{calls}).
\]

This symbolic core is expanded into the frozen production vector from Stage 0;
different operation types are not collapsed into one incomparable count.  The
candidate must be funded by a removed baseline component in one frozen graph.
Packed data, integer tables, routers, indexes, metadata, alignment, CPU work,
host transfers, compilation, and amortization are charged.  Confidence
intervals accompany physical measurements.  Training and serving ledgers
remain separate.

### L6. End-effect bridge

Write the end gain in its actual metric with exactly one unresolved empirical
quantity.  For an additive loss `ell`, a selected candidate path `T`, and
baseline fallback outside selection `S`, the exact gain is

\[
G=R_B-R_C
=\mathbb E\!\left[\mathbf 1_S
  (\ell(B,Y)-\ell(T,Y))\right].
\]

For hard accuracy, with candidate correctness `T` and baseline correctness
`B`, this becomes

\[
\Delta A=P(S\cap T\cap\neg B)-P(S\cap\neg T\cap B).
\]

Before running, freeze the required threshold for the one unresolved term and
provide non-vacuous bounds for every other term.  A generic `repair <= 1,
harm >= 0` ceiling is not admission evidence.  If the resulting optimistic
ceiling is below `tau`, close the idea.  The experiment may estimate the named
term; it may not replace this equation with global component accuracy.

The unresolved quantity must be local to exactly one declared block or DAG
edge.  It may not be `G`, end-to-end accuracy, joint acquisition success, or a
scalar that bundles several learned failures.  The map from that local quantity
to `G`, including every coefficient and harm term, must already be proved or
independently bounded.

## Every proof is formal and explainable

A mathematically correct but opaque proof is a fragile research artifact.  Each
lemma is stored with:

1. **formal statement** — variables, quantifiers, assumptions, conclusion;
2. **plain explanation** — what is conserved or why the result must hold;
3. **small witness** — the smallest hand-checkable positive example;
4. **adversary** — the smallest counterexample when an assumption is removed;
5. **claim boundary** — what the lemma does not establish.

If the plain explanation cannot be written without “the model learns it,” the
missing step is empirical and must be isolated as such.

## Stage 4: typed block graph

Only after the proof ladder survives do we decompose the implementation:

```text
raw observations
  -> observable statistic
  -> writer / persistent representation
query + persistent representation
  -> router
  -> reader or executor
  -> answer-bearing state
  -> decoder
```

Each block card contains:

- exact input/output shapes and dtypes;
- semantic meaning;
- formula or reference algorithm;
- invariant and assumptions;
- byte, operation, traffic, state, and depth functions;
- exhaustive small domain or frozen sample domain;
- independent implementation/check;
- positive, zero, random, shuffled, collision, and shortcut controls;
- one kill result; and
- the larger claim that a block pass does not prove.

A block with two independent unknowns is split again.  The block dependency DAG
must also state how local approximation errors, failure probabilities, traffic,
workspace, and critical-path costs compose into the end-to-end budgets.

## Stage 5: microbenchmarks answer one binary question

A microbenchmark is admitted only when its block has passed the paper gate and
the result can change a decision.

The preregistration must say:

```text
hypothesis:
theorem already established:
unresolved empirical fact:
domain and controls:
pass threshold -> next permitted action:
fail threshold -> closed claim:
resource measurement:
```

Order:

1. exhaustive or symbolic check of the local algebra;
2. raw-data census of the theorem's observable assumptions;
3. reference implementation of one block;
4. numerical/physical primitive benchmark;
5. two-block composition;
6. small causal model A/B;
7. target GPU;
8. preregistered multi-seed end-to-end Pareto test on untouched data;
9. scale.

Passing a later cheap-looking test never excuses a missing earlier proof.

### Local-first target-hardware admission

A target-GPU rental is not a substitute for a small experiment.  Before any
multi-hour rented run, the candidate must pass all of the following on local
resources:

1. the paper packet, counterexample search, and strongest-control comparison;
2. an exhaustive CPU check of the smallest nontrivial construction;
3. a reduced-shape local GPU run through the same semantic and numerical
   codepath intended for the target GPU;
4. positive, zero, shuffled, and strongest-control results large enough to
   clear the frozen local kill margins; and
5. a measured scaling model for time, peak bytes, traffic, and expected effect,
   including the uncertainty interval and the first scale at which the local
   conclusion could change.

If the local GPU is temporarily occupied, wait or continue paper work; do not
rent an H100 merely to perform a falsification that fits locally.

The rental preregistration must additionally name the one target-hardware or
production-scale uncertainty that cannot be answered locally, the exact
machine stratum, maximum wall time and cost, intermediate checkpoints, and a
machine-checkable early-stop rule.  The first rented action is a short smoke
block.  A multi-hour continuation is admitted only if that block reproduces
the local direction and remains inside the frozen scaling envelope.

## Discovery loop for humans and LLMs

The LLM is a proposal generator and adversarial algebra assistant, not the
evaluator.

1. Freeze one obstruction, one witness, one serving ledger, and one strongest
   control.
2. Generate object transformations, not named architectures.
3. Translate each transformation into typed equations immediately.
4. Search for a positive construction and a counterexample in parallel.
5. Reject type errors, circular assumptions, control containment, information
   violations, ledger overflow, and sub-material effect ceilings before code.
6. Preserve counterexamples and closed lanes so later searches cannot rename
   them.
7. For a finite/searchable algebra, use a tiny executable skeleton, an exact
   evaluator, diverse populations, and rediscovery of known constructions as
   a sanity check.
8. Require an independent paper audit whose reviewer is asked to disprove the
   claim, not improve the presentation.
9. Admit a microbenchmark only if exactly one important local empirical fact
   remains.

## Application to T34

T34 fails before implementation:

- its MDL premise assumes the semantic recovery it claims to prove;
- a matched conditional-memory control computes its complete local function;
- global path coverage does not imply repair minus harm;
- the architecture and serving ledger are not one frozen graph; and
- natural compilation, routing, safe selection, and residual use leave several
  independent empirical miracles.

Therefore typed pages remain a vocabulary and exact finite operators remain
reusable blocks, but T34 authorizes no microbenchmark or H100 work.

## Admission condition for the next direction

The next paper must begin with one of two things:

1. a raw-observable structural statistic with a non-circular recovery and
   approximation theorem; or
2. a new operation/resource model with a strict separation that the matched
   dense, MoE, recurrent, conditional-memory, and compiled controls cannot
   inherit for free.

Until one exists, generating more module diagrams is not progress.
