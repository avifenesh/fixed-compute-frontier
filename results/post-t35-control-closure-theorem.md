# Post-T35 raw-plane control-closure theorem

Date: 2026-08-01  
Status: **ACTIVE SEARCH BOUNDARY; NO EXPERIMENT OR GPU RUN ADMITTED**

## Purpose

T35 exposed a recurring false source of capability: a candidate transforms the
raw corpus into a useful sidecar and gives that sidecar a reader, but compares
against a raw-memory control that was denied the same transform or reader.  The
candidate then appears stronger only because the control was artificially
weakened.

This note states the closure precisely.  Its job is not to rule out external
memory.  Its job is to identify what a successor must contribute beyond a new
name for a program the strongest raw-memory control can already execute.

## Frozen objects

Let:

- `X` be the raw training corpus;
- `Q` be the served query/prompt;
- `Y` be the answer;
- `A` be the complete legal training algorithm, including all raw-only proxy
  objectives, random seeds, optimization, and offline compilation;
- `K_A(X)` be every persistent artifact produced by training or compilation;
- `I` be the frozen finite-precision serving instruction set;
- `G_A(Q,K_A(X))` be the compiled serving program;
- `C(G,K)` be the complete serving vector from `RESOURCE_CONTRACT.md`.

The strongest lossless raw-plane control class `R` may:

1. keep any injective representation of `X` that fits the persistent-byte cap;
2. run any legal training/compilation algorithm on `X`;
3. use the same learned payload, indexes, metadata, prompt substitutions,
   adapters, gates, and candidate instructions already present in the frozen
   instruction set `I`; and
4. spend no more than the candidate in any serving-resource coordinate.

The control is not automatically granted a genuinely new instruction.  If a
candidate introduces one, its implementation, operands, workspace, traffic,
latency, and simulation cost under `I` must be named.  Otherwise the word
“primitive” can hide an ordinary program and make the comparison vacuous.

## Theorem 1: artifact-and-reader absorption

### Formal statement

Let candidate `H` be trained from the same raw corpus and compute

\[
H_X(Q)=G_H(Q,K_H(X)).
\]

Assume there exists a control parameterization `r_H in R` such that:

\[
K_{r_H}(X)=K_H(X)
\]

up to an invertible relabeling, and the control executes the same finite
instruction trace as `G_H` on corresponding states.  Assume all persistent
artifacts and runtime resources are charged symmetrically.  Then, for every
`X,Q`,

\[
r_{H,X}(Q)=H_X(Q)
\]

and

\[
C(r_H,K_{r_H})=C(G_H,K_H).
\]

Therefore `H` cannot have a positive worst-case, expected, or held-out
capability separation from the strongest control at that same resource point:

\[
\sup_{r\in R:C(r)\preceq C(H)}\operatorname{Score}(r)
\ge \operatorname{Score}(H).
\]

### Proof

Initialize the control with the candidate artifact under the invertible
relabeling.  At serving instruction zero their corresponding states agree.
If the states agree before instruction `j`, executing the same deterministic
finite-precision instruction on the same operands produces the same next
state.  Induction over the instruction trace gives identical outputs and final
state.  Coupling the same random tape gives the randomized case.  The resource
vectors are identical because every artifact, instruction, state byte,
workspace byte, and transfer is duplicated rather than omitted.  Since this
parameterization belongs to the maximization over controls, its score lower
bounds the control optimum.  QED.

### Plain explanation

If the control is allowed to carry the same book and run the same lookup
procedure, calling the book a graph, page, handle plane, compiler output, or
memory does not make the candidate a stronger model.  The control can be the
candidate.

### Small witness

Suppose raw text contains `Ada was born in 1815`, an offline compiler stores
`(Ada, birth_year, 1815)`, and a learned reader maps “When was Ada born?” to
that cell.  A conditional-memory control given the same triple, query mapper,
lookup, and decoder returns exactly the same answer for exactly the same work.

### Adversary

Deny the control the query mapper.  The candidate now wins, but the gap is an
unmatched component, not an architectural result.  Restoring the mapper erases
the claimed separation.

### Claim boundary

The theorem does not say the candidate cannot train more reliably than another
parameterization, or that two equivalent instruction traces have equal wall
time on real hardware.  It says those are the only kinds of claims left after
functional absorption, and they must be stated and tested as learnability or
physical-execution claims.

## Corollary 1: lossless formats do not create capability by themselves

Any deterministic lossless compiler, including token tables, grammars,
self-indexes, typed occurrence lists, graphs, or compressed weight-funded raw
planes, is absorbed when the control receives the same encoded bits and legal
operations.  Such a format can still provide a byte, traffic, or latency edge;
that is a systems claim, not a function-class capability edge.

## Corollary 2: ordinary readers do not escape closure

If the reader is composed of ordinary gather, matmul, attention, recurrence,
comparison, branching, or finite table lookup already available under `I`, the
control can execute the same composition.  Reversing a projection, shifting a
value stream, changing an attention schedule, or injecting into existing
prompt positions is not a capability-bearing escape unless its physical trace
has a strict measured resource separation.

## Corollary 3: offline compilation is not a free distinction

Serving cost may exclude one-time compilation, but the control must then be
allowed the same offline stage.  If extra compilation/training cost is the
alternative currency, it is reported separately and compared symmetrically.
Moving work before serving changes the cost coordinate; it does not by itself
prove a better served function.

## Corollary 4: acquisition is the residual only when isolated

If candidate and control have the same representable function at the same
serving ledger but use different training parameterizations, the admissible
hypothesis is:

> under the frozen raw data and training budget, one named parameterization
> learns one named edge more reliably.

This hypothesis cannot bundle semantic extraction, query routing, evidence
selection, role binding, and answer use.  Every other edge in the block graph
must be exact, proved, or independently bounded.

## The only admissible escape classes

A successor must begin by naming one of these, not by drawing another sidecar.

### E1. Strict online resource separation

Give an exact or epsilon-bounded operator `P` and a target function family
`W_n` such that:

\[
C_P(W_n) \prec C_{I\text{-control}}(W_n)
\]

in at least one frozen serving coordinate and is noninferior in all others.
The lower bound must apply to the strongest legal simulation under the frozen
instruction set and hardware stack.  The saved resource must fund a
capability-bearing component whose repair-minus-harm effect is large enough.

This is a physical/algebraic escape, not a claim that the control may never
adopt `P`.  If the control adopts it, it has become the proposed system; the
research contribution is the new primitive and resource exchange.

### E2. Identified task-sufficient quotient

Declare a source/task family in which a raw-observable statistic `Z(X)` is
constructively recoverable and sufficient:

\[
Y\perp X\mid (Q,Z(X)).
\]

Prove finite-sample recovery and an end-risk bound, then show `Z` requires
strictly fewer served bits/work than the strongest lossless control for that
family.  Natural prose satisfying the declared family may be the sole local
empirical link.  Without the source/task law, task-agnostic lossy usefulness is
unidentifiable.

### E3. Finite-budget learnability separation

Keep the same serving function class and ledger, but prove or isolate that one
named parameterization has lower sample or optimization complexity under the
same raw data and training compute.  The claim is about acquisition, not
expressivity.  A valid experiment changes only this parameterization and must
include a reparameterized strongest control with the same inference function.

### E4. Explicit approximation currency

Accept a frozen nonzero error `epsilon`, randomness, or workload prior and
prove that it buys a strict state/work/traffic reduction with a propagated
end-risk bound.  The approximation budget and its failures are part of the
claim; they cannot disappear inside average benchmark accuracy.

## Admission test for T36

The proposed associative bit-parallel regular-query reader is admitted to a
paper screen only under `E1`.  Its candidate contribution is not regex, Bitap,
an NFA, or a raw index; those are classical.  The possible contribution is a
fixed-cost language-model integration in which:

1. an exact associative word operator scans packed raw token IDs;
2. only matched spans are expanded into continuous vectors;
3. the complete scan plus gather has a strict traffic/work advantage over the
   strongest ordinary continuous reader on a frozen witness family; and
4. the saved serving resource funds a material knowledge/reasoning gain.

The paper is killed before code if any of the following holds:

- the strongest standard control implements the same scan at the same ledger;
- the claimed bound compares against a deliberately dense reader while a
  legal FM-index, inverted index, sparse gather, or automaton control matches
  it;
- question-to-program induction, semantic field discovery, role binding, and
  answer use leave more than one unresolved learned edge;
- the query program merely restates the answer-bearing relation unavailable
  from raw-only training; or
- the optimistic repair-minus-harm ceiling cannot reach the frozen material
  gain.

No CPU, local GPU, or rented run is authorized by this theorem.  T36 must first
provide its exact monoid, strongest sparse/discrete controls, finite resource
ledger, one learned edge, and end-effect equation.
