# Autonomous exact-title structure witness

Status: **FROZEN STAGE -2 SEARCH CONTRACT; NO CANDIDATE ADMITTED**  
Date: 2026-07-31

## Purpose

This witness is the smallest natural domain that preserves the full project
claim without conflating three unsolved problems.

The query itself literally names two target corpus titles, but may also mention
other title-shaped predicates or categories.  A deterministic raw-only matcher
may generate literal candidates; the router must identify the two grammatical
arguments.  Alias resolution and implicit retrieval
are a later, separately priced witness.  The active question is:

> Can from-zero raw-only training compile each document into a lossless digital
> plane plus useful executable structure, then use that structure to answer
> unpredictable title-disjoint natural reasoning questions substantially better
> than the strongest matched dense and conditional-memory controls at identical
> complete serving cost?

This is stronger than the current T32 result.  T32 established information
availability to a 9B explicit-text reader.  The witness requires the actual
small trained model, the exact deployed bottleneck, raw-only compilation, and
all physical resource gates.

## Frozen scope

### Training input

- raw titled documents only, serialized as `title + separator + prose`;
- no questions, answers, supporting titles, supporting sentences, relation
  labels, pretrained encoder, parser, embedding API, or teacher model;
- from-zero model and compiler parameters;
- compiler work and derived self-supervised examples are charged to training,
  not serving.

### Evaluation input

- a raw natural-language question containing exactly two non-overlapping corpus
  title surfaces;
- titles disjoint from any task-like compiler validation examples;
- answer and support fields sealed until the representation, router, reader,
  and thresholds are frozen;
- comparison/reasoning answer scored from complete answer strings.

### Serving contract

- equal total resident bytes, counted by dtype;
- equal or lower p50 and p95 on frozen H100/H200 request shapes;
- equal or lower peak allocated and reserved memory;
- no extra KV entries, visible prompt tokens, host/network retrieval, auxiliary
  model, or per-request storage read;
- all digital state resident in the served checkpoint allocation;
- training/compilation may cost more, but the cost is disclosed.

## Required candidate interface

A candidate supplies these typed functions:

```text
compile(raw_documents)
    -> lossless_base, structural_sidecar, frozen_reader_state

route(raw_question, title_dictionary)
    -> two typed document handles

read(query_state, handles, lossless_base, structural_sidecar)
    -> fixed-shape evidence code

reason(backbone_state, evidence_code)
    -> ordinary answer logits
```

The lossless base must reconstruct every admitted token record.  The structural
sidecar is where autonomous extraction is claimed.  It must contain observable
objects—integer span pointers, discrete codes, graph edges, grammar productions,
or another declared program—not opaque per-question labels.

## Why the lossless base is mandatory

The task-agnostic extraction no-go proves that a lossy raw-only compiler cannot
guarantee arbitrary future fact preservation.  A lossless base prevents a
failed sidecar from silently deleting knowledge and separates two claims:

1. the corpus information is resident;
2. the learned structure makes that information usable at fixed compute.

The sidecar may be lossy because it is an index/program over the lossless base,
not the sole copy of the evidence.

## Minimal mathematical witness

Let documents be token records `X_j` and future queries be pairs
`q=(j,k,r)` whose answer depends on bounded evidence from records `j` and `k`
followed by reasoning operator `r`.

The router supplies `j,k` by typing literal title candidates in the raw
question.  Let a sidecar
extract at most `a` addressable units per record and let the reader expose at
most `s` fixed-width evidence units.  A candidate paper must provide:

1. a family under which the sidecar construction covers every sufficient
   evidence set with probability at least `p`;
2. a constructive reader setting that selects a covered evidence unit with
   declared error `epsilon`;
3. an error propagation bound from two record reads to final reasoning.

For a two-record task that needs both reads, even optimistic independence gives

\[
P(\text{both evidence sets available})\le p^2.
\]

An 80% end-task target therefore needs `p >= sqrt(0.8) = 0.8944` before reader
and reasoner errors.  A four-head or four-anchor design is not admitted merely
because four sounds sufficient; its frozen raw-only sidecar must clear the
coverage bound on held-out natural documents.

## Required proof packet for a candidate

### W0. Structural statistic

Define exactly what the sidecar extracts and what raw observable causes every
write.  If it selects spans, specify the selection objective and tie-breaking.
If it emits codes, specify their decoding semantics.

### W1. Positive construction

Exhibit a generative family and an explicit compiler/reader setting for which
the sidecar recovers the answer-bearing statistic.  A teacher-generated key or
support-label selector is not a construction.

### W2. Failure theorem

Give an adversarial raw corpus or query on which the sidecar loses coverage,
aliases addresses, or exceeds its program budget.  The implementation must
detect or fail exactly at this boundary.

### W3. Strong-control absorption

Try to reproduce the same behavior with the strongest legal control.  If a
standard self-index, direct token table, learned summary, or Engram-like lookup
implements the same operator inside the ledger, narrow the claim before code.

### W4. Complete bytes and work

Count lossless-base bytes, sidecar bytes, reader weights, title dictionary,
workspace, extraction traffic, active operations, critical path, and the FFN
capacity removed to fund them.  Integer buffers are charged by bytes, not
pretend parameters.

### W5. Effect-size path

Propagate sidecar coverage, reader accuracy, reasoning accuracy, and protected
quality into a predicted natural effect.  The candidate must plausibly exceed
the best matched control by ten points, not merely the dense question-only
baseline.

## Strongest mandatory controls

All controls receive the same raw document stream and charged training work
where applicable.

1. **Dense frontier:** width-1,024 dense model with no plane.
2. **Byte-matched free state:** reduced-width backbone plus the same resident
   bytes exposed through the strongest non-document interface.
3. **Direct typed raw plane:** `uint16` token IDs, lengths, exact-title table,
   and the strongest legal matched reader.
4. **Compressed self-index:** lossless compressed text supporting exact search
   and extraction, plus the same reader.  FM-index/Infini-gram-like machinery is
   a control, not candidate novelty.
5. **Learned document summaries:** equal-byte opaque summaries with the same
   query and decoder opportunity.
6. **Conditional learned memory:** Engram-like or strongest available
   content-addressed learned lookup at matched bytes and active work.
7. **Candidate with sidecar destroyed:** zero, random, shuffled-between-records,
   and position-permuted sidecars.
8. **Candidate with base destroyed:** wrong-record and shuffled-record controls
   proving that answers are caused by the stored evidence rather than sidecar
   leakage.

The candidate must beat the best control, not a convenient dense baseline.

## Explainable block microbenches

### B0. Exact-title router

Inputs are raw question and frozen title dictionary.  Require 100% route
agreement on the frozen domain after labels are revealed, case invariance,
longest-nonoverlap behavior, and declared failure on removed/aliased titles.
The block does not claim semantic retrieval.

### B1. Lossless base

Require exhaustive token/length round-trip, deterministic build, corruption
detection, exact byte count, random extraction, title search, and order
invariance where declared.  Compare direct `uint16` and compressed self-index
bytes and operation counts.

### B2. Structural sidecar

Freeze from raw-only data.  Then reveal support spans only to score coverage,
not to train or alter it.  Require entity renaming, document shuffling,
paraphrase, relation reversal, and distractor controls.  Report coverage as a
function of sidecar bytes and address count.

### B3. Reader

Use an exhaustive small world with an explicit parameter witness, then unseen
random records.  Require correct/zero/random/shuffled/wrong-record controls,
score-margin distributions, numerical bounds, and exact output shape.

### B4. Bottleneck information ceiling

Give a stronger fixed reader only the evidence amount the deployed bottleneck
could possibly expose.  Do not substitute 48 or 128 explicit tokens for a
32-dimensional evidence code without labeling it as a looser ceiling.

### B5. Raw-only learnability

Train only the sidecar/reader proxy on raw-derived objectives.  Freeze it before
natural support/answer access.  Require at least three seeds and compare all
matched memory interfaces.

### B6. Small from-zero composition

Train the complete 36.6M served graphs at matched optimizer opportunity, token
pool, derived examples, parameters/bytes, and active work.  Require:

- at least 80% absolute frozen natural accuracy;
- at least +10 points over the best matched control;
- correct-minus-shuffled causal gap at least 15 points;
- protected validation NLL within 0.5%;
- replicated direction on every seed.

### B7. Physical integration

Only after B0-B6 survive, measure the non-dominated complete implementation on
target H100/H200 shapes.  A primitive kernel pass cannot replace the complete
graph.

## Candidate search object

The discovery system may vary only the structural sidecar and its fixed-shape
reader until B0-B5 pass:

```text
observe(raw record) -> raw events
build(multiset[events]) -> discrete sidecar
explain(sidecar item) -> raw witnesses
read(query, sidecar, lossless base) -> evidence code
resource_count(shape) -> exact ledger
```

Search populations remain separated by algebra rather than fashionable names:

1. extractive span/pointer programs;
2. lossless grammar and shared-subprogram references;
3. sparse incidence/set programs;
4. permutation/graph paths with explicit raw edges;
5. predictive codes with a proved observable recovery assumption;
6. physical schedules that preserve one admitted semantic operator.

An LLM may propose programs.  It cannot see natural support or answer labels,
and it cannot decide admission.  Exactness, counterexamples, control absorption,
resource limits, and frozen microbench thresholds decide survival.

## Current decision

No structural sidecar is admitted yet.  The direct typed raw plane, compressed
self-index, and exact-title router are controls.  The next candidate must begin
with W0-W5 and an explicit `p >= 0.8944` coverage path before reference code.
