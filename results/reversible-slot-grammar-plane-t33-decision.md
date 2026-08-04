# Reversible slot-grammar plane T33 — decision and key shift

Date: 2026-07-31  
Decision: **CLOSED; no model or GPU stage admitted**

## What was tested

T33 asked whether raw prose alone exposes recurring, reversible one-hole
surface programs that can place useful typed span pointers into the existing
128-token raw plane.  The compiler was frozen before opening support metadata.
It used no model, embeddings, parser, questions, supports, or answers.

The pre-run program contained:

- a restricted recovery theorem and explicit non-identifiability theorems;
- an exhaustive direct-vs-reference anti-unifier check over 595,686 pairs;
- ten planted/adversarial worlds;
- an actual aligned packed image and byte ledger;
- a sealed support-location evaluator that never loaded the source answer
  column.

The first CPU attempt exposed a conformance bug: the indexed occurrence
matcher omitted the frozen 127-token hole bound.  A 128-token adversary was
added, the implementation was corrected without changing the method or gate,
and the identical census was rerun.  All twelve local/remote tests then passed.

## Frozen candidate result

Phase A passed its raw-only construction gates:

- 2,405 documents and 9,799 eligible compiler sentences;
- 3,577 proposals, 1,861 positive-saving proposals;
- 1,785 positive proposals rejected for observable conflicts;
- 76 retained rules, 604 logical occurrences, and 431 served edges;
- zero witness failures;
- 643,584 resident bytes, below the 974,848-byte Stage-0 cap;
- 23.62 CPU seconds and 18.75 wall seconds, with no CUDA-visible device.

The sealed result failed by a decisive margin:

| measure | required | observed |
|---|---:|---:|
| required support sentences with a usable edge | 89.44% | **2.76% (6/217)** |
| questions with usable edges in both documents | 80.00% | **0.00% (0/104)** |

The six hits were not useful relation templates.  Their decoded boundaries
were examples such as `K ... .`, `S ... .`, `L ... .`, `A ... .`, and
`and ... .`.

The 128-token record boundary cannot explain the miss.  There were 173
retained occurrences outside the served record.  Even granting every one as a
different missing support hit gives at most `6+173=179`; the gate needs at
least 195 of 217.

## Was this a wall or an implementation choice?

Both, at different levels.

The immediate 2.76% failure was caused by the conservative selector.  A
post-hoc privileged ceiling kept all ambiguous rules without changing their
raw construction:

| diagnostic | support | both documents | bytes | exactly one edge |
|---|---:|---:|---:|---:|
| all 1,861 positive rules | 83.41% | 75.96% | 755,840 | 16.59% |
| all 3,577 recurring rules, ignoring MDL | 94.47% | 90.38% | 807,872 | 18.43% |

Thus conflict rejection discarded coverage.  But retaining ambiguity turns
the grammar into a redundant span lattice: 205 covered support sentences had
921 usable edges, about 4.49 competing spans each.  The semantic selection
problem was deferred rather than solved.

The mandatory simple control then dominated it:

| representation | support | both documents | bytes | exactly one span |
|---|---:|---:|---:|---:|
| direct raw sentence-boundary index | **99.54%** | **99.04%** | **669,120** | **91.24%** |
| all recurring T33 rules | 94.47% | 90.38% | 807,872 | 18.43% |

After the common aligned raw base, the sentence index costs 48,512 bytes and
the relaxed grammar costs 187,264 bytes: the grammar sidecar is 3.86 times
larger.  It is also less complete and much more ambiguous.

This is not a threshold accident.  Any T33 edge is already an interval inside
a raw sentence.  A full sentence pointer is an information superset of that
edge.  T33 can win only by cheaply identifying the task-relevant subinterval;
its raw surface recurrence did not do that.  Once all alternative intervals
are retained, the simpler sentence index absorbs the claimed structural gain.

## Key shift

The bottleneck is no longer knowledge storage or support containment.  A
lossless raw plane plus a 48.5-KiB sentence index already exposes virtually all
required evidence.  The missing operation is **query-conditioned semantic
selection and reasoning**.

Therefore another task-agnostic corpus compiler is the wrong next move.  A new
candidate must attack the conditional computation itself and pass these paper
gates before code:

1. the baseline function is contained as an exact setting;
2. a declared task family has an exact candidate construction and a lower
   bound against the matched dense/sparse control at the same state and served
   work;
3. the query-to-operation link is learned from the allowed from-zero signal,
   not supplied by a support label, parser, teacher, or arbitrary rule ID;
4. the complete method beats raw plane plus sentence index, not a weaker
   no-index control;
5. any extra routing, tables, branches, and physical kernel overhead are
   charged.

The next algebra search should examine context-addressed reuse of a fixed
weight bank or another conditional circuit family: the opportunity is to make
one physical parameter participate in several query-dependent logical
features without adding a second matrix multiplication.  This is a search
direction, not an admitted method.

## Artifacts

- `results/reversible-slot-grammar-plane-t33-stage0-phase-a.json`
- `results/reversible-slot-grammar-plane-t33-stage0-phase-a.bin`
- `results/reversible-slot-grammar-plane-t33-stage0-coverage.json`
- `results/reversible-slot-grammar-plane-t33-postmortem.json`
- `experiments/reversible_slot_grammar_plane_t33_stage0.py`
- `experiments/reversible_slot_grammar_plane_t33_coverage.py`
- `experiments/reversible_slot_grammar_plane_t33_postmortem.py`
- `tests/test_reversible_slot_grammar_plane_t33_stage0.py`
- `tests/test_reversible_slot_grammar_plane_t33_coverage.py`
