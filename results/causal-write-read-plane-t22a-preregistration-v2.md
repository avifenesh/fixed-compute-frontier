# Causal write/read plane T22a-v2 — preregistration addendum

Status: **FROZEN AFTER CPU PREFLIGHT, BEFORE ANY T22 GPU TRAINING OR RESULT**  
Date: 2026-07-31

This file supersedes the writer-held-out-set clauses of
`causal-write-read-plane-t22a-preregistration.md`, whose SHA-256 is
`e6517016b796caef7d4ba8d4e628efa38d5aa16dd3b508d96f84d01205554445`.
Every other model, data, objective, seed, optimizer, schedule, compute ledger,
control, threshold, gate, and decision rule in that preregistration remains
unchanged.

## Why a pre-result addendum is required

The frozen CPU preflight applied T21a's label-blind router: among all raw titles
appearing in a question, greedily choose the two longest nonoverlapping spans.
It selected the sealed supporting-title pair on 103 of 104 development rows.
On row `5ac0d9a35542992a796ded90` it selected `Hungry Hungry Hippos` and
`Parker Brothers`, while the sealed support pair is `Hungry Hungry Hippos` and
`Parcheesi`:

`Are Hungry Hungry Hippos and Parcheesi both published by Parker Brothers?`

Changing the router to the support pair would leak evaluator structure and
erase a real semantic-routing error.  The router therefore remains unchanged,
and the mismatched row remains fully priced in development accuracy.

## Corrected writer-held-out set

Before any T22 model update, the benchmark harness forms the union of:

1. all 208 distinct sealed development supporting titles; and
2. every raw title selected by the unchanged label-blind development router.

The frozen union contains exactly **210** raw documents.  The writer learns
only from the remaining **2,195** documents.  After specialization, every one
of the 210 held-out documents receives exactly one gradient-free writer pass.
The writer and its loss still receive no question, answer, supporting-title,
supporting-sentence, or evaluator field.

The dense-causal control may receive gradients from all 2,405 documents,
including all 210 held-out documents.

## Corrected isolation gate

Gate 4 is replaced by all of the following:

- the held-out set contains exactly 210 documents and the writer-training set
  exactly 2,195;
- every title selected by the development raw-title router belongs to the
  writer-held-out set;
- no title selected by a QA-training row belongs to the writer-held-out set;
- learned-writer specialization presents zero held-out documents to a
  gradient;
- each held-out document has writer compile minimum = maximum = 1, for exactly
  210 total held-out compile passes.

Held-out read evaluation therefore contains
`210 * 8 = 1,680` frozen probes.  All capability thresholds and the requirement
for at least 75% development QA remain unchanged.

