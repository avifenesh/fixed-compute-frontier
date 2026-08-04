# Raw-prose equality plane T10 — matched-training pilot protocol

Status: frozen before pilot execution  
Date: 2026-07-31

## Purpose

This pilot is allowed to run only after the exhaustive BF16 T10 representation
artifact passes.  It tests the strongest causal objection to the candidate:
perhaps all gain comes from giving a compiler structured facts that ordinary
training never receives.

Every gradient control therefore receives:

1. the exact same raw prose strings;
2. one compiler-derived direct QA label for every recovered fact;
3. compiler-derived equality demonstrations on a support subset;
4. the same mixed natural/knowledge continuation stream.

The candidate receives the same raw corpus and extracted table, but writes the
table directly and deletes the compiler at export.

## Frozen data split

The corpus, model, tokenizer, initialization, and T10 layout are unchanged.

Direct-label training contains exactly one query per canonical entity/relation:
the lexicographically first alias, first role-A anchor, and first role-B anchor.
All other aliases and role-A surfaces are held out.

Equality-label training contains one surface for canonical entities whose
index is divisible by four and ordered relation pairs satisfying
`(relation_a + relation_b) mod 2 = 0`.  Evaluation excludes every exact labeled
four-token query and reports the remaining exhaustive surface combinations.

Thus controls are given all fact values and a substantial equality curriculum,
but held-out evaluation still requires alias, paraphrase, and composition
generalization.

## Frozen arms

- `compiler_muon_1x`: direct write, no gradient prefix, 1,000 mixed steps;
- `muon_labels_1x`: one raw/direct/equality prefix pass, 1,000 mixed steps;
- `muon_labels_2x`: two prefix passes, 2,000 mixed steps;
- `adamw_labels_2x`: two prefix passes, 2,000 mixed steps.

The Muon rate is the previously calibrated `0.005`; AdamW uses `3e-4`.  Each
prefix pass presents every raw sentence once, every direct fact label once, and
every equality support label once.  Prefix order and arm order are seeded.

Every twentieth mixed step is a knowledge step, alternating direct and
equality batches.  Other steps use the protected document-disjoint natural
stream.  Candidate program entries are gradient-masked and restored after
every update.

## Pilot gates

The single-seed pilot advances to a three-seed sealed run only if:

1. T10 exhaustive BF16 representation gates passed before training;
2. candidate direct and held-out-equality accuracy remain exactly 100% at
   steps 250, 500, and 1,000, with positive minimum margins;
3. candidate natural NLL is within 0.5% of `muon_labels_1x` at matched mixed
   checkpoints;
4. the best twice-trained control remains below 90% on either held-out direct
   or held-out equality, creating a qualitative—not sub-percent—gap;
5. all arms start from the identical random checkpoint and remain finite;
6. raw strings, derived targets, optimizer updates, token presentations, GPU
   seconds, HBM, and writer/compiler costs are fully reported.

If both twice-trained controls exceed 90% on both held-out capabilities, this
scale is non-discriminating and cannot support a breakthrough claim.  The next
test must increase independently addressable facts or linguistic ambiguity,
not weaken the controls.
