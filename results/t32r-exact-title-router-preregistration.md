# T32R exact-title router — CPU preregistration

Status: **FROZEN BEFORE EXECUTION**  
Date: 2026-07-31

## Claim

The frozen router is an autonomous deterministic compiler from raw question
text and the raw corpus-title dictionary to exactly two document handles on the
104-question domain.  It does not read evaluator support or answer fields.

It claims no alias, implicit-reference, fuzzy, or semantic retrieval.

## Frozen inputs

- corpus:
  `data/hotpot-real-prose-t12/candidate-raw-prose.jsonl`, SHA-256
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`;
- evaluator:
  `data/hotpot-real-prose-t12/sealed-evaluator.jsonl`, SHA-256
  `5a0413f53bfc0fc73b5e66c941c708077624908095934af740c84d234b9fea6b`;
- exactly 2,405 unique titles and 104 questions;
- frozen longest-first, non-overlapping, case-folded matcher from
  `scale_referenced_whole_record_t32_information_oracle.py`.

## Frozen tests

1. Every raw question routes to exactly two distinct corpus titles.
2. Only after routes are frozen, their unordered title pairs equal the sealed
   supporting-title pairs for all questions.
3. Swapping the case of every question does not change route indices.
4. Replacing the two matched title spans by `Entity A` and `Entity B` causes
   exact routing to reject every question.
5. Removing either matched title causes exact-two routing to reject.
6. Mutating sealed support-title fields without changing the raw question does
   not change any route.
7. Synthetic nested titles prove longest-first non-overlapping selection.
8. Corpus title order permutation does not change selected title strings.
9. No answer value is accessed; execution uses CPU standard library only.

## Kill conditions

One route failure, one support mismatch, one case change, one successful alias
route, one label-mutation effect, one overlap/order failure, or any GPU/model
access closes exact-title routing as an autonomous block.

## Non-claim

A pass establishes only exact-surface routing on this domain.  It does not
establish semantic structure extraction, rank-32 reader sufficiency, model
learnability, natural accuracy, or serving performance.

