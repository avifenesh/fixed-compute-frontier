# Hotpot internal entity address T15 — preregistration

Status: frozen before execution  
Date: 2026-07-31

## Purpose

T12 through T14 close post-hoc property-similarity compilation.  T15 tests the
new boundary: can the exact contextual vector already entering an existing FFN
identify a natural multi-token entity title, so a compiled dense record can be
loaded before the ordinary model performs question-dependent reasoning?

T15 does not compile records, use answer labels, train weights, or measure QA.

## Frozen state

T15 uses the unchanged T12 checkpoint and T12 real-prose boundary.  Integrity
anchors include:

- checkpoint SHA-256:
  `a2900585a6e9afe4a9fba4f55afca30df5bd79c335d646cda5b9e940b72d693b`;
- terminal state SHA-256:
  `53b3117727045ad31b52efd719b55fb251a557b2256008c924318c3cc244de57`;
- T14 result SHA-256:
  `4e80b58fb9d2d6372b8093792ba57c133eb6ea27b78ec01b63330325cd019839`;
- candidate corpus SHA-256:
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`.

No optimizer, learned projection, adapter, title dictionary at inference,
external encoder, or parameter update is allowed.

## Frozen address algebra

The address interface is the exact normalized tensor entering the final
block's existing `gate.weight` and `up.weight` matrices, as contract-tested in
T13.

1. For each of all 2,405 canonical candidate titles, tokenize a leading space
   followed by the title without special tokens.  Encode at most the final 128
   tokens and retain the L2-normalized interface vector at the last real token.
2. For every title surface in the 146 training and 104 development natural
   questions, locate the case-insensitive exact character span, tokenize the
   full question with offset mappings, and retain the interface vector at the
   final token overlapping that span.
3. Retrieve the nearest canonical title by cosine similarity against all 2,405
   canonical keys.  Report top-1 accuracy and target-minus-best-competitor
   margins separately for training and development surfaces.
4. Repeat exactly on the untouched seed-8,209 initialization.

Supporting-title annotations identify evaluation targets only.  Answer values
are never read.  There is no threshold, pooling, layer, prompt, or metric
choice.

## Admission gates

All must pass:

1. every integrity check and exact-interface contract passes;
2. all 2,405 canonical keys are finite and have a positive nearest-neighbor
   margin;
3. trained top-1 retrieval is 100% on both train and development surfaces;
4. the trained worst-case target margin is positive on both sets;
5. checkpoint bytes and trained/random states remain unchanged;
6. zero training updates and zero answer-label reads occur.

Random accuracy is diagnostic, not an admission comparison: entity identity
may be supplied structurally by token identity before language training.

Passing admits a finite-capacity entity-record loading experiment.  It does not
show that raw prose can be compressed into useful records, that records can
coexist without interference, that natural QA improves, or that a strict
serving-cost Pareto gain exists.

Failure closes this untrained contextual title-address interface and requires a
co-trained address protocol rather than a representation sweep.

## Holdout accounting

These questions are a development resource.  T15 reads no answer values and
makes no capability claim.
