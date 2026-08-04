# Raw self-query functional record — Stage-0 decision

Status: **algebra/data gate passed; GPU capability result not yet run**  
Date: 2026-07-31

## Decision

Admit one preregistered GPU test of a quantized document-conditioned read
function.  Do not claim a model improvement from Stage 0.

The construction uses only each raw document's `document_id`, `title`, and
`text`.  Corpus-scale document frequency defines a quotient of local word
windows: common anchor words and relative positions remain, while the target
and rare values are erased.  A record must therefore implement

`document x relation-skeleton -> missing value`.

Identical skeletons across two documents also provide raw-only equality labels,
so the same records can be trained under a composed two-record read without a
teacher, parser, QA answer, or supporting-fact field.

## Frozen Stage-0 result

- 2,405 documents, all with disjoint train and evaluation reads;
- 108,570 unique functional edges with disjoint train/evaluation assignment;
- 3,392 single-valued ambiguous skeleton classes covering 2,248 documents;
- 2.5677 bits of conditional target entropy;
- 225 training equality classes and 562 positive cross-document pairs;
- 17 evaluation equality classes and 36 positive pairs;
- 267 multi-valued `(document, skeleton)` assignments excluded;
- unchanged logical record budget: 2,116,400 bits.

All seven Stage-0 gates passed.  The result JSON SHA-256 is
`f46a36b5c617bb62e18e9fc81111dc3df2a8e8d351d5808e6173deaa79c1d660`.

## Claim boundary

This proves that the corpus contains an automatically derived, causally
ambiguous read game.  It does not prove that a 220-cell record can learn the
function, that the function transfers to held-out natural questions, that it
beats stronger dense training, or that virtual records can be compiled into
ordinary weights.

The reused Hotpot development split is already development evidence.  Even a
pass there only admits physical export and a new untouched capability set.
