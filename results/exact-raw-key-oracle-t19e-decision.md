# Exact raw-key oracle T19e — decision

Status: **FAIL; CLOSE lexical locally-decodable indexing**  
Date: 2026-07-31

## Decision

Do not compact, hash, quantize, widen, stem, or otherwise rescue raw lexical
keys.  The collision-free oracle is strictly more informative than any Bloom
filter, perfect hash, CountSketch, or locally decodable code over the same
unigram/bigram/year/decade key set, and it failed.

The next representation must change what a stored item *means*.  It must not
merely change how the same lexical item is located.

## Evidence

- All five implementation contracts passed on the rented H100 host.
- The raw-only compiler assigned one exact coordinate to each of **129,971**
  distinct keys; there were no hashes, collisions, truncation, quantization,
  or physical-capacity constraints.
- The exact reader fitted all 146 training examples at **100%**.
- Held-out accuracy was **56.7308%** (59/104), failing the 75% gate.
- T19b's compressed CountSketch was **60.5769%**; removing every collision
  made the result **3.8462 points worse**, not better.
- Query-only accuracy was also **56.7308%**.  Exact document payloads added
  zero held-out points over question wording.
- A fixed document shuffle reduced accuracy by only **5.7692 points**, below
  the required 10-point causal drop.
- The optimizer converged in 34 iterations with finite weights and logits.

Result SHA-256:
`415ba8a2242f00bc8e86dd17d6507613b04f7a75dae33a00437089c2f392201c`.

Executed source SHA-256:
`efefc775e2173441be16ce83dea602e280842be57a5b5b284ee31e2d0d1a4dd4`.

Preregistration SHA-256:
`6ae71b6d66fd5abd848d21f76b352b7e21272bcec0b496ea455e7744b6ce2bd6`.

## What failed

Lexical presence is not relation truth.  The questions require typed claims:
occupation, nationality, chronology, membership, medium, comparison, and
negation.  A collision-free word dictionary knows that a page contains
`actor`, a year, or a name, but not which entity-role-value assertion those
tokens instantiate.  The train reader can memorize sparse predicates, but
title-disjoint development pages expose that no shared relation algebra was
learned.

This separates the boundary cleanly:

1. T19d: raw residual information fits, but entropy decoding is sequential.
2. T19e: exact random access is cheap, but lexical keys are semantically
   insufficient.

Storage and lookup are independently solved; the missing object is a
raw-trained, relation-bearing sufficient statistic.

## Admissible successor

The next gate may use the model's *functional learning signal*, not another
post-hoc state similarity.  For raw token context `h_t`, target `y_t`, tied
embedding `E`, and model distribution `p_t`, define the hidden prediction
residual

`r_t = E[y_t] - p_t E`.

The document's first-order learning operator is

`G_d = sum_t h_t outer r_t`.

This operator records how the raw document would change the model's behavior,
not merely which words occur.  A raw-only separable projection
`M_d = A^T G_d B` can fit in the same 220-cell record and a query operator can
interact with it through the existing SwiGLU elementwise product.  It earns an
experiment only with an exact query-only control, random-subspace control,
document-shuffle ablation, BF16/write ledger, and the same 75%/+10-point gates.

This is not yet evidence that functional residual memory works.  It is the
first successor that changes the algebra implicated by the failure.
