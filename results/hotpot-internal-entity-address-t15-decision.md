# Hotpot internal entity address T15 — decision

Status: **FAIL exactness; RETAIN strong entity-identity signal**  
Date: 2026-07-31

## Decision

Do not load records using raw final-FFN contextual vectors.  The identity metric
misroutes too often for a digital memory.  Retain the entity-record boundary and
test one co-trained, foldable address protocol rather than returning to
property-similarity metrics.

## Evidence

- All seven contracts passed; zero answer values or training updates were used.
- All 2,405 canonical title vectors were distinct.
- Trained contextual title retrieval reached 85.6164% on 292 training surfaces
  and 81.2500% on 208 development surfaces.
- Random initialization reached 27.0548% and 30.2885%, respectively.
- The trained mean target margin was positive on both sets, but the worst
  margins were -0.078952 (train) and -0.199634 (development).
- The checkpoint remained byte-identical.
- Result artifact SHA-256:
  `b88c42915fd4b7e594a3fb337d3153c90db25be33fb3fce5a5a538c2044c36a0`.

## Interpretation

Unlike T12-T14 property matching, entity identity is strongly present in the
served interface.  The failure is contextual gauge drift: the same title alone
and inside a question does not land at exactly the same point.  An 81.25%
router is unusable, but the 51-point gain over random is sufficient evidence to
test an explicit invariant subspace.

## Direct refinement

T16 trains a single 64-by-384 linear projection using only candidate titles and
prefixes cut from candidate raw prose.  It never reads questions or answers
until the projection is sealed and hashed.  Canonical projected addresses are
folded back into 384-dimensional FFN key rows:

`k_e = P^T normalize(P h_e)`.

An ordinary served gate then computes

`h_q^T k_e = (P h_q)^T normalize(P h_e)`

with no served projection, parameter, or operation.  `P` is a train-time
compiler object and is discarded after key generation.

T16 must retrieve all 2,405 titles exactly from held-out raw-prefix contexts
and retrieve every natural-question title surface exactly.  Anything below
100% remains a failed digital address.
