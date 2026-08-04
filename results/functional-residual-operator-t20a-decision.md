# Functional residual operator T20a — decision

Status: **FAIL; CLOSE post-hoc energy-optimal learning operators**  
Date: 2026-07-31

## Decision

Do not tune the HOSVD ranks, checkpoint layer, context length, residual sign,
normalization, covariance weighting, reader, or threshold.  The frozen
first-order operator is closed.

The result rejects a stronger premise than “220 dimensions were too small.”
The raw HOSVD captured most of the compressible learning-operator energy and
still carried no causal held-out relation signal.  A better low-rank
approximation to the same next-token gradient is therefore not the missing
knowledge representation.

## Evidence

- All five pre-run algebra/integrity tests passed on the rented H100.
- The raw-only 20x11 HOSVD retained **53.7813%** of normalized document
  operator energy in 220 cells.
- A same-size random orthonormal subspace retained only **0.1518%**.  The
  learned subspace therefore changed the representation radically and was not
  a random sketch in disguise.
- Minimum BF16 cosine was **0.9999963** and every payload had unit norm within
  `1.79e-7`.
- The physical write was exactly **743,670** existing parameter entries, with
  identical 36,577,152-parameter state schemas.
- The HOSVD reader fitted all 146 training examples at **100%**.
- Held-out HOSVD accuracy was **44.2308%** (46/104), failing 75% and scoring
  **16.3462 points below** T19b.
- Random-subspace accuracy was **50.0000%**.
- Query-only accuracy was **59.6154%**; adding the HOSVD documents reduced
  accuracy by **15.3846 points**.
- The fixed document shuffle produced exactly the same **44.2308%** score: a
  **0-point causal drop**.
- Peak HBM allocation was 1.711 GB; the failure is not a runtime artifact.

One non-capability contract also missed narrowly: float32 eigenspaces had
orthonormal max errors `1.92e-4` and `1.75e-4` against a frozen `1e-4` limit.
This cannot explain the result: payload fidelity and norm contracts passed,
the HOSVD/random energy separation was 354x, and document shuffling had zero
effect.

Result SHA-256:
`c108b1a7a7f4531ec70858657fb81b1f0aa7bfa6b649c2f1a5cbf37b30b9a327`.

Executed source SHA-256:
`033af02eebfc49616eaf62ba330e7913fd6298f3f71c95e276b068a77bed0d25`.

Preregistration SHA-256:
`29b5ec50896be72886b038121c44b36020817e79c13156fb74d24dfadfaffc8d`.

## What the result establishes

For next-token loss, high energy is not high semantic utility.  The dominant
gradient directions encode frequent surface, syntax, and prediction
corrections.  Relation truth can live in low-energy directions that HOSVD is
specifically designed to discard.  Conversely, retaining over half of total
gradient energy does not make two title-disjoint pages comparable under a
natural property question.

Together T12-T20 now reject four post-hoc assumptions:

1. ordinary contextual geometry exposes exact entity keys;
2. pooled hidden state is a stable semantic schema;
3. exact lexical keys expose typed relation truth;
4. energy-optimal next-token gradients expose typed relation truth.

The common failure is **non-identifiability**.  Raw likelihood constrains the
model's output distribution but does not select a unique internal
entity/relation/value factorization.  Any invertible hidden gauge can preserve
the same likelihood while changing the coordinates a post-hoc compiler tries
to interpret.  More importantly, many predictors with equal likelihood need
not make paraphrased relation truth linearly or bilinearly addressable.

## Key shift

Stop extracting a semantic plane after ordinary LM training.  The plane must
be part of the from-zero learning contract.

The next architecture must make relational structure identifiable through a
symmetry or intervention that ordinary next-token prediction lacks.  A viable
contract is entity-renaming equivariance:

- a consistent permutation of entity identities must permute only entity
  addresses;
- the relation/value program must remain invariant;
- swapping entity addresses between records must swap which entity owns the
  relation/value program without changing the program itself;
- raw reconstruction prevents the invariant code from collapsing.

This is an architectural training constraint, not a richer teacher or
post-hoc metric.  It targets the precise ambiguity exposed by the failures.
The next experiment must be a matched from-zero candidate/control comparison,
not another frozen-state payload probe.
