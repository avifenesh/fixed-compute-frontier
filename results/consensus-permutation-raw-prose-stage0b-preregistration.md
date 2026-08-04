# Consensus-permutation raw-prose compiler — corrected Stage 0b preregistration

Status: frozen after invalidating Stage 0 XOR and before Stage 0b execution  
Date: 2026-07-31

## Correction being tested

Stage 0's direct extraction was valid, but its XOR composition metric used the
generator-private mapping from opaque value words to integers.  Stage 0b
removes that hidden semantic dependency.

The corrected held-out operation is categorical equality:

```text
equal(entity, relation_a, relation_b)
    := recovered_value(entity, relation_a)
       == recovered_value(entity, relation_b).
```

Equality is unchanged by every permutation of the opaque value vocabulary.
The compiler can therefore compute it using recovered surface strings alone;
it needs no value labels, numerical interpretation, truth table, or operation
example.

## Frozen corpus and extractor

The world is unchanged from Stage 0:

- 128 entities, 6 relations, and 16 opaque value words;
- 3 disjoint alias views and 2 disjoint prose frames per relation/view;
- 63 mentions per fact/frame with 15% independent contradictions;
- shuffled raw strings as the compiler's only input;
- pair-mask literal frame induction, majority recovery, relation alignment,
  and complete-signature alias alignment.

Relation histograms may be used as an alias-permutation invariant when unique;
otherwise the compiler may exhaust relation permutations for at most eight
relations.  Both routes must be verified by complete row-signature agreement.

## Frozen gates

All must pass:

1. Exact frame count, one-frame-per-sentence coverage, and exact alias-view
   count.
2. 100% direct fact, entity-alias, and relation-paraphrase recovery.
3. 100% categorical equality on every entity and ordered relation pair, with
   the comparison performed directly on recovered opaque strings.
4. Independent-view and duplicate-signature null corpora are rejected.
5. Majority-error union bound is below 1%.
6. `compile_corpus` accepts only raw strings.
7. The two-channel SwiGLU product identity is numerically exact on bipolar bits;
   this is only the algebra needed for later in-model equality, not evidence
   that integration already works.

## Admission rule

Passing admits a GPU **representation integration** test, not a breakthrough
claim.  That test must write only compiler-recovered facts, execute direct
lookup and equality inside an unchanged Transformer, and compare against
gradient controls that receive the same raw prose and compiler-derived labels.
