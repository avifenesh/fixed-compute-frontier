# T32R exact-title router — CPU decision

Status: **FAIL; CLOSE LONGEST-TWO LITERAL MATCHER AS AUTONOMOUS ROUTER**  
Date: 2026-07-31

## Verdict

The source-level provenance correction stands: routing uses raw question text
and corpus titles, not sealed support labels.  But the frozen exact-title router
failed its stronger autonomous-correctness gate.

It produced two distinct literal matches for all 104 questions, was
case-invariant, rejected every two-title alias replacement, and was invariant
to corpus title order.  Only 102 routes matched the sealed supporting-title
pairs.  Six single-title deletions exposed another valid corpus-title mention
instead of causing exact-two rejection.

## Counterexamples

1. `Hungry Hungry Hippos` and `Parcheesi` are the compared entities, while
   `Parker Brothers` is a third corpus title in the predicate.  Longest-two
   matching chose `Hungry Hungry Hippos` and `Parker Brothers`.
2. `Hanafuda` and `Okey` are compared, while `Card game` is also a corpus title.
   The matcher chose `Hanafuda` and `Card game`.

The failure is structural: literal membership cannot distinguish grammatical
arguments from title-shaped predicate or category mentions.  More aggressive
substring ranking is not a semantic fix.

## Decision boundary

Close only this operator:

```text
raw question + title dictionary
    -> choose the two longest non-overlapping literal title matches
```

Retain:

- raw-only route provenance;
- literal matching as a candidate generator;
- the 102/104 exact-support result as a measured primitive;
- case and corpus-order invariance;
- the declared alias/implicit boundary.

Do not repair it by hand-coding these question templates or consulting support
labels.  A successor must type mentions by their role in the question using a
raw-only learned or provably recoverable operator, and it must compare against
the literal candidate generator.

## Effect on prior T32 evidence

The information oracle intentionally routed from raw question strings and did
not use support labels.  Its +40-point correct-versus-shuffled result therefore
remains valid for **the records selected by the frozen raw router**.  It does
not establish perfect supporting-document retrieval.  The two mismatches make
the earlier broad “correct two documents” wording too strong and must be
carried as a provenance qualification.

No H100, model, corpus mutation, retry, or threshold change occurred.

