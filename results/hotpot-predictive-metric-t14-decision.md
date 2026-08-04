# Hotpot predictive metric T14 — decision

Status: **CLOSED NEGATIVE for the model-output predictive metric**  
Date: 2026-07-31

## Decision

Do not continue searching for a post-hoc similarity metric between questions
and raw-document states.  `M = E^T E / V` failed, and T12 through T14 now close
the static, Euclidean-contextual, and model-native predictive variants of that
family at the frozen scale.

## Evidence

- Seven implementation/algebra contracts passed on the H100.
- The trained metric was highly non-isotropic: condition number 292,438.69,
  versus 1.414 at random initialization.  It therefore changed the geometry
  materially rather than reproducing Euclidean cosine.
- Despite that change, the trained arm reached only 57.5342% on its fitting set
  and 43.2692% on the 104-question development evaluator.
- Random initialization reached 49.0385% on the evaluator.
- The trained arm failed the 70% absolute gate, the 10-point lexical-improvement
  gate, and all three superiority comparisons.
- The checkpoint and both model states remained unchanged; zero updates ran.
- Result artifact SHA-256:
  `4e80b58fb9d2d6372b8093792ba57c133eb6ea27b78ec01b63330325cd019839`.

## Interpretation

Next-token predictive similarity is not relation-truth identity.  A question
and a document can induce related output distributions without the minimum
page similarity separating whether both named entities share the asked
property.  The result rejects the premise that a fixed post-hoc metric can make
the compiler understand the query.

No centering, whitening, Fisher, learned bilinear, layer, pooling, or metric
sweep follows T14.  Those would continue the rejected family after observing
the evaluator.

## Architectural reframe

The compiler should not semantically match arbitrary properties.  It should:

1. address a named entity exactly;
2. load a dense entity record compiled once from that entity's raw prose;
3. let the served model's ordinary attention and FFN computation select and
   reason over the loaded records under the natural question.

This changes the writer/reader boundary.  The compiler only maps
`entity -> record`; it never sees the question.  Query semantics remain the
served model's job.  A record is read through existing FFN channels, so the
served parameter count and operation graph need not grow.

T15 tests the first necessary condition: whether the existing final-FFN
interface can identify natural multi-token title spans against all 2,405
candidate titles without external semantic matching.  It is still only a
development feasibility gate.
