# Hotpot contextual address T13 — decision

Status: **CLOSED NEGATIVE for untrained Euclidean matching at the final FFN interface**  
Date: 2026-07-31

## Decision

Do not compile property keys by copying raw-document contextual vectors into
the final FFN gate.  The exact `M = I` contextual-cosine interface is closed at
the frozen scale.

## Evidence

- All eight implementation contracts passed on the H100.
- The tested vector was exactly the normalized tensor consumed by the existing
  final-block `gate.weight` and `up.weight` matrices.
- No parameter update occurred.  The checkpoint remained byte-identical at
  SHA-256
  `a2900585a6e9afe4a9fba4f55afca30df5bd79c335d646cda5b9e940b72d693b`.
- The trained checkpoint reached only 57.5342% on its threshold-fitting set and
  53.8462% on the 104-question development evaluator.
- Random initialization reached 47.1154% on the evaluator.
- The trained contextual interface was worse than T12's trained static
  interface at 60.5769%, below the 70% absolute gate, and below the required
  10-point gain over the 55.7692% lexical reference.
- The result artifact has SHA-256
  `a09fc367ad52bfe0b357690e9b0aed6e252c5c774b9efaeae8b8f8c906347329`.

The low fitted-training accuracy matters: the failure is not explained by a
threshold that happened not to transfer.  Yes/no cases are not separated in
this Euclidean coordinate system.

## Interpretation

An ordinary LM has no objective requiring a question-final state and a raw
document-token state to share an identity metric.  T13 assumed that they did.
This assumption is now rejected; a layer or pooling sweep would not repair the
missing writer/reader coordinate contract and is forbidden by the protocol.

This is not yet a digital-plane capacity wall.  A compiler key need not equal
the raw document vector.  If the model supplies a fixed metric `M`, the compiler
can write `M d` offline and the unchanged serving matrix can read it with the
existing dot product `q^T (M d)`.

## Direct refinement

T14 tests the one model-native metric already trained by next-token prediction:
`M = E^T E / V`, where `E` is the tied output embedding matrix.  This makes
`q^T M d` exactly the average dot product of the two states' complete standard
logit vectors.  `M d` is computed only by the offline compiler; serving still
performs one existing row dot product.

T14 remains a development diagnostic on the previously observed evaluator.
Failure closes this exact predictive metric; it does not permit metric,
centering, whitening, layer, or pooling sweeps.
