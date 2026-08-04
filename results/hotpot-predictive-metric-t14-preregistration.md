# Hotpot predictive metric T14 — preregistration

Status: frozen before execution  
Date: 2026-07-31

## Purpose

T12 rejected static token rows as a semantic key space.  T13 rejected direct
Euclidean matching between natural-question and raw-document contextual states.
T14 tests whether the language model's own learned output geometry supplies the
missing writer/reader coordinate transform without adding any serving work.

This is a fixed-checkpoint privileged development diagnostic.  It neither
trains nor compiles a digital plane.

## Frozen state and inputs

T14 uses the exact T12 shared-base checkpoint, model, tokenizer, candidate raw
prose, 146-question threshold-fitting set, and 104-question development set.
The frozen integrity anchors are:

- checkpoint SHA-256:
  `a2900585a6e9afe4a9fba4f55afca30df5bd79c335d646cda5b9e940b72d693b`;
- terminal model-state SHA-256:
  `53b3117727045ad31b52efd719b55fb251a557b2256008c924318c3cc244de57`;
- T13 result SHA-256:
  `a09fc367ad52bfe0b357690e9b0aed6e252c5c774b9efaeae8b8f8c906347329`;
- candidate corpus SHA-256:
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`.

No optimizer, update, learned projection, adapter, centering, whitening,
temperature, or new parameter is permitted.  Checkpoint bytes and model state
must remain unchanged.

## Frozen algebra

Let `E` be the checkpoint's tied `V x H` input/output embedding matrix and let
`z(x)` be the ordinary final normalized hidden state for a token position.
Freeze

`M = E^T E / V`.

For every raw-document token state `d`, the offline compiler key is

`k(d) = M d`.

For a natural-question state `q`, the read score is

`q^T k(d) = q^T E^T E d / V = logits(q)^T logits(d) / V`.

Thus T14 compares the complete standard next-token logit geometry while the
served read remains one ordinary dot product against a precomputed row.  The
compiler does not contain a model and cannot answer a query; it applies a fixed
384-by-384 matrix derived from the served model's own weights.

## Frozen encoding and score

- Raw page prose is tokenized without special tokens, truncated to 512 tokens,
  and encoded in consecutive non-overlapping 128-token chunks.
- Every real-token final normalized hidden state is transformed by `M` offline.
- The full question is tokenized without special tokens, truncated from the
  left to 128 tokens, and its final real-token final normalized hidden state is
  the query.
- For each of the two gold supporting pages, take the maximum `q^T k(d)` over
  page tokens; take the minimum of those two maxima.
- Fit one scalar threshold and one direction on the 146 training questions;
  evaluate unchanged on the 104 development questions.
- Run the exact same algebra on the untouched seed-8,209 initialization.

There is no normalization in the `M` metric beyond the model's existing final
RMS normalization and the fixed division by vocabulary size.  No alternative
metric or pooling is allowed.

## Admission gates

All must pass:

1. every integrity check passes and all scores are finite;
2. a contract test proves `q^T M d` equals the full-logit dot product divided
   by `V`;
3. checkpoint bytes and trained/random model states remain unchanged;
4. trained predictive-metric accuracy is at least 70%;
5. it is at least 10 percentage points above the 55.7692% lexical reference;
6. it exceeds random initialization, T12 static accuracy (60.5769%), and T13
   direct-contextual accuracy (53.8462%);
7. zero training updates occur.

Passing admits a separately preregistered finite-capacity compilation test on
a newly frozen holdout.  It does not prove that many keys can coexist, that a
writer can assign values, that QA improves, or that a production Pareto gain
exists.

Failure closes `M = E^T E / V` at this scale.  It does not admit a post-result
metric family sweep.

## Holdout accounting

The evaluator remains a development set because its T12 aggregate was observed
before T14.  No result here is fresh held-out evidence.
