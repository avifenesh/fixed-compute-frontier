# Hotpot folded entity address T16 — preregistration

Status: frozen before execution  
Date: 2026-07-31

## Purpose

T15 found strong entity identity at the existing FFN interface but failed exact
routing under context shift.  T16 tests a train-time-only invariant address
projection that is algebraically folded into ordinary FFN key rows, adding no
served parameter or operation.

This remains an address feasibility gate.  It does not compile records, read
answer values, train the served model, or measure QA.

## Frozen state and integrity

- served checkpoint SHA-256:
  `a2900585a6e9afe4a9fba4f55afca30df5bd79c335d646cda5b9e940b72d693b`;
- terminal served state SHA-256:
  `53b3117727045ad31b52efd719b55fb251a557b2256008c924318c3cc244de57`;
- T15 result SHA-256:
  `b88c42915fd4b7e594a3fb337d3153c90db25be33fb3fce5a5a538c2044c36a0`;
- candidate corpus SHA-256:
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`;
- candidate titles: all 2,405 canonical titles.

The 36,577,152 served model parameters are frozen throughout.  Its checkpoint
must remain byte-identical.  The compiler projection is saved and hashed before
any natural question is loaded.

## Raw-prose-only compiler training data

For every canonical title `e`:

1. canonical view: a leading space followed by the title;
2. five context views: 32 consecutive tokens deterministically selected from a
   different candidate raw-prose document, followed by the same title tokens;
3. four context views are compiler-training views; the fifth is a raw-prose
   held-out context view.

Source documents and offsets are selected by SHA-256 of the title and view
number.  Every sequence is capped at the final 128 tokens.  No question,
answer, supporting-fact field, template, synthetic label, external corpus, or
external encoder enters compiler training.

The frozen served checkpoint encodes every view at the exact final-block
normalized pre-FFN interface.  The checkpoint is never differentiated.

## Frozen projection training

- projection: bias-free `P` with shape 64 by 384;
- seed: 9,109;
- initialization: orthogonal rows;
- updates: 800;
- batch: 256 titles, one of four training contexts selected independently;
- objective: cross-entropy over all 2,405 canonical titles using cosine logits
  between L2-normalized projected context and canonical vectors;
- temperature: 0.05;
- optimizer: AdamW, learning rate 0.003, zero weight decay;
- no schedule, retry, model update, or hyperparameter sweep.

Only `P` is optimized.  It is sealed to
`hotpot-folded-entity-address-t16-projection.pt` before question data is opened.

## Frozen serving fold

For canonical interface vector `h_e`, define address

`a_e = normalize(P h_e)`

and the served FFN key row

`k_e = P^T a_e`.

For any served query vector `h_q`, the unchanged FFN dot product is exactly

`h_q^T k_e = (P h_q)^T a_e`.

`P` is therefore absent at serving.  Its effect is already contained in the
ordinary 384-scalar key row that stores the entity record route.

## Frozen evaluation order

1. verify inputs and checkpoint state;
2. construct raw-prose views;
3. encode views with the frozen model;
4. train, seal, and hash `P`;
5. evaluate canonical and fifth-view raw-prose routing;
6. only then load the Hotpot split and evaluate all 292 training and 208
   development title surfaces; answer values are never accessed;
7. re-hash checkpoint and served model state.

Retrieval always compares against all 2,405 folded keys.  No threshold, title
subset, metric, layer, projection width, step, or seed choice follows result
observation.

## Admission gates

All must pass:

1. every integrity and algebraic-fold contract passes;
2. compiler training is finite with exactly 800 updates and no retry;
3. all 2,405 canonical titles retrieve themselves exactly with positive worst
   target margin;
4. all 2,405 held raw-prose context views retrieve their title exactly with
   positive worst target margin;
5. all 292 train-question and 208 development-question surfaces retrieve their
   title exactly with positive worst target margin;
6. the served checkpoint and model state remain unchanged;
7. the projection is sealed before questions are loaded;
8. zero answer-label reads and zero served-model updates occur.

Passing admits finite-capacity dense-record loading.  It does not demonstrate
record usefulness, record coexistence, QA gain, or a strict Pareto result.

Failure closes this linear raw-prose context-invariance protocol.  No
post-result projection dimension, augmentation, optimizer, or training sweep is
allowed.
