# Hotpot distributed entity keys T17 — preregistration

Status: frozen before execution  
Date: 2026-07-31

## Purpose

T16's shared rank-64 address improved routing strongly but confused
class-specific near neighbors.  T17 is the sole analog successor: optimize the
actual per-entity key row that can be written into an existing FFN gate, and
distribute the 2,405 rows across three physically available 1,024-row FFNs.

T17 remains an address gate.  It does not yet overwrite the served checkpoint,
attach record payloads, read answers, or measure QA.

## Frozen state

- served checkpoint SHA-256:
  `a2900585a6e9afe4a9fba4f55afca30df5bd79c335d646cda5b9e940b72d693b`;
- served state SHA-256:
  `53b3117727045ad31b52efd719b55fb251a557b2256008c924318c3cc244de57`;
- T16 result SHA-256:
  `f3bb7cc9b2182334e5ec9faa1dbe6b779a38491e7fb06baa1ef0b51db620a917`;
- T16 projection SHA-256:
  `70861caae201e145e156bcadbe2330990f7a09ab98b91f912623b9cd2d29b08c`;
- candidate corpus SHA-256:
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`.

The served model is frozen and never differentiated.  Compiled keys are sealed
and hashed before any natural question is loaded.

## Frozen physical allocation

Entity title `e` is assigned to one of zero-based blocks 7, 8, and 9 by

`SHA256("t17-layer-" + e) mod 3`.

Each block receives at most 1,024 entities; this is checked before training.
The contextual input for a key assigned to block `l` is exactly that block's
normalized pre-FFN tensor, the vector multiplied by its existing gate row.

One unit-norm 384-scalar key is learned per entity.  The total is 923,520
scalars, or 2.525% of the existing 36,577,152 parameters.  These are not added
parameters: a successful key replaces one already executed `gate.weight` row
in its assigned block.  Tensor shapes, dense FFN MACs, and served graph remain
unchanged.

## Frozen raw-prose views

Reuse T16's exact deterministic construction:

- canonical leading-space-plus-title view;
- four training views, each with 32 tokens cut from SHA-selected candidate raw
  prose before the title;
- one held raw-prose prefix view;
- final 128-token cap;
- all 2,405 titles.

No question, answer, supporting annotation, template, external corpus, or
external encoder enters key training.

## Frozen key training

- initialization: the unit-normalized canonical interface at each entity's
  assigned layer;
- seed: 9,323;
- steps: 800;
- batch: 256 entities and one independently chosen training view;
- logits: for each query, dot its block-7 vector against block-7 keys, its
  block-8 vector against block-8 keys, and its block-9 vector against block-9
  keys; concatenate all 2,405 scores in canonical entity order;
- each key is L2-normalized inside the loss;
- common score scale: `1 / (sqrt(384) * 0.05)`;
- objective: cross-entropy to the entity identity;
- optimizer: AdamW at 0.003, zero weight decay;
- no schedule, retry, projection, bias, title subset, or sweep.

Only the key matrix is differentiated.  Unit-normalized final keys plus their
layer assignments are sealed to
`hotpot-distributed-entity-keys-t17-keys.pt` before questions are opened.

## Frozen evaluation

For each title occurrence, compute all three physical interface vectors and
all 2,405 physically assigned row scores.  Retrieval is the single largest raw
dot product across all rows and all three blocks.  No L2 normalization of the
query is performed; only stored rows are unit norm, matching the served gate.

Evaluate, in order:

1. all canonical title views;
2. all four training raw contexts;
3. all held raw contexts;
4. after key sealing, all 292 train-question and 208 development-question title
   surfaces, without reading answer values.

## Admission gates

All must pass:

1. every integrity, allocation, and exact-interface contract passes;
2. no layer receives more than 1,024 keys and the stored-key ledger is exactly
   923,520 scalars;
3. training is finite with exactly 800 updates and no retry;
4. canonical, training-context, held-context, train-question, and
   development-question retrieval are each 100%;
5. every set has a strictly positive worst target-minus-competitor margin;
6. keys are sealed before questions are opened;
7. checkpoint bytes and served model state remain unchanged;
8. zero answer reads and zero served-model updates occur.

Passing admits record-payload compilation plus an actual same-shape model
write.  It does not prove that overwriting the rows preserves language quality,
that records are useful, that QA improves, or that a strict Pareto gain exists.

Failure closes analog contextual entity addressing.  The next admissible
address would require an explicit discrete sequence sideband, not another
analog rescue.
