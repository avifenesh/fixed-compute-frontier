# Digital relation plane T8 — preregistration

Frozen before any T8 training measurement: 2026-07-31

## Question

Can the same 110,776,960-parameter causal LM hold and exactly retrieve tens of
thousands of arbitrary facts at one amplitude-coded payload scalar per fact,
while ordinary calibrated Muon and AdamW training fail after two complete fact
exposures and twice the mixed-step budget—and while natural language modeling
remains protected?

This is a dense-knowledge test, not an NLL-improvement claim.

## Counting bound and construction

The table contains 512 entities, 64 relations, and one of 16 values for every
entity–relation pair: 32,768 arbitrary facts.  There are

```text
16^32768 = 2^131072
```

possible tables.  Any zero-error representation of an arbitrary table must
therefore distinguish at least 2^131072 states and carry at least 131,072
logical payload bits.

The candidate assigns the 16 values to the odd amplitudes `-15,-13,...,+15`
and stores exactly one such scalar in each entity–relation coordinate.  Its
32,768 payload scalars therefore carry exactly four logical bits each, or
131,072 logical bits total.  This meets the counting lower bound in logical
16-ary payload cells.  It is not a claim of physical bit optimality: the
checkpoint stores higher-precision floating-point scalars and has shared
decoder/isolation overhead.

RMSNorm would normally rescale those amplitudes differently for different
entity rows.  The compiler adds one compensation scalar per entity so every
entity has exactly the same squared norm, 14,401.  Two existing attention heads
copy the 64 amplitudes and compensation into a 6-bit relation query.  Sixty-four
shared SwiGLU channels select one relation; 48 shared channels decode the chosen
amplitude to a four-bit output code.  The decoder width is independent of the
number of entities/facts.

The sealed BF16 predecessor evaluated all 32,768 facts with zero errors, a
minimum correct-class margin of 6.625, and 6.62% accuracy on 4,096 unseen random
queries (16-way chance is 6.25%).

## Fixed deployed model

- Vocabulary 49,152, width 640, 16 pre-RMSNorm blocks, ten 64-wide heads,
  SwiGLU width 1,728, tied output embedding, and 110,776,960 parameters.
- Every arm has the exact same vocabulary, dense graph, checkpoint shape,
  precision, KV state, and inference FLOPs.
- The candidate uses 79 existing hidden coordinates, two heads, and at most 64
  channels in any FFN block.  It fixes 20,361,151 entries, mostly isolation
  zeros, and 38,211 nonzeros; 32,768 nonzeros are fact payloads.
- Entity, relation-query, and result IDs come from 925 token rows absent from
  both sealed natural streams.  The candidate adds no row, module, sidecar,
  retrieval store, expert, or runtime operation.

## Data and controls

The arbitrary fact table is fixed by seed 4,000,037.  Evaluation uses an
independent unseen-query seed 4,100,041.  Model seeds are 4,123, 4,487, and
4,889, with rotated arm order.

Every gradient control receives direct 16-way cross-entropy on every fact.
There is no hidden compositional inference requirement: the label presented is
the exact value to memorize.  The 2x controls receive two independently shuffled
complete passes over all 32,768 facts before 2,000 mixed steps.  Every twentieth
mixed step presents 256 additional fact queries; the other 19 use paired
FineWeb-Edu natural batches.

- `muon_1x`: calibrated hybrid Muon/AdamW, one complete fact pass, 1,000 mixed
  steps.
- `compiler_muon_1x`: direct table compilation, calibrated hybrid Muon/AdamW,
  no prefix-gradient pass, 1,000 mixed steps.
- `muon_2x`: calibrated hybrid Muon/AdamW, two complete fact passes, 2,000
  mixed steps.
- `adamw_2x`: AdamW at 3e-4, two complete fact passes, 2,000 mixed steps.

Muon uses the T7-sealed matrix learning rate 0.005, five Newton–Schulz steps,
`match_rms_adamw`, momentum 0.95, Nesterov updates, and AdamW at 3e-4 for the
embedding/norm vectors.  All arms use BF16 forwards, FP32 loss, clip norm 1,
and the same warmup/cosine schedule.

A schema-informed exact assignment control is algebraically identical to the
candidate transition, not a distinct learning baseline.  The adversarial
question here is whether current gradient training can use the same model and
direct labels to acquire comparable exact storage with more exposure/compute.

## Frozen gates

All gates must pass independently in all three seeds:

1. Every arm begins from an identical pre-compilation model hash.
2. At candidate mixed steps 250, 500, and 1,000, all 32,768 compiled facts are
   exact with zero errors and positive worst-case logit margin.
3. At those checkpoints, unseen random-query accuracy remains between 4% and
   9%, rejecting leakage or a constant lucky code.
4. Both twice-exposed, 2,000-step Muon and AdamW controls remain below 50%
   accuracy over the full arbitrary table.
5. Candidate natural validation NLL is no more than 0.5% above its paired
   `muon_1x` control at steps 250, 500, and 1,000 and improves from 500 to
   1,000.
6. All losses, pre-clip gradient norms, parameters, and evaluations are finite;
   maximum loss is below 100; no step is skipped/retried and no sample is
   discarded.
7. The deployed parameter count, graph, vocabulary, precision, and inference
   FLOPs remain identical across arms.

## Interpretation boundary

Passing would prove a large dense-knowledge capability change: 32,768 arbitrary
16-way facts, exact at every checkpoint, in the same deployed model where
twice-exposed frontier optimizers remain partial.  It would also establish a
logical payload construction at the counting lower bound.

It would not yet prove natural-language fact extraction.  The entity/relation
schema and unused token IDs are explicit, and the compiler is handed the parsed
table.  A pass must therefore advance to extraction from ordinary token
sequences and paraphrased/compositional queries before a general LLM-knowledge
claim.

Failure of exact retention, protected natural NLL, or control separation closes
this construction without changing thresholds, seeds, or code levels.

## Frozen integrity hashes

- Training runner: `3b7e6ddd1ca3d6c873d65896e83d5fb0e8dc4342d317582e426580a41965670d`
- Representation source: `950808d4493b6aa8307f0f481917022c7f8d64e9cd1bdf134a684f0177da9ca4`
- Structural tests: `0b63013ed5c03244878c5a312650340aa3ce792264fbb7bd379c73750d3792d7`
- Exact BF16 predecessor: `a213f320bfa88742688ad11714e4ec3a467353b8a0a645893afde083666980d5`

