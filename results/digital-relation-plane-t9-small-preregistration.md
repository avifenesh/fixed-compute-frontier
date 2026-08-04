# Digital relation plane T9 — second-scale preregistration

Frozen before any T9 training measurement: 2026-07-31

## Purpose

T8 installed 32,768 arbitrary 16-way facts exactly in a 110.8M-parameter LM
while twice-exposed Muon and AdamW controls reached only 11.86% and 7.27% on
average.  T9 is the untouched second-scale replication required before the
dense-memory result can satisfy the project breakthrough contract.

No fact-count reduction is allowed.  The 36,577,152-parameter model must carry
the same 32,768 facts and 131,072 logical payload bits at one scalar per fact.

## Frozen model and construction

- Vocabulary 49,152, width 384, ten pre-RMSNorm blocks, six 64-wide heads,
  SwiGLU width 1,024, tied output embedding, context 128, and 36,577,152
  deployed parameters.
- The digital plane uses 79 existing hidden coordinates, two existing heads,
  and at most 64 channels in a block.  Four language heads remain.
- Model graph, parameter count, vocabulary, precision, KV state, checkpoint
  shape, and inference FLOPs are identical across arms.
- The new fact table uses seed 5,000,051; unseen evaluation uses seed 5,100,053;
  model seeds are 5,123, 5,461, and 5,819 with rotated arm order.
- The sealed BF16 quick result has zero errors over all 32,768 facts, minimum
  margin 2.625, and 5.86% unseen random accuracy.

The same norm-compensated 16-level payload, 6-bit relation router, and shared
48-channel decoder are used without post-T8 modification.  At this scale the
candidate fixes 9,872,267 entries, mostly isolation zeros, and 37,263 nonzeros.

## Independent optimizer calibration

A natural-only development pilot tested AdamW 3e-4 and hybrid Muon rates 0.0025,
0.005, 0.01, and 0.02 for 250 steps from one excluded seed.  The frozen rule
selected the finite Muon arm with lowest terminal NLL.  Rate 0.005 won at
6.60553, versus 6.68001 for AdamW and 6.62350/6.68628 for adjacent Muon rates.
The pilot artifact SHA-256 is
`1dd04991690a7c1867d84a1ec5d88a068b56be5cd1506186ef07c08ed7df6026`.

## Arms

Every control receives direct 16-way labels for all facts.  The 2x controls get
two complete independently shuffled table passes plus 2,000 mixed steps.  The
1x control gets one table pass plus 1,000 mixed steps.  The compiler consumes
the same table directly and runs 1,000 mixed steps.  Every twentieth mixed step
uses 256 fact queries; the rest use paired sealed FineWeb-Edu batches.

- `muon_1x`
- `compiler_muon_1x`
- `muon_2x`
- `adamw_2x`

Muon uses the independently selected 0.005 matrix rate with
`match_rms_adamw`; embeddings/norms use AdamW 3e-4.  BF16 forwards, FP32 loss,
clip norm 1, zero weight decay, and the schedule match across applicable arms.

## Frozen gates

All gates must pass in all three seeds:

1. Identical pre-compilation initialization hashes across arms.
2. Candidate fact accuracy is exactly 100%, with zero errors and positive
   worst-case margin, at mixed steps 250, 500, and 1,000.
3. Candidate unseen random-query accuracy stays between 4% and 9% at all three
   checkpoints.
4. Both twice-exposed Muon and AdamW controls remain below 50% full-table
   accuracy at step 2,000.
5. Candidate natural NLL is no more than 0.5% above paired Muon 1x at steps
   250, 500, and 1,000, and improves from 500 to 1,000.
6. All losses and pre-clip gradients are finite, maximum loss is below 100,
   no step is skipped/retried, and no sample is discarded.
7. Deployed resources and serving computation remain identical.

Passing, together with T8, establishes a two-scale, six-seed qualitative
acquisition result at the counting-bound logical payload: exact dense knowledge
where calibrated controls fail with twice the exposure and mixed budget.

It still does not establish natural-text extraction.  The table schema and
special entity/relation IDs are explicit.  That limitation remains the next
research front after replication, not a reason to relabel this result as a
general natural-language benchmark gain.

## Frozen integrity hashes

- Training runner: `2580f5902546404f406782f2fd92cf5216e032061ac2574f6e746007a8fa0860`
- Core/compiler source: `4770f93d63be76e73d040b06837c6c3ec857d7a7c1e60762fa1bcf8a47c03b00`
- Muon pilot: `1dd04991690a7c1867d84a1ec5d88a068b56be5cd1506186ef07c08ed7df6026`
- Exact quick predecessor: `2dd20d2f56eab1b1a46ae78d851c6e9944d70d11308c83d1d598810e6aefdd6f`

