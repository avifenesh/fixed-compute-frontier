# Causal write/read plane T22a — preregistration

Status: **FROZEN BEFORE IMPLEMENTATION RESULT**  
Date: 2026-07-31

## Claim under test

Can the same from-zero Transformer learn a raw-prose write function that emits
one finite record per document, then use one-pass records for documents that
never supplied candidate gradients to outperform an equal-or-stronger dense
training control on title-disjoint natural QA at identical serving cost?

T22a is a virtual representation-and-reader gate.  Passing admits, but does
not replace, physical compilation into the existing ordinary-Transformer
entries, final untouched evaluation, and replication.

## Frozen model, data, and split

- Same 36,577,152-parameter, 384-wide, ten-layer, six-head causal Transformer,
  tokenizer revision, protected natural streams, 2,405 raw-only documents,
  146 QA-training rows, and 104 development rows as T21a.
- Model seed 8,209; probe seed 22,001; fixed-random-record seed 22,003;
  shuffle seed 22,005.
- The benchmark harness takes the 208 supporting titles already sealed in the
  development file as the writer-held-out title set.  It exposes no question,
  answer, support sentence, or support field to the writer or its loss.
- Candidate write/read specialization samples only the other raw documents.
  Every held-out raw document is consumed exactly once by the frozen writer
  after specialization, with no gradient or per-document optimization.
- Dense causal specialization may train on all 2,405 raw documents, including
  writer-held-out prose.  This deliberately strengthens the ordinary control.

## Common natural base

Train one shared base for exactly 19,000 protected-natural updates, batch 16,
context 128, BF16 forward, Muon/AdamW peak rates 0.005/0.0003, 500-step warmup,
cosine-to-10%, clip 1.0, and no retry.  Clone its state byte-identically into
all specialization arms.

## Same-model discrete writer

For a raw document `X`, the ordinary model computes its contextual state.  The
record is the fixed 220-coordinate slice of the final active token:

`C_theta(X) = 4 Q16(tanh(h_theta(X)_last[64:284]))`.

`Q16` rounds to the 16 uniformly spaced odd levels
`{-15,-13,...,15}/15` and uses a straight-through derivative.  There is no
external encoder, learned projection, per-document parameter, teacher, parser,
generated question, or pretrained model.  The same Transformer writes and
reads.  After specialization, all records are recomputed once and frozen.

The table has `2,405 * 220 = 529,100` four-bit logical cells.  The prospective
ordinary-graph ledger remains exactly 743,670 existing entries, below the
743,734 cap.

## Raw-only read probes

Each document is tokenized as `title + newline + text`, truncated to leave one
answer-carrier position.  A deterministic probe:

1. chooses one non-title, non-EOS target token;
2. replaces it with a dedicated target-mask token;
3. independently replaces every other body token with a second mask token at
   probability 0.5;
4. appends a dedicated answer-carrier token;
5. asks the tied output head at that carrier to predict the withheld token.

The three carrier rows are selected from tokens absent from protected natural
train/validation, all raw documents, and both QA files.  They do not add model
state.  Two independently seeded probes are made per sampled document.  The
record is added at the final title-token position before the ordinary blocks.

Held-out read evaluation uses eight new frozen probes per held-out document.
No evaluation probe seed appears during specialization.

## Specialization arms and exact compute ledger

All model-specialization optimizers use BF16 forward, Muon/AdamW peak rates
0.001/0.0001, 100-step warmup, cosine-to-10%, clip 1.0, and no retry.

1. **learned-writer:** 1,000 updates.  Each update processes 16 full writer
   sequences and 32 read sequences (two probes for each writer sequence).
   Read-token cross entropy backpropagates through both reader and writer.
2. **fixed-random-writer:** 1,000 updates over the identical read probes with a
   deterministic 220-cell Q16 code derived only from document ID.  It tests
   whether arbitrary identity keys plus a shared reader suffice.
3. **zero-record-read:** 1,000 updates over the identical read probes with no
   record.  It prices the corrupted-view objective itself.
4. **dense-causal-3x:** 3,000 ordinary causal raw-document updates, batch 16,
   sampling all documents.

The primary candidate and dense control each process exactly
`6,144,000 = 1,000*(16+32)*128 = 3,000*16*128`
specialization sequence-token positions.  The dense arm receives next-token
supervision at every active position, whereas the candidate receives only the
two withheld-token targets per document presentation.  Writer quantization and
the final one-pass compile are disclosed separately.

## QA reader and evaluation

Freeze and hash every record before QA training.  Each arm receives the same
3,000 batch-32 updates on the 146 QA-training rows, with T21a's binary tied
` yes`/` no` head, peak Muon/AdamW rates 0.001/0.0001, 300-step warmup,
cosine-to-10%, clip 1.0, and no retry.

For learned and fixed-random arms, exact case-insensitive raw-title matching
selects the two records and adds them at the last token of each title.  Codes
remain frozen.  Development is evaluated once, followed without refit by zero
and fixed-permutation ablations of the learned records.

## Mandatory gates

All gates are required:

1. Frozen input/preregistration/tokenizer hashes match; writer/read
   specialization sees only permitted raw fields and never QA/support data.
2. Every arm starts from the identical common-base state, finishes every
   update with finite loss/gradient, and has zero retry.
3. Learned records contain only the fixed 16 levels, are finite, preserve exact
   decoded indices through BF16, and remain hash-identical during QA training.
4. Every development support document is writer-held-out and receives exactly
   one gradient-free frozen-writer compile pass.
5. On writer-held-out probes, correct learned records reduce NLL by at least
   20% relative to both zero and shuffled records.
6. Correct learned records reduce held-out probe NLL by at least 10% relative
   to the correctly paired fixed-random-record arm.
7. Learned-record development QA is at least 75% and at least 10 absolute
   points above the best of dense-causal-3x, zero-record-read, and
   fixed-random-writer.
8. Zeroing and shuffling learned records each reduce development QA by at least
   10 absolute points.
9. Learned-record protected natural NLL after QA is no more than 0.5% worse
   than dense-causal-3x.
10. All arms retain identical served state schema, parameter count, precision,
    attention/FFN graph, and KV shape; prospective writes equal 743,670 and do
    not exceed 743,734.

## Decision rule

- **Fail acquisition:** if gates 3-6 fail, close this same-model write map; do
  not tune slice, scale, quantization, masks, steps, or thresholds.
- **Fail transfer:** if held-out probe causality passes but QA gates fail, close
  the raw-mask functional interface as a semantic-query wall.
- **Pass:** admit exactly one physical same-graph export, then a newly frozen
  evaluation and unseen-seed replication.  T22a alone is not a production-model
  result because title lookup and virtual record addition remain privileged.

