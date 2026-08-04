# Joint quantized document code T21a — preregistration

Status: **frozen before implementation result**  
Date: 2026-07-31

## Question

T19c showed that a mean-pooled hidden state is not an identifiable document
schema.  T20a showed that even a 220-cell HOSVD retaining 53.78% of raw
learning-operator energy carries no causal held-out relation signal.  Both were
extracted after ordinary LM training.

T21a asks whether identifiability appears when the document record is itself a
from-zero optimization variable whose only job is to reduce raw-prose
prediction loss:

> Can a 220-cell, 16-level record learned only from each document's prose give
> a full Transformer substantially better held-out knowledge/reasoning than
> dense training, including a control with twice the document exposure?

This is a representation-and-reader upper bound.  Title-to-record selection is
an exact training-time string lookup and record injection is virtual.  Passing
would admit, but not replace, an ordinary-graph physical export experiment.

## Frozen data and model

- Same 2,405 raw documents and title-disjoint 146-train/104-development QA
  split as T12-T20.
- Same protected natural train/validation streams.
- Same 36,577,152-parameter, 384-wide, ten-layer, six-head, SwiGLU causal
  Transformer and SmolLM2 tokenizer revision.
- Model seed 8,209 in every arm.
- Raw document batches contain `title + newline + text`, at most 128 causal
  next-token transitions.  Padding is excluded from loss.
- No support annotation, QA row, answer, parser, external model, pretrained
  teacher, or evaluator field enters pretraining or the document-code update.

## Digital record

The candidate owns one training-only real parameter vector `u_d in R^220` per
raw document, initialized from seed 21,001 with normal standard deviation 0.05.
At every forward pass it is transformed coordinatewise by

`c_d = 4 * Q16(tanh(u_d))`,

where `Q16` rounds to the nearest of the 16 uniformly spaced odd levels
`{-15,-13,...,15}/15`; a straight-through estimator supplies gradients.

`c_d` is added to hidden coordinates 64 through 283 at the first token of that
document before the ordinary ten Transformer blocks.  The code and model are
jointly optimized by next-token cross entropy.  The code table is updated only
on raw-document steps and is frozen before any QA-format training.

The table contains exactly `2,405 * 220 = 529,100` four-bit logical cells.  The
already frozen title-code/address layout would make the compiler-specific
physical ledger 743,670 existing entries, 64 below the 743,734 cap.  T21a
verifies this ledger and BF16 level stability but does not claim its virtual
injection is already an exported in-graph reader.

## Pretraining arms

All models start byte-identically and use T12's Muon/AdamW split, peak rates
0.005/0.0003, 500-step warmup, cosine-to-10% schedule, BF16 forward, gradient
clip 1.0, batch 16, context 128, and no retry.

1. **dense-1x:** 20,000 updates: 19,000 protected-natural and 1,000 raw-document
   updates, with a document update every twentieth step.
2. **joint-code-1x:** exactly the dense-1x schedule plus the quantized record;
   `u_d` uses Adam, learning rate 0.03, zero weight decay.
3. **dense-2x-doc:** 21,000 updates: the same 19,000 natural updates and 2,000
   raw-document updates.  The first 20,000 steps use a document update every
   tenth step; the final 1,000 are natural.  This control gets twice the raw
   document presentations and more total training compute.

Document batches and natural batches use frozen arm-independent seeds.  The
candidate's record optimizer work is reported separately rather than hidden in
the serving ledger.

After pretraining, report full-corpus document NLL for the candidate with the
correct, all-zero, and seed-21,005 shuffled records, plus dense-arm document
NLL and protected natural NLL.

## QA-format training and evaluation

The document table is frozen and hashed before QA training.  Each arm then
receives 3,000 identical batch-32 QA updates on the 146 training rows.  The
question is the only model input.  Binary logits are the tied rows for the
single tokens ` yes` and ` no`.  All ordinary model parameters train with peak
Muon/AdamW rates 0.001/0.0001, 300-step warmup, cosine-to-10%, BF16 forward,
clip 1.0, and no retry.

For the candidate only, the two longest nonoverlapping raw titles are found by
exact case-insensitive string match.  Their frozen records are added at the
last token position of each matched title.  No development support-title or
supporting-sentence field is read.  Development is scored once after training.

The fitted candidate is also scored without refit using:

- both records replaced by zero;
- one fixed seed-21,005 permutation of all development record selections.

The document-code hash must be unchanged across QA training.

## Gates

All gates are required:

1. Frozen data/tokenizer/preregistration hashes match; pretraining document
   code sees only raw documents and its same-model loss.
2. All three arms start from identical model hashes and finish every update
   with finite loss/gradient and zero retry.
3. Candidate codes use only the frozen 16 logical levels, remain finite, and
   retain identical decoded level indices after BF16 roundtrip.
4. Candidate code hash is unchanged by QA-format training.
5. Exactly two raw titles are found in every QA row without support fields;
   both classes occur in train and development.
6. Candidate and controls retain identical served state schemas, parameter
   counts, precision, attention/FFN graph, and KV shape.  The prospective
   compiler ledger is exactly 743,670 and at most 743,734.
7. Correct candidate records reduce full-corpus document NLL by at least 10%
   relative to both zero and shuffled records.
8. Candidate development QA accuracy is at least 75% and at least 10 absolute
   points above the better of dense-1x and dense-2x-doc.
9. Zeroing or shuffling candidate records reduces development accuracy by at
   least 10 absolute points each.
10. Candidate protected natural-validation NLL after QA training is no more
    than 0.5% worse than dense-1x.

## Decision rule

- **Fail:** close this frozen quantized auto-decoding record.  Do not tune code
  width, level count, amplitude, injection coordinate, optimizer, exposure,
  QA schedule, or thresholds.  The ablations determine whether the failure is
  record acquisition, reader use, or natural-quality interference.
- **Pass:** admit one physical same-graph export using only existing title and
  FFN weights, followed by an untouched final set and replication.  T21a alone
  cannot satisfy the production goal because exact string routing and virtual
  addition are privileged.

No post-result rescue is allowed.
