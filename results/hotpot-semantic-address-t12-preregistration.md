# Hotpot semantic address T12 — from-scratch feasibility preregistration

Status: frozen before GPU execution  
Date: 2026-07-31

## Purpose

T11 showed that an exact digital plane can coexist with language modeling, but
its inputs were special one-token symbols.  T12 first tests the missing
interface on real prose: can a from-scratch model learn enough label-free
semantic geometry that ordinary multi-token entity names and natural property
phrases can address compiled channels?

No plane is compiled in this gate.  Failure stops the writer before another
synthetic or hand-routed result can be mistaken for progress.

## Frozen data boundary

The source is `hotpotqa/hotpot_qa` distractor validation parquet at Hub commit
`1908d6afbbead072334abe2965f91bd2709910ab`, 27,452,575 bytes, SHA-256
`c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`.

The passed T12 data manifest exposes 2,405 deduplicated title/prose documents
(1,180,913 characters) to the shared model and compiler.  It withholds every
question, answer, and supporting-fact annotation from this corpus.  A
document-disjoint router/control split contains 146 training questions and 104
sealed evaluation questions over 205 unseen supporting page titles.  The
candidate writer is not admitted in this stage and receives no QA record.

The protected FineWeb-Edu-dedup train and validation streams retain their T10
checksums.

## Frozen shared-base training

- model: the existing 36,577,152-parameter, 10-layer, 384-hidden,
  6-head/1,024-SwiGLU causal Transformer;
- initialization seed: 8,209;
- tokenizer: `HuggingFaceTB/SmolLM2-135M` at
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;
- steps: 20,000;
- batch: 16 sequences of 128 input tokens;
- token presentations: 40,960,000;
- every twentieth step draws only from the title-plus-raw-prose stream;
- all other steps draw from the protected natural stream;
- Muon matrix learning rate: 0.005;
- AdamW embedding/norm learning rate: 0.0003;
- 500-step linear warmup, then cosine decay to 10% of peak at step 20,000;
- BF16 forward, FP32 cross-entropy, gradient norm clipped to 1.0;
- natural checkpoints at steps 5,000 and 20,000.

Questions and labels may be opened only after the final checkpoint is saved and
hashed.  They cannot affect model weights, step count, optimizer, or schedule.

## Frozen compositional entity address

Each tokenizer row appearing in a candidate document title receives a
deterministic 32-bit bipolar code derived from SHA-256.  A multi-token title's
address is the normalized mean of its subword codes.  This is an algebraic
screen only; the codes are not yet written into the model.

The screen reports canonical title collisions and, for every sealed supporting
title, retrieves among every candidate title using the exact natural question
surface.  This construction replaces the rejected ill-conditioned arbitrary
phrase-code solve; no regression or fitted codebook is allowed.

## Frozen semantic diagnostic

This diagnostic is deliberately privileged and cannot be reported as a model
result.  It uses the sealed supporting-page identities only after training:

1. remove the two exact entity-title strings and a frozen auxiliary stop list
   from the natural question;
2. tokenize the remaining property phrase;
3. for each property subword, take its maximum cosine similarity to any input
   embedding row occurring in each supporting page's raw prose;
4. average within a page and take the minimum over the two pages;
5. choose one scalar threshold and direction on the 146 document-disjoint
   training labels;
6. report accuracy on the 104 sealed questions.

The same diagnostic is run on the untouched random initialization after the
trained checkpoint is sealed.  The existing privileged exact-lexical probe is
55.7692% on the sealed set.

## Admission gates

All must pass:

1. every data checksum and label-blind corpus gate passes;
2. training is finite with zero failed/retried updates;
3. final protected natural NLL is below both initial NLL and 5.5;
4. all canonical candidate titles have distinct 32-dimensional addresses;
5. sealed question surfaces retrieve all 205 supporting titles exactly, with a
   positive worst-case cosine margin;
6. trained semantic diagnostic accuracy is at least 70%;
7. trained semantic diagnostic exceeds the privileged lexical probe by at
   least 10 percentage points and exceeds the random-initialization diagnostic;
8. the checkpoint is saved before any label-bearing diagnostic and all source,
   data, preregistration, state, and checkpoint hashes are emitted.

Passing admits a separately preregistered real-prose channel compiler and
same-serving-cost controls.  It does not demonstrate autonomous extraction or
a smarter production model.

Failure closes this static-input-embedding semantic interface at the frozen
training scale.  No post-result pooling, stopword, threshold, code dimension,
or step-count change belongs to T12.
