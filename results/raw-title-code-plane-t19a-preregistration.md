# Raw-title code plane T19a — from-zero routing preregistration

Status: **frozen before implementation result**  
Date: 2026-07-31

## Question

Can a training-only raw-title compiler install a discrete compositional address
inside an otherwise ordinary Transformer, such that from-zero co-training
routes unseen natural question surfaces exactly while matched gradient training
with the same targets fails even at twice the steps?

This is the first corrected T19 gate.  It does not test document payloads or
question answering.

## Frozen inputs

- Candidate corpus:
  `data/hotpot-real-prose-t12/candidate-raw-prose.jsonl`, SHA-256
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`.
- The compiler accepts only `document_id`, `title`, and `text`; it uses only
  `document_id` and `title` in this gate.
- Tokenizer: `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Protected natural train and validation streams are the unchanged T10 files.
- T12's 104 questions / 208 title surfaces are development evaluation only.
  Neither compiler nor routing training may read them.

## Frozen address algebra

For tokenizer row `t`, the first 32 bits of repeated
`SHA256("t19a-token-{t}-{counter}")` form a deterministic bipolar vector
`r_t in {-1,+1}^32`.  Only rows occurring in canonical strings
`" " + title` are writable.  Candidate embedding cells `E[t, 0:32]` are set
to `0.02 * r_t` and restored after every optimizer step.

The target address for title `i` is

`c_i = normalize(mean(r_t for t in tokenize(" " + title_i)))`.

No fitted projection, semantic encoder, question, answer, supporting-page
label, or external model enters the compiler.

The write count is exactly `32 * number_of_active_token_rows` and must not
exceed T11's 743,734-entry isolation budget.  Candidate export must have the
same class, state-dict keys, tensor shapes, parameter count, and forward graph
as the controls.

## Training arms

All arms start from model seed 9,019 and receive identical natural batches and
identical raw-only routing examples over their shared prefix.

1. `compiled_1x`: deterministic code-cell compilation, then 3,000 mixed steps.
2. `gradient_1x`: no compilation, 3,000 mixed steps.
3. `gradient_2x`: no compilation, 6,000 mixed steps.

Every step has one 16x128 protected next-token batch and one 64-example routing
batch.  Routing training uses only three frozen wrappers ending in a raw title:

- the bare title;
- `Article about {title}`;
- `Document title: {title}`.

The loss is natural next-token cross entropy plus mean cosine distance between
the final hidden state's first 32 coordinates and `c_i`.  Muon/AdamW grouping,
peak rates 0.005/0.0003, 500-step warmup, BF16 autocast, and gradient clipping
at 1.0 match T12.  Each arm has its own cosine schedule over its frozen step
count.  Failed or retried steps are forbidden.

## Frozen evaluation

Top-1 retrieval compares the normalized final hidden first-32 vector against
all 2,405 frozen title codes.  Evaluate:

- canonical bare titles;
- two raw-only held wrappers: `Information concerning {title}` and
  `Read the entry for {title}`;
- the 208 title-ending prefixes cut from T12's development natural questions.

For `compiled_1x`, deterministically permute the active token rows' 32 code
cells after training, reevaluate all three surfaces, and restore the exact
state.  This is the causal ablation.

## Admission gates

T19a passes only if all conditions hold:

1. Input hashes and raw-only compiler interface are exact.
2. The write count is at most 743,734 entries and compiled cells remain
   bit-exact after training.
3. Candidate and controls have identical ordinary served model structure and
   parameter count.
4. `compiled_1x` reaches 100% canonical, 100% held-wrapper, and 100% natural
   question-surface top-1 retrieval, each with positive BF16 score margin.
5. On every surface family, `compiled_1x` exceeds `gradient_2x` by at least
   10 absolute percentage points.
6. Shuffling compiled cells reduces natural-surface accuracy by at least 30
   absolute points and below 70%.
7. Candidate protected natural NLL is no more than 0.5% worse than
   `gradient_1x` after 3,000 steps.
8. Every loss and gradient is finite, with zero failed/retried steps.

Failure closes deterministic token-superposition initialization as the T19
address solution.  It may not be rescued with more dimensions, more wrappers,
question-derived training, a learned projection, more steps, or relaxed
exactness.  Passing admits a separately frozen raw-prose payload experiment; it
is not a smarter-model claim.
