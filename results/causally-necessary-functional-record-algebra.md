# Causally necessary functional records — algebra and next-gate boundary

Status: **design boundary; not yet a candidate or positive result**  
Date: 2026-07-31

## Goal-preserving question

Can from-zero training turn raw prose into a small digital object that improves
held-out natural-language knowledge and reasoning, then compile that object into
weights already present in an ordinary Transformer so that checkpoint bytes,
active parameters, attention/FFN graph, KV state, and serving FLOPs do not grow?

T11 established that a hand-written digital relation plane can survive ordinary
training at the same serving cost.  T12--T20 failed to recover a useful natural
semantic payload after ordinary LM training.  T21a tests whether an optional
per-document code learned by causal next-token loss is enough.  The design here
does not rescue or tune T21a.  It states the different assumption that would be
tested only after T21a is closed.

## Why optional autoregressive codes are non-identifiable

Let a raw document be `D=(T,X)`, with title `T`, body tokens
`X=(x_1,...,x_n)`, and a bounded record `C_D`.  T21a minimizes

`L_AR = sum_t -log p_theta(x_t | T, x_<t, C_D)`.

The ordinary solution that ignores `C_D` is always in the hypothesis class.
At the population optimum, the maximum benefit available to the record is

`Delta_AR = sum_t I(x_t; C_D | T, x_<t)`.

The prefix already exposes wording, syntax, and much of the local fact.  Thus a
record can be ignored, or can encode residual surface prediction, without ever
becoming a reusable semantic interface.  Correct reconstruction would still not
identify how an unseen natural query should read the record: jointly applying
an invertible change of record coordinates and its inverse in the decoder leaves
the reconstruction loss unchanged.

This is a wall in the objective's identifiability, not evidence that bounded
digital records or same-graph compilation are impossible.

## Replace a stored vector with a trained read function

For each document, construct raw-only probe pairs

`(Q_m(D), Y_m(D))`, for masks/views `m ~ M`,

where `Y_m(D)` is removed from `Q_m(D)`.  Examples include a masked content
span, a distant-span target, and a position carrier with no body-token prefix.
No answer, QA row, support annotation, parser, teacher, or external model is
used.  A shared ordinary Transformer plus the record defines the function

`F_D(q) = Reader_theta(q, C_D)`.

Training minimizes

`L_read = E_D E_m[-log F_D(Q_m(D))[Y_m(D)]]`.

The artifact being trained is therefore the behavior of one record across many
independent reads, not the value of a latent vector under one decoder prompt.
Coordinate gauge freedom may remain, but the read function is behaviorally
identified by its probe distribution.

For a probe family with genuine answer collisions,

`Delta_read = I(Y; C_D | Q)`

is measured directly by correct-record versus zero-record and shuffled-record
loss.  If `H(Y|Q)>0`, the no-record reader has a nonzero information-theoretic
floor.  The record is no longer merely available: it is causally necessary for
the part of the answer not determined by the query.

### A raw-only quotient that supplies such reads

Let `df(w)` be the number of documents containing word `w`.  Define the anchor
alphabet without a parser or label as

`A = {w : ceil(N/100) <= df(w) <= floor(N/2)}`

and allow a target value when

`1 <= df(w) <= ceil(N/20)`.

For a target occurrence, project its four-word left/right neighborhood onto
`A`: retain anchor words and relative offsets, erase the target, and map every
other rare word to the same blank symbol.  Two windows are equivalent when
this projection is identical.  The quotient class is a corpus-discovered
relation skeleton; its varying erased targets are the values.

The record is consequently trained as a bounded sparse function

`f_D : relation-skeleton -> value`,

not as an undifferentiated document embedding.  When the same skeleton occurs
in two documents, raw text also supplies the exact self-supervised composition

`equal(f_D(q), f_E(q))`.

The frozen Stage-0 construction yields 108,570 unique functional edges with
disjoint train/evaluation assignment,
3,392 single-valued ambiguous quotient classes, 2.568 bits of conditional target
entropy, and raw equality pairs without inspecting QA or supporting-fact fields.  These
counts establish a nontrivial training game, not that the model can learn or
export it.

## Frozen logical budget

Retain the existing prospective plane:

- 2,405 document records;
- 220 cells per record;
- 16 exactly representable levels per cell;
- `2,405 * 220 * 4 = 2,116,400` logical record bits;
- 880 logical bits per document;
- at most 743,734 overwritten entries in the unchanged BF16 checkpoint;
- no additional served token, buffer, retrieval index, module, parameter,
  activation state, KV entry, or operator.

At roughly 128 body tokens and about five nats of residual uncertainty per
token, 880 bits is near the order of a full document's residual code length.
That estimate is motivation, not a capacity claim; the experiment must report
the actual rate-distortion curve and cannot count logical bits as physical
checkpoint savings.

## A valid first experiment

The first gate should compare from identical initialization:

1. ordinary causal dense training on the raw prose;
2. a dense probe control trained on the exact same raw-derived reads but with no
   record;
3. the quantized functional-record candidate trained on those reads;
4. a stronger dense control receiving at least twice the raw-document exposure.

Natural-token presentations, raw-token presentations, optimizer work, elapsed
time, and training FLOPs are reported separately.  Extra compiler training is
allowed as a disclosed training-only cost; it cannot be called free.

After raw-only acquisition, freeze and hash every record.  Train the same small
natural-question reader in all arms.  Score title-disjoint held-out questions
once, then score the fitted candidate without refit using zero and permuted
records.

The virtual gate is worth running only with all of these frozen requirements:

- correct records reduce held-out raw-probe NLL at least 20% versus both zero
  and shuffled records;
- candidate held-out QA is at least 75% and at least 10 absolute points above
  the best dense control;
- zeroing and shuffling records each reduce QA by at least 10 absolute points;
- protected natural NLL is no more than 0.5% worse than the matched ordinary
  model;
- every served tensor shape, parameter count, precision, operator, and KV shape
  is identical;
- no development answer or support field enters record acquisition or routing.

A virtual pass admits only physical export into the existing plane.  It is not
the project result.  Physical export must reproduce the gain with ordinary
weights and operators, no runtime record addition or exact external lookup,
then survive untouched holdout and unseen-seed replication.

## Failure localization

- **Correct ~= zero ~= shuffled on probes:** the record transport or bounded
  capacity is unusable under forced reads.  Close this record family.
- **Probe gain, no QA gain:** the raw-derived query family does not identify a
  reusable natural-language function.  Close masked-view functional records;
  do not tune code width or exposure.
- **Virtual QA gain, physical-export loss:** same-graph compilation is the wall.
- **Physical gain with natural regression:** capacity was reallocated rather
  than made denser; the fixed-cost claim fails.
- **All gates pass:** replicate and scale before using the phrase smarter model.

## Prior-art boundary checked on 2026-07-31

- *Memory Tokens* optimizes one continuous embedding per fixed sequence and
  demonstrates reconstruction, but reports that use beyond reconstruction is
  unresolved and retains an extra inference token.
- *Latent Context Compilation* uses a disposable LoRA compiler and portable
  buffer tokens, with random-query regularization, but the compiled memory
  remains an extra inference artifact.
- *Large Language Model as Token Compressor and Decompressor* learns discrete
  variable-length Z-tokens with pretrained-model adapters, but also serves the
  compressed token sequence.
- *Knowledge Capsules* injects external attention-compatible KV memory.

Those results support feasibility of compression; none establishes the claim
here: from-zero raw-prose acquisition of a query-conditioned record, physically
embedded into already-paid ordinary-model weights, with a substantial held-out
reasoning gain at identical serving cost.

Primary sources:

- https://arxiv.org/abs/2506.15001
- https://arxiv.org/abs/2602.21221
- https://arxiv.org/abs/2603.25340
- https://arxiv.org/abs/2604.20487
