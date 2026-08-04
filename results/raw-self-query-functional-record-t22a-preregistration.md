# Raw self-query functional record T22a — preregistration

Status: **frozen before any T22a capability result**  
Date: 2026-07-31

## Question

Can from-zero training use raw prose to learn a bounded, document-conditioned
read function that substantially improves title-disjoint natural knowledge and
comparison reasoning over stronger dense training, while fitting the already
frozen same-checkpoint digital-plane ledger?

T22a is not a wider/longer rescue of T21a's optional causal-next-token code.
It changes the identifiable object.  The raw corpus mechanically supplies
queries for which the missing value is not present in the query, plus exact
two-record equality composition.  The record is trained as

`document x relation-skeleton -> value`,

not as an unconstrained latent document vector.

T22a is a virtual representation-and-reader gate.  Passing admits a physical
ordinary-weight export; it does not itself establish the project or production
claim.

## Frozen inputs

- The same 2,405 raw documents and title-disjoint 146-train/104-development QA
  split as T12--T21.
- Candidate corpus SHA-256:
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`.
- Stage-0 result SHA-256:
  `f46a36b5c617bb62e18e9fc81111dc3df2a8e8d351d5808e6173deaa79c1d660`.
- Same protected natural train/validation streams and frozen hashes.
- Same SmolLM2 tokenizer revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Same 36,577,152-parameter, 384-wide, ten-layer, six-head, SwiGLU causal
  Transformer and model seed 8,209 in every arm.
- The compiler-visible document fields are exactly `document_id`, `title`, and
  `text`.  No QA row, answer, supporting title/sentence, parser, teacher,
  pretrained model, external model, or evaluator field enters functional-edge
  construction or record optimization.

QA strings may only be checked after raw-only control-token allocation to prove
that the selected unused token rows do not occur naturally.  They cannot change
which rows are selected.

## Raw-only quotient and functional edges

Lowercase alphanumeric words are extracted mechanically.  With `N=2,405`, the
document-frequency thresholds are scale-derived and frozen:

- anchor words: `ceil(N/100) <= df(w) <= floor(N/2)`, hence 25 through 1,202;
- target words: `1 <= df(w) <= ceil(N/20)`, hence 1 through 121.

For each target occurrence, inspect four words on either side.  Retain only
anchor words and their signed offsets; erase the target and every non-anchor.
This projected tuple is a relation skeleton.  Duplicate
`(document, skeleton, target)` occurrences are collapsed before splitting.
Mappings are assigned to an 80/20 SHA-256 split, repaired per document so every
document has disjoint train and evaluation mappings.  A position-only fallback
is allowed only when a document has fewer than two nonempty mappings and is
excluded from structural capability gates.

Within each split, the GPU corpus retains one representative for every
single-valued structural `(document, skeleton, target)` edge.  If a document
has no structural collision edge in that split, retain its deterministic
lowest-hash raw edge solely to keep that record addressable; such coverage rows
are excluded from unary structural evaluation and equality construction.

Any `(document, skeleton)` with more than one target is excluded.  A structural
class must have at least two documents and two different targets.  Frozen Stage
0 contains:

- 108,570 unique functional edges;
- 3,392 single-valued ambiguous skeleton classes covering 2,248 documents;
- 2.5677 bits of conditional target entropy;
- 225 training equality classes with 562 positive cross-document pairs;
- 17 evaluation equality classes with 36 positive pairs.

These counts admit the training game only.  They are not model evidence.

## Model-visible raw probes

Five vocabulary rows absent from protected natural text and all raw documents
are selected deterministically as read, blank, answer, compare, and second-read
control tokens.  Vocabulary size and embedding/output tensors do not change.

For a unary read, the input is:

`READ, eight projected offset slots, ANSWER, target-prefix`.

Absent/erased offset slots use the one blank token.  The record is added at
`READ`.  Cross-entropy applies only to the complete tokenization of the erased
target word; no surrounding body token or target token is visible before it is
predicted.

For equality, select two different documents in one single-valued skeleton
class.  The input contains the same projected skeleton once, with the two
records added at `COMPARE` and `SECOND-READ`.  The exact raw target strings
supply `yes` iff equal and `no` otherwise.  Positive and negative batches are
balanced.  Negative pairs may come only from skeleton classes that also contain
a positive pair, preventing skeleton identity from revealing the class.

Within each split, one representative raw occurrence realizes each functional
edge.  Training samples documents uniformly, then one of that document's
available functional reads uniformly, so verbose documents do not receive more
record capacity.

## Digital record

The candidate owns `u_d in R^220` per document, initialized at standard
deviation 0.05 from seed 22,001.  At every use,

`c_d = 4 * Q16(tanh(u_d))`,

where `Q16` rounds to 16 uniform levels and uses a straight-through gradient.
The record is added to hidden coordinates 64 through 283 at the selected input
position before the ordinary ten blocks.

The table is exactly `2,405 * 220 = 529,100` logical cells and 2,116,400 logical
bits.  A prospective physical export is exactly 743,670 overwritten existing
BF16 entries, below the 743,734 frozen cap.  Logical four-bit capacity is not
claimed as physical checkpoint compression.

The record optimizer is Adam, learning rate 0.03, zero weight decay.  It steps
only on unary/equality updates.  The table is frozen and hashed before QA-format
training.

## Pretraining arms

All arms start byte-identically.  Model optimization is T12's Muon/AdamW split,
peak rates 0.005/0.0003, 500-step warmup, cosine-to-10%, BF16 forward, gradient
clip 1.0, batch 16, and no retry.  Natural batches retain context 128; functional
reads use context 64.

1. **dense-probe-1x:** 29,000 updates in 1,000 identical 29-step cycles:
   19 natural, 8 unary, and 2 equality updates.  It receives every transformed
   training target and label but no record.
2. **functional-record-1x:** exactly the same model initialization, batches,
   targets, schedule, and model optimizer as dense-probe-1x, plus the bounded
   record and its disclosed optimizer work.
3. **dense-probe-2x:** 39,000 updates in 1,000 identical 39-step cycles:
   19 natural, 16 unary, and 4 equality updates.  It receives the same natural
   work and twice the functional exposure/model-update compute.

Thus each 1x code receives about 66 uniformly sampled unary/equality
presentations per document in expectation.  This exposure was fixed before a
capability result so a 16-level code can traverse its range; it is not a
post-result T21 adjustment.  Every training token, update, elapsed second,
optimizer byte, and peak HBM byte is reported.  Extra training-only compiler
work is a disclosed cost, not free compute.

## Raw-read evaluation

After pretraining, evaluate only disjoint evaluation functional edges and
equality pairs.  The candidate is scored without refit using correct, all-zero,
and seed-22,005 permuted records.  Dense arms receive only correct/no-record
evaluation.

The equality set is exactly balanced by retaining every positive pair up to the
smaller class count and the same number of deterministically ordered negatives.

## Natural QA training and evaluation

All three arms then receive T21a's same 3,000 batch-32 QA-format updates on the
146 training rows, Muon/AdamW peaks 0.001/0.0001, 300-step warmup, cosine-to-10%,
BF16 forward, clip 1.0, and no retry.  The record table remains frozen.

Questions are the only model input.  Exact case-insensitive occurrence of the
two longest nonoverlapping raw titles selects the two records, which are added
at the last token of each matched title.  Binary output uses the existing
single-token rows ` yes` and ` no`.  No development support field is read.

Development is scored once.  The fitted candidate is then scored without refit
with both records zeroed and with record selections permuted by seed 22,005.

## Gates

Every gate is required:

1. All frozen hashes, raw-only signatures, Stage-0 gates, control-token
   nonoccurrence, and exactly three compiler fields pass.
2. All arms start from the same model hash, complete every frozen update, and
   have finite loss/gradient with zero retry.
3. Candidate records lie exactly on the 16 levels, preserve decoded indices in
   BF16, remain finite, and retain an identical hash across QA training.
4. Correct records reduce structural unary evaluation NLL by at least 20%
   relative to both zero and shuffled records.
5. Candidate raw equality accuracy is at least 80%, at least 20 absolute points
   above the better dense arm, and falls at least 20 points under both zero and
   shuffled records.
6. Candidate development QA accuracy is at least 75%, at least 10 absolute
   points above the better dense arm, and falls at least 10 points under both
   zero and shuffled records.
7. Candidate protected natural NLL after QA training is no more than 0.5% worse
   than dense-probe-1x.
8. Served parameter count, state schema, precision, attention/FFN graph, KV
   shape, and prospective physical ledger are identical; the ledger is exactly
   743,670 entries and no larger than 743,734.

## Decision rule

- **Fail acquisition:** if correct records do not causally improve unary reads,
  close this quotient-record transport.  Do not tune width, levels, amplitude,
  exposure, optimizer, or threshold.
- **Unary pass, equality fail:** close the shared functional-composition claim.
- **Raw functions pass, natural QA fail:** the raw quotient is not a sufficient
  natural-language semantic interface.  Close it rather than adding a parser or
  benchmark-derived probes.
- **Virtual pass:** admit one physical same-graph export into existing title and
  FFN weights, then untouched evaluation and unseen-seed replication.  Do not
  call T22a alone a smarter production model.

No post-result rescue is allowed under T22a.
