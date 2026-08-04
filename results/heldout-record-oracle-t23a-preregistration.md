# Held-out forced-read record oracle T23a — preregistration

Status: **FROZEN BEFORE IMPLEMENTATION AND BEFORE ANY T23a CAPABILITY RESULT**  
Date: 2026-07-31

## Decision question

Did T22a fail because its one-pass amortized writer could not construct a
useful record, or because the existing 220-cell virtual record/read interface
cannot carry useful raw-prose information even when its write is optimized?

T23a is a wall-localization oracle, not a production candidate.  It removes
the autonomous writer and explicitly optimizes held-out document records.  A
pass can justify rethinking the writer.  The optimized write itself cannot
justify a smarter-production-model claim.

## Frozen inheritance

T23a inherits without change from the executed causal-write/read T22a:

- the 36,577,152-parameter, width-384, ten-layer, six-head causal Transformer;
- model seed 8,209, tokenizer revision, protected natural streams, 2,405 raw
  documents, and raw fields exactly `document_id`, `title`, and `text`;
- the title-disjoint partition of 2,198 reader-training documents and 207
  held-out documents;
- the three raw-only carrier tokens, target masking, 0.5 other-body masking,
  context 128, title-position record injection, tied output head, and record
  coordinates 64 through 283;
- the exact 1,656 T22a evaluation probes: eight seed-22,007 probes for each of
  the 207 held-out documents;
- the 16 uniform levels, amplitude four, 220 cells, BF16 round-trip rule, and
  prospective 743,670-entry physical ledger under the 743,734 cap.

The authoritative T22a result SHA-256 is
`be7ce8c4d31250910e806770606fdb25381283584e20bd529d90008767bdd10b`.
Its source and authoritative preregistration hashes are respectively
`c59457f73c4b7770afcbf6c82b5254ee0cda2cea233a8bdc7b859c41fa37f222`
and
`f3dd0a9634efd1aca50c6c267c9980f8a14487737e8b0e4a5b20e1df3a7ea3e9`.

## Common base and byte-matched meta-readers

Reproduce T22a's 19,000-update natural common base exactly.  Clone two models
byte-identically and allocate one trainable 220-scalar parameter row for every
raw document in each arm, initialized identically at standard deviation 0.05
from seed 23,001.

For exactly 1,000 reader updates per arm, sample only the 2,198 reader-training
documents with T22a's identical batch, seed, two-probe construction, optimizer,
learning-rate schedule, BF16 forward, and clip.  The arms differ only in the
record transform:

1. **continuous reader:** `c_d = 4 tanh(u_d)`;
2. **Q16 reader:** `c_d = 4 Q16(tanh(u_d))` with the straight-through
   derivative.

Cross-entropy updates that arm's reader and selected rows.  Both record
optimizers are Adam at learning rate 0.03, zero weight decay, and step only on
forced-read updates.  Initial model bytes, code parameters, document/probe
order, targets, optimizer settings, and issued reader work are otherwise
identical.

Every row belonging to the 207 held-out documents must remain byte-identical
during these 1,000 model updates.  No held-out raw sequence, probe, code, QA
row, question, answer, or support annotation enters a model gradient.

This phase meta-learns a common code-reading convention on other documents.
It does not establish that an unseen document can be encoded.

## Held-out raw-only compile

Freeze every model parameter.  Rebuild the exact T22a held-out evaluation
probe batch first and seal its `(document, target_position)` pairs.

For each held-out document, deterministically rank every remaining body
position by SHA-256 of

`t23a-compile | document_id | target_position`.

Select the first 16 positions, or every available position when fewer than 16
remain.  None may be an evaluation target position.  Construct two independently
masked raw views per selected position using seeds 23,007 and 23,009.  These
compile views use only title and raw body tokens.  They use no evaluation target
position, QA string, answer, supporting title, or generated text.

Create two held-out parameter tables from the same seed-23,003 initialization:

1. **continuous oracle:** `c_d = 4 tanh(u_d)`;
2. **Q16 oracle:** `c_d = 4 Q16(tanh(u_d))` with a straight-through gradient.

Optimize only these held-out tables for exactly 40 balanced rounds of the 207
documents, in ascending batches of at most 16, learning rate 0.03 and zero
weight decay.  Each document therefore receives exactly 40 optimizer
presentations and 80 compile views.  The two oracle modes receive the same
document order, positions, masks, targets, number of updates, and initial
parameters.  Each arm's frozen reader receives no gradient or optimizer state.

The compile cost, elapsed time, optimizer state, target-token presentations,
and peak HBM are reported.  This is deliberately a stronger iterative writer
than the claimed autonomous path; it is not free compute.

The deterministic nearest-Q16 projection of the final continuous records is
also frozen.  It is evaluated through the continuous reader without any STE
optimization or refit.  This separates finite-level approximation from the
ability to optimize through the quantizer.

## Matched one-pass writer

Freeze the Q16 reader.  Create a separate compiler with exactly the T22a
one-pass writer architecture and common-base initialization: the same ordinary
Transformer consumes one raw document and returns

`4 Q16(tanh(h_last[64:284]))`.

For exactly 1,000 updates, use the same 2,198 training-document samples, raw
writer inputs, probes, targets, model optimizer, schedule, and quantized read
loss as T22a, but backpropagate through the frozen T23a Q16 reader into only
the compiler.  The compiler and reader do not share mutable weights.  This is
a stronger training-only writer than the production constraint and is charged
as an extra full model; it disappears after compile.

After training, consume every held-out raw document exactly once with no
gradient and evaluate its emitted record through the same frozen Q16 reader on
the same T22a probes.  This matched arm changes only the record source, unlike
the historical T22a comparison.

## Frozen controls and evaluation

Evaluate the frozen reader on the exact 1,656 T22a held-out probes with:

1. correct continuous optimized records through the continuous reader;
2. seed-23,011 shuffled continuous records, continuous fixed-random records,
   and zeros through that same reader;
3. nearest-Q16 projection of the continuous optimum, its seed-23,011 shuffle,
   fixed-random Q16, and zeros through the continuous reader;
4. correct STE-optimized Q16 records through the Q16 reader;
5. seed-23,011 shuffled STE-Q16 records, fixed-random Q16 records derived from
   `document_id` and seed 23,013, and zeros through that same reader;
6. matched one-pass writer records, their seed-23,011 shuffle, fixed-random
   Q16 records, and zeros through the frozen Q16 reader.

The continuous fixed-random record maps successive SHA-256 16-bit words for
the exact UTF-8 byte string
`t23a-random|23013|{document_id}|{counter}` uniformly onto `[-4, 4]`.  The Q16
fixed-random record maps nibbles from the same digest stream onto the 16 frozen
levels.

No condition receives refitting on an evaluation position.  Zero, shuffle, and
random are evaluated through their matched continuous or Q16 reader; they are
causal controls, not separately trained arms.

For a reference gain

`g(reference, candidate) = (NLL_reference - NLL_candidate) / NLL_reference`.

## Gates and classification

All integrity gates are mandatory: frozen hashes, exact T22a split and
evaluation probes, zero held-out model-gradient presentations, byte-identical
held-out initialization rows through meta-reader training, disjoint compile
and evaluation targets, finite optimization, exact Q16 levels, and BF16-stable
level indices.

The capability gates are:

1. continuous correct-record NLL gain is at least 20% independently versus
   its zero, shuffled-continuous, and fixed-random-continuous controls;
2. nearest-Q16 projection retains at least 90% of the continuous record's NLL
   gain over zero through the same continuous reader;
3. STE-optimized Q16 correct-record NLL gain is at least 20% independently
   versus its zero, shuffled-Q16, and fixed-random-Q16 controls;
4. STE-Q16 retains at least 90% of its continuous arm's NLL gain over zero;
5. continuous, projected-Q16, and STE-Q16 token accuracy each exceed their
   matched controls by at least ten absolute points;
6. protected natural NLL for each meta-reader is no more than 0.5% worse than
   the frozen common base;
7. the matched one-pass writer's NLL gain is at least 20% independently versus
   zero, its own shuffle, and fixed-random Q16, and its token accuracy exceeds
   each by at least ten absolute points.

Classification is frozen:

- **continuous fails:** the present meta-reader/injection law cannot exploit
  even a strong document-local write on unseen positions.  Close this exact
  virtual read interface; do not call it a four-bit capacity theorem;
- **continuous passes, projected Q16 fails:** the local 16-level projection is
  the boundary;
- **projected Q16 passes, STE-Q16 fails:** optimization through the quantizer,
  not existence of a discrete record, is the boundary;
- **STE-Q16 passes, matched one-pass fails:** the one-pass amortized writer is
  the localized wall.  Advance to an autonomous error-correcting or distilled
  writer, not a wider record;
- **matched one-pass passes:** T22a failed from its shared co-adaptation path,
  not the frozen forward writer function class;
- **Q16 raw probes pass:** only then may the frozen records enter the existing
  natural-QA causal gate and a separately proved physical ordinary-graph
  export.  Probe success alone is compression, not a smarter model.

No post-result change to steps, seeds, split, masks, optimizer, learning rate,
amplitude, level count, code width, injection coordinates, or thresholds is
allowed inside T23a.

## Prior-art boundary

Per-context optimized memory is already established by GradMem, and ridge or
delta updates are instances of test-time regression.  T23a therefore makes no
novelty claim.  Its purpose is only to localize this project's negative result
without confusing writer amortization, quantization, and reader capacity.

Primary comparisons:

- GradMem: <https://arxiv.org/abs/2603.13875>
- Test-time regression: <https://arxiv.org/abs/2501.12352>
- Gated KalmaNet: <https://arxiv.org/abs/2511.21016>
