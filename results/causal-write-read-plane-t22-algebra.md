# Causal write/read plane T22 — algebra and admission boundary

Status: **DERIVATION ONLY; not preregistered and not yet a candidate**  
Date: 2026-07-31

## Required outcome

A from-zero model must read raw prose, emit a finite digital record, and use
that record to improve title-disjoint held-out natural QA substantially over
the strongest dense control.  The record must ultimately be written into the
already budgeted ordinary-model entries, so the deployed parameter count,
precision, graph, KV shape, and per-token FLOPs remain unchanged.

Reconstruction, compression, or a causal record ablation by itself is not the
outcome.  Those are only mechanism gates.

## Why T21a's objective can ignore its record

For document tokens `X=(X_1,...,X_n)` and record `C`, T21a minimizes

`L_AR = sum_t -log p_theta(X_t | X_<t, C)`.

Relative to a no-record model, the population value available from the record
at position `t` is

`I(X_t ; C | X_<t)`.

The teacher-forced prefix already carries syntax, local wording, repeated
entities, and much of the document identity.  Consequently a valid stationary
solution is to set every record-dependent Jacobian to zero and let the ordinary
model absorb the document distribution.  Even when a record lowers `L_AR`, the
objective identifies only a next-token residual.  It does not identify a
function that can answer a new query.

This is not repaired by increasing code width, levels, amplitude, exposure, or
changing one injection coordinate.  Those changes can increase storage while
leaving the missing read contract unchanged.

## Changed algebra: train a function over withheld reads

Let `g_theta` be the same causal Transformer used by the served reader.  A
writer pass consumes the complete raw document and takes a fixed slice of the
last contextual state; no teacher, parser, answer, or support annotation is
available:

`C_theta(X) = 4 Q16(tanh(s * g_theta(X)_last[64:284]))`.

There is no learned external encoder or projection in this definition.  The
same model writes and reads.  `Q16` has 16 fixed levels and a straight-through
training derivative.  After training, one 220-cell four-bit record is sealed
per raw document and the writer pass disappears from serving.

For a raw-only random mask `M`, construct a read view `Q_M(X)` that contains
the document title and a corrupted natural span but never the target span
`Y_M`.  The read loss is

`L_read = E_M[-log p_theta(Y_M | Q_M(X), C_theta(X))]`.

The available record signal is now

`I(Y_M ; C_theta(X) | Q_M(X))`.

Masks are admissible only when a frozen no-record diagnostic shows material
conditional uncertainty; otherwise the prefix has merely been replaced by
another shortcut.  Multiple independently sampled masks make each stored
record define a family of query-to-answer behaviors rather than one
reconstruction trajectory.

The record has the hard information bound

`H(C) <= 220 log2(16) = 880 bits/document`.

On the frozen 2,405-document corpus, truncation leaves 241,541 content tokens,
or 100.433 per document.  The full logical table therefore supplies 2,116,400
bits, 8.762 bits/token, or 6.073 nats/token of maximum discrete capacity.  This
is not a proof of usable coding, but it prevents a failed one-pass writer from
being misreported as a dimension-count impossibility.

For an optimal reader, the zero-versus-correct read-loss difference is exactly
`I(Y_M; C | Q_M)` in nats.  The measured finite-model difference is therefore
an operational test of whether the channel carries target information, while
the shuffled ablation rejects a corpus-wide bias.

There is also a separating-mask identifiability condition.  If, for every pair
of documents that differ in a stored fact, the frozen mask family contains a
view with the same visible query but different target, then zero read error
requires either different records or an impossible deterministic collision.
This does not prove natural-question semantics, which is why held-out QA and
its record ablations remain mandatory.

Natural next-token updates remain in the schedule.  The record path therefore
trades training work for served knowledge density; it does not call that work
free.

## Why this answers the compiler objection

The writer is not a stronger deployed model and not a weaker heuristic
extractor.  It is the same from-zero model, given one offline full-document
pass and trained through the downstream read loss.  The only persistent output
is the finite record.  A separate optimizer, pretrained teacher, semantic
parser, generated QA corpus, retrieval store, or serving-time memory is not
part of the mechanism.

The writer can still fail.  Quantization can collapse, the common state slice
can remain surface-coded, or the reader can fail to generalize from corrupted
prose to natural questions.  Each failure is separately observable.

## Mandatory controls

1. **Writer-held-out documents:** use the already frozen title-disjoint
   benchmark partition before any model update.  The benchmark harness may
   identify the held-out titles, but no support field, question, or answer is
   exposed to the writer.  Learn the write/read algebra only on the remaining
   raw documents.  After the writer is frozen, each held-out raw document gets
   exactly one full-document forward write and quantization; neither the model
   nor its record receives a gradient on that document.
2. **Dense causal control:** ordinary next-token training with the same natural
   and raw documents.
3. **Dense read control:** identical corrupted views and target losses, but the
   record is always zero.  This prices the self-supervised objective itself.
4. **Training-compute control:** additional ordinary/raw updates priced to at
   least the writer plus reader forward/backward work.
5. **Correct/zero/shuffle record:** evaluated without refitting on raw reads and
   held-out QA.
6. **Random fixed writer:** a deterministic random code with the same logical
   capacity, including unseen document IDs, to reject mere identity keys plus
   reader memorization.
7. **Title-disjoint evaluation:** unchanged sealed T12 evaluation titles and
   no support fields in writing or routing.

The dense compute control may receive gradients on the writer-evaluation prose;
the candidate may only read that prose during its frozen one-pass write.  This
makes the baseline stronger and tests whether a learned compiler can ingest new
knowledge more effectively than ordinary weight updates at the same served
capacity.

## Admission gates

All are required before physical export:

- every record uses exactly 220 four-bit cells, is finite, and survives BF16
  decode exactly;
- on writer-held-out documents, correct records reduce sealed raw-read NLL by
  at least 20% relative to both zero and shuffled records;
- the learned same-model writer beats the random fixed writer on unseen-mask
  reads from writer-held-out documents, not only on masks or documents sampled
  during updates;
- held-out natural QA is at least 75% and at least 10 absolute points above the
  best dense/control arm;
- zeroing and shuffling records each reduce held-out QA by at least 10 points;
- protected natural NLL is no more than 0.5% worse than the dense causal
  control;
- prospective physical writes remain at most 743,734 existing entries;
- no serving resource increases.

If raw-read causality passes but QA causality fails, the branch has reached the
semantic-query wall and is closed.  If the writer loses to optimized or random
codes, the learned write map is closed.  If the virtual mechanism passes, only
then is an ordinary-graph physical export justified.

## Prior-art boundary

Optimized memory embeddings and latent context compression already show that
text can be reconstructed from compact continuous or discrete tokens.  Those
results establish feasibility of compression, not this claim: their memory
artifacts remain serving inputs or KV-bearing tokens, and reconstruction need
not produce title-disjoint QA capability.  T22 is admissible only if the
record is autonomously written from zero, causally useful for new natural
queries, and physically absorbed into an otherwise identical served model.

GradMem is the sharpest negative prior for the one-pass writer: it reports that
per-context gradient writes scale capacity better than forward-only writers.
T22 deliberately tests the harder amortized one-pass case because a
per-document optimized state would not demonstrate an autonomous learned
compiler.  Failure therefore closes this one-pass write map, not all
loss-driven memory writes.

Primary comparisons:

- Memory Tokens: <https://arxiv.org/abs/2506.15001>
- Latent Context Compilation: <https://arxiv.org/abs/2602.21221>
- Large Language Model as Token Compressor and Decompressor:
  <https://arxiv.org/abs/2603.25340>
- GradMem: <https://arxiv.org/abs/2603.13875>
