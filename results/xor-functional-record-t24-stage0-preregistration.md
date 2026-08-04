# XOR functional record T24 — Stage 0 preregistration

Status: **FROZEN AFTER EXPLORATORY DIMENSION AUDIT, BEFORE AUTHORITATIVE STAGE 0**  
Date: 2026-07-31

## Decision question

Can the 220 four-bit cells already budgeted per raw document represent the
real-corpus functional quotient as an exact query-to-value table, without a
learned latent coordinate convention, a per-document optimizer, or a stronger
semantic compiler?

T23 showed that an arbitrary injected vector remains unreadable even when its
continuous values are optimized.  T24 changes the algebra: a query identifies
three fixed cells, and their bitwise XOR is the value.  The writer solves a
finite linear system; the reader is specified exactly.

Stage 0 is only a storage/read-identity gate.  It cannot establish natural
question parsing, useful knowledge, physical Transformer integration, or a
smarter production model.

## Frozen raw quotient

Reuse the raw-only construction from
`raw_self_query_functional_record_stage0.py` without changing its tokenizer,
frequency thresholds, radius, target removal, deduplication, or corpus:

- corpus SHA-256:
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`;
- fields exactly `document_id`, `title`, and `text`;
- 2,405 documents;
- only non-fallback skeletons that occur in at least two documents and take at
  least two distinct single-valued targets;
- no parser, teacher, QA row, answer, supporting title, or latent label.

The previous sealed audit reported 3,392 such skeletons, 14,146
document-skeleton edges, 2,248 covered documents, and 2.568 bits of conditional
target entropy.  Stage 0 must reproduce those values before testing the new
record.

For each skeleton `s`, sort its observed target strings by UTF-8 lexical order
and assign the target rank

`y_s(t) in {0,...,|V_s|-1}`.

The exploratory audit found `max_s |V_s| = 65`; seven payload bits are
therefore necessary and sufficient for this frozen quotient.  This lexical
rank has no semantic meaning.  A later model would still have to learn or
compile the shared `(s, rank) -> target surface` decoder.

## Frozen retrieval algebra

Each document has 109 logical slots.  Canonically serialize a skeleton as
compact JSON over its ordered `(relative_offset, anchor_word)` pairs.  For the
fixed family ID `1`, SHA-256 domain

`xor-functional-t24-v1 | family | skeleton | counter`

supplies three distinct indices in `{0,...,108}`.  SHA-256 is used only by the
offline Stage-0 construction.  A physical model would require the relation
front end to emit the already compiled three indices; no claim is made that a
Transformer computes SHA-256.

For document `d`, let `A_d in GF(2)^(n_d x 109)` contain ones at the three
indices of each registered skeleton.  Let `Y_d in GF(2)^(n_d x 7)` contain the
seven-bit target ranks.  The writer solves

`A_d Z_d = Y_d  (over GF(2))`.

Free variables are fixed to zero.  The first 109 physical four-bit cells store
bits 0--3 of each row of `Z_d`; the next 109 cells store bits 4--6 with the
fourth bit zero.  The final two cells are reserved zeros.  A read XORs the
three selected low cells and the three corresponding high cells and
concatenates the seven result bits.

The exact identity is immediate: if the solve succeeds, then for every
registered key `i`,

`read(d, s_i) = (A_d Z_d)[i] = Y_d[i]`.

This is a static retrieval data structure related to Bloomier/XOR retrieval;
that data-structure idea is prior art.  Stage 0 tests only whether its algebra
matches this project's raw quotient and fixed 220-cell plane.

## Frozen controls

Evaluate all registered edges and the previously frozen train/evaluation
occurrence split under:

1. correct document record;
2. a fixed seed-24,001 permutation of document records;
3. all-zero records;
4. fixed seed-24,003 random four-bit records.

Controls use the correct query hashes and target-rank dictionaries.  No record
is refit for a control.

## Mandatory gates

All must pass:

1. the raw quotient exactly reproduces 3,392 skeletons, 14,146 edges, 2,248
   covered documents, maximum 65 targets per skeleton, and maximum 101
   registered skeletons per document;
2. every `A_d` has full row rank under the single fixed hash family; no
   per-document seed or retry is used;
3. every seven-bit solve is finite and all 14,146 correct-record reads are
   bit-exact, including every frozen evaluation edge;
4. every low cell is in `[0,15]`, every high cell in `[0,7]`, and exactly 220
   cells are emitted per document;
5. mapping cell indices to the odd integer levels `2*i-15` is exact through a
   BF16 round trip and back to the same indices;
6. shuffled, zero, and random records each score at least 20 absolute points
   below the correct record on evaluation edges;
7. the maximum semantic payload bound, the 763-bit logical solution, and the
   fixed 880-bit physical record are all reported without calling unused bits
   compression gains.

Failure closes this exact 109-slot, seven-bit XOR record.  No post-result hash,
slot count, family, rank assignment, target filtering, or threshold change is
allowed inside T24 Stage 0.

## Admission boundary

A pass admits exactly one learned-interface experiment.  That experiment must
train from zero on raw-derived views so an ordinary model maps language to the
three relation indices and maps `(relation, rank)` back to a target surface.
The dense controls receive the identical raw-derived views and target strings.
Natural QA, correct/zero/shuffle causality, protected language NLL, physical
same-graph export, untouched holdout, and unseen-seed replication remain
mandatory before any smarter-model claim.

Primary prior-art boundary:

- XOR filters: <https://arxiv.org/abs/1912.08258>
- Binary fuse filters: <https://arxiv.org/abs/2201.01174>
