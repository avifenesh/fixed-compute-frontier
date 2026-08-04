# Reversible slot-grammar plane T33 — frozen Stage-0 CPU preregistration

Status: **FROZEN BEFORE IMPLEMENTATION, NATURAL CENSUS, OR SUPPORT ACCESS**  
Date: 2026-07-31

## Decision question

Does the fixed raw corpus expose an unambiguous positive-saving one-hole rule
edge inside at least 89.44% of the support sentences required by the frozen
two-record natural questions, while fitting the declared sidecar envelope?

This stage tests B0--B3 only.  It cannot establish that a natural question maps
to a rule, that the middle span is answer-sufficient, that a small model can
reason from it, or that the complete served graph is faster.

## Phase boundary

The executable has two processes and two source files:

- `experiments/reversible_slot_grammar_plane_t33_stage0.py` is Phase A;
- `experiments/reversible_slot_grammar_plane_t33_coverage.py` is Phase B.

### Phase A — raw compiler

May read only:

`data/hotpot-real-prose-t12/candidate-raw-prose.jsonl`

Expected SHA-256:

`a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`

and the locally cached tokenizer artifacts for
`HuggingFaceTB/SmolLM2-135M` revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.  Network access is forbidden and
every tokenizer artifact path/hash is reported.

Accessible corpus fields are exactly `document_id`, `title`, and `text`.  Questions,
answers, supporting titles/sentences, the Hotpot source parquet, trained
models, parsers, embeddings, external corpora, and network access are
forbidden.  The pinned tokenizer is a deterministic coordinate system, not a
semantic model.

Phase A writes a deterministic compiler result and its SHA-256.  Phase B may
not run unless all Phase-A integrity and synthetic gates pass.

### Phase B — sealed coverage evaluator

May read the frozen Phase-A result, candidate corpus, sealed evaluator, and
pinned Hotpot source only to map support sentence identities back to raw
sentences.  It may not alter a token rule, threshold, proposal, edge, ID,
ordering, or byte field.

Expected sealed evaluator SHA-256:

`5a0413f53bfc0fc73b5e66c941c708077624908095934af740c84d234b9fea6b`

Expected source parquet SHA-256:

`c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`

Answers are not needed; Phase B may parse the sealed schema but must never
access, branch on, serialize, or report the `answer` value.  Support metadata
is used only for scoring after the compiler object is frozen.

## Frozen token and sentence observer

1. Serialize each record exactly as `title + "\n" + text`, matching T32.
2. Use the pinned fast tokenizer with `add_special_tokens=False` and offset
   mapping.  The first 128 token IDs are the lossless served record; later IDs
   may help discover rules but cannot receive a served edge.
3. Split the original `text` at the raw character immediately after `.`, `?`,
   or `!`; retain punctuation and a final unterminated nonempty span.  Shift
   these spans by `len(title)+1` into serialized-string coordinates.
4. A tokenizer token belongs to a sentence iff its nonempty character interval
   lies wholly inside that sentence.  A token crossing a sentence boundary
   makes that sentence ineligible and is reported.
5. Find nonoverlapping, case-sensitive exact raw-character occurrences of the
   document's own title inside each sentence.  Replace every contiguous set of
   tokenizer tokens wholly covering such an occurrence by the abstract symbol
   `<SELF>=65535`; partial-token boundaries make that title occurrence
   ineligible and are reported.
6. Preserve an alignment from every normalized symbol boundary to the original
   serialized tokenizer boundary.  A proposed middle emits a served edge only
   when both boundaries map exactly, the raw span is nonempty, and
   `raw_start + raw_length <= 128`.

All ordinary normalized symbols are the original tokenizer IDs in
`[0,49151]`; only `<SELF>` uses 65535.  Do not casefold, prepend the document
container, resolve aliases or pronouns, stem, lemmatize, delete stopwords,
normalize numbers, or merge sentences.

## Frozen pair anti-unification

Only pairs of normalized sentences from distinct documents are compared.
Pairs must have the same first token and the same last token; both become part
of nonempty anchors.  Identical sequences are excluded from one-hole proposal
generation and reported as the exact-repeat control.

For a pair `(x,y)`:

1. compute the full common-prefix length `L_p` and full common-suffix length
   `L_s`;
2. enumerate positive `p<=min(L_p,127)` and `s<=min(L_s,127)` that leave a
   one-through-127-token residual middle in both strings;
3. select maximum `p+s`, breaking ties by larger `p` and then lexical proposal
   serialization;
4. propose the selected prefix/suffix, or nothing if the feasible set is
   empty.

Proposal serialization is the compact UTF-8 JSON encoding of `(a,b)`.

The independent reference exhaustively enumerates every nonempty
prefix/suffix pair that reconstructs both strings and selects maximum total
anchor length, breaking ties by longer prefix then lexical serialization.  It
must equal the direct LCP/nonoverlap-LCS implementation on every pair over an
alphabet of size three with lengths one through six.

## Frozen occurrence and code-gain rule

A proposal `(a,b)` matches normalized sentence `x` iff:

- `x` starts with `a` and ends with `b`;
- the two anchors do not overlap; and
- the residual middle has length one through 127.

Occurrences are deduplicated by `(document_id, sentence_index, proposal)` and
retain exact raw witnesses.  A proposal needs at least two occurrences in two
distinct documents.  An occurrence count above 16,383 is a declared format
failure; no clipping or saturation is allowed.

Let `Sigma` be the distinct normalized token alphabet including `<SELF>` and
let

`B = 16`, because the frozen alphabet contains 49,152 tokenizer IDs plus one
typed `<SELF>` symbol and fits in 65,536 codes.

For proposal anchor length `A`, occurrence count `m`, fixed metadata cost
`H=60` bits, and per-occurrence logical reference cost `R=19` bits, define

`G = B*(m-1)*A - H - m*R`.

`R=19` is twelve rule-ID bits plus seven middle-length bits.  `H` is two
seven-bit anchor lengths, one fourteen-bit occurrence count, and one 32-bit
integrity checksum.  The checksum itself is stored later; `G` is a frozen MDL
selection score, not credit against physical resident bytes.

A proposal is eligible iff `G>0` and the support conditions above hold.

## Frozen ambiguity rejection

Build the complete eligible proposal set first.  For each proposal, form its
set of `(document_id,sentence_index)` occurrences.

A proposal is retained iff its occurrence set is disjoint from the occurrence
set of every other eligible proposal.  Therefore any sentence matched by two
eligible proposals causes every proposal touching that sentence to be
rejected.  No coverage, gain, anchor length, frequency, lexical, or semantic
tie-breaker may rescue a conflict.

Retained rule IDs are assigned by lexical proposal serialization.  More than
4,096 retained rules is a failure; no truncation is allowed.

Each retained occurrence emits one logical edge:

```text
(document_id, rule_id, raw_bpe_start, raw_bpe_length,
 raw_sentence_witness)
```

More than one edge for the same `(document_id,rule_id)` is a declared duplicate
and not silently selected.  Report it; the keyed-reader census treats that key
as unusable.

## Frozen persistent-byte census

Count actual serialized binary fields, not JSON development artifacts:

- direct raw base: `2405*128*2` token-ID bytes plus one `uint16` length per
  document;
- rule anchors: one `uint16` per normalized anchor token plus fixed offsets,
  lengths, and checksums;
- each materialized edge: exactly four bytes for the 32-bit layout in the
  paper;
- per-document edge offsets/counts;
- duplicate-key and corruption metadata;
- no credit for raw MDL gain because the lossless base remains present.

The candidate image is little-endian and has these frozen sections: a 64-byte
header, `uint16[documents,128]` raw tokens, `uint16[documents]` raw lengths,
16-byte rule records, `uint16` anchor tokens, 8-byte document-directory
records, and 4-byte edges.  Every section starts at the next 64-byte boundary
and all padding is charged.  A rule record contains `uint32 anchor_offset`,
`uint8 prefix_length`, `uint8 suffix_length`, `uint16 occurrence_count`,
`uint32 CRC32`, and four zero reserved bytes.  A document record contains
`uint32 edge_offset`, `uint16 edge_count`, and `uint16 flags`.

An edge packs, from least-significant to most-significant bits, a 12-bit rule
ID, 7-bit raw start, 7-bit `(length-1)`, and six flags.  Flags are: unique
`(document,rule)` key, normalized-witness exact, raw-witness exact, span inside
the stored record, dictionary-checksum exact, and reserved zero.  Duplicate
keys remain stored with the unique bit clear and are unusable; they receive no
unpriced side table.  The 64-byte header contains magic/version, section
counts, eight reserved zero bytes, and a 32-byte SHA-256 integrity field;
section offsets are derived from the frozen order, counts, element sizes, and
64-byte alignment.  Development JSON,
timings, witnesses, titles, and document IDs are not resident-state bytes.

Report total candidate-state bytes before any query mapper or neural reader.
For reference, 48 units of ten-layer width-384 SwiGLU cost
`48*10*3*384*2 = 1,105,920` BF16 bytes.  Stage 0 must leave at least 131,072
bytes inside that envelope for a future mapper/reader; therefore raw base plus
all grammar state must be at most **974,848 bytes**.  This is a paper budget,
not evidence that width 976 preserves quality.

## Frozen synthetic microbench worlds

Phase A tests all before natural output:

1. exact two-rule, three-instance planted world satisfying every theorem
   assumption;
2. a one-shot rule, which must not be retained;
3. boundary-nondiverse fills, which must not claim exact boundary recovery;
4. two rules with one broad positive-saving cross-rule proposal, causing
   conflict rejection;
5. two latent rules sharing exactly the same observable anchors; the raw
   compiler must produce the same merged surface rule as a one-rule world, and
   the planted evaluator must label the latent partition non-identifiable;
6. a sentence with two eligible holes, causing rejection;
7. exact repeated sentences, counted only in the exact-repeat control;
8. own-title normalization, repeated own-title occurrences, Unicode,
   punctuation, tokenizer tokens crossing a sentence/title boundary, empty
   residuals, and 127/128-token record boundaries;
9. reverse document order and reverse proposal construction order;
10. corrupt witness and corrupt checksum detection.

The direct matcher and an independent prefix/suffix inverted-index matcher
must produce identical proposals, occurrences, conflicts, and retained rules
in all worlds.

## Frozen natural metrics

Phase A reports:

- documents, raw/normalized sentences, full/stored BPE tokens, and tokenizer
  provenance;
- sentence-pair comparisons after first/last bucketing;
- unique proposals, eligible proposals, and exact-repeat groups;
- eligible and retained occurrence counts;
- proposals and sentences rejected for ambiguity;
- retained rules and their occurrence/gain/anchor distributions;
- raw document coverage;
- edges per document, duplicate `(document,rule)` keys, and documents above 24
  edges;
- exact normalized and original BPE witness failures, boundary rejections, and
  pointers outside the stored 128-token record;
- deterministic object hashes;
- binary persistent-byte ledger;
- CPU seconds, wall seconds, and peak RSS.

After the compiler hash is frozen, Phase B reports:

- evaluator questions and required `(title,sentence_id)` support items;
- required support items mapped exactly back to candidate raw documents;
- fraction of required support sentences containing at least one retained,
  unique-key edge;
- fraction containing exactly one such edge;
- per-question fraction with usable edges in both required documents;
- edge middle-span and anchor lengths on support sentences;
- shuffled-title and shuffled-rule coverage controls without refitting, using
  respectively fixed PRNG seeds 33001 and 33003.

Support-sentence edge coverage is a structural necessary condition only.  It
does not prove that the middle is answer-sufficient.

## Mandatory gates

All must pass:

1. Every input hash, tokenizer revision/artifact, and field boundary matches;
   Phase A never opens a forbidden file, accesses the network, or imports a
   model/parser/embedding package beyond the pinned tokenizer runtime.
2. Exhaustive anti-unifier agreement, all ten synthetic worlds, witness
   reconstruction, checksums, independent matchers, and order invariance pass.
3. At least one and at most 4,096 rules are retained with zero unresolved
   ambiguous occurrence admitted.
4. At least 89.44% of required per-document support sentences contain a
   retained edge with a unique `(document,rule)` key.
5. At least 80% of evaluator questions have such usable edges in both required
   documents.
6. At least 95% of documents containing any retained edge have at most 24
   edges; no truncation or priority rule is allowed.
7. Raw base plus complete grammar state is at most 974,848 resident bytes,
   leaving the frozen 128-KiB mapper/reader reserve.
8. Phase B maps every support item to exactly one raw source sentence and does
   not read answers.
9. Every object is finite and the stage uses CPU only.

## Decision rule

Failure closes this exact reversible one-hole grammar bridge.  Do not change
tokenization, normalization, sentence boundary, pair buckets, hole length,
code constants, ambiguity rule, rule cap, edge cap, byte cap, or coverage
thresholds after observing the result.

Passing admits only a separately frozen decoded-middle information oracle.  It
does not admit cloze/query-mapper training, a model download, H100 work,
physical integration, or a smarter-model claim.
