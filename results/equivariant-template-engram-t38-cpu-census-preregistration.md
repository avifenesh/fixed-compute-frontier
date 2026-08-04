# T38 equivariant template Engram — CPU opportunity-census preregistration

Date frozen: 2026-08-01  
Status: **FROZEN BEFORE CORPUS MATERIALIZATION OR CENSUS; CPU ONLY; NO MODEL OR GPU**

## Binary question

At equal table-head count and admitted-record capacity, does the exact
renaming quotient plus covariant `COPY/LITERAL` payload expose at least a
three-point decoded next-token opportunity over the strongest concrete lexical
dictionary on untouched natural prose, mathematical text, and code?

This census cannot establish model capability, benchmark gain, or serving-cost
parity.  It can cheaply falsify the necessary natural-coverage and reuse premise
before any model is trained.

## What is already proved

The paper derivation proves that the template is a complete orbit label under
the declared token-permutation group, the binding vector co-transforms, and a
valid relative operation is exactly equivariant.  It proves an orbit/record-
sharing factor against a concrete-key dictionary on an equivariant source law.
It does not prove that natural text follows that law often enough to matter.

## Sole empirical question

The unresolved fact is the joint natural frequency of document-held-out events
for which:

1. a template learned from raw next-token counts is stable;
2. its concrete binding tuple was not observed under that template in train;
3. the covariant operation decodes the correct token; and
4. an equal-capacity concrete lexical dictionary is wrong or abstains.

No task labels, questions, answers, parser, entity recognizer, pretrained model,
or benchmark scores may enter the compiler.

## Frozen raw sources

All sources are read as independent documents and are pinned before download:

| stratum | repository/config | revision | rows at freeze | text/id fields |
|---|---|---:|---:|---|
| prose | `HuggingFaceTB/smollm-corpus`, `fineweb-edu-dedup` | `3ba9d605774198c5868892d7a8deda78031a781f` | 190,168,005 | `text`, `id` |
| math/reasoning | `open-web-math/open-web-math`, `default` | `fde8ef8de2300f5e778f56261843dab89f230815` | 6,315,233 | `text`, `url` |
| code | `transformersbook/codeparrot`, `default` | `1525880546992f12c04c5ae4cf5c4d1e80ca04a4` | 477,249 | `content`, `repo_name:path` |

The corpus builder must verify each Hub repository SHA before and after fetch.
Because the public rows endpoint is not revision-addressable, every block must
also report the frozen `num_rows_total` above and `partial=false`; otherwise the
run fails.  The seal records the final materialized corpus SHA-256.  This binds
all future replay to the fetched bytes, while explicitly not claiming a
cryptographic proof that Hugging Face's viewer derivative was built from the
named Git commit.
It queries the public datasets-server rows endpoint in **32 deterministic
blocks per stratum, 32 rows per block**.  For block `i`, the offset is

\[
\operatorname{uint64be}(\operatorname{SHA256}(
\texttt{"t38-census-v1|"}+\text{revision}+\texttt{"|"}+i)[0:8])
\bmod (R-31).
\]

Canonical IDs must be nonempty.  Duplicate canonical IDs within a source and
exact duplicate text within or across sources are deduplicated before
selection.  Documents are ranked by SHA-256 of the byte-exact serialization
`UTF8(source + "|" + canonical_id + "|" + lowercase_hex_SHA256(text))`, and
the first **768 per stratum** are retained.  Fewer than 768 is a hard data
failure; no replacement sample is allowed.  Empty documents are removed.  The
materialized JSONL and its SHA-256 are recorded.

Document split is independent and deterministic:

\[
s=\operatorname{uint64be}(\operatorname{SHA256}(
\operatorname{UTF8}(\texttt{"t38-split-v1|"}+source+\texttt{"|"}+id+
\texttt{"|"}+\operatorname{lowercase\_hex\_SHA256}(text)))[0:8])\bmod20.
\]

- train: `s < 14`;
- development audit: `14 <= s < 17`;
- untouched adjudication: `17 <= s < 20`.

The adjudication split is opened exactly once by the census.  No threshold or
candidate may be changed afterward.

## Frozen tokenizer and truncation

- tokenizer: `HuggingFaceTB/SmolLM2-135M`;
- revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;
- implementation: `tokenizers==0.22.2` loading the pinned `tokenizer.json`;
- test runner: `pytest==9.0.2`, executed before the adjudication seal;
- no special tokens are inserted;
- at most the first **1,024 token IDs** of each document are used;
- contexts never cross document boundaries;
- no EOS prediction event is added.

The script records tokenizer-file SHA-256, vocabulary size, per-stratum
documents, tokens, and split counts.

## Frozen quotient

The literal set `L` is the **8,192 most frequent token IDs in train only**,
with smaller token ID breaking frequency ties.  All other vocabulary IDs form
`E`.  Development and adjudication frequencies cannot alter the partition.

Three heads use context lengths `n in {8,16,32}`.  For each train event:

- the T38 key is the exact tagged literal/equality template;
- the T38 outcome is `LITERAL(y)`, `COPY(j)`, or bottom;
- bottom occurrences count against operation purity and are never predicted;
- outcome-count ties choose the smaller integer encoding; because bottom is
  encoded as `-1`, a bottom/non-bottom tie excludes the cell rather than
  optimistically predicting;
- the lexical-control key is the concrete token-ID `n`-gram and its outcome is
  the concrete next token.

Each head admits at most **65,536 records**.  T38 and the static-template
control require a key to occur at least 16 times in at least four train
documents; T38 additionally requires train winner purity at least 98%.
Concrete lexical and concrete-relative controls receive **no support,
document, or purity floor**: they may spend the same capacity on any observed
training key.  Within a head, keys are admitted by descending winning-outcome
count, then descending total support, then ascending SHA-256 of the exact
serialized key.  This gives the primary concrete control the strongest frozen
raw-count policy rather than inheriting the candidate's structural filter.

The logical serving record is frozen at 24 bytes for both systems: 128-bit key
fingerprint, 16-bit payload, 8-bit confidence, 8-bit flags, and 32-bit saturated
train support.  The 8-bit confidence is
`floor(255 * WilsonLCB)/255`; support is used by the frozen tie-break and is not
free metadata.  The census therefore compares three reads and at most
4,718,592 logical record bytes per system.  It uses exact tuple dictionaries
for measurement.  A later packed index, slot load factor, traffic, and latency
ledger are still required; these logical bytes are not a physical-serving
proof.  A keyed 128-bit fingerprint introduces the explicitly reported
union-bound error currency `epsilon <= N*Q/2^128`; any observed retained-key
collision fails.

## Frozen read and fallback

Each matching head emits one token and its quantized train Wilson 95% lower
confidence bound.  Across heads, choose the largest lower bound, then larger
train support, then longer context.  If no head matches, abstain.  The same rule
is used for T38 and every control.  There is no language-model fallback in this
opportunity census.

For every adjudication token event define

\[
d_i=1[T38_i=Y_i,\ Lex_i\ne Y_i]
    -1[Lex_i=Y_i,\ T38_i\ne Y_i],
\]

where abstention is not correct.  Wrong non-abstaining predictions are charged.
The stratum score is the mean of `d_i` over **all** eligible token events; the
aggregate is the unweighted mean of the three stratum scores.

## Frozen controls

At identical head count and record cap, report:

1. concrete lexical `n`-gram -> concrete token (primary control);
2. concrete lexical `n`-gram -> relative opcode where representable;
3. equality template -> static concrete token, without covariant read;
4. T38 with operation payloads permuted across admitted records using seed
   `38001`, separately within literal payloads and within copy payloads of the
   same template arity so the shuffle cannot create invalid copy indices;
5. T38 queried with a deterministic nonzero cyclic shift of every binding
   vector of arity at least two using seed `38002`; and
6. full T38.

Exhaustive unit tests over vocabularies of size at most four must verify
canonicalization, the converse orbit test, operation validity, equivariance,
copy-index examples, deterministic admission, and repair-minus-harm scoring.

## Frozen statistics

- report every metric by stratum, context length, split, and control;
- report raw event counts, abstention, correct, wrong, and matched-cell counts;
- report train and adjudication operation purity;
- report distinct binding tuples per template, excluding exact duplicate
  documents and exact repeated contexts;
- separately report adjudication events whose `(template,binding)` tuple was
  absent in train;
- paired document bootstrap: 10,000 resamples, seed `38003`, nearest-rank
  percentile 95% interval for the equal-stratum aggregate advantage; the lower
  endpoint uses zero-based index `ceil(0.025*B)-1`, a conservative convention;
- no multiple candidate, threshold, literal-boundary, or window sweep is
  permitted beyond the three frozen heads.

## Decision

Advance only if **all** gates pass:

1. aggregate T38-minus-lexical advantage is at least **3.0 absolute points**
   and its paired-bootstrap lower bound exceeds **2.0 points**;
2. aggregate T38 purity is at least **97%**, and is also at least **97%** on
   unseen-binding events;
3. at least one T38 template has 20 adjudication occurrences, and every such
   active template has at least **95%** purity;
4. median distinct train binding tuples among admitted templates is at least
   **8**, and the 75th percentile at least **16**;
5. correct non-literal `COPY` predictions cover at least **3% of all
   adjudication token events** in the equal-weighted math/code aggregate;
6. shuffled payload loses at least 90% of T38's positive aggregate advantage;
   on events where the unshuffled T38 read selects a `COPY` with binding arity
   at least two, the nonzero binding shift removes at least 90% of T38's correct
   predictions (the binding control is not compared against literal or unary
   advantage); and
7. no data, hash, deterministic-replay, or exhaustive-algebra check fails.

The implementation creates an exclusive one-shot opening seal **before the
first corpus row is fetched or tokenized**.  It binds SHA-256 hashes of this
preregistration, the implementation, and its exhaustive test file.  Immediately
after corpus materialization and structural verification, an exclusive
companion seal binds the opening-seal SHA-256, corpus SHA-256, and manifest
SHA-256; the pair is the complete adjudication seal.  There is no prepare-only
mode.  A pre-existing corpus without that pair, a failed sealed run, or an
existing result forbids automatic rerun or overwrite.

A failure closes the fixed single-token, fixed-window T38 candidate.  It does
not close multi-token identifier quotients, structured span binders, or other
equivariant memory algebras.  A pass admits only a finite packed-table ledger
and a CPU lookup microbenchmark; it does **not** admit model training or GPU
use.
