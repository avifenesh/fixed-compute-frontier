# T32R proof-first capability audit

Status: **PRE-RUN HOLD; FROZEN H100 IMPLEMENTATION IS NOT ADMITTED**  
Date: 2026-07-31

No H100 timing cell, implementation source, or result existed when this audit
was written.  The earlier preregistration remains an immutable historical
artifact; this audit prevents its execution rather than changing it after a
result.

## Verdict

T32R contains three useful primitives:

1. an exact whole-record codec;
2. an exact path from stored token IDs back to the model's shared token
   embeddings;
3. a bounded query-conditioned scan with an exact streaming-softmax schedule.

Those primitives do not yet compose into the active research goal:

> from-zero training reads raw prose, autonomously extracts useful structure
> into the digital plane, and demonstrates substantially better held-out
> knowledge and reasoning than frontier training at identical serving cost.

The decisive gaps are upstream of GPU timing:

- the compiler stores raw tokens; it does not extract semantic structure;
- the query contains exact document-title surfaces and a deterministic
  corpus-title tokenizer recognizes them; this is raw-only routing on the
  frozen domain, but it does not cover aliases or implicit references;
- the natural information ceiling used a 9B reader over explicit text, not the
  proposed rank-32 reader in the 36.6M model;
- the conditional softmax theorem assumes the useful score margin instead of
  constructing it from raw-only training;
- the BF16 radix table is byte-dominated by a direct typed `uint16` token plane
  before any hardware measurement.

Therefore an H100 result would answer only whether a dominated physical block
fits in one implementation.  It could not decide whether T32R makes a smarter
model.

## Goal-alignment audit

| Required property | Current evidence | Decision |
|---|---|---|
| reads raw prose | deterministic tokenization and exact pack | pass |
| autonomously extracts useful structure | pack preserves the entire token sequence; it compiles document identity but no semantic relation or evidence structure | fail beyond exact record identity |
| autonomous addressing | deterministic case-folded matching uses only the raw question and title dictionary, but longest-two selection matches the sealed support pair on only 102/104 questions | raw-only provenance passes; autonomous route correctness fails |
| large causal information availability | correct records gave a fixed 9B reader about +41 to +43 points over shuffled/question-only controls | pass as an information ceiling only |
| proposed bottleneck can use it | rank-32 scan has only synthetic algebra and a conditional margin theorem | open |
| better than strongest matched memory | no learned-summary, typed-token, Engram-like, or equally routed control result | open |
| unchanged serving cost | arithmetic ledger is plausible; latency is unmeasured | open |
| substantially smarter production LLM | no candidate-model result | fail/not established |

## Theorem 1: the record plane addresses a real information problem

Let a record be

\[
X=(X_1,\ldots,X_L),\qquad X_i\overset{iid}{\sim}\operatorname{Unif}([V]),
\]

and let a query request any coordinate `i`.  Suppose persistent state `S` has
at most `B` bits and a decoder estimates every requested token with average
error at most `epsilon`.  Fano's inequality and independence give

\[
B\ge I(X;S)
  \ge L\left(\log_2V-h_2(\epsilon)
             -\epsilon\log_2(V-1)\right).
\]

**Proof.**  For each coordinate, Fano bounds

\[
H(X_i\mid S)\le h_2(\epsilon_i)+
                    \epsilon_i\log_2(V-1).
\]

Subadditivity gives `H(X|S) <= sum_i H(X_i|S)`.  Applying concavity to the
average error and subtracting from `H(X)=L log2(V)` proves the inequality.

For `V=49,152` and `L=128`:

| target error | required state bits |
|---:|---:|
| 0% | 1,994.875 |
| 1% | 1,964.585 |
| 5% | 1,858.473 |
| 10% | 1,735.357 |

The T32 code exposes `381*6 = 2,286` robust logical payload bits, only 14.59%
above the zero-error fixed-length entropy.  Thus an exact record is a valid
answer to the **multi-query arbitrary-record** storage witness.  A single
32-dimensional BF16 learned summary has at most 512 physical bits and cannot
solve that witness for arbitrary records.

What this proves:

- a small fixed per-document summary cannot retain every possible fact in a
  high-entropy record;
- preserving token identity can create a large advantage on unpredictable
  future queries.

What it does not prove:

- natural prose behaves like independent uniform tokens;
- natural questions require all record entropy;
- T32R can locate or reason over the requested information;
- BF16 radix amplitudes are the best physical representation.

## Theorem 2: the frozen BF16 record table is byte-dominated

T32R does not transport its record through RMSNorm.  It indexes a persistent
table, decodes IDs, and immediately gathers the shared embedding.  The
scale-referenced BF16 amplitude code therefore protects against a transformation
that this reader never applies.

The frozen table uses

\[
2405\cdot381\cdot2=1,832,610\text{ bytes}
\]

for 307,840 token IDs, or 5.953125 bytes per token.  A direct typed plane uses:

| component | resident bytes |
|---|---:|
| `2405 x 128` `uint16` token IDs | 615,680 |
| 2,405 `uint16` lengths | 4,810 |
| non-record scanner state, 27,233 BF16 entries | 54,466 |
| fixed `128 x 32` BF16 position code | 8,192 |
| total | **683,148** |

One unit of SwiGLU width across ten width-384 layers costs

\[
10\cdot3\cdot384\cdot2=23,040\text{ resident bytes}.
\]

Therefore the entire direct typed plane fits after reducing width by

\[
\left\lceil683,148/23,040\right\rceil=30,
\]

from 1,024 to **994**, with 8,052 resident bytes left.  The frozen BF16 design
reduces width to 940.

The typed plane also removes base-64 digit reconstruction.  It is therefore
strictly better in persistent bytes and declared decode work.  Its realized
latency still requires a target-hardware microbench, but the width-940 BF16
implementation is no longer the strongest legal construction and must not be
timed as the active candidate.

At width 994, shrinking all ten FFNs saves 345,600 multiplies per processed
token.  Against the frozen 3,230,144-multiply reader, arithmetic break-even is
ten total processed prompt-plus-generated tokens.  This is an accounting
result, not a latency result.

## Theorem 3: exact-surface routing has a declared autonomy boundary

The frozen implementation does **not** read `supporting_titles` to route.  It
case-folds the raw question, searches the corpus-title dictionary, chooses two
non-overlapping title spans, and replaces them with typed handles.  This is a
legitimate raw-only candidate generator, not a hidden evaluator-label channel.
However, the subsequently frozen route audit found only 102/104 supporting-pair
matches.  Predicate/category mentions such as `Parker Brothers` and `Card game`
are themselves corpus titles and can displace a grammatical argument.

Let the corpus be records `D_1,...,D_N`, let the current router return a small
set of supplied handles `r(q)`, and let the data-plane contribution to the
model be

\[
F(q,D)=G(q,D_{r(q)}).
\]

For any target record `k` not in `r(q)`, choose two corpora `D` and `D'` that
are identical on all routed records but whose correct answer to `q` differs
only through record `k`.  Then

\[
F(q,D)=F(q,D'),
\]

so this path cannot answer both correctly.  This is not a weakness of scan
rank or training; the target record is causally absent.

Consequently the present architecture can claim **raw-only literal title
candidate generation**, not a correct autonomous router.  It also cannot claim
alias resolution, implicit-reference retrieval, or open-domain semantic
addressing.  Selecting the grammatical argument mentions from the literal
candidate set is a missing block with its own state/compute ledger.

## Reader algebra: what is and is not proved

The proposed scanner computes, per document:

1. a shared rank-32 projection of each gathered token embedding;
2. a depthwise three-token context feature;
3. four diagonal-metric attention scores;
4. four eight-dimensional convex summaries;
5. a role-aware linear injection into the boundary hidden state.

The concentration result

\[
\alpha_*\ge \frac{1}{1+127e^{-\Delta}}
\]

is valid **if** training constructs a score gap `Delta` at the useful position.
It does not construct the query state, projection, or metric that creates that
gap from raw prose.

Also, the statement “rank 32 necessarily aliases vocabulary tokens” is too
strong.  A generic linear projection can be injective on a finite set even
when it is non-injective on the ambient continuous space.  The correct
limitation is:

- arbitrary width-384 vectors and operations cannot be preserved through a
  rank-32 linear map;
- finite token identities may remain distinct, but their numerical margins,
  semantic geometry, and task-relevant relations are not guaranteed.

The natural oracle does not close this gap.  It gave a deep 9B model 48 or 128
explicit tokens per document.  T32R gives a 36.6M model one 32-dimensional
query-dependent summary per document.  The former is an information ceiling,
not an upper bound on the latter reader's attainable accuracy.

## Strongest-control audit

The eventual comparison must include, at equal persistent bytes and complete
served cost:

1. width-1,024 dense training;
2. width-matched free learned state without records;
3. direct `uint16` records with the same routing and reader;
4. learned per-document summaries;
5. an Engram-like conditional learned memory;
6. correct, shuffled, zero, and wrong records;
7. the same raw-only auxiliary training views for every legal arm.

A control granted the same typed conditional-addressing operator can reproduce
the T32 representation.  Therefore the possible edge is not a larger function
class than general conditional memory.  It is the conjunction of:

- near-entropy token storage;
- reuse of one shared embedding table;
- raw-only acquisition without gradient memorization;
- a learned query-dependent reader.

That conjunction, not the BF16 codec alone, is what a future experiment must
separate.

## Breakthrough-size ceiling

On the exact-title-routed 104-question slice, the information oracle improved by at most
43.2692 points over question-only.  Even granting that entire effect to the
deployable reader and assuming zero protected-quality loss, a ten-point
aggregate improvement requires at least

\[
f\ge 0.10/0.432692=23.11\%
\]

of evaluated requests to exercise this exact knowledge path.  This does not
rule out a large targeted-domain win, but it prevents a general-smarter-model
claim from a narrow title-routed benchmark.

The model reallocates 967,680 of 36,577,152 BF16 entries in the frozen design,
2.646% of parameters, and reduces every FFN by 8.20% of width.  A typed plane
reduces the required width sacrifice to 2.93%, but neither version has measured
the protected general-language cost.

## Required block DAG before another expensive run

```text
raw prose
  -> autonomous extractor/writer
  -> typed persistent state
query
  -> non-privileged router
  -> addressed record(s)
  -> proved bounded reader
  -> evidence code with a natural sufficiency ceiling
  -> matched small reasoner
  -> target-hardware serving graph
```

Each block needs a card under the v2 protocol.  The next fatal tests, in order,
are:

1. **Goal witness:** retain literal-title questions first, but require the
   router to distinguish compared arguments from title-shaped predicate and
   category mentions using raw-only training; treat alias/implicit retrieval as
   a separate harder witness.
2. **Strong-control construction:** freeze the direct typed-token plane and the
   best learned-memory control before choosing a new representation.
3. **Router microbench:** unseen aliases, distractor corpus growth, absent keys,
   entity permutation, and no privileged metadata.
4. **Reader microbench:** an exhaustive small key/value world with a
   constructive parameter witness, then held-out random records with
   zero/shuffle/wrong-record controls and measured score margins.
5. **Natural bottleneck ceiling:** test the exact rank and summary count that
   the small model receives, not explicit text given to a stronger reader.
6. **Raw-only learnability:** train the isolated router/reader without QA labels
   and compare to all matched memory interfaces.
7. **Physical microbench:** only after the semantic path survives, freeze the
   non-dominated typed representation and measure its complete H100 graph.

## Retained knowledge and closed surface

Retain:

- T32's exact BF16 scale-referenced codec as a construction for data that must
  survive a shared residual normalization path;
- the +40-point controlled natural-information result;
- exact embedding reuse after token-ID expansion;
- the rank-32 scanner reference, role-separation witness, and streaming
  online-softmax identity;
- the parameter, operation, and physical-residency accounting discipline.

Close before execution:

- the exact `BF16 radix table + width-940 FFN` H100 implementation as the
  active candidate;
- any claim of correct autonomous routing, semantic extraction, alias
  resolution, or implicit retrieval from longest-two literal title matching;
  only raw-only candidate generation remains admitted;
- any inference from the 9B explicit-text oracle to rank-32 small-model
  learnability.

This is a pre-run mathematical correction, not a negative benchmark and not a
post-result rescue.
