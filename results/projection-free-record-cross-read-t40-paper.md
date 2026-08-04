# T40 projection-free record cross-read — paper candidate

Date: 2026-08-01  
Status: **CLOSED BY INDEPENDENT PAPER AUDIT; ZERO RUNS; RETAIN ALGEBRA ONLY**

## Independent audit verdict

T40 is not admitted to a CPU census, natural bottleneck oracle, model run, or
GPU work.  Its exact token-to-embedding expansion, streaming-softmax identity,
conditional concentration bound, and rank-32 side-path separation are valid.
They do not isolate a capability-bearing edge.

Four failures are decisive:

1. a strongest control carrying the same tape, matcher, and identity reader is
   exactly T40, so post-T35 artifact-and-reader absorption applies;
2. raw-only acquisition still bundles question-to-relation matching,
   token-position selection, role binding, coordination across four reads, and
   downstream answer use rather than one identifiable learned edge;
3. no repair-minus-harm equation propagates T32's explicit-text 9B ceiling
   through the proposed four-summary 36.6M interface; and
4. the stated resident and operation ledgers omit the exact raw-title
   normalization/index, candidate metadata, position/role codes or their
   generation work, and several numerical/runtime operations.

The proposal below is retained as the object that was audited.  Its admission
order is historical and superseded by the final decision at the end.

## P0. One-sentence edge

At unchanged checkpoint bytes, request tokens, KV shape, and generated-token
graph, replace 29 units of every width-1,024 SwiGLU by an exact raw-token plane
and four projection-free boundary reads, with the goal of retaining at least a
ten-point held-out natural-knowledge gain while protected language quality and
measured serving latency remain noninferior.

This is a continuation of T32, not a new storage idea.  T32 proved that the
correct raw records contain a 40-point natural-information advantage.  T32R
then inserted a learned rank-32 projection before the record scan and could
not transfer that information result through the bottleneck.  T40 asks whether
the model can read the exact token embeddings in their existing width-384
coordinate system, without a document projection or new Q/K/V matrices.

## P1. Named obstruction

### The T32R value bottleneck

T32R maps every record token through `P^T:R^384 -> R^32`, forms four
eight-dimensional summaries, and maps the result back through a width-32
output.  The side-path values therefore lie in a subspace of dimension at most
32.  For any embedding table whose token rows span more than 32 dimensions,
some token-embedding direction is unavailable to that side path regardless of
storage exactness.

T40 removes this loss rather than increasing the rank.  It reads the already
resident input embedding `E[token_id] in R^384` directly as both key material
and value material.  A selected value is therefore a full embedding, not a
rank-32 reconstruction.

### The T32R route bottleneck

The longest-two title rule excluded a correct document in two of 104 frozen
questions because a third literal title occurred in the predicate.  T40 does
not guess which two mentions are grammatical arguments.  It retains every
exact literal title match when the complete set has at most `H=8` records and
abstains from the data-plane path when there are more.  Relevance is left to
the learned read, and candidate-set recall is measured before training.

This fixes causal absence only on the declared exact-title domain.  It does
not solve aliases or implicit reference.

## P2. Typed raw compiler

For a corpus of `N=2,405` records, each with title `a_j` and tokenized
`title + newline + body` record `x_j` of length at most `W=128`, compile:

```text
record_ids[j, 0:128] : uint16
record_length[j]     : uint16
title_length[j]      : uint8
title_hash[j]        : uint64
sorted_doc_id[j]     : uint16
```

The hash is only a candidate index.  A query match is accepted only after the
raw token span is compared exactly with the title prefix stored in
`record_ids`; hash collision cannot create a false match.  If two documents
have the same exact title-token sequence, or a document/token ID exceeds the
declared integer domain, the compiler rejects the corpus.

The raw prompt remains unchanged.  The tokenizer-side matcher returns a
bounded list

\[
C(q)=(j_1,\ldots,j_c),\qquad c\le H=8,
\]

containing all exact title matches.  Candidate IDs and match spans are charged
request metadata, not new prompt tokens or KV positions.  If more than eight
distinct documents match, `C(q)=empty` and the ordinary model path is used.

This is an autonomous, explainable structure extractor only at the declared
level: it builds an exact `title -> raw record` relation from raw fields and
returns the exact prompt spans witnessing every route.  It does not claim to
extract semantic predicates or triples.

## P3. Projection-free repeated read

Let `h_l in R^384` be the hidden state of the final observed prompt token at a
chosen layer `l`.  For read round `r in {1,2,3,4}`, use learned diagonal query
scale `a_r`, query bias `b_r`, and output gate `g_r`, all in `R^384`:

\[
q_r=a_r\odot RMSNorm(h_l)+b_r.
\]

For candidate document `j` and valid position `i`, form

\[
k_{j,i}=E[x_{j,i}]+p_i+d_j,
\qquad
v_{j,i}=E[x_{j,i}]+\tilde p_i+\tilde d_j,
\]

where `p_i, d_j, p_tilde_i, d_tilde_j` are fixed, generated signed-sinusoidal
codes.  They are not learned tables.  The candidate-role code depends on the
order of exact prompt occurrences, not on evaluator labels or hidden support
documents.

One read is ordinary attention with identity K/V projections and a diagonal Q
projection:

\[
z_{j,i}=q_r^T k_{j,i}/\sqrt{384},
\]

\[
o_r=\sum_{j\in C(q)}\sum_{i<length_j}
    softmax(z)_{j,i}v_{j,i},
\]

\[
h_l' = h_l + g_r\odot o_r.
\]

Place the four reads before four distinct late blocks so ordinary model
computation changes the query between reads.  Only the final observed prompt
position receives the side update.  Later generated tokens obtain its effects
through the ordinary cached K/V states; the record path is not rerun during
decode.

The exact layer indices, normalization placement, finite precision, and
whether the fixed position/role codes enter keys, values, or both must be
frozen before an executable reference.  This draft does not authorize choosing
them after a result.

## P4. Positive theorems

### Theorem 1: exact persistence and semantic expansion

For every admitted record token,

\[
E[record\_ids[j,i]]=E[x_{j,i}].
\]

The persistent plane is an exact typed token tape.  Unlike T32's BF16 radix
carrier, it does not pass through RMSNorm before the embedding gather.

### Theorem 2: no value-rank bottleneck

Ignoring fixed position/role codes, the set of possible single-token values is
the row set of `E`.  Its linear span has dimension `rank(E)`, which can be 384.
T32R's side output lies in the column space of a `384 x 32` map and has
dimension at most 32.  Therefore, whenever `rank(E)>32`, there exists a token
embedding direction representable by a concentrated T40 read and not by a
T32R value path in isolation.

This is a side-path transport separation, not a whole-model expressivity
separation; the backbone residual may already contain related information.

### Theorem 3: exact streaming schedule

The softmax result can be computed in one pass without materializing scores.
For running maximum `m`, denominator `s`, and numerator `u in R^384`, update
with score `z` and value `v` by

\[
m'=max(m,z),
\]

\[
s'=e^{m-m'}s+e^{z-m'},
\]

\[
u'=e^{m-m'}u+e^{z-m'}v.
\]

At the end, `u/s` equals ordinary softmax attention in exact arithmetic.  The
workspace is one width-384 accumulator plus scalar state, not `H*W` scores or
`H*W*384` expanded records.

### Theorem 4: selected full-value approximation

If one desired cell has score at least `Delta` above every other one among at
most `n=H*W=1,024` cells, then

\[
alpha_*\ge\frac{1}{1+(n-1)e^{-Delta}}.
\]

If every value norm is at most `M`,

\[
\|o-v_*\|\le2M(n-1)e^{-Delta}.
\]

Thus T40 can return a full width-384 token value to any declared error when a
sufficient score margin exists.  The theorem does not construct that margin
from raw prose.

### Theorem 5: boundedness

Before the learned output gate, `o_r` lies in the convex hull of the finite
value set.  Consequently

\[
\|o_r\|\le\max_{j,i}\|v_{j,i}\|.
\]

The read itself cannot exhibit the recursive polynomial explosion that closed
the Cayley program tree.  Stability after the learned gate and ordinary blocks
still requires a finite-precision reference.

## P5. Incomplete prospective resource ledger

The frozen small model has ten layers, hidden width 384, BF16 weights, and
baseline SwiGLU width 1,024.  Removing one FFN-width unit from every layer
saves

\[
10\cdot3\cdot384\cdot2=23,040\text{ resident bytes}
\]

and 11,520 multiplications per processed token.

### Resident state

| component | bytes |
|---|---:|
| `2,405 x 128` uint16 record IDs | 615,680 |
| 2,405 uint16 record lengths | 4,810 |
| 2,405 uint64 title hashes | 19,240 |
| 2,405 uint16 sorted document IDs | 4,810 |
| 2,405 uint8 title lengths | 2,405 |
| four BF16 query scales, biases, and output gates | 9,216 |
| **total** | **656,161** |

Removing 29 width units supplies 668,160 bytes, leaving 11,999 bytes of
matched slack and a candidate FFN width of **995**.  Alignment/padding must fit
inside this slack or force a newly declared width before execution.

### Active arithmetic upper bound

At the worst declared `H=8`, `W=128`, and four reads, query scaling, score dot
products, weighted values, and output gates need at most

\[
4(2\cdot8\cdot128\cdot384+2\cdot384)
=3,148,800
\]

multiplications per request, plus additions, 4,096 exponentials, comparisons,
integer indexing, and normalization.  The 29-unit FFN reduction saves

\[
29\cdot11,520=334,080
\]

multiplications per processed prompt or generated token.  Arithmetic break-
even is therefore ten total processed tokens.  This is not a latency claim.

Without cross-read caching, worst-case BF16 embedding traffic is

\[
4\cdot8\cdot128\cdot384\cdot2=3,145,728\text{ bytes/request},
\]

plus 8,192 record-ID bytes and small metadata.  The online-softmax accumulator
needs about 1.5 KiB in FP32, excluding implementation padding and query state.
No record token obtains a KV entry, and the generated-token path performs no
record scan.

The tokenizer-side title matcher must separately report rolling-hash work,
binary searches, exact verification comparisons, request metadata, and p50,
p95, and p99 CPU latency.  The 64-bit hash is never trusted without exact
verification.

## P6. Counterexample and scope

T40 deliberately fixes the key/value gauge to the input-embedding coordinate.
Apply an arbitrary orthogonal rotation to the backbone hidden state while
leaving `E` fixed.  An unrestricted dense Q projection can undo the rotation;
a diagonal scale and bias generally cannot.  Therefore projection-free reads
are not universally as expressive as learned Q/K/V attention.  Their possible
advantage is an acquisition and resource bias: one shared residual coordinate
must serve language modeling and exact record access.

Other hard limits are explicit:

- exact-title matching does not recover aliases or implicit entities;
- if more than eight records match, the data path abstains;
- one read is one width-384 convex summary and cannot preserve an arbitrary
  collection of 1,024 independent width-384 values;
- four reads do not prove four distinct or correct selections;
- ordinary next-token training may never create the needed score margins;
- the compiler extracts an exact title/record index, not latent semantic
  predicates; and
- logical arithmetic parity does not imply H100/H200 latency parity.

The raw-only training hypothesis must therefore be isolated.  A record-
reconstruction or held-out-span objective may be derived from the same raw
documents, but every matched arm receives the identical derived examples,
tokens, and training compute.  Questions, answers, support labels, pretrained
teachers, and evaluator feedback are forbidden to the writer and primitive
reader training.

## P7. Strongest controls

Any eventual composition screen must include, at equal complete serving cost:

1. the width-1,024 dense frontier baseline;
2. width 995 plus all 656,161 bytes as unrestricted learned parameters;
3. an Engram-like conditional learned memory;
4. learned per-document summaries with the same literal candidate matcher;
5. the T32R rank-32 projected scan;
6. the same uint16 tape with a parameter-matched learned Q/K projection;
7. T40 with correct, zero, shuffled-record, wrong-record, and shuffled-title
   controls; and
8. explicit prompt/RAG evidence, priced separately as an information ceiling.

A control granted the same tape, matcher, and identity reader is T40 and must
tie it.  The research claim is not a larger function class than conditional
memory.  The possible separation is that typed raw acquisition plus shared
embedding reuse spends the same bytes and online work more effectively than
learned document payloads.

## P8. Prior-art boundary

Lookup memory, external memory, identity/value-only attention, and layer-wise
token injection are not new families.  Relevant strong controls include:

- [Engram](https://arxiv.org/abs/2601.07372) for iso-parameter,
  iso-active-FLOP conditional lookup;
- [TIDE](https://arxiv.org/abs/2605.06216) for token-ID memories injected at
  every layer;
- [Bank of Values](https://arxiv.org/abs/2606.02780) for context-free
  token-specific deep-layer values; and
- [Language Model Memory and Memory Models for Language](https://arxiv.org/abs/2602.13466)
  for the evidence that causal next-token training alone forms weak arbitrary-
  access memories and that explicit retention objectives are a strong
  control.

T40 claims none of those components independently.  Its provisional research
object is the fixed-budget conjunction of an exact raw title/record compiler,
no new per-record continuous vectors, reuse of one existing token embedding
table, full-rank identity-coordinate values, and repeated prefill-only reads
with no added prompt or KV tokens.

## P9. Proposed breakthrough propagation and admission order

T32's correct-record ceiling improved by 43.2692 points over question-only and
41.3462 points over shuffled records.  A ten-point candidate gain requires the
deployable path to retain roughly one quarter of that observed causal ceiling.
This is large enough to test; it is not a prediction that T40 will do so.

Before any model or GPU run:

1. an independent reviewer must validate or reject this algebra and ledger;
2. a sealed CPU census must measure exact-title candidate count, full support
   recall as evaluator-only evidence, abstention rate, exact index bytes, title
   token collisions, prompt-length distribution, and the arithmetic margin;
3. two independent CPU implementations must agree on streaming-softmax output
   and gradients on exhaustive small worlds;
4. an adversarial gauge-rotation world must fail exactly as P6 predicts;
5. an exact-key/value world must demonstrate full-value transport and show the
   rank-32 control's declared subspace failure; and
6. a natural bottleneck oracle must establish that four width-384 repeated
   summaries can retain a breakthrough-sized fraction of T32's evidence gain
   before from-zero language training.

Only after those blocks pass may a small from-zero, multi-seed composition
screen be preregistered.  The required model result remains at least ten
absolute title-disjoint knowledge/reasoning points over the best matched arm,
protected NLL within 0.5%, identical checkpoint/KV/request-token bounds, and
noninferior p50/p95/p99 serving behavior.  No CPU corpus run, local GPU run, or
rental is admitted by this draft.

## Final decision

The independent review rejected the proposed admission order before corpus
access.  In particular:

- the listed 656,161 bytes are internally arithmetically correct, but they do
  not include an exact raw-byte title dictionary or all boundary-sensitive
  tokenizer normalization state;
- storing full key/value position and role code families would add 208,896
  bytes, while generating them must instead charge the complete arithmetic,
  constants, traffic, and workspace;
- the 3,148,800 multiplication count omits RMSNorm, online-softmax scalar
  rescaling, divisions, matcher work, code generation, and the exact
  finite-precision schedule;
- a valid T32R control would need the same four-layer schedule, not its earlier
  one-read placement; and
- a label-privileged natural oracle could at most create another transport
  ceiling.  It could not prove raw-only acquisition or same-control
  superiority.

Causal use is possible only when a read occurs before the selected block's
Q/K/V projection, so the updated boundary state enters that layer's cached K/V.
The paper never froze that placement or its custom/fused execution cost.

Retain only the following blocks:

1. direct `uint16` records expand exactly through the shared token embedding;
2. identity-coordinate values avoid the declared rank-32 side-output subspace;
3. streaming softmax is exact in real arithmetic with width-sized accumulator
   state; and
4. score-margin concentration and pre-gate convex-hull bounds are correct.

These are transport primitives, not a semantic reader or a model result.  No
CPU census, local GPU experiment, rental, or rescue modification is admitted.
