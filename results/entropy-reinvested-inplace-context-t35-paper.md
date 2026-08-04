# Entropy-reinvested in-place context T35 — paper candidate

Status: **CLOSED ON PAPER; CONTROL-ABSORBED, NO MEASUREMENT OR GPU RUN**  
Date: 2026-08-01

## P0. End claim

T35 asks whether a from-zero language model can preserve its complete dense
backbone while placing lossless raw prose inside the served checkpoint and
reading that prose through already-existing prompt states, at no increase in
complete resident bytes, mathematical work, KV state, or latency.

The proposed exchange is:

```text
statistical redundancy in the lossless encoding of dense BF16 weights
    -> exact raw-token plane and title index

redundant multi-token title surfaces in a routed question
    -> one bounded in-place memory read into existing prompt positions
```

The target remains the frozen exact-title natural comparison witness: at least
80% accuracy, at least ten absolute points above the strongest same-cost
control, at least a 15-point correct-versus-shuffled causal gap, protected
quality within 0.5%, and noninferior complete serving cost.

This paper does not claim that lossless compression, title routing, raw-token
memory, reverse-projected attention, or cross-attention is individually new.
The research claim is their closed resource composition: no dense capacity is
deleted, no evidence tokens or KV positions are added, and the raw plane is
written deterministically from the same prose used for from-zero training.

## P1. Construction

Train the ordinary dense backbone from random initialization on raw prose,
then freeze it.  Compile the raw plane and train only a memory-specific
rank-`a` correction to the reused Q/K/V/O projections plus scalar gates, using
raw-derived objectives with no QA labels, teacher, parser, or pretrained
encoder.  This is staged from-zero training, not adaptation of an externally
pretrained model.  After training, encode every frozen BF16 backbone tile with
an exact lossless codec.  A serving kernel reconstructs the identical BF16
tile before its ordinary matrix operation.

Use only recovered checkpoint bytes for:

1. `uint16` token IDs and lengths for every admitted raw document;
2. a deterministic exact-title dictionary and bounded candidate list;
3. codec and integrity metadata; and
4. the explicitly byte-counted memory-only low-rank corrections and gates.
   The backbone matrices themselves are reused unchanged; the corrections are
   never evaluated on the inactive path.

For an eligible question, the tokenizer-side compiler enumerates every exact
case-folded corpus-title interval.  It proceeds only if the complete set has no
overlaps and contains at most `C` occurrences and at most `C` distinct titled
records; otherwise it
abstains.  It replaces **every** retained occurrence by a typed occurrence
handle.  The handle carries only its raw occurrence index; the corresponding
candidate record receives the same fixed role index in the memory read.  The
`C` input-only handle embeddings and the matching fixed document-role features
are charged to `B_G`; they are not output-vocabulary rows or free metadata.
The compiler does not read supporting-title labels or decide which mentions
are grammatical arguments.  Unlike the failed longest-two router, extra
title-shaped mentions remain in both the prompt and candidate set for the
model to disambiguate.  T35 never resolves a nested-title ambiguity with a
semantic or length heuristic.

At one frozen layer, `R` existing late prompt positions read the candidate
records.  The read enriches those positions in place.  Later ordinary layers
write their normal K/V entries for the same shortened prompt; no memory token
is appended and no additional decode-time KV entry exists.

Prompts without a valid bounded literal-title candidate set follow the exact
dense path with their original text.  The memory path never runs during
ordinary decode; generated tokens attend to the already enriched prompt K/V.

## P2. Exact dense-backbone preservation

Let the trained BF16 weight bit string be `W`, let `E` be the lossless encoder,
and let `D` be its serving decoder.  Require

\[
D(E(W))=W
\]

bit for bit for every tile.

### Theorem 1: quality identity on the inactive domain

If a prompt does not enter the routed memory branch, the decompressed model
has exactly the same token IDs, BF16 weights, operations, and state transitions
as the dense baseline.  Its logits are therefore bit-identical, subject only
to using the same deterministic kernel schedule and accumulation order.

If the compressed kernel changes the reduction schedule, weight identity
alone gives real-arithmetic equivalence rather than bit identity.  The local
numerical gate must then establish the frozen output tolerance.  T35 may not
describe that weaker result as bit-exact.

This theorem protects model function, not latency.  Codec work and physical
scheduling remain charged.

## P3. Exact byte admission

Let the baseline contain `P` BF16 values and occupy `16P` payload bits.  Let
the complete encoded weights and their mandatory decode metadata occupy
`B_W` bits.  Define recoverable slack

\[
S=16P-B_W.
\]

Let `B_T`, `B_I`, `B_R`, and `B_G` be the complete raw-token plane,
title/index, memory-reader correction, and gate/integrity bits.  Byte admission
is the exact inequality

\[
B_T+B_I+B_R+B_G \le S.
\]

Nothing is admitted from entropy estimates.  The inequality uses the actual
exported byte streams, alignments, tile tables, checksums, decoder tables, and
allocator residency.

For the current `P=36,577,152` scratch model, the inherited T32R bundle is
683,148 bytes.  It is only a conservative threshold: that bundle includes
54,466 bytes of old scanner state and 8,192 bytes of the old rank-32 position
code, while T35 would require a different exact index/reader ledger.  Funding
this conservative bundle requires an average code length no greater than

\[
16-\frac{8(683148)}{36577152}=15.8506
\]

bits per BF16 value: a 0.934% reduction.  A one-megabyte all-in side budget
requires about 15.7706 bits per value.  These are admission thresholds, not a
claim that the current from-zero checkpoint achieves them.

For four rank-`a` corrections represented by two dense factors each,
`P_R=8da` BF16 values before biases and metadata; at `d=384,a=16`, that is
49,152 values or 98,304 raw bytes.  This is an illustrative paper rank, not
evidence that rank 16 is sufficient.

The dense control receives the same lossless codec and exactly the same total
served-byte envelope.  Compression itself is not credited as candidate
capability.

## P4. In-place raw-memory operator

Let `Z in R^(M x d)` be shared token embeddings plus frozen position/document
features for at most `C` records of length `L`, so `M=C L`.  Let `h_j` be one
of `R` existing prompt states.  Reuse an existing layer's `H` attention heads
with the memory-only rank-`a` corrections.  Ordinary self-attention continues
to use the unmodified frozen matrices.
For head `a`, ordinary raw-memory cross-attention is

\[
q_{ja}=h_jW_Q^a,\quad K_a=ZW_K^a,\quad V_a=ZW_V^a,
\]

\[
\alpha_{ja}=\operatorname{softmax}(q_{ja}K_a^T+b_{ja}),\quad
o_{ja}=\alpha_{ja}V_a.
\]

The exact reverse association retained from the earlier audit is

\[
\tilde q_{ja}=q_{ja}(W_K^a)^T,
\]

\[
\alpha_{ja}=\operatorname{softmax}(\tilde q_{ja}Z^T+b_{ja}),\qquad
o_{ja}=(\alpha_{ja}Z)W_V^a.
\]

It avoids persistent or on-demand document K/V matrices.  It is functionally
ordinary cross-attention and receives no capability credit.  Its role is to
make the complete raw-ID path potentially fundable.

With a fused stream over `Z`, the logical MAC count for `R` prompt positions is

\[
C_{mem}=R(4d^2+2HMd)+C_{adapter}.
\]

At the paper shape

```text
d=384, H=4, C=8, L=128, M=1024, R=4
```

The base part is 14,942,208 MACs.  For arbitrary per-head rank-`a` corrections,
Q and O cost `4Rda`; reverse-K and post-V each cost `(H+1)Rda`.  The complete
unfused correction bound is therefore `(2H+6)Rda`; at `H=4,a=16,R=4` this is
344,064 MACs.  The kernel must include gate, normalization,
position/document features, embedding gathers, online softmax, residual
injection, and workspace in the complete ledger; this formula is not a latency
claim.

## P5. Title-handle compute funding

Let a baseline prompt tokenize to `q_0` positions.  Replacing every one of the
bounded `c <= C` unambiguous literal-title occurrences by one typed occurrence handle
saves `s` positions and produces `q=q_0-s`.  Let `C_base(q)` be the exact dense
prefill MAC count, including attention's length-dependent term.  T35 is
MAC-admissible on a request only
if

\[
C_base(q_0)-C_base(q) \ge C_{mem}(q,C,L,R).
\]

For an arithmetic lower bound that omits attention-score savings, a width-384,
width-1024 SwiGLU Transformer layer spends

\[
4d^2+3dm=1,769,472
\]

MACs per prompt token on projections and FFN.  Across ten layers, one removed
position saves 17,694,720 MACs, already larger than the 15,286,272-MAC
base-plus-rank-16 paper reader.  This comparison covers the MAC category only.  Title matching,
handles, embedding gathers, ANS decode, online-softmax scalar work, and index
traffic remain separate integer/SFU/traffic/critical-path currencies.  They
must meet issued-work, latency, energy, memory, and cost noninferiority directly;
saved MACs do not erase them.  The exact graph, not this coarse bound, decides
admission.

If `s` is insufficient, the branch does not run and the original prompt is
preserved.  Average savings cannot subsidize a request that exceeds the frozen
per-request or workload-cell budget.

### State consequence

The candidate stores `q_0-s` ordinary prompt K/V positions, not `q_0+M`.
Temporary online-softmax state and embedding traffic are charged, but raw
memory does not persist as decode K/V.  The candidate must remain no larger in
allocated, reserved, and persistent state at every frozen concurrency cell.

## P6. Why this is not the closed T32R reader

T32R reduced every FFN and compressed two records into one rank-32 boundary
summary.  T35 changes both fatal choices:

1. the dense backbone is losslessly preserved rather than narrowed; and
2. `R` existing prompt positions are enriched at full residual width, so later
   self-attention receives a distributed evidence interface instead of one
   32-dimensional summary.

This does not prove sufficiency.  `R*d` scalars still cannot losslessly expose
arbitrary `M*d` semantic vectors, and ordinary attention mixtures may discard
binding or order.  It merely removes the specific one-summary inference used
to block T32R.

## P7. Positive construction and limits

Consider documents containing bounded literal records

```text
TITLE ; KEY_i ; VALUE_i ; ...
```

and questions that name two titles and two literal keys.  Replace every title
occurrence, including distractor titles, and carry only its raw occurrence
index into the correspondingly indexed record.  Choose linearly
independent token embeddings for keys and values.  Two heads can score the two
key occurrences, additive relative bias can select their following value
tokens, and four prompt positions can retain the two values plus typed role
bits.  A later FFN can implement equality or a finite ordered comparison.

Thus there is an explicit bounded family on which raw title matching, raw-ID
storage, in-place attention, and later reasoning solve two-record queries
without learned per-document vectors.

The construction proves function existence only.  Natural questions use
paraphrases, syntax, scope, and implicit typing.  The current formulation
actually leaves several learned blocks: handle/query adaptation, evidence
scoring, role binding, and downstream use.  Calling them one transfer
uncertainty would violate the one-unknown gate.  A synthetic world that writes
the desired key into the question does not establish any of them.

## P8. Fatal counterexamples

T35 fails or abstains on each of the following unless a later paper supplies a
new proved block:

- supporting entities are aliases or implicit and have no literal title;
- more than `C` nonoverlapping title candidates occur, forcing abstention;
- title tokenization saves too little work to fund the memory read;
- context-free token embeddings plus position bias do not preserve the syntax
  required by the question;
- four mixtures lose two independently required evidence spans;
- the question/document relation is absent from every raw-only training
  signal;
- compressed-weight decoding plus memory gather exceeds latency, energy,
  workspace, or issued-work limits; or
- a same-byte learned conditional memory or self-index obtains the same or
  better capability.

The task-agnostic usefulness theorem remains binding.  T35 is lossless on the
documents and makes no universal semantic-extraction guarantee.

## P9. Strongest controls

Every arm uses the same from-zero corpus and training-accounting rules.

1. Uncompressed dense frontier.
2. Losslessly compressed dense frontier using slack for batch/runtime only.
3. Same compressed backbone with the free bytes exposed as the strongest
   byte-matched dense or opaque learned state.
4. Engram-like conditional learned memory at equal total bytes and active work.
5. Direct `uint16` records with the strongest legal one-summary reader.
6. Compressed self-index with the same title candidates and evidence budget.
7. T35 with correct, zero, shuffled, wrong-document, position-permuted, and
   gate-disabled memory.
8. Explicit retrieved text as a separately priced information ceiling.

The lossless codec is granted to every legal control.  T35 must beat the best
complete control, not an uncompressed straw baseline.

## P10. Current prior-art boundary

- The 2026 lossless-weight work aligns ANS decode with GEMM tiles and reports
  near-entropy coding plus serving throughput gains.  It supplies a possible
  physical codec, not the reinvested raw-prose architecture.
- Engram establishes iso-parameter, iso-FLOP conditional N-gram memory and is a
  mandatory learned-memory control.
- INTRA pre-encodes evidence and uses decoder attention for retrieval; its
  reverse-key algebra absorbs T35's attention reassociation.
- TokenMem uses a dedicated cross-attention channel and a small trained gate to
  improve knowledge compliance in frozen LLMs.
- T32/T32R already establish near-entropy raw-ID storage, a large explicit-text
  information ceiling, and the failure of one rank-32 boundary reader.

T35 therefore claims no new primitive.  Its admissible novelty question is
whether a *closed, baseline-preserving resource loop* can turn encoding
redundancy and prompt redundancy into materially better internal knowledge at
identical complete serving cost.

## P11. Breakthrough-size path

The frozen best dense/reference accuracy is 56.7308%; explicit correct records
read by the earlier 9B oracle reached 89.4231%.  Reaching 80% requires recovering

\[
\frac{80-56.7308}{89.4231-56.7308}=0.7118
\]

or 71.18% of that observed gap before protected-quality penalties.  This is
only an information-ceiling ratio.  It is not an end-gain theorem.

For selected memory path `S`, path correctness `T`, and baseline correctness
`B`, the exact change is

\[
\Delta A=P(S\cap T\cap\neg B)-P(S\cap\neg T\cap B).
\]

Reaching 80% from 56.7308% requires this repair-minus-harm quantity to be at
least 0.232692, while beating the strongest control requires its paired
candidate-minus-control analogue to be at least 0.10.  Global route coverage
or a 9B explicit-text ceiling cannot substitute for either joint quantity.

The candidate is killed before end-to-end training unless:

1. the complete bounded raw candidate pool contains both supports on every
   admitted row without argument selection or truncation;
2. the handle-funded branch is available on enough rows to make 80% possible;
3. a same-size small model given explicit records clears the 80% information
   ceiling;
4. the exact `R`-position deployed bottleneck retains at least 71.18% of the
   dense-to-explicit gap under frozen correct/shuffled controls; and
5. the same-byte strongest conditional-memory control remains at least ten
   points behind the final candidate.

No sub-percent endpoint is a positive result.

## P12. Local-first admission ladder

No rented GPU is admitted by this paper.  If independent review does not find
a fatal algebraic or control error, proceed only in this order:

1. **CPU route/work census:** freeze `C,R` and the exact enumeration/abstention rule first; report all
   title-candidate counts, support coverage revealed only for scoring, title
   token savings, per-request work inequalities, and the maximum active-row
   ceiling.
2. **CPU entropy/codec census:** measure actual tile entropy and complete
   lossless encoded bytes on current from-zero checkpoints; round-trip every
   bit and include metadata/alignment.
3. **Local reduced operator:** on the local GPU only when free, verify
   ordinary-versus-reverse attention, numerical error, allocated bytes, and
   small-shape timing through the same codepath.  If the GPU is busy, wait.
4. **Local capability ceilings:** establish the same-size explicit-record
   ceiling and then the exact four-position bottleneck with correct, shuffled,
   zero, and wrong records.
5. **Local small from-zero A/B:** only after all preceding gates pass, run the
   smallest model that can falsify transfer and control absorption.
6. **Target GPU:** rent only if local results pass and the remaining question
   is production-shape physical realization.  Freeze maximum dollars, wall
   time, checkpoints, a short smoke block, and machine-checkable early stop.

## Independent audit result

The review found multiple blockers before measurement.

### A1. The stated positive construction needs a shifted value stream

With `Z` equal to context-free current-token embeddings, a head that matches a
`KEY_i` position returns the value projection of `KEY_i`, not the following
`VALUE_i`.  Additive position bias cannot dynamically move from whichever key
matched to its successor.  The construction would need, for example,

\[
K_i=E[t_i]W_K,\qquad V_i=E[t_{i+\delta}]W_V,
\]

or a second dependent hop.  A fixed shifted stream is raw-computable and can
be funded with a revised traffic ledger, but it is the already-known shifted
value-transport primitive.  It repairs the toy construction without solving
natural relation selection.

### A2. Strongest-control absorption is fatal

P9 incorrectly restricted the direct `uint16` control to a one-summary reader.
The strongest legal direct-plane or general conditional-memory control receives
the same:

- lossless weight codec and recovered bytes;
- complete all-title compiler and typed occurrence handles;
- raw-ID records and shared embeddings;
- `R` existing prompt destinations;
- reverse-projected attention, shifted values if added, adapters, and gates;
- prompt-token compute savings and complete serving ledger.

It therefore reproduces the complete T35 function, acquisition path, and
resource point.  T35's required ten-point advantage over that control is
mathematically impossible: the control can be parameterized as T35 itself.
Lossless entropy reinvestment and handle-funded in-place context are useful
implementation rules, but neither is a capability-bearing object outside the
strongest raw-memory control.

### A3. The learned path is not one isolated uncertainty

The phrase `raw-derived objectives` supplies no concrete observation model,
training target, identifiability condition, or local measurable quantity.
Handle/query adaptation, evidence scoring, role binding, and downstream use
are separate learned blocks.  The earlier 9B explicit-text result gives only a
global information ceiling; it does not supply the selected-branch
repair-minus-harm law above or propagate a margin against the best legal
control.  This independently forbids a corpus or model census.

### A4. The physical packet is incomplete

Even after correcting the adapter work to `(2H+6)Rda`, no finite ledger exists
for the `R*H` online-softmax accumulators, lifted queries, typed-handle branch,
codec decode scratch, allocator alignment, or peak transient state.  Promising
future accounting is not a Stage-5 resource proof.

The named 36.6M scratch backbone has six attention heads, while the illustrative
reader silently uses four.  A four-of-six subset needs exact projection/output
shapes and a residual join.  Using all six heads raises the base reader to
21,233,664 MACs before adapters, above the claimed one-token saving of
17,694,720 MACs.  The retained one-token funding statement is therefore not
valid for the named graph.

### A5. Lossless decode is not zero work

Even inactive prompts decode compressed weight tiles.  That adds integer,
addressing, and decode work relative to the uncompressed dense baseline, and
the operation contract counts those categories separately.  T35 cannot claim
componentwise no-work increase against that baseline.  Making the
losslessly-compressed dense model the resource comparator repairs the type
error, but then A2 grants the comparator the same recovered bytes and absorbs
the complete candidate.

## Paper decision

Close T35 before the CPU route/entropy census.  No local GPU check, small-model
run, or rental is admitted.  Retain only:

1. the general byte inequality for funding a raw plane from lossless weight
   entropy, with the 683,148-byte number retained only as a conservative
   inherited bundle rather than exact T35 state;
2. the symbolic per-request MAC inequality for funding a bounded read from
   internal title-token savings, not the invalid four-head/named-graph numeric
   claim; and
3. the requirement that any future raw-memory control receive both
   optimizations.

The next candidate must add a capability-bearing algebra that the optimized
direct-plane/general-conditional-memory control cannot instantiate at the same
bytes, work, state, and training information.  A better codec or another
ordinary attention schedule is not such an algebra.
