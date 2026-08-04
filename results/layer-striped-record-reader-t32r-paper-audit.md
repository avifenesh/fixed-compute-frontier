# T32R exact-record reader — algebra and collision audit

Status: **PAPER PASS TO RAW HANDLE/COMPUTE CENSUS ONLY**  
Date: 2026-07-31

## Verdict

The T32 information pass removes payload sufficiency as the active uncertainty,
but it does not make a standard Transformer an exact reader. Two negative
facts force an implementation change:

1. a tokenizer ID is an arbitrary symbol, so its three radix cells do not
   contain semantic geometry; and
2. one ordinary attention value from one record token is a fixed linear
   projection, so it cannot expose two complete records to a query-dependent
   reader in one width-384 state.

The broad alternatives—per-layer token embeddings, context-free value banks,
conditional lookup memory, hashed FFN replacement, and document parameters—
are already occupied prior art. T32R therefore claims no novelty for lookup or
per-layer injection. Its narrower candidate is an exact in-parameter document
tape whose IDs address the model's own token embeddings, followed by a small
query-conditioned scan paid for by reducing dense FFN width.

Only a CPU census of routing, handle compression, and the complete static
resource/compute inequalities is admitted by this paper. No model training or
GPU reader run is admitted yet.

## R0: what is now established

For 104 frozen title-disjoint comparison questions, a fixed 9B reader reached
89.4231% from the two decoded 128-token records and 88.4615% from two selected
48-token windows, versus 46.1538% question-only and 48.0769% shuffled evidence.
The correct payload therefore carries more than 40 points of causal usable
information.

This is an information ceiling over explicit token sequences. It gives no
free license to replace those sequences by one continuous vector.

## R1: token semantics lower bound

Let the vocabulary have `V` symbols and let `S: {0,...,V-1} -> R^r` be their
semantic vectors. T32 stores an injective digital code `C(t)` for each symbol
ID `t`. For an arbitrary assignment `S`, any exact reader of `S(t)` from
`C(t)` must represent the `V*r` scalar values of `S` somewhere; the ID digits
alone impose no relation among them.

Equivalently, permuting the semantic assignment among token IDs preserves the
record codec and all integer arithmetic while changing the required language
behavior. A decoder whose learned state is independent of that permutation
cannot be correct for both assignments.

Therefore T32 cannot turn IDs into meaning through radix algebra alone. It
must either:

- look up the already-resident token embedding `E[t]`;
- store an additional semantic table; or
- assume and validate a lower-dimensional structure in `E`.

T32R chooses the exact existing-embedding lookup. This is sparse memory access,
not a claim that semantics emerged from the integer code.

## R2: one-record-token attention bottleneck

For one head at one layer, an ordinary record position contributes

\[
\alpha(q,r)W_Vr
\]

to a query position. The query changes only the scalar attention coefficient;
the value projection `W_V r` is fixed before seeing that query. Consequently a
single head cannot perform query-dependent coordinate selection inside the
record before transfer.

More generally, simultaneously transferring two arbitrary width-384 records
through one width-384 query state by a linear value path is non-injective: the
map has domain dimension 768 and rank at most 384. A later FFN cannot recover
information already lost by that map.

Layer striping can repair transport for bounded token reads, but not token
semantics: different layers can copy different record coordinates, yet the
reader still needs the `ID -> embedding` operation proved necessary in R1.

## R3: collision boundary

The following families are not novelty claims:

- [Gemma 3n Per-Layer Embeddings](https://ai.google.dev/gemma/docs/gemma-3n)
  inject token-specific parameters at successive layers and cache/offload them;
- [TIDE](https://arxiv.org/abs/2605.06216) maps token IDs to context-free
  semantic vectors and injects them at every layer;
- [Bank of Values](https://openreview.net/forum?id=YoQ0VK3JnP) learns
  token-specific context-free attention values in deeper layers;
- [Engram](https://arxiv.org/abs/2601.07372) provides deterministic O(1)
  conditional memory and already beats an iso-parameter, iso-FLOP MoE
  baseline;
- [MemoryFormer](https://openreview.net/forum?id=04EC4ZnZJj) replaces dense
  feature transforms by hashed lookup vectors;
- [Parametric RAG](https://arxiv.org/abs/2501.15915) compiles documents into
  parameters injected into FFNs.

Any eventual T32R claim must beat an Engram-like learned lookup and a free
learned document-vector control at the same total parameters and active FLOPs.
Beating only a dense Transformer would not establish the new conjunction.

## R4: typed refined path

T32R is invoked once at the assistant-generation boundary of a fully observed
prompt:

```text
route(raw prompt titles)          -> two document handle indices
unpack(record_table[handle])      -> two exact 128-token ID sequences
gather(shared_input_embedding,ID) -> two 128 x 384 semantic sequences
project(shared P)                 -> two 128 x r sequences
scan(query code, sequences)       -> two bounded r-state summaries
inject(summaries, boundary state) -> ordinary width-384 decoder state
```

`route` is part of the tokenizer/model input contract, not an uncharged search
service. The record table is resident checkpoint state. `gather` uses the
model's existing token embedding table. `scan` is a small conditional encoder
that runs during prefill only; decode reuses the resulting ordinary KV state.

The path is causal with respect to output generation: it sees the complete
user prompt but no future assistant token. It is not a causal next-token layer
for arbitrary positions, and training must preserve that boundary.

## R5: exactness theorem up to the learned bottleneck

Let `P,U` be the passed T32 pack/unpack maps and `E` the model's input embedding
table. For every stored record `t`,

\[
E[U(P(t))_i]=E[t_i]
\]

at every retained position `i`. Thus the sparse expansion recovers exactly the
same semantic input vectors the model would receive if the 128 IDs had appeared
in context. No learned document compiler intervenes before the projection `P`.

All information loss begins at the declared rank-`r` projection and bounded
scan. That is the sole composition hypothesis; it must be tested directly.

## R6: parameter-neutral construction

Use the frozen small-model shape:

```text
layers L = 10
hidden d = 384
baseline SwiGLU width m = 1024
records N = 2405
record digit cells = 381
scan rank r = 32
```

Reduce every SwiGLU from width 1,024 to 940. The removed parameters are

\[
10(1024-940)(3\cdot384)=967,680.
\]

Allocate at most:

| component | entries |
|---|---:|
| exact record digits, `2405*381` | 916,305 |
| shared token/query projection, `384*32` | 12,288 |
| scan query/key/value maps, `3*32^2` | 3,072 |
| bounded local scan block, at most `3*32^2` | 3,072 |
| summary output, `32*384` | 12,288 |
| total | **947,025** |
| removed dense entries | **967,680** |
| unallocated matched slack | **20,655** |

The checkpoint is therefore no larger. A matched dense control receives the
same width 940 plus 947,025 free learned entries exposed through the strongest
non-record interface allowed by the preregistration.

This ledger is prospective. The scan equations and exact use of the 20,655
slack must be frozen before implementation.

## R7: active multiplication break-even

The document projection dominates the side path:

\[
C_{doc}=2(128)(384)(32)=3,145,728
\]

multiplies per request. Query projection costs `12,288*q` for `q` prompt
tokens. Two token-sequence key/value transforms cost at most
`2*128*2*32^2 = 524,288`, scan attention/recurrent arithmetic is lower order,
and summary output costs 24,576. A conservative bound before kernel overhead is

\[
C_{scan}(q)\le3,700,000+12,288q.
\]

Shrinking all ten FFNs saves

\[
C_{save}(q)=q\cdot967,680
\]

multiplies during prompt processing and 967,680 multiplies for every generated
token. Hence `C_save(q) >= C_scan(q)` for every prompt with at least four tokens
under the conservative bound. The natural prompts are expected to exceed this,
but the CPU census must verify it without labels.

This is only arithmetic. Small gathers and scans may be latency-bound even with
fewer multiplies, so target-H100 p50/p95 remains mandatory.

## R8: why handle compression matters but is not credited twice

Replacing a multi-token exact title by one input-only document handle reduces
ordinary Transformer prompt work and KV entries. The CPU census must report the
actual token saving on all routed questions. Those savings are a real serving
effect, but the parameter/FLOP gate above must pass even if no title-token
saving is credited. This prevents tokenizer compression from hiding an
expensive scanner.

Handles live in an input-only table and are not output vocabulary rows. Their
record digits are charged in R6. No external retrieval index, document text,
or extra prompt tokens may be present at serving.

## R9: admitted CPU microbench

Freeze and run one raw-only census that reports:

1. exact two-title routing on all evaluator questions without reading answers
   or support fields;
2. base-tokenizer title lengths, per-question title tokens removed, and handle
   token counts;
3. prompt lengths and the minimum `C_save-C_scan` arithmetic margin;
4. record-table bytes, exact ID expansion, embedding gather shapes, and all
   allocation totals;
5. whether any document record refers to an input-only handle namespace;
6. source/data/tokenizer hashes, CPU time, and zero GPU operations.

One collision, one nonpositive compute margin, one route failure, or one ledger
overflow closes this exact refinement before a physical reader.

## R10: later matched gates

If the CPU census passes, the next paper must freeze these physical arms:

- width-1,024 dense baseline;
- width-940 dense plus equal free learned memory;
- width-940 Engram-like conditional lookup;
- width-940 learned document-summary lookup;
- width-940 T32R exact record expansion and scan.

All receive identical raw tokens, training work, and query supervision. T32R
must show at least a ten-point title-disjoint natural gain over the best matched
arm, protected NLL within 0.5%, no increase in model bytes/KV/request tokens,
and noninferior H100 p50/p95 at the frozen batch/concurrency points. A smaller
gain is useful engineering, not the requested breakthrough.

## Paper decision

The full-record information result survives, but the naive ordinary-Transformer
reader does not. The exact next step is a CPU handle/compute census for the
parameter-neutral sparse-expansion path. The H100 remains idle until that census
passes and the bounded scan itself is fully specified.
