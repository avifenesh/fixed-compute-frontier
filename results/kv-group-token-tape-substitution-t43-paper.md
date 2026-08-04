# T43 KV-group token-tape substitution — paper packet

Date: 2026-08-01  
Status: **CLOSED BY INDEPENDENT PAPER AUDIT; NO CPU, MODEL, LOCAL-GPU, OR RENTAL RUN ADMITTED**

## Independent audit verdict

T43 is a no-go as a capability candidate.  Its logical K/V-group credit and
token-tape arithmetic are mostly correct, but the proposed reader fails at the
answer-bearing edge:

1. `k_i` and `v_i` are slices of the same token.  Matching a relation/key token
   therefore returns that token, not its unknown successor, object, answer, or
   neighboring evidence.  A shifted/contextual value stream or dependent
   second hop is required.  That is a different operator with additional
   traffic and is already inside the T35/T40 control boundary.
2. At `d=384,h=64`, one group supplies `98,304` bytes, while the established
   `2,405 x 48` token population alone needs `230,880` bytes before metadata.
   At least three such group credits are required, and every lost history group
   must be charged as a separate capability removal.
3. A fixed `h`-coordinate value slice is a rank-`h` transport bottleneck.  The
   retained output map cannot reconstruct token information discarded by that
   slice.
4. Exact record identity does not identify the evidence position.  T43-B also
   adds the separate, unproved requirement that raw-prefix and later-question
   surfaces emit the same address.
5. The identical tape-and-reader control ties exactly.  The residual claim is
   a resource/acquisition hypothesis, not a new function class, and no
   repair-minus-lost-context bridge was supplied.

No shifted-stream repair, local census, or kernel test follows this verdict.
The historical proposal remains below so the valid arithmetic and the precise
failure cannot be renamed later.

## End claim and reason for this object

The target is a from-zero language model that converts raw prose into an
internal digital plane and gains materially more held-out knowledge and
reasoning at no greater complete serving cost.

T32 established that exact raw records can contain a breakthrough-sized
amount of causal information, but T32R/T40 did not provide a funded reader
whose placement and resource exchange were complete.  T43 changes the object
that pays for the reader:

> Replace one complete self-attention K/V group by a persistent exact token
> tape.  Retain that group's ordinary query and output paths, but let its
> queries cross-attend to a bounded routed record whose keys and values are
> fixed slices of the already resident token embedding.

This removes two dense projections and one group's request KV cache.  It is
not another side branch added to a full Transformer.

T43 has two deliberately separate addressing stages:

1. **T43-A, exact raw structure:** a raw title or another declared exact span
   identifies a record.  This isolates token-tape acquisition and reading.
2. **T43-B, learned latent structure:** existing query coordinates emit a
   discrete address learned from raw next-token training.  This is the end-goal
   interface, but it is not admitted until T43-A closes the reader block.

The stages may not be bundled into one end-to-end success scalar.

## 1. Replaced GQA group

Let model width be `d`, query-head width `h`, and grouped-query ratio
`g=H_q/H_kv`.  One ordinary GQA group contains:

- `g` query heads: `W_Q in R^(d x gh)`;
- one key head: `W_K in R^(d x h)`;
- one value head: `W_V in R^(d x h)`; and
- the corresponding `gh` input columns of `W_O in R^(gh x d)`.

T43 retains `W_Q` and the corresponding `W_O` block.  It removes `W_K` and
`W_V` only.  At weight precision `b_w` bits, the exact persistent-byte credit is

\[
B_{credit}=2dh\,b_w/8.
\]

The removed projection work is exactly `2dh` scalar MACs per processed token.
The removed mutable state is `2h b_k/8` bytes for every cached token, where
`b_k` is the K/V precision.

The candidate no longer performs this group's ordinary attention over the
request history.  That lost context path is a real capability cost and is a
mandatory control, not hidden slack.

## 2. Exact raw token tape

For fixed record width `W` and vocabulary `V<=65,536`, store

```text
record_ids[A, W] : uint16
record_valid[A]  : packed bit
record_length[A] : optional uint8 or uint16
```

Padding uses a declared reserved token and a validity/length field; no raw
token is confused with absence.  In the simplest dense-address realization,
the address itself is the row number and no key/fingerprint table exists.

The payload cost is

\[
B_{tape}=2AW
\]

bytes before validity, lengths, alignment, writer receipts, or any exact-title
index.  Every one of those additions is charged against `B_credit`.

### Finite target cell

At `d=4096`, `h=128`, BF16 weights, and `W=48`:

\[
B_{credit}=4dh=2,097,152\text{ bytes}.
\]

Choosing `A=2^14=16,384` gives

\[
B_{tape}=16,384\cdot48\cdot2=1,572,864\text{ bytes},
\]

leaving `524,288` bytes for validity, lengths, alignment, the exact raw index,
and any fixed code metadata.  This is a prospective upper envelope, not a
complete index ledger.

At the existing width-384 research geometry with `h=64`, the same exchange
provides `98,304` bytes, exactly enough for `1,024` fixed-width 48-token
payloads before metadata.  Therefore a real small-model implementation must
either reduce `A`, reduce `W`, or fund metadata from an explicitly removed
additional component.

## 3. Projection-free record attention

Let `E in R^(V x d)` be the model's shared token embedding.  Freeze two
coordinate selectors `S_K,S_V` that each choose `h` coordinates; they contain
no learned scalars.  For routed record `r=(x_1,...,x_m)`, `m<=W`, define

\[
k_i=S_K\,\bar E[x_i],\qquad
v_i=S_V\,\bar E[x_i],
\]

where `bar E[x]` is either the raw embedding row or a precisely specified
row-normalized embedding.  Row normalization is not free: using it requires a
charged vocabulary-scale table or the full reduction work and traffic at each
read.  The first executable packet must select one.

For retained query head `q_a in R^h`, head `a=1,...,g`, compute

\[
\alpha_{a,i}=\operatorname{softmax}_i
\left(q_a^T R_i k_i/\sqrt h\right),
\qquad
o_a=\sum_i\alpha_{a,i}v_i.
\]

`R_i` is the already specified positional operation for record position `i`
(for example, the model's existing RoPE rotation).  Concatenate the `g`
outputs and apply the retained `W_O` columns exactly as the ordinary group.

The read is placed where the replaced group's self-attention output would have
been placed.  It therefore has an unambiguous residual and cache schedule; it
does not need a pre-block side update followed by a second Q/K/V evaluation.

## 4. Positive lemmas

### L1. Exact typed persistence

For every valid cell and position, gathering `record_ids[c,i]` selects exactly
the same embedding row as the corresponding vocabulary token:

\[
E[record\_ids[c,i]]=E[x_i].
\]

**Plain explanation.** The digital plane stores token identity, not a learned
summary of it.  The shared embedding performs the only semantic expansion.

**Small witness.** With vocabulary rows `{red, blue, green}` and record IDs
`[2,0]`, the two gathered rows are exactly `[E[green],E[red]]`.

**Adversary.** If vocabulary size exceeds the integer domain or padding aliases
a real token without a length/valid bit, exactness fails.

### L2. Exact attention equivalence on the declared group

Consider an ordinary GQA group whose K/V inputs are the sequence
`(bar E[x_1],...,bar E[x_m])` and whose K/V maps are the fixed selectors
`S_K,S_V`.  For the same queries, position operation, softmax schedule, and
`W_O`, T43 computes exactly the same real-arithmetic output.

**Plain explanation.** T43 changes where the record sequence lives.  It does
not approximate the declared attention calculation.

**Adversary.** An unrestricted learned dense K/V projection is a strictly
larger map family than a fixed coordinate selector.  T43 does not reproduce an
arbitrary ordinary attention group.

### L3. Concentrated-token recovery

If one desired record position has score at least `Delta` above every other
valid position, then for `m<=W`

\[
\alpha_*\ge {1\over1+(m-1)e^{-\Delta}}.
\]

If every value norm is at most `M`,

\[
\|o_a-v_*\|\le2M(m-1)e^{-\Delta}.
\]

The lemma bounds a read after a margin exists.  It does not derive a natural
question-to-position margin from raw prose.

### L4. Arbitrary-record information boundary

For iid uniform record tokens `X_1,...,X_W in [V]`, any state `S` from which a
decoder recovers a uniformly requested coordinate with average error at most
`epsilon` needs

\[
B\ge W\left[\log_2V-h_2(\epsilon)
-\epsilon\log_2(V-1)\right]
\]

bits.  A `uint16` tape is within the fixed-length integer overhead of the
zero-error bound when `V<=65,536`.  A learned `h`-dimensional vector can beat
the tape only by exploiting a non-arbitrary natural-record distribution or by
accepting error; it cannot solve the arbitrary coordinate witness with fewer
than the bound's bits.

This separates exact token records from small continuous summaries on the
declared multi-query witness.  It does not prove that natural QA needs all
record entropy.

### L5. Dense-address collision ceiling

Let a raw-only writer assign every observed continuation record `R` to address
`C`.  If one table record is retained per address, the maximum possible exact
record recall of any writer using only `C` is

\[
P_1=\sum_c P(C=c)\max_r P(R=r\mid C=c).
\]

With at most `J` variants per address it is

\[
P_J=\sum_c P(C=c)\sum_{r\in TopJ(c)}P(R=r\mid C=c).
\]

**Proof.** Conditional on `C=c`, the optimal one-record decision chooses the
modal record; the optimal `J`-record set chooses the `J` largest conditional
masses.  Average over `c`.

This is a fatal, raw-computable ceiling for T43-B.  Nominally having `2^b`
addresses does not establish a semantic memory.  If the query and writer
surfaces do not share an address, or the address has low conditional purity,
no reader repairs the absent record.

## 5. Active work and mutable-state exchange

For one routed record of length `m<=W`, the QK and AV work of all `g` heads is

\[
C_{record-attn}=2gmh
\]

MACs, plus softmax, position operations, gathers, validity handling, and the
unchanged output projection.  The removed ordinary group would have used

\[
C_{history-attn}=2gTh
\]

MACs at cached history length `T`, plus the removed `2dh` K/V projection MACs.
Thus matrix/attention MACs are strictly lower whenever

\[
gmh < dh + gTh,
\]

which holds trivially at the target cell for bounded `m`; this is not a latency
proof.  The candidate introduces random record-ID reads, embedding gathers,
validity branches, and a different cache/critical-path schedule.

At `d=4096,h=128,g=4,W=48`, the worst record attention is `49,152` MACs while
the removed K/V projections alone are `1,048,576` MACs per token.  The
candidate also removes `512` BF16 K/V bytes per cached token for that group.

The physical packet must separately price:

- address generation and lookup;
- token-ID and metadata traffic;
- `m*h` embedding-slice traffic and cache behavior;
- normalization, RoPE, masking, softmax, and reductions;
- divergence when batch items route to different records;
- the loss of contiguous FlashAttention over the replaced history group;
- workspace, registers, occupancy, p50/p95/p99, throughput, and energy.

## 6. Raw writer and the compiler-strength objection

T43 does not permit an offline model that semantically rewrites prose.

### T43-A writer

The writer may only tokenize a declared raw record, verify exact title/span
identity, and copy up to `W` token IDs into the assigned cell.  This is a typed
data movement algorithm, not a semantic teacher.  It is independently
checkable and deleted after export.

### T43-B writer

The only proposed learned address is derived from coordinates of the retained
query projection already computed by the serving model.  During an offline
table-build pass, the same model maps raw prefixes to addresses and copies raw
continuations.  No larger model, question labels, answer labels, support
documents, or external embeddings are allowed.

This answers only the narrow compiler paradox: the writer need not be a
stronger reasoner because it does not generate a semantic value.  It records
evidence visible in the training sequence.  Whether raw prose and later
questions acquire the same address is still the central unresolved semantic
problem, governed first by `P_J` and then by held-out route recall.

## 7. Strongest controls

Every candidate comparison receives the same raw corpus and training compute.
At equal complete serving cost it must include:

1. the unchanged dense/MoE model with the ordinary GQA group;
2. the group removed, with its K/V bytes reinvested in the strongest ordinary
   dense or routed parameters permitted by the active-MAC ceiling;
3. the same group substitution with continuous learned vector records;
4. Engram and Lngram-style conditional memories funded from the same resident
   and active budgets;
5. the same token tape and exact route but zero, shuffled, wrong-record, and
   reader-disabled arms;
6. the same token tape with a learned low-rank K/V map funded from fewer or
   shorter records;
7. explicit prompt/RAG evidence with prompt tokens, KV, retrieval, CPU/network,
   and latency priced rather than hidden; and
8. an identical tape-and-reader control, which must tie T43 and prevents a
   claim that renaming conditional memory creates a new function class.

The possible claim is a strict storage/acquisition point relative to continuous
memory and an end-model Pareto gain relative to the frontier baseline.  It is
not superiority over an identical conditional-memory machine.

## 8. Current proof-ladder audit

| lemma | status | evidence/boundary |
|---|---|---|
| L1 positive construction | fail for a useful record reader | only same-token ID storage/gather passes; key-to-successor transport is absent |
| L2 information/approximation | hold | Fano and conditional concentration are valid; slice sufficiency and a useful margin are not |
| L3 non-circular identifiability | fail | T43-A identifies only a record; within-record evidence and T43-B cross-surface address are unresolved |
| L4 strongest-control separation | fail | arbitrary-record storage is not target-risk separation and the identical reader ties |
| L5 complete resource exchange | hold/fail | logical tensor counts pass; physical layout, traffic, metadata, and loss of context capacity do not |
| L6 end-effect bridge | fail | T32's 9B explicit-text ceiling does not determine this group's repair-minus-harm effect |

T43 therefore does **not** authorize even a small model run.  It identifies a
plausible logical resource source and integration site, but it does not
materially repair T40's answer-bearing semantic edge or end-effect theorem.

## 9. Next admissible paper work

Before code, one successor packet must do both:

1. freeze a natural, raw-only T43-A reader witness whose evidence-token
   recovery maps to the end metric without bundling routing, reading,
   reasoning, and decoding; and
2. complete the exact small-model and target-shape index/metadata/traffic
   ledgers, including the strongest vector-memory and reinvested-group controls.

Only if that packet leaves one binary reader-acquisition fact unresolved may
an exhaustive CPU reference be admitted.  T43-B latent addressing remains a
later stage and must first pass the raw-computable `P_J` collision ceiling plus
held-out route-recall gate.

No local GPU is needed now.  No rented instance is justified.

## 9a. Additional physical corrections from audit

The exact logical counts above must not be quoted as a realized resource win:

- the replaced layer count must be explicit, and every additional layer gets
  its own byte credit and lost context path;
- removing one logical KV head may not shrink padded tensor shapes, packed
  kernels, allocator state, or cache stride on a real runtime;
- disjoint key/value selectors logically load up to `2mh` elements, while
  row-major gathers may touch substantially more physical bytes;
- pre-extracted vocabulary K/V slices would cost
  `V*|S_K union S_V|` static elements, while request caching would cost
  `2mh*b/8` mutable bytes;
- query-position and record-position RoPE must use the exact relative rotation
  `R_t^T R_i`, not an unspecified `R_i`; and
- Cross-Layer Attention, TIDE, Bank of Values, Memorizing Transformers, and
  current head-aware KV systems belong in the strongest control set.

## 9b. Retained scale law only

For a fixed record width `W`, the **logical** record capacity funded by one
removed BF16 K/V group is `Theta(dh/W)`, while one record read is
`Theta(gWh)` arithmetic.  If `h` grows proportionally with `d`, this exchanges
quadratic resident projection bytes for linear active record-read work.  That
is the useful retained systems law.  It carries no capability claim: the
reader must still produce an answer-bearing value and beat the lost
self-attention group plus every matched memory control.

## 10. Prior boundary checked on 2026-08-01

- [Memory3](https://arxiv.org/abs/2407.01178) already establishes explicit
  memory as a cheaper knowledge substrate than ordinary parameters or RAG.
- [Engram](https://arxiv.org/abs/2601.07372) establishes iso-budget token-based
  conditional memory.
- [Lngram](https://arxiv.org/abs/2605.24869) learns discrete hidden-state
  addresses and exact latent n-gram lookup; its 2B experiment reallocates MoE
  experts and reports a `+1.41` point task-balanced average over its MoE
  baseline.
- [Memory Grafting](https://arxiv.org/abs/2605.20948) uses a stronger offline
  pretrained model to construct latent values; T43 explicitly forbids that
  source of writer capability.

T43 does not claim explicit memory, discrete latent addressing, or token
lookup as new components.  The provisional object is their exact
`K/V-group -> uint16 token tape` resource substitution with shared-embedding
identity K/V and no stronger offline semantic model.  Novelty remains
unestablished and is irrelevant unless the proof ladder first admits a useful
model result.
