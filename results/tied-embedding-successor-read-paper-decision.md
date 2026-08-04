# Tied-embedding successor read — pre-run decision

Status: **PAPER FAIL; NO IMPLEMENTATION OR GPU RUN**  
Date: 2026-07-31

## Decision in plain language

Do not implement the projection-free tied-embedding successor reader.

The reader has a clean conditional correctness theorem, but its condition is
the missing capability: the question hidden state must already rank the raw
document anchor above every distractor.  Tying the token embedding to the LM
head does not create that ranking.  At an intermediate state it supplies a
logit-lens coordinate; at the final pre-unembedding state it supplies an
ordinary next-token logit.  Neither identity is a relevance guarantee.

The proposal also does not define a sufficiently separated operator family.
T27 already proves the successor-transport object with a different exact
two-token matcher.  With learned query and key transforms the proposal becomes
ordinary attention/induction.  With an external table it enters the
established explicit-memory family.  No strongest-control separation remains.

This is a wall for the **projection-free interface**, not for raw evidence.
T32 has already shown that the full raw record contains a breakthrough-sized
amount of causal information.  The implementation must be rethought around an
identifiable query interface rather than another storage codec.

## Proposed operator

Let the tied input/output embedding table be
`E in R^(V x d)`, and let a raw token tape be
`X=(x_1,...,x_L)`.  For a query hidden state `h`, the proposed read is

\[
\alpha_i(h)=
\frac{\exp(\beta h^\top E_{x_i})}
     {\sum_{j=1}^{L-1}\exp(\beta h^\top E_{x_j})},
\qquad
r(h,X)=\sum_{i=1}^{L-1}\alpha_i(h)E_{x_{i+1}}.
\]

The hoped-for edge was:

- raw token IDs are the persistent memory;
- the tied table supplies both keys and returned token features;
- successor shifting supplies the value relation;
- no new learned Q/K/V projection is used by the read.

## Positive theorem: what the operator really proves

Let `L >= 2`, `beta > 0`, and `Delta > 0`.  Assume every embedding row has norm at most `B`.
Suppose position `j in {1,...,L-1}` is the desired anchor and has query margin

\[
h^\top E_{x_j}\ge h^\top E_{x_i}+\Delta
\quad\text{for every }i\ne j.
\]

Let

\[
S=(L-2)e^{-\beta\Delta}.
\]

Then

\[
1-\alpha_j\le \frac{S}{1+S}\triangleq\delta
\]

and therefore

\[
\|r-E_{x_{j+1}}\|_2\le 2B\delta.
\]

If the desired successor `v=x_(j+1)` also has tied-output separation

\[
E_v^\top E_v-E_t^\top E_v\ge\gamma
\quad\text{for every }t\ne v,
\]

then `v` remains the output argmax whenever

\[
\gamma>4B^2\delta.
\]

### Proof

Every incorrect softmax numerator is at most `e^(-beta Delta)` times the
correct numerator.  Summing the `L-2` incorrect terms gives the mass bound.
The read is a convex combination, so replacing the correct value by values of
norm at most `B` moves it by at most `2B` times the incorrect mass.  Finally,
for any competing output row,

\[
|(E_v-E_t)^\top(r-E_v)|
\le \|E_v-E_t\|\,\|r-E_v\|
\le 4B^2\delta.
\]

Thus the original output margin survives under the stated inequality for the
standalone tied readout `E r`.  Residual addition, an output projection, final
normalization, or output bias needs a separate end-to-end margin/Lipschitz
bound.

This theorem is useful as a block invariant.  It is not a query theorem: it
starts by assuming the correct semantic address.

## No implication theorem: tying does not identify the address

At the final normalized state of a tied, bias-free LM head, the score used
above is exactly the token logit

\[
\ell_t(h)=h^\top E_t
\]

With an intermediate state, it is only the corresponding logit-lens score.
The next-token objective directly trains the final score to rank a likely
**continuation token**.  It can indirectly shape an intermediate TESR state,
but tying alone supplies no equation requiring either state to rank a token
whose occurrence in a separate declarative record is relevant to the current
question.

A minimal counterexample to the claimed implication uses vocabulary
`{a,b,d,e}` with four orthonormal unit rows.  Let the raw record be `a b d e`, with the desired read
`a -> b`, and let the correct next token after the question also be `b`.  The
perfectly predictive final state `h=E_b` gives

\[
h^\top E_b=1,\qquad h^\top E_a=h^\top E_d=h^\top E_e=0.
\]

It is optimal for next-token prediction, but at finite positive `beta`, TESR
puts its largest mass on eligible anchor `b`, making successor `d`—not `b`—the
unique standalone tied-readout argmax.  Tying is fully satisfied.  The address
guarantee therefore does not follow from tying even in the noiseless
orthogonal case.

This does not prove that an end-to-end trained TESR can never learn a useful
intermediate address state.  It proves that such learning is the unresolved
mechanism rather than a consequence of weight tying.  Two concrete ways to
create the missing alignment are:

1. Use the hidden state at a literal occurrence of `a`.  This recovers exact
   token matching, not unseen paraphrases, predicate roles, aliases, or
   question-to-declaration alignment.
2. Learn a map `q=W_Q h` and compatible document keys `k_i=W_K z_i`.  This
   restores the query/key transform that the proposal intended to remove.

Other learned intermediate constructions are possible, but then the paper
must identify their training signal, strongest control, and complete cost; the
tied table itself is not that construction.

Recent mechanistic evidence points in the same direction: tied embeddings are
shaped more strongly by the output-prediction role than by the input-role
geometry, so shared rows should not be treated as a free semantic index
([Lopardo et al., 2026](https://arxiv.org/abs/2603.26663)).

## Strongest-control absorption

### Existing project controls

- T27 already proved exact unique two-token matching, successor selection, and
  tied-output token decoding inside ordinary Transformer widths.
- T32R already separated literal raw-title candidate generation from semantic
  role typing and found the literal router correct on only 102/104 questions.
- AFTA already treated shifted-value/successor transport as an established
  primitive rather than a novel semantic reader.

TESR changes the numeric similarity function used by T27, but does not repair
its natural query-to-key boundary.

### Ordinary attention control

An induction circuit performs the same `[A][B] ... [A] -> [B]` behavior by
using a previous-token transport, a learned QK match, and a copying OV circuit.
This behavior and its two-head construction are established in
[Olsson et al.](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/).
With a previous-token head or an externally shifted value tape, identity-like
token maps reduce that circuit to TESR's successor behavior.  A single ordinary
causal head with identity maps is not sufficient by itself.  TESR is
nonetheless contained in the broader attention-circuit family, not a strictly
larger function class.

### Conditional-memory control

Engram already separates static lookup from dynamic neural computation using
deterministic n-gram memory, learned projections, and context gating, and its
iso-parameter/iso-FLOPs experiments reallocate sparse expert capacity to
memory ([Cheng et al., 2026](https://arxiv.org/abs/2601.07372)).  Memorizing
Transformers and Memory3 already store or reuse attention key-values as
explicit memory.  RetrievalAttention already searches CPU-resident KVs using
the model's attention queries.  The newer INTRA direction explicitly makes
decoder attention queries retrieve pre-encoded evidence.

Consequently, restoring the missing learned transform yields a valid research
family, but not a new candidate without an additional non-absorbed object or
resource law.

## Resource boundary

For a routed record of length `L`, TESR still performs `Theta(Ld)` score work
and reads `Theta(L)` token IDs plus the referenced embedding rows.  Across an
unrouted corpus of `N` records it becomes `Theta(NLd)` or needs an external
index.  The index construction/search state and traffic are part of serving
cost.

Removing per-record Q/K/V projections is a real local arithmetic saving, but
it is not yet an end-to-end Pareto result.  A complete ledger must count raw-ID
and sentence-index bytes, embedding-row traffic, score and weighted-sum passes
(or their fused equivalent), online-softmax state, residual injection, and
every routing/index operation.  The matched control is raw-tape attention with
the same route and memory.  Granting free document routing would repeat the
supplied-handle error already excluded by the research protocol.

## Breakthrough-size boundary

The current natural ledger is:

- best dense/reference result: `56.7308%`;
- full raw-record information ceiling: `89.4231%`;
- required target: at least `80%`;
- required fraction of the observed oracle gap: `71.18%`.

TESR proves no lower bound on natural address accuracy.  Its optimistic end
gain is therefore unquantified, while the required composition would need high
accuracy on routing, per-record anchor selection, returned evidence, and
two-record reasoning simultaneously.  A block with no query-accuracy bound
cannot support the breakthrough arithmetic.

## What is retained

Retain three facts, without retaining the candidate:

1. The softmax margin and output-separation bounds above are valid block
   checks for any successor reader.
2. Raw token IDs plus sentence/order indices remain the strongest compact
   writer representation currently established in this project.
3. Causal prefix K/V reuse gives a computational identity under strict
   conditions.  Complete per-layer states reproduce `P || Q` only when they
   were computed from the identical prefix `P` with the same order,
   tokenization, position IDs/encoding, masks, model/adapters, and numerical
   semantics.  A document encoded alone generally cannot be spliced after a
   different prefix or reordered.  The identity removes recomputation; it
   does not solve relevance, retrieval, or compression.

The third fact is an implementation rethink, but existing explicit-memory and
retrieval-attention work already covers its broad form.  The next research
packet must isolate something stronger than “precompute K/V and retrieve it.”

## Next search object, not yet a candidate

The useful mathematical target is an **attention-behavior codec**:

```text
same-model causal document pass
    -> compact, inspectable state C(document)

existing model query + C(document)
    -> epsilon-accurate outputs of selected memory heads
```

It must satisfy all of the following before code:

1. derive an error bound over the model's declared reachable query set, not
   over a supplied correct handle;
2. beat raw-token plus sentence-index, full/sparse KV, product-quantized KV,
   linear-attention moment, and learned-memory controls at the same complete
   storage/search/read ledger;
3. use no stronger writer than the recipient model;
4. expose a minimal witness where the new code is smaller than every matched
   control while preserving a query distinction that they lose;
5. propagate its error and coverage to an `>=80%` end-capability ceiling;
6. name the different paid currency if GPU compute, VRAM, latency, model bytes,
   and prompt tokens remain fixed.

Until such a separating witness exists, a K/V codec sweep would be ordinary
compression research and no H100 run is admitted.
