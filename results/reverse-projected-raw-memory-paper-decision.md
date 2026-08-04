# Reverse-projected raw memory — pre-run decision

Status: **INDEPENDENTLY AUDITED PAPER FAIL AS A BREAKTHROUGH CANDIDATE; RETAIN AS AN EXACT IMPLEMENTATION PRIMITIVE**  
Date: 2026-07-31

## Decision in plain language

Do not implement or time this proposal as a new smarter-model candidate.

Moving the key projection to the query side and the value projection after
attention is an exact reassociation of ordinary linear cross-attention.  It can
make an on-demand raw-token memory read much cheaper and can avoid a large
persistent K/V table.  It does not change the function that the corresponding
ordinary cross-attention layer computes, solve autonomous routing, or prove
that one boundary summary can transport the semantic information measured by
the T32 full-record oracle.

The useful result is therefore narrower:

> reverse projection is a potentially valuable physical implementation of a
> raw-memory reader, not the source of a new capability.

This distinction matters because the project requires a smarter production
model at fixed serving cost, not a faster implementation of an unproved
reader.  No CPU microbenchmark or H100 run is admitted by this paper.

## Exact operator

Let

- `Z in R^(L x d)` be a routed static memory sequence, such as gathered token
  embeddings plus any position feature already expressed in the same `d`
  dimensional input space;
- `h in R^d` be one query state;
- `H` be the number of query heads;
- `G` be the number of GQA key/value groups;
- `r` be the head width, with `Hr=d`;
- `g(j)` identify the KV group used by query head `j`.

For head `j`, ordinary cross-attention computes

\[
q_j=hW_Q^{(j)},\qquad
K_g=ZW_K^{(g)},\qquad
V_g=ZW_V^{(g)},
\]

\[
\alpha_j=\operatorname{softmax}\!\left(
q_jK_{g(j)}^T/\sqrt r+b_j
\right),\qquad
o_j=\alpha_jV_{g(j)}.
\]

Here `W_Q^(j),W_K^(g),W_V^(g) in R^(d x r)`, and `b_j` may contain an
additive mask or relative bias.

Associativity gives

\[
\tilde q_j=q_j(W_K^{(g(j))})^T\in\mathbb R^d,
\]

\[
\boxed{
\alpha_j=\operatorname{softmax}\!\left(
\tilde q_j Z^T/\sqrt r+b_j
\right),\qquad
o_j=(\alpha_j Z)W_V^{(g(j))}.
}
\]

The concatenated heads then use the unchanged output projection `W_O`.

### Equality proof

For every head and position,

\[
q_j(ZW_K^{(g)})^T
=q_j(W_K^{(g)})^TZ^T.
\]

Therefore the logits and softmax weights are identical.  Linearity then gives

\[
\alpha_j(ZW_V^{(g)})=(\alpha_jZ)W_V^{(g)}.
\]

The head outputs and final projected output are consequently identical in exact
arithmetic.  This is equality of functions, not an approximation theorem.

The result also survives a constant value-projection bias in serving/evaluation
mode by explicitly adding that bias after the reverse output, because
`sum_i alpha_i=1`.  It does not move a token-dependent nonlinear value map,
position-dependent value matrix, content gate, or attention-dropout realization
whose surviving weights no longer sum to one through the weighted sum.

### Positional and normalization boundary

The reassociation is exact when any static normalization or absolute position
feature is already part of `Z`, and additive attention biases stay in the
logits.  Query-side normalization is also compatible when it is already part
of `q`.  A nonlinear normalization applied **after** the memory key projection,
such as per-token RMSNorm/LayerNorm of `ZW_K`, cannot generally be moved into
one common lifted query.

A key-side position-dependent linear operator such as full RoPE generally
prevents one common lifted query: the reverse map then depends on the memory
position.  Special commuting maps can be exceptions, but a position-specific
lift loses the common-query implementation benefit.  A memory sublayer can
avoid that obstruction by using a position-invariant key projection and
representing order in `Z` or the additive bias, but that is an architectural
choice and must be controlled.

All equalities in this paper are exact-arithmetic statements.  BF16, FP8, or
quantized reassociation changes rounding boundaries and therefore needs a
separate numerical-equivalence tolerance before physical deployment.

## Exact logical arithmetic ledger

For one query and one routed memory sequence of length `L`, ordinary on-demand
GQA cross-attention performs

\[
C_{ordinary}=2d^2+2L dGr+2LHr
\]

MACs: Q and O projections, memory K/V projections, and QK/PV products.

The reassociated form performs

\[
C_{reverse}=4d^2+2HLd
\]

MACs: Q, per-query-head reverse-K, raw-space scores, raw-space weighted sums,
per-query-head post-V, and O.  Post-V cannot generally be shared among query
heads in one GQA group because their attention weights differ.

For the frozen scratch shape

```text
d=384, H=6, G=2, r=64, L=256
```

the complete counts are:

| operation | ordinary on-demand | reverse-projected |
|---|---:|---:|
| Q projection | 147,456 | 147,456 |
| memory K/V projection | 25,165,824 | 0 |
| reverse K on query | 0 | 147,456 |
| scores | 98,304 | 589,824 |
| weighted values/raw states | 98,304 | 589,824 |
| post-attention V | 0 | 147,456 |
| O projection | 147,456 | 147,456 |
| **total** | **25,657,344** | **1,769,472** |

At this shape, reverse projection uses fewer logical MACs than computing raw
K/V on demand for `L>=4` **for one query and no reusable K/V cache**.  This is a
substantial implementation result, but both columns compute the same function.

## Persistent-state trade

A cached ordinary K/V representation for one 128-token document and one reader
layer contains

\[
2LGr=2\cdot128\cdot2\cdot64=32,768
\]

scalars.  In BF16 that is 65,536 bytes per document, or 157,614,080 bytes for
2,405 documents (about 150.3 MiB).  The corresponding fixed 128-token `uint16`
tape is 615,680 bytes for the entire corpus (about 0.587 MiB), before lengths,
title routing, and index metadata.  The raw-ID ratio to this one-layer BF16 K/V
table is exactly 256:1.

If K/V are already cached, ordinary one-query attention over two records costs

\[
2d^2+2LHr=491,520
\]

MACs, so the reassociated read spends about `3.60x` more arithmetic while
eliminating that K/V table.  A fused online-softmax kernel could read each raw
embedding once and maintain `H*d=2,304` running value accumulators, but that is
a physical hypothesis rather than part of the algebraic proof.  It also needs
the six transformed `d`-wide queries resident or repeatedly read.

There is a second amortization boundary.  If ordinary K/V are projected once
for the routed records and reused for `R` query tokens, their total logical cost
is

\[
2LdGr+R(2d^2+2LHr),
\]

whereas repeating the reverse reader costs `R(4d^2+2HLd)`.  At this frozen
shape and `L=256`, reverse projection is cheaper only through `R=19`; at
`R>=20`, projecting once and reusing ordinary K/V has fewer MACs.  The intended
one-boundary read therefore cannot be generalized to per-token cross-attention.

The table comparison is deliberately not called a complete serving ledger:
quantized/PQ K/V controls, raw embedding gather traffic, routing metadata,
workspace, and the cost of locating the two records still have to be charged.

## A possible fixed-byte/fixed-MAC composition—and why it is not admitted

If the reader reuses an existing layer's Q/K/V/O matrices, it adds no learned
projection weights.  The 615,680 raw-ID bytes equal 307,840 BF16 weight slots.
Across ten width-384 SwiGLU layers, removing 27 intermediate channels frees

\[
27\cdot10\cdot3\cdot384=311,040
\]

BF16 weights and the same number of dense MACs per processed token.  If the
memory reader executes `R` times over a sequence containing `T` ordinary
processed tokens, the logical MAC condition is

\[
R\cdot1,769,472\le T\cdot311,040,
\qquad\text{or}\qquad T/R\ge5.6889.
\]

Thus six or more tokens fund the reader only when it executes once.  A
per-token memory read fails this funding argument.

The byte match is similarly fragile.  Twenty-seven channels free 622,080
bytes, only 6,400 more than the raw ID tape.  One `uint16` length per document
uses another 4,810 bytes and leaves 1,590 bytes—already insufficient for a real
title/router index.  The 27-channel statement is therefore a raw-tape-only
lower bound, not a complete stored-state equality.  Width 997 is also not a
friendly tensor-core tile, so the logical dense-MAC reduction may not become a
physical saving.

This construction shows that a raw-memory hybrid could fit the coarse global
byte and MAC envelopes without adding projection weights.  It does **not**
establish a Pareto model:

1. reducing every FFN from 1,024 to 997 may damage protected language ability;
2. sharing one layer's projections between causal self-attention and memory
   cross-attention is an unproved optimization constraint;
3. routing and index bytes are missing from the equality above;
4. logical MAC equality does not imply equal H100 latency;
5. residual injection, gating, normalization, and their state/operations are
   not yet funded;
6. the reader output remains one `d`-dimensional mixture, with no proof that it
   preserves the two-record relations needed by natural QA.

These are multiple independent empirical miracles, so the construction fails
the pre-run rule.

## Strongest-control and prior-art audit

### Function-class absorption

The reverse form is exactly absorbed by ordinary cross-attention.  It cannot
provide a separating function witness against that control.  Any capability
gain comes from granting the model routed raw evidence, not from reassociation.
An optimizing ordinary-attention control must therefore receive the same
compiler rewrite; a comparison against naively materialized K/V is only an
implementation baseline, never a capability control.

### Existing project controls

- T32 proved that correctly routed raw records contain a large natural
  information gain, but used a 9B reader and did not prove this bottleneck.
- T32R already proposed a query-conditioned raw-token scan and failed admission
  because rank-32 semantic sufficiency and autonomous routing were unproved.
- AFTA already covers shifted/local value transport and shows that sharing one
  address across multiple payload offsets is a workload specialization rather
  than a general replacement for independent heads.
- Direct title-shaped routing reached only 102/104 supporting pairs and is not
  an autonomous semantic router.

Full-width reverse projection removes T32R's declared rank-32 information
kernel, but it does not prove that the learned query selects the correct raw
tokens or that the resulting mixtures preserve syntax and relations.

There is also a strict representation dichotomy:

- If `Z` is the gathered raw token embedding plus fixed position features, the
  tiny raw-ID ledger is valid but `Z` has no contextual encoding and carries no
  proved syntax/relation representation.
- If `Z` is a contextual encoder state as in INTRA, the reader is stronger, but
  the raw-ID ledger is incomplete: the system must store `Z` or pay and account
  for the encoder pass that creates it.

The algebraic identity cannot erase this cost/capability choice.

### External collision

[INTRA](https://arxiv.org/abs/2605.05806) already derives Reverse-QWK for GQA,
moves layer/head-specific key projections onto the query, stores one shared
encoder representation, computes values on demand, and uses decoder attention
queries for retrieval.  Its Appendix A gives the same per-head lifted-query
construction and explicitly treats the result as an implementation device.

The post-softmax value reassociation above is a valid additional linear
identity, but it does not create a new model class.  INTRA also relies on a
pretrained encoder-decoder, retrieval tokens, supervised oracle evidence, and
large stored encoder states, so it is not a solution to this project's
from-zero/raw-ID objective.  It does, however, occupy the broad novelty claim
that attention-native retrieval plus reversed key projection is a new
direction.

## Breakthrough-size boundary

The natural target still requires moving from the best dense/reference result
of 56.7308% to at least 80%.  Correct explicit raw evidence gave a 9B reader
89.4231%, so a proposed small reader must recover at least 71.18% of the
observed oracle gap while protecting ordinary language ability.

Reverse projection has no semantic lower bound.  Because it is equal to its
ordinary cross-attention control, it also has no capability advantage to
propagate into that target.  The optimistic natural gain of the **hybrid memory
architecture** may be large; the gain of the **reassociation itself** is
exactly zero at equal inputs and weights.

## Explainable block cards

### Reverse-key block

| field | content |
|---|---|
| type | `q: R^r, W_K: R^(d x r) -> q_tilde: R^d` |
| meaning | the ordinary attention query expressed in the unprojected memory basis |
| construction | `q_tilde=q W_K^T` |
| invariant | `q_tilde Z^T = q(ZW_K)^T` exactly |
| assumption | position-independent linear key map after `Z` is defined |
| resource | `dr` MACs per query head; no per-memory-token K state |
| adversary | position-dependent key rotation |
| kill condition | equality failure under the declared memory attention semantics |
| non-claim | no relevance or routing guarantee |

### Post-attention value block

| field | content |
|---|---|
| type | `alpha: R^L, Z: R^(L x d), W_V: R^(d x r) -> o: R^r` |
| meaning | the ordinary head value without preprojected per-token V |
| construction | `o=(alpha Z)W_V` |
| invariant | `(alpha Z)W_V = alpha(ZW_V)` exactly |
| assumption | linear, position-independent value map; normalized `alpha` for bias movement |
| resource | `Ld+dr` MACs per query head; no persistent V state |
| adversary | token-dependent nonlinear value transform |
| kill condition | equality failure or a ledger dominated by cached/quantized V |
| non-claim | no evidence sufficiency or reasoning guarantee |

These cards are retained for future architectures that independently solve the
semantic and routing gates.  They do not authorize a standalone benchmark.

## What would reopen implementation

Implementation is admitted only as part of a new packet that supplies all of:

1. an autonomous raw-only router with a frozen corpus-growth test;
2. a minimal natural or controlled witness where the proposed memory
   composition preserves a query distinction that every byte/MAC-matched dense
   control loses;
3. a construction showing how local order and multi-token relations survive
   the one-boundary bottleneck;
4. an end-to-end ledger including title/index metadata, quantized K/V and
   raw-tape controls, traffic, workspace, and protected-path capacity;
5. a derived ceiling capable of reaching at least 80% natural QA without
   degrading protected NLL;
6. exactly one unresolved composition hypothesis for the eventual run.

Until then, the correct next action is further paper search—not a microbench.
