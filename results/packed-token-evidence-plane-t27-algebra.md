# Packed token evidence plane T27 — algebra and paper gate

Status: **STAGE 0 PASS; DECODED-PREFIX ORACLE FAIL; T27 CLOSED**  
Date: 2026-07-31

## Key shift

T12-T26 repeatedly tried to make a small continuous or discrete record mean a
semantic relation.  The representation then failed before physical storage
became relevant.

T27 does not ask the compiler to understand prose.  It stores an ordered raw
token prefix exactly, with a fixed tokenizer-defined meaning, and makes the
served model perform query-dependent reads over that packed evidence.  The
only raw compiler operation is tokenization, truncation/padding, and radix-16
packing.

This is not yet the full production objective.  It is a direct test of a more
fundamental hypothesis:

> Can exact raw evidence, packed into existing weights and read by a fixed
> algebra, outperform implicit gradient memorization without adding a served
> token, layer, parameter, KV entry, or matrix operation?

If the answer is no, semantic extraction was not the first bottleneck.  If the
answer is yes, later work may replace prefix selection with a learned or
lossless evidence selector while preserving the same read contract.

## P0: promised edge

The prospective edge is qualitative raw-knowledge acquisition at identical
served model structure and cost.  Candidate and frontier controls export the
same ordinary Transformer class, parameter count, tensor shapes, tokenizer,
precision, context state, and dense operation graph.  The candidate reallocates
some existing weight cells from analog training to exact compiled evidence.

No capability claim is allowed at Stage 0.  A paper pass admits only:

1. logical/BF16 primitive tests;
2. an information oracle over the decoded prefix;
3. a physical same-graph microbenchmark if the oracle passes.

## P1: baseline witness

On the current raw-prose development lane, ordinary from-zero training and
post-hoc semantic records have not made document evidence causally usable:

- the strongest frozen lexical payload reached 60.5769% held-out comparison
  QA;
- the T24 exact table was 100% correct on its own records, but its perfectly
  decoded local quotient reduced held-out QA to 45.1923%;
- the T23 continuous optimized record did not improve its held-out raw reads.

Those failures leave one untested object: an exact, ordered, tokenizer-level
evidence record with a constructive content-read law.

For arbitrary token tapes, any pooled 220-dimensional analog statistic can
alias distinct ordered tapes.  T27 is injective on its declared 55-token
domain.

## P2: typed blocks and meanings

Let the tokenizer vocabulary satisfy `V <= 65,536`.  For token ID `t`, define
four radix-16 digits

\[
n_k(t)=\left\lfloor t/16^k\right\rfloor\bmod 16,
\qquad k=0,1,2,3.
\]

The blocks are:

```
pack(tokens[0:55]) -> 220 radix-16 cells
load(title, model weights) -> packed record at the addressed title state
match2(query token pair, packed record) -> 55 position scores
select(position scores, packed record) -> four nibbles of a following token
decode(four nibbles) -> fixed 16-bit vocabulary code / ordinary logits
```

Every intermediate has a fixed observable meaning:

- cell `4*j+k` is digit `k` of tokenizer token `j`;
- position score `j` is the squared radix distance from the query pair;
- the selected four digits are a tokenizer ID;
- the final 16 coordinates are that ID's binary code.

There is no learned latent convention in storage or logical reading.

## P3: positive constructions

### Capacity and injectivity

`65,536 = 16^4`, so four four-bit cells encode one vocabulary token.  Exactly
220 cells encode 55 tokens:

\[
55\cdot4=220,
\qquad
55\cdot16=880\text{ logical bits}.
\]

Radix expansion is unique, hence `unpack(pack(x)) = x` for every declared
tape.  The odd amplitude map

\[
a(n)=2n-15\in\{-15,-13,\ldots,15\}

\]

is exact in BF16.  It gives adjacent nibble values a distance of two.

### Squared-distance primitive in one SwiGLU

For any scalar difference `d`, the identity

\[
d\operatorname{SiLU}(d)+(-d)\operatorname{SiLU}(-d)=d^2

\]

is exact in real arithmetic because `sigma(d)+sigma(-d)=1`.

A SwiGLU channel can put `d` in both its gate and value branches.  A second
channel uses `-d` in both.  Their down-projection sum is `d^2`.  Therefore a
two-token key over four nibbles per token and 55 candidate positions requires

\[
2\text{ signs}\cdot2\text{ tokens}\cdot4\text{ nibbles}\cdot55
=880

\]

intermediate channels.  This fits an existing width-1,024 SwiGLU.  The
down-projection emits the 55 total squared distances

\[
D_j=\sum_{u=0}^{1}\sum_{k=0}^{3}
  \left(a(n_k(q_u))-a(n_k(t_{j+u}))\right)^2.

\]

`D_j = 0` exactly for a matching pair.  Every non-match has `D_j >= 4` in real
arithmetic, leaving a finite threshold interval `(0,4)`.

### Selection primitive

Given match indicators `m_j` and a unique matching position, four selected
digits are

\[
o_k=\sum_j m_j a(n_k(t_{j+2})).

\]

This needs `55*4 = 220` gate/value products in a later SwiGLU, again below
width 1,024.  A soft SiLU threshold may approximate `m_j`; exact discrete
selection is the Stage-0 reference, and BF16 margin is a separate physical
microbenchmark.

### Token-output identity

The four selected nibbles can be converted to the token ID's 16 ordinary bits
by four copies of the retained 16-level triangular decoder.  The prior digital
plane used 48 shared channels for one four-bit value; four nibbles therefore
need at most 192 channels.

Reserve 16 coordinates of the tied embedding/output rows for the fixed bipolar
binary code `c(t) in {-1,+1}^16`.  The correct token scores 16.  Any different
token differs in at least one bit and scores at most 14, giving a noiseless
margin of two.

The complete logical reader therefore fits in three ordinary later SwiGLUs
with maximum required widths 880, 220, and 192.  It adds no layer or channel to
the served graph; it reallocates existing ones.

## Physical placement sketch

Use one existing FFN channel per document across three early blocks, as in the
T19 ledger.  A title-address gate activates that channel at its title token and
the channel's down row writes the 220 packed cells into reserved hidden
coordinates.  Later attention can keep the two addressed document records at
their two distinct title positions.  Three later FFNs implement match,
selection, and token-code decoding.

The sketch preserves dense tensor shapes and nominal MACs.  It is not yet a
physical proof: RMSNorm scaling, residual contamination, multiple matches,
SiLU threshold margin, layer placement, and protected language capacity must
all be measured before training.

## P4: explicit obstruction and scope

1. **Length:** the exact uncompressed tape holds only 55 tokenizer tokens.
   Truncation is an information loss, not a reader failure.
2. **Selection:** a raw prefix is not a learned evidence selector.  A win on a
   lead-sentence-heavy corpus cannot be called a production extractor.
3. **Surface addressing:** exact lookup finds tokenizer sequences, not unseen
   paraphrases or logical predicates.  Natural query-to-key alignment remains
   the one composition hypothesis after an information oracle passes.
4. **Ambiguity:** a repeated two-token key has several successor positions;
   the simple selector is exact only for a unique match.  A later construction
   must return a set, add context, or define a deterministic tie rule.
5. **Output:** the proved reader returns a token code.  Multi-token answers and
   arbitrary reasoning still require the ordinary model.
6. **Finite precision:** the squared-distance identity is algebraic; exact
   BF16 margin through RMSNorm and real Transformer residuals is unproved.
7. **Capacity reallocation:** using 220 hidden coordinates and up to 880/220/192
   FFN channels can harm general language modeling even though served FLOPs do
   not increase.

These boundaries prevent T27 from inheriting the final goal merely because its
pack/read identity is exact.

## P5: matched control

An ordinary Transformer of the same size can embed the identical circuit.
T27 does not enlarge its function class.  The claim is a learning/allocation
edge: exact raw facts are placed into a canonical finite subspace rather than
being rediscovered through gradient superposition.

Controls must receive:

- the same tokenizer and raw-token pool;
- the same fixed token-code auxiliary targets;
- the same reserved coordinates/channels or an equal-capacity dense
  reallocation;
- matched charged training work and initialization;
- the same title-routing views.

The only candidate-specific operation is the raw-derived digital weight write.

## P6: resource ledger

For 2,405 documents, 4,150 active title-token rows, 32 title-address
coordinates, and 220 payload cells:

| write | existing entries reassigned |
|---|---:|
| title token codes | 132,800 |
| per-document FFN gate keys | 76,960 |
| thresholds | 2,405 |
| up constants | 2,405 |
| packed token payloads | 529,100 |
| 16-bit codes for all 49,152 tokenizer rows | 786,432 |
| **total** | **1,530,102** |

This is 4.183% of the 36,577,152-parameter model.  It exceeds T11's narrow
743,734-entry isolation precedent and therefore receives no inherited
no-regression result.  The controls must reserve the same capacity, and
protected natural NLL must be re-established.

The exported state dict, bytes, BF16 precision, forward graph, matrix shapes,
MACs, KV cache, and request length remain unchanged.  Compiler tokenization and
writes are training-only work and are charged.

## P7: breakthrough-size plausibility

The mechanism can create a qualitative acquisition difference: on every
covered prefix and exact key, the candidate can retain and read the observed
token with zero logical error, while gradient controls have previously failed
exact arbitrary fact acquisition even at twice the exposure.

The natural effect is not established.  An exploratory, non-authoritative
length census performed after deriving the 55-token capacity found:

- 88.86% of document first sentences fit completely in 55 SmolLM2 tokens;
- 97.06% of valid development questions have both support sentences start
  before token 55;
- only 71.57% have both support sentences finish before token 55.

Therefore full-sentence coverage alone does not clear the 75% natural target.
The next gate must use a stronger decoded-prefix reader and causal controls;
model training is forbidden until that oracle shows a large gain.

An exploratory static Huffman census reached 94.12% complete two-sentence
coverage inside 880 logical bits, but variable-length decoding lacks the
fixed-cost read proof above.  It is a later compression possibility, not part
of T27 admission.

## Paper decision

T27 passes the paper gate only for two cheap stages:

1. a CPU logical/BF16 primitive microbenchmark that cross-checks packing,
   squared-distance matching, unique selection, and token decoding;
2. a separately frozen stronger-information oracle over the exactly decoded
   55-token prefixes, with correct/zero/shuffled controls.

Failure of the primitive closes this read circuit.  Failure of the decoded
prefix oracle closes fixed-prefix evidence before any physical implementation.
Passing both may admit a same-graph physical reader microbenchmark; it still
does not admit a from-zero LM run or a production claim.

## Observed decision

Stage 0 passed, but the first and only decoded-prefix scoring attempt produced
an exact BF16 yes/no logit tie in the full-document condition.  The frozen
oracle declared any tie invalid, so no rerun or score repair is permitted.
T27 is closed before physical implementation.  Its exact radix packing and
content-read identities remain reusable primitives, not evidence for the
natural payload.  See
[`packed-token-evidence-plane-t27-prefix-oracle-decision.md`](packed-token-evidence-plane-t27-prefix-oracle-decision.md).

## Prior-art boundary

Reversible memory embeddings, compressed memories, and lossless neural text
compression are prior directions.  For example,
[Memory Tokens](https://arxiv.org/abs/2506.15001) demonstrates reconstruction
from optimized embeddings, while the current
[Language Model Memory](https://arxiv.org/abs/2602.13466) work studies
information-retaining objectives.

T27 claims no novelty yet.  Its narrower testable conjunction is: tokenizer-ID
radix cells compiled into existing weights, a constructive content lookup
using ordinary-width SwiGLUs, and no added served artifact or operation graph.
Novelty and utility remain open until behavioral and physical evidence exists.
