# Scale-referenced whole-record plane T32 — paper gate

Status: **PAPER PASS TO FROZEN CPU CODEC GATE ONLY**  
Date: 2026-07-31

## Verdict

T32 is the first post-T27 representation that removes both of T27's declared
payload defects on the frozen corpus domain:

1. it stores all 128 tokenizer slots rather than a 55-token prefix; and
2. every token is directly addressable without running an entropy decoder.

The construction uses 382 of a width-384 state. It is exact, finite, and has a
scale reference that makes its 64 logical levels invariant to a common
RMSNorm scale. It does not yet prove that natural questions can select useful
evidence, that a ten-layer Transformer can execute the complete reader, or
that reserving the weight rows preserves language quality.

Only an exact CPU codec/numerics/resource gate is admitted. No model training,
natural evaluator, H100 work, or production claim is admitted by this paper.

## P0: promised edge

At unchanged served checkpoint shape, parameter count, BF16 precision, KV
shape, request tokens, and dense matrix shapes, place the complete frozen
128-token document record into existing weights and let ordinary model layers
perform query-dependent reads over it. The intended gain is qualitative
held-out knowledge acquisition relative to gradient memorization, not a small
NLL improvement.

The different currencies are compiler work and reallocated weight/features.
Neither is free. The final claim requires identical service latency and
protected quality, not merely nominally unchanged GEMMs.

## P1: baseline and predecessor obstruction

T19d losslessly entropy-coded the frozen document tokens in 473,358 four-bit
cells, but exact decoding depended on all earlier decoded tokens and one
record exceeded its 350-cell layout. T27 instead used an exact random-access
radix-16 tape, but 220 cells held only 55 tokens. Its decoded-prefix oracle was
closed after one exact BF16 decision tie, and no payload-sufficiency result was
obtained.

T32 changes the representation rather than relaxing either result. It uses
the full width-384 document carrier and a higher but still finite digit radix.
It does not inherit a natural oracle pass from T27.

## P2: typed codec

Let a document record contain `L <= 128` token IDs padded to
`t_0,...,t_127`, with every `t_i < 65,536`. For tokens `i=0,...,126`, define

\[
\ell_i=t_i\bmod64,\qquad
m_i=\lfloor t_i/64\rfloor\bmod64,\qquad
h_i=\lfloor t_i/4096\rfloor\in\{0,...,15\}.
\]

Define twelve two-bit auxiliary digits:

- `a_0,...,a_7` are consecutive two-bit chunks of `t_127`;
- `a_8,...,a_11` are consecutive two-bit chunks of the eight-bit length `L`;
- `a_i=0` for `i>=12`.

The three stored base-64 cells for regular token `i` are

\[
(\ell_i,m_i,c_i),\qquad c_i=4h_i+a_i.
\]

Coordinates `0..380` contain these 127 triples. Coordinate 381 is the fixed
scale reference `1`. Coordinates 382 and 383 are reserved and zero in T32.

The maps have explicit meanings:

```text
pack(token_ids, length) -> 384 finite record cells
unpack(record)           -> exact token_ids and length
load(title, weights)     -> addressed record at a title state
match(query code, record)-> token-position evidence scores
select(scores, record)   -> a bounded raw token window
reason(question, window) -> answer-bearing ordinary hidden state
```

Only `pack` and `unpack` are admitted to implementation by this paper.

## P3: exact capacity and inverse theorem

### Theorem 1: injectivity

For every declared record, `unpack(pack(t,L))=(t,L)`.

**Proof.** `ell_i` and `m_i` are stored directly. Exact division of `c_i` by
four recovers quotient `h_i` and remainder `a_i`. Therefore

\[
t_i=\ell_i+64m_i+4096h_i
\]

for `i<127`. Concatenating the two-bit remainders `a_0,...,a_7` reconstructs
all 16 bits of `t_127`; concatenating `a_8,...,a_11` reconstructs `L`. Every
operation is over bounded integers, so the inverse is unique. QED.

The record uses

\[
127\cdot3+1=382
\]

active coordinates and leaves two reserved. Its 381 base-64 cells have 2,286
logical bit positions. They carry 2,048 token bits plus eight length bits; the
remaining capacity is structural slack required by direct access.

This is not entropy compression. It is a finite, aligned, random-access code.

## P4: common-scale decoding theorem

Encode a base-64 digit `d` as the odd integer

\[
A(d)=2d-63\in\{-63,-61,...,63\},
\]

and encode the scale-reference coordinate as `1`. Suppose RMSNorm or another
common positive scale maps the ideal pair to

\[
x_d=sA(d),\qquad x_r=s,
\]

for `s>0`. The boundary between digits `k` and `k+1` is the even integer

\[
\tau_k=2k-62.
\]

The sign of

\[
x_d-\tau_k x_r=s(A(d)-\tau_k)
\]

therefore determines which side of every boundary contains `d`, independently
of `s`. The minimum exact-real margin is `s`.

For BF16 round-to-nearest inputs with unit roundoff `u=2^-8`, let the rounded
pair be `x_d+e_d,x_r+e_r`. Ignoring underflow, the comparison error is bounded
by

\[
|e_d-\tau_k e_r|
\le us(|A(d)|+|\tau_k|)
\le 125us
=0.48828125s<s.
\]

Thus BF16 rounding of the normalized cells alone cannot cross a digit
boundary when products/subtractions accumulate in FP32. This theorem does not
cover arbitrary residual contamination, BF16 accumulation, or a physical
Transformer path; the CPU gate must test the exact declared arithmetic and a
separate perturbation margin.

Once `c_i` is decoded, `h_i=floor(c_i/4)` and `a_i=c_i mod 4` are exact. The
auxiliary packing therefore does not weaken the token code.

## P5: direct query access

The record has no dependency chain. Any regular token needs exactly its three
cells; the last token needs the twelve registered auxiliary remainders. A
query token is represented by its own low-six, middle-six, and high-four
parts.

The retained paired-SiLU identity

\[
z\operatorname{SiLU}(z)+(-z)\operatorname{SiLU}(-z)=z^2
\]

can compare low and middle digits. For positions `i>=12`, `a_i=0`, so the high
cell is exactly `4h_i` and is compared the same way. The first twelve
positions require a bounded four-value membership check, and token 127 a
bounded auxiliary reconstruction. Those are finite circuit obligations, not
assumed free.

A one-token scan over the 115 uniform positions needs

\[
2\text{ signs}\times3\text{ fields}\times115=690
\]

square channels, below width 1,024. Handling the twelve auxiliary-bearing
positions and the last token is deferred to a separately proved reader gate.
Two-token keys, window scoring, and two-document reads may consume additional
layers but may not increase their width.

This count proves only that the uniform bulk is not immediately too wide. It
is not a whole-model placement proof.

## P6: information and resource boundary

For 2,405 documents, the raw payload writes are

\[
2,405\times382=918,710
\]

existing BF16 entries. This exceeds the old 743,734-entry narrow-isolation
precedent but is only 2.5128% of the 36,577,152-parameter model before routing
and shared-reader entries. T32 receives no inherited no-regression claim.

The Stage-0 resource gate must include title codes, one addressed route per
document, thresholds/constants, payload rows, shared token codes, and shared
reader reserve. The complete prospective write must remain below 5% of model
entries. A later matched control reserves the same entries.

The record contains the observed token information; it does not create new
information or compress arbitrary facts below entropy. Its possible advantage
is that exact evidence is acquired by a deterministic write rather than by
gradient superposition.

## P7: natural sufficiency gate before physical integration

If the CPU codec passes, the next artifact must freeze a stronger-information
oracle before reading evaluator answers. It must compare:

1. question only;
2. the two correct full 128-token records decoded to text;
3. a deterministic raw-token local-window selector over those records;
4. shuffled-document versions of both evidence conditions.

The full-record condition must exceed 80% held-out accuracy and beat both
question-only and shuffled evidence by at least 15 points. The local-window
condition must exceed 75%, lie within five points of full-record accuracy, and
lose at least ten points when documents are shuffled. Complete forced-string
answer likelihood, not a single BF16 token logit, must be specified before the
run.

Failure closes the representation before reader circuits or model training.
Passing only shows that the information is sufficient; it does not show that
the fixed graph can read it.

The raw `document_id` source field may be used only to audit uniqueness and
order invariance. It is not codec input; the packed record is derived solely
from tokenizing `title + "\n" + text`.

## P8: matched controls and kill conditions

Any later model experiment must compare against:

- ordinary from-zero NTP;
- the same raw-document and query-derived training views without a record;
- equal-capacity free learned rows;
- equal charged training work;
- correct, zero, shuffled-document, and shuffled-record records;
- a prompt/RAG information ceiling priced separately;
- protected natural NLL and capability slices.

Kill the branch on any of the following:

- one codec collision or BF16 boundary error;
- tokenizer ID or length outside the declared domain;
- more than 384 state cells or 5% prospective weight writes;
- failure of the stronger-information oracle;
- a reader needing extra prompt tokens, KV entries, parameters, layers,
  workspace, or unpriced operations;
- serving latency or protected-quality regression;
- only a small endpoint gain.

## P9: prior-art boundary

[Parametric RAG](https://openreview.net/forum?id=kcKZoLKv1m) parameterizes
documents offline, [Knowledge Capsules](https://arxiv.org/abs/2604.20487)
compile structured document knowledge into externally injected KV, and
[MemoryFormer](https://openreview.net/forum?id=04EC4ZnZJj) replaces dense FFN
transforms with hashed lookup tables. RAG, FFN memories, document parameters,
external KV, finite coding, and exact lookup are all established families.

T32 claims no novelty for those blocks. The research conjunction is narrower:
an exact scale-referenced full-record code, compiled from raw tokens into
already-resident ordinary-model entries, with a query-dependent reader and a
breakthrough-size title-disjoint gain at no served-resource or protected-
quality increase.

## Paper decision

The code has an explicit inverse, a finite-precision boundary theorem, a
direct-access law, an initial channel bound, a resource ceiling, and a stronger
information oracle that precedes physical work. It therefore earns one frozen
CPU codec gate. Nothing else is admitted yet.
