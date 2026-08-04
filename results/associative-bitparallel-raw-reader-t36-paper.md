# T36 associative bit-parallel raw reader — paper screen

Date: 2026-08-01  
Decision: **CLOSED ON PAPER; EXACT ALGEBRA RETAINED, NO CPU/GPU/RUN ADMITTED**

## Intended breakthrough claim

T36 asks whether a language model can replace continuous attention over a raw
token plane with an exact packed discrete query machine:

```text
natural question -> short regular program -> packed raw-token scan
                 -> exact matched span -> small continuous gather -> answer
```

The hoped-for exchange is large: read two-byte token IDs and execute word
logic over them, expand only a few matched spans into width-`d` vectors, and
spend the avoided continuous memory traffic and matmul on a stronger dense
backbone at identical served cost.

The algebra is exact and clean.  It does not survive the strongest-control and
acquisition gates.

## P1. Restricted positive witness

Let the raw plane be tokens `t_1,...,t_n`.  A length-`m` query program is a
sequence of token classes

\[
A_0,A_1,\ldots,A_{m-1},\qquad 1\le m\le w,
\]

where `w` is the machine-word width.  A match ending at `i` exists when

\[
t_{i-m+1+j}\in A_j\quad\text{for every }j\in\{0,\ldots,m-1\}.
\]

Define the word mask

\[
M(x)_j=
\begin{cases}
\mathbf 1[x\in A_j], & 0\le j<m,\\
0, & m\le j<w.
\end{cases}
\]

and the Shift-And state

\[
D_i=((D_{i-1}\ll 1)\;|\;1)\;\&\;M(t_i),\qquad D_0=0.
\]

Bit `m-1` of `D_i` is one exactly when the program matches a suffix ending at
`i`.

### Exactness proof

Invariant: after token `i`, bit `j` of `D_i` is one exactly when the length
`j+1` prefix of the query program matches the length-`j+1` suffix ending at
`i`.

For `j=0`, OR with one starts a new candidate and AND with `M(t_i)` retains it
exactly when `t_i in A_0`.  For `j>0`, left shift moves the truth of prefix
`j` at `i-1` to prefix `j+1` at `i`; AND with `M(t_i)` retains it exactly when
the new token belongs to `A_j`.  Induction over `i,j` proves the invariant and
the accept-bit statement.

### Small witness

For the program `[Ada] [was] [born]`, use masks:

```text
M(Ada)  = 001
M(was)  = 010
M(born) = 100
```

Starting from zero, the states are `001`, `010`, `100`; the final high bit is
the exact match.  A following fixed-width value can be returned by position,
not by attending to the key token's value vector.  This avoids T35's
key-to-successor error.

### Boundary

This witness proves exact bounded lexical pattern matching.  It does not prove
that a natural question identifies the correct token classes, relation, value
boundary, record among aliases, or answer format.

## P2. Associative segment algebra

For `0 <= k <= w`, let `U_k` be the word mask with its lowest `k` bits
cleared (`U_w=0`).  Use canonical tuples satisfying `A=A&U_k`; shifts are
truncated to `w` bits.  Define

\[
F_{k,A,B}(x)=((x\ll k)\;\&\;A)\;|\;B.
\]

One token `t` has transform

\[
T_t=(1,M(t)\;\&\;U_1,M(t)\;\&\;1).
\]

If `F=(k,A,B)` is followed by `G=(l,C,D)`, their composition is

\[
G\circ F =
\left(
k+l,
(A\ll l)\;\&\;C,
((B\ll l)\;\&\;C)\;|\;D
\right),
\]

where `k+l` is capped at `w` because larger shifts are zero.  The resulting
`A` is canonical: shifting canonical `A` by `l` clears its lowest `k+l`
bits, and the capped-at-`w` case is zero.

### Composition proof

Direct substitution gives

\[
\begin{aligned}
G(F(x))
&=((((x\ll k)\&A)|B)\ll l)\&C)|D\\
&=((x\ll(k+l))\&(A\ll l)\&C)
  |((B\ll l)\&C)|D.
\end{aligned}
\]

This is the stated tuple.  On canonical tuples the identity is
`(0, all_ones, 0)` on both sides.

### Associativity theorem

The tuple operation is associative on the represented transforms.

Proof: the formula above is exact function composition, and direct expansion
of three canonical tuples gives the same capped shift, `A`, and distributed
AND/OR expression in either grouping.  Canonicality is preserved, and the
two-sided identity was established above.  Therefore these tuples form a
monoid.  Function-composition associativity gives the same result semantically.

Consequently a work-efficient parallel prefix scan can compute every `D_i`
with `O(n)` tuple combines and `O(log n)` dependency depth.  A sequential scan
uses one state word and `O(n)` word operations.

### Claim boundary

Associativity removes the algebraic sequential dependency.  It does not prove
that a GPU prefix scan is faster or cheaper than a fused continuous attention
kernel.  A scan materializes or communicates segment tuples; that physical
traffic must be counted.

## P3. Exact capture

For a fixed-width record grammar, include boundary classes in the program and
return the accepted end position plus fixed offsets.  Gather only those token
IDs or their shared embeddings.  This construction is exact.

A variable value of at most `c` tokens would need an explicit `c`-state capture
automaton or a separately defined boundary transition and deterministic
selection rule.  T36 does not provide that construction, so variable capture
is not part of the retained theorem.

This is a correct discrete position transport.  It does not require an
attention head to make a dynamic key-to-successor hop.  It also does not infer
the record grammar: the program already specifies it.

## P4. Idealized resource upper bound

For one `m<=64` program scanning `n` uint16 token IDs sequentially, the logical
work is approximately:

- `2n` raw-ID bytes read;
- one mask-table access per token, which may hit cache but is not free;
- per token: one shift, one OR, one AND, plus loop/addressing work;
- `O(occ)` position writes and at most `K*d` gathered continuous elements.

The mask table is query-specific, so it is not a free constant.  Constructing
`M(x)` requires reading the emitted program, writing or clearing its token-class
representation, and either building the touched token-ID entries or performing
membership tests during the scan.  Preallocating dense masks instead charges
`vocabulary_size * w` bits per live program plus initialization traffic.  T36
does not provide a finite bound for this setup, so the idealized count above is
not a complete serving ledger.

An ordinary continuous read that expands every raw token and scores it against
a width-`d` query requires `Theta(nd)` continuous-element traffic and
`Theta(nd)` multiply-accumulate work before the selected-value reduction.
Thus the discrete scan has a real idealized separation from that deliberately
dense reader when `d` is large and `K << n`.

This comparison is insufficient for admission:

1. an FM/self-index can answer exact substring queries in work tied mainly to
   pattern length and occurrences rather than scanning all `n` tokens;
2. inverted/multigram indexes can prefilter regular-query candidates;
3. the strongest discrete raw-memory control can execute the exact same
   Shift-And or segment-monoid scan;
4. integer issue rate, cache misses, scan synchronization, tuple workspace,
   sparse gathers, and launch latency have no finite physical ledger; and
5. saving work against one auxiliary dense reader does not yet fund a stronger
   backbone under the complete end-to-end serving vector.

The `O(n)` versus `Theta(nd)` observation is retained as a systems reason to
prefilter.  It is not a fixed-cost model-capability theorem.

## P5. Strongest controls

T36 must beat all of these at identical persistent bytes and serving cost:

1. **Compressed self-index:** lossless corpus plus exact substring locate.
2. **Sparse regular-query index:** multigram/posting prefilter plus exact
   automaton verification.
3. **Direct automaton control:** the identical packed token IDs, query program,
   masks, Shift-And/segment scan, captures, gather, and decoder.
4. **Continuous raw-memory control:** the best learned sparse/dense retrieval
   reader, not only full cross-attention.

Control 3 is terminal.  If T36 includes a learned question-to-program mapper
`P_phi` and answer join `J`, the direct control computes

\[
J\left(Q,\operatorname{Gather}
  (X,\operatorname{Scan}(X,P_\phi(Q)))\right)
\]

with the same program, artifact, instruction trace, and resource vector.  By
the post-T35 control-closure theorem, the control can be parameterized as T36
itself.  T36 therefore has no capability or resource separation from its
strongest legal control.

Calling the bitwise scan a new neural head does not change this result.  The
operation is a classical automaton program already legal to the discrete
control.

## P6. Acquisition audit

Even if the direct automaton control were omitted, the natural-language path
would not meet the one-uncertainty rule.  Raw next-token training must cause:

1. the corpus compiler to expose a record grammar and capture boundaries;
2. a question hidden state to emit the corresponding token-class program;
3. aliases/paraphrases to choose the right lexical anchors and record; and
4. the model to use or copy the captured span without harming baseline-correct
   cases.

The first three are not consequences of the exact scan theorem.  A synthetic
cloze objective generated from a known grammar would supervise them, but then
the grammar constructor is the missing semantic compiler.  A learned regex
from labeled positive/negative examples changes the raw-data contract.

The raw-only proxy-to-natural-question transfer cannot absorb all these edges
into one empirical scalar.  This repeats the T22/T34 failure with a cleaner
executor.

## P7. End-effect audit

Let `S` select the discrete path, `T` mean the final T36 answer is correct, and
`B` mean the baseline is correct.  The exact accuracy change remains

\[
\Delta A=P(S\cap T\cap\neg B)-P(S\cap\neg T\cap B).
\]

On the inherited held-out task, moving from `56.7308%` to `80%` requires

\[
\Delta A\ge 0.232692.
\]

Exact lexical-match accuracy supplies neither term: it does not say the path
fires on baseline errors or abstains on baseline-correct cases.  Against the
direct automaton control, the paired gain ceiling is exactly zero because the
control can reproduce every selection and answer.

## P8. Prior-art boundary

No novelty is claimed for the components:

- Shift-And/Bitap is a classical bit-parallel NFA simulation; a modern exact
  invariant appears in
  [bit-parallel sequence-to-graph alignment](https://pmc.ncbi.nlm.nih.gov/articles/PMC6761980/).
- The parallel-prefix move is directly occupied by Mitani, Ino, and Hagihara's
  [inclusive-scan GPU formulation of Shift-Or and Wu-Manber](https://doi.org/10.1109/TPDS.2016.2645222),
  which defines associative companion operators and benchmarks a hybrid
  scan/segmentation/bit-level implementation.  T36's tuple notation therefore
  does not establish a new GPU systems direction either.
- [HybridSA](https://doi.org/10.1145/3689771) compiles regular expressions to
  masks and executes Shift-And-derived bit-parallel kernels on GPUs, including
  a CPU/GPU split for patterns with unfavorable access behavior.  It is a
  direct collision with the proposed packed GPU executor.
- GPU bitvector/semiring implementations already exploit low-branching packed
  regular-language operations in
  [Search-Based Regular Expression Inference on a GPU](https://arxiv.org/abs/2305.18575).
- exact substring lookup has strong compressed controls such as
  [prefix-free-parsed FM indexes](https://arxiv.org/abs/2305.05893) and
  [Infini-gram mini](https://arxiv.org/abs/2506.12229), whose FM index is both
  an exact locate/count structure and a compressed representation of the raw
  corpus.
- regular-expression indexing already combines multigram posting filters with
  exact verification in
  [An Index for Regular Expression Queries](https://arxiv.org/abs/1108.1228).
- unsupervised corpus automata have already been coupled to language models in
  [RetoMaton](https://arxiv.org/abs/2201.12431).
- explicit memory has already been trained with a from-scratch language model
  in [Memory3](https://arxiv.org/abs/2407.01178).
- [S3-Attention](https://arxiv.org/abs/2601.17702) is a strong continuous/sparse
  reader control: it derives sparse feature IDs from a model's own projections,
  builds a CPU inverted index over positions or spans, and retrieves compact
  evidence instead of retaining the full KV cache.  It is not an exact T36
  collision, but prevents comparison only against dense continuous attention.

The tuple composition above is a useful derivation for parallelization, not a
claim that finite-state transition composition is new.

## Decision

Close T36 before even an exhaustive CPU check.  The mathematical recurrence
and segment monoid are already proved by direct algebra; executing small words
would only test transcription.  The hypothesis that matters—natural semantic
program acquisition with a material end effect—is not isolated, and the
strongest direct automaton control contains the complete method.

Retain only:

1. the exact Shift-And segment monoid;
2. exact fixed-offset positional capture as a repair for key-to-successor
   transport;
3. the idealized `O(n)` word-logic versus `Theta(nd)` continuous-reader
   separation as a possible implementation primitive; and
4. the rule that any future raw-memory baseline receives the strongest
   self-index, sparse index, and discrete automaton controls.

No CPU benchmark, local GPU run, or rental is scientifically admitted.

Any later C/C++ implementation of the retained algebra must implement a
capped shift `shl_w(x,s)=0` for `s>=w`; a native shift by the machine-word
width is undefined behavior even though the mathematical operator is sound.
