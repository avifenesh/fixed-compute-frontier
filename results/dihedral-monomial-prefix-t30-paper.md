# Dihedral-monomial prefix operators T30 — paper gate

Status: **EXACT PRIMITIVE RETAINED; BREAKTHROUGH DIRECTION CLOSED ON PAPER**  
Date: 2026-07-31

## P0. One-sentence edge

Replace the homogeneous part of T29's 110-coordinate diagonal recurrence by a
109-coordinate dihedral permutation followed by a diagonal scale, preserving
the 220-byte recurrent-state envelope, the 220-cell document-record envelope,
and one multiply/add per state coordinate while adding exact noncommutative
state routing and retaining exact document-operator composition.

The hoped-for edge is not a smaller algebra test score.  It is a hybrid model
that can acquire a raw document once, compose previously unseen document
combinations at title triggers, and perform state tracking that a diagonal
head cannot perform, without a larger served checkpoint or recurrent state.

## Explain it without notation

Treat the memory as 109 labeled drawers arranged on a ring.  A token performs
three operations:

1. rotate or mirror the drawer labels;
2. multiply each drawer by one coefficient;
3. add one value to each drawer.

Rotating or mirroring does not perform a dense matrix multiplication.  It only
changes which logical label refers to which physical drawer.  The entire
document can therefore be stored as:

- the final rotate/mirror orientation: eight bits;
- 109 final scale values;
- 109 final offset values.

Reapplying those three pieces once must equal replaying all document tokens.
This statement is algebraic and independently testable.  Whether the drawers
hold useful language features is separate.

## P1. The diagonal obstruction

For T29's homogeneous update, the state Jacobian is diagonal:

\[
J_x=\operatorname{diag}(a(x)).
\]

Every pair of such Jacobians commutes and every coordinate subspace is
invariant.  The affine offsets can make whole token transitions order
sensitive, but no existing state component can move to a different coordinate.

This blocks faithful linear realization of even two noncommuting state
actions.  Let `rho` rotate 109 positions by one and let `tau` reflect them.
They obey

\[
\tau\rho\tau=\rho^{-1},
\]

so `tau*rho != rho*tau`.  Two simultaneously diagonal state actions cannot
satisfy that faithful relation because diagonal matrices commute.

This is a limitation of the homogeneous transition class, not a universal
language-model lower bound.  A nonlinear decoder can decode a fragile scalar
sequence code, a dense recurrence can route state, and multiple layers can
simulate richer computation at other cost.

## P2. Typed operator

Let `m=109`.  The dihedral group `D_109` consists of pairs

\[
g=(s,r),\qquad s\in\{-1,+1\},\quad r\in\mathbb Z_{109},
\]

acting on drawer indices as `i -> s*i+r (mod 109)`.  It has `2*109=218`
elements, so its complete state fits in eight bits.  Composition is

\[
(s_2,r_2)(s_1,r_1)
=
(s_2s_1,\ r_2+s_2r_1\pmod {109}).
\]

Let `P_g` be the corresponding permutation operator, with
`P_g P_f = P_(gf)`.  One token or compiled block is the typed map

\[
T_{g,a,b}(h)=a\odot P_g h+b,
\]

where `h,a,b` are real vectors of width 109.

The blocks are:

```text
token features       -> dihedral element g, scale a, offset b
raw document fold    -> one monomial-affine operator (g_D,A_D,B_D)
title precomposition -> one replacement title operator
stored record        -> 2 group nibbles + 109 A cells + 109 B cells
recurrent update     -> lazy permutation view, multiply, add
reader               -> ordinary downstream hidden computation
```

Every stored field names a directly executable state transformation.

## P3. Positive construction

### Closed composition law

Using `P_g(a odot z)=(P_g a) odot (P_g z)`, applying operator 1 and then
operator 2 gives

\[
T_{g_2,a_2,b_2}\circ T_{g_1,a_1,b_1}
=
T_{g_2g_1,
   a_2\odot P_{g_2}a_1,
   a_2\odot P_{g_2}b_1+b_2}.
\]

The identity is `(e,1,0)`.  Closure and associativity follow either by direct
expansion or because these triples denote ordinary affine functions and the
displayed law equals function composition.  A balanced prefix scan therefore
has `O(log n)` dependency depth and `O(nm)` work.

### Exact document substitution

For any prefix `P`, document `D`, suffix `Q`, and incoming state `h`, fold the
raw document transitions to `Phi(D)`.  Precompose `Phi(D)` with the final title
token transition.  Replacing that one title update by the result gives exactly
the same state as processing the title and all document transitions before
`Q`.  This remains true for multiple documents because the operator family is
closed and ordered composition is associative.

### Strict homogeneous separation from T29

T29 is the `g=e` special case.  T30 can set `a=1,b=0` and apply the two
generators `rho=(+1,1)` and `tau=(-1,0)`.  Starting from a vector with two
differently valued adjacent markers, all 218 dihedral products give their
exact group action, including noncommuting order.  T29's diagonal homogeneous
actions cannot faithfully realize these generators in one layer because they
commute.

This proves a real operator-class increase at linear state-update cost.  It
does not prove a language gain.

### Stability and quantization

Permutation preserves every `l_p` norm.  If `||a_t||_infinity <= alpha < 1`
and `||b_t||_infinity <= beta`, then

\[
||h_t||_\infty
\le
\alpha^t||h_0||_\infty+\frac{\beta(1-\alpha^t)}{1-\alpha}.
\]

For an exactly stored group element and quantized `A,B`, one trigger has the
same error bound as T29:

\[
||\widehat h-h||_\infty
\le
\epsilon_A||h||_\infty+\epsilon_B.
\]

Later monomial-affine steps multiply this error by at most their scale norms;
the permutations do not amplify it.  Multiple quantized document triggers
obey the corresponding recursive sum.  No semantic tolerance follows from
this bound.

### Lazy physical representation

Let logical state be `h=P_f u`, where `u` is the physically stored vector and
`f` is an eight-bit orientation.  For a new operator `(g,a,b)`, set `f'=gf`
and update

\[
u'=(P_{f'}^{-1}a)\odot u+P_{f'}^{-1}b.
\]

The physical vector is never shuffled.  The kernel changes an orientation
word and reads the scale/offset arrays in a rotated or reversed order.  It
still reads and writes each coordinate once and performs one multiply and one
add per coordinate.  This is a logical traffic claim; H100 latency and
coalescing remain measured primitives.

## P4. Scope and obstructions

1. `D_109` has only 218 group elements.  It is not an arbitrary permutation
   group and does not inherit general finite-state expressivity for free.
2. The exact separation concerns one-layer homogeneous state actions.  Deep
   diagonal models or nonlinear decoders can trade depth, conditioning, or
   precision for some of the same functions.
3. Token-to-group routing is not solved.  A fixed route may be semantically
   useless; a learned hard route introduces an optimization block that must be
   isolated.
4. The stored payload still contains at most 880 logical bits per document.
   It cannot losslessly store arbitrary prose above that entropy.
5. A 109-coordinate state can still suffer interference.  Moving information
   is not the same as selecting the right information.
6. The document operator is valid only for the declared input-driven head.
   It does not compile a whole nonlinear stacked hybrid.
7. Lazy rotations and reflections preserve nominal work and traffic counts but
   may hurt real GPU memory access or fusion.
8. Reassigning checkpoint entries to routing and records can hurt protected
   language capability.

## P5. Matched controls

The eventual causal screen requires:

1. byte-matched diagonal-affine T29 head (`g=e`);
2. same dihedral head with group routes shuffled across token types;
3. same head with identity document operators;
4. state-only document caches `B_D`, which alias arbitrary-prefix insertion;
5. a permutation-diagonal recurrent control without document compilation;
6. a dense or block-diagonal recurrent control charged for every parameter,
   operation, state byte, and training token;
7. online raw-document execution as the exact reference.

A matched dense recurrent RAM can embed the finite behavior.  The claimed edge
is a structured learnability/serving trade, not a larger finite machine.

## P6. Fixed-resource equation

### Recurrent state

```text
109 BF16 state values = 218 bytes
one uint16 orientation =   2 bytes
total                  = 220 bytes
```

This equals T29's 110 BF16 values.  Candidate and controls reserve the same
220 bytes.

### Per-document record

```text
109 four-bit scales  = 109 cells
109 four-bit offsets = 109 cells
dihedral code        =   2 cells
total                = 220 cells = 880 logical bits
```

For 2,405 documents, payload entries remain `2,405*220=529,100` and the full
prospective write ledger remains 743,670 entries under the 743,734 cap, with
64 spare.

### Recurrent work

The logical update performs 109 multiplies and 109 additions, versus 110 of
each in T29.  The permutation is a lazy index-frame change, not a matrix
multiply.  Token feature projections, hard group routing, table lookup,
decoding, actual bytes transferred, and launch/fusion behavior must be funded
and measured before a same-cost production claim.

## P7. Breakthrough-size path

The proposed qualitative gain is exact noncommutative state tracking and
arbitrary-prefix document composition at the byte and arithmetic envelope of a
diagonal head.  A successful model could write each document once and answer
new ordered combinations without replaying document tokens or retraining
shared weights.

The direction is killed as a breakthrough candidate if any of these occurs:

- it does not exceed the byte-matched diagonal head by at least 30 percentage
  points on a frozen noncommutative state-tracking/composed-document task;
- correct operators do not exceed zero and shuffle controls by at least 30
  points;
- an unquantized natural information oracle is below 80% or lacks a 15-point
  causal document-state gap;
- the physical update increases target-H100 p95 latency, state bytes, or
  checkpoint bytes beyond the frozen envelope;
- protected language quality regresses beyond its frozen noninferiority bound.

## Prior-art boundary

[PD-SSM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/77b830c18836a9b2e1395a4936dd687a-Abstract-Conference.html)
already establishes permutation-diagonal transitions, linear-cost scans, and
strong finite-state tracking.  The complexity separation of
[permutation-diagonal LRNNs](https://arxiv.org/abs/2603.03612) and work on
[block-diagonal state mixing](https://arxiv.org/abs/2602.12021) further close
any broad novelty claim for adding permutations to an SSM.

Composable attention memories also exist in
[Models Take Notes](https://arxiv.org/abs/2606.17107) and
[C2KV](https://arxiv.org/abs/2607.17715), while
[sparse recurrent prefix caching](https://arxiv.org/abs/2605.05219) stores exact
recurrent checkpoints.

T30 therefore claims no novelty for permutation-diagonal recurrence,
associative scan, recurrent caching, or composable cache representations.  The
unverified conjunction is the eight-bit `D_109` controller, closed affine
offset operator, exact 220-cell document serialization, title substitution,
lazy 220-byte state implementation, and a large causal natural-language gain.

## Block microbench contract

| block | explainable invariant | first kill test |
|---|---|---|
| dihedral controller | two integers encode all rotations/reflections and compose exactly | exhaustive 218-element closure, inverse, and ordered composition |
| monomial-affine fold | one triple equals every sequential state update | independent sequential, left-fold, balanced-fold, and black-box reconstruction |
| noncommutative routing | order changes the homogeneous action | explicit `rho,tau` witnesses and all group commutators |
| title substitution | one operator equals title plus raw document for every incoming state/suffix | exhaustive small sequences, repeated titles, wrong/reversed documents |
| lazy frame | logical permutation equals a pointer/orientation update | materialized-versus-lazy exhaustive controller/state comparison |
| codec | group code is exact and A/B errors obey the norm bound | all codes, BF16 levels, clipping, multi-trigger accumulation |
| ledger | state, record, and arithmetic caps are exact | independent byte/cell/operation accounting |
| learnability | routing is used rather than ignored | multi-seed noncommutative composed-document screen with diagonal/zero/shuffle controls |
| natural information | raw one-pass states contain answer-relevant evidence | frozen unquantized online/compiled oracle before quantization or physical work |
| physical operator | permutation is actually free enough | target-H100 p50/p95, traffic, workspace, occupancy, and replay equality |

## Paper decision

T30 passed its frozen
[exact CPU reference gate](dihedral-monomial-prefix-t30-stage0-decision.md).  It
has a strict homogeneous separation, a closed operator, an unchanged
record/state ledger, and explicit prior-art limits.  The first separately
frozen learnability screen was
[withdrawn before execution](dihedral-monomial-prefix-t30-learnability-pre-run-audit.md)
because its document arm compared against only an additive T29 subcase and did
not exercise noncommutativity.  Hard routing on raw prose, natural usefulness,
quantized model behavior, and physical cost remain unproved.  No further GPU
run is admitted until the natural-information bridge survives a full
diagonal-affine matched-control construction audit.

That subsequent
[natural-information bridge audit](dihedral-monomial-prefix-t30-natural-bridge-audit.md)
failed on paper.  Once a semantic key is known, factual overwrite is already a
diagonal-affine operation; T30 does not construct the key or relational join,
does not add storage entropy, and overlaps prior permutation-diagonal
state-tracking work.  The exact operator is retained, but T30 is closed as the
active breakthrough direction without a training run.
