# T72 finite mechanism grammar — affine calibration and no-go

Date: 2026-08-02  
Status: **INDEPENDENTLY REVISED; AFFINE V0 RETAINED AS CALIBRATION ONLY; NO CODE OR RUN**

## 0. Purpose and disposition

T71 changes the project from selecting a memory module to testing whether a
model can become a substantially better learner. T72 drafted the smallest
non-prose world in which acquisition, active experiment choice, reusable-rule
induction, long execution, revision, and retention might be separated.

Independent review found a decisive no-go: for the full affine class, a fixed
basis schedule is already minimax. Adaptivity buys exactly zero probes. Affine
v0 is therefore retained only as a calibration specification for persistence,
leakage, system identification, state accounting, and horizon execution. It
cannot admit a neural run or natural pilot.

## 1. Correct affine world

Fix a prime field `F_p` and vector width `d`. A latent program is a scalar DAG:
leaves are coordinates `input_i`; internal nodes are typed scalar constants,
addition, and scalar multiplication; and `d` ordered roots form

\[
P:\mathbb F_p^d\to\mathbb F_p^d,
\qquad P(x)=Ax+b.
\]

This scalar-node/output-root definition is required; vector-valued `add` and
`scale` alone do not generate an arbitrary matrix.

At the start of a lifetime, one shared affine-monomial state encoding `S`
permutes coordinates and applies an invertible affine value map per coordinate.
The observed transition is the conjugate

\[
\widetilde P=S\circ P\circ S^{-1},
\]

so

\[
\widetilde P^h=S\circ P^h\circ S^{-1}=(\widetilde P)^h.
\]

Independent input/output encodings are forbidden: without a charged bridge,
their one-step map does not determine the intended rollout. Program-plus-binding
worlds are enumerated; a fixed hidden AST-size bound is not claimed to remain
closed under rendering.

The interface is:

```text
PROBE x       -> FEEDBACK P_t(x)
PREDICT x     -> private scored P_t(x)
ROLLOUT x,h   -> private scored static P_t^h(x)
```

No hidden change occurs inside one scored rollout. A complete lifetime world
includes the initial map, change time, and change kernel.

## 2. T72.1 — observable and scoring equivalence

Two worlds are observationally equivalent when every legal adaptive probe
policy induces the same distribution over learner-visible transcripts:

\[
w\sim_{obs}w'
\Longleftrightarrow
\forall\pi,\quad
\mathcal L(\tau_{obs}\mid w,\pi)=
\mathcal L(\tau_{obs}\mid w',\pi).
\]

Define `w ~score w'` when every legal private scored query has the same target.
The protocol is learnable only if exhaustive enumeration proves

\[
w\sim_{obs}w'\Longrightarrow w\sim_{score}w'.
\]

Private targets cannot appear in the observable relation. Exact hidden-AST
recovery is not scored unless syntax is canonical and identifiable; otherwise
behaviorally equivalent programs receive identical score.

## 3. T72.2 — sufficient-state information floor

Let `H` be uniform over `N` score-distinct quotient classes after conditioning
on all meta-training information and the complete pre-erasure transcript. Let
`M` be the sole cross-boundary state through a discrete or explicitly quantized
`B`-bit channel. If a decoder identifies `H` from `M` with error `p_e`, Fano's
inequality and `I(H;M)<=H(M)<=B` give

\[
B\ge
\log_2N-h_2(p_e)-p_e\log_2(N-1).
\]

For exact class identification, `B>=ceil(log2 N)`. Prediction performance
implies this floor only if the frozen query battery decodes the class;
otherwise a rate-distortion bound for the declared prediction loss is needed.

Conversely, after exact identification, a memory with the shared codebook can
store the class index in `ceil(log2 N)` bits. A learned latent state therefore
has no fundamental bit advantage over an optimal class-index code. Legitimate
edges are fewer interactions, lower update/use work, better transfer of the
identification algorithm, longer execution, more selective revision, or better
finite-data robustness at the same complete resource point.

## 4. T72.3 — generic query bounds and affine no-go

For finite version space `V` and probe `a`, define

\[
V_{a,y}=\{h\in V:h(a)=y\},
\quad
\alpha(V)=\min_a\max_y\frac{|V_{a,y}|}{|V|},
\quad
\bar\alpha=
\max_{V\text{ reachable},|V|>1}\alpha(V).
\]

If `bar_alpha<1`, greedy splitting identifies a noiseless class in at most

\[
\left\lceil\frac{\log N}{\log(1/\bar\alpha)}\right\rceil
\]

probes. If each probe has at most `K` outcomes, every adaptive decision tree
needs at least `ceil(log_K N)` probes. These are generic bounds; exhaustive
minimax dynamic programming provides the exact small-world counts. The
nonadaptive control is the optimal set or sequence fixed before outcomes.

For the complete affine class,

\[
|H|=p^{d(d+1)}.
\]

Each vector probe has at most `p^d` outcomes, so exact identification needs at
least `d+1` probes. The fixed schedule

\[
0,e_1,\ldots,e_d
\]

recovers `b` and every column of `A` in exactly `d+1` probes. Thus adaptive and
nonadaptive minimax counts are equal. The active-learning ratio is `1`, not
`2`. A restricted AST subset may differ only if exact decision trees prove it.

## 5. Representation and execution ledgers

An arbitrary truth table for a map `F_p^d -> F_p^d` contains

\[
d p^d\log_2p
\]

output bits. A scalar DAG with `n` nodes, schema library size `|L|`, fan-in at
most `k`, and at most `r` field parameters per node has the loose program bound

\[
B_{DAG}\lesssim n[
\lceil\log_2|L|\rceil+
k\lceil\log_2(n+d)\rceil+
r\lceil\log_2p\rceil]
\]

plus a prefix/topological encoding, exact arity/type tags, and

\[
B_{out}\le d\lceil\log_2(n+d)\rceil
\]

for output roots. Parameterized schema names and their values are not counted
twice. A loose binding charge is

\[
B_{bind}\le
\lceil\log_2(d!)\rceil+d[
\lceil\log_2(p-1)\rceil+
\lceil\log_2p\rceil].
\]

If state stores the canonical observed affine map rather than the hidden DAG,
charge `d(d+1)ceil(log2 p)` bits and do not claim AST recovery. All tiny-world
reports must give numerical totals; asymptotics against an unrestricted truth
table do not establish a practical compression win.

For a static affine map,

\[
P^h(x)=A^h x+\sum_{i=0}^{h-1}A^i b.
\]

Augmenting `[x;1]` and exponentiating a `(d+1)x(d+1)` matrix evaluates this in
`O(d^3 log h)` field operations. This exact control separates identifying the
rule from imitating short rollout traces and prevents credit for rediscovering
known affine algebra.

## 6. Revision and retention calibration

For pre/post maps define

\[
\Delta(x)=(A'-A)x+(b'-b).
\]

The change family must be frozen so both of these sets have preregistered
minimum size:

\[
Q_{changed}=\{x:\Delta(x)\ne0\},
\qquad
Q_{protected}=\{x:\Delta(x)=0\}.
\]

A pure nonzero bias shift can change every full-vector query and is rejected,
or evaluated under a separately frozen coordinate-level contract. Measure
probes/regret to recover changed behavior, maximum protected loss, calibration
around the first contradiction, and recovery when the old map is restored.

There is no `Q_alias`: if an internal reparameterization is behaviorally
equivalent, Section 2 proves the interface cannot detect it. Paired-world state
swaps may compare worlds differing in the declared behavioral mechanism; an
opaque state has no intrinsic AST node to swap.

## 7. What a future CPU calibration could do

After a separate freeze, an exhaustive CPU enumerator could:

1. enumerate program-plus-binding worlds and both equivalence relations;
2. prove observable equivalence refines score equivalence;
3. report the fixed-basis optimum and exact restricted-class decision trees;
4. compute conditional sufficient-state, transcript, DAG, binding, canonical
   affine, and memorized-policy bits;
5. construct changes with nonempty frozen changed/protected sets; and
6. verify that longer static horizons add execution depth but no new
   identification information.

That would be a theorem/calibration microbenchmark only. It is not admitted in
this round and cannot unlock the T71 neural objective ladder.

## 8. Decision

T72 retains three useful boundaries:

1. latent state cannot beat an optimal sufficient-state code in information
   bits;
2. scoring must respect what is identifiable from learner-visible evidence;
   and
3. affine identification cannot be called active intelligence when a fixed
   basis schedule is minimax.

Every affine composition collapses behaviorally to one `(A,b)`. Larger AST and
graph-shape holdouts do not prove reuse of internal composition. A successor
grammar must add a proved adaptive/nonadaptive gap, guaranteed protected
changes, and non-collapsing held-out compositions before any neural run.

No enumeration, neural code, local GPU use, or rental is admitted by T72.

Independent review:
[`finite-mechanism-grammar-t72-independent-audit.md`](finite-mechanism-grammar-t72-independent-audit.md).
