# T39 residual decision DAG — paper no-go

Date: 2026-08-01  
Status: **CLOSED BEFORE IMPLEMENTATION; NO CPU, MODEL, OR GPU RUN ADMITTED**

## Proposed direction

Train an ordinary raw-prose model, discretize selected hidden features, compile
the model's persistent next-token residual into a shared reduced ordered
decision DAG, replace an equal-byte FFN or MoE allocation by that DAG, and
execute one root-to-terminal path per token.

The attraction was real: different contexts could share suffix subprograms,
only one path would be active, and an offline compiler could spend more
training work without increasing serving cost.  The claim does not survive the
matched-control and goal-alignment audit.

## Exact served function

Let

\[
b(h)\in\{0,1\}^k
\]

be the discretized hidden code, let `tau(b(h))` be the reached terminal, and
let `v_j in Q^r` be terminal payload `j`.  The proposed update is

\[
F(h)=h+v_{\tau(b(h))}.
\]

This is exactly all of the following:

1. a hard hierarchical MoE whose experts are constant vectors;
2. a Fast Feedforward Network whose leaf subnetworks are constants;
3. a conditional-memory table keyed by a hierarchical router; and
4. a legal post-T35 raw-plane control carrying the identical DAG and reader.

The fourth construction is fatal.  Give the strongest control the compiled
DAG, comparisons, successors, and terminal table.  Its instruction trace and
output equal the candidate's at every input and its resource vector is
identical.  The post-T35 artifact-and-reader absorption theorem therefore
applies directly.  Suffix sharing can beat an *unrolled* tree, but a strongest
control is allowed to share the same suffixes.

This collision is not merely terminological.  Binary-tree conditional
execution is already the defining mechanism of
[Fast Feedforward Networks](https://arxiv.org/abs/2308.14711), differentiable
sparse tree layers already provide true conditional forward and backward
execution in the
[Tree Ensemble Layer](https://arxiv.org/abs/2002.07772), and current
[Engram](https://arxiv.org/abs/2601.07372) supplies the relevant
key-to-vector conditional-memory control at iso-parameter and iso-active-FLOP
allocation.

## Exact resource floor

Let the reachable DAG contain `n` internal nodes, `L` distinct terminals,
`N=n+L`, payload precision `p` bits, and maximum path depth `H`.

Even before alignment, allocator, and integrity metadata, it needs

\[
B_{payload}=Lrp
\]

payload bits and at least

\[
B_{child}=2n\lceil\log_2N\rceil
\]

child-index bits.  If variable levels are not implicit, add

\[
n\lceil\log_2 k\rceil
\]

variable tags, plus thresholds or quantizer state.  A reachable binary DAG
satisfies `n >= L-1`.  Reaching `L` different outputs requires

\[
H\ge\lceil\log_2L\rceil,
\]

so one served token pays at least `H` dependent comparisons and successor
loads, one `rp`-bit terminal read, and `r` residual additions.  Oblique tests
add either `kD` precomputation or `HD` path work.

For the frozen width-384, width-1,024 BF16 SwiGLU,

\[
3\cdot384\cdot1024\cdot2=2,359,296\text{ bytes}.
\]

Ignoring every router byte, this funds at most 3,072 width-384 terminal
vectors, but only 24 full 49,152-logit terminal vectors.  A shared output basis
reduces payload bytes only by imposing its rank.  If a hidden payload is
expanded by the tied unembedding, the matrix of representable logit residuals
has rank at most 384; an arbitrary exact residual can require vocabulary rank.
A token ID plus amplitude is cheaper, but it is not a general next-token
residual.

## What ROBDD reduction does not prove

Reduced ordered binary decision diagrams are canonical, and minimal in the
relevant reduced representation class, for an exact Boolean function under
one fixed variable order.  Vector-valued residuals require a multi-terminal
BDD or algebraic decision diagram, not an ordinary ROBDD.

The reduction proves none of the needed comparisons:

- it is not minimal across variable orders;
- it is not minimal among free branching programs, arithmetic circuits,
  neural networks, soft or oblique routers, MoEs, or approximate functions;
- it supplies no sample-complexity or optimization theorem;
- it says nothing about GPU divergence, gather efficiency, or latency; and
- on observed contexts it is minimal only for the chosen completion of unseen
  bit patterns.

A concrete order obstruction is equality on two `m`-bit strings.  Under the
order `x_1,...,x_m,y_1,...,y_m`, all `2^m` assignments to `x` induce distinct
residual functions of `y`, so the OBDD needs width at least `2^m` after the
`x` prefix.  An interleaved order needs only linear size, and a general circuit
computes equality with `O(m)` comparisons and conjunctions.  Fixed-order
minimality is therefore not general circuit efficiency.  The classical
boundary is stated in Bryant's
[original OBDD paper](https://www.cs.cmu.edu/~bryant/pubdir/ieeetc86.pdf).

## Two additional definition failures

1. The DAG is fitted to residuals produced by a baseline hidden-state
   distribution.  Removing the FFN that produced those states changes the
   rollout distribution; retaining it violates the equal-byte exchange.  The
   proposed exact compilation is therefore actually approximate distillation
   with rollout error.
2. One discretized code generally occurs with multiple next tokens and
   residuals.  A single persistent terminal has no exact target until an
   aggregation rule is declared, and the within-cell conditional variance is
   irreducible collision error for that code.

## Escape-class audit

### E1: no strict online-resource separation

The DAG may read fewer values than a dense FFN, but FFF, a hierarchical MoE,
or conditional memory can execute the identical DAG.  There is no lower bound
against the strongest legal simulation and no new hardware instruction.

### E3: no finite-budget learnability separation

Offline residual fitting is a training algorithm available to the matched
control.  ROBDD reduction gives no sample- or optimization-complexity
separation.  Comparing against a gradient-only control would repeat the
post-T35 weakened-control error.

An E4 approximate compression study could still measure a frozen error/latency
trade against identical-DAG, FFF, hierarchical-MoE, and Engram controls.  That
would be a systems/distillation study, not the active capability objective.

## Goal alignment

The compiled object is next-token behavior over a hidden quantization.  It is
not an entity/relation/value quotient, an exact raw record, or a bridge from a
natural question to stored evidence.  T20 already showed that dominant
next-token residual structure need not carry relation truth, and the feature-
DAG screen already showed that an efficient structured circuit can fail after
a small generic rotated component is added.

Therefore T39 does not advance the required chain

```text
raw prose -> autonomous useful structure -> digital plane
          -> held-out knowledge/reasoning -> identical serving cost
```

and is closed before implementation.  No residual-DAG CPU reference, model
run, GPU work, or rental is admitted.
