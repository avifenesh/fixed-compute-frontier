# Full-rank conditional tree scaling law

Status: **algebraic result; language capability and hardware realization are unproved**  
Date: 2026-07-30

## Result

A same-byte hierarchical MoE can reduce active FFN work from quadratic to
approximately `O(D log D)`, but it does so by making each selected expert only
`O(log D)` wide.  Inside a hard-routing region, its residual update Jacobian
therefore has rank at most `O(log D)`.

A balanced tree of Cayley-conjugated diagonal GLU edges has the same
`O(D^2)` resident storage and `O(D log D)` active-work scaling while every
selected edge generically has a rank-`D` update Jacobian.  This is a
strict local algebraic separation from narrow routed SwiGLU experts.  It is not
yet a language-model or speed result.

## Dense baseline

For model width `D` and SwiGLU hidden width `M=cD`, one FFN stores and applies

\[
P_{dense}=3DM=3cD^2
\]

learned scalars/MACs per token.

For a width-`r` SwiGLU expert,

\[
f(x)=W_d[\operatorname{SiLU}(W_gx)\odot W_ux],
\]

the Jacobian of `f` factors through the `r` columns of `W_d`, so

\[
\operatorname{rank}J_f(x)\le r.
\]

Hard top-1 routing is locally constant away from boundaries and does not
change this bound.  Multiplying by a differentiable scalar routing gate can
add at most one rank-one term.

## Equal-storage hierarchical MoE

With `E` equal experts under the same resident budget, each expert has width
approximately `M/E`.  A hierarchical router can choose one expert in
`O(D log E)` work rather than paying a dense `D x E` router.  Its selected
expert costs `O(DM/E)`.

Choosing

\[
E=\Theta(D/\log D)
\]

makes total active work `O(D log D)`, but selected expert width and local
update rank become only `O(log D)`.  The compute reduction and the rank loss
are the same trade.

## Equal-storage full-rank program tree

Give every internal binary-tree node two edges.  Each edge stores gate, up,
and down diagonal `D`-vectors: `3D` scalars.  With `N` internal nodes the tree
stores approximately `6DN` scalars.  Choosing

\[
N\simeq M/2=\Theta(D)
\]

matches the dense FFN's `3DM=Theta(D^2)` resident budget.  A balanced tree has
maximum path length

\[
L=\lceil\log_2(N+1)\rceil=\Theta(\log D).
\]

For a selected edge with an invertible Cayley basis `Q`, use

\[
F(x)=Q^T\left[d\odot\operatorname{SiLU}(g\odot Qx)
                 \odot(u\odot Qx)\right].
\]

Its Jacobian is `Q^T diag(s(x)) Q`.  Outside the zero set of its diagonal
entries it has rank `D`, despite reading only `3D` learned payload scalars.
The exact Cayley response is dense; the implemented fixed-order Neumann
polynomial is sparse at large `D`, although it remains invertible under the
frozen norm bound.  In particular, with `||A||_2 <= alpha = 1/4` and order
`T=4`, its operator error from the exact orthogonal Cayley response is bounded
by

\[
\lVert\widehat Q-Q\rVert_2
\le \frac{2\alpha^{T+1}}{1-\alpha}
=0.0026042,
\]

so `sigma_min(Qhat) >= 0.9973958`.  Therefore
`Qhat^T diag(s) Qhat` is rank `D` whenever every entry of `s` is nonzero.
Global support must come from composing depths and cycling bases, not from
pretending that one truncated edge is dense.
Applying that approximation and its transpose costs `O(D)` per edge for
frozen degree and Neumann order.  A path reads `3DL` payload scalars and
performs `O(DL)=O(D log D)` multiply-like work.

The executable finite-width rank witness is
`results/cayley-conditional-expert-rank.json`; the conditional path witness is
`results/cayley-program-tree-stage0.json`.

## Concrete ledger

The table uses three degree-three bases and four Neumann terms in each
direction.  Incomplete balanced trees are allowed at widths where the exact
resident budget is not one less than a power of two.

| `D` | `M` | internal nodes | max depth | dense FFN MACs | tree active upper bound | ratio | ideal reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 1,024 | 511 | 9 | 1,179,648 | 96,768 | 8.203% | 12.19x |
| 4,096 | 14,336 | 7,166 | 13 | 176,160,768 | 1,490,944 | 0.846% | 118.15x |
| 8,192 | 28,672 | 14,334 | 14 | 704,643,072 | 3,211,264 | 0.456% | 219.43x |

The ratios exclude route/index arithmetic, additions, nonlinearities,
workspace, gathers, divergence, launches, and synchronization.  They are not
wall-clock predictions.

## Hard boundary

This construction does **not** make a generic dense matrix cheap.  One active
path has only `O(D log D)` learned degrees of freedom and cannot cover an open
set in `D^2`-dimensional matrix space.  The stored paths also share prefixes
and bases; they are not independent dense experts.

The only possible win is distributional: language computation must reuse
global bases and compositional path structure.  A real advance therefore
requires all three outcomes:

1. matched language quality against dense SwiGLU and an exact-byte,
   active-work-matched hierarchical MoE;
2. causal use of multiple paths rather than route collapse;
3. a fused implementation with a large end-to-end serving gain after every
   omitted byte and operation is charged.

Without those, the result is only a structural theorem, not a breakthrough.
