# Cayley program tree — Stage-0 preregistration

Status: **frozen before execution**  
Date: 2026-07-30

Pre-language amendment and rerun: formula-derived affine matchings replace
stored/random permutation tables so the exact resident-byte claim holds.  The
Stage-0 result is rerun from scratch after this change and before any tree GPU
model execution.

Pre-GPU topology integrity amendment: those one-round affine matchings split
width 384 into three disconnected 128-channel components and were rejected
before any tree GPU execution.  The executable topology now uses reversible
quadratic-affine-quadratic permutations.  Every one of its three degree-three
basis graphs must be 3-regular and connected, with exact diameter 10 at width
384 and diameters 16, 15, and 15 at width 4,096.

Runtime-faithfulness amendment: path outputs and Jacobians are evaluated with
the same four-term Neumann polynomial used by the language implementation,
not with the exact Cayley inverse.  The exact orthogonal response remains the
reference; the runtime polynomial must satisfy the preregistered operator
error bound and remain invertible.

## Why this is the same research branch

The sparse-Cayley bank tests whether cheap global noncommutative operations are
learnable with only width-32-equivalent FFN storage.  If it learns but lacks
quality, the missing currency is independent payload capacity.  This successor
keeps the same global operator and spends the full ordinary FFN byte budget on
path-specific diagonal payloads.

It is not admitted to language training by this Stage 0.

## Operator

At tree depth `t`, transform the state into one of three cyclic sparse-Cayley
bases `Q_t`.  A hard binary route selects one of two node edges.  That edge
stores three `D`-vectors `(g,u,d)` and applies

\[
q=Q_t h,
\quad
h' = h + \frac{1}{\sqrt L}Q_t^T
\left[d\odot \operatorname{SiLU}(g\odot q)\odot(u\odot q)\right].
\]

The token then visits the chosen child.  A depth-`L` path therefore executes
`L` nonlinear global residual micro-operations but reads only the payloads on
that path.

Three independent sparse skew generators are cycled across depths.  Using one
basis at every depth would be a fatal coordinate-wise control: after one basis
change all node operations would remain diagonal in the same coordinates.

## Exact width-384 ledger

For `D=384`, ordinary intermediate width `M=1024`, tree depth nine, 511
internal nodes, three Cayley bases, degree three, and four Neumann steps:

- path payloads: `511 * 2 * 3 * 384 = 1,177,344` scalars;
- three sparse generators: `3 * 3 * 384 / 2 = 1,728` scalars;
- one route threshold per node: `511` scalars;
- route coordinates: implicit integer hash, zero stored table bytes;
- sparse matching indices: implicit reversible polynomial formulas, zero
  stored table bytes (runtime index arithmetic/workspace remains excluded);
- total candidate FFN: `1,179,583` scalars;
- ordinary SwiGLU: `3 * 384 * 1024 = 1,179,648` scalars.

The 65-scalar slack is 130 BF16 bytes per layer; charging one eight-byte
formula seed still leaves 122 bytes.  No tensor index table is hidden.

The candidate has 65 fewer learned scalars per layer.  A hard path reads 27D
payload scalars.  Its conservative multiply-like ledger is

\[
9(2\cdot4\cdot3D+4D)=96,768,
\]

or 8.203125% of the ordinary dense MAC count.  This excludes route formation,
indices, additions, SiLU, gathers, serial dependencies, kernel overhead,
backward work, and batch divergence.

## Frozen algebra gate

At `D=8`, tree depth six, three bases, degree three, `alpha=.25`, and seed 31:

1. every exact Cayley basis is dense and orthogonal to `1e-12`;
2. every runtime Neumann basis is dense at this width, within
   `2*alpha^5/(1-alpha)` of its exact response, and has minimum singular value
   at least one minus that bound;
3. all 64 runtime forced paths produce distinct outputs at ten decimals;
4. every runtime path Jacobian at the frozen probe is dense and rank eight;
5. the 64 flattened runtime path Jacobians span all 64 dimensions of `M_8(R)` at
   relative tolerance `1e-9`;
6. finite-difference and analytic path Jacobians agree to `1e-5` relative;
7. the exact width-384 parameter ledger is no larger than SwiGLU and the ideal
   active ratio is below 9%;
8. a same-basis coordinate proof is emitted to prevent falsely attributing
   cross-coordinate composition to repeated diagonal updates.

## Claim boundary

A pass proves that a dense-FFN-sized tree can store many independently learned
node payloads while each selected runtime-polynomial path has dense, full-rank nonlinear
Jacobians spanning the whole toy matrix space at near-linear ideal active work.

It does not prove route learnability, useful language specialization,
independent knowledge for every path, kernel speed, or novelty over the broad
families of hierarchical MoE, neural decision trees, implicit GNNs, and
structured orthogonal networks.  Those are mandatory controls before a claim.
