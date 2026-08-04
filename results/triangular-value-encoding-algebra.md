# Triangular value encoding: the algebraic core

## The key shift

A dense projection is expensive because it implements an arbitrary learned
linear map. Replacing it with a cheaper structured map usually pays for speed by
losing functions. TVE takes a different path: retain the transformer's dense
matmuls, choose an equivalent gauge in which some physical weight coordinates
are exactly zero, and reuse those coordinates to fund a cheap nonlinear map on
the value path.

For one coordinate block, let `C` be strictly lower triangular and let `phi`
act coordinatewise. TVE writes

`y_j = v_j + sum_{i<j} C[j,i] phi(v_i)`.

The tested feature is `phi(x) = x |x|`. Coefficients are the strict-lower
entries of a representative output-projection block divided by tau. They are
not separately stored.

## Theorem: the cache transform is information-preserving in exact arithmetic

The Jacobian is triangular:

`d y_j / d v_j = 1`,

`d y_j / d v_i = C[j,i] phi'(v_i)` for `i < j`, and zero for `i > j`.

Therefore every diagonal entry is one and

`det(J) = 1`.

More strongly, the inverse exists constructively for any well-defined `phi`:

`v_0 = y_0`,

`v_j = y_j - sum_{i<j} C[j,i] phi(v_i)`.

Thus the real-arithmetic transform is a bijective, volume-preserving additive
triangular flow. It changes the feature coordinates written to KV cache without
compressing or discarding the original value information. BF16 rounding weakens
literal invertibility, but it does not require extra cache values.

## Why the ordinary transformer is contained

Value/output attention has an orthogonal gauge: rotating a KV head's value
coordinates and applying the inverse rotation to the corresponding output
blocks preserves the represented function. RQ gauge fixing can choose an
equivalent representative whose selected output block is upper triangular.
Its strict-lower physical entries are therefore exactly zero.

Setting all TVE coefficients to zero recovers that function-preserving
gauge-fixed transformer. Consequently the standard transformer's function class
is contained in the TVE parameterization in exact arithmetic. The candidate can
ignore the new path; exploiting it is optional.

After learning, the reused physical entries participate both in the ordinary O
projection and in the nonlinear value transform. This is deliberate weight
reuse, not free independent parameters. It can create useful coupling, but it
does not grant two independent matrices.

## What extra algebra is purchased

At fixed attention weights, an ordinary value/output path is linear in `v`.
TVE adds signed-quadratic terms before values are cached and mixed across
tokens. Each later layer can linearly recombine those features and apply another
triangular lift. The useful resource is therefore not more dense width but more
nonlinear algebraic depth per stored parameter and per dense matmul.

For block size `B`, KV heads `H_kv`, and head dimension `d_h`, the logical
coefficient MAC count per token and layer is

`H_kv * (d_h / B) * B(B-1)/2 = H_kv * d_h * (B-1)/2`.

At SmolLM2-360M geometry (`H_kv=5`, `d_h=64`, `B=16`), this is 2,400 logical
MACs per token and layer, versus the dense QKV/FFN work at hidden 960 and
intermediate 2,560. Tensor-core padding executes 5,120 MAC slots, still about
0.05 percent of dense training MACs. There are no new model values, buffers,
runtime metadata bits, or KV coordinates.

## The remaining head-local budget

The orthogonal group in `d_h` dimensions has `d_h(d_h-1)/2` continuous degrees
of freedom. This exactly matches the number of strict-lower zeros produced by a
full-head RQ representative. At head dimension 64, that is 2,016 slots per KV
head. Block-16 TVE uses only four within-block triangles, or 480 slots per KV
head; 76.19 percent of the stable orthogonal gauge budget is still unused.

A full-head triangular flow would use all 2,016 slots while retaining the same
bijection and standard-model containment proof. At 360M geometry it costs
10,080 logical coefficient MACs per token and layer, still tiny beside dense
projections. The frozen gauge-budget screen tests whether this additional
feature algebra helps enough to justify its larger local kernel.

The larger general linear gauge could force an invertible representative block
all the way to identity and expose more zero coordinates, but its inverse can be
ill-conditioned and amplify value activations. Orthogonal RQ is therefore the
current stability boundary; nonorthogonal gauge funding is a separate, gated
refinement rather than a free extension.

## Why this is not yet a breakthrough claim

- Elementwise AdamW is not invariant to the orthogonal gauge change, so raw and
  gauge-canonical training can drift even from function-equivalent starts.
- The 100M-token three-seed run showed a persistent TVE effect but formally
  failed one canonical-control validity gate.
- The current custom backward needs its revised numerical/operator gate and
  full-model trajectory equivalence.
- H100 v6 covers QKV plus cache-write epilogue, not the fastest native full
  attention path.
- Larger-model, second-domain, and downstream capability evidence do not yet
  exist.

The research direction survives because its central guarantees are structural:
standard-function containment, exact-arithmetic information preservation, zero
new stored model state, unchanged KV size, and a very small arithmetic lift.
The remaining work is empirical falsification of usefulness and systems cost,
not invention of another unconstrained architecture.
