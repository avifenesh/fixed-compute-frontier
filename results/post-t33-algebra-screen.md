# Post-T33 algebra screen — paper decisions before execution

Date: 2026-07-31  
Status: **no executable candidate admitted**

## Search constraint inherited from T33

The raw plane plus a direct sentence index already reaches 99.54% support-span
coverage and 99.04% two-document coverage in 669,120 bytes.  Another offline
span representation cannot create a capability edge.  The remaining target is
query-conditioned computation that is strictly more useful per complete
served byte and operation than a matched dense, sparse, recurrent, and indexed
control.

## A. Context-addressed physical weight reuse

### Clean algebra

For a context code `c`, use cheap signed/permutation actions around one shared
matrix:

`y = D_out(c) P_out(c) W P_in(c) D_in(c) x`.

This creates many logical linear maps from one physical `W`; the identity code
contains the ordinary shared map.  A `log2(K)`-bit router and procedurally
generated Hadamard/Rademacher codes avoid a `K*d` code table.

On a restricted task family whose target maps lie in the signed-permutation
orbit of one matrix, the construction represents `K` contextual maps exactly
with one matrix and `O(d log K)` router state.  A context-blind single linear
map cannot represent two different outputs for the same `x`.

### Why this does not pass

The useful statement is restricted to an orbit that shares singular values
and parameter magnitudes.  It does not store `K` arbitrary learned matrices;
the independent degrees of freedom remain those of `W`.  A matched SwiGLU or
modulated/shared-expert control already has multiplicative context gating, so
the context-blind linear lower bound is not the relevant control.

The main mechanism is established prior art:

- [Superposition of many models into one](https://openreview.net/pdf?id=SJewgBBlLH)
  binds model parameters with task context;
- [Parameter-Efficient Mixture-of-Experts](https://aclanthology.org/2022.coling-1.288/)
  shares an MPO core and specializes experts with auxiliary tensors;
- [MoFME](https://arxiv.org/abs/2312.16610) instantiates multiple logical
  experts through feature-wise modulation of one shared block;
- [ModularMoE](https://aclanthology.org/2026.findings-acl.174/) explicitly
  builds experts from combinations of shared modules.

Decision: **closed as a standalone novelty/capability claim; no microbench**.

## B. Tensor-sketched implicit conjunction features

### Clean algebra

The degree-`p` tensor feature `x^(tensor p)` contains `d^p` monomials.
TensorSketch maps it to `m` coordinates and approximately preserves polynomial
kernel inner products with error controlled by sketch width and failure
probability.  FFT-based convolution can construct the sketch without an
explicit `d^p` tensor.  This is a valid way to expose combinatorial
interactions without FFN up/down matrices.

### Why this does not pass

An `m`-coordinate sketch still exposes only `m` trainable readout degrees of
freedom.  It approximates a fixed kernel; it does not store `d^p` independently
learned features or prove higher natural-language capability than a matched
adaptive SwiGLU.  The strongest valid claim is an operation-count reduction
for a kernelized model.

That claim is already represented by
[PolySketchFormer](https://proceedings.mlr.press/v235/kacham24a.html), which
uses polynomial sketches with approximation guarantees for linear-time
attention and reports speed without observed quality loss.  General fast
polynomial-kernel sketching is also established by
[Song et al.](https://proceedings.mlr.press/v139/song21c.html).

Decision: **retained as an efficiency/control primitive, rejected as a smarter
FFN proof; no microbench**.

## C. Budget-conserving asynchronous/fixed-point reasoning

### Clean algebra

For a contractive recurrent block `h_(t+1)=F(h_t,x)`, plain iteration converges
linearly.  Anderson/Chebyshev or Krylov extrapolation can reduce fixed-point
error faster for the same number of expensive `F` evaluations, with small
history-vector work.  Separately, a fixed total token-update budget can be
allocated unevenly so a short reasoning chain receives more sequential depth
than uniformly updating every token.

### Why this does not yet pass

The convergence theorem proves a better solver for an already meaningful,
stable fixed point.  It does not prove that LM loss trains `F` to have a useful
fixed point, that extrapolated hidden states remain on the learned manifold,
or that the extra reductions/history traffic are free.  A matched recurrent
model can also be trained with the same solver.

The obvious architecture is a crowded active line:

- [Inner Thinking Transformer](https://aclanthology.org/2025.acl-long.1369/)
  performs adaptive token routing and recurrent residual thinking;
- [Mixture-of-Recursions](https://openreview.net/forum?id=YtQtGsNr64) allocates
  token-level recursive depth;
- [Predictive Scheduling](https://arxiv.org/abs/2602.01237) reports a 7.9-point
  reasoning gain at identical token cost by redistributing budget;
- [Fixed-Point Reasoners](https://openreview.net/pdf/a3ff2634e3fdaea1aa9ccf8967f9da0244a9760e.pdf)
  target stable adaptive looped reasoning;
- [DEQ Jacobian regularization](https://proceedings.mlr.press/v139/bai21b.html)
  already uses equilibrium solvers including Anderson acceleration;
- [Anderson acceleration theory](https://arxiv.org/abs/1810.08455) gives the
  relevant first-order convergence improvement under fixed-point assumptions.

Decision: **the numerical-solver theorem is valid but not a novel capability
bridge.  No implementation until a stricter language-specific separation from
these controls is proved**.

## Current research state

No post-T33 direction passes the paper gate.  This is intentional: an existing
operator with a renamed context code, a kernel approximation, or adaptive
depth is not the requested breakthrough.

The next candidate must introduce a new invariant or resource trade, not just
a new module.  In particular, it must explain which natural-language function
class is compressed or solved more efficiently, how the query supplies the
needed conditional information, and why the strongest matched existing
architecture cannot realize the same function at the same complete serving
cost.
