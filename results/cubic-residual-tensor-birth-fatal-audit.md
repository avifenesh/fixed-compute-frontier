# Cubic residual tensor birth — T0 fatal audit

Status: closed before experiment  
Date: 2026-07-30

## Claim tested

A weak SwiGLU channel has leading contribution

```text
Delta H_n = 0.5 (a^T x_n) (c^T x_n) d.
```

The proposed mechanism formed the residual tensor

```text
T = sum_n e_n tensor x_n tensor x_n
```

and used alternating tensor contractions to birth a useful channel.  The hoped
for distinction was that `T` exposed a quadratic feature that ordinary
first-order training could not see.

## Fatal identity

For loss differential `sum_n e_n^T Delta H_n`, the ordinary gradients are

```text
grad_a L = 0.5 sum_n x_n (e_n^T d) (x_n^T c)
grad_c L = 0.5 sum_n x_n (e_n^T d) (x_n^T a)
grad_d L = 0.5 sum_n e_n (x_n^T a) (x_n^T c).
```

These are exactly the three contractions used by alternating tensor power
iteration.  Normalizing a contraction and assigning it to one factor changes
the metric and takes a finite block-coordinate step; it does not expose a new
training statistic.  From a standard nonzero random initialization, all three
contractions are nonzero almost surely when the tensor has signal.  Adam-like
normalization and matrix-normalized controls therefore attack the claimed
advantage directly.

An exact-zero channel does make every ordinary factor gradient zero, but
reserving zero channels only for the candidate manufactures the separation.
Giving the controls the same channel birth/reinitialization opportunity removes
that argument.

## Prior-art collision

Tensor or moment initialization followed by gradient descent already has
recovery guarantees for one-hidden-layer networks, and tensor-derived neural
objectives have been used to avoid bad local geometry:

- Zhong et al., *Recovery Guarantees for One-hidden-layer Neural Networks*:
  <https://arxiv.org/abs/1706.03175>
- Ge et al., *Learning One-hidden-layer Neural Networks with Landscape
  Design*: <https://arxiv.org/abs/1711.00501>
- Zhang et al., *Learning One-hidden-layer ReLU Networks via Gradient
  Descent*: <https://arxiv.org/abs/1806.07808>

## Decision

Close this lane.  It fails T0 because its supposed information advantage is an
ordinary gradient identity, while its remaining finite-step advantage is
inside the optimizer/initialization control envelope.  No small-system or GPU
run is justified.  The useful retained lesson is to distinguish a genuinely
new statistic from a repackaged contraction of the existing gradient.

