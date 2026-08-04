# Associative nonlinear reduction — Stage 0 preregistration

## Fixed question

Can the real local capacity gain of nonlinear K-reduction be retained while
removing the serial proposal's arbitrary block order and unstable growing-prefix
products?

## Frozen algebra

Represent each gate/up block partial as a dual number `w=g+eps*u`, with
`eps^2=0`, and define

```
w plus_a v = w + v + a*w*v
```

or, in two real M-wide states,

```
p' = p + g + a*p*g
q' = q + u + a*(p*u + q*g)
```

Because `1+a*(w plus_a v)=(1+a*w)(1+a*v)`, the operation is commutative and
associative.  At `a=0` it is ordinary addition and the final
`SiLU(p)*q` is exact SwiGLU.

One selected post-attention RMSNorm slot is a carrier, not a forward scale.
At conversion its old scale is folded into the matching gate/up columns;
normalization uses one at that coordinate forever.  The stored slot supplies
`a=0.25*tanh(carrier-1)`.  Baseline and candidate therefore have identical raw
parameters and bytes, while the baseline carrier is function-null.

## Frozen gate

- Double precision, seed 47, `D=8,M=4,O=8,k=4`, 100 Gaussian probes.
- Verify zero-alpha endpoint and non-unit RMSNorm gauge conversion to `1e-12`.
- Verify forward/reverse/left/right-balanced dual reductions and the closed
  form agree to `1e-12` on random dual inputs.
- Compute sampled functional-Jacobian ranks for carrier-null SwiGLU, the serial
  shared-alpha recurrence, and the associative reducer at the common endpoint.
  Both nonlinear reducers must exceed baseline rank; the dual candidate's new
  singular value must exceed rank tolerance by at least `1e6`.
- The carrier JVP must be zero for baseline, nonzero for both candidates, and
  agree with centered finite differences under the same zero/nonzero criteria
  as the preceding Stage 0.
- Under 500,000 samples of four iid `N(0,1/4)` block gates at `a=0.25`, report
  the per-sample worst absolute gain over all up-contribution paths.  Serial
  paths use the implemented clipped prefix factors from their insertion point
  onward; associative paths use every individual-block factor except their own.
  Associative p99 must be below 2.0 and below half the serial p99; sampled
  associative path sign-flip rate must be zero.
- Exact raw parameter and `3DM` dense-weight/MAC ledgers, hashes, and finiteness
  are mandatory.

Passing selects the associative reducer for an H100 register/latency fatal gate.
It does not authorize language training.
