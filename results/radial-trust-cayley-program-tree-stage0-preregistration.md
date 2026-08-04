# Radial-trust Cayley program tree Stage-0 preregistration

## Frozen hypothesis

The v1 program tree failed because recursive quadratic updates were unbounded.
Apply the basis-invariant map

`S(d) = d / sqrt(1 + mean(d^2))`

to every raw edge update before adding it to the hidden trajectory.  This fixes
`tau=1` and retains the v1 `1/sqrt(depth)` residual scale.  It changes neither
payloads, routes, topology, nor parameter count.

Frozen source hashes:

- Module: `cf64ab47717ec0cb8a8720fbd768adf48ca9785f0169253966f5a1539ff70226`
- Stage-0 executable:
  `ae85b5072128e2f508b7113b1849e45c1bc3a192950522bd5fb6dbfda4231889`
- Tests: `d1d5dc22b069f569ea6958c610df272b15dde2a743aefa5d7c28a353b84c4fa5`
- Theorem: `0bb93ce6760f87a9e9976b9d586e4692843423f695a95c43f67d3cadee9c40af`

## Required gates

All gates must pass:

1. Candidate and v1 have exactly the same parameter count.
2. Candidate and v1 expose exactly the same checkpoint keys; the trust region
   adds no learned state or persistent buffer.
3. Random raw updates with RMS far above one map to RMS strictly below one.
4. The trust-region Jacobian at a random finite point has full rank.
5. Both Gaussian RMS-one input and the adversarial RMS-one vector
   `(sqrt(D), 0, ..., 0)` remain finite through all nine depths.
6. Every trusted update has RMS below one and every hidden state respects the
   theorem bound `RMS(h_k) <= RMS(h_0) + k/sqrt(L)`.
7. The unchanged ideal active-FFN ratio remains below 8.3%.

Passing Stage 0 establishes stability, full-rank radial flow, and accounting
integrity only.  It does not establish useful LM capability.  Failure closes
this exact successor.  Passing authorizes one frozen GPU smoke test.
