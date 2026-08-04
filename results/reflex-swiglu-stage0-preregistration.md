# Reflex-SwiGLU stage-0 preregistration

Status: frozen before execution.

Scope: exact endpoint, trainability at that endpoint, artificial degree witness, and resource ledger only.

## Candidate

```text
g  = G x
u  = U x
z0 = SiLU(g) * u
r  = clip(z0, -1, 1)
z1 = SiLU(g + alpha * r) * u
y  = V z1
```

`alpha in R^M` is learned per hidden feature, initialized to zero, and constrained to `[-2,2]` during training. Inference stores the effective bounded value.

## Frozen G0 gates

1. `alpha=0` contains ordinary SwiGLU with float64 maximum absolute error at most `1e-12`.
2. The analytic derivative with respect to `alpha` is nonzero at the zero endpoint for a frozen nondegenerate point, so the new parameter can leave the endpoint on the first optimizer step.
3. Shared `G,U,V` derivatives at `alpha=0` equal ordinary SwiGLU derivatives algebraically.
4. In the identity-clip, square-activation analogue, the construction has a nonzero degree-7 term while any ordinary square-gated FFN has degree at most 3. This is not evidence for deployed SiLU superiority.
5. At `D=4096,M=14336`, the same-width parameter overhead `M/(3DM)=1/(3D)` is below `0.009%`; the exact equal-parameter width loses at most 2 features.

Passing G0 authorizes a fused H100 latency gate, not training or capability claims.

