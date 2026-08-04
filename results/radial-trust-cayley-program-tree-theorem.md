# Radial-trust Cayley program tree: stability theorem

Let the raw conditional edge update at depth `k` be `d_k in R^D`.  Replace the
unbounded v1 recurrence with

```
S_tau(d) = d / sqrt(1 + ||d||_2^2 / (tau^2 D))
h_(k+1) = h_k + S_tau(d_k) / sqrt(L).
```

## Stability

`RMS(S_tau(d)) < tau` for every finite `d`.  Therefore the triangle inequality
gives

```
RMS(h_L) <= RMS(h_0) + sqrt(L) * tau.
```

For the frozen `L=9`, `tau=1`, RMS-normalized transformer input, every internal
state has RMS below 4 regardless of payload values or route choices.  This
removes the repeated-quadratic explosion found in v1.

## No rank bottleneck

Write `a = 1/(tau^2 D)` and
`c = (1 + a ||d||^2)^(-1/2)`.  The Jacobian is

```
J_S(d) = c I - a c^3 d d^T.
```

Its `D-1` tangential eigenvalues are `c`; its radial eigenvalue is `c^3`.
All are positive for finite `d`, so `rank(J_S)=D`.  The trust region is a
smooth radial diffeomorphism from `R^D` to the open radius-`tau*sqrt(D)` ball,
not a low-rank projection or hard clip.

## Resource effect

The map adds no checkpoint parameters or persistent buffers.  It adds one
per-token reduction and scalar square root per active tree edge, so active work
remains `O(D log D)`.  Resident parameters, hard-path payload, and the 8.203125%
ideal active-FFN ratio are unchanged from v1.

This theorem establishes stability and preserves the scaling separation.  It
does not establish language-model capability; that remains an empirical gate.
