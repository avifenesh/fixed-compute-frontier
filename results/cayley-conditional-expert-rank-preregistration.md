# Cayley conditional expert — local-rank separation preregistration

Status: **frozen before execution**  
Date: 2026-07-30

## Claim

A hard-routed additive vector-memory layer needs at least `D` active experts to
produce a locally full-rank state update, while one Cayley-conjugated diagonal
GLU edge produces a generically full-rank update using only three active
`D`-vectors plus a shared near-linear mixer.

This is a local differential expressivity separation.  It is not a language
quality, independent-memory, or hardware-speed claim.

## Proof target

Inside a region where the selected expert set `S` is fixed, an additive vector
memory has

\[
f(x)=\sum_{i\in S}a_i(x)v_i,
\qquad
J_f=\sum_{i\in S}v_i\nabla a_i(x)^T,
\]

so `rank(J_f) <= |S|`.  A residual layer therefore has
`rank(J-I) <= |S|`.

For one Cayley-diagonal edge,

\[
F(x)=x+Q^T[d\odot\operatorname{SiLU}(g\odot Qx)\odot(u\odot Qx)],
\]

where `Q` is orthogonal.  Its update Jacobian is

\[
J_F-I=Q^T\operatorname{diag}(s(x))Q.
\]

Whenever all coordinates of `s(x)` are nonzero, this has rank `D`.

## Frozen executable gates

At `D=16`, seed 20260730:

1. analytic and finite-difference Jacobians agree within `1e-5` relative;
2. one, four, and eight additive vector experts have update rank no greater
   than one, four, and eight respectively;
3. the Cayley-diagonal update Jacobian has rank 16 and at least 90% numerical
   density;
4. its active learned payload is exactly `3D` scalars;
5. a dense full-rank expert uses `D^2` learned payload scalars and dense MACs;
6. at `D=4096`, degree three and four Neumann terms for both `Q` and `Q^T`,
   the Cayley edge's conservative multiply-like count `28D` is below 1% of
   `D^2`;
7. the result explicitly states that a single fixed basis spans only a
   `D`-dimensional family of symmetric update Jacobians.  Varying bases and
   path composition are required for broader matrix-space coverage.

## Controls and boundary

The rank bound includes input-dependent scalar gates; it does not assume
constant expert weights.  It excludes discontinuous derivatives exactly on
hard-routing boundaries.  Soft routing over all resident experts can have
higher rank, but it reads/evaluates those experts and is not the same active
budget.  A conventional dense MoE expert is full-rank but pays `D^2` payload
and work.
