# Independent adversarial audit: action Lie algebra variable formation (T58)

Date: 2026-08-01  
Scope: theorem, distinctness, resource, and experiment-gate audit; no experiments run

## Verdict

**RETAIN AS A LOCAL GEOMETRIC-CONTROL DIAGNOSTIC; NO-GO AS A DISTINCT ACQUISITION UPDATER; NO RUN.**

T58's mathematical primitive is real: for smooth full-state coordinates, Lie
brackets are natural under diffeomorphisms; executable inverse-flow loops expose
bracket directions; and independent commuting fields admit local simultaneous
straightening. These facts can diagnose local noncommutativity, accessibility,
and action coupling without semantic labels.

They do **not** establish variable formation or cross-interface learning savings.
An abstract action algebra omits its realization on state, orbit and stabilizer
geometry, drift, passive variables, global topology, control limits, reward,
costs, and observation/readout maps. Algebra matching therefore need not reduce
state representation, readout, reward, or controller acquisition at all.

The exact online-probe-plus-transfer conjunction is not shown by one reviewed
system, but its recovery stage is a known control primitive and its transfer
stage is false without stronger task-family assumptions.

## Geometry and commutator-loop corrections

Let the observation be `y=h(x)`. A pushed-forward vector field on observation
space is well-defined automatically only when `h` is a local diffeomorphism, or
an embedding and the field is considered on `h(M)`. For a noninjective partial
sensor, distinct latent states in one observation fiber can induce different
velocities; no unique `h_*X` exists. Exact coordinate invariance therefore does
not provide raw-observation identifiability.

With T58's convention

\[
[X,Y]=JY\,X-JX\,Y
\]

and its right-to-left composition

\[
C(t)=\Phi_Y^{-t}\circ\Phi_X^{-t}\circ\Phi_Y^t\circ\Phi_X^t,
\]

the leading sign is `+[X,Y]`; it is not ambiguous after conventions are fixed.
Swapping the fields reverses it. Smoothness on a compact local neighborhood is
needed for a uniform `O(t^3)` remainder.

The inverse flow exists mathematically only locally. It is not thereby a legal
action. Dissipation, contact, hysteresis, stochasticity, actuator saturation,
and a control-affine drift can make `Phi^{-t}` unavailable. In

\[
\dot x=f_0(x)+\sum_i u_i X_i(x),
\]

reversing a command does not generally reverse the flow because `f_0` remains.
The standard bracket maneuver is a classical driftless/symmetric-control
construction; higher nested brackets require longer pulse words and expose
signals of progressively higher order. “Nested loops create any missing
direction” must be replaced by “appropriate product formulas approximate
bracket directions under symmetric-control and regularity assumptions.”

## T58.1: Frobenius scope

If `X_1,...,X_k` are pointwise independent throughout a neighborhood and
pairwise commute there, simultaneous straightening gives local coordinates with
`X_i=partial/partial z_i`. This is correct.

It yields only a local orbit chart. The transverse `d-k` coordinates remain
arbitrary, origins may vary with transverse coordinates, and a constant change
of action basis changes the action coordinates. Periodic flows, topology,
singular orbits, rank changes, and holonomy obstruct a global chart. The fields
define action directions, not semantic variables or a complete minimal state.
More general Frobenius involutivity straightens the **distribution**, not the
original noncommuting fields individually.

## T58.2: finite closure and structural decomposition

Finitely many smooth vector fields generally generate an infinite-dimensional
Lie algebra. A finite basis `E_1,...,E_r` requires identities on an open set

\[
[E_i,E_j]=\sum_k c_{ij}^{k}E_k
\]

with **constant scalars** `c_ij^k`. State-dependent coefficients are structure
functions of a frame/distribution, not the claimed finite-dimensional algebra.
Pointwise dependence at finitely many noisy states cannot prove equality of
vector fields, exclude a later independent nested bracket, or establish global
closure. A valid recovery theorem needs a specified finite function class,
open-set excitation, uniform approximation, a smallest-singular-value margin,
and a stopping/error guarantee.

The structural statements need these qualifications:

- If `g=i direct-sum j` as ideals, then `[i,j]=0`; this is algebraic commutation,
  not a product state space, independent reward, or independent controller.
- For a finite-dimensional real Lie algebra, the solvable radical `r` is
  canonical and Levi's theorem gives `g=r semidirect s` for some semisimple
  Levi subalgebra `s`. The Levi factor is not canonical, and `s` is not generally
  an ideal. Its action `s -> Der(r)` is essential data.
- Ideal decompositions can be nonunique, and approximate structure tensors can
  cross algebra-isomorphism boundaries under arbitrarily small perturbations.
- A local Lie algebra does not determine the global group or action. `SO(3)` and
  `SU(2)` share a Lie algebra; quotients, discrete components, boundaries, and
  stabilizers are invisible.
- Cartan decomposition is a different structure: for a semisimple algebra with
  an involution, `g=k+p` with `[k,k]` in `k`, `[k,p]` in `p`, and `[p,p]` in `k`.
  `p` is not generally an ideal. It cannot be relabeled as a direct or
  semidirect subsystem decomposition.

For a constant algebra-basis change `E'_a=A_a^i E_i`,

\[
c'_{ab}{}^c=A_a{}^iA_b{}^j(A^{-1})_k{}^c c_{ij}{}^k.
\]

This is the correct basis law. A state-dependent frame introduces derivative
terms because the vector-field bracket is not bilinear over smooth functions.
Under an exact observation diffeomorphism with the paired pushed-forward basis,
the constants remain the same. Across unknown interfaces, that pairing and its
time/action calibration are precisely part of the unsolved representation map.

## T58.3: epsilon bound and what it does not cover

The displayed pointwise expansion is correct in a fixed chart with compatible
vector/operator norms:

\[
\|[\widehat X,\widehat Y]-[X,Y]\|
\le 2(B_1\epsilon_0+B_0\epsilon_1+\epsilon_0\epsilon_1).
\]

Setting both errors to `epsilon` and `B=max(B_0,B_1)` gives
`4 B epsilon + 2 epsilon^2`. This does not yet bound closure recovery. Projection
onto an estimated generator span adds basis error and inverse dependence on its
smallest singular value; nested brackets add derivative orders and accumulated
error. The separation `gamma` must be a uniform fixed-chart margin over a
declared domain. Norms, `B`, `epsilon`, and `gamma` are not invariant under an
ill-conditioned observation diffeomorphism, even though exact bracket zero is.

For `K t + sigma/t^2`, the exact optimizer is

\[
t_*=(2\sigma/K)^{1/3},
\]

and the minimum is

\[
(2^{1/3}+2^{-2/3})K^{2/3}\sigma^{1/3}.
\]

The order statement in T58 is therefore correct. It omits reset error, timing
and inverse-flow error, process noise, chart curvature, and uncertainty in the
starting state. Deterministically bounded sensor error does not shrink by
repetition. With independent noise whose endpoint standard error scales as
`n^{-1/2}`, the optimized bracket error improves only as `n^{-1/6}`.

## Decisive end-model counterexample

Take

\[
M=G\times W,
\]

let the recovered algebra act only on `G`, and let every action leave the
arbitrary passive factor `W` unchanged. For every choice of `W`, every smooth
observation diffeomorphism, and every reward `r(g,w,a)`, all commutator probes
and the abstract action algebra can be identical. Yet representing `W`, inverting
the observation map, learning the reward, and choosing the optimal policy can be
arbitrarily hard. The same construction works for abelian, ideal-decomposed, and
semidirect algebras.

Thus algebra matching alone supplies zero general reduction in state dimension,
readout complexity, reward sample complexity, or decision regret. Even on the
active factor, one abstract algebra has many inequivalent actions and orbit
geometries. A controller transfers only if the two systems also match in their
Lie-algebra action representations, drift/diffusion, action automorphism and
scale, constraints, costs, and reward pullback.

A defensible gain theorem must posit an equivariant map `F` and action
automorphism `A` satisfying, at minimum,

\[
DF\,\rho_s(\xi)=\rho_t(A\xi),\quad
DF\,f_s=f_t,\quad
r_t(Fx,Au)=r_s(x,u),
\]

plus compatible costs, constraints, observability, and bounded-complexity
interface maps. It must then prove that learning within this constrained class
is cheaper than matched nonlinear system identification. Merely recovering the
isomorphism class of `g` proves none of this.

## Current-literature boundary

- [LieNLSD (ICML 2025)](https://proceedings.mlr.press/v267/hu25o.html) discovers dimensions and explicit nonlinear infinitesimal symmetry generators from dynamics using a function library, surrogate Jacobians, and SVD.
- [LieDynNet](https://openreview.net/forum?id=VgZ8BJkneV), an ICLR 2026 submission rather than settled peer-reviewed evidence, enforces finite/infinitesimal validity, closure, antisymmetry, Jacobi, and dimension selection.
- [WLA](https://arxiv.org/abs/2503.09911) learns compositional Lie-action representations across environments and adapts to novel action sets.
- [AIR/VAIR](https://arxiv.org/abs/2602.06741) derives action-dependent disentanglement under explicit minimality, invertibility, and open-latent assumptions.
- [LieGAN](https://arxiv.org/abs/2302.00236) and [LaLiGAN](https://arxiv.org/abs/2310.00105) recover Lie-algebra symmetry bases, including nonlinear latent actions, and reuse them downstream.
- [VPSD-RL](https://arxiv.org/abs/2605.06500) exposes the missing end condition: value preservation requires the controlled generator **and reward functional**, not merely an isomorphic algebra.
- [Symmetric Space Learning](https://openreview.net/forum?id=e8t9F4vX9N) learns a Cartan-decomposed algebra for combinatorial generalization.
- Classical geometric control already uses bracket motion and Lie closure for accessibility; finite-dimensional complete-field assumptions permit equivalence to systems on Lie groups or homogeneous spaces.

## Hidden resource bill

T58 must charge manifold/state estimation, full-state observability, action/time
calibration, legal inverse compilation, drift cancellation, resets/safety, real
actions, denoising, vector-field/Jacobian fitting, excitation, nested pulse
length, rank conditioning, basis alignment, orbit/global-group recovery, target
representation, reward/controller learning, outer training, memory, and compute.

## Exact corrections and experiment disposition

1. Restrict invariance to diffeomorphic full-state charts or projectable
   embeddings; mark partial observation as unsolved.
2. Fix the commutator sign and state smoothness, driftless symmetric-control,
   inverse-action, timing, and reset assumptions.
3. Present Frobenius as local orbit straightening with explicit gauge, not
   semantic or global variable recovery.
4. Make finite-dimensional constant-coefficient closure an assumption until an
   open-set, conditioned identification theorem is supplied.
5. Correct the ideal/Levi/Cartan claims and separate abstract algebra from its
   state representation and global group.
6. Restrict the epsilon bound to one chart and add basis/subspace conditioning,
   nested-bracket, reset, timing, and noise terms.
7. Delete the claim that matching the algebra implies learning “only” a new
   representation/readout; those terms may contain the whole problem.
8. Replace the end gain with a theorem over equivariantly matched dynamics,
   rewards, costs, constraints, and bounded interface maps, against matched
   nonlinear system identification.

**No experiment is earned.** A CPU toy that generates a known algebra would only
confirm classical geometry. The next work is paper-only: formalize the restricted
task family and prove an acquisition-complexity separation that survives the
`G x W` passive-factor counterexample. Only then could a small analytic falsifier
be admitted. No neural run, GPU, rented resource, or benchmark construction is
justified now.
