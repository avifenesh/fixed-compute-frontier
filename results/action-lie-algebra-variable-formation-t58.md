# T58 action Lie algebra — variable formation from commutator experiments

Date: 2026-08-01  
Status: **LOCAL CONTROL DIAGNOSTIC RETAINED; DISTINCT UPDATER AND TRANSFER EDGE REJECTED; NO RUN**

## 0. Result in one sentence

Smooth reversible actions induce vector fields whose Lie brackets are natural
under diffeomorphic full-state coordinates. Legal inverse-flow loops can
therefore diagnose local action coupling under strong control assumptions. The
primitive does not form a sufficient state or a transferable controller: the
same algebra can coexist with arbitrarily difficult passive state, observation,
reward, and policy learning. Independent audit therefore closes the model-level
claim without a run.

## 1. Why this is different from naming predicates

Let the observable state lie on a smooth manifold `M`, possibly embedded in a
very high-dimensional sensor space. A locally continuous control `i` induces a
smooth vector field `X_i` and flow `Phi_i^t`:

\[
\frac{d}{dt}\Phi_i^t(x)=X_i(\Phi_i^t(x)).
\]

No semantic name, object slot, latent coordinate, or symbolic predicate is
assumed. The primitive observation is only what happens after executing short
legal actions.

If another interface uses full-state coordinates `y=psi(x)` for a local
diffeomorphism `psi` (or an embedding onto its image with projectable fields),
the corresponding field is the pushforward `psi_*X_i`. Naturality of the Lie
bracket gives

\[
\boxed{\psi_*[X_i,X_j]=[\psi_*X_i,\psi_*X_j].}
\]

Thus exact commutativity and the exact generated algebra do not depend on that
coordinate chart. A partial or noninjective observation need not admit a unique
pushed-forward field, so this does not solve raw-observation identifiability.
It weakens T54's affine-sensor assumption only when local full-state
invertibility, smooth legal inverse actions, and local excitation are supplied.

## 2. Observable commutator probe

For two reversible flows, execute the four-step loop

\[
C_{ij}(t)
=\Phi_j^{-t}\circ\Phi_i^{-t}\circ\Phi_j^t\circ\Phi_i^t.
\]

With the displayed bracket and composition conventions, the
Baker--Campbell--Hausdorff expansion gives

\[
C_{ij}(t)(x)=x+t^2[X_i,X_j](x)+O(t^3).
\]

With the displayed bracket and composition conventions the leading sign is
fixed; swapping `i,j` reverses it.

This is not a free oracle. It requires executable negative flows rather than
merely mathematical inverse flows, calibrated timing, no uncancelled drift,
four controlled segments, chart containment, uniform smoothness, adequate
reset/repeat access, and sensor resolution fine enough to see an `O(t^2)`
displacement. Dissipation, contact, stochasticity, hysteresis, saturation, or a
control-affine drift can invalidate the maneuver.

## 3. T58.1 — local coordinate birth by Frobenius

Suppose `X_1,...,X_k` are linearly independent on a neighborhood and pairwise
commute:

\[
[X_i,X_j]=0.
\]

The simultaneous straightening theorem gives local coordinates
`z_1,...,z_d` in which

\[
X_i=\frac{\partial}{\partial z_i},\qquad i\le k.
\]

Therefore independent commuting interventions define only local orbit
coordinates, up to a large chart gauge. Transverse coordinates remain
arbitrary; topology, rank changes, singular orbits, and holonomy can obstruct a
global chart. This is not a semantic or complete-state construction.

The theorem is classical and already used as a criterion for disentangled
representations. It cannot be claimed as a new model. It also deliberately
excludes common noncommuting structure such as rotations acting on
translations.

## 4. T58.2 — do not force real structure to commute

Assume the action-generated fields close on an open set as a
finite-dimensional Lie algebra with constant scalar coefficients

\[
\mathfrak g=\operatorname{Lie}\{X_1,\ldots,X_m\},
\qquad
[X_i,X_j]=\sum_{k=1}^{r}c_{ij}^{k}X_k.
\]

The constants `c_ij^k` are a multiplication table for local action
composition. This closure is an assumption, not something certified by
pointwise noisy rank tests: generic smooth fields generate an
infinite-dimensional algebra, and state-dependent coefficients define
structure functions rather than this finite algebra. Appropriate product
formulas only approximate nested bracket directions under symmetric-control
and regularity assumptions.

Instead of flattening this algebra into independent coordinates:

- a direct-sum decomposition into ideals identifies mutually commuting action
  subsystems;
- the solvable radical and a Levi factor separate a finite-dimensional algebra
  into solvable and semisimple pieces; and
- semidirect products retain directional couplings, such as one subsystem
  transforming another.

Under a constant change of action basis, the structure constants obey the
ordinary basis transformation law, while Lie-algebra isomorphism remains. A
state-dependent frame introduces derivative terms. Under an observation
diffeomorphism the constants agree only when the pushed-forward basis and time
calibration are already paired. The abstract algebra does not determine its
state action, orbit/stabilizer geometry, global group, drift, or readout.

The decomposition is not semantic. The solvable radical is canonical, but a
Levi factor generally is not; ideal decompositions can be nonunique, and the
Levi action on the radical is essential data. Algebraic ideals do not imply a
product state, independent rewards, or transferable controllers. Two tasks
with the same action algebra may have unrelated observations, rewards,
constraints, costs, and useful policies.

## 5. T58.3 — epsilon bill for learned brackets

In an observed coordinate chart, use

\[
[X,Y]=JY\,X-JX\,Y.
\]

Assume

\[
\|X\|,\|Y\|\le B_0,\quad
\|JX\|,\|JY\|\le B_1,
\]

and estimates satisfy

\[
\|\widehat X-X\|,\|\widehat Y-Y\|\le\epsilon_0,
\quad
\|J\widehat X-JX\|,\|J\widehat Y-JY\|\le\epsilon_1.
\]

Expanding the two bilinear products gives

\[
\|[\widehat X,\widehat Y]-[X,Y]\|
\le
2(B_1\epsilon_0+B_0\epsilon_1+\epsilon_0\epsilon_1).
\]

For `epsilon_0=epsilon_1=epsilon` and `B=max(B_0,B_1)`, this is at most

\[
4B\epsilon+2\epsilon^2.
\]

Consequently a zero/nonzero decision with separation margin `gamma` is stable
only when the complete bracket error is below `gamma/2`. This pointwise bound
uses compatible norms in one fixed chart. Closure recovery additionally pays
generator-basis and subspace conditioning, higher-derivative and nested-bracket
errors, and a uniform open-set identification bill. Its numerical margin is
not invariant under an ill-conditioned observation diffeomorphism.

Direct commutator loops have a separate resolution tradeoff. If the local BCH
remainder divided by `t^2` is bounded by `K t` and endpoint sensor error
contributes at most `sigma/t^2`, then

\[
\epsilon_{loop}(t)\lesssim Kt+\frac{\sigma}{t^2}.
\]

Its best scale is of order `(sigma/K)^(1/3)`, with minimum error of order
`K^(2/3)sigma^(1/3)`. Very small actions amplify sensor noise; larger actions
leave the local bracket regime. Repeating noisy measurements only weakly
improves this indirect derivative unless the sensor or trajectory model shares
information across states.

## 6. Candidate learned updater

A bounded lifetime-trained control diagnostic could implement:

```text
estimate local action flows from raw transition pairs
choose short reversible commutator experiments
fit generated vector fields and structure constants
enforce antisymmetry, Jacobi, and closure residuals
split the algebra into ideals / semidirect dependencies
retrieve a candidate prior representation for matched algebra pieces
measure the full new representation, task-readout, and controller bill
```

The persistent record could be a versioned local algebra hypothesis plus
evidence and environment-specific maps. It is not a distinct updater: closure
and generator birth require a specified function class and conditioned
open-set identification theorem, while representation matching can contain the
entire original learning problem.

## 7. What this could buy

In a deliberately restricted equivariant family, an algebraic learner might
reuse:

- the number and composition of controllable mechanisms;
- which action subsystems are independent or coupled;
- controllers and planners expressed in abstract generator coordinates; and
- a finite multiplication table instead of raw trajectories.

No general end-model gain follows. Let the state be `M=G x W`, let the recovered
algebra act only on `G`, and leave arbitrary passive state `W` untouched. Every
commutator probe and abstract algebra can remain identical while representing
`W`, inverting the observation, learning the reward, and choosing the policy
are arbitrarily hard. A gain theorem would need compatible equivariant state
actions, drift/diffusion, action calibration, rewards, costs, constraints, and
bounded-complexity interface maps, then beat matched nonlinear system
identification. T58 provides no such theorem.

## 8. Current literature boundary

The core ingredients are occupied:

- commutativity as a necessary/sufficient local-chart disentanglement
  criterion was explicitly developed in Frank Qiu's 2023 dissertation;
- World Modeling through Lie Action learns shared continuous compositional
  action representations across environments, but assumes selected matrix
  group families and fixed slot/action capacity;
- action-induced representations give provable action-dependent
  disentanglement from experimental outcomes;
- VPSD-RL discovers exact/approximate value-preserving Lie-group operators in
  continuous RL; and
- causal representation learning, Koopman methods, geometric control, and
  nonlinear system identification are mandatory controls.

The narrower conjunction not established by these sources is online discovery
of an initially unknown, potentially noncommutative finite action algebra from
raw commutator experiments, followed by ideal/semidirect factor transport to a
held-out observation interface and task.

Primary references:

- https://escholarship.org/uc/item/8xb4737b
- https://arxiv.org/abs/2503.09911
- https://arxiv.org/abs/2602.06741
- https://arxiv.org/abs/2605.06500
- https://arxiv.org/abs/2604.23800

This remaining conjunction is an unproved capability conjunction, not a
distinct updater or architectural novelty.

## 9. Fatal tests before any experiment

Close the lane without a run if any of the following holds:

1. a matched nonlinear system-identification or Lie-action control already
   recovers the same algebra with lower acquisition and compute cost;
2. inverse/commutator actions are supplied only in a synthetic benchmark and
   cannot be compiled from the raw interface without an oracle;
3. matching the algebra fails to reduce representation/readout learning cost;
4. different reward/constraint laws erase policy transfer despite identical
   action algebra;
5. the generated Lie algebra is infinite-dimensional or has no stable margin;
6. the gain disappears against a recurrent world model trained with identical
   commutator-loop augmentation; or
7. learned vector-field/Jacobian error is already larger than every useful
   bracket separation.

## 10. Final disposition

No experiment is earned. A CPU toy generating a known algebra would only
confirm classical geometry. The lane is retained as a local diagnostic for
noncommutativity, accessibility, and coupling. It is rejected as a distinct
acquisition updater and as a general cross-interface intelligence mechanism.

The independent audit supplies the complete geometric, identifiability,
resource, and literature corrections:

- [T58 independent audit](action-lie-algebra-variable-formation-t58-independent-audit.md)
