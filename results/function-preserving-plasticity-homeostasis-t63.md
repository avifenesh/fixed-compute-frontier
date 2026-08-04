# T63 function-preserving plasticity homeostasis — exact rank no-go

Date: 2026-08-01  
Status: **OPTIMIZER CONTROL RETAINED; INTELLIGENCE MECHANISM REJECTED; NO RUN**

## 0. Result in one sentence

Moving a trained network along an exact function-preserving gauge orbit can
change the conditioning seen by coordinate-dependent optimizers, but every
smooth invertible such move preserves the model's functional tangent space and
Jacobian rank exactly; it can rebalance existing learning directions, not create
missing directions.  This rank result does **not** bound the finite-budget gain
from conditioning, which can be large; the rejection is of mechanism novelty
and rank restoration, not of practical optimizer value.

## 1. Candidate and intended gain

The candidate was a homeostatic operation between learning episodes:

1. keep the represented function and all current predictions fixed;
2. move to a different parameter representative on the same function orbit;
3. choose the representative with a better-conditioned empirical NTK or Fisher;
4. continue training from there with unchanged architecture and serving cost.

Examples of exact transformer symmetries include reciprocal changes of basis in
the query/key channels, inverse changes of basis across value/output projections,
permutations, and limited reciprocal scaling across an MLP hidden feature and its
down projection.  Residual paths, normalization, and non-homogeneous SiLU gates
restrict the available group; they do not enlarge it.

## 2. Exact gauge-rank theorem

Let `f: Theta -> R^m` be continuously differentiable and let `T` be a local
diffeomorphism that is an exact model symmetry:

\[
f(T(\theta))=f(\theta)
\]

in a neighborhood of `theta`.  Write

\[
\theta'=T(\theta),\qquad R=D T(\theta),\qquad
J_\theta=\frac{\partial f}{\partial\theta}.
\]

Differentiating the symmetry identity gives

\[
J_{\theta'}R=J_\theta,
\qquad
J_{\theta'}=J_\theta R^{-1}.
\]

Because `R` is invertible,

\[
\operatorname{rank}J_{\theta'}=\operatorname{rank}J_\theta,
\qquad
\operatorname{im}J_{\theta'}=\operatorname{im}J_\theta.
\]

Thus a smooth exact re-gauging cannot make any previously unreachable
first-order output change reachable.  For the Euclidean empirical NTK,

\[
K_{\theta'}
=J_\theta R^{-1}R^{-T}J_\theta^T.
\]

Its positive eigenvalues and condition number may change, but its range and
nullspace do not.  The candidate can precondition the directions already
present; it cannot restore functional rank, generate a new feature, revive a
direction annihilated by a gate, acquire evidence, or prevent forgetting.

## 3. Why larger gradients are not free plasticity

For the scalar two-factor model

\[
f_{a,b}(x)=abx,
\qquad
(a',b')=(sa,b/s),
\]

the function is unchanged, while the empirical NTK becomes

\[
K'=(s^2a^2+b^2/s^2)XX^T.
\]

The gradient scale can be made arbitrarily large by extreme `s`, but the rank
and condition number of `XX^T` are unchanged.  In the local squared-loss/NTK
regime with this uniform kernel scaling, a stable learning rate falls as the
largest curvature scale grows, cancelling the apparent gain.  This is not a
global theorem for Adam, nonlinear feature learning, or nonuniform gauges.  The
balanced value `s^4=b^2/a^2` minimizes the factor scale and improves numerical
balance; it does not create a new tangent direction.

More general gauge groups can redistribute anisotropy, so a useful conditioning
gain is possible.  But that gain exists only because ordinary SGD and Adam use a
coordinate-dependent metric.  A fully reparameterization-invariant optimizer
would produce the same functional update.  Adam moments must also be transported
consistently; resetting or leaving them in the old coordinates introduces a
separate optimizer-state intervention.

## 4. Current-work collision

The broad mechanism is already occupied:

- Neural Teleportation moves networks between function-equivalent parameter
  representatives, changes gradient scale and local loss geometry, and reports
  faster gradient descent both at initialization and after a mid-training move.
- Natural Neural Networks/PRONG uses function-preserving activation whitening to
  improve Fisher conditioning and amortizes the reparameterization cost.
- Path-normalized and symmetry-invariant optimizers explicitly remove or exploit
  the same positive-scaling gauges.
- Current continual-learning work connects plasticity to NTK anisotropy and
  applies full-spectrum isometry regularization/AdamO.  It reports broad gains in
  supervised and RL streams, but only a preliminary small transformer study and
  no large-LLM result.
- Plasticity Injection already supplies a zero-output fresh residual learner
  while freezing the old head.  It can produce large gains, but increases total
  parameters, memory, and training/inference work.

Primary references:

- https://arxiv.org/abs/2012.01118
- https://arxiv.org/abs/1507.00210
- https://arxiv.org/abs/2606.09762
- https://arxiv.org/abs/2305.15555

## 5. The constructive boundary

The theorem identifies the next admissible class.  To increase local functional
rank while preserving the current function, a method must leave the smooth
invertible symmetry class.  At least one of the following currencies is required:

1. **representational slack:** move to a singular or redundant same-function
   factorization containing zero-output but gradient-active features;
2. **architecture/parameter change:** add a fresh residual learner, as plasticity
   injection does;
3. **temporary function deviation:** cross a loss barrier while replay or a
   verifier protects old capabilities; or
4. **higher-order escape:** use directions invisible at first order, accepting
   slower credit and no immediate NTK-rank lift.

This is a stricter target than “improve the NTK.”  A surviving mechanism must
show that it creates useful new tangent directions at fixed served size, not
merely rescales gradients, and that the capacity reserved for future learning
does not reduce current capability by an equal or larger amount.

## 6. Run decision

**No run.**  The central algebra is exact and current work already demonstrates
the remaining conditioning effect.  A small optimization benchmark could
reproduce neural teleportation but would not test a distinct intelligence
mechanism.  This decision does not assert that a future transformer-specific
conditioning method cannot have a large finite-training gain.

The next theorem target is a same-size **fiber-rank lift**: an exact or protected
move to a function-equivalent parameter point with a strictly larger useful
functional tangent space.  It must expose and charge the representational slack
that makes the lift possible.
