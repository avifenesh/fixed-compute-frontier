# Independent audit — T63/T64 function-preserving plasticity

Date: 2026-08-01  
Verdict: **Both no-run decisions survive, but T63 overstates what rank invariance
rules out and T64 must make its conditional rank gain and full slack bill explicit.**

## Executive result

- T63's local gauge theorem and empirical-NTK range/nullspace result are correct.
  They rule out creation of new first-order output directions by a smooth exact
  symmetry. They do **not** rule out a large finite-budget plasticity gain from
  improving the spectrum inside the unchanged image.
- The scalar example correctly shows that uniform NTK amplification is cancelled
  by a correspondingly smaller stable step in the local squared-loss/NTK regime.
  It is not a general no-go for anisotropic gauge preconditioning.
- T64's two-layer linear dimension formula is correct for perturbations of the
  full input-output matrix. It is not automatically the rank of an empirical
  Jacobian on a finite dataset.
- The zero-output SwiGLU construction is exact and can raise local Jacobian rank,
  but the increment is conditional on transversality. Initially only the down
  vector learns; the feature weights remain gradient-flat. The construction spends
  a pre-existing dormant channel and supplies a finite, one-shot reserve.
- Neural Teleportation/PRONG/AdamO occupy conditioning; Continual Backpropagation,
  Plasticity Injection, zero-init LoRA, and Gate-Zero Growth occupy the practical
  zero-output/recycling/growth geometries. A narrow same-width implementation detail
  remains testable, but no breakthrough-sized intelligence mechanism survives.

## 1. T63 gauge-rank theorem

Let `T` be a local diffeomorphism and suppose `f(T(theta))=f(theta)` throughout a
neighborhood, not merely at one point. With `theta'=T(theta)` and
`R=DT(theta)`, the chain rule gives

\[
J_{\theta'}R=J_\theta,
\qquad
J_{\theta'}=J_\theta R^{-1}.
\]

Since right multiplication by an invertible matrix preserves column space,

\[
\operatorname{rank}J_{\theta'}=\operatorname{rank}J_\theta,
\qquad
\operatorname{im}J_{\theta'}=\operatorname{im}J_\theta.
\]

For the Euclidean empirical NTK,

\[
K_{\theta'}=J_\theta R^{-1}R^{-T}J_\theta^T.
\]

Because the middle factor is positive definite,

\[
\operatorname{im}K_{\theta'}=\operatorname{im}J_\theta
=\operatorname{im}K_\theta,
\qquad
\ker K_{\theta'}=\ker J_\theta^T=\ker K_\theta.
\]

This is exact for the output/sample-space nullspace. The **parameter-space**
nullspace is not literally unchanged; it transforms as
`ker(J_theta') = R ker(J_theta)`. T63 should name which nullspace it means.

The theorem also requires the chosen homeostatic map itself to be smooth and
locally invertible. A data-dependent canonicalization that is discontinuous,
clips scales, changes rank, or lands on a singular fiber is outside the theorem.
That exception is precisely where T64 operates.

### The inference drawn from the theorem is too strong

Rank preservation says that no new first-order direction appears. It does not say
that the existing directions remain learnable within a finite step/data budget.
The positive eigenvalues of `K` can change arbitrarily in principle, and a badly
conditioned full-rank kernel can behave almost like a rank-deficient one at finite
precision or finite training time. A gauge move that raises small eigenvalues
relative to the largest can therefore restore substantial measured plasticity
without changing rank.

T63 already concedes that conditioning can improve, but then says the move cannot
“prevent forgetting” or provide substantial restoration. Neither follows from the
rank theorem. A different coordinate-dependent optimizer trajectory can alter
both new-task learning and interference with old tasks. The valid conclusion is
narrower: gauge homeostasis cannot enlarge the instantaneous functional tangent
space or add evidence/capacity. The no-run case must rest on lack of a new gauge
selection rule, prior-art collision, and the project gate—not on rank alone.

The transformer-symmetry examples also need architecture qualifiers. An arbitrary
query/key change of basis is exact for plain dot-product attention, but with RoPE
it must commute appropriately with the position rotations. LayerNorm/RMSNorm,
biases, tied weights, quantization, and SiLU restrict other nominal gauges.

## 2. T63 scalar stability argument

For `f(x)=abx`, the parameter derivatives are `bx` and `ax`. Under
`(a',b')=(sa,b/s)`, the empirical NTK is indeed

\[
K'=(s^2a^2+b^2/s^2)XX^T.
\]

For nonzero `a,b`, the coefficient is minimized at
`s^4=b^2/a^2`. Its variation only multiplies every nonzero eigenvalue by the
same scalar, so it cannot improve the positive-spectrum condition number.

In local squared-loss or Gauss-Newton/NTK dynamics, stability requires the step
size to scale inversely with the largest eigenvalue. Therefore making the scalar
factor large forces an inverse learning-rate change and does not improve the best
stable functional step. This portion of T63 is correct.

Two qualifications are required:

1. Away from interpolation, the exact Hessian has residual-dependent second-order
   terms, so “must cancel” is a local/linearized statement, not a global theorem
   for arbitrary loss and optimizer.
2. The example tests only uniform scale. Multi-dimensional gauges can rebalance
   anisotropy, which is exactly the useful effect pursued by preconditioning.
   Transporting Adam moments is necessary to call the operation a pure coordinate
   change; resetting or retaining untransformed moments is a separate intervention.

## 3. T64 two-layer linear Jacobian dimension

Take `W1 in R^(h x d)`, `W2 in R^(o x h)`, with ranks `r1,r2`. The tangent image
in matrix space is

\[
\mathcal A+\mathcal B
=\{\delta W_2W_1\}+\{W_2\delta W_1\}.
\]

`A` contains all `o x d` matrices whose row space lies in `row(W1)`, so
`dim A=o r1`. `B` contains all matrices whose column space lies in `col(W2)`, so
`dim B=r2 d`. Their intersection has both restrictions and dimension `r1 r2`.
Hence

\[
\dim\operatorname{im}J=o r_1+r_2d-r_1r_2.
\]

The formula is correct for the Jacobian of the represented matrix
`M=W2 W1`, equivalently for access to all input directions. On an empirical input
matrix `X`, the observed perturbation is `delta M X`; its rank can be smaller and
is additionally capped by `o rank(X)`. T64 should not move between this structural
dimension and empirical-NTK rank without the dataset evaluation map.

## 4. T64 exact zero-output SwiGLU rank lift

For one channel

\[
f(x)=f_{old}(x)+d\,\phi_{u,g}(x),
\qquad
\phi_{u,g}(x)=\operatorname{SiLU}(g^Tx)(u^Tx),
\]

`d=0` makes the contribution identically zero for all finite inputs. Reseeding
`u,g` while holding `d=0` is therefore exactly function preserving, and

\[
\partial f/\partial d=\phi_{u,g}(x).
\]

For `N` examples and `o` output coordinates, let
`Phi=(phi(x_1),...,phi(x_N))`. The Jacobian block for the vector `d` is, up to
layout, `Phi tensor I_o`, with rank `o` when `Phi` is nonzero. The actual increment
over the old model is

\[
\Delta r=\operatorname{rank}((I-P_{old})(\Phi\otimes I_o)),
\qquad 0\leq\Delta r\leq o.
\]

It is zero if the feature block already lies in the old Jacobian image. With `q`
channels, the maximum increment is `o rank([Phi_1,...,Phi_q]) <= oq`, subject to
the empirical output-dimension cap. “Choose an independent feature” is therefore
a conditional transversality assumption, not a universal guarantee.

At the zero-output point,

\[
\partial f/\partial u=\partial f/\partial g=0.
\]

Only `d` receives first-order credit. The feature parameters start learning only
after `d` moves away from zero, which creates a two-stage/higher-order credit
bottleneck. The construction exposes a random or externally chosen feature bank
immediately; it does not expose unrestricted fresh feature learning at first
order. Feature scale, correlation, numerical rank, and target alignment must be
measured, not inferred from algebraic rank.

This does not contradict T63: the translation of `u,g` preserves the function
only on the singular stratum `d=0`; it is not a model symmetry on a neighborhood.

## 5. The slack bill is real but incomplete

For the specific channelwise construction, if `q` down vectors are exactly zero,
deleting those channels preserves the current function. The function at that
point is therefore representable with at most `h-q` active channels. That is an
exact and useful accounting fact.

The broader “no universal construction” sentence is not proved by this example.
Other singular fibers can use duplicated or cancelling units rather than
individually zero output factors. They still appear to spend redundancy, but T64
does not establish a general lower bound equating every possible fiber-rank lift
with `q` deletable channels or a specified loss in approximation quality.

The operational bill must include:

- parameters and checkpoints for dormant `u,g,d`, plus optimizer states;
- dense training FLOPs unless the reserve uses structured masking;
- compiler/redeployment cost if dormant channels are physically omitted at serve;
- normal memory traffic and FLOPs after activation; and
- exhaustion of the finite reserve. Repeated continual learning requires later
  compression/recycling, new growth, or tolerated drift.

Standard dense kernels do not obtain a free per-channel skip merely because a
down vector is zero. Conversely, if an overparameterized deployed model already
contains unused channels, spending them may give practical gains with no measured
current-quality loss. “It uses slack” is a cost statement, not proof of uselessness.

## 6. Prior-art collision, checked against primary sources

### T63 conditioning family

- [Neural Teleportation](https://arxiv.org/abs/2012.01118) explicitly moves along
  function-equivalent parameter representations and changes local geometry and
  backpropagated gradient scale.
- [Natural Neural Networks / PRONG](https://arxiv.org/abs/1507.00210) performs
  function-preserving representation whitening to improve Fisher conditioning and
  amortizes reparameterization cost.
- [Preserving Plasticity via Dynamical Isometry](https://arxiv.org/html/2606.09762)
  connects finite-budget plasticity to empirical-NTK anisotropy and introduces
  AdamO. It is an ICML 2026 paper, but its transformer evidence is explicitly a
  preliminary four-block CIFAR-10 pixel-permutation study, not an LLM result.

These works make a generic “choose a better-conditioned representative” proposal
occupied. They also reinforce the correction above: spectrum, not only rank, can
matter substantially to plasticity.

### T64 zero-output/recycling family

- [Continual Backpropagation](https://arxiv.org/html/2306.13812) replaces
  low-utility units in a fixed-capacity network and initializes their outgoing
  weights to zero so the newly inserted unit initially adds nothing. Removing the
  old low-utility unit and transferring its mean contribution to a bias is only an
  approximate preservation step, so it does not fully prove T64's exact variant.
- [Plasticity Injection](https://arxiv.org/html/2305.15555) exactly preserves
  predictions by adding cancelling fresh residual heads while freezing the old
  head. It keeps trainable-parameter count fixed but explicitly increases total
  parameters, memory, and training time.
- [Gate-Zero Growth](https://arxiv.org/html/2607.14571) gives the same conditional
  projected-rank formula for zero-gated residual blocks and explicitly unifies
  zero-init LoRA, ReZero, adapters, and zero-output projections. Its Transformer
  result spends `300M -> 857M` parameters.

Gate-Zero is a single-author July 2026 v1 preprint with a strong conditional
theorem but limited empirical certainty: main cells use one seed and one dataset
order, and the reported transversality diagnostic projects against only 200
sampled old-Jacobian directions. Independence from that sampled subspace does not
establish independence from the full old-Jacobian image; the paper acknowledges
possible unsampled overlap. This weakens the empirical claim, not the elementary
zero-gate algebra.

Collectively, the sources bracket T64 closely. The exact use of a pre-existing
zero-output SwiGLU channel at unchanged width is a narrower implementation point
than growth or approximate recycling, but its geometry is the same zero-init
output-factor/LoRA geometry and its resource is the same fixed-capacity reserve.

## 7. Required corrections and verdict

1. Keep T63's gauge theorem, but distinguish output-space NTK nullspace from the
   transformed parameter nullspace.
2. Remove the claims that rank preservation alone bars substantial plasticity
   restoration or forgetting mitigation. It bars new tangent directions, not
   conditioning gains in the old image.
3. Scope the scalar cancellation to local squared-loss/NTK dynamics and emphasize
   that it treats uniform scale, not anisotropic rebalancing.
4. Add RoPE/normalization/quantization restrictions to the claimed Transformer
   symmetry group.
5. Label T64's linear formula as a full-matrix structural dimension and add the
   empirical-data rank cap.
6. Replace the informal SwiGLU “adds a direction” claim with the projected-rank
   formula; state that `u,g` are gradient-flat at `d=0`.
7. Present the deletable-channel slack result as exact for this construction, but
   do not call it a universal lower bound over all singular fibers without proof.
8. Charge dormant storage/optimizer state, dense training, compiler/redeployment,
   activated serving cost, and reserve exhaustion.
9. Qualify Gate-Zero's status and diagnostic, while retaining it, Plasticity
   Injection, and Continual Backpropagation as direct mechanism collisions.

**Final verdict: no breakthrough mechanism survives.** T63 retains a valid
preconditioning/optimizer control. T64 retains a valid local plasticity primitive
and a narrow same-width systems hypothesis. Neither supplies evidence selection,
long-horizon credit, abstraction formation, retention beyond finite reserve, or a
strict fixed-compute Pareto case over the cited controls.

**No run is justified for the project's primary-mechanism search.** For T63, the
reason is prior-art saturation and absence of a transformer-specific gauge policy,
not a rank-based performance impossibility. For T64, the reason is that the exact
primitive is already bracketed by zero-init adapters/growth and fixed-capacity
recycling, while the proposed reserve is finite and fully charged. A future
plasticity-subsystem study would need a preregistered repeated-stream result at
unchanged total served parameters/FLOPs, with current capability, amortized
training cost, reserve exhaustion, and matched Continual Backprop, Plasticity
Injection, LoRA, Gate-Zero, and AdamO controls.
