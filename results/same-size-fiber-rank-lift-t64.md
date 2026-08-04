# T64 same-size fiber-rank lift — possible only by spending slack

Date: 2026-08-01  
Status: **EXACT BOUNDARY RETAINED; DISTINCT MECHANISM REJECTED; NO RUN**

## 0. Result in one sentence

A fixed-size network can gain new first-order learning directions without
changing its current function by reseeding parameters hidden behind an exact
zero-output factor, but those directions consume representational slack; the
operation reduces to neuron recycling or a preallocated plasticity reserve, while
function-preserving growth obtains the same benefit by explicitly adding capacity.

## 1. Exact constructive example

Consider one hidden feature in a residual MLP,

\[
f(x)=f_{old}(x)+d\,\phi_{u,g}(x),
\qquad
\phi_{u,g}(x)=\operatorname{SiLU}(g^Tx)(u^Tx),
\]

where `d` is its output vector.  At `d=0`, the feature contributes exactly zero
for every input, so `u` and `g` may be replaced arbitrarily without changing the
represented function.  Yet

\[
\frac{\partial f(x)}{\partial d}=\phi_{u,g}(x).
\]

Choosing `u,g` so that the projected feature is linearly independent of the
existing functional Jacobian adds a new first-order output direction.  At the
zero-output point only `d` receives this first-order learning signal; the
feature parameters `u,g` remain flat until `d` moves away from zero.  This does not
contradict T63: reseeding behind `d=0` is a move available only on a singular
zero-product stratum, not a smooth invertible symmetry of a neighborhood.

The same fact is visible in the scalar product model `f(a,b)=ab`.  Both `(0,0)`
and `(1,0)` represent the zero function, but the Jacobian ranks are respectively
zero and one.  Moving the hidden factor from zero to one while the output factor
remains zero preserves the function and restores a learning direction.

## 2. Linear-network rank accounting

For a two-layer linear map

\[
f(x)=W_2W_1x,
\]

the first-order function change is

\[
\delta M=\delta W_2W_1+W_2\delta W_1.
\]

If `rank(W1)=r1` and `rank(W2)=r2`, the dimension of the reachable **full matrix**
perturbations is

\[
\dim\operatorname{im}J
=o r_1+r_2 d-r_1r_2.
\]

The first term consists of perturbations whose row space lies in `row(W1)`; the
second consists of perturbations whose column space lies in `col(W2)`; their
intersection has dimension `r1*r2`.  On a finite empirical input set the
observed Jacobian rank may be smaller.  A product factor that is zero on one side
but diverse on the other can therefore preserve `W2 W1` while changing the
available tangent directions.

## 3. The slack bill

For `q` zero-output SwiGLU channels, the current function is represented by at
most the remaining `h-q` active channels.  The dormant channels can be skipped by
an inference compiler while their output factors stay exactly zero, but after
activation they consume their normal parameters, memory traffic, and FLOPs.

Therefore one of two things must be true:

1. the old function was representable with `h-q` channels, in which case the
   model contained at least `q` channels of architectural slack; or
2. reserving/recycling those channels reduces current approximation quality.

This exact zero-channel construction cannot avoid that choice.  If every
channel is required for the current function under the chosen approximation
tolerance, it cannot create an exact zero-output reserve without first
compressing the function, changing the architecture, or accepting deviation.
This is not a universal theorem over every singular same-function fiber.

## 4. Current-work collision

The constructive mechanisms are already occupied at each currency point:

- Continual Backpropagation continually reinitializes low-utility units in a
  fixed-capacity network and reports sustained plasticity across thousands of
  continual ImageNet tasks.  Its utility test and resets are approximate rather
  than exact function preservation.
- Plasticity Injection freezes the old head and adds a fresh zero-output residual
  head.  It preserves predictions and the number of *trainable* parameters, but
  increases total parameters, memory, and compute.
- Gate-Zero Growth adds zero-gated residual blocks.  Its theorem gives conditional
  Jacobian-rank additivity through gate directions, and its transformer experiment
  grows `300M -> 857M`; the strong preservation result is bought with explicit
  capacity growth.
- Fixed-capacity neuron-reset, pruning/recycling, and spectral/isometry methods
  already cover practical approximations to maintaining a plastic reserve.

Primary references:

- https://arxiv.org/abs/2306.13812
- https://arxiv.org/abs/2305.15555
- https://arxiv.org/abs/2607.14571
- https://arxiv.org/abs/2606.09762

## 5. What this does and does not solve

Fiber-rank lifting is a real local answer to “how can an old model learn again?”
It can prevent optimization capacity from collapsing and can preserve the old
function at the instant fresh directions are exposed.

The full ledger includes dormant parameter storage, optimizer state, reseeding
and training FLOPs, compiler/skip metadata, normal serving cost after activation,
and reserve exhaustion or replenishment.  Counting only trainable parameters
or the zero-gate instant hides these costs.

It does not decide what evidence to seek, assign long-horizon causal credit,
form a correct world model, discover reusable abstractions, consolidate an
unbounded experience stream into finite capacity, or reason reliably.  Even
perfect plasticity only ensures that a supplied learning signal can move the
function efficiently.

## 6. Run decision

**No run.**  A toy zero-product experiment would only confirm the exact derivative
above, and fixed-capacity neuron recycling plus explicit growth already establish
the empirical effect.  The candidate cannot meet the substantial-intelligence
gate without a mechanism that produces better evidence, abstractions, or credit.

The next search returns to the full intelligence loop.  Plasticity is retained as
a required subsystem and diagnostic, not promoted to the primary breakthrough.
