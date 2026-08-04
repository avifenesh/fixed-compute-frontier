# Heterogeneous algebraic compilation — adversarial T1b preregistration

Frozen before result generation: 2026-07-30

## Purpose

T1 passed on exposed binary coordinates, but current theory shows that
Rademacher initialization can make almost-full parity learnable by gradient
descent.  T1b attacks both weaknesses before any language-model advancement.

## Adversarial representations

The same three sealed worlds (`731, 947, 1213`) are reused with two new data
views:

1. **Unknown finite-field mixing.** Latent 32-bit vectors are multiplied by a
   secret dense invertible `32 x 32` matrix over `GF(2)`.  Labels are a latent
   24-bit parity.  The compiler sees only the mixed bits and labels; it must
   recover the induced parity in observed coordinates.
2. **Raw recurrence windows.** The induced observed-coordinate mask defines an
   order-32 binary linear recurrence.  The compiler receives 256 consecutive
   context/next-bit windows from one stream, not independent sampled contexts.
   Validation uses a fresh initial state.

The deployed model is exactly the T1 78,404-parameter residual-SwiGLU model.
Every arm starts from the same Gaussian model initialization before its charged
transition.

## Candidate

The candidate uses the same overdetermined `GF(2)` elimination and exact
SwiGLU product-tree compiler as T1.  It may compile only on rank-32 consistent
systems.  It takes no gradient step.

## Initialization-aware controls

After consuming the same 256-example prefix, controls discard their weights
and perform a data-independent Rademacher rebirth.  This is a training
transition from the common initialization, not a different deployed model.

- gate/up rows: independent signs scaled by `m/sqrt(68)`;
- down matrices: independent signs scaled by
  `m/(sqrt(64)*sqrt(6))`;
- output head: independent signs scaled by `m/sqrt(68)`;
- multipliers `m in {0.25, 0.5, 1.0}`;
- AdamW learning rates `{1e-3, 3e-3}`;
- both binary cross-entropy and signed hinge loss;
- batch 64, gradient norm clipping 1.0, no weight decay, 2,000 steps;
- the best endpoint is selected optimistically per world, view, and loss.

This directly tests the initialization escape identified by Abbe et al.
(<https://arxiv.org/abs/2412.04910>), while retaining the exact fixed
architecture.  Their theorem uses a wider two-layer ReLU construction, so this
is an adversarial transfer test rather than a claimed reproduction.

## Frozen gates

Advance only if all conditions hold in all three worlds and both views:

1. candidate full, parity, and protected accuracy are each at least `99%`;
2. recovered masks exactly match the induced hidden masks;
3. every mixing matrix is invertible and is not a permutation matrix;
4. the best Rademacher BCE control and the best Rademacher hinge control remain
   below `80%` parity accuracy after 2,000 steps;
5. at least one Rademacher control per view reaches `95%` protected accuracy;
6. candidate solver and compile work, control dense work, data exposure, and
   wall time are reported separately;
7. source, test, preregistration, predecessor result, and result hashes are
   recorded.

Passing remains an exact-system finding.  It advances only to a causal-model
compilation gate; it is not a natural-language or novelty claim.

