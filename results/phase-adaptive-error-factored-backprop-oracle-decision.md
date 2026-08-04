# PAEFB exact-error oracle: closed negative result

Date: 2026-07-30  
Decision: **close before recursive simulation or kernels**

## Fatal result

The run was stopped when the width-768, step-64 cell completed because its
frozen cost gate was irreversibly false.

Across the 21 sampled Q/K/V/O and FFN maps:

- charged backward ratio: **84.6425%** of dense;
- frozen ceiling: **8%**;
- dense fallbacks: **10 of 21**;
- minimum accepted squared cosine: **0.99000018**.

The cell misses the required cost by **10.58x**. Later checkpoints or other
widths cannot restore the all-gates decision.

Dense fallbacks included all three output projections and all three FFN down
projections, plus the first sampled layer's Q, K, gate, and up projections. The
first layer's three FFN maps alone represent 25% of sampled learned-projection
work, already more than three times the total frozen allowance.

## What the two consecutive oracles prove

The prior CPAC oracle found that completed weight gradients `dW` become very
cheap to approximate at 90% direction quality. This oracle measured the actual
linear backward ingredients at 99% joint `(dX,dW)` quality and found them
mostly expensive.

There is no contradiction:

`dW = E^T X`.

The product can have a concentrated singular spectrum even when the arriving
error `E` is not cheaply factorizable. Computing a low-rank approximation of
the **finished** `dW` after the dense multiplication therefore does not remove
the multiplication that produced it. Likewise, communication or optimizer
memory savings from gradient compression are not automatically training-FLOP
savings.

## Why lowering the direction threshold is not a repair

The 99% gate protects rare and compositional error directions. Current work on
the LM-head gradient bottleneck reports that projection can suppress most
logit-gradient norm and make simple patterns unlearnable; deliberately adding
another aggressive backward bottleneck would require capability evidence, not
a looser NLL-only approximation gate. The project contract forbids rescuing a
failed compute claim by silently discarding more learning signal.

## Retained boundary

1. Weight-gradient low rank is real and improves with width in these small
   from-zero models.
2. It is useful for optimizer state, communication, or update representation.
3. It does **not** expose a cheap exact-forward dense-backward path at protected
   direction quality.
4. Do not pursue randomized SVD, Walsh-Hadamard projections, recursive error
   compression, or custom kernels for this formulation.

The partial raw record is
`results/phase-adaptive-error-factored-backprop-oracle.log`. No result JSON was
written because sequential falsification stopped the run before the remaining
unneeded cells.
