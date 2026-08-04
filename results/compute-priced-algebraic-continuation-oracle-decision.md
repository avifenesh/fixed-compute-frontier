# CPAC gradient oracle: closed negative result

Date: 2026-07-30  
Decision: **do not implement CPAC; retain the phase-dependent gradient fact**

## Frozen result

CPAC failed four of its six gates:

- cheap mixed structure in seven of nine cells: **fail, 6/9**;
- mixed algebra 25% cheaper than low-rank in six cells: **fail, 0/9**;
- cost decreases with width at every checkpoint: **pass, 3/3**;
- isotropic gradients require at least 85% dense cost: **pass, 90.03%**;
- widest cell projects to at least 2x at every checkpoint: **fail, step 0
  is only 1.4679x**;
- all models learn: **pass**.

The proposal is therefore neither a from-zero large-gain mechanism nor
distinct from incremental low-rank training.

## The decisive table

Every direction retained at least 90% squared cosine to the exact dense
gradient. Costs are ideal structured matrix-apply cost divided by dense cost.

| width | step | mixed cost | low-rank cost | mixed saving | projected frontier efficiency |
|---:|---:|---:|---:|---:|---:|
| 192 | 0 | 73.42% | 77.69% | 5.50% | 1.186x |
| 192 | 64 | 5.92% | 6.45% | 8.08% | 3.963x |
| 192 | 512 | 6.60% | 6.60% | 0.00% | 3.732x |
| 384 | 0 | 66.00% | 76.13% | 13.31% | 1.253x |
| 384 | 64 | 3.79% | 4.65% | 18.53% | 4.218x |
| 384 | 512 | 3.44% | 3.47% | 0.94% | 4.127x |
| 768 | 0 | 52.94% | 54.88% | 3.54% | 1.468x |
| 768 | 64 | 1.51% | 1.51% | 0.00% | 4.571x |
| 768 | 512 | 1.67% | 1.90% | 12.02% | 4.386x |

At step 64, plain low-rank won all 21 width-768 matrices. At step 512 it won
20 of 21. Kronecker was never selected. The mixed family adds no new training
algebra; its best observed saving over low-rank was 18.53%, below the frozen
25% distinction gate and absent in the cells with the strongest ideal gains.

## Important correction to the optimistic ledger

The 4x figures are **gradient-update oracle values, not executable whole-step
speedups**. CPAC proposed to make the forward weight itself a sum of fast
atoms. But initialization requires 53%-73% of dense algebra, and the ordinary
random initial matrix is already dense. Once a dense base exists, adding a
low-rank update does not remove the dense forward multiplication.

Therefore the post-step-64 projection in the preregistered JSON is an upper
bound that assumes the structured current gradient can retroactively make the
current weight structured. It cannot. This is the same distinction between a
low-rank update and a low-rank function. Any executable follow-up must leave
the dense forward charged and attack backward computation only.

## Retained finding

Exact from-zero LM **weight gradients undergo a strong phase change**:

- initialization is broad and expensive to approximate;
- after 64 steps, 90%-quality gradient directions cost only 5.92%, 3.79%, and
  1.51% at widths 192, 384, and 768;
- the required fraction decreases sharply with width;
- isotropic gradients remain near dense, so this is learned structure rather
  than a favorable oracle ledger.

This is consistent with existing gradual-rank/incremental-low-rank results and
is not itself novel. It does justify one different fatal question: whether the
token-by-channel **error matrices** entering each linear backward are low-rank
enough to avoid both dense backward matmuls while preserving an exact dense
forward and ordinary deployed model.

Artifacts:

- `results/compute-priced-algebraic-continuation-oracle.json`
- `results/compute-priced-algebraic-continuation-oracle.log`
- `results/compute-priced-algebraic-continuation-oracle-preregistration.md`
- `experiments/compute_priced_algebraic_continuation_oracle.py`
