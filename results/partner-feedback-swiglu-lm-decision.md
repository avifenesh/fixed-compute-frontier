# Gauge-exposed partner-feedback SwiGLU — 50M-token decision

Decision: **reject the fixed partner-feedback recipe.  Do not tune its
coefficient, matching, bound, or training length.**

The mechanism passed algebra, exact count, export, and H100 serving gates.  It
then failed every terminal language-capability and causal-ablation gate.

## Terminal validation NLL

| Arm | NLL | Relative to raw SwiGLU |
|---|---:|---:|
| Raw status-quo SwiGLU | 5.535243854 | — |
| Carrier-null parameterization | 5.535241853 | 0.000036% better |
| Self-feedback | **5.535193104** | 0.000917% better |
| Partner gate-only | **5.535192754** | 0.000923% better |
| Full scale-sensitive partner feedback | 5.535271119 | 0.000493% worse |

Carrier-null matched raw SwiGLU within `0.0000020` NLL, proving that the clean
replacement-scalar training chart did not create a material baseline penalty.
Self-feedback and partner-gate-only agreed within `0.00000035` NLL.  Their tiny
gain is about 54 times below the frozen `0.05%` baseline gate.

Full partner feedback was best at 10M tokens (`6.309270211` versus
`6.309430279` raw), but reversed by 50M.  Terminal paired candidate-minus-raw
mean was `+0.00002727` with interval `[-0.00000493,+0.00005946]`.  It was
decisively worse than partner-gate-only: mean `+0.00007837`, interval
`[+0.00004633,+0.00011040]`.

## Causal ablations

| Evaluation of trained candidate | NLL |
|---|---:|
| Full partner | 5.535271119 |
| Feedback zeroed | 5.535273559 |
| Partner replaced by self | **5.535227615** |
| Partner map shifted | **5.535248749** |

Zeroing feedback changed loss by only `0.00000244`; its paired interval crossed
zero.  Replacing the partner with self-feedback improved paired block loss by
`0.00004350`, with the full-partner-minus-self interval entirely positive:
`[0.00002192,0.00006509]`.  The trained model therefore did not organize a
useful partner-specific interaction.

The carrier was live rather than dead: terminal mean absolute `lambda` was
`0.005655`, clip occupancy was `8.76%`, and maximum scale condition factor was
only `1.0469`.  This is a negative preference result, not a saturation or
optimization-failure result.

## What is retained

1. A SwiGLU scale gauge can be converted into real functional directions with
   no added serialized weights or dense MACs.
2. A fixed partner second-gate pass is hardware-cheap: measured full-path H100
   overhead was `0.04%` to `0.42%` on the frozen grid.
3. Those facts do not create useful language capacity.  Across Reflex,
   triangular microdepth, tile re-pairing, and now partner feedback, richer
   within-token activation algebra has produced only negligible or negative
   NLL movement.

This changes the admission rule.  Do not propose another within-token
activation interaction merely because it is a strict, cheap function-class
superset.  A successor must change what information is preserved or exchanged
by the expensive projections, or first establish a task-side unmet structure
that the new algebra targets.

Result SHA-256:
`4b1ea7bb272c31a5afc287da9bf5d57d583cd9bb5b938948c04d9f991df8b600`.

