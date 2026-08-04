# TVE gauge-budget screen decision

## Decision: retain block-16; reject full-head growth

The preregistered promotion gate for full-head TVE failed. On seed `14321` at
9,994,240 training tokens, terminal validation NLL was:

| Arm | Slots per layer | Terminal NLL |
| --- | ---: | ---: |
| Canonical control | 0 active | 6.5462803 |
| TVE block-16 | 960 | **6.5280969** |
| TVE full-head-64 | 4,032 | 6.5294473 |

Full-head TVE remained better than control, but was worse than block-16 by
`0.0013504` NLL. The paired 64-batch 95% diagnostic interval was
`[0.0010083, 0.0016925]`, entirely in the wrong direction for promotion.
Block-16 beat control by `0.0181834` NLL, with interval
`[-0.0186185, -0.0177483]`.

This is a one-seed development screen, not evidence that block-16 is globally
optimal. It does reject the tested assumption that exposing 4.2x as many
gauge-reused nonlinear coordinates naturally improves the model. The retained
design principle is now narrower: spend a capped local feature budget and let
depth compose it, instead of maximizing within-head nonlinear connectivity.

All frozen integrity, data, equality, finite-value, coefficient, export, and
serving-bridge checks passed. Only the preregistered full-head superiority gate
failed. The result file is
`results/triangular-value-encoding-gauge-budget-screen.json`.
