# Cycle-factor FFN: frozen algebra and 10M-token screen

## Hypothesis

Ordinary width-`M` SwiGLU buys `M` interaction features from `2M` independently learned factor rows:

`g = Gx`, `u = Ux`, `a_i = SiLU(g_i) u_i`, `y = Va`.

The cycle-factor FFN learns one width-`M` generator bank and reuses every row once as a gate and once as a neighboring value:

`h = Wx`, `a_i = SiLU(h_i) h_pi(i)`, `y = Va`.

`pi` is frozen as the next coordinate inside disjoint cycles of length four. There is no router, learned edge, gather bank, wider feature tensor, recurrent state, bias, auxiliary loss, or additional projection.

The candidate has `2DM` projection weights/MAC coefficients and an `M`-wide intermediate. Full SwiGLU has `3DM`. The strongest equal-`2DM` control is an independently factored SwiGLU with total width `2M/3`; at `M=1024,L=12`, its frozen per-layer widths are `(683,683,682)` repeated four times, so its aggregate projection count is exactly equal to the candidate.

Each narrow control layer multiplies its activation by the foldable constant `sqrt(M / width_l)`. This restores the full-width SwiGLU initialization variance without changing learned parameters, served MACs, or the three-matrix function class; omitting it would handicap the control by a factor `2/3` in output variance.

## Why the function is different

Near zero, a self-product feature has quadratic term `0.5 (w_i^T x)^2`, a rank-one positive-semidefinite quadratic form. A cycle feature has `0.5 (w_i^T x)(w_pi(i)^T x)`, generically a rank-two indefinite form. It therefore cannot be reduced to a self-square by a fixed coordinate permutation.

Compared with equal-cost narrow SwiGLU, the candidate trades factor independence for feature/readout count:

- narrow SwiGLU: `4M/3` independently learned factor rows, `2M/3` features/readout columns;
- cycle factor: `M` learned factor rows reused on graph edges, `M` features/readout columns.

The frozen Stage-0 test must verify equal raw parameters, graph regularity, indefinite cross forms, and sampled functional-Jacobian rank against narrow SwiGLU and self-product.

## Frozen language screen

- H100; Torch `2.5.1+cu124`; CUDA `12.4`; Transformers `4.57.6`
- scratch parallel-attention model: `D=384,M=1024,L=12`, six attention heads, two KV heads
- arms in order: full SwiGLU, exact-`2DM` narrow SwiGLU, width-`M` variance-matched self-product, width-`M` cycle factor
- common seed `815`; same token stream and optimizer batches
- sequence 512, microbatch 32, accumulation 2
- 305 optimizer steps = 9,994,240 prediction tokens
- 30-step warmup, AdamW `3e-4`, betas `(0.9,0.95)`, weight decay `0.1`, clip `1.0`
- 128 fixed validation batches at initialization and step 305

The cycle candidate advances to an untouched 50M-token replication only if all ledgers are valid and either:

1. it beats full SwiGLU by at least 0.05% with a wholly favorable paired 95% interval; or
2. it is noninferior to full SwiGLU within 0.05% (paired upper bound inside that margin) and beats both equal-`2DM` controls by at least 0.05%, with wholly favorable paired intervals.

The post-training self-edge and second-neighbor ablations are diagnostic, not evidence of superiority by themselves. A 10M pass is not a breakthrough; it only authorizes long-horizon replication and a fused scale H100 gate.
