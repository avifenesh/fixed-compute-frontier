# Depth-curvature residual: frozen 10M-token screen

## Algebra

Let an ordinary parallel Transformer layer produce raw update

`Delta_l = Attention_l(Norm(x_l)) + FFN_l(Norm(x_l))`

and baseline output `x_(l+1)=x_l+Delta_l`.

The candidate retains the previous raw update for one layer and uses

`x_(l+1) = x_l + Delta_l + beta_l (Delta_l - Delta_(l-1))`.

The first layer uses the ordinary Euler update. `beta_l=0.5 tanh(raw_l)`, with one raw scalar per 32 hidden channels and zero initialization. It therefore starts exactly at the baseline. The previous update is ephemeral depth state, not request-persistent state or a learned tensor.

The matched residual-scale control uses the same grouped scalars but

`x_(l+1) = x_l + Delta_l + beta_l Delta_l`.

This separates reuse of the previous update from simple learned LayerScale. Both arms add 144 scalar parameters (0.000381% of the 37.75M model), no dense projection weights/MACs, and O(D) pointwise work per layer. The candidate temporarily retains one `[batch,tokens,D]` update until the next layer.

## Frozen screen

- same H100/runtime, seed 815, data, 305-step optimizer schedule, and 9,994,240 prediction tokens as the valid cycle-factor predecessor
- frozen full-SwiGLU baseline from predecessor SHA-256 `c66dfe6f5a6e3a1c4032932b9cbd2c491812af2e86a314e9cc6d46e83aba4029`
- arms: residual-scale control, depth-curvature candidate
- grouped scalar parameters have zero weight decay; all baseline weights retain the original AdamW treatment
- no coefficient bound, grouping, sign, schedule, or update-definition sweep

Promotion requires the candidate to improve full SwiGLU by at least 0.05% and the scale control by at least 0.025%, both with wholly favorable paired 95% intervals. Its learned coefficient magnitude must exceed 0.001 and zeroing coefficients after training must hurt by at least 0.01%. A pass authorizes long-horizon replication and a memory/latency gate.
