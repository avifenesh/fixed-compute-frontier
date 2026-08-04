# Temporal factor FFN: frozen 10M-token successor

## Key shift

Arbitrary same-token factor reuse was active but harmful. This successor reuses a factor with independent semantics: the previous token's generated value.

For `h_t = W x_t`, disjoint group-4 route `pi`, and one learned angle per 32 channels:

`v_ti = cos(theta_g) h_ti + sin(theta_g) h_(t-1),pi(i)`

`a_ti = s(theta_g) SiLU(h_ti) v_ti`

`y_t = V a_t`.

At a sequence boundary, `h_(t-1)` is defined as `h_t`. The angle is parameterized as `theta=(pi/2)tanh(raw)` and initialized at zero, so the initial forward is the width-`M` self-product control. The frozen moment scale is

`s(theta) = SELF_SCALE / sqrt(cos(theta)^2 + SELF_SCALE^2 sin(theta)^2)`.

It equals `SELF_SCALE` at the self endpoint and 1 at the independent temporal cross-factor endpoint under the standard-normal initialization model.

There are 32 angles per layer, 384 total. Served weights are 33,035,520 versus 37,753,728 for full SwiGLU: 12.498% smaller overall. Dense FFN projection weights/MAC coefficients remain exactly `2DM`, two thirds of full SwiGLU. Decode state is one BF16 width-`M` vector per layer: 24,576 bytes per request at this screen shape. Prefill uses a vectorized one-token shift, not a scan.

## Frozen experiment

- exact predecessor controls and validation batches from `cycle-factor-ffn-lm-screen.json`, SHA-256 `c66dfe6f5a6e3a1c4032932b9cbd2c491812af2e86a314e9cc6d46e83aba4029`
- same H100/runtime, data manifest, seed 815, 305 steps, batches, optimizer, learning-rate schedule, and 9,994,240 prediction tokens
- existing weights use the predecessor's AdamW settings; only the new angle parameters have zero weight decay
- no angle/group/route/lag sweep

Promotion requires:

- initial candidate NLL within `2e-5` of the frozen self-product endpoint;
- candidate noninferior to full SwiGLU within 0.05%, with the paired interval upper bound inside that margin;
- candidate at least 0.05% better than both equal-2DM controls, with wholly favorable paired intervals;
- finite training and nonzero learned temporal sine magnitude;
- forcing all angles to zero after training worsens NLL by at least 0.01%, with a wholly favorable paired interval.

A pass authorizes an untouched 50M-token replication and a real decode-state/fused H100 gate. It is not yet a breakthrough claim.
