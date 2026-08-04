# BBCM ternary-residual endpoint H100 gate v2

Status: frozen after invalidating v1's scale format, before v2 execution.

## Repair

Both base and residual scales are rounded to BF16 before code assignment and decoding. The deployed residual alphabet is frozen to ternary `{-1,0,1}` in a two-bit field. Each of four Lloyd scale updates is BF16-rounded before the next assignment. Zero residual rows store BF16 `1` and remain exact.

The model, validation data, software, H100, 90 matrices, batch geometry, and frozen baseline anchor are identical to the INT8 v2 gate.

## Frozen gates

1. Exact environment, data, model, and baseline anchor.
2. INT8-plus-ternary relative NLL degradation is at most 0.05%.
3. Combined relative-L2 error is lower than INT8-only for every matrix.
4. Codes contain zero, positive, and negative values and all statistics are finite.
5. Every base and residual scale is exactly BF16-representable.

Passing establishes only an exact-format checkpoint reconstruction endpoint. The four routes are identical; specialization, packing, and runtime remain unproved.
