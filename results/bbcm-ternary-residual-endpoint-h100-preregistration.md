# BBCM ternary-residual endpoint H100 gate

Status: frozen after the INT8-only result, before execution.

## Question

Can one 2-bit delta plane reconstruct enough of the INT8 shared-base error to begin routed upcycling near the BF16 checkpoint rather than paying the measured 0.1406% NLL tax?

## Frozen quantizer

For every output row of every FFN gate/up/down matrix:

1. Form the frozen per-row symmetric INT8 base `B`.
2. Let residual `R = W - B`.
3. Quantize `R` to ternary codes `{-1,0,1}`, stored in two bits.
4. Initialize the per-row delta scale at `2/3 * max(abs(R))`, then perform four deterministic Lloyd updates: assign nonzero when `abs(R) >= scale/2`, then replace scale by the mean absolute assigned residual.
5. Decode `W_hat = B + scale * ternary_code`.

The same decoded delta is copied to all four routes at upcycling initialization; no route has an endpoint advantage. The other three stored copies are not used in this gate.

## Frozen protocol and gates

The model, validation tokens, H100/software environment, batch geometry, and baseline anchor are identical to the INT8-only gate.

1. Exact environment/data/model protocol and 90 quantized matrices.
2. Independently measured BF16 baseline NLL matches `2.8295718903541565` within `1e-7`.
3. INT8-plus-ternary-residual relative NLL degradation is at most 0.05%.
4. Combined weight relative-L2 error is strictly lower than INT8-only error for all 90 matrices.
5. Statistics are finite and ternary codes include zeros and nonzeros.

Passing establishes a low-damage initialization only. It does not show expert specialization, final quality, or a fast decoder.

