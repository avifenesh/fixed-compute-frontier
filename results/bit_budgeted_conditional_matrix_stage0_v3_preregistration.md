# Bit-Budgeted Conditional Matrix stage-0 v3 preregistration

Status: frozen after identifying v2's codebook mismatch, before v3 execution.

## Repair

V3 uses the deployed two-bit ternary alphabet `{-1,0,1}` from the checkpoint endpoint. One physical bit pattern is unused; no `2^16` semantic-cardinality claim is made. The four-route witness is two-dimensional so a shared per-output scale does not trivialize four scalar values.

- Inputs are `(-2,0)`, `(0,-2)`, `(0,2)`, `(2,0)`.
- Bias-free router rows and ternary delta rows are respectively `(-1,0)`, `(0,-1)`, `(0,1)`, `(1,0)`.
- The INT8 base row is `(0,0)` and all scales equal one.
- Top-1 routing must select routes `0,1,2,3` and output two for every input.
- The best single bias-free linear row must have MSE four, while the routed map has zero MSE. A general nonlinear FFN can fit the witness; no superiority over SwiGLU is claimed.

## Frozen gates

1. The full candidate including BF16 scale sets and affine router fits below the single-GPU BF16 FFN storage ledger at aligned width 14,320.
2. Selected-matrix plus router MACs do not exceed baseline matrix MACs. Decode, dispatch, physical traffic, and tensor-parallel replication remain outside this gate.
3. The specific allocation `8 + K*2 = 16` caps `K` at four. This is allocation-specific, not a natural universal cap.
4. The exact affine router realizes all four frozen routes.
5. INT8-base plus deployed ternary deltas exactly realize the target and separate it from one bias-free linear row.
6. Physical payload is exactly 16 bits per coordinate; semantic delta combinations are `3^4`, not `2^8`.

Passing repairs only internal consistency of the arithmetic witness. It does not establish LLM quality, runtime, or novelty.
