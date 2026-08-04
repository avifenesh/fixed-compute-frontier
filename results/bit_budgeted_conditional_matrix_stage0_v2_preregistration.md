# Bit-Budgeted Conditional Matrix stage-0 v2 preregistration

Status: frozen after invalidating v1, before v2 execution.

## Repair

The router is affine, `logits_e(x)=a_e*x+b_e`. Four BF16 biases are included in the storage ledger. The scalar witness freezes:

- router slopes `[-3,-1,1,3]` and biases `[0,4,4,0]`, whose upper envelope selects four consecutive regions;
- shared INT8 code `Q0=0` with scale 1;
- four 2-bit codebook values `[-3,-1,1,3]` with shared delta scale `0.5`, decoding route weights `[-1.5,-0.5,0.5,1.5]`;
- inputs away from router ties and independent expected route labels.

The predicted route must come from `argmax(a_e*x+b_e)` and equal the independent expected label. The decoded route weight is then multiplied by `x` once.

## Frozen gates

1. Candidate payload, BF16 per-output scales, BF16 `D x 4` router weights, and four BF16 router biases fit under baseline BF16 FFN bytes.
2. Candidate FFN matrix MACs plus router MACs do not exceed baseline FFN matrix MACs. Decode, dispatch, additions, scratch, and physical traffic are explicitly outside this stage-0 MAC count and require later measurement.
3. Natural cap `K=floor((16-8)/2)=4`.
4. Affine router produces the frozen four routes.
5. Explicit INT8-base plus 2-bit-code decoding represents the piecewise target exactly with one scalar multiplication after routing; the best single affine predictor is inexact.
6. Raw code cardinality remains `2^16`.

Passing establishes only the exact abstract bit/MAC ledger and a realizable artificial separation. Independent base/delta scales make the decoded weight a floating-point value; a one-MMA implementation would require fused load-time decode/add before BF16/FP8 tensor-core input. It is not an INT8 GEMM and is not established here.

