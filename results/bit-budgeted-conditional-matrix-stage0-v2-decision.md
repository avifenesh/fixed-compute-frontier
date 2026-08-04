# BBCM stage-0 v2 decision

Decision: **pass the repaired abstract ledger; remain unproven as an architecture.**

## Exact ledger

- BF16 target FFN: 352,321,536 resident matrix bytes; 176,160,768 matrix MAC/token.
- BBCM at aligned `M=14320`, including five sets of per-output BF16 scales, a BF16 `4096 x 4` router, and four router biases: 352,288,456 bytes.
- Candidate FFN matrices plus router: 175,980,544 MAC/token; four router-bias additions are recorded separately.
- Slack: 33,080 bytes and 180,224 MAC/token.
- Payload allocation: shared INT8 base plus four 2-bit deltas, exactly 16 bits per matrix coordinate.

The repaired affine router realizes all four frozen scalar regions, and explicit base/codebook decoding represents the artificial piecewise map exactly where one affine map cannot.

## What this proves

The same 16-bit code entropy can parameterize a context-indexed family instead of one context-independent scalar. It is a genuine conditional function-class change at equal abstract resident bits and active matrix MACs.

## What this does not prove

- The witness is deliberately artificial and says nothing about language loss.
- Per-output scales make the decoded composite a floating-point weight. A one-MMA implementation needs fused code load, two dequantizations, addition, and BF16/FP8 tensor-core feed. This is not an INT8 GEMM.
- Decode operations, token dispatch, scratch, and physical weight traffic are not funded by the MAC ledger.
- DeltaMoE is a direct conceptual collision: shared bases plus routed low-bit expert deltas already exist. The remaining distinction is exact dense-BF16 bit/MAC budgeting, end-to-end training under that budget, dense two-bit deltas, and a possible single-mainloop decoder.

## Correct quality condition

In a local Fisher/Hessian model, the candidate can win only when

`S > Q_joint + R + W`

where:

- `S` is the between-route loss paid by the best single shared map;
- `Q_joint` is the loss from forcing all ideal route maps into one jointly feasible INT8-base plus four-INT2-delta codebook, including interactions among SwiGLU's three matrices;
- `R` is routing-confusion loss;
- `W` is the loss from shrinking hidden width from 14,336 to 14,320.

Hardware is a separate gate: fused decode, dispatch, and traffic must keep target serving latency within the fixed-cost contract.

Evidence: `bit-budgeted-conditional-matrix-stage0-v2.json`, SHA-256 `2e0e2ab0faa22051f252be4c62b63b4630744045b664290ac14c3feaa5d519e8`.
