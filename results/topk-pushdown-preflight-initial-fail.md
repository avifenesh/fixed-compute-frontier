# CPKV-TOPK-001 initial timing preflight

Status: **closed implementation failure; no model result**

The first length-stratified CPU projection was written to
`results/topk-pushdown-preflight.json` before the harness optimization pass.  It
used seed 1701, 512 base samples plus missing length strata, two measured
training passes, two development passes, and the unchanged 5.7 CPU-hour safety
gate below the six-hour hard run limit.

| family | projected all-arm CPU hours | result |
|---|---:|---|
| D1 | 4.0090 | pass |
| A1 | 4.3250 | pass |
| A2 | 6.2494 | fail |

A2 decomposition:

| arm | projected CPU hours |
|---|---:|
| S32 | 2.8577 |
| LINK32 | 1.5524 |
| DIRECT3 | 0.7909 |
| HARD3 | 0.6558 |
| RNN32 | 0.2038 |
| MLP | 0.1889 |

This does not test the candidate's capability.  It proves that the first
research implementation lacked enough runtime margin, concentrated in the
differentiable S32 beam and LINK32.  The gate and training protocol remain
unchanged; a replacement receipt is admissible only after algebra-preserving
optimization plus differential output/loss/gradient tests.

Initial receipt source hashes:

- training: `dafc4cacb1d2f3cb42474b7851ba3531fa5adbf97de4adbeb5a70f0c43ebeb69`
- locked evaluator: `42f60de4f711973d621c490ef1072e4a42dff62e185bb9f67cda4fedd090b104`
- preflight: `e3dfe88bfc462bd7636ddb7a62cc3800f33efd50856ecc4f2fe77f28ba9c9d74`
