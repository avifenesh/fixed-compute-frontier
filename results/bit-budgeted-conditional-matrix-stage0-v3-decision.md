# BBCM stage-0 v3 decision

Decision: pass, narrowly, as an internally consistent arithmetic witness.

- Exact single-GPU storage at width 14,320: 352,288,456 bytes versus 352,321,536 BF16 baseline bytes.
- Selected matrices plus router: 175,980,544 MAC/token versus 176,160,768 baseline matrix MAC/token.
- The two-dimensional witness uses the deployed ternary delta alphabet, routes all four inputs correctly, has zero routed MSE, and has MSE four for the best single bias-free linear row.
- Result SHA-256: `8a0c0f40e6e596fb7d8840b45511be19ca202f10baac4f057fc5e51de2f1ac90`.

This is not evidence against a nonlinear dense FFN. The four-expert cap is specific to `8 + K*2 = 16`. Tensor-parallel replication, decode work, grouped routing, and runtime are not funded by this single-GPU ledger.
