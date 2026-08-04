# Self-product scale H100 eager decision

The corrected resident-BF16 eager gate is a formal no-go.

- Cached decode candidate ratio: 0.962 at batch 1 and batch 8.
- Prefill candidate ratio: 0.990 at batch 1, 0.943 at batch 8, and 1.044 at
  batch 32. Both wall and CUDA clocks fail the batch-32 gate.
- Static BF16 parameters are exactly equal at 102,247,040 / 204,494,080 bytes.
- Candidate allocated and reserved peaks exceed SwiGLU in every measured
  decoder-core and service cell.
- The candidate performs 50% more scalar SiLUs/products. The eager
  implementation also materializes multiple width-2,688 temporaries.

The 100M-token scale LM run is cancelled under this artifact. One separately
preregistered fused-serving refinement is allowed because in-place activation
changes the measured runtime/memory implementation, not the learned function
or the failed gate. If the fairly fused candidate still fails any original
latency/memory cell, close the self-product branch.

Eager result SHA-256:
`7e4c7b6b1686a1631c1b90aad8437141ff0299dd15c312aeb76fe62f27415fc8`.
