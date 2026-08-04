# TVE prefill-compatible H100 admission gate v2

This supersedes v1 before the v2 run. V1 wrote encoded values only to the cache and therefore did not cover prefill consumers that read the QKV V slice.

## Frozen executor

- Packed BF16 QKV GEMM.
- FP32 split-half RoPE with one BF16 store.
- The candidate computes block-16 triangular value encoding while writing the required KV cache and writes the same encoded V to the QKV V slice.
- BF16 projected values, BF16 signed-square features, BF16 scaled coefficients, FP32 tensor-core accumulation, BF16 delta boundary, FP32 residual addition, and BF16 final stores.
- Control and candidate have identical learned tensors, persistent state, cache shape, cache dtype, and cache bytes.

## Frozen screen

- NVIDIA H100 80GB HBM3.
- Rows 1, 8, 32, 128, 512, and 2048.
- Thirty warmups and 200 randomized paired trials per cell.
- Arithmetic read/write touch of a 128 MiB BF16 tensor plus synchronization before each timed arm.
- Five thousand matched bootstrap median-ratio replicates.

## Frozen gates for every cell

1. Control QKV, K cache, and V cache are bit exact.
2. Candidate K cache is bit exact.
3. Candidate QKV and V cache each have max absolute drift at most 0.015625, mean drift at most 1e-8, and mismatch fraction at most 1e-6. This admits only rare final-BF16 differences from legal tensor-core reduction ordering.
4. Propagating that drift through a fixed attention mixture and O projection has max error at most 0.015625 and mean error at most 1e-7; proxy NLL changes by at most 1e-6.
5. The nonlinear candidate delta is nonzero.
6. The upper 95 percent latency ratio is at most 1.025.
7. Added learned-weight bytes, KV bytes, and metadata bits are zero.

The formal source, executor, attention module, tests, preregistration, and runtime are hash-bound before execution. A pass admits a one-seed matched LM discovery pilot, not a breakthrough claim.
