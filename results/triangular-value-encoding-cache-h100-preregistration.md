# Triangular value encoding H100 admission gate

This document freezes the serving-cost gate before its formal run.

## Fixed candidate

- Canonicalize each GQA value/output gauge with deterministic sign-canonical RQ.
- Reuse only the strict-lower entries of eight 16 by 16 blocks in the representative output-head block.
- Encode each newly projected value once, immediately before the required KV-cache write.
- Use BF16 projected values, BF16 signed-square features, BF16 physical coefficients scaled by 1/0.125, FP32 tensor-core accumulation, a BF16 delta boundary, FP32 residual addition, and one BF16 cache store.
- Use packed BF16 QKV projection and FP32 one-store split-half RoPE for both arms.
- Do not add parameters, persistent buffers, KV elements, cache bytes, or metadata.

## Fixed H100 screen

- Device: NVIDIA H100 80GB HBM3.
- Rows: 1, 8, 32, 128, 512, and 2048.
- Per cell: 30 warmups and 200 paired randomized trials.
- Before each timed arm, touch a 128 MiB BF16 tensor with arithmetic read/write and synchronize.
- Timed scope: BF16 QKV GEMM plus Q/K RoPE plus the required K/V cache write.
- Compare the candidate with the canonical control using the same cache shape, dtype, task grid, and output buffers.
- Bootstrap 5,000 matched median-ratio replicates.

## Frozen pass conditions

Every row cell must pass all conditions:

1. Control QKV, K cache, and V cache are bit exact to the independent reference.
2. Candidate QKV and K cache are bit exact to the independent reference.
3. Candidate V-cache maximum absolute error is at most 0.015625 and the nonlinear delta is nonzero.
4. The upper 95 percent bootstrap bound for candidate/control latency is at most 1.02.
5. Added learned-weight bytes, KV bytes, and metadata bits are zero.

The source, tests, preregistration, and runtime are hash-bound before the formal run. Passing this gate admits a matched LM-quality pilot; it is not itself a capability claim.
