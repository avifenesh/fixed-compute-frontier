# Fused Feedback-SwiGLU H100 preregistration

Status: frozen before execution.

The stock executor failed by 4.2% to 5.9%. This materially different executor fuses the entire hidden feedback operation between the existing up/gate and down GEMMs.

## Frozen implementation

- One Triton baseline kernel computes `SiLU(g) * u`.
- One Triton candidate kernel per token and fixed feedback group computes `z0`, all 8 compressed state coordinates, `R tanh(h)`, and refined `z1` without materializing `z0`, `h`, or `delta`.
- CUDA Graphs capture the complete two-up-GEMM, activation/feedback, down-GEMM paths for both baseline and candidate.
- No approximation and no change to the candidate algebra.

## Frozen grid and gates

Use the same device, BF16 shapes, 8 groups, widths, batches, 40 warmups, 200 randomized samples, and thresholds as `feedback-swiglu-h100-preregistration.md`:

- same-width and equal-parameter candidate at most `1.02x` baseline for token batches 1 and 8;
- at most `1.05x` baseline for every batch in `{1,8,32,128}`;
- zero-feedback fused output maximum row-relative error at most `0.2%` versus the fused baseline;
- source/prereg hashes and the exact Triton launch configuration must be recorded.

Passing this gate establishes only deployable single-H100 latency plausibility. It does not establish training speed, tensor-parallel speed, or model quality.

