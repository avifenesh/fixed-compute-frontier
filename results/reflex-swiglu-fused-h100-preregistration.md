# Reflex-SwiGLU fused H100 preregistration

Status: frozen before execution.

## Frozen implementation and grid

- BF16 on H100, `D=4096`, baseline `M=14336`.
- Same-width candidate has one learned effective `alpha` per hidden feature.
- Equal-parameter candidate aligns `floor(3DM/(3D+1))=14334` down to 8: `M=14328`.
- A single Triton kernel fuses `z0`, clip, self-feedback, and the refined activation. No intermediate is materialized.
- CUDA Graphs capture both up/gate GEMMs, the activation kernel, and the down GEMM.
- Batches `{1,8,32,128}`, 40 warmups, 200 randomized interleaved CUDA-event samples.

## Frozen gates

1. Complete valid protocol and source/preregistration hashes.
2. `alpha=0` fused output maximum row-relative error at most `0.2%` versus fused SwiGLU baseline.
3. A nonzero-`alpha` fused activation test covering `z0<-1`, `|z0|<1`, and `z0>1` has maximum row-relative error at most `1%` versus a float32 PyTorch reference.
4. Same-width and equal-parameter median latency at most `1.02x` baseline for batches 1 and 8.
5. Same-width and equal-parameter median latency at most `1.05x` baseline for every frozen batch.

Passing establishes single-H100 serving plausibility only. It does not establish training speed or model quality.
