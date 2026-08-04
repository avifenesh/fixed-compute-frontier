# Feedback-SwiGLU H100 execution preregistration

Status: frozen before execution.

This gate tests whether a stock compiled PyTorch executor can hide the candidate's low-rank feedback cost. It does not test model quality.

## Frozen grid

- Device must contain `H100`.
- BF16, `D=4096`, baseline `M=14336`, `q=8`, 8 fixed feedback groups.
- Equal-parameter candidate uses the largest multiple of 64 (8 groups times tensor-core alignment 8) not exceeding `floor(3DM / (3D+2q))`: `M=14272`.
- Token batches: `{1,8,32,128}`.
- 40 warmups and 200 randomized, interleaved CUDA-event samples per cell.
- Compile all three paths with `torch.compile(fullgraph=True, mode="reduce-overhead")`.

## Paths

1. Baseline SwiGLU.
2. Same-width Feedback-SwiGLU, isolating executor overhead at only `0.1302%` extra MACs/parameters.
3. Equal-parameter Feedback-SwiGLU, paying for `P,R` by reducing hidden width from `14336` to `14272`.

The hidden features are partitioned into 8 fixed groups. Each group computes its own rank-8 state and refinement. Groups execute in parallel. A tensor-parallel deployment assigns whole groups to shards, so the candidate adds no pre-down-projection collective; a global-state variant is explicitly out of scope.

## Frozen gates

All must pass:

1. Complete valid protocol and frozen grid.
2. With `R=0`, compiled same-width feedback and baseline have maximum row-relative difference at most `0.2%`.
3. Same-width feedback median latency is at most `1.02x` baseline for `B in {1,8}` and at most `1.05x` on every cell.
4. Equal-parameter feedback median latency is at most `1.02x` baseline for `B in {1,8}` and at most `1.05x` on every cell.

Failure closes only the stock PyTorch executor. A custom fused activation/feedback kernel is permitted as a materially different implementation, but it must use this same grid and thresholds.
