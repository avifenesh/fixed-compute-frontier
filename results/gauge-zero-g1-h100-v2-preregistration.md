# Gauge-zero G1 H100 v2 preregistration

Status: frozen before execution  
Date: 2026-07-27

## Question

After unifying the quality prototype and serving epilogue, does deterministic
zero-pivot G1 preserve BF16 Q/K outputs while adding at most 2% incremental
latency at decode rows 1 and 8, and at most 3% across the full tested grid?

## Frozen numerical contract

Both paths consume ordinary BF16 dense QKV projection output. The candidate
loads its coefficient from the deterministic diagonal of the same dense K
weight, computes the quadratic shear and split-half RoPE in FP32, and rounds
once at the final BF16 Q/K store. The control performs the same fused FP32 RoPE
without the shear. There is no pivot atlas, sidecar, extra learned tensor, or
extra KV coordinate.

The H100 tests must directly bridge the LM projection implementation to this
Triton kernel for both control and candidate with exact BF16 equality.

## Frozen performance protocol

- hardware/runtime: H100, Torch 2.5.1+cu124, CUDA 12.4, Triton 3.1.0;
- shape: hidden 4096, 32 query heads, 8 KV heads, head dimension 128;
- normal BF16 `torch.mm` QKV projection followed by the same one-pass Triton
  epilogue in both arms;
- rows/trials: 1/4000, then 8, 32, 128, 512, 2048 each with 400 trials;
- 50 warmups per cell; seed 7907 plus the source-frozen per-cell offset;
- randomized matched arm order, cold L2 before each timed call, CUDA events,
  5,000 paired bootstrap median-ratio resamples.

Every gate must pass:

1. candidate and control are independently bit-exact to their FP32-one-round
   BF16 references in every cell;
2. every candidate/control median ratio is at most 1.02;
3. every bootstrap 95% upper ratio is at most 1.03;
4. rows 1 and 8 bootstrap 95% upper ratios are at most 1.02;
5. resident dense weight bytes are equal and pivot metadata is zero bits.

This gate covers incremental cost on this exact normal-GEMM plus fused-RoPE
executor. Passing does not establish parity with a production FlashAttention
stack or zero overhead for an entire serving request.
