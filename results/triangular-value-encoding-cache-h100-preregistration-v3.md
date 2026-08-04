# TVE serving-aligned H100 admission gate v3

V2 exposed rare legal reduction-order differences between the PyTorch algebra path and the Triton serving path. V3 removes ambiguity from the model contract: candidate training and evaluation use the exact Triton serving forward, with gradients supplied by the algebraically equivalent PyTorch expression. The forward values consumed by attention therefore match exported serving values exactly.

## Frozen executor and screen

- Packed BF16 QKV, FP32 one-store RoPE, and encoded V written to both QKV and the unchanged BF16 KV cache.
- Block size 16; BF16 signed-square features and scaled coefficients; FP32 tensor-core accumulation; BF16 delta; FP32 residual addition; BF16 stores.
- H100 80GB; rows 1, 8, 32, 128, 512, and 2048.
- Thirty warmups, 200 randomized paired trials, 128 MiB arithmetic L2 touch, and 5,000 bootstrap median-ratio replicates per cell.

## Frozen gates for every cell

1. Control QKV and caches are bit exact.
2. Candidate QKV and V cache are bit exact to the serving-aligned model forward; K cache is bit exact.
3. The candidate nonlinear delta is nonzero.
4. Difference from the algebraically equivalent PyTorch reduction has max error at most 0.015625, mean at most 1e-8, and mismatch fraction at most 1e-5.
5. Propagated O-output max error is at most 0.015625, mean at most 1e-7, and proxy NLL difference at most 1e-6.
6. The upper 95 percent candidate/control latency ratio is at most 1.025.
7. No learned-weight bytes, KV bytes, or metadata bits are added.

Sources, tests, preregistration, and runtime are hash-bound before execution. Passing admits a matched LM discovery pilot; capability still must be demonstrated independently.
