# Budget-neutral soft record T18 — H100 runtime preregistration

Status: frozen before execution  
Date: 2026-07-31

## Purpose

T18 Stage 0 passed exact parameter/MAC and operator contracts, but the dense
reference materializes a 2,406-way softmax.  This gate measures whether the
unfused PyTorch operator is already noninferior on the retained H100.  Training
is not admitted if the served runtime surface fails.

## Frozen artifacts

- Stage-0 result SHA-256:
  `ba2ac04c370a4c000b513438ac30ce5b5fccd0d6856ba98861198ea83db47126`;
- Stage-0 source SHA-256:
  `25c01d95905ffb6026cefa855d9a882f4deecd76a9ea63e2f9e9ef0e1908c628`;
- model parameters per arm: 36,577,152;
- replaced matrix MACs per token per arm: 2,359,296;
- precision: BF16 autocast with FP32 parameters;
- device: the retained H100/H200 only.

## Frozen benchmark surfaces

Block-local pair, excluding identical attention:

- `(batch, tokens) = (1,1), (1,128), (8,128), (32,128)`.

Whole model, including tied output logits:

- `(1,1), (1,128), (8,128)`.

The block-local baseline is two residual width-1,024 SwiGLUs with two RMSNorms.
The candidate is one residual 2,406-key/value record memory followed by one
residual width-444 SwiGLU with two RMSNorms.  Their parameter and matrix-MAC
counts must be exactly equal.

The whole-model arms are the exact Stage-0 baseline and candidate classes.
Weights are random because this gate measures shapes/operators, not quality.

## Timing protocol

- seed: 10,109;
- five interleaved trials per arm/surface, alternating arm order;
- 20 warmups before each measured trial;
- 200 measured iterations when `batch*tokens <= 128`, otherwise 50 for
  block-local;
- 50 measured iterations when `batch*tokens <= 128`, otherwise 10 for whole
  model;
- CUDA events around the complete loop, one synchronization after the end
  event;
- report every trial, median milliseconds, and candidate/baseline ratio;
- report peak incremental allocated HBM from one synchronized inference after
  allocator warmup.

No CUDA graph, `torch.compile`, custom kernel, selective reporting, or retry is
allowed.

## Gates

1. environment is H100/H200 and every result is finite;
2. block-pair parameters and matrix MACs remain exactly equal;
3. candidate block median latency is at most 1.05x baseline on every surface;
4. candidate whole-model median latency is at most 1.05x baseline on every
   surface;
5. candidate peak incremental allocated HBM is at most 1.05x baseline on every
   corresponding surface;
6. source, preregistration, and Stage-0 hashes are emitted.

Passing admits from-scratch training design.  Failure identifies a served
implementation blocker: a fused online-softmax record kernel must pass the same
gate before training.  Runtime failure does not justify ignoring non-matrix
cost or claiming equality from MAC counts.
