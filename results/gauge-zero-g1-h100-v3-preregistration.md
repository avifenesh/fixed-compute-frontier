# Gauge-zero G1 H100 v3 preregistration

Status: frozen before execution  
Date: 2026-07-27

This is the final cost/correctness re-gate after one precision defect was found
by the trained-weight bridge: the LM prototype had read the reused coefficient
from the FP32 optimizer master while serving necessarily reads the BF16 dense K
weight. The prototype now casts that physical weight to the dense-projection
dtype before the fused FP32 epilogue. A trained-weight smoke bridge is bit-exact.

The frozen contract is BF16 dense projection and coefficient; quadratic shear
plus split-half RoPE in FP32; one BF16 final Q/K store. The deterministic pivot
is derived from shape and adds no tensor or metadata.

Protocol: H100; Torch 2.5.1+cu124; CUDA 12.4; Triton 3.1.0; hidden 4096;
32 query heads; 8 KV heads; head dimension 128; rows/trials 1/4000 and
8,32,128,512,2048/400 each; 50 warmups; seed 11939 plus the source-frozen
per-cell offset; randomized matched order; cold L2; CUDA events; 5,000 paired
bootstrap median-ratio resamples.

All gates must pass: independent candidate/control BF16 exactness; every median
ratio at most 1.02; every bootstrap upper 95% ratio at most 1.03; rows 1 and 8
upper 95% at most 1.02; equal 50,331,648-byte dense weight storage; zero pivot
metadata bits. Scope remains incremental overhead on this exact normal-GEMM
plus fused-RoPE executor, not a complete production serving stack.
