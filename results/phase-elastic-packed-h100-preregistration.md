# Phase-elastic packed H100 physical gate

Status: **protocol drafted; run blocked by Vast credit**  
Date: 2026-07-28

## Frozen object once compiler smoke passes

- exact 50M artifact SHA-256
  `d7bb0164c71ee0c01cb80dd68711302a2776d86fc1fcd99bb25143c6c8965402`;
- full non-MIG 80GB H100 SXM with 132 SMs, Torch 2.5.1+cu124, CUDA 12.4,
  isolated Triton 3.6;
- dimensions `D=384`, base width 1,024, branch width 1,472;
- rows `{1,8,32,128,512,2048}`;
- one monolithic 128-byte-aligned 2,338,816-byte layer blob;
- no globally materialized expanded weights.

## Candidate executor

Two CUDA-graphed kernels:

1. fused gate+up: reconstruct INT8+ternary or W4 fragments in registers,
   issue one BF16 MMA per effective weight matrix, preserve the artifact's
   BF16 linear/SiLU/multiply boundaries, and store only the activated
   intermediate;
2. fused down: reconstruct base and optional branch fragments, preserve the
   artifact's independent BF16 base/branch output rounding and BF16 addition,
   and write one result.

INT8 and ternary may not execute as separate GEMMs.  Doing so changes the base
from 1x to 2x Tensor work and the full path from 2.4375x to 3.4375x, invalidating
the tested algebra.

Ternary and W4 streams must be loaded at their compact byte cardinality once
per tile, then expanded by register gather/bit extraction.  Per-coordinate
duplicate packed-byte loads do not satisfy the physical traffic claim even if
the coalescer happens to merge some transactions.

## Controls and timing

- the dense denominator is the faster same-run p50 of two frozen BF16 controls
  at each row: a custom fused Triton gate/up/SwiGLU/down path and a packed
  gate+up cuBLAS/PyTorch path;
- explicit unpacked-artifact oracle for base and full paths;
- 200 warmups and 1,000 CUDA-event replays per cell;
- candidate and controls are randomly interleaved once per paired timing round;
- graph-warm timing and cold-L2 timing after a 128 MiB flush are both reported;
- both a single-layer diagnostic and a primary 12-unique-layer streamed graph
  are measured; only the streamed ratios carry performance gates;
- clocks, power, temperature, and P-state are reported before and after each
  timing block;
- p50, p99, and paired-bootstrap 95% ratios are reported.

The previous-instance dense CUDA-graph snapshot is diagnostic only:
`{1:13.933,8:19.141,32:19.596,128:22.028,512:24.747,2048:39.900}` microseconds.
The selected dense single-layer control must be no more than 5% slower than
that snapshot at every row; otherwise the denominator is disqualified.

## Frozen correctness/storage gates

1. candidate output is finite, relative L2 `<=1e-3`, and max error `<=2` BF16
   ULPs versus explicit artifact decode at every row and mode;
2. monolithic persistent bytes `<=2,359,296` per layer, including all codes,
   scales, and padding;
   all 12 blobs must be physically allocated, 128-byte aligned, exact-sized,
   and every segment must share its blob storage at the frozen offset; after
   deleting the loaded artifact source and emptying the cache,
   `torch.cuda.memory_allocated` must equal exactly 12 logical blobs (allocator
   reserved bytes are reported separately);
3. no expanded-weight global buffer or hidden repack/scheduler workspace;
4. full model-plus-workspace is no larger than dense at decode rows 1, 8, 32;
5. p99 ratio is no more than p50 ratio plus 0.10.
6. selected packed kernels report zero PTXAS spills, no PTX local-memory
   declaration, and an MMA tensor-core instruction; chosen configs, registers,
   shared memory, and PTX hashes are recorded.
7. current clocks are at least 80% of max SM and 95% of max memory clock before
   and after each block, with unchanged P-state.

The unpacked oracle and dense controls necessarily materialize expanded BF16
weights in this benchmark process.  They are forbidden as candidate-kernel
arguments and excluded from the candidate ledger.  A compiler resource audit
for local-memory spills plus CUDA-graph pool accounting is required after the
prototype timing; prototype gates alone cannot prove the no-extra-VRAM claim.

## Engineering admission

- base paired-bootstrap upper 95% ratio `<=1.10x` dense at every row;
- full upper 95% ratio caps at rows `{1,8,32,128,512,2048}` are
  `{1.10,1.10,1.15,1.35,2.10,2.56}`.

## Breakthrough pass

- in both warm and cold-L2 12-layer streams, base p50 `<=1.00x` dense at rows
  128 and 512 and `<=1.05x` at 2,048, with each upper 95% bound no more than
  0.02 above its point cap;
- in both warm and cold-L2 12-layer streams, full p50 `<=1.00x` dense at rows 1
  and 8 and `<=1.10x` at 32, with each upper 95% bound no more than 0.02 above
  its point cap;
- the 12-layer FFN-only estimate for `512` prompt tokens plus `128` batch-1
  decode tokens has p50 `<=1.00x` and upper 95% `<=1.02x` dense in both warm
  and cold-L2 streams;
- all correctness, storage, workspace, and engineering gates pass.

This is deliberately a single-layer FFN physical gate.  It is not labeled an
end-to-end model benchmark.  Passing it authorizes compiler-spill inspection
and a model-integrated phase-switch run; only those together may set
`physical_breakthrough_proven=true`.

Kill the no-cost claim if a separate INT8/ternary MMA is required, the packed
layout exceeds BF16, expanded weights reach global memory, or a CuTe RS
implementation still misses full rows 1/8 by more than 10%.

This file is not yet externally hash-pinned because the H100 exited during
compiler smoke and no replacement could be rented.  Pin source and protocol
only after the smoke compiles on the resumed H100.  The pin includes the source,
this protocol, and every imported helper that affects unpacking, the storage
ledger, hashing, or result writing; Python, NumPy, driver, UUID, and power limit
are recorded in the result.
