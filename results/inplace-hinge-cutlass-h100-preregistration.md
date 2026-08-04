# In-place hinge — native CUTLASS H100 gate

Status: **frozen before compiling or timing the modified CUTLASS kernel**  
Date: 2026-07-28  
Hardware: retained H100 SXM 80 GB  
CUTLASS: v3.5.1, SM90a

## Why this gate exists

The Triton W8 gate proved the in-place algebra can execute with zero register
delta and near-zero relative cost against its own dense kernel.  That dense
kernel is not the production ceiling.  This gate injects the active
`A <- A + |A|` mutation into the same CUTLASS TMA/WGMMA mainloop used by the
fastest retained native S8 baseline.

This is a separate fatal gate.  The Triton pass cannot substitute for it.

## Frozen implementation

- Baseline executable: the already-built unmodified CUTLASS profiler containing
  only the exact native kernel
  `cutlass3x_sm90_tensorop_i64x128x32gemm_s8_s8_s32_s8_s8_128x128x128_2x1x1_0_tnn_align16_warpspecialized_cooperative_epi_tma`.
- Candidate: a separate build from the same v3.5.1 tree and kernel filter with
  `CUTLASS_ENABLE_MIDPOINT_HINGE=1`.
- In the SM90 shared/shared warp-specialized collective, record the original K
  tile count.  Immediately before consuming the first tile of the second half:
  wait for all outstanding WGMMA groups, fence accumulator access, replace each
  INT32 accumulator element by `value > 0 ? 2*value : 0`, fence it again, and
  resume the unchanged pipeline.
- No output workspace, second accumulator, weight, scale, or extra global read.
- The active path is `alpha=1`; it is not a compiler-null timing proxy.

The patch is limited by `if constexpr` to INT32 accumulators.  The baseline
binary is not rebuilt after modifying headers.

## Frozen cells and measurement

Use `K=N=4096`, rows `{1,8,64,256,1024,4096}`.  For each binary/kernel/cell:

- 20 warmups and 200 profiler iterations;
- verification disabled because CUTLASS's ordinary GEMM reference does not
  implement the midpoint algebra;
- persist CSV output and command lines;
- extract kernel resource usage and instruction evidence from the cubin.

The independent active-path correctness witness is already frozen in
`inplace-hinge-reduction-alpha-one-evidence.json`: exact error zero, 64
registers, no spills, two WGMMA instructions for K=128, and 32 `abs.s32`
operations per thread fragment.

## Frozen decision

Advance to a full-FFN/learning gate only if all hold:

1. the candidate kernel is the exact same native S8 kernel family, with the
   same WGMMA instruction count and no extra global workspace;
2. compiled candidate register use is no greater than baseline and neither
   spills;
3. candidate/baseline runtime is at most `1.01` for rows 256 and 1,024;
4. it is at most `1.02` for every row;
5. disassembly contains the midpoint integer hinge instructions between two
   WGMMA regions, proving the active mutation survived compilation.

No threshold or tile shape is tuned after timing.  Failure closes the current
native executor even if the weaker Triton comparison passed.

