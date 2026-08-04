# In-place hinge at FP8 promotion — native H100 gate

Status: **frozen before compiling or timing the modified FP8 kernel**  
Date: 2026-07-28  
Hardware: retained H100 SXM 80 GB  
CUTLASS: v3.5.1, SM90a

## Claim under test

The accurate CUTLASS FP8 mainloop already keeps two accumulator fragments:

- a temporary WGMMA fragment;
- a main FP32 accumulator.

At the configured promotion interval it executes `warpgroup_wait<0>()`, adds
the temporary fragment into the main accumulator, and clears the temporary
fragment.  For the selected `TileK=128` E4M3 kernel, the default promotion
interval is four WGMMA operations—the four K32 operations in each K tile—so a
promotion already occurs at every K128 boundary.

At K/2, immediately after the existing promotion, the candidate mutates the
main accumulator in place:

\[
A \leftarrow A+|A|=2\operatorname{ReLU}(A).
\]

It adds no wait, accumulator fragment, global state, or operand read.  It then
continues the unchanged second half.  The active result is `B+2ReLU(A)`.

## Frozen kernel

Both binaries contain only:

`cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_2x1x1_0_tnn_align16_warpspecialized_cooperative_epi_tma`

- Baseline: same v3.5.1 source with no hinge macro.
- Candidate: `CUTLASS_ENABLE_FP8_PROMOTION_HINGE=1` and the frozen patch.
- Active `alpha=1`; no compiler-null timing arm.
- `K=N=4096`, rows `{64,256,1024,4096}`.
- 20 warmups and 200 iterations; verification disabled because the ordinary
  CUTLASS reference does not implement the modified reduction.

Fast-accumulation FP8 kernels are excluded: they do not provide the existing
promotion boundary being claimed.  Stream-K and split-K are excluded because
the hinge is not distributive over independently reduced K partitions.

## Frozen decision

Advance to the learning/control screen only if all hold:

1. baseline and candidate have identical register, local, and shared-memory
   resource reports;
2. candidate adds no warpgroup wait/dependency barrier relative to baseline;
3. the same WGMMA instruction family and count remain;
4. candidate/baseline runtime is at most `1.005` at rows 256 and 1,024;
5. it is at most `1.01` at every row;
6. disassembly proves the active in-place hinge occurs after an existing
   promotion and before later WGMMA work.

No promotion interval, tile, threshold, or row cell is tuned after timing.
Passing authorizes a matched learning screen against post-sum hinge and
ordinary SwiGLU.  It does not establish a decode path; that remains a separate
synchronous weight-only/GEMV gate.

