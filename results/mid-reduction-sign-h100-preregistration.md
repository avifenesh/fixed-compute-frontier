# Mid-reduction sign checkpoint — H100 development gate

Status: **frozen before observing candidate kernel timings**  
Date: 2026-07-28  
Hardware: retained H100 SXM 80 GB

## Claim under test

An ordinary integer projection destroys every intermediate partial sum when it
finishes `y=Wx`.  We can retain one additional fact per output lane without a
second accumulator bank:

1. accumulate the first half of K into the ordinary INT32 accumulator `A`;
2. checkpoint `c = +1` when `A >= 0`, otherwise `c = -1`;
3. continue the second half in the **same** accumulator, producing `s=A+B`;
4. expose `(s,c)` to the epilogue.

This is strictly more information than `s`: `(A,B)=(1,-1)` and `(-1,1)` both
produce `s=0`, but have opposite checkpoints.  The checkpoint is a predicate,
not a second full-precision projection.  W8 weights, operand reads, WGMMA
instructions, output shape, and the long K reduction remain unchanged.

The timed null computes `s + alpha*c` with an INT32 `alpha` loaded at runtime
and set to zero.  Therefore the result must be bit-exact to baseline while the
compiler cannot delete the live checkpoint.  This epilogue is only a hardware
proxy; it is not a language-capability claim.

## Frozen kernel matrix

Projection shape is `K=N=4096`, with rows `{1,8,64,256,1024}`.  Baseline and
candidate use the same Triton configuration grid:

- `BLOCK_M=64`, `BLOCK_K=32`;
- `BLOCK_N in {64,128}`;
- `num_warps in {4,8}`;
- `num_stages in {3,4}`.

Every configuration is correctness-checked before timing.  Each arm may select
its own fastest valid configuration per row count.  Compiler metadata records
registers, spills, shared memory, and PTX Tensor Core evidence.

## Frozen decision

The one-bit placement advances only if all hold:

1. every valid runtime-zero configuration is bit-exact to baseline;
2. the best kernels expose SM90 Tensor Core PTX rather than scalar dot loops;
3. the best checkpoint/baseline median ratio is at most `1.02` at rows 64 and
   256;
4. the ratio is at most `1.03` at every row, including 1,024;
5. no selected checkpoint kernel spills;
6. the candidate's selected register count is no more than eight above its
   same-configuration baseline.

The tighter all-cell bound distinguishes a compact checkpoint from the closed
two-bank executor, which cost 7.33% at rows 1,024 and 32 extra registers.
Thresholds, checkpoint location, and row cells will not be tuned after timing.

## Authorization after a pass

A pass authorizes only an algebra/control screen for a non-absorbable use of
`(s,c)`, followed by a fused W8 SwiGLU gate.  Mandatory controls are:

- ordinary W8 SwiGLU;
- checkpoint-null with all executor work live;
- an output-derived sign control using `sign(s)` rather than `sign(A)`;
- a matched learned bias/activation control;
- a candidate with a zero endpoint that preserves the full ordinary FFN.

No language-model training is authorized by this projection-only gate.

