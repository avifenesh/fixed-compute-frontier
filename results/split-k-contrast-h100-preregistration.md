# Split-K contrast projection — H100 development gate

Status: **frozen before observing kernel timings**  
Date: 2026-07-28  
Hardware: retained H100 SXM 80 GB

## Claim under test

For a fixed sign partition `P=diag(+1,-1,+1,-1,...)`, split the ordinary W8
projection reduction into two native accumulator banks:

\[
A=W_{+}x_{+},\qquad B=W_{-}x_{-},\qquad
s=A+B=Wx,\qquad d=A-B=WPx.
\]

The sum is the exact ordinary projection.  The contrast is a second tied
linear statistic computed from the same W8 weights, scales, activation values,
weight/input reads, and WGMMA instructions.  The changed physical resources
are a second accumulator bank and an epilogue add/sub/use.

The algebra gate is already frozen in
`results/split-k-projection-algebra.json`.  It proves exact INT8 reconstruction,
rank `2 -> 4` for a width-four witness, quadratic Hessian rank `2 -> 4` for the
minimum baseline-containing SwiGLU correction, and a live zero endpoint.

## Development kernels

One Triton kernel accumulates every K tile into one INT32 bank and stores the
ordinary INT32 result.  The split kernel accumulates the first and second K
partitions into separate INT32 banks, then computes

\[
o=s+\alpha d

\]

with `alpha` loaded at runtime.  The timed null uses `alpha=0`, keeping the
contrast live to the compiler while requiring bit-exact equality to baseline.
This linear epilogue is only a register/occupancy proxy; it is not the proposed
language activation and creates no capability claim.

Frozen projection shape is `K=N=4096`, with rows `{1,8,64,256,1024}`.  Both
kernels are compared under the same Triton tile for every configuration:

- `BLOCK_M=64`, `BLOCK_K=32`;
- `BLOCK_N in {64,128}`;
- `num_warps in {4,8}`;
- `num_stages in {3,4}`.

Each cell reports every valid configuration and separately compares the best
baseline with the best split kernel.  Correctness is checked before timing.
The result records compiler/SASS evidence where the installed Triton runtime
exposes it; missing inspection metadata cannot be called a pass.

## Frozen decision

This development gate passes only if all hold:

1. null output is bit-exact in every valid cell;
2. both kernels compile to SM90 Tensor Core instructions rather than scalar
   dot loops;
3. the best split/baseline median latency ratio is at most `1.02` for rows
   `64` and `256`;
4. the ratio is at most `1.05` for every row;
5. neither best split kernel spills local memory;
6. the split configuration's register count does not make the pass depend on
   an unrepresentative single-CTA occupancy accident.

A pass authorizes only a fused W8 gate/up plus contrast-correction FFN kernel.
Language learning remains blocked until that complete FFN matches the fastest
ordinary fused W8 SwiGLU path across decode and prefill.  A fail closes this
placement; thresholds, partitions, and row cells are not tuned afterward.

## Mandatory later controls

- ordinary W8 SwiGLU;
- split-null, which reconstructs ordinary SwiGLU while paying the two-bank
  executor cost;
- a foldable linear use of `d`, proving that merely rotating the fixed
  partition can be absorbed into ordinary weights;
- the minimum non-absorbable correction
  `SwiGLU(s_g,s_u) + lambda*d_g*d_u`;
- a grouped-half FFN that receives the same two partial projections but has no
  baseline sum path.

