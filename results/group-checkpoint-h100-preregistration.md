# Group mid-reduction checkpoint — H100 development gate

Status: **frozen before observing candidate kernel timings**  
Date: 2026-07-28  
Hardware: retained H100 SXM 80 GB

## New claim under test

The rejected per-output sign kept accumulator-shaped state even though its
values were Boolean.  This successor reduces state cardinality by a fixed
factor of 64.

Partition output features into model-semantic consecutive groups `G` of 64.
After accumulating the first half of K into the ordinary INT32 accumulator
`A`, retain one group decision per token:

\[
c_{t,G}=\operatorname{sign}\left(\sum_{j\in G} A_{t,j}\right).
\]

Then reuse the same accumulator for the second K half to produce the ordinary
projection `s=A+B`.  The output exposes `(s,c)`, with each group decision
broadcast only to its 64 member features.

This is strictly more information than `s`: two partial-reduction states can
have the same final projection while their first-half group sums have opposite
signs.  It also introduces group-conditioned feature behavior before the
ordinary FFN down projection.  It is not a per-output second projection.

The timed null computes `s + alpha*c` with runtime INT32 `alpha=0`, requiring
bit-exact dense output while keeping the checkpoint live.  No global
intermediate is allowed.

## Frozen kernel matrix

Projection shape is `K=N=4096`, rows `{1,8,64,256,1024}`, and semantic group
width is exactly 64.  Both arms use:

- `BLOCK_M=64`, `BLOCK_K=32`;
- `BLOCK_N in {64,128}`; a 128-wide tile holds two independent semantic groups;
- `num_warps in {4,8}`;
- `num_stages in {3,4}`.

Each arm may select its fastest valid configuration per row.  Correctness,
registers, spills, shared memory, and Tensor Core PTX evidence are recorded.

## Frozen decision

Advance only if all hold:

1. every valid runtime-zero output is bit-exact to baseline;
2. every selected kernel exposes SM90 Tensor Core PTX;
3. best candidate/baseline median is at most `1.02` for rows 64 and 256;
4. it is at most `1.03` for every row, including 1,024;
5. selected candidates have no spills;
6. selected candidate register delta versus its same-configuration baseline is
   at most eight.

Group width, midpoint, thresholds, and cells will not be tuned after timing.
A pass authorizes a non-absorbability/control proof and then a fused-FFN gate,
not language training.

## Mandatory later controls

- ordinary W8 SwiGLU;
- group-checkpoint null;
- final-output group sign `sign(sum_G s)`, which needs no partial state;
- an equally cheap global/token sign;
- a same-byte narrower independent-gate FFN;
- exact zero endpoint and matched learned-parameter ledger.

