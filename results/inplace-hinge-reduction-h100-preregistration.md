# In-place hinge reduction — H100 development gate

Status: **frozen before observing candidate kernel timings**  
Date: 2026-07-28  
Hardware: retained H100 SXM 80 GB

## Claim under test

Let an ordinary W8 projection reduction be partitioned once:

\[
A=W_0x_0,\qquad B=W_1x_1,\qquad s=A+B.
\]

Instead of retaining `A`, mutate the one live accumulator at the midpoint:

\[
A \leftarrow A + \alpha |A|,
\qquad y_\alpha=A+B+\alpha|A|.
\]

Then forget the old `A` and continue the native K loop.  At `alpha=0`, this is
the exact ordinary projection.  At `alpha=1`, the first partition becomes
`2 ReLU(A)`, so one learned row supplies a conditional hinge plus the ordinary
second-partition contribution.  No checkpoint state survives the mutation.

The new map is not a reweighted dense projection.  For independent `A` and
`B`, it has different input gradients on the two sides of `A=0`.  A final-only
univariate control `s + alpha|s|` has gradients confined to the single dense
row direction, while the midpoint hinge depends separately on the `A` and `B`
directions.  A later algebra gate must formalize this distinction before
training.

## Frozen timed null

The candidate loads one runtime INT32 scalar `alpha=0`, executes
`accumulator += alpha * abs(accumulator)` once after K/2, overwrites the same
SSA accumulator, and resumes the second K half.  The output must be bit-exact
to dense.  This measures the transient mutation cost while preventing
compile-time removal.

Shape is `K=N=4096`, rows `{1,8,64,256,1024}`.  Both arms independently choose
their best configuration from:

- `BLOCK_M=64`, `BLOCK_K=32`;
- `BLOCK_N in {64,128}`;
- `num_warps in {4,8}`;
- `num_stages in {3,4}`.

## Frozen decision

Advance only if all hold:

1. every valid runtime-zero output is bit-exact;
2. every selected kernel exposes SM90 Tensor Core PTX;
3. best hinge/dense median ratio is at most `1.01` at rows 64 and 256;
4. it is at most `1.02` at every row, including 1,024;
5. selected candidates have no spills;
6. selected register delta versus the same-configuration dense kernel is at
   most four.

This is tighter than exported-state gates because the claim is zero surviving
state.  Thresholds, midpoint, hinge formula, and cells will not be tuned after
timing.

A pass authorizes only: (a) an algebra/rank proof against `s+alpha|s|`; and (b)
a complete fused W8 SwiGLU hardware gate.  Language training remains blocked.

## Mandatory later controls

- ordinary W8 SwiGLU;
- in-place hinge null;
- final-output hinge `s+alpha|s|`;
- matched two-half grouped FFN;
- learned-activation control with the same scalar count;
- exact parameter-byte funding for any served `alpha` values.

