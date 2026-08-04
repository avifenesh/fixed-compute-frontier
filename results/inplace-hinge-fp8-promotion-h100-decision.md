# FP8 promotion hinge — H100 decision

Status: **closed as an executor; algebra retained**  
Date: 2026-07-28  
Hardware: H100 SXM 80 GB  
CUTLASS: v3.5.1, SM90a

## Result

The active midpoint hinge reused CUTLASS accurate-FP8's existing accumulator
promotion wait.  Baseline and candidate resource reports were identical, and
both binaries contained five `WARPGROUP.DEPBAR` instructions.  The candidate
therefore fixed the synchronization failure seen in the S8 experiment.

It still failed the frozen timing gate:

| rows | baseline ms | hinge ms | ratio |
|---:|---:|---:|---:|
| 64 | 0.0165280 | 0.0172258 | 1.04222 |
| 256 | 0.0164426 | 0.0171026 | 1.04014 |
| 1,024 | 0.0304179 | 0.0313552 | 1.03081 |
| 4,096 | 0.1091140 | 0.1124140 | 1.03024 |

Disassembly explains the residual cost.  The candidate added 128 `FSEL` and
128 `FADD` instructions while leaving the dependency-barrier count unchanged.
Applying a scalar nonlinearity to every live output accumulator is itself a
full output-tile pass; an existing wait does not make that pass free.

## Decision

Do not advance this kernel to the learning screen.  Close both tested
per-output placements:

- a new midpoint wait is too expensive;
- an existing midpoint wait removes that expense, but the tile-wide mutation
  still costs 3.0%-4.2% in the isolated projection.

Retain the mathematical function `A+B+alpha|A|`, but move its next successor
to the shared input side of a projection group.  An `O(d)` nonlinear chart
computed once for Q/K/V or gate/up can be amortized across their `O(d*m)`
linear projections; repeating a nonlinear scalar operation for all `m`
outputs cannot.

Evidence:

- `results/inplace-hinge-fp8-promotion-h100-preregistration.md`
- `results/cutlass-fp8-promotion-hinge-gate-summary.json`
- `experiments/cutlass_fp8_promotion_hinge_v351.patch`
- `experiments/run_cutlass_fp8_promotion_hinge_gate.sh`

