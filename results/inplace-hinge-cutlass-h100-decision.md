# In-place hinge — native CUTLASS S8 decision

Decision: **close the exact-width S8 executor; retain the algebra and search
for an already-synchronized precision path.**

The active `alpha=1` hinge was injected into the exact native SM90a
TMA/WGMMA kernel.  It preserved production resource shape—168 registers,
zero local bytes, and the same reported shared bytes—but added one mandatory
warpgroup drain.  That bubble is visible in native timing.

| rows | baseline (ms) | hinge (ms) | ratio |
|---:|---:|---:|---:|
| 64 | 0.0148754 | 0.0149482 | 1.004894 |
| 256 | 0.0142930 | 0.0145506 | **1.018023** |
| 1,024 | 0.0286662 | 0.0289482 | 1.009837 |
| 4,096 | 0.0926938 | 0.0938298 | 1.012255 |

The exact kernel does not admit rows 1 or 8, so it also cannot establish a
decode path.  At rows 256 the 1.80% overhead fails the preregistered 1% key
cell bound.  The exact-width S8 implementation therefore does not advance to
learning as a zero-cost executor.

This is not a mathematical rejection.  The cost comes from creating a new
synchronization point, not from the hinge arithmetic or storage.  CUTLASS's
accurate FP8 mainloop already drains and promotes a temporary WGMMA
accumulator into a main FP32 accumulator at fixed intervals.  The next branch
may apply the same in-place hinge at an existing promotion boundary; it must
not add a wait.

Evidence:

- frozen protocol: [`inplace-hinge-cutlass-h100-preregistration.md`](inplace-hinge-cutlass-h100-preregistration.md)
- result: [`cutlass-inplace-hinge-gate/summary.json`](cutlass-inplace-hinge-gate/summary.json)
- source patch: [`../experiments/cutlass_midpoint_hinge_v351.patch`](../experiments/cutlass_midpoint_hinge_v351.patch)
- runner: [`../experiments/run_cutlass_inplace_hinge_gate.sh`](../experiments/run_cutlass_inplace_hinge_gate.sh)

