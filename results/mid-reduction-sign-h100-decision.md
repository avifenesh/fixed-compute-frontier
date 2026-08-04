# Per-output mid-reduction sign — H100 decision

Decision: **close the per-output checkpoint representation.**

The frozen candidate retained `sign(A)` after the first half of a W8
projection, reused the same accumulator for the second half, and reconstructed
the ordinary result exactly at runtime `alpha=0`.  Although the logical state
was one Boolean per output, the compiled Triton kernel held one full register
lane per checkpoint.  Every selected candidate used 96 registers per thread
versus 64 for its same-configuration baseline: exactly the 32-register cost of
the closed second INT32 accumulator.

| rows | dense median (ms) | checkpoint median (ms) | ratio |
|---:|---:|---:|---:|
| 1 | 0.106336 | 0.106304 | 0.999699 |
| 8 | 0.106336 | 0.106496 | 1.001505 |
| 64 | 0.108256 | 0.108256 | 1.000000 |
| 256 | 0.125536 | 0.124320 | 0.990314 |
| 1,024 | 0.269760 | 0.287104 | **1.064294** |

The result failed both frozen discriminators: register delta `+32 > +8` and
large-prefill latency `1.0643 > 1.03`.  It therefore does not authorize an
algebra, fused-FFN, or language screen.

This does not show that partial-reduction information is intrinsically
expensive.  It shows that one checkpoint per output has the same live-state
*shape* as the accumulator unless the implementation bit-packs it.  The next
allowed branch must reduce the semantic state dimension itself—one checkpoint
per fixed output group—or prove a true packed representation before timing.

Evidence:

- frozen protocol: [`mid-reduction-sign-h100-preregistration.md`](mid-reduction-sign-h100-preregistration.md)
- result: [`mid-reduction-sign-h100-development.json`](mid-reduction-sign-h100-development.json)
- executable: [`../experiments/mid_reduction_sign_h100.py`](../experiments/mid_reduction_sign_h100.py)

