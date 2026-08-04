# Split-K contrast projection — H100 decision

Decision: **close the two-accumulator placement before fused-FFN or language training.**

The frozen Triton W8 gate preserved the exact dense result at runtime
`alpha=0`, used Tensor Core instructions in every selected kernel, exposed
compiler metadata, and produced no spills.  It was effectively free through
256 rows, but failed the preregistered full-envelope latency gate at 1,024
rows.

| rows | dense median (ms) | split median (ms) | split / dense |
|---:|---:|---:|---:|
| 1 | 0.106464 | 0.106176 | 0.997295 |
| 8 | 0.106656 | 0.106528 | 0.998800 |
| 64 | 0.108352 | 0.108384 | 1.000295 |
| 256 | 0.125536 | 0.125504 | 0.999745 |
| 1,024 | 0.269312 | 0.289056 | **1.073313** |

The best baseline used 64 registers per thread and the best split kernel used
96.  This is consistent with the extra INT32 accumulator bank.  The 7.33%
large-prefill cost exceeds the frozen 5% all-cell limit, even before holding
gate and up projections together in a fused SwiGLU kernel.

The algebra remains valid: the same W8 products can expose both `Wx` and a
tied contrast `WPx`, and a nonlinear use is not generally absorbable into an
ordinary projection.  What failed is retaining that information at full
INT32 precision for every output lane.

Retain the narrower successor rule: a partial reduction may export only a
compact quantized checkpoint before reusing the ordinary accumulator.  It
must preserve the long Tensor Core reduction, add no global intermediate, and
pass the complete serving envelope before any learning screen.

Evidence:

- frozen protocol: [`split-k-contrast-h100-preregistration.md`](split-k-contrast-h100-preregistration.md)
- machine-readable result: [`split-k-contrast-h100-development.json`](split-k-contrast-h100-development.json)
- algebra result: [`split-k-projection-algebra.json`](split-k-projection-algebra.json)
- executable: [`../experiments/split_k_contrast_h100.py`](../experiments/split_k_contrast_h100.py)

