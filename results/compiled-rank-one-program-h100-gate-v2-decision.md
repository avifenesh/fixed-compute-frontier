# Compiled rank-one program H100 gate v2 decision

Status: **closed as an unfused executor; retain the exact algebra only**.

## Provenance

- Device: NVIDIA H100 80GB HBM3, 132 SMs.
- Frozen grid: `D=4096`, `L in {4,8,16,32}`, `B in {1,8,32,128}`, FP16 and BF16.
- Timing: 40 warmups and 200 randomized, interleaved samples per cell.
- Source SHA-256: `0c13bdf32739a67d5995b882e5293f8c65d087d024a073690054118a417492bf`.
- Result SHA-256: `3a33b37507c7980fe5bc1a42b97d478b69128bced5e131d9c3b985f3af0109ad`.

## Valid conclusion

The compiled PyTorch executor did not produce the promised serving advantage:

- It was slower than direct sequential execution in every measured cell.
- Median compiled/direct latency ratios ranged from about `1.01x` to `1.57x`.
- At the key `L=16, B=8` cells, compiled/additive was `1.270x` in FP16 and `1.217x` in BF16, exceeding the preregistered `1.15x` ceiling.
- Overhead increased with program length, reaching roughly `1.49x` to `1.93x` versus additive at `L=32`.

This rejects the current unfused execution path. Another run is justified only for a materially different fused kernel, not threshold tuning or more samples.

## Invalid numerical conclusion

The two update-relative error gates are not interpretable. The harness formed the update as `low_precision(x + update) - x`; small updates were then lost to residual-add rounding before subtraction. That measures cancellation relative to the residual scale, not error in the compiled recurrence.

Evidence for the diagnosis:

- FP16 compiled full-output max-row error stayed near `0.021%`, while the reported update-relative error ranged from about `0.55%` to `13.62%`.
- BF16 compiled full-output max-row error stayed near `0.15%` to `0.17%`, while the reported update-relative error ranged from about `4.38%` to `69.94%`.
- Direct execution often showed even larger update-relative error, despite being the reference-shaped computation.

Therefore the failed numerical gates do **not** reject the algebra. Any future fused-kernel gate must compare pre-residual updates at a common dtype and separately report deployable residual-add error.

## Retained result

The exact reduction from serial D-wide rank-one updates to two wide passes plus a scalar triangular recurrence remains mathematically valid. What failed is the measured PyTorch/H100 realization and its latency claim.
