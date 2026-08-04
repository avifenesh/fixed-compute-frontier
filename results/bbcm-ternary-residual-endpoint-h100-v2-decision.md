# BBCM ternary-residual endpoint H100 v2 decision

Decision: pass as an exact-format checkpoint reconstruction endpoint.

- BF16 baseline NLL: `2.8295718897134066`.
- INT8-plus-ternary decoded NLL: `2.8298953603953123`.
- Relative NLL degradation: `0.0114317888%`.
- Paired absolute-NLL 95% interval: `[0.0002429777, 0.0004039636]`.
- The ternary residual recovers `91.67%` of the INT8 base's absolute NLL tax.
- Combined weight error is lower for every one of 90 matrices; maximum combined relative L2 is `0.0035411`.
- Base and delta scales are exactly BF16-representable; deployed codes use ternary `{-1,0,1}`.
- Result SHA-256: `df736cbbdc45387120b7bd328e27b30c9198ffd17246b1e7d43717871eaaeef7`.

The four residual routes are still identical. This establishes a low-damage initialization, not specialization or a fast serving implementation. The endpoint also retains width 1,536; exact small-model byte equality requires a width-funded learning control.
