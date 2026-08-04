# Reflex-SwiGLU fused H100 v2 decision

Decision: **pass the serving gate; authorize the matched language-learning screen.**

This is not a capability result. It establishes that the proposed FFN algebra has a credible fixed-serving-cost implementation on one H100.

## Validated result

- Exact SwiGLU endpoint: maximum output error `0.0` at `alpha=0`.
- Shared-weight endpoint gradients: finite-difference maximum difference `0.0`.
- Live new parameter: `d output / d alpha = 0.0621888431` in the frozen witness.
- Nonzero-alpha fused reference: maximum row-relative error `0.00368152`, with 148 negative-clipped, 472 unclipped, and 148 positive-clipped coordinates.
- Same-width median latency ratios for batches 1, 8, 32, 128: `0.99616`, `1.00088`, `1.00107`, `1.00116`.
- Equal-parameter aligned median ratios: `1.01152`, `1.00591`, `1.01090`, `1.00891`.
- Same-width parameter overhead: `1/(3D) = 0.0000813802`; dense matrix MAC count is unchanged.

## What passed

Reflex-SwiGLU adds a value-conditioned second gating step without another matrix multiplication, reduction, collective, or hidden-state materialization:

```
z0 = SiLU(g) * u
r = clip(z0, -1, 1)
alpha = 2 * tanh(beta / 2)
z1 = SiLU(g + alpha * r) * u
```

The fused pointwise executor stayed below the frozen `1.02x` key-cell latency limit at both same width and aligned equal parameter count.

## What remains unproved

The exact function class is larger, but a larger class may optimize worse or converge to the same language loss. The next gate must compare terminal validation loss against equal-storage controls, including a per-channel gate bias and stopped-feedback-gradient variants. Early-only improvement is insufficient.

Evidence:

- `reflex-swiglu-stage0-v2.json`, SHA-256 `d60e7360ce2b19aaf37bab1d34af0a10a6d1b13eb4d325b83dac601c99b374f6`
- `reflex-swiglu-fused-h100-v2.json`, SHA-256 `2c2fb0fa5c622b279a87a5b7c0a0d7f3c97885c1de6065645762f4f95e9ee6fa`
