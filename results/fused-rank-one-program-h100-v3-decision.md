# Compiled rank-one program H100 gate v3 — fused decision

Decision: **the original frozen grid fails; retain an r<=8 hardware primitive.**

This was a materially different implementation from the rejected PyTorch
executor: additive, compiled, and direct sequential execution each used one
custom CUDA launch, one block per token program, the same already-selected
expert vectors, and FP32 accumulation.  It repaired the old launch and
materialization overhead, but it exposed a real program-length boundary.

## Result

At `D=4096`, both FP16 and BF16, and batches `1,8,32,128`:

- `L=4`: compiled/additive median latency was `1.001x-1.008x`.
- `L=8`: compiled/additive was `1.004x-1.023x`; compiled was about
  `0.677x-0.693x` the latency of fused direct sequential execution.
- `L=16`: compiled/additive jumped to `1.674x-1.708x`, failing the frozen
  `<=1.15x` key-cell gate and sometimes losing to direct execution.
- `L=32`: compiled/additive returned near `1.024x-1.038x`, but absolute
  latency and compiled/direct ratios were poor; this does not rescue the
  failed monotone length grid.
- Maximum row-relative error was below `0.00022` in FP16 and `0.00174` in
  BF16, within the frozen `0.005` bound.

The non-monotone `L=16` cliff is consistent with a generated-code/register or
instruction-layout regime change.  It is not legitimate to relabel the old
full-grid result as a pass.  The old routed program remains closed because the
router, bank gather, and language-quality questions are also unresolved.

## Retained consequence

Groups capped at eight are a measured, conditional executor primitive: a
short triangular scalar recurrence can add ordered nonlinear dependence for
roughly 0-2.3% over a matched additive rank-one executor.  This observation
may be used to design a new mechanism only if that mechanism freezes `r<=8`
before its own quality result and does not inherit the old uncharged router.

The block-triangular microdepth FFN follows that rule: it removes routing,
places every coefficient in a fixed group of eight, and tests whether the
cheap scalar dependency is more useful than spending the same parameters on
an ordinary wider activation bank.

Result SHA-256:
`f19438c3a278331b1b93d4d0f698dbb50b06549a8d239d28f40c3cbdbd33c4a2`.
