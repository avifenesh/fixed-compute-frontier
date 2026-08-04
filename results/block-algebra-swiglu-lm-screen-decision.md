# Rank-three block-algebra SwiGLU — matched LM screen decision

Decision: **reject this exact fixed complex-algebra recipe; do not fuse, tune,
or extend it to quaternion/C4.**

The rank-three algebra is mathematically genuine and the run was healthy, but
it learned language worse than ordinary SwiGLU at the same parameter count,
dense-matmul budget, width, data, and optimizer schedule.

## Frozen terminal result

| Arm | Validation NLL | Relative to baseline |
|---|---:|---:|
| Ordinary SwiGLU | **5.528720897** | — |
| Split-complex, rank 2 | 5.538607392 | 0.17882% worse |
| Complex, real rank 3 | 5.537440911 | 0.15772% worse |

Complex was 0.02106% better than the equal-arithmetic split-complex control,
but the preregistered requirement was at least 0.1%.  More importantly, its
paired 95% candidate-minus-baseline interval was entirely worse:
`[0.0074711, 0.0099689]` NLL.  The interval versus split-complex was narrowly
favorable, `[-0.0022246, -0.0001084]`, so nonsplit algebra recovered a small
part of the cross-channel mixing penalty rather than creating a net gain.

The same ordering was already visible at 10M tokens: baseline 6.303944770,
split-complex 6.316574570, and complex 6.316114038.  There is no late reversal
or checkpoint-selection ambiguity.

## The negative result is valid

- Every arm had exactly 37,758,336 total and trainable parameters.
- Initial activation RMS ratios were 1.0087 for split-complex and 1.0118 for
  complex relative to baseline, inside the frozen 25% bound.
- All 49,971,200-token runs were finite.  Maximum pre-clip gradient norms were
  3.31 or lower and maximum step losses were 11.27 or lower.
- Source, preregistration, LM tests, algebra source, algebra tests, algebra
  result, data, and runtime protocol hashes validated before arm one.
- Ten focused tests and full-model BF16 forward/backward smoke passed on the
  retained H100.

Thus the failure cannot be assigned to parameter mismatch, scale mismatch,
dead gradients, numerical divergence, an absorbable real basis change, or a
broken implementation.

## What was learned

The algebra gate proved that complex block multiplication raises the leading
real bilinear tensor rank from 2 to 3 per coordinate pair while preserving the
three dense FFN matrices.  The language screen shows that **more local
multiplicative rank is not automatically more useful capacity**.  Both
cross-channel algebras damaged optimization/generalization relative to the
coordinatewise product; the nonsplit/rank-three law only reduced that damage
slightly.

This closes the frozen 12-layer, 384-wide, 1,024-FFN, `1/sqrt(2)` fixed-core,
50M-token, seed-223 recipe.  One seed does not close all nonsplit algebras, but
there is no evidence to spend serving-kernel or quaternion work on this family.
Revival requires a materially different hypothesis explaining why its
interaction law aligns with language features, not a different threshold,
seed, scaling constant, or longer run.

## Cost boundary

The eager direct-product arms ran about 11.2% slower than baseline (267–268k
versus 301k tokens/s), despite negligible arithmetic overhead on paper.  That
would normally motivate fusion, but the capability gate failed first, so no
serving optimization is authorized.

Result SHA-256:
`860a1440cda362568857a8c91f80b18ed1212b11494e354504d68d1d35ff7fd0`.
