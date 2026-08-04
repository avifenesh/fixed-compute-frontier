# Static heterogeneous SwiGLU: long-horizon decision

## Decision

**Closed for the fixed 50/50 RMS-matched recipe.**

The 10M-token development gain was real at that checkpoint, but it did not
survive longer optimization.  The frozen confirmation was stopped after two
of five seeds because both completed candidate arms lost at 50M tokens.  The
preregistered requirement was at least four wins in five seeds; after two
losses that gate was mathematically impossible, so running the remaining
candidate arms could not change the decision.

## Evidence

All integrity checks passed.  Raw and canonical-null controls remained closely
matched, excluding the fixed chart as the source of the result.

| Seed | 10M best control | 10M candidate | Relative | 50M best control | 50M candidate | Relative |
|---:|---:|---:|---:|---:|---:|---:|
| 2741 | 6.3057530 | 6.3041366 | +0.02563% | 5.5268270 | 5.5306529 | -0.06922% |
| 2753 | 6.2978849 | 6.2967944 | +0.01732% | 5.5244162 | 5.5291721 | -0.08609% |

The result is consistent across these untouched initialization seeds: a small
early optimization advantage reverses into a small terminal disadvantage.
This also explains why the larger 10M development effect was not an adequate
capability claim.

## What remains true

- The alternative gate has a different feature order:
  `g*tanh(g)` starts quadratically as a gate, so the complete GLU channel
  starts cubically, whereas ordinary SwiGLU begins quadratically.
- `g*tanh(g) = 0.5*(SiLU(2g) + SiLU(-2g))`; one such channel compresses a tied
  pair of ordinary SwiGLU channels.
- A static heterogeneous partition uses the same three dense matrix shapes,
  learned-scalar count, optimizer-state size, and dense MAC count.
- These algebraic facts did not produce a better trained model under the
  tested recipe.

## Reopen condition

Do not tune the ratio, RMS multiplier, STE, seed, or schedule to rescue this
branch.  Reopen only if a different algebraic construction supplies a reason
that heterogeneous feature order should improve the converged function—not
merely early optimization—and freeze a new fatal long-horizon control before
training.

Evidence artifact:
`results/heterogeneous-swiglu-long-horizon.json` (SHA-256
`6beab37c3e54e8f172b89acf65e7e76d9bbcd658d512982318bc832f281d789d`).
