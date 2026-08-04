# Spectral successor Stage-0 preregistration

Status: frozen before executing the Stage-0 result  
Date: 2026-07-30

## Frozen artifacts

- Source: `94260afe2bf2107ea23d67ffcfbba7fe3d9813c3b172484c0442f5743c6a6465`
- Tests: `26c415dbc2e5b81f59977bdff56fcf9f0736d2797792e2dcb897be6d4c7f14fb`
- Hypothesis:
  `8e4fa680d5222bc9dfe358ce2abcc461d2dc506cba8acf449c59f2182cce1294`
- Seed 260730; 4,096 permutation pairs.
- Token-code width 4; eight complex frequencies; target width 64.
- Spectral radius 0.97 with uniformly spaced half-bin-offset frequencies.

## Required gates

All must pass without changing the frozen construction:

1. Reverse recurrence and direct sum agree within `1e-12` in float64.
2. Reverse recurrence and direct sum agree within `1e-4` in float32.
3. Fixed-code bag-of-words collides for all 4,096 same-multiset permutation
   pairs.
4. The spectral target has zero exact float32 collisions on those pairs.
5. Its first-percentile relative separation margin is at least 1%.
6. Eight distinct spectral evaluations reconstruct an eight-step, width-four
   future through the Vandermonde inverse within `1e-10`, with condition number
   at most 2.
7. The 64-value BF16 target has fewer bits than an arbitrary 512-token,
   49,152-symbol suffix, explicitly preserving the finite-capacity wall.
8. Exported parameter and inference-operation deltas are exactly zero.

Passing proves only the target algebra and accounting.  It authorizes the
predeclared latent-plan learning falsification test.  It does not establish
novelty, natural-language gain, compute efficiency, or a breakthrough.
