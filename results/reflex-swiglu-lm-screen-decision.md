# Reflex-SwiGLU matched LM screen decision

Decision: **reject this retrofit recipe; do not replicate.**

All provenance, endpoint, resource, and stability gates passed. Every quality gate failed.

## Terminal validation loss

| Arm | NLL at step 640 | Relative change vs baseline |
|---|---:|---:|
| baseline | 2.815088324 | — |
| reflex | 2.815111125 | -0.000810% |
| reflex_detach | 2.815082494 | +0.000207% |
| gate_bias | 2.815275539 | -0.006650% |
| gate_temperature | 2.815129142 | -0.001450% |

The best Reflex-family arm was `reflex_detach`. Its nominal paired 95% interval for candidate-minus-baseline NLL was `[-0.00007810, +0.00006644]`, spanning zero. Its relative improvement was `0.000207%`, about 966 times smaller than the preregistered `0.2%` gate.

## What the failure says

- The new parameters learned: Reflex effective alpha reached mean absolute `0.004762` and maximum `0.05764` without saturation.
- The recurrent Jacobian was not the main blocker: detaching the feedback gradient changed terminal NLL by only `0.00002863` versus ordinary Reflex.
- A generic per-channel parameter was not enough: both gate-bias and gate-temperature controls were worse than baseline.
- More exact scalar function-class expressivity did not translate into useful language capability. The useful frontier likely requires new conditional inter-feature or inter-token maps, not additional pointwise curvature.

## Cost evidence retained

The target-scale fused serving implementation remains valid: same-width latency was `0.996x` to `1.001x` baseline and added parameter storage was `0.0081%`. This executor result is retained, but there is no quality gain to deploy.

## Scope

This rejects the frozen 21M-token SmolLM2 retrofit protocol. It does not prove the mathematical activation class useless under from-scratch training. Reviving it would require a new preregistered training hypothesis, not threshold or LR tuning around this miss.

Evidence: `reflex-swiglu-lm-screen.json`, SHA-256 `c465888821490e41214557c5a8f1756115f1f95f162b058caf5e530119aefb95`.
