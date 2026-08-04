# Gauge-curvature learned-routing screen — decision

Status: **cyclic writer rejected; no confirmation run**  
Date: 2026-07-26  
Hardware: one Vast H100 NVL 96 GB, PyTorch `2.7.1+cu128`  
Frozen evidence: [`gauge-curvature-learned-routing-screen.json`](gauge-curvature-learned-routing-screen.json)

## Decision

Reject the gauge-funded cyclic writer as an architecture hypothesis on this
mechanism. It lost the primary held-out NLL comparison to the fixed strongest
control, paper-style GLU Attention, in **all five worlds**. The preregistration
required at least four wins.

Median relative excess-NLL improvement above the finite-test Bayes marginal was
`-27.79%`: cyclic increased, rather than closed, the remaining NLL gap. Its
target-bag attention mass was also lower than GLU in every world by 0.10–0.90
percentage point, so the loss is not hidden by better routing. Do not run the
two confirmation seeds, frozen-router diagnostic, unpaired task, mean task, or
an LM gate for cyclic.

## Five-world result

All learned variants have 203,328 trainable parameters except BDA (201,792)
and square (203,232). Cyclic and dense cache 96 K+V scalars per token/layer;
GLU caches 84.

| variant | median test NLL | median accuracy | median best validation NLL | median target-bag mass | median train seconds | median incremental peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.87582 | 84.81% | 0.31923 | 97.80% | 22.17 | 166.9 |
| BDA | 0.97319 | 83.64% | 0.33351 | 97.37% | 24.10 | 165.4 |
| **cyclic** | **0.97594** | **86.18%** | **0.31616** | **96.69%** | **27.62** | **183.8** |
| square | 1.01703 | 83.37% | 0.34111 | 97.40% | 26.92 | 176.4 |
| source MLP | 0.95781 | 83.98% | 0.33940 | 97.51% | 26.38 | 176.4 |
| matched G2 | 1.01044 | 83.34% | 0.33589 | 97.63% | 26.34 | 175.3 |
| full G2 | 0.90654 | 85.19% | 0.32446 | 97.95% | 23.35 | 170.9 |
| **GLU, value width 12** | **0.81652** | **86.60%** | **0.29575** | **97.35%** | **23.30** | **170.9** |
| G1 | 0.93411 | 85.10% | 0.32480 | 97.92% | 23.52 | 172.5 |

The constrained Bayes oracle was 92.76–93.33% accuracy and 0.1665–0.1809
NLL. Address-only and uniform-all-bags controls were 50%; breaking query/source
addresses drove the cyclic model below chance as expected. Declared and actual
parameter counts matched, all nonlinear initializations had nonzero gradients,
and dense/BDA/cyclic/source-MLP/matched-G2 initial functions agreed to within
`6e-8` on H100.

## Why the answer is negative

The cyclic writer did add the intended pre-cache quadratic family, but that
family was not a better fixed-budget inductive bias than narrower GLU values or
the standard alternatives. Cyclic's accuracy was not disastrous; its failure
was generalization and calibration. Training NLL continued toward zero while
validation NLL bottomed near steps 150–300 and then rose. The frozen protocol
used the final 1,500-step checkpoint, so every variant was overconfident.

This schedule weakness does not reverse the comparison:

- GLU had lower final test NLL in all five worlds;
- GLU also had lower **best validation NLL in all five worlds**;
- cyclic did not gain routing quality;
- the naive PyTorch cyclic implementation was slower and used more temporary
  memory, although this screen is not a fused served-kernel benchmark.

We therefore do not rescue cyclic by selecting an earlier checkpoint after
seeing the curves. A later research program may reuse the addressed-bag
benchmark with a preregistered common early checkpoint, but it must be a new
hypothesis, not a rerun intended to save this writer.

## What survives

1. **Algebra only:** the cyclic mask still spans every scalar quadratic
   monomial for `r>=3`, and its raw value multiplication/addition/cache ledger
   matches dense. Its 48 learned pivot scales are V/O-gauge directions, so the
   actual added quotient family is `r^2-r`, not `r^2`.
2. **Testing method:** fresh per-scene orthogonal addresses, exactly two
   positive/two negative bags, joint 2-of-4 Bayes marginals, and broken-address
   controls form a useful learned-routing mechanism gate.
3. **Control result:** GLU was the only observed Pareto-shaped writer here:
   lower median NLL, slightly higher accuracy than cyclic, equal raw parameter
   count, and 12.5% fewer combined K+V cache scalars. This corroborates prior
   art; it is not a new architecture claim and still needs an LM and served
   kernel gate under a non-overtrained protocol.

## Audit trail

- Frozen protocol: [`gauge-curvature-learned-routing-preregistration.md`](gauge-curvature-learned-routing-preregistration.md)
- H100 CUDA self-test: [`gauge-curvature-learned-routing-h100-self-test.json`](gauge-curvature-learned-routing-h100-self-test.json)
- Excluded 30-step smoke: [`gauge-curvature-learned-routing-h100-smoke.json`](gauge-curvature-learned-routing-h100-smoke.json)
- Complete screen: [`gauge-curvature-learned-routing-screen.json`](gauge-curvature-learned-routing-screen.json)
- Runner: [`../experiments/gauge_curvature_learned_routing_torch.py`](../experiments/gauge_curvature_learned_routing_torch.py)

An independent adversarial review found and forced correction of the
constrained Bayes oracle, routing validity, redundant source-control scale,
raw-versus-quotient accounting, and G2/G1 explicit multiplication ledger before
the frozen screen ran.
