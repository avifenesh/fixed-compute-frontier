# Raw-prose equality plane T11 — pilot decision

Status: **single-seed Pareto pilot passed; production claim not admitted**  
Date: 2026-07-31

## Result

T11 repaired the measured T10 capacity tradeoff without changing the serving
graph.  The frozen footprint fell from 7,824,855 entries (21.39%) to 743,734
entries (2.03%), while the exhaustive BF16 representation gate remained exact.

After matched training, held-out direct and equality accuracy remained 100% at
steps 250, 500, and 1,000.  Relative natural NLL versus `muon_labels_1x` was:

| Step | Candidate NLL | Matched control NLL | Relative delta |
|---:|---:|---:|---:|
| 250 | 6.631265 | 6.644705 | -0.2023% |
| 500 | 6.362011 | 6.358583 | +0.0539% |
| 1,000 | 6.146454 | 6.128630 | +0.2908% |

Every preregistered gate passed.  The best twice-trained gradient control
reached 8.41% held-out direct accuracy and 77.51% equality accuracy.

## Claim boundary

This is the first positive Pareto pilot in this lane: a controlled structured
capability was added at identical deployed parameter count, tensor shapes,
attention/SwiGLU graph, KV state, precision, and per-token serving FLOPs,
without exceeding the frozen natural-quality tolerance.

It is not evidence for a smarter production LLM.  It is one model seed, a
synthetic corpus with exact repeated structure, a four-token lexical query
protocol, and a 36.6M-parameter research model.  The gradient controls are
stronger than T10's candidate budget but are not frontier-scale pretraining.

The BF16 minimum margins also fell from above 10 before training to 0.0708
(direct) and 0.0615 (equality) at step 1,000.  Their signs remained exact, but
unseen-seed and longer-scale robustness must be tested before treating the
compiled plane as stable.

## Decision

Retain T11 and advance it unchanged to three unseen model seeds.  Do not change
the mask, output scale, optimizer, schedule, threshold, data, or controls.
Passing replication admits a scaled real-prose/unrestricted-QA test; it still
does not complete the project goal.
