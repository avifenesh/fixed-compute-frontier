# Raw-prose equality plane T11 — replication decision

Status: **controlled-scale Pareto result replicated; production goal remains open**  
Date: 2026-07-31

## Decision

T10 hit an implementation failure, not a demonstrated algebraic wall.  The
T11 implementation package preserves the same compiled lookup/equality
algebra and passes the fixed natural-quality gate across three previously
unseen model seeds.

This does not identify which T11 change repaired T10 by itself.  T11 changed
both the isolation mask and the special output-row scale under one frozen
preregistration.  The result proves that the controlled construction can avoid
the T10 tradeoff; it does not separately estimate the causal contribution of
each implementation change.

## Sealed replication result

Seeds 6,703, 6,997, and 7,307 each passed every pilot gate.  Across nine
candidate-versus-matched-control checkpoints:

- worst natural-NLL delta: **+0.1784%** (gate: at most +0.5%);
- best natural-NLL delta: **-0.8368%**;
- held-out direct accuracy: **100% at every checkpoint**;
- held-out equality accuracy: **100% at every checkpoint**;
- minimum direct BF16 margin: **0.1021**;
- minimum equality BF16 margin: **0.0449**;
- failed or retried steps: **0**.

No averaging exception was used: every seed passed independently.

## What is now known

At this controlled scale, a compiler-written digital plane can add a
qualitatively stronger exact held-out factual/compositional capability than
the matched and twice-trained gradient controls, while keeping deployed
parameters, tensor shapes, graph, KV state, precision, and serving FLOPs
unchanged and keeping natural NLL inside a narrow fixed tolerance.

## What is not known

The result does not yet show that the model can:

1. extract stable structure from messy, non-repeated real prose;
2. route unrestricted natural questions into the plane without a hand-built
   four-token query protocol;
3. outperform a genuinely strong frontier-style training control at scale;
4. preserve positive BF16 margins over production-length training;
5. improve broad downstream capability enough to call the deployed model
   smarter.

## Next research boundary

Do not tune the synthetic lookup task further.  The next experiment must test
the language-to-plane interface: learned natural-language addressing into the
same fixed-cost plane, with document-held-out real prose, natural questions,
strong gradient/RAG-style controls priced honestly, and the same no-regression
language gate.

Passing that test would still require larger-scale confirmation before a
production claim.  Failing it would be evidence that the hand-designed
interface, not the plane algebra, is the current wall.
