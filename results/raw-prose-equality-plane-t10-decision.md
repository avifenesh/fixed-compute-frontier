# Raw-prose equality plane T10 — decision

Status: **closed as non-Pareto; representation result retained**  
Date: 2026-07-31

## Decision

T10 is not a smarter production LLM and does not satisfy the project goal.
It adds an exact compiler-written factual/equality capability at unchanged
serving graph and parameter count, but it weakens protected natural-language
modeling beyond the preregistered tolerance.

The concrete T10 construction is closed.  Its negative result must not be
relabelled as success, and its exact representation result alone is not a
production-capability claim.

## Measured result

The candidate retained 100% held-out direct-fact and equality accuracy with a
positive BF16 margin at steps 250, 500, and 1,000.  The best twice-trained
gradient control reached only 8.41% direct accuracy and 77.51% equality
accuracy, so the added structured capability was qualitative rather than a
sub-percent benchmark fluctuation.

The candidate nevertheless lost protected natural quality against the matched
`muon_labels_1x` control:

| Step | Candidate NLL | Matched control NLL | Relative loss | Gate |
|---:|---:|---:|---:|:---:|
| 250 | 6.674094 | 6.644705 | +0.4423% | pass |
| 500 | 6.407662 | 6.358583 | +0.7718% | fail |
| 1,000 | 6.188970 | 6.128630 | +0.9846% | fail |

The frozen writer occupies 7,824,855 of 36,577,152 parameter entries
(21.39%).  In particular, it removes the first 55 coordinates from every
ordinary token embedding and isolates two attention heads plus 64 SwiGLU
channels through all ten layers, although the exact computation is completed
by block 2.  This is a measured footprint and a plausible explanation for the
growing NLL gap, not yet a causal finding.

## What survives

Only the following result is retained: a raw-string compiler can recover the
controlled prose table and write an exact direct/equality computation into the
ordinary Transformer graph without adding deployed parameters, KV state, or
serving operations.

## Next admissible test

T11 may test the same digital-plane mechanism with minimal triangular
isolation: preserve only the entries required for the three-block computation
and final answer registers, while returning the remaining embedding,
attention, and FFN capacity to language learning.  It must rerun the identical
matched controls and the same 0.5% natural-NLL gate.  Failure closes that
capacity-preserving refinement; it does not justify moving the goalposts.

Evidence:

- `raw-prose-equality-plane-t10-quick.json`
- `raw-prose-equality-plane-t10-training-pilot.json`
- `raw-prose-equality-plane-t10-training-pilot-protocol.md`
