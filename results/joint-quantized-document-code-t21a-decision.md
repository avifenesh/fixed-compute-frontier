# Joint quantized document code T21a — decision

Status: **FAIL; CLOSE THE FROZEN FREE-CODE AUTOREGRESSIVE RECORD**  
Date: 2026-07-31

## Decision

Do not physically export this record.  Do not tune its width, level count,
amplitude, injection coordinate, optimizer, raw-document exposure, QA schedule,
title router, or thresholds.

T21a learned a weak document-specific side channel, but not a dense knowledge
representation.  Its correct records produced small causal effects and no
held-out advantage over dense-1x training.  The result fails the preregistered
acquisition, capability, and record-dependence gates.

This closes the frozen mechanism in which one independently optimized
220-cell free parameter vector per document is trained only through ordinary
teacher-forced next-token loss.  It does not say that the code path was broken
or completely ignored: code gradients were nonzero and both document NLL and
QA moved under ablation.  The effect was simply too small.

## Document-channel evidence

The full 2,405-document corpus produced:

| arm / code mode | document NLL |
|---|---:|
| joint code, correct | 4.885034 |
| joint code, zero | 5.085703 |
| joint code, shuffled | 4.927442 |
| dense-1x | 4.887899 |
| dense-2x-document | 4.455406 |

The correct record improved NLL by only **3.9458% over zero** and **0.8606%
over shuffle**, below both required 10% gates.  Its NLL was effectively the
same as dense-1x and materially worse than simply giving the dense model twice
the document exposure.

The code path did receive learning signal: maximum code-gradient norm was
**4.8746**.  Therefore the failure is not zero gradient, missing injection, or
serialization.  Teacher-forced raw prediction assigned too little useful work
to the optional code.

## Held-out natural-QA evidence

All three arms fitted the 146 QA-training examples.  On the 104 title-disjoint
development rows:

| arm / ablation | accuracy |
|---|---:|
| joint code, correct | 60.5769% |
| joint code, zero | 56.7308% |
| joint code, shuffled | 55.7692% |
| dense-1x | 60.5769% |
| dense-2x-document | 54.8077% |

The candidate gained **0 points** over the best dense arm, versus the required
10 points, and remained below the required 75% accuracy.  Zeroing cost only
**3.8462 points** and shuffling cost **4.8077 points**, both below the required
10-point causal drops.

The small ablation effects establish some record use, but not enough to claim
better knowledge density.  In particular, correct code did not outperform the
dense-1x model at all.

## Contracts that passed

- Frozen raw, QA, tokenizer, natural-data, and preregistration hashes matched.
- All arms began from byte-identical model states and finished every training
  step without failure or retry.
- The candidate used all and only the 16 frozen levels, and the decoded indices
  survived BF16 roundtrip exactly.
- The record hash was unchanged through QA training.
- Candidate and controls retained one 36,577,152-parameter served schema.
- The prospective physical ledger was exactly 743,670 existing entries, below
  the 743,734 cap.
- Protected natural NLL passed: 6.707012 for the candidate versus 6.827360 for
  dense-1x, a 1.7627% improvement rather than degradation.

The first execution completed its learning and evaluation path but failed only
while serializing a scalar tensor at the final JSON endpoint.  Two accidental
duplicate replay processes were detected and terminated before evidence was
read.  The exact single replay used only a serialization conversion, ran from
step zero with no other ML process on the H100, and produced this result.

## What was learned

An optional free record can absorb a small amount of document identity and
prediction residual, but ordinary autoregressive likelihood does not make that
record carry enough query-relevant structure.  More storage or more aggressive
tuning would not repair the missing learning contract and is disallowed by the
frozen decision rule.

T21a was an upper bound rather than an autonomous compiler because it optimized
one persistent parameter vector for every document.  Its failure makes a
forward-only autonomous writer harder, not easier.  T22a is separately frozen
to force disjoint raw-derived functional reads and equality composition.  It
has not yet produced a capability result; its purpose is to test a different,
causally necessary learning contract rather than rescue this closed objective.

## Evidence hashes

- Result JSON SHA-256:
  `604ebdaff1e176b1b9193cc440e124acc6be9b10511b52f1e5b2c3d8f0536ecd`
- Exact replay log SHA-256:
  `b70af1e04d66b0e641f39e91e9da33fbe43f7bd55301d3e5ea988cd9476bf629`
- Preserved first-run serialization-failure log SHA-256:
  `06b25b8ede5d33d0ade9f30dab6868d56abd6671373e02b752c767db8d5d00b9`
- Preserved duplicate-abort log SHA-256:
  `8f975b6eb6da7e7f738e51f5de0d31e3eec57900f0dc558298f900df5d53a593`
- Executed source SHA-256:
  `c21688b70a5582c6138202f36189d2a615a7f8cbb2c15c37c31d52cd7574a044`
- Preregistration SHA-256:
  `5550066c5661dd1d91d7beff6edb7f7b4612c904161eb948ff1a2e90774164a8`
