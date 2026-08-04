# Raw self-query functional record T22a — decision

Status: **FAIL; CLOSE THE FROZEN QUOTIENT-RECORD TRANSPORT**  
Date: 2026-07-31

## Decision

Do not tune the record width, levels, amplitude, optimizer, exposure,
relation-window quotient, thresholds, control tokens, or QA schedule.  Do not
attempt physical export.  Close this frozen mechanism.

T22a successfully made the record useful on exposed training functions, but it
did not make the record a reusable document program.  On disjoint raw-derived
edges, the correct record was barely better than zero or a wrong document's
record, lost to both dense controls, and transferred only 0.96 points to
title-disjoint natural QA.  It also violated the protected natural-quality
gate substantially.

This is not a repeat of T21a's optional-code failure.  T22a removed the target
from the query, supplied exact raw-only function labels, balanced composition,
and gave every document a train/evaluation split.  The candidate fit those
training functions strongly.  The new negative result is failure to compress
that fitted lookup behavior into structure that extrapolates to an unseen
query of the same document.

## Raw functional evidence

The candidate's held-out structural unary NLL was:

| record mode | NLL |
|---|---:|
| correct | 8.706794 |
| zero | 8.760444 |
| shuffled | 9.061359 |

The correct-record gains were only **0.6124% over zero** and **3.9129% over
shuffle**, versus the required 20% for both.  Dense-1x scored **8.286604** and
dense-2x scored **8.223974**, so the candidate's correct record was worse than
either ordinary model on the disjoint reads.

Balanced held-out equality accuracy was:

| arm / record mode | accuracy |
|---|---:|
| functional record, correct | 58.3333% |
| functional record, zero | 50.0000% |
| functional record, shuffled | 56.9444% |
| dense-1x | 61.1111% |
| dense-2x | 61.1111% |

Correct code gained only **8.3333 points over zero** and **1.3889 points over
shuffle**, missed the 80% absolute gate, and scored **2.7778 points below** the
best dense control.  Most of the equality behavior therefore survived using
the wrong document.

The path was not broken.  Code-gradient norm reached **21.0093**, all 16
quantization levels were used, and the final exposed equality batch reached
**0.00100 loss**, compared with **0.15550** for dense-1x.  The sharp separation
between exposed-batch fit and disjoint-edge evaluation identifies memorization,
not missing gradients or insufficient training fit.

## Natural transfer and interference

On 104 title-disjoint QA rows:

| arm / record mode | accuracy |
|---|---:|
| functional record, correct | 57.6923% |
| functional record, zero | 48.0769% |
| functional record, shuffled | 55.7692% |
| dense-1x | 56.7308% |
| dense-2x | 50.0000% |

The candidate gained only **0.9615 points** over the best dense arm, far below
the required 10 points, and missed the 75% absolute gate.  Zeroing cost
**9.6154 points**, narrowly below the causal threshold, but shuffling cost only
**1.9231 points**.  This means QA learned to expect a record-like activation;
it did not depend strongly on selecting the correct document record.

After QA training, protected natural NLL was **6.794118** for the candidate and
**6.437711** for dense-1x: a **5.5362% regression** against the allowed 0.5%.
Before QA, the candidate was only 0.3803% worse, so most interference appeared
while the natural reader adapted to the record-conditioned QA path.

## Contracts that passed

- Raw-only compiler fields, all frozen data hashes, Stage-0 counts, disjoint
  edges, control-token nonoccurrence, and context bounds matched.
- All three arms began byte-identically and completed every frozen update with
  finite loss/gradient and zero retry.
- The candidate record remained on the 16 frozen levels, preserved its decoded
  indices through BF16, and was unchanged during QA training.
- Candidate and controls retained one 36,577,152-parameter state schema.
- The prospective physical ledger remained exactly **743,670** existing
  parameter writes, within the 743,734 cap.

These contracts rule out implementation failure as an explanation for the
capability result.

## Mechanistic conclusion

A free bounded vector plus a shared nonlinear reader can memorize exposed
`document x skeleton -> value` edges without learning a coordinate system that
supports unseen edges.  Making record use causally necessary solves the
*ignore-the-record* problem but does not solve the *what must the record mean*
problem.  Per-document gauge freedom remains: many code/reader combinations
fit the same exposed functions while disagreeing on every withheld function.

The next architecture therefore needs an enforced algebra over record state,
not only more queries against a free vector.  A valid successor must make raw
token updates compose through a shared, document-independent operation whose
effect on a query is determined before document-specific fitting.  It must be
tested with unseen documents as well as unseen edges, so an arbitrary
per-document lookup cannot satisfy the gate.

## Evidence hashes

- Result JSON SHA-256:
  `cdba1e8279c77d7cdac6ab52883cf11623c59b41fc6d457af05227515eaa9866`
- Clean run log SHA-256:
  `e0aff119d0f2dcd72ffac306908943a8a341e1d40a2d0a9e6da32accbc1f8f7a`
- Executed source SHA-256:
  `ff68db68976581ac70fdd4117d195b4c535c57ab9ec2e3af97fe87f98c750375`
- Test source SHA-256:
  `50cb983e0402c46989b2e66d8d5b57b48d2f6c0214e665ca017ca42fb9fc6678`
- Preregistration SHA-256:
  `fa70cd79d4d632c524db5da3db0e326edf932f5a996990d94e1bb8145c88e1fc`
- Stage-0 result SHA-256:
  `f46a36b5c617bb62e18e9fc81111dc3df2a8e8d351d5808e6173deaa79c1d660`
