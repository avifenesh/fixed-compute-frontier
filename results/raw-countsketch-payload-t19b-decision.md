# Raw CountSketch payload T19b — decision

Status: **FAIL; close lexical CountSketch payload**  
Date: 2026-07-31

## Decision

Do not train the T19b LM.  The exact physical construction passes, but the
frozen raw unigram/bigram/year payload is not sufficient for held-out natural
comparison reasoning.  Do not rescue it with different hashes, dimensions,
features, classifiers, thresholds, or regularization.

## Result

The fixed reader fit all 146 training questions exactly and then scored only
63/104 = **60.5769%** on the development questions.  The preregistered gate was
at least 75% and at least ten points above T12's 60.5769% semantic probe.  Both
primary gates failed.

This is not a storage or numerical failure:

- 129,971 raw lexical features were compiled;
- minimum BF16-to-FP32 payload cosine was 0.9999971;
- all 2,405 payloads were finite and nonzero;
- all 250 questions found exactly two raw titles without support annotations;
- the fixed logistic reader was finite and converged in 25 closure calls;
- train and evaluation each contained both labels.

## Exact retained physical result

The following ordinary-Transformer write layout is valid:

| write | entries |
|---|---:|
| title token codes | 132,800 |
| FFN gate keys | 76,960 |
| gate thresholds | 2,405 |
| up constants | 2,405 |
| 220-dimensional down payloads | 529,100 |
| **total** | **743,670** |

This uses 2,405 existing channels across three ordinary FFNs and remains 64
entries below the 743,734-entry budget, with identical class, state-dict
structure, parameter count, and serving graph.

## Interpretation

The payload preserved lexical evidence but not the abstractions required by
the questions.  Whole-document token overlap cannot reliably distinguish:

- a type or occupation from a merely mentioned word;
- a birth/death interval from an unrelated date;
- a shared latent value such as country when the question does not name it;
- relation-specific comparisons such as before/after, equality, and overlap.

The perfect training fit plus weak evaluation makes the missing object clear:
more storage or a more flexible decoder would only memorize the 146 relation
surfaces.  The compiler needs raw-only **typed relational structure**, not a
larger lexical sketch.

This does not authorize a pretrained semantic extractor.  The next admissible
question is whether corpus-wide repeated syntax can induce typed relation/value
slots through a bounded graph/permutation procedure cheaper than a language
model.  If it cannot, the stronger-compiler objection becomes fundamental.

## Evidence

- Result SHA-256:
  `1d914d1a18994c9892fab1b16b6b053498895e08eea171399be901b6880bb0a3`.
- Source SHA-256:
  `17b030f69b86dc7be68dfc63e363eb45962032393cddff360a333fe412263598`.
- Preregistration SHA-256:
  `861f5690a99d70f232373bfd6c071cf1035bba8a26d3b6ec2e169d66d70110d9`.
