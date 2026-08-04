# Radial-trust Cayley B2 FP64 forced-route oracle audit receipt

Date: 2026-08-01  
Verdict: **PASS — CPU FP64 forced-route oracle slice only**

## Audited artifacts

- Oracle:
  `experiments/radial_trust_cayley_b2_fp64_oracle.py`
- Oracle SHA-256:
  `9f3d6acedeac2ce9b44ebda56020f59a0c42467c1118715aad65be84f368a08b`
- Tests:
  `tests/test_radial_trust_cayley_b2_fp64_oracle.py`
- Tests SHA-256:
  `6ea976ea5a84ca965e618c03ac17b9e16411ce8efba5549d19b57ea1d5d8cd15`
- Frozen parent SHA-256:
  `31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`
- Typed map SHA-256:
  `e96dfb57d3ed4c44da62925f3bdb8fd3d8f1c8be09a61bb6dad564578d9c484c`

The final combined CPU-specification and oracle suite produced:

```text
25 passed in 0.12s
```

## Independent falsification history

Reviewer: `/root/tesr_claim_review`

The first audit failed despite 22 passing tests. It demonstrated that finite
BF16 artifacts with basis coefficients `2^43` overflowed the naive FP64 square,
silently produced a zero trust scale, returned an unchanged finite output, and
recorded an infinite raw RMS. It also found that the frozen wrapper accepted
arbitrary FP64 inputs and that the tests did not causally protect route
coordinates or nonzero end-to-end wiring.

The corrected oracle uses a scaled sum-of-squares RMS and `hypot(1,rms)` radial
denominator, checks every trust intermediate, and produces a finite nonzero
update for that extreme case. The strict frozen wrapper now accepts only one
C-order raw BF16 `[B,4096]` input and owns its exact FP64 decode. An independent
dense-matrix calculation protects payload-branch selection, depth-basis
selection, transpose use, and complete nonzero forced paths.

The second audit found that toy-width coordinate literals constrained route
constants only modulo eight. The final tests add literal width-4,096 checks for
the seed base, node multiplier, depth stride, layer stride, and a combined
maximum-node case. The reviewer reran the final suite and found no remaining
numerical or contract issue in this slice.

## Authorization boundary

This oracle is a mathematical FP64 diagnostic. It is not the staged-BF16
serving reference and has not executed a target-width packed layer. This receipt
does not establish target-shape correctness, CUDA correctness, physical
efficiency, language capability, or production-model improvement. It authorizes
no GPU check, local GPU run, rental, or B2 physical measurement.
