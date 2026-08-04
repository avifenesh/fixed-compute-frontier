# Radial-trust Cayley B2 route-checker audit receipt

Date: 2026-08-01  
Verdict: **PASS — independent CPU route-checker slice only**

## Audited artifacts

- Checker:
  `experiments/radial_trust_cayley_b2_trace_checker.py`
- Checker SHA-256:
  `94d57dcff01b269d9eb672e557189d37b80236efa10536d48ec7c02be82d1c81`
- Tests:
  `tests/test_radial_trust_cayley_b2_trace_checker.py`
- Tests SHA-256:
  `9a2ef2ae23384dc697d89047a3c46cba9b6c7752a24f9afd7a622cb4fe6ec2a0`
- Current typed-map SHA-256:
  `8317e77188033d50014fe57a69bc77d6ebaac97fb29389df62a8e2b2355908c0`

The final combined CPU-specification, FP64-oracle, and route-checker suite
produced:

```text
36 passed in 0.14s
```

The independent reviewer reproduced 36 passes in 0.13 seconds.

## Independent falsification history

Reviewer: `/root/tesr_claim_review`

The first review accepted the current SplitMix64 values, wrap sequence, route
coordinate formula, collapsed/uniform leaves, node/child/payload-edge
recurrences, and 12/13-edge termination, but failed the slice for three gaps:

1. fractional forced bits were truncated by `int` and accepted as binary;
2. collapsed routes bypassed frozen row/layer domain checks; and
3. target-width coordinate literals used only layer zero, so a material layer-
   stride mutation could survive.

The corrected checker requires the integer index protocol for every bit,
validates row and layer before choosing either diagnostic regime, and protects
the nonzero-layer target-width coordinate
`route_coordinate(7165,12,11,4096)==400`. The reviewer verified that the prior
mutations now fail and found no new mathematical, termination, strict-wrapper,
independence, or storage-contract drift.

## Authorization boundary

The checker defines semantic route records only. It does not choose or approve
a CUDA trace-buffer dtype, padding sentinel, shape, or storage layout. It does
not establish staged-BF16 agreement, target-shape execution, CUDA correctness,
physical efficiency, language capability, or production-model improvement. It
authorizes no GPU check, local GPU run, rental, or B2 physical measurement.
