# Radial-trust Cayley B2 physical preregistration Amendment 2 audit

Date: 2026-08-01  
Verdict: **PASS — normative clock-list call protocol is frozen**

## Effective paper identity

- Parent SHA-256:
  `31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`
- Amendment 1 SHA-256:
  `9259dc8c5c4e0bb3f3ac0bb6bda207f87b8d840dc93bd36b7e562b0d7affa900`
- Amendment 2 SHA-256:
  `746dfcf756038f950cd393a79ec30e499944897bd19c972ce9da7112a97e9564`
- Pinned-image `nvml.h` SHA-256:
  `28b51fbd44df16adf1e58229778414a4d1e7e05fdd4a74526ef0affb75f18416`

The effective B2 preregistration identity is the ordered hash triple above.
The parent and Amendment 1 remain byte-for-byte unchanged.

## Independent audit

Reviewer: `/root/tesr_claim_review`

The reviewer found that the first clock-list implementation incorrectly sorted
numeric MHz values instead of bytewise ASCII decimal keys and silently borrowed
the parent process-list maximum-three-attempt rule. The implementation also did
not carry explicit evidence of the first `count=0,NULL` input or the second
input capacity/pointer state.

Amendment 2 resolves those gaps with exactly one fail-closed two-call pair per
supported-clock query, complete input/output transcript fields, checked exact
allocation, zero tail, duplicate rejection, no retry after any second-call
failure, exact memory-to-graphics coverage, and ASCII-decimal bytewise call and
record order. It explicitly pins the same full-GPU handle and graphics selector
across both calls, original-index raw-buffer recording, and allocation-failure
invalidation.

The reviewer judged the state machine total, ABI-feasible, consistent with the
parent's general unsupported-query rule, and independent of the distinct
process-list retry protocol. The final amended SHA passed with no remaining
finding.

## Authorization boundary

This receipt activates Amendment 2 only for CPU implementation and independent
falsification. It does not establish NVML availability, GPU identity, stable or
nonempty supported-clock lists, physical feasibility, performance, or model
capability. It authorizes no local GPU sampling, GPU execution, rental,
instance creation, or B2 measurement.
