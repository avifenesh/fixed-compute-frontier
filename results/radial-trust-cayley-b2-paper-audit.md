# Radial-trust Cayley B2 paper-audit receipt

Date: 2026-08-01  
Verdict: **PASS — ready for implementation audit**

## Hashes

- Independently audited content hash:
  `fd5bf6f788600e38f1baa8385f25620c593ae622652c061826c269007346a23d`
- Frozen paper hash after the status-only mutation:
  `31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`
- Paper:
  `results/radial-trust-cayley-b2-physical-preregistration.md`

Replacing the frozen status line with the prior draft status line reconstructs
the independently audited content hash exactly. No scientific, measurement,
gate, acquisition, or teardown text changed during the freeze mutation.

## Independent review

Reviewer: `/root/tesr_claim_review`

The final review accepted:

- the narrow one-instance paired randomized physical-survivor claim;
- the exact scalar binary64 energy estimator;
- the exact single-pass CUPTI counter protocol;
- the persistent PID 1 auditor and separate measurement-process snapshots;
- the universal precheck, warmup, measurement, synchronization, postcheck order;
- exact `get_mempolicy` arguments and failure rules; and
- exact NVML v3 process enumeration, retry, measurement, and failure rules.

## Authorization boundary

This receipt authorizes implementation and independent code audit only. It
does not authorize a GPU check, local benchmark, rental, physical gate run,
language training, capability claim, or frontier-model claim. Section 15's
local-first check remains mandatory immediately before any later rental.
