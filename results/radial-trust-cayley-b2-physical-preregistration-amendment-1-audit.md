# Radial-trust Cayley B2 physical preregistration Amendment 1 audit

Date: 2026-08-01  
Verdict: **PASS — Amendment 1 is normative and active**

## Effective paper identity

- Historical parent:
  `results/radial-trust-cayley-b2-physical-preregistration.md`
- Parent SHA-256:
  `31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`
- Normative Amendment 1:
  `results/radial-trust-cayley-b2-physical-preregistration-amendment-1.md`
- Amendment 1 SHA-256:
  `9259dc8c5c4e0bb3f3ac0bb6bda207f87b8d840dc93bd36b7e562b0d7affa900`

The effective B2 physical preregistration is the ordered pair of these hashes.
Neither historical file nor any earlier audit receipt is rewritten.

## Finding and resolution

The parent used an undefined `<BDF>` in mandatory sysfs and NVIDIA procfs
paths. In the pinned NVML header, the v3 `busId` uses eight uppercase domain
digits and `busIdLegacy` uses four uppercase domain digits; Linux path bytes
use the four-digit lowercase PCI basename. The alternatives are not
behaviorally interchangeable on a case-sensitive filesystem.

Amendment 1 derives one lowercase `path_bdf` from the v3 numeric tuple, requires
the two returned logical strings to match their exact uppercase formats for
that tuple, checks the function-zero representation and numeric ranges, and
forbids every lookup or fallback. At every invariant snapshot the v3 call
precedes dependent reads. Its exact return code is always recorded; any
non-success invalidates B2 and permits neither derivation nor either read.

The independent reviewer verified the parent and amendment hashes, reproduced
the pinned `nvml.h` SHA-256 and relevant definitions inside the exact container
without GPU exposure, and approved the final total control flow. The added BDF
fields remain additive to the complete v3 result.

## Authorization boundary

This audit activates only the BDF definition in Amendment 1. It establishes no
NVML success, GPU identity, filesystem availability, physical validity,
performance, or capability. It authorizes CPU implementation and falsification
of the amended platform schema only. It authorizes no local GPU sampling, GPU
execution, rental, or B2 measurement.
