# Radial-trust Cayley B2 runtime-platform identity audit receipt

Date: 2026-08-01  
Verdict: **PASS — CPU NVML-PCI result and Linux-BDF derivation slice only**

## Audited artifacts

- Platform identity evaluator:
  `experiments/radial_trust_cayley_b2_runtime_platform.py`
- Evaluator SHA-256:
  `a92b6f58eaa6cc35ff76c1a7a0e3259d9c92cfb7ed5cff7bdf48095f5caa0078`
- Tests:
  `tests/test_radial_trust_cayley_b2_runtime_platform.py`
- Tests SHA-256:
  `011ad4cfe8f70287da146c2dec5fb3ec3c4ca9fa43675692f551ef7043cb0bb9`
- Frozen parent SHA-256:
  `31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`
- Normative Amendment 1 SHA-256:
  `9259dc8c5c4e0bb3f3ac0bb6bda207f87b8d840dc93bd36b7e562b0d7affa900`
- Pinned amd64 container digest:
  `nvcr.io/nvidia/pytorch@sha256:4bf906c628d572681a977681765a44a3c1d7e34b39633b7160a84de707608139`

The final focused platform suite produced 45 passes. The combined B2 CPU suite
produced:

```text
146 passed in 1.99s
```

The container ran with `--network none` and without GPU device exposure.

## Independent falsification history

Reviewer: `/root/tesr_claim_review`

The first audit found that callers could construct a frozen `PciIdentity` with
an out-of-range tuple, inconsistent NVML strings, or a traversal-valued
`path_bdf`, then obtain a purportedly valid record and BDF-dependent paths. It
also found an unhandled wrong-object result after `NVML_SUCCESS` and incomplete
zero/maximum tests for the five unsigned numeric fields.

Both output boundaries now independently revalidate the complete identity.
The evaluator treats a missing or wrong-typed success result as total invalid,
and the tests pin exact zero and maximum for domain, bus, device, PCI device ID,
and PCI subsystem ID.

The second audit defeated ordinary revalidation using a `bytes` subclass whose
overloaded equality falsely accepted `../../forbidden`; the forged value then
survived path concatenation. The final boundary accepts only exact built-in
`int` and `bytes` values and exact `RawPciInfo` and `PciIdentity` classes before
performing comparison or formatting. Adversarial tests cover deceptive bytes,
mutable byte arrays, deceptive integers, and direct record-class subclasses.
The reviewer reproduced all 45 focused passes and confirmed that weakening the
exact-class checks to `isinstance` makes the new subclass regressions fail.

## Authorization boundary

This pass covers only pure evaluation of the mandatory
`nvmlDeviceGetPciInfo_v3` return/result, fixed-array NUL handling, tuple/string
agreement, and derivation of the one lowercase Linux BDF permitted by Amendment
1. It does not call or load NVML, read sysfs or procfs, establish GPU presence,
collect the remaining NVML fields, compose the complete container invariant,
supervise PID 1, initialize CUDA, measure performance, or establish model
capability. It authorizes continued CPU implementation only and no local GPU
sampling, GPU execution, rental, or B2 physical measurement.
