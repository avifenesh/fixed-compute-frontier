# Radial-trust Cayley B2 runtime NVML scalar-schema audit receipt

Date: 2026-08-01  
Verdict: **PASS — pure CPU NVML result-schema slice only**

## Audited artifacts

- Schema evaluator:
  `experiments/radial_trust_cayley_b2_runtime_nvml_schema.py`
- Evaluator SHA-256:
  `cbad8ceb144ba47fcc7530968cbb2bc3828305990431d68efc1507fc115841c0`
- Tests:
  `tests/test_radial_trust_cayley_b2_runtime_nvml_schema.py`
- Tests SHA-256:
  `d99932ddc8f5880690479e3e2025fa08a62865cf48d577b3f016acdeed6978cc`
- Frozen parent SHA-256:
  `31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`
- Normative Amendment 1 SHA-256:
  `9259dc8c5c4e0bb3f3ac0bb6bda207f87b8d840dc93bd36b7e562b0d7affa900`
- Pinned amd64 container digest:
  `nvcr.io/nvidia/pytorch@sha256:4bf906c628d572681a977681765a44a3c1d7e34b39633b7160a84de707608139`
- Pinned-image `/usr/local/cuda/include/nvml.h` SHA-256:
  `28b51fbd44df16adf1e58229778414a4d1e7e05fdd4a74526ef0affb75f18416`

The final focused schema suite produced 29 passes. The combined B2 CPU suite
produced:

```text
192 passed in 2.09s
```

The header was rehashed inside the exact pinned image. The test container ran
with `--network none` and without GPU device exposure.

## Independent falsification

Reviewer: `/root/tesr_claim_review`

The audited partition contains exactly four fixed-buffer string queries, five
unsigned scalar queries, and three paired queries after excluding the separately
audited PCI result and the still-unimplemented variable-length clock lists.
Successful strings require a nonempty first-NUL logical value in the complete
zero-initialized ABI-sized buffer; bytes after that first NUL are ignored.
Successful integers are exact uint32 values and enum-valued queries reject
unknown values.

Only `NVML_ERROR_NOT_SUPPORTED` is representable as a non-success invariant
record. Every other return invalidates; a not-supported result carrying stale
payload also invalidates. Exact result classes, primitive types, immutable query
specifications, uniqueness, completeness, bytewise query sorting, power-range
ordering, and configured-limit consistency are enforced. The final test change
independently pins the complete mapping `GSP=64`, `infoROM=16`, `UUID=96`, and
`VBIOS=32`, so swapping two correct constants cannot evade the suite. The
reviewer reproduced all 29 focused passes and found no remaining slice-local
correctness or provenance blocker.

## Authorization boundary

This pass proves only conversion of already captured NVML ABI results into the
frozen canonical-tree records. It does not load NVML, resolve a device handle,
execute an NVML call, implement the two-call supported-clock enumeration,
collect PCI identity, compose the complete container invariant, initialize
CUDA, measure a GPU, or establish model capability. It authorizes continued
CPU implementation only and no local GPU sampling, GPU execution, rental, or
B2 physical measurement.
