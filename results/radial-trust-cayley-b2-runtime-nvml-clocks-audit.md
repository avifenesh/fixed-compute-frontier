# Radial-trust Cayley B2 runtime NVML clock-schema audit receipt

Date: 2026-08-01  
Verdict: **PASS — pure CPU supported-clock transcript slice only**

## Audited artifacts

- Transcript evaluator:
  `experiments/radial_trust_cayley_b2_runtime_nvml_clocks.py`
- Evaluator SHA-256:
  `1d9a4cb37d69c0daf01ca0ad17f7984346c204e21c3c2cdca059d34194502b2f`
- Tests:
  `tests/test_radial_trust_cayley_b2_runtime_nvml_clocks.py`
- Tests SHA-256:
  `0adfc1ab1a365f898f63fc24bbe1c2abd9ee95fcc10259904ae7d67377a170d9`
- Effective-paper Amendment 2 SHA-256:
  `746dfcf756038f950cd393a79ec30e499944897bd19c972ce9da7112a97e9564`
- Pinned-image `nvml.h` SHA-256:
  `28b51fbd44df16adf1e58229778414a4d1e7e05fdd4a74526ef0affb75f18416`
- Pinned amd64 container digest:
  `nvcr.io/nvidia/pytorch@sha256:4bf906c628d572681a977681765a44a3c1d7e34b39633b7160a84de707608139`

The final focused clock suite produced 29 passes. The combined B2 CPU suite
produced:

```text
221 passed in 2.08s
```

The container ran with `--network none` and without GPU device exposure.

## Independent falsification history

Reviewer: `/root/tesr_claim_review`

The first implementation was rejected because it numerically sorted MHz values
instead of sorting their canonical decimal-string keys bytewise, silently
borrowed the process-list maximum-three-retry rule, and omitted the call inputs
needed to prove `count=0,NULL` and exact second-call capacity. That defect caused
Amendment 2 to be frozen and independently audited before implementation.

The replacement contains exactly one call-pair transcript with explicit first
input count/pointer, first return/count, second capacity/pointer, second
return/count, and complete original-index zero-initialized buffer. It rejects
every second-call failure without retry, nonzero unused tail, duplicates, wrong
or missing fields, and wrong primitive or record types. Published values use
bytewise ASCII-decimal order; the raw `[900,1000]` regression publishes
`["1000","900"]` while retaining the raw buffer order. Graphics transcripts
must occur exactly once per returned memory clock and in that canonical order.

The implementation audit then found that a deceptive `str` subclass could
forge the private outcome union and publish an arbitrary status. The final
publication boundary requires an exact built-in string before any equality and
re-evaluates the immutable source transcript. The reviewer reproduced all 29
focused passes and found no remaining outcome, transcript, or record-field
forgery.

## Authorization boundary

This pass establishes only the pure transcript state machine, canonical order,
and memory-to-graphics completeness. It does not allocate a C buffer, prove
allocation-failure handling, load NVML, prove identical device handles or
graphics selectors in a caller, execute either supported-clock API, initialize
CUDA, measure a GPU, or establish model capability. It authorizes continued CPU
implementation only and no local GPU sampling, GPU execution, rental, or B2
physical measurement.
