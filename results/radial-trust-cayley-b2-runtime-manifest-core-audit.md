# Radial-trust Cayley B2 runtime-manifest core audit receipt

Date: 2026-08-01  
Verdict: **PASS — CPU runtime-manifest core only**

## Audited artifacts

- Core:
  `experiments/radial_trust_cayley_b2_runtime_manifest.py`
- Core SHA-256:
  `3f007ef1b9fa1fe52ef215b4a004640daedc3123c6a11a2471e8b0afef8e3292`
- Tests:
  `tests/test_radial_trust_cayley_b2_runtime_manifest.py`
- Tests SHA-256:
  `a03f4c37d8c615fc5a85e0474fbea931b86b61466ca2ecbe14a38bea882f2b17`
- Frozen parent SHA-256:
  `31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`
- Pinned amd64 container digest:
  `nvcr.io/nvidia/pytorch@sha256:4bf906c628d572681a977681765a44a3c1d7e34b39633b7160a84de707608139`
- Exact NumPy 2.3.5 wheel SHA-256:
  `0d8163f43acde9a73c2a33605353a4f1bc4798745a8b1d73183b28e5b435ae28`

The final focused core suite produced 18 passes. The combined B2 CPU suite
produced:

```text
68 passed in 2.04s
```

The container ran with `--network none` and without GPU device exposure.

## Independent falsification history

Reviewer: `/root/tesr_claim_review`

The first audit found a real filesystem-root join bug, noncanonical cgroup
paths whose recorded and accessed forms could diverge, tests coupled to the
same frozen-name constants they were meant to protect, incomplete secret-name
mutation coverage, a mismatch between the ASCII canonical subset and accepted
values, and cached manifest hashes that could be forged.

The corrected core special-cases root joins, accepts only `/` or canonical
absolute cgroup components, independently pins both schema names, every named
cgroup file, every raw environment allowlist entry, and every secret fragment,
and tests all four secret fragments case-insensitively inside otherwise
publishable names. It restricts the manifest tree to the stated ASCII/no-JSON-
number subset and revalidates cached hashes, exact envelopes, embedded schema,
and canonical bytes at comparison boundaries.

The second audit constructed internally consistent blobs that bypassed the
normal constructor with an unknown schema or non-object payload. The final
boundary validator independently enforces both constructor invariants. The
regressions use canonical, correctly hashed adversarial blobs, so their
rejection is causal. The reviewer reproduced all 18 focused passes and found no
remaining slice-local correctness or security issue.

## Authorization boundary

This receipt covers only canonical bytes, base64url and decimal encodings,
secret-safe environment records, exact-errno path records, unified-cgroup-v2
path resolution, named cgroup collection, manifest hashing, and exact boundary
comparison. It does not establish the complete `b2-container-runtime-v1` or
`b2-process-runtime-v1` field sets, PID-1 supervision, mount/device/PCI
collection, NVML behavior, NUMA/rlimit/maps capture, CUDA allocation tracing,
or any GPU property. It authorizes continued CPU implementation only and no
local GPU sampling, GPU execution, rental, or B2 physical measurement.
