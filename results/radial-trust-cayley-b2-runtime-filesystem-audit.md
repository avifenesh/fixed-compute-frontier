# Radial-trust Cayley B2 runtime-filesystem audit receipt

Date: 2026-08-01  
Verdict: **PASS — CPU mount and NVIDIA-device filesystem slice only**

## Audited artifacts

- Collector:
  `experiments/radial_trust_cayley_b2_runtime_filesystem.py`
- Collector SHA-256:
  `2b5488174de25f81f3206f239f56f4070f28221eee2356b769aa2b1be3497f67`
- Tests:
  `tests/test_radial_trust_cayley_b2_runtime_filesystem.py`
- Tests SHA-256:
  `d0903533db766d915e3f92aea829823773d96591046f58db3e4a31300c4320f7`
- Frozen parent SHA-256:
  `31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`
- Pinned amd64 container digest:
  `nvcr.io/nvidia/pytorch@sha256:4bf906c628d572681a977681765a44a3c1d7e34b39633b7160a84de707608139`

The final focused filesystem suite produced 33 passes. The combined B2 CPU
suite produced:

```text
101 passed in 1.94s
```

The container ran with `--network none` and without GPU device exposure.

## Independent falsification history

Reviewer: `/root/tesr_claim_review`

The first audit found that text-style `splitlines()` corrupted a legal raw CR
inside a mount pathname, mountinfo grammar was underchecked, pathname-based
device recursion could follow a directory raced into a symlink, and regular-
file hashes were not tied to the mount IDs they claimed to describe.

The rewritten parser is LF-byte-delimited, requires the kernel-emitted terminal
LF, validates positive IDs, canonical `major:minor`, exact post-separator
arity, and distinct pathname versus kernel-mangled escape sets while leaving
optional fields extensible. The device walker now uses descriptor-relative
`openat`/`stat`/`readlink`, `O_NOFOLLOW`, inode checks, and before/after listing
and target checks. A deterministic directory-to-symlink swap fails closed.

The second and third audits found over-rejection of legal `\\043` and raw-CR
field data, false errno provenance, blocking `O_RDONLY` classification of a
FIFO, and an ineffective mount-ID after-check on the same pinned descriptor.
The final design classifies with `O_PATH|O_NOFOLLOW`, reads only a verified
regular inode through its procfd, rejects stacked mountpoints, verifies the
opened mount ID, then reopens the pathname and verifies both inode identity and
mount ID again. Named-file hashing does not call fdinfo, and infrastructure
failures cannot be emitted as false named-path errno records. The reviewer
reproduced all 33 focused passes, loop-probed for descriptor leaks, and found no
remaining slice-local correctness or security blocker.

## Authorization boundary

This pass covers strict mountinfo consumption, stable regular-file mount hashes,
hash-or-exact-errno named files, and recursive non-following `/dev/nvidia*`
metadata. It does not cover PCI/sysfs/module collection, the exact complete
container payload, NVML, PID-1 supervision, process snapshots, allocation
tracing, CUDA, performance, or capability. It authorizes continued CPU
implementation only and no local GPU sampling, GPU execution, rental, or B2
physical measurement.
