# Radial-trust Cayley B2 runtime-platform filesystem audit receipt

Date: 2026-08-01  
Verdict: **PASS — CPU BDF/proc/module filesystem slice only**

## Audited artifacts

- Collector:
  `experiments/radial_trust_cayley_b2_runtime_platform_filesystem.py`
- Collector SHA-256:
  `e39323c7b7fa44788f805ea37db21ac381b07d3dd4159c4f21bf6e53005cc055`
- Tests:
  `tests/test_radial_trust_cayley_b2_runtime_platform_filesystem.py`
- Tests SHA-256:
  `0f20089c9effe3500e66503890674316790121447d9a503db2e33b84cb368be2`
- Frozen parent SHA-256:
  `31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`
- Normative Amendment 1 SHA-256:
  `9259dc8c5c4e0bb3f3ac0bb6bda207f87b8d840dc93bd36b7e562b0d7affa900`
- Audited PCI-identity implementation SHA-256:
  `a92b6f58eaa6cc35ff76c1a7a0e3259d9c92cfb7ed5cff7bdf48095f5caa0078`
- Pinned amd64 container digest:
  `nvcr.io/nvidia/pytorch@sha256:4bf906c628d572681a977681765a44a3c1d7e34b39633b7160a84de707608139`

The final focused platform-filesystem suite produced 17 passes. The combined
B2 CPU suite produced:

```text
163 passed in 2.02s
```

The container ran with `--network none` and without GPU device exposure.

## Independent falsification history

Reviewer: `/root/tesr_claim_review`

The first audit found three blockers. A file that permitted `O_PATH` but denied
the verified read lost its required exact `EACCES`; a read failure could be
recorded after the pathname had already changed inode; and a regular file or
symlink substituted for a module-parameter directory was underreported as an
ordinary `ENOTDIR` record.

The corrected reader pins the exact named inode, maps only permission denial on
that verified inode to a named errno, and revalidates the read descriptor, path
descriptor, and freshly reopened pathname on success and failure. Module
directories are first pinned with `O_PATH|O_NOFOLLOW`, required to be actual
directories, opened through their verified procfd, and revalidated after the
one-level snapshot. Structural substitution is invalid rather than absence.

The second audit reproduced Linux's enumerable-but-non-searchable directory
case: listing exposed a child name while `stat_at` returned `EACCES`. The final
collector records that exact errno against the listed child path before its type
is knowable; `ENOENT` after listing remains a race and invalidates the snapshot.
The reviewer reproduced the real mode-`0444` directory and mode-`000` file
cases, all 17 focused passes, the inode-replacement and structural-substitution
regressions, and a descriptor stress test whose count remained `4 -> 4 -> 4`
across 50 success and 50 exception collections.

## Authorization boundary

This pass covers only the 13 frozen files at the one Amendment-1-derived BDF,
the exact NVIDIA `version`, `params`, and per-BDF `information` proc files, and
direct regular or symlink entries under the two exact module-parameter
directories. It establishes bytewise path sorting, exact named-path errno
provenance, symlink non-following, and fail-closed race/structure checks. It
does not collect NVML fields, compose the complete container invariant,
supervise PID 1, capture a process snapshot, initialize CUDA, measure a GPU, or
establish model capability. It authorizes continued CPU implementation only
and no local GPU sampling, GPU execution, rental, or B2 physical measurement.
