# Radial-trust Cayley B2 staged BF16 reference audit receipt

Date: 2026-08-01  
Verdict: **PASS — pinned-runtime CPU staged-reference slice only**

## Audited artifacts

- Reference:
  `experiments/radial_trust_cayley_b2_staged_reference.py`
- Reference SHA-256:
  `d64d4aa35cf453361ea0bb59bc826c787f44a112f976b032b8639a970c803df8`
- Tests:
  `tests/test_radial_trust_cayley_b2_staged_reference.py`
- Tests SHA-256:
  `7f20232362791e48339d636c95701a058fbbe1076c001ed532b89f130d57f0f3`
- Frozen parent SHA-256:
  `31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`
- Current typed-map SHA-256:
  `8317e77188033d50014fe57a69bc77d6ebaac97fb29389df62a8e2b2355908c0`
- Pinned amd64 container digest:
  `nvcr.io/nvidia/pytorch@sha256:4bf906c628d572681a977681765a44a3c1d7e34b39633b7160a84de707608139`
- Exact NumPy 2.3.5 wheel SHA-256:
  `0d8163f43acde9a73c2a33605353a4f1bc4798745a8b1d73183b28e5b435ae28`

The final combined CPU-specification, FP64-oracle, route-checker, and staged-
reference suite produced:

```text
50 passed in 2.00s
```

The container ran with `--network none` and without GPU device exposure.

## Independent falsification history

Reviewer: `/root/tesr_claim_review`

The initial audits found that a merely deterministic nonzero test did not
independently protect the complete staged computation, target-width topology
and route constants were underconstrained, dtype and residual checks were
self-referential, and the independent helper did not preserve matching order.

The final suite independently constructs the scalar matching operators, uses
libm FP32 `fmaf`, `expf`, and `sqrtf` at the frozen boundaries, pins asymmetric
forced-path output and trust-scale bits, compares every width-4,096 descriptor,
and checks nonzero-layer target-width route literals and the exact residual
constant bits. A rounding-sensitive U1 counterexample with terms
`[2^24,-2^24,1]` distinguishes the frozen accumulation order from a reordered
third term.

The last audit failure showed that every natural nonzero fixture selected only
right edges, so an always-right router could survive. The superseding test
feeds two independent nonzero rows through the real router and pins both the
independent scalar result and staged result to literal BF16 outputs and full
per-edge traces: one path has bits `(1,1)` and the other `(0,0,0)`. Thus both
always-right and always-left mutations fail, while a separate zero-input case
protects the exact tie rule `>=0 -> 1`. The reviewer found no remaining
self-reference, algebra/order, BF16-boundary, route, payload, strict-wrapper,
target-shape, or claim-scope issue in this slice.

## Authorization boundary

This pass establishes the CPU behavior of the serving reference in the pinned
software image. CPU execution does not reproduce SM90 instruction selection,
libdevice bits, SASS, CUDA graph behavior, device allocation accounting, HBM
traffic, latency, or energy. It does not establish useful learned routing,
language capability, or production-model improvement. It authorizes only the
next CPU implementation step: the runtime invariant/snapshot recorder and
manifest schema. It authorizes no local GPU sampling, GPU execution, rental,
or B2 physical measurement.
