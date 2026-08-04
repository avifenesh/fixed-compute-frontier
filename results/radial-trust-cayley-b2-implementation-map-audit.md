# Radial-trust Cayley B2 implementation-map audit receipt

Date: 2026-08-01  
Verdict: **PASS — typed implementation map only**

## Frozen inputs

- Parent physical preregistration SHA-256:
  `31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`
- Audited implementation map:
  `results/radial-trust-cayley-b2-implementation-map.md`
- Corrected map SHA-256:
  `e96dfb57d3ed4c44da62925f3bdb8fd3d8f1c8be09a61bb6dad564578d9c484c`

## Independent falsification and correction

Reviewer: `/root/tesr_claim_review`

The first map draft, SHA-256
`651dcef73b5431a8f5461f1bbc4fbbde51a04a58a081d72bd563456eab9b9543`,
failed review for three contract drifts:

1. it typed the 12-layer panel as a data chain instead of 12 independent layer
   inputs and outputs under serial launch order;
2. it mislabeled a valid predicted-versus-observed traffic mismatch as a harness
   failure instead of an instrumentation warning; and
3. it aggregated route and batch controls in a way that implied a prohibited
   Cartesian product.

The corrected map was independently re-audited and accepted because it now:

- types 12 distinct input/output pairs and prohibits output-to-input chaining;
- distinguishes exact requested-byte or counter-protocol invalidity from a
  greater-than-20-percent model warning and from valid unfavorable traffic; and
- states the separate natural, forced-route, post-selection, one-layer,
  high-batch, route-diagnostic, and cold schedules without adding cells.

The reviewer found no new drift in the corrected or surrounding text. The
depth-dependent operation/byte totals, U0--U5 algebra, rounding boundaries,
trust invariant, and implementation order also matched the frozen parent.

## Authorization boundary

This receipt accepts only the typed extraction of the frozen B2 graph. It does
not establish implementation correctness, CUDA correctness, physical advantage,
language capability, or production-model improvement. It authorizes no GPU
check, local run, rental, or physical measurement.

## Audited order amendment

After the FP64 oracle exposed that the pinned PyTorch/libdevice runtime was not
available in the current CPU environment, Section 6 reordered two reference
slices: the CPU FP64 oracle and trace checker now precede the target-shape
staged-BF16 reference in the pinned runtime. The parent requires all four
references before performance measurement but imposes no order among them.

The independent reviewer reconstructed the prior two-line block in memory and
obtained the exact earlier SHA-256 above, proving that the order swap and the
clarifying words `in the pinned runtime` were the only byte-level mutation.
The amended map SHA-256 is:

`8317e77188033d50014fe57a69bc77d6ebaac97fb29389df62a8e2b2355908c0`

Verdict on the amendment: **PASS**. It does not change a dependency, gate,
scientific claim, or GPU authorization boundary.
