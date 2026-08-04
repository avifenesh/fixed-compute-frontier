# Radial-trust Cayley B2 CPU-specification audit receipt

Date: 2026-08-01  
Verdict: **PASS — CPU specification slice only**

## Frozen inputs

- Physical preregistration:
  `results/radial-trust-cayley-b2-physical-preregistration.md`
- Physical preregistration SHA-256:
  `31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`
- CPU specification:
  `experiments/radial_trust_cayley_b2_spec.py`
- CPU specification SHA-256:
  `feca8452423f6d07ef5e66079dda39713ecc1d839bcbc7698e0cd6bd69700411`
- CPU tests:
  `tests/test_radial_trust_cayley_b2_spec.py`
- CPU tests SHA-256:
  `9134121734b8c4492d48f50e63a273cbdc7b9b4a57be478a97729a0b84401233`

## Independent review

Reviewer: `/root/tesr_claim_review`

The final review independently accepted that:

- `generate_frozen_trace(family, *, tuning=False)` exposes no arbitrary shape
  or seed override, and its tests cover all four exact evaluation and tuning
  contracts;
- the complete timing, counter, energy, and memory schedule permutations are
  protected by full serialized SHA-256 goldens;
- every one of the 36,864 raw descriptor `uint32` words is independently
  formula-derived, packed as `partner | pair << 12 | sign << 23`, and compared;
- the candidate payload is one C-order `[node, bit, 3, coordinate]` little-endian
  `uint16` allocation; and
- candidate RNG order and the small trace artifacts remain independently
  reconstructed and hash-guarded.

The reviewer reran the focused suite and observed:

```text
12 passed in 0.10s
```

A same-turn primary rerun observed:

```text
12 passed in 0.10s
```

## Authorization boundary

This receipt closes the CPU-specification audit gate only. It does not establish
CUDA correctness, kernel equivalence, physical efficiency, language capability,
or a production-model improvement. It does not authorize a GPU check, local GPU
run, rental, or B2 physical gate. Those require the remaining implementation
slices and independent code audit under the frozen preregistration.
