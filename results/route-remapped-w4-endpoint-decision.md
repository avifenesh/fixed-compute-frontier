# Route-remapped W4 fatal endpoint — decision

Decision: **reject this no-recovery shared codebook substrate; do not advance
its route decoder branch.**

The hash-bound SmolLM2-135M evaluation quantized every gate/up/down FFN matrix
to symmetric signed W4 with one BF16 scale per output-row group of at most 128
input weights.  The static endpoint raised validation NLL from
`2.8295718897` to `3.0682800580`: an `8.43619%` degradation, 42.18 times the
frozen `0.2%` limit.  The paired 95% interval for decoded minus baseline NLL
was `[0.2364039193, 0.2410124172]`, also wholly beyond the allowed boundary.

The failure is not an accounting or execution failure.  All 90 matrix keys,
shapes, group boundaries, 64-value gate/up tails, codes, BF16 scales, runtime
bindings, and 256 evaluation records passed.  The exact resident ledger was
79,626,240 codes plus 668,160 BF16 scales, or 329,195,520 bits total:
`4.134259` resident bits per FFN weight.

This closes only the exact static symmetric group-128 W4 codebook as a
quality-preserving base that needs no recovery.  It does not close learned or
data-aware W4, smaller groups, mixed precision, outlier paths, fine-tuning, or
routing in general.  Any such recovery must be introduced as a new costed
hypothesis; it cannot be counted as validation of this failed substrate.

Evidence hashes:

- result: `0ab0855d244d617ea1aa0f4241a35bbd4546883f9ed3382328daf6d13023476a`
- log: `bd281b28bc264e7cfecc8094527b5bd7c6352975ee59dc9b572001dab4f25a38`
- source: `689e1eb851b4685a5f5bc1ea07f05ea4ce84495c3d63c69a8c45564902cedc6c`
- preregistration: `7d7cb030cdd8ab4038f6e163dd02cd0cd4eac92ec07668dc23490f3131bf722f`
- integrity manifest: `31055a4542b9a43e36fdf4baa4190e9c5d34d74180d5a999f158e30c6c245688`
