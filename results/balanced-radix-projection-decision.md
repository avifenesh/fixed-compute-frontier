# Balanced-radix projection multiplexing — exact algebra, closed executor

Status: **closed before language training**  
Date: 2026-07-28

## Result

Two independent binary-weight projections sharing a ternary activation can be
encoded exactly into one signed-INT8 dot-product stream.  For each K=32 tile,

\[
P = W_0 + 64W_1,\qquad S=A P=S_0+64S_1.
\]

Both logical dots have the parity of the number of nonzero activations, so
nearest-even decoding of `S/64` resolves the only touching residues at
`+/-32`.  Exhaustive digit tests and randomized K=32,64,384 tests passed.

The construction does **not** scale through the ordinary long-K accumulator.
Every K=32 partial must be decoded before the next partial is added; otherwise
the low digit carries into the high digit and the result is ambiguous.

## H100 kill test

The real-K Triton kernel used K=4096, radix-64 decoding after every K=32 dot,
and separate low/high accumulators.  It was bit-exact in every cell but slower
than a fused dual-INT8 kernel that shared activation loads:

| Rows | Packed median | Fused dual median | Packed speedup |
|---:|---:|---:|---:|
| 16 | 0.054880 ms | 0.052704 ms | 0.9603x |
| 128 | 0.060480 ms | 0.057216 ms | 0.9460x |
| 512 | 0.194240 ms | 0.184992 ms | 0.9524x |

This is already a fail against a byte-expanded INT8 baseline.  Native INT4
and binary-popcount representations are stronger controls: they retain the
long reduction, store the two binary streams in no more space, and avoid the
per-K32 radix extraction.  A learning screen therefore cannot rescue the
served-resource claim.

Evidence:

- `experiments/balanced_radix_projection.py`
- `tests/test_balanced_radix_projection.py` (11 passed on H100)
- `experiments/balanced_radix_projection_h100.py`
- `tests/test_balanced_radix_projection_h100.py` (passed on H100)
- `results/balanced-radix-projection-h100-development.json`, SHA-256
  `51871216d38aa84cf516d1df91a11744749fc6cea042430b197cb95b9147eed9`

## Retained principle

Representation slack can encode more logical projections, but an execution
codec is useful only if decoding occurs outside the critical long reduction.
Do not interrupt a production matmul mainloop to recover extra digits.  Search
instead for a second algebra over the same operands that can run alongside the
ordinary reduction, or for native low-bit conditional capacity whose selected
expert still uses an unmodified long-K instruction.
