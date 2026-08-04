# Predictive residual coder T19d — decision

Status: **FAIL physical-width gate; finite coder retained**  
Date: 2026-07-31

## Decision

T19d does not admit an in-model decoder or QA training.  The exact top-256
finite coder round-trips the corpus and fits the aggregate write budget, but at
least one document exceeds the one-record hidden-width limit.  Do not rescue
the sealed layout with a larger top-k, changed frequency precision, overflow
channel, truncation, relaxed width, or a second record.

## Exact finite result

- 2,405/2,405 documents and 242,537 tokens round-tripped bit-exactly.
- Every integer frequency was positive and every CDF summed to 65,536.
- Uniform, skewed, randomized, and escape arithmetic-coder contracts passed.
- Top-256 escape rate after the first token was 23.0823%.
- Ideal length including the uniform first token was 1,799,509.77 bits.
- Actual finite length was 1,889,614 bits, 1.05007x ideal.
- Actual padded payload was 473,358 four-bit cells.
- Bitstream SHA-256:
  `5b53e752722d66ac96b47fcf32f8004afc5a1f93adcdf04179f448ef3515f7b3`.

The aggregate physical ledger passed:

| write | entries |
|---|---:|
| title codes | 132,800 |
| FFN address/gate | 81,770 |
| finite payload | 473,358 |
| zero end sentinels | 2,405 |
| **total** | **690,333** |

This leaves 53,401 entries under the 743,734 cap, above the required 50,000
decoder reserve.

## Failed gate

The largest bitstream used **361** four-bit cells.  The standard 384-wide
record has room for at most 350 after 32 address coordinates, one constant,
and one sentinel.  The preregistered maximum-width gate therefore fails.

This is not relabeled as a near pass.  Aggregate spare weights do not make an
oversized record readable in one hidden state.  An overflow record changes the
address/depth/read protocol and would require a new claim.

## Deeper computation boundary

Even if the 11-cell overflow were removed, the cached-CDF roundtrip is not a
served decoder.  At token `t`, the arithmetic interval depends on the writer's
distribution conditioned on decoded tokens `0..t-1`.  For an arbitrary causal
Transformer this is a genuine dependency chain, not a batched matrix layout.
Recomputing all 128 prefixes would add document-length inference work, which
violates the fixed served graph/cost target.

The T19 sequence now exposes a three-way boundary:

1. **Cheap random access:** lexical sketches fit and read cheaply, but discard
   typed semantic information.
2. **Semantic-looking static state:** generic self-latents are not identifiable
   or composable and scored at chance.
3. **Lossless semantic-free storage:** predictive residual coding preserves the
   raw prose, but exact access is sequential and one record overflows.

A successor must be query-addressable without full decompression.  It cannot
be another static pooled vector or ordinary entropy code.  The admissible
frontier is a jointly learned, reconstructable residual index whose query
operation is already one of the Transformer's fixed parallel primitives and
whose strongest same-information control is priced explicitly.

## Evidence

- Result SHA-256:
  `6195cab4edeefb09912bb273dada4309532cbc3a606ee977016d1acb507060e7`.
- Source SHA-256:
  `fd9b6c24a7bbebdb5604afea57fae78090c2e7475ddf5b822bdbe58fbefdd43f`.
- Preregistration SHA-256:
  `3d3018fe92c44f97ccf8ac44fc23738b696244f97e999d9c7a52d0a7f0c2075d`.
