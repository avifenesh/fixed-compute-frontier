# Packed token evidence plane T27 — Stage 0 preregistration

Status: **FROZEN BEFORE EXECUTION**  
Date: 2026-07-31

## Question

Does the exact logical primitive claimed by
`packed-token-evidence-plane-t27-algebra.md` work over its full finite token and
nibble domains, including BF16 storage and the stated per-layer SwiGLU width
ceilings?

This Stage 0 uses no corpus, question, answer, model checkpoint, optimizer, or
GPU.  It cannot establish natural-language sufficiency, physical Transformer
integration, language quality, or a smarter model.

## Frozen domain

- Vocabulary domain: every integer token ID in `[0, 65,535]`.
- Tape: 55 token IDs, four radix-16 digits per token, exactly 220 cells.
- Amplitude map: `a(n) = 2*n - 15` for `n in [0,15]`.
- Virtual positions 55 and 56 are padding token zero; they are not stored.
- A two-token key may start at any stored position `j in [0,54]`; reads return
  the token at `j+2`, including virtual padding near the end.
- Random seed: 27,001.
- Random tapes: 1,000 independently sampled 55-token tapes.

## Frozen reference functions

The source must expose:

```
token_to_nibbles(token_id)
nibbles_to_token(nibbles)
pack_tokens(tokens)
unpack_tokens(cells)
amplitude(nibble)
amplitude_to_nibble(value)
bf16_roundtrip_float32(values)
silu_square_pair(d)
match_positions(tape, query_pair)
read_unique_successor(tape, query_pair)
token_binary_code(token_id)
binary_code_to_token(code)
channel_ledger()
run_stage0()
```

The reference reader is discrete.  A repeated query pair must return an
explicit ambiguity instead of silently choosing one occurrence.

## Mandatory gates

All gates are required:

1. Every token ID in `[0,65,535]` survives token-to-nibbles-to-token exactly.
2. Packing and unpacking every deterministic boundary tape is exact; every
   emitted cell is in `[0,15]` and the tape has exactly 220 cells.
3. All 16 odd amplitude levels survive a round-to-nearest-even BF16 round trip
   and decode to the same nibble.
4. For every amplitude difference induced by two nibble levels,

   `d*SiLU(d) + (-d)*SiLU(-d) = d^2`

   within absolute error `1e-10` in float64.  Zero distance is exactly zero and
   every nonzero nibble mismatch has squared distance at least four.
5. For each of 1,000 random tapes, select one two-token pair that occurs
   exactly once and verify that the returned successor equals the tape/virtual
   padding token at `j+2`.  If a sampled tape has no unique pair, the run fails;
   no retry or regenerated tape is allowed.
6. A frozen adversarial tape containing a repeated two-token pair reports both
   positions and returns `ambiguous`, never a value.
7. Every 16-bit token code round-trips exactly.  The correct bipolar dot score
   is 16, and every frozen one-bit neighbor scores 14, establishing margin two.
8. The channel ledger is exactly:
   - two-token squared-distance upper bound: 880;
   - unique-successor selection upper bound: 220;
   - four-nibble token decoder upper bound: 192;
   and each is at most the existing width 1,024.
9. The result records the preregistration, source, and test hashes and reports
   every gate independently.

## Kill conditions

Any failed gate closes this exact radix-16 packing/read identity.  Do not
change tape length, vocabulary size, amplitude levels, random seed, ambiguity
rule, numerical tolerance, activation identity, or channel accounting after a
failure.

A pass admits only a separately frozen decoded-prefix natural-information
oracle.  It does not admit physical model code or H100 training.
