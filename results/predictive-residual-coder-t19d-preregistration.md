# Predictive residual coder T19d — finite coder preregistration

Status: **frozen before finite-code result**  
Date: 2026-07-31

## Question

Can the same raw-trained model losslessly encode every document's first 128
tokens into robust four-bit payload cells while leaving enough of the existing
write budget for a shared in-model decoder?

This gate implements and round-trips the residual code.  It does not claim
that the ordinary Transformer can decode or reason over it at serving.

## Frozen inputs

- T12 raw-only writer checkpoint and SHA-256:
  `a2900585a6e9afe4a9fba4f55afca30df5bd79c335d646cda5b9e940b72d693b`.
- Candidate corpus SHA-256:
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`.
- Frozen SmolLM2 tokenizer/revision from T12.
- For each document, encode the first 128 tokens of
  `title + "\n" + text` without special tokens.

The compiler receives no question, answer, support label, ontology, external
model, or semantic feature.

## Frozen finite code

1. Encode the first token uniformly over the 49,152-token vocabulary.
2. For every later token, run the frozen writer in BF16 on the true prefix.
3. Select exactly the top 256 logits.  Their probabilities and one escape mass
   are quantized to positive integer frequencies summing to 65,536: assign one
   count to every symbol, floor the remaining proportional masses, then assign
   leftover counts by descending fractional remainder with symbol index as the
   tie break.
4. Arithmetic-code the selected rank, or the escape symbol followed by the
   original token uniformly over the full vocabulary.
5. Use a 32-bit integer arithmetic coder with standard E1/E2/E3
   renormalization and one final termination bit.

Encoding and decoding must use the same frozen integer CDFs.  T19d may cache
those CDFs only to test the finite coder; cached distributions are forbidden
from the deployed-resource ledger and do not solve the later in-model decoder.

## Physical accounting

Each emitted bitstream is padded only to a four-bit cell boundary.  Its nibbles
map to the 16 nonzero odd amplitudes already validated by the digital plane;
one following zero-amplitude cell is an end sentinel with margin one from the
nearest payload level.  Cells after the sentinel are ignored and need not be
written.  The complete write count is:

`132,800 title cells + 81,770 address cells + payload cells + 2,405 sentinel cells`.

The hard cap is 743,734.  This gate additionally requires at most 693,734
total writes, reserving at least 50,000 entries for a future shared decoder and
isolation program.  Each document may use at most 350 payload cells, so its 32
address coordinates, one constant, one sentinel, and payload fit within hidden
width 384.

## Gates

All gates are mandatory:

1. Frozen input/checkpoint metadata and raw-only compiler interface match.
2. Arithmetic-coder unit contracts pass for uniform, skewed, escape, and
   randomized symbol streams.
3. Every one of 2,405 document token sequences round-trips bit-exactly with no
   retry, and all decoded token IDs are in range.
4. Integer frequency tables are positive and sum exactly to 65,536 at every
   step.
5. Maximum payload length is at most 350 four-bit cells.
6. Complete writes are at most 693,734, leaving at least 50,000 decoder entries
   under the 743,734 cap.
7. Code length is reported against ideal writer surprisal, including escape
   literals, padding, and arithmetic termination; no idealized length is used
   in the physical gate.

Failure closes this exact top-256 finite residual code.  Passing admits only a
fixed-depth parallel decoder theorem/operator gate.  It does not admit QA
training or a smarter-model claim.
