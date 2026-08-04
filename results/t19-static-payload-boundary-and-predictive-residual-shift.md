# T19 static payload boundary and predictive-residual shift

Status: **T19a–T19c closed; exploratory capacity result retained**  
Date: 2026-07-31

## Closed static routes

The three corrected T19 screens isolate the failure:

| route | development result | conclusion |
|---|---:|---|
| deterministic title code, compiled 1x | 99.5192% routing | causal but only +1.4423 points over gradient 2x |
| 220d lexical CountSketch payload | 60.5769% QA | faithful storage, insufficient typed structure |
| 220d mean-pooled self-latent payload | 50.0000% QA | same-model latent is not a composable schema |

Do not tune or combine these static vectors.  Addressing is learnable, and
copying either a lexical summary or a generic hidden summary does not create
held-out reasoning.

## Why the failure is not a memory-capacity wall

An exploratory H100 audit used the sealed T12 raw-trained checkpoint as an
entropy model over each document's first 128 tokens.  Per-token ideal code
length was `-log2 p(token | prefix)`.  This is a reproducible development
measurement, not a preregistered or final result and not yet an implemented
finite-precision coder.

Measured over all 2,405 documents:

- total ideal predictive code: **1,762,027.92 bits**;
- mean per document: **732.65 bits**;
- median: **788.99 bits**;
- p90: **1,030.32 bits**;
- maximum: **1,387.96 bits**.

The retained physical plane has 529,100 per-document payload cells.  At the
already demonstrated robust 16-level / four-bit alphabet, it carries
2,116,400 bits.  Per document, the 384-wide hidden state leaves 351 coordinates
after 32 address coordinates and one constant.  The largest observed ideal
code requires `ceil(1387.96 / 4) = 347` payload coordinates.

Using variable-length allocation and one length cell per document:

| write | entries |
|---|---:|
| title codes | 132,800 |
| FFN address/gate cells | 81,770 |
| ideal-code payload cells, per-document rounded | 441,693 |
| zero end-sentinel cells | 2,405 |
| **total** | **658,668** |

This leaves **85,066** entries under the 743,734 cap for a shared decoder and
isolation.  Every document fits in at most 348 payload-plus-sentinel coordinates,
below the 351-coordinate per-record limit.

The information is concentrated: the 32 most surprising tokens account for
63.41% of document surprisal on average; the median document needs 28 tokens
to account for half its surprisal.

## What this changes

The compiler need not understand or summarize semantics.  The same model can
identify and entropy-code only its predictive residual—the raw information it
does not already carry.  That representation is lossless in principle and
avoids both failed static-vector assumptions:

- it does not discard typed facts like CountSketch;
- it does not assume arbitrary hidden coordinates form a stable schema.

The remaining problem is computation, not storage.  A conventional arithmetic
code is sequential and nonlocal.  Installing its bits in weights is useless
unless an ordinary fixed-depth Transformer can consume the residuals without
adding tokens, layers, KV state, or serving work.

## Next admissible gate

T19d must be an algebra/runtime gate before QA training:

1. implement a deterministic finite-precision predictive coder and prove exact
   roundtrip for all 2,405 raw documents;
2. keep the complete address, payload, length, shared-decoder, and isolation
   writes within 743,734 entries;
3. expose a bounded parallel residual interpreter using only the existing
   attention/SwiGLU graph;
4. reject the route if decoding requires document-length sequential depth,
   added inference tokens, external state, or a stronger served model;
5. only after exact parallel readout exists, compare knowledge/reasoning with
   controls given the same residual token examples and total training work.

This is an implementation rethink grounded in the failures: preserve raw
innovations exactly first, then make the existing model interpret them.  It is
not another post-hoc semantic vector.
