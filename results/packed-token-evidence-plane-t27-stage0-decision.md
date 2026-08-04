# Packed token evidence plane T27 — Stage 0 decision

Status: **PASS PRIMITIVE; ADMIT ONE DECODED-PREFIX ORACLE**  
Date: 2026-07-31

## Decision

The finite pack/read primitive is internally valid.  Freeze it and proceed to
one stronger-information oracle over the exactly decoded 55-token prefixes.

Do not implement the Transformer placement and do not train a language model
yet.  Stage 0 proves only that the proposed digital object has a fixed meaning
and that its logical content-read circuit fits the nominal widths of three
ordinary SwiGLUs.

## Evidence

All frozen gates passed:

- every one of 65,536 token IDs survived radix-16 pack/unpack;
- 55 tokens occupied exactly 220 four-bit cells / 880 logical bits;
- all 16 odd amplitude levels survived the simulated BF16 round trip exactly;
- across all 31 possible nibble-level differences, the paired-SiLU square
  identity had maximum absolute error `1.14e-13`;
- zero distance remained zero and the minimum nonzero squared distance was
  `4.0` within floating error;
- all 1,000 frozen random tapes contained a unique sampled two-token key and
  returned the exact successor;
- the repeated-key adversary returned both positions and `ambiguous`;
- all 16-bit vocabulary codes round-tripped, with correct score 16, nearest
  one-bit-neighbor score 14, and margin two;
- width ceilings were 880 for pair matching, 220 for selection, and 192 for
  token decoding, each within width 1,024.

The targeted test suite passed `22/22` tests locally.

## What the pass establishes

T27 no longer depends on an arbitrary record coordinate convention.  Every
cell denotes one radix digit of one tokenizer token, pair matching has a
constructive squared-distance identity, ambiguity is explicit, and the output
code has a fixed decoding margin.

This repairs the storage/read identifiability failure that invalidated generic
continuous records.  It does not show that the first 55 tokens contain enough
evidence, that natural questions form usable lookup keys, or that the physical
model can preserve the numerical margin after RMSNorm and residual mixing.

## Next sole gate

Freeze a decoded-prefix oracle before execution.  It must:

1. use exactly the first 55 frozen tokenizer IDs per raw document;
2. decode them perfectly outside the candidate model;
3. compare correct, zero/omitted, and shuffled document prefixes;
4. use a strong fixed reader only as an information upper bound;
5. require a breakthrough-sized absolute result and causal correct-versus-
   shuffled gain before physical work.

Failure closes fixed-prefix evidence.  Passing admits only a same-graph
physical read microbenchmark.

## Integrity

- preregistration SHA-256:
  `771646b606589317f1817cd676b6b7e2b4686b33d36475ed41e93e6a920c80a8`
- source SHA-256:
  `162cc9be2e5aaa4b98f543fe27ba746113c6c70f58e4d5bfc7452281eb03d1b7`
- tests SHA-256:
  `0cc7e108e324c95b77e29443b09dd9233d05b3e9db247eefce98f7400d3cb695`
