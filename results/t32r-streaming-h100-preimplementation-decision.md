# T32R streaming H100 gate — preimplementation decision

Status: **WITHDRAWN BEFORE IMPLEMENTATION OR TIMING**  
Date: 2026-07-31

## Decision

Do not implement or execute the frozen `BF16 radix table + width-940 FFN`
H100 benchmark.

The proof-first audit found two upstream fatal issues:

1. the physical representation is byte-dominated by direct `uint16` token
   storage, which funds a width-994 candidate under the same resident-byte
   ledger; and
2. the current title-routed, rank-32 path has no pre-run construction or
   microbench connecting raw-only training to the natural score margins needed
   by its conditional attention theorem.

The first issue makes the frozen implementation the wrong physical candidate.
The second means a timing pass could not advance the smarter-model claim.

## Integrity

At withdrawal:

- `experiments/t32r_streaming_h100.py` did not exist;
- `tests/test_t32r_streaming_h100.py` did not exist;
- `results/t32r-streaming-h100.json` did not exist;
- no timing cell had been observed;
- the retained H100 had not been used for this gate.

The original preregistration and positional-residency erratum remain unchanged
as evidence of the superseded construction.

## CPU proof-regression result

Five tests passed in 0.02 seconds.  They verify:

- the Fano information lower-bound values;
- the exact direct-typed-plane resident ledger;
- strict record-byte improvement over the BF16 table;
- ten-token arithmetic break-even at width 994;
- a finite-set counterexample to the overbroad claim that low ambient rank
  necessarily aliases token identities.

No GPU, corpus, model, or training was accessed.  This result proves neither
latency nor learnability.

## Artifact hashes at withdrawal

- proof-regression source:
  `c71fb66f768fb6f6cfef55820f0e134d0bd3a297bea1c8865256f7b427ff18d7`
- proof-regression tests:
  `5297c7b296b2b78ef9b4421920a61e086bc4c819d0c4a1f30018914d01a21692`
- capability audit:
  `64e38f9d7238dcbab013ec51f8f05d15c77f11047f9ca3e05bb1ea3d4ff1fc5d`
- methodology v2:
  `2ab92b0c14d322eddb17f4d4c8ca8b42a3d98978c8b183f88b34be426a4c1550`
- superseded H100 preregistration:
  `52dfcd9f0587391cb601809941271c694136223e6770f01b81bef3f880abf402`
- preregistration residency erratum:
  `9a97b7b6a42b6e6e2c6e7dbe31ef84b6595b8d6ba6a7ed5ac68eb78e577d1553`

## Retained next question

The active research object is not a new kernel.  It is a typed digital plane
whose literal-title matcher is retained only as a raw-only candidate generator,
whose router types grammatical argument mentions, whose extractor goes beyond
record identity, and whose exact bottleneck beats the strongest matched
conditional-memory control before physical integration.  Alias and implicit
retrieval remain a separate harder scope.
