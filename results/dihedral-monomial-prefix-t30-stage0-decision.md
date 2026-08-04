# Dihedral-monomial prefix operators T30 — Stage-0 decision

Status: **EXACT PRIMITIVE PASS; FIRST LEARNABILITY SCREEN WITHDRAWN PRE-RUN**  
Date: 2026-07-31

## Verdict

The exact `D_109` controller, monomial-affine composition, arbitrary-prefix
title substitution, independent black-box reconstruction, lazy state frame,
four-bit codec bounds, and fixed-resource ledger all passed their frozen CPU
gates.

This is a strict algebraic improvement over T29's one-layer homogeneous
transition: T30 realizes noncommuting rotations/reflections while T29's
homogeneous diagonal maps commute.  It is not yet evidence that training uses
the extra state routing, that raw prose becomes answerable knowledge, or that a
GPU kernel has unchanged latency.

## Authoritative result

- unit tests: 8/8 passed;
- group: 218 elements, 47,524 ordered pairs and 10,360,232 ordered triples,
  with zero code, inverse, action, or associativity failures;
- exact folds: 1,365 sequences and 5,460 state evaluations, zero mismatches;
- black-box reconstruction: 9,265 basis probes, zero mismatch;
- separation: all 218 actions distinct, 12 ordered token pairs with
  noncommuting homogeneous actions, and two-marker difference `2.0`;
- title substitution: 148,176 single-title and 200,000 two-title cases, zero
  mismatch;
- lazy state: 2,778,192 multiply/add/read/write operations with zero state or
  frame mismatch and no physical state permutation copy;
- codec: 114,660 one-trigger and 37,044 two-trigger executions, zero clipping
  and zero bound violations;
- ledger: 109 BF16 values plus one uint16 frame = 220 bytes; 220 four-bit cells
  per document; 743,670 entries under the 743,734 cap, leaving 64.

The maximum one-trigger quantization error was `0.29470486111111116` under a
maximum bound of `0.307525634765625`.  The maximum two-trigger error was
`0.23046875` under a maximum bound of `0.31901041666666663`.  These are valid
bounds, not semantic noninferiority evidence.

## Integrity and execution

- paper SHA-256:
  `79093931f90ef2afc4dc46b4924d7c6248e56eccee231dcca6bf913e781564a3`
- preregistration SHA-256:
  `e8daed5f6c0f41fa8c8123086249a113c73132fbcd91b97ea228267266177297`
- implementation SHA-256:
  `5432c7d564db6ab6f71e5164ec3c4d772180b6978cbee9f760e9b24b6311c382`
- tests SHA-256:
  `01320ee5ceb7168cc1e349071baa3f573f3bf3839fcb2913ab59b9da4af12ef4`

The unchanged artifacts ran on the retained research box with CUDA hidden in
the isolated CPU PyTorch environment.  No GPU kernel or language model ran.

## Prior-art and admission boundary

Permutation-diagonal SSMs and their finite-state advantage are prior art.
Stage 0 therefore establishes a correct compact specialization, not broad
architectural novelty.

The first newly reasoned screen was later
[withdrawn before execution](dihedral-monomial-prefix-t30-learnability-pre-run-audit.md):
its document arm fixed the diagonal scales to one and used rotations only, so
it did not test the named T30-versus-T29 separation.  The corrected arbitrary-
incoming-state theorem is retained as a deterministic operator microbench,
but that property substantially overlaps PD-SSM prior art.  No new GPU run is
admitted until a natural-information argument explains why this compact
specialization should improve raw-prose capability against a full matched
diagonal-affine or hybrid control.  No natural-language, quantized-model,
physical-kernel, or production claim is admitted.
