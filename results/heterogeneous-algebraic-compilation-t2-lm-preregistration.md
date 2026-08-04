# Heterogeneous algebraic compilation — 37M LM coexistence preregistration

Frozen before result generation: 2026-07-30

## Purpose

The exact causal gate passed.  This screen measures the cost of placing that
capability inside one fixed-budget model that is simultaneously pretrained on
real language.  It is an admission test, not yet a breakthrough adjudication.

## Fixed model

- tied vocabulary/input-output embedding with 49,152 ordinary token IDs and
  four reserved algebraic IDs;
- hidden size 384, 10 causal blocks, six heads of width 64, SwiGLU width 1,024;
- learned absolute positions, context 128, no bias or normalization;
- residual output matrices use depth-scaled initialization;
- 36,619,776 deployed parameters in every arm;
- identical dense token embedding, position embedding, Q/K/V/O, gate/up/down,
  causal softmax, and tied output operation at inference.

The no-normalization decoder is intentionally the smallest architecture that
contains the already-proved causal construction.  Passing does not establish
transfer to RMSNorm/RoPE production LMs.

## Data and objective

- natural data: frozen document-disjoint FineWeb-Edu uint16 streams at
  `data/block-algebra-scratch-v4576/{train,validation}.uint16.bin`;
- natural batches: 16 sequences x 128 predictions;
- algebraic batches: 64 sequences from the T1b dense-mixing worlds, with 32
  observed bit tokens, a parity/copy query, and a two-token conditional output;
- every 20th post-prefix step is algebraic; all other steps are natural;
- optimizer: AdamW, LR `3e-4`, betas `(0.9, 0.95)`, no weight decay, gradient
  clip 1.0, BF16 forward and FP32 loss;
- paired model/world seeds: `(731, 947)`;
- natural validation and fresh algebraic validation are fixed per seed.

## Arms

1. `baseline_1x`: four gradient prefix batches plus 1,000 mixed steps;
2. `compiler_1x`: consumes the same 256-example prefix, solves and compiles,
   then runs the same 1,000-step stream;
3. `baseline_2x`: same prefix plus 2,000 mixed steps.

The compiler reserves nine of 384 hidden coordinates, two scalar attention
components in the first block, and at most 19 of 1,024 first-block FFN
channels.  Natural token output rows are fixed to zero in compiler coordinates.
Cross-lane entries are fixed to zero, while all language-language entries keep
their paired random initialization and train normally.  Later blocks preserve
the compiler coordinates by ordinary residual identity.  A training-time mask
restores fixed entries after optimizer steps; it is deleted at export.

All prefix solving, fixed-entry writes/restores, algebraic steps, natural steps,
evaluation, and optimizer state are charged.  Candidate and controls retain the
same deployed tensor shapes and inference operations.

## Checkpoints and frozen gates

Evaluate at mixed steps 250, 500, and 1,000; the 2x control is additionally
evaluated at 2,000.  Advance only if both paired seeds satisfy all conditions:

1. compiler parity and protected-copy accuracy are each at least `99%` at
   steps 250, 500, and 1,000;
2. `baseline_2x` parity accuracy remains below `80%` at step 2,000 while copy
   accuracy reaches at least `95%`;
3. compiler natural validation NLL is no more than `0.5%` above paired
   `baseline_1x` at steps 250, 500, and 1,000;
4. compiler terminal natural NLL is no worse than its own preceding checkpoint;
5. candidate support recovery is exact and all initial hashes match;
6. every arm stays finite with maximum pre-clip gradient norm below 100;
7. parameter count, tensor shapes, inference graph, and output vocabulary are
   identical after deleting training-only masks;
8. source, tests, preregistration, data files, and T1c predecessor are
   hash-bound.

Failure on language noninferiority closes fixed-coordinate reservation at this
size.  Passing advances to a standard RMSNorm/RoPE model and stronger optimizer
envelope; it still does not satisfy the multi-scale breakthrough contract.

