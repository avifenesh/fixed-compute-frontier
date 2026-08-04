# Shared algebraic interpreter T3 — preregistration

Frozen before implementation or measurement: 2026-07-30

## Question

Can one fixed in-place neural interpreter store and execute 256 independently
discovered rules while paying a rule-description cost rather than one circuit
per rule, without changing the exported model shape or inference graph?

This is a capacity-density test.  It is not a refinement of the closed T2
no-normalization protocol.

## Model and data

- One tied-embedding causal language model with vocabulary 49,152, hidden width
  384, ten residual blocks, six 64-wide attention heads, SwiGLU width 1,024,
  pre-RMSNorm, causal masking, and RoPE on the four language heads.  The two
  compiler-capable heads are unrotated in every arm so their query-key algebra
  is position independent.
- All arms have identical parameter shapes, parameter count, sequence graph,
  attention/FFN FLOPs, and initial random state before compilation.
- Natural training and validation use the sealed FineWeb-Edu uint16 streams
  from T2 with their existing hashes.
- Algorithm tokens are selected deterministically from vocabulary IDs with zero
  occurrences in both sealed natural streams.  No vocabulary row is added.
- There are 256 rule IDs.  Each rule is an independently sampled 16-of-32
  parity support.  Inputs contain a rule token, 32 position-specific binary
  tokens, and a parity-or-protected-copy query.
- The protected route copies the first observed bit and does not depend on the
  hidden rule.

## Candidate compiler

For every rule, a sealed 64-example prefix is solved over GF(2).  The compiler
must recover all 256 supports exactly or the arm fails.

The recovered supports are written as 256 sign vectors of length 32 into the
existing rule-token embedding rows: exactly 8,192 scalar rule-description
entries.  A single shared interpreter occupies at most 41 hidden coordinates,
two existing attention heads, and 64 existing SwiGLU channels per block.  It:

1. copies the selected rule description to the query;
2. uses query-key scores to select the 16 supported input positions;
3. reduces their signs to the integer sum in `{-16,-14,...,16}`;
4. uses one shared BF16-stable piecewise SwiGLU decoder to emit parity;
5. uses a separately gated path for protected copy.

Natural tokens have zero activation in the compiled coordinates.  Compiler
entries and the cross-partition zeros needed to preserve the circuit are fixed
during candidate training.  The exported checkpoint contains no solver,
sidecar, adapter, extra expert, retrieval store, or added parameter.  The
ordinary dense matrices contain the merged circuit.

## Arms and budgets

Two paired model seeds are used.  Every seed runs:

- `baseline_1x`: ordinary gradient training on the complete 16,384-example
  charged prefix followed by 1,000 mixed steps;
- `compiler_1x`: exact support recovery/compilation followed by the same 1,000
  mixed steps;
- `baseline_2x`: the same charged-prefix gradient training followed by 2,000
  mixed steps.

Mixed training uses one algorithm batch every 20 steps and natural language on
the other 19.  Natural and algorithmic batches, optimizer, schedule, clipping,
precision, and checkpoint examples are paired within a seed.  AdamW uses
betas `(0.9, 0.95)`, zero weight decay, 50-step warmup, peak learning rate
`3e-4`, cosine decay over a fixed 2,000-step horizon to 10% of peak, BF16
forward arithmetic, FP32 losses, and global clipping at 1.

## Frozen gates

Every gate must pass in both paired seeds:

1. Identical pre-compilation initialization hashes and exact recovery of all
   256 supports.
2. At steps 250, 500, and 1,000, the candidate has at least 99% parity accuracy
   for every rule and at least 99% protected-copy accuracy.
3. At step 2,000, the ordinary control has mean parity accuracy below 80% while
   retaining at least 95% protected-copy accuracy.
4. Candidate natural validation NLL is no more than 0.5% above `baseline_1x`
   at steps 250, 500, and 1,000, and candidate NLL improves from 500 to 1,000.
5. Every loss is finite, maximum pre-clip gradient norm is below 100, and
   maximum training loss is below 100 in every arm.
6. Candidate rule-description entries equal `256 * 32 = 8,192`; interpreter
   hidden width does not depend on rule count; model parameter count and graph
   are identical across arms.

## Interpretation boundary

Passing establishes a large qualitative optimization-and-density separation:
one fixed same-size model holds 256 exact rules that twice the ordinary
gradient budget does not acquire, while preserving language quality.  It does
not by itself establish automatic rule-family discovery, natural-language
reasoning gains, or novelty over all neural compilation work.  Failure on
language, stability, or BF16 execution closes this particular partitioned
shared-interpreter construction rather than inviting threshold tuning.
