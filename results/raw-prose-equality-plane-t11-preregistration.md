# Raw-prose equality plane T11 — minimal-isolation preregistration

Status: frozen before GPU execution  
Date: 2026-07-31

## Question

Did T10 lose natural-language quality because the digital plane inherently
competes with language modeling, or because T10 isolated far more of the model
than its three-block computation required?

T11 changes only the isolation mask.  The raw corpus, compiler, recovered
facts, four-token query protocol, model, initialization, attention/SwiGLU
graph, and direct/equality algebra remain fixed.

## Frozen construction

1. Special token rows are zero outside the program and frozen.  Natural token
   rows retain all 384 coordinates; no embedding coordinate is globally
   reserved.
2. Block 0 fixes only Q/K entries that read program coordinates in the first
   two heads, all V entries that read program coordinates, the 38 output
   coordinates actually carrying compiled values, and the FFN `up` entries
   required to make the unused branch zero on program-only inputs.
3. Blocks 1 and 2 make attention zero on program-only inputs by fixing only V's
   program-input columns.  Their FFNs isolate respectively 64 selector and 8
   product channels using a triangular zero in each inactive `up` branch.
4. After block 2, residual identity preserves exactly nine answer coordinates:
   the constant, four selected-value bits, and four product bits.  Later
   attention and FFN branches may read them but may not write them.
5. T10's output-row scale is reduced from 30 to 2.  The decision boundary is
   algebraically unchanged, while unused special rows exert less pressure on
   the natural-language softmax.

This mask is static and input-independent.  It adds no branch, parameter,
activation, KV state, serving FLOP, or precision change.

## Representation gates

All T10 exhaustive compiler, direct-query, equality-query, BF16-margin,
same-shape, same-parameter-count, and destructive perturb/enforce gates remain
mandatory.  In addition:

- the frozen-entry count must be lower than 25% of T10's 7,824,855 entries;
- no ordinary token row may be frozen merely because a coordinate belongs to
  the program plane;
- exactly nine coordinates may be protected after block 2;
- all 100% accuracy results must survive the smaller output scale.

Any representation error closes T11 before matched training.

## Matched-training admission

A representation pass admits exactly the T10 single-seed matched-training
protocol: identical raw strings, compiler-derived labels, controls,
initialization, Muon/AdamW settings, token presentations, checkpoints, and
resource accounting.

At steps 250, 500, and 1,000 the candidate must retain exact held-out direct
and equality accuracy with positive margins, and natural NLL must be no more
than 0.5% above `muon_labels_1x`.  The best twice-trained control must remain
below 90% on at least one held-out capability.  Passing only admits a sealed
three-seed run; it does not establish a smarter production LLM.

## Kill rule

If natural NLL exceeds the 0.5% gate at any checkpoint, minimal triangular
isolation has not repaired T10's tradeoff and this refinement is closed.  No
post-result threshold, scale, or mask change may be reported under T11.
