# Regenerated packed memory T5 — preregistration

Frozen before implementation or T5 measurement: 2026-07-30

## Predecessor and single change

T4 is closed at its BF16 quick gate.  Its packed decoder recovered every one
of 24,576 support signs, but decoded magnitudes ranged from 0.1416 to 1.6016
globally.  Dot-product softmax converted that amplitude spread into nonuniform
routing and destroyed the selected-bit sum.

T5 retains T4's 768 rules, eight nibbles per rule, 6,144 rule-data scalars,
model shape, unused vocabulary rows, support solver, natural streams, arms,
budgets, optimizer, and acceptance thresholds.  It adds exactly one conceptual
operation before attention: a shared symbol-regeneration block.

## Regeneration algebra

Raw decoded support signs occupy 32 transient coordinates.  A second shared
FFN writes 32 canonical coordinates using

`g(x) = -a + ReLU(x+a) - ReLU(x-a)`, with `a = 1/16`.

For every T4-decoded value, `|x| >= 0.1416 > 2a`; therefore the ideal map emits
exactly `+a` or `-a`.  The implementation uses BF16-stable
`SiLU(beta*x)/beta` hinges with `beta = 32`.  All three terms share the same
RMS and task gate, so their output has a rule-dependent positive common scale
but equal magnitude across all 32 coordinates of a rule.  Attention may vary
in temperature between rules; it may not vary selection weight within a rule.

The regeneration block uses at most 66 active SwiGLU channels and 32 additional
transient hidden coordinates.  Program state is temporally isolated through
five blocks (decode, regenerate, copy, select, count-decode); later blocks
protect only the result sign.  Model parameter count, vocabulary, forward
graph, dense matmul shapes, and inference FLOPs remain identical in all arms.

## Frozen gates

Before any full training, the sealed BF16 quick gate requires:

1. 100% raw decoded sign accuracy across all 24,576 support bits;
2. 100% regenerated canonical sign accuracy;
3. within each rule, maximum/minimum canonical absolute magnitude ratio no
   greater than 1.05;
4. at least 99% parity accuracy for every one of 768 rules and at least 99%
   protected-copy accuracy.

If the quick gate passes, the same two paired-seed full protocol and gates from
T4 run: every candidate rule at least 99% through steps 250/500/1,000, 2x
ordinary mean parity below 80%, copy at least 95%, language NLL within 0.5%,
terminal candidate improvement, finite training, maximum loss below 100, and
identical model shape/initialization.

Passing would establish a 12x rule-table density increase over T3 with an
explicit error-correcting algebraic boundary.  It remains a synthetic GF(2)
finding, not an automatic general rule-family learner.  Any BF16 quick failure
closes regenerated nibble memory without tuning `a`, `beta`, or attention
weights after measurement.
