# Heterogeneous algebraic compilation — 37M LM coexistence v3 preregistration

Frozen before any v3 arm: 2026-07-30

## Predecessor

V2's BF16 compiler passed, but its constant-LR `baseline_2x` became non-finite
at step 1,500.  V2 emitted no result and is closed in
`heterogeneous-algebraic-compilation-t2-lm-v2-decision.md`.

## Only protocol change

All architecture, initialization, data, two paired seeds, arm definitions,
compiler masks/weights, BF16 arithmetic, objectives, batch sizes, 1x/2x step
budgets, evaluation checkpoints, ledgers, and frozen acceptance gates from the
v1 and v2 preregistrations remain unchanged.

The sole v3 change is a common AdamW learning-rate schedule:

- peak `3e-4`;
- 50-step linear warmup;
- cosine decay over the fixed 2,000-step schedule horizon;
- terminal learning rate 10% of peak;
- 1x arms stop at step 1,000 on the same schedule used by the 2x arm;
- the four control prefix updates use schedule steps 1 through 4;
- candidate and controls use identical mixed-step learning rates.

The finite-gradient gate remains maximum pre-clip norm below 100 and maximum
loss below 100 in every arm.  The language noninferiority tolerance remains
0.5%; the capability gates remain 99% candidate and below 80% 2x control.  No
failed or partial v2 measurement may be used in the v3 decision.

