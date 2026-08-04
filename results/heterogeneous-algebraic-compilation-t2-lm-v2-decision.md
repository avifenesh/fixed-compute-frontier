# Heterogeneous algebraic compilation — 37M LM coexistence v2 decision

Status: closed on non-finite control before result emission  
Date: 2026-07-30

The BF16 product-tree compiler passed both sealed quick checks at 100% parity
and protected-copy accuracy, and all six implementation tests passed.  The full
run did not complete: `baseline_2x` produced a non-finite loss at mixed step
1,500 under the frozen constant `3e-4` learning rate.  The script aborted before
writing a result, so no candidate/control language comparison is admissible.

This closes the exact v2 constant-learning-rate protocol.  It does not diagnose
algebraic compilation because the failing arm contained no compiler.  The
unstable no-normalization baseline is not a valid frontier control.

V3 may change only the common optimizer schedule to 50-step linear warmup and
cosine decay from `3e-4` to 10% of peak over 2,000 steps.  Architecture,
initialization, data, seeds, arms, compiler, accounting, checkpoints, and all
quality gates remain unchanged.

The v2 run emitted this terminal error:

```text
RuntimeError: nonfinite baseline_2x loss at step 1500
```

