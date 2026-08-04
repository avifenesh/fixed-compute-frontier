# Norm-matched folded direction: frozen 10M screen

## Hypothesis

The earlier rank-16 folded-metric optimizer selected an ordinary dense
checkpoint with 0.0507% lower terminal NLL than a first-step LR-matched AdamW
control, at zero serving overhead.  Its effective per-matrix update norm was
not fixed, so a changing learning-rate multiplier remained a fatal
explanation.

This successor removes that degree of freedom.  For every targeted matrix and
every optimizer step, let `u` be its ordinary AdamW update and `v` the
training-only rank-16 factor update.  Form

`v_perp = v - <v,u> u / ||u||^2`

and apply

`u_candidate = ||u|| (u + v_perp) / ||u + v_perp||`.

Thus `||u_candidate|| = ||u||` exactly in real arithmetic, while the factor can
only rotate the update.  Weight decay is already part of `u`.  Factors remain
optimizer state: the model, checkpoint, and served graph always contain only
the ordinary dense matrix.

## Controls and frozen protocol

- `fixed_direction`: deterministic orthonormal `A`, trained `B`;
- `learned_direction`: trained `A` and `B`;
- frozen ordinary AdamW reference: seed-223 baseline from the prior
  folded-metric screen.

Both new arms use exactly the prior seed, initialization, data batches,
schedule, target matrices, rank, factor seed, and optimizer hyperparameters:
12-layer `D=384,M=1024`, sequence 512, microbatch 32, accumulation 2, AdamW
`3e-4`, 100-step warmup, weight decay 0.1, clip 1.0, and 305 steps
(9,994,240 prediction tokens).  Validation uses the same 128 batches.

The implementation records the maximum per-matrix relative norm mismatch over
all 305 steps.  It must be at most `2e-5`; the median factor/base dot after
projection must be numerically zero.  All endpoints must reload into the plain
factor-free model with bitwise-equal BF16 logits.

## Decision

The direction advances to an untouched 50M replication only if the learned arm:

1. beats ordinary AdamW by at least 0.10% relative NLL;
2. beats the fixed-direction control by at least 0.05%;
3. has wholly favorable paired 95% intervals for both comparisons;
4. passes exact-update-norm, finiteness, integrity, and plain-export gates.

This is deliberately stricter than merely reproducing the earlier 0.0507%
edge.  Failure closes this rank-16 folded rotation, without negating the broader
training-cost-for-serving-quality exchange.

