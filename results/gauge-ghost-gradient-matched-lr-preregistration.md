# Gauge Ghost Gradient matched learning-rate frontier

## Question

Does the backward-only TVE virtual Jacobian beat an ordinary Transformer after
both arms receive the same learning-rate search, or did the original `3e-4`
comparison only observe a displaced optimizer optimum?

This is a final development-seed falsifier.  It uses seed `21017` only and does
not consume any of the five sealed replication seeds.

## Frozen initial frontier

Use the exact mechanism-v2 initialization, deterministic math-SDPA backend,
data order, AdamW schedule, 305 steps, 9,994,240 prediction tokens, evaluation
batches, clipping, weight decay, and model geometry.  Compare canonical and
backward-only TVE at the log-spaced grid

```text
1.5e-4,
3e-4/sqrt(2),
3e-4,
3e-4*sqrt(2),
6e-4,
6e-4*sqrt(2),
1.2e-3.
```

Reuse only hash-bound valid arms: both arms at `3e-4`, and canonical arms at
the other four points through `6e-4`.  Train six new ghost arms and two new
canonical arms.  Arm order is interleaved at the two new upper points, with the
remaining ghost-only points filled afterward.  Quality is deterministic; the
ordering only limits thermal/time confounds in throughput diagnostics.

Every new arm must reproduce the frozen initial state hash and all 64 initial
losses exactly.  It must record exactly 305 finite step losses, exactly
9,994,240 training prediction tokens, and 64 finite validation batches / 1,048,576
prediction tokens at steps 0, 61, 152, and 305.  Parameters, buffers, optimizer
state, metadata, and ordinary inference semantics must remain structurally
matched.  Exported ghost attention tensors are validation artifacts, not
served state.

The eight-arm run uses an ordered, hash-bound completion journal.  A completed
prefix may resume only after every stored arm and export is revalidated.  If a
crash occurs after the helper writes a ghost export but before the parent
payload checkpoint, the bound and fully valid orphan is moved to a
hash-recorded quarantine and that arm is retrained; unbound or malformed
artifacts fail closed.

## Frozen decision

Let `Lc` and `Lg` be the minimum terminal NLLs over the complete canonical and
ghost grids.

- If either minimum occurs at either grid endpoint, return
  `endpoint_extension_required`; extend only that side before interpretation.
- If `Lg / Lc - 1 >= 0.0005`, return
  `ghost_fails_matched_lr_frontier`.
- If `Lc / Lg - 1 >= 0.0005` and the paired 64-batch ghost-minus-canonical
  normal-approximation 95% interval has upper bound below zero, return
  `ghost_survives_matched_lr_frontier`.
- Otherwise return `inconclusive_matched_lr_frontier`.

Any hash, protocol, data, determinism, structure, accounting, initial-equality,
artifact, or finite-value failure yields only `invalid_control`.

A failure closes Ghost Gradient as the current breakthrough direction while
retaining its same-LR optimization effect as a mathematical observation.  A
survival authorizes the five sealed seeds but is still not an A-E result:
training compute and the scrambled-Jacobian control remain mandatory.
