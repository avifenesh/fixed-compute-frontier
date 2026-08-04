# Gauge Ghost Gradient fatal learning-rate control

## Why this precedes fresh-seed replication

On development seed `21017`, backward-only TVE improved terminal validation
NLL from `6.5238095224` to `6.5070632696`, a gain of `0.0167462528`. Its
pre-clip gradient norms were larger than canonical. Before attributing the gain
to the structured virtual Jacobian, test whether ordinary global AdamW update
scale explains it.

This is a falsifier on the already-used development seed. It contributes no
new replication seed.

## Frozen control

The existing `3e-4` canonical and ghost arms are hash-bound from the valid
deterministic mechanism result. Train four new canonical arms at the same
initialization, data order, 9,994,240 prediction tokens, optimizer, warmup,
weight decay, clipping, deterministic math SDPA, and evaluation batches.

The full log-symmetric grid is:

```text
1.5e-4, 3e-4/sqrt(2), 3e-4, 3e-4*sqrt(2), 6e-4.
```

Every new arm must reproduce the prior canonical initial state hash and all 64
initial per-batch losses exactly; record exactly 305 finite step losses,
9,994,240 training tokens, and 64 finite batches / 1,048,576 tokens at each of
steps 0, 61, 152, and 305. Parameter/state shape is unchanged and no artifact
or metadata is added.

## Frozen classification

Let `G0 = 0.0167462528`, let `Lghost` be the frozen ghost terminal NLL, and let
`Lbest` be the best terminal NLL over the five canonical learning rates.

- If the best canonical LR is either grid endpoint, return
  `endpoint_extension_required`; do not interpret the mechanism.
- Return `learning_rate_explanation_fatal` if LR tuning captures at least 80%
  of `G0`: `(Lbase - Lbest) / G0 >= 0.80`.
- Return `ghost_survives_global_lr_control` only if ghost beats the best
  non-endpoint canonical arm by at least `0.05%` relative NLL and retains at
  least 50% of `G0`, i.e. `Lbest - Lghost >= 0.5 * G0`.
- Otherwise return `inconclusive_lr_control`.

Any integrity, protocol, data, determinism, structure, realized accounting,
initial equality, or finite-value failure yields only `invalid_control`.

A surviving result permits the five fresh-seed replication, but does not prove
TVE-specific alignment. A later scrambled-ghost control must preserve forward
identity and compute while destroying coordinate alignment. Training-compute
efficiency also remains open because the current exact ghost surrogate is
about 1.5x slower than canonical.
