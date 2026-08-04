# Gauge Ghost Gradient matched-LR upper extension

## Why

The valid seven-point matched frontier put both canonical and backward-only TVE
minima at the shared upper endpoint `1.2e-3`.  At that point canonical NLL was
`6.0990843549` and ghost NLL was `6.1027457491`.  Extend only the unresolved
upper boundary on the already-used seed `21017`; do not consume replication
seeds.

## Frozen extension

Train both arms at

```text
1.2e-3 * sqrt(2), 2.4e-3, 2.4e-3 * sqrt(2), 4.8e-3.
```

Execute the grid from highest LR downward, canonical then ghost.  This exposes
an unstable upper boundary before spending the lower arms.  A non-finite arm
still invalidates this run; it is not silently treated as an infinite loss or
used to declare an interior optimum.  A follow-up lower-bound protocol would
then be required.

Reuse the complete hash-bound prior frontier through `1.2e-3`.  Every new arm
uses the identical initialization, data order, deterministic math-SDPA backend,
AdamW schedule, clipping, geometry, 305 steps, 9,994,240 prediction tokens, and
64 fixed validation batches at steps 0, 61, 152, and 305.  It must reproduce
the prior initial state hash and all 64 step-zero losses exactly.  Parameters,
buffers, optimizer state, metadata, and ordinary inference semantics remain
matched.  Ghost exports are required validation artifacts but are not served
state.

This is an explicit all-or-nothing restart-only run.  The result and every
expected export path are preflighted absent before any arm starts; no partial
payload is written.  If interrupted, any exports are archived out of the
frozen paths and all eight arms restart.  Partial arms are never admitted as
evidence.

## Frozen decision

Combine the prior seven points with the four extensions.  Let `Lc` and `Lg` be
the best terminal canonical and ghost losses.

- If either arm's best loss ties or occurs at the new `4.8e-3` endpoint, return
  `endpoint_extension_required`.
- If `Lg / Lc - 1 >= 0.0005`, return
  `ghost_fails_matched_lr_frontier`.
- If `Lc / Lg - 1 >= 0.0005` and the paired ghost-minus-canonical 95% interval
  has upper bound below zero, return `ghost_survives_matched_lr_frontier`.
- Otherwise return `inconclusive_matched_lr_frontier`.

Any integrity, protocol, data, deterministic-backend, structure, accounting,
initial-equality, artifact, or finite-value failure yields `invalid_control`.
Only a non-endpoint survival may release the five sealed seeds.
