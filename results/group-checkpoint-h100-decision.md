# Group mid-reduction checkpoint — H100 decision

Decision: **close exported group checkpoints.**

The frozen candidate reduced the retained checkpoint cardinality by 64: one
sign of the first-half accumulator sum per token and fixed 64-feature group.
It preserved the dense result exactly at runtime zero and used Tensor Cores
without spilling, but the cross-lane group reduction created a high temporary
register peak.

| rows | dense/checkpoint ratio | candidate registers | delta |
|---:|---:|---:|---:|
| 1 | 1.004511 | 88 | +24 |
| 8 | 1.007218 | 88 | +24 |
| 64 | 1.008287 | 88 | +24 |
| 256 | 1.009207 | 88 | +24 |
| 1,024 | **1.082045** | 88 | +24 |

It failed the frozen `+8` register limit and `1.03` all-cell latency limit.
The cost is no longer long-lived accumulator-shaped checkpoint state; it is
the peak live state required to summarize the accumulator tile.  Neither form
is a serving win.

Retain the next boundary: do not export partial-reduction history.  If the
history can improve the function, apply a baseline-containing nonlinear
mutation directly to the one live accumulator and then resume the native
reduction.  That changes the reduction algebra without any state surviving
the checkpoint.

Evidence:

- frozen protocol: [`group-checkpoint-h100-preregistration.md`](group-checkpoint-h100-preregistration.md)
- result: [`group-checkpoint-h100-development.json`](group-checkpoint-h100-development.json)
- executable: [`../experiments/group_checkpoint_h100.py`](../experiments/group_checkpoint_h100.py)

