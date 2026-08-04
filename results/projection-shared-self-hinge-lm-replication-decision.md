# Projection-shared self-hinge — 50M replication decision

Status: **direct RMS-gain replacement closed**  
Date: 2026-07-28

Self-hinge repeated its early advantage on an independent seed, but did not
retain it at the frozen 50M endpoint.

| arm | 10M NLL | 50M NLL |
|---|---:|---:|
| raw RMS gain | 6.29844994 | **5.52353868** |
| self-hinge | **6.29824674** | 5.52481525 |
| folded null | 6.29880837 | 5.52588144 |

At 50M, self-hinge minus raw was `+0.00127658`, paired 95% interval
`[+0.00116901,+0.00138414]`.  It still beat folded-null, and turning gamma off
worsened NLL by `0.00525890`; the nonlinear feature was learned and useful.
It was not useful enough to replace the learned RMS gain over the longer
horizon.

Retain the feature and change the cost domain.  During training, keep both the
ordinary gain and the nonlinear chart.  At export, fold the gain exactly into
the columns of every immediate downstream matrix and delete it.  This spends
`O(Ld)` extra training parameters and optimizer state while preserving the
served parameter count, matrix shapes, activation width, and KV cache.

