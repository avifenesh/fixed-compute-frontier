# Projection-shared coupling — 10M development decision

Status: **two-way coupling closed; self-hinge promoted for replication**  
Date: 2026-07-28

All five arms had 37,758,336 parameters and identical initial validation loss.
The two-way candidate beat raw RMSNorm with a paired interval wholly below
zero, and turning its coupling off hurt.  It nevertheless lost significantly
to the simpler same-budget self-hinge control.

Terminal validation NLL:

| arm | NLL |
|---|---:|
| self-hinge | 6.54706765 |
| two-way coupling | 6.54709728 |
| one-way coupling | 6.54733408 |
| raw RMS gain | 6.54751825 |
| folded null | 6.54756417 |

Self-hinge minus two-way was `-0.00002963`, paired 95% interval
`[-0.00004956,-0.00000970]`.  Cross-channel coupling supplied no advantage at
this horizon.  Close it and replicate self-hinge directly.

