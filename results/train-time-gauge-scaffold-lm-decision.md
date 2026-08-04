# Training-time gauge scaffold — 10M decision

Status: **nonlinear charts closed; shared shift promoted**  
Date: 2026-07-28

The screen validated the cost trade but rejected its original nonlinear
winner rule.  The shared-shift control was decisively best:

| arm | terminal NLL |
|---|---:|
| scaffolded shared shift | **6.56207684** |
| scaffolded hinge | 6.56247661 |
| scaffolded scale | 6.56319776 |
| scaffolded centered hinge | 6.56325645 |
| scaffolded null | 6.56332934 |
| raw RMS gain | 6.56333854 |

Shared shift improved over raw by `0.00126170` NLL (`0.0192%`) while preserving
the exact exported parameter count.  The absorbable scale chart improved only
`0.00014079`; training overparameterization explains part, but not most, of the
shift result.  Hinge lost to shift by `0.00039976` with a paired interval wholly
favoring shift.  Close the nonlinear charts.

Promote the simpler successor: train `g*(RMS(x)+b)`, export by folding `g` into
all immediate matrix columns, and serve `RMS(x)+b`.  This replaces the served
RMS gain multiply with a shift add at the same `d` parameters.  A new-seed 50M
screen must compare raw gain, direct shift without scaffold, an absorbable
scale scaffold, and the full scaffolded shift.

Note: the nonzero terminal hinge ablations in the development payload are
invalid because arm dispatch overrode the requested ablation mode.  The zero
ablation is valid; no promotion claim relies on the invalid entries.

