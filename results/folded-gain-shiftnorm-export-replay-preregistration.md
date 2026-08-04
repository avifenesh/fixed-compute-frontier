# Folded-gain ShiftNorm — frozen export replay

Status: **frozen before replay**  
Date: 2026-07-28

Replay the exact seed-813 scaffolded-shift arm for 1,525 steps.  Require its
pre-export validation NLL to reproduce the recorded `5.527982976287603` within
`2e-6`.  Then materialize every gain into Q/K/V or gate/up input columns,
replace the training norm by gain-free ShiftNorm, and reevaluate the same 128
validation batches.

Pass export if served parameter count becomes exactly 37,758,336, matrix/cache
shapes are unchanged, absolute mean NLL drift is at most `2e-5`, and the paired
95% interval contains zero.  This tests actual BF16 execution, not only real
algebra.

