# Served ShiftNorm — frozen H100 gate

Status: **frozen before execution**  
Date: 2026-07-28

Compare the ordinary fused RMSNorm output `z*gain` with served ShiftNorm
`z+bias`, each followed by the same BF16 projection.  Both read one `d`-vector,
write one normalized `d`-vector, use the same projection tensor, and allocate
the same output/scratch shapes.

Frozen grid: hidden 4,096; output widths 6,144 and 14,336; rows
`{1,8,64,256,1024}`; 50 warmups and 500 interleaved samples.  Pass if resources
are identical, candidate/baseline is at most 1.01 at rows 256 and 1,024, at
most 1.03 everywhere, and BF16 correctness error is at most 0.02.

