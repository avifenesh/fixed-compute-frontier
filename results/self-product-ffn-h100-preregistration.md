# Folded self-product FFN — H100 preregistration

Freeze before measurement:

- Same H100/software/model shape and `(1,1)`, `(1,512)`, `(8,512)`,
  `(32,512)` cells as the generator-edge timing gate.
- Arms: independent parallel width-1,024 SwiGLU, folded width-1,536 wide SiLU,
  folded width-1,536 self-product.
- The fixed activation scales are multiplied into each down weight once before
  timing.  No runtime scale multiply remains.
- All models resident; same input per cell; five warmups; 30 synchronized
  forwards per arm in independently shuffled order each round; BF16 inference,
  TF32, no cache/compile.
- Exact equal learned parameter counts, source hashes, all samples, medians,
  p10/p90/min, and ratios are recorded.

Self-product passes only if median latency is at most 1.02x parallel SwiGLU in
all four cells.  This admits the LM screen; it is not a production serving
claim.
