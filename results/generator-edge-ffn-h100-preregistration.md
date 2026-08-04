# Generator-edge FFN — unfused H100 preregistration

Freeze before measurement:

- Existing Vast H100 SXM 80 GB, PyTorch 2.5.1+cu124, Transformers 4.57.6.
- The 12-layer hidden-384 scratch model using the independent one-norm
  parallel attention/FFN block.
- Four exact-FFN-parameter arms: ordinary width-1,024 SwiGLU,
  width-1,536 self-product, duplicate-edge, and two-permutation
  generator-edge.
- BF16 autocast, inference mode, TF32 enabled, no cache, no compilation.
- Cells `(batch,tokens)`: `(1,1)`, `(1,512)`, `(8,512)`, `(32,512)`.
- All four models remain resident.  Five warmups per model/cell, followed by
  30 synchronized forwards per arm in independently shuffled arm order each
  round.  Same input tensor within a cell.
- Report median, p10, p90, minimum, peak allocated bytes, exact parameter
  counts, device/software versions, and source hashes.

The candidate passes feasibility only if its median latency is at most 1.05x
ordinary parallel SwiGLU in every cell.  This reference implementation
materializes the `2M` edge tensor, so a failure closes the unfused path but
does not prove a fused tile generator impossible.  A pass only admits the LM
screen; it is not a production serving claim.
