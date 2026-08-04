# Self-product FFN scale-matched H100 preregistration

- Exact scale architecture from the scale LM preregistration.
- Deployment-fold both scaled arms before timing.
- Convert the full model to resident BF16 before warmup; no timed weight casts.
- Prefill cells: `(1,512)`, `(8,512)`, `(32,512)`.
- Cached decode cells: batch 1 and 8, one token after a populated 512-token KV
  cache. Empty-context `(1,1)` is not called decode.
- Five warmups and 30 seed-271828 interleaved repetitions per cell.
- Record synchronized wall and CUDA-event samples. Candidate median must be no
  greater than SwiGLU and the fixed-bootstrap 95% upper ratio at most 1.02 in
  every cell for both clocks.
- In an allocation-clean phase, build one arm at a time and record allocated
  and reserved base/peak/increment bytes for both decoder-core (no LM-head
  logits) and end-to-end calls. Candidate increments must be no greater than
  SwiGLU in every prefill cell; absolute allocated/reserved peaks must also be
  no greater.
- A non-tensor CUDA-library runtime floor may remain after all live tensors are
  deleted. Record its allocated/reserved bytes and require the exact same floor
  before every isolated arm; do not misclassify it as model memory or silently
  require an impossible zero.
- Record and require identical BF16 parameter/buffer bytes and dtypes.
- Require H100, Torch 2.5.1+cu124, CUDA 12.4, Transformers 4.57.6, and exact
  source/dependency hashes. No LM training if any gate fails.
