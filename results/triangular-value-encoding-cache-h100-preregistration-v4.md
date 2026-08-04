# TVE serving-aligned H100 admission gate v4

V4 keeps the exact serving-forward contract from v3. It replaces v3's arbitrary sub-micro error thresholds with thresholds tied to BF16 resolution and to the later capability gate. The formal seed is fresh.

## Frozen screen

- H100 80GB; packed BF16 QKV; FP32 one-store RoPE; encoded V written identically to QKV and unchanged KV cache.
- Rows 1, 8, 32, 128, 512, and 2048; 30 warmups; 200 randomized paired trials; 128 MiB arithmetic L2 touch; 5,000 bootstrap replicates.
- Training/evaluation candidate forward uses the exact serving kernel. The PyTorch expression supplies its algebraically equivalent gradient.

## Frozen gates per cell

1. Control is bit exact. Candidate is bit exact to the serving-aligned forward. The nonlinear delta is nonzero.
2. PyTorch-reduction drift is below one conservative BF16 absolute step (0.015625), has mean at most 2e-8, and affects at most 1e-5 of V values.
3. After fixed attention mixing and O projection, max drift is below 0.015625 and mean is at most 1e-6. Proxy NLL drift is at most 1e-5, over 100 times smaller than the later minimum quality-gain threshold.
4. The upper 95 percent latency ratio is at most 1.025.
5. Added learned-weight bytes, KV bytes, and metadata bits are zero.

All inputs are hash-bound before the run. A pass only admits the quality experiment.
