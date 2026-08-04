# Self-product fused-serving scale admission preregistration

This runner, preregistration, and test are hash-frozen before observing the
fused H100 result.

The 102.247M-parameter / 99.94M-prediction-token scale LM experiment may run
only when all conditions hold:

1. The exact eager H100 artifact remains a frozen latency/memory failure.
2. The separately frozen fair in-place fused H100 artifact passes every wall,
   CUDA-event cached-decode and prefill latency gate, every prefill
   absolute/incremental allocated/reserved peak gate, and every static BF16
   byte, embedded semantic-equivalence, environment, and hash gate.
3. The untouched 37.75M confirmation still says `advance_to_scale_test=true`.
4. The original scale architecture, data, optimizer, quality thresholds,
   activation-safety gates, and arm order remain unchanged.

The admission runner changes only which serving artifact satisfies the H100
gate. It cannot alter model algebra, training, data, or quality thresholds.
The fused result must embed the hashes of this exact runner, preregistration,
test, precommit manifest, confirmation result, and scale data manifest; the
runner verifies them again before scale training.
