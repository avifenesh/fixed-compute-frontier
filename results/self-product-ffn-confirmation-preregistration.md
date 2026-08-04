# Self-product FFN untouched confirmation preregistration

This document and its runner/test hashes are frozen before observing seed 2718.

## Question

At untouched seed 31415, does the exact same self-product FFN again improve
quality at identical served parameters and dense MACs, while passing the
prospectively selected symmetric activation-safety test?

## Frozen protocol

- Candidate and controls: identical to discovery and seed-2718 revised
  exploratory replication.
- Seed: 31415.
- Same H100, data ledger, arm order, model, optimizer, schedule, 1,525 steps,
  49,971,200 prediction tokens per arm, and 128 validation batches.
- Both scaled arms use fresh BF16 evaluation after deployment folding.
- Imported base, discovery, replication, preregistration, runner, test, and
  integrity-chain hashes must all close before training.

## Confirmation gates

All must pass:

1. Candidate terminal NLL is at least 0.05% below SwiGLU and at least 0.025%
   below folded wide SiLU; both paired 95% intervals are wholly favorable.
2. Candidate at 10M tokens is noninferior to SwiGLU within 0.05%.
3. Plain-SiLU ablation hurts at least 0.01% with a wholly favorable interval.
4. All exact parameter/MAC, fold-equivalence, finite-training, and frozen H100
   checks pass.
5. At initialization and terminal, candidate median activation RMS,
   worst-layer p99, and worst-layer absolute maximum are each no greater than
   SwiGLU's corresponding value.
6. Seed 2718 passed the already frozen revised-exploratory gates, while the
   original seed-1907 formal failure remains explicitly preserved.
7. The quality thresholds and favorable paired intervals hold independently
   at all three seeds, ignoring only the original asymmetric activation-tail
   gate that this untouched confirmation is designed to adjudicate.

Passing promotes the architecture to a larger model/token scale test. It does
not by itself establish broad scaling, downstream benchmark gains, or novelty.
