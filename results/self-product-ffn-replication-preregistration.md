# Self-product FFN replication preregistration

## Frozen question

Does the seed-1907 quality result repeat at seed 2718 under a prospectively
revised symmetric activation-safety diagnostic, when every training token,
parameter, dense MAC, validation batch, BF16 deployment fold, and H100 serving
gate is held fixed?

This is a replication of the same architecture, not a modified candidate:

`y = B [0.44642046792894413 * SiLU(Ax) * (Ax)]`, width 1,536.

Controls remain parallel SwiGLU width 1,024 and parameter-matched wide SiLU
width 1,536. All three have identical total/trainable parameters and
1,179,648 FFN dense parameters/MAC coefficients per layer.

## Frozen protocol

- Seed: 2718.
- Same H100, data ledger, model, optimizer, schedule, arm order, 1,525 steps,
  49,971,200 prediction tokens per arm, and validation batches as discovery.
- The frozen discovery artifact and this source/preregistration/test are
  hash-checked before training.
- The untouched seed-31415 confirmation source, preregistration, and test are
  also hash-frozen before this seed-2718 run begins.
- Both scaled arms are evaluated after folding their scale into the down
  weights; all quality promotion comparisons use those BF16-folded scores.

## Revised exploratory gates

All must pass:

1. The candidate is at least 0.05% below SwiGLU terminal NLL and its paired
   95% interval is wholly favorable.
2. The candidate is at least 0.025% below folded wide-SiLU terminal NLL and
   its paired 95% interval is wholly favorable.
3. The candidate is noninferior to SwiGLU within 0.05% at 10M tokens.
4. The plain-SiLU ablation hurts by at least 0.01% with a wholly favorable
   paired interval.
5. Parameter, MAC, fold-equivalence, finite-training, and frozen H100 gates
   all pass.
6. Symmetric activation safety at initialization and terminal, independently
   in both seeds: candidate median RMS, worst-layer p99, and worst-layer
   absolute maximum are each no greater than the corresponding SwiGLU value.
   This replaces, but does not erase, the discovery screen's failed asymmetric
   fixed-threshold gate.
7. Averaging the two seed differences within each of the same 128 validation
   batches gives a wholly favorable batch-clustered descriptive interval versus
   each control. This is not interpreted as 256 independent observations or as
   a seed-level confidence interval.

Because the symmetric safety metrics were chosen after inspecting seed 1907,
passing only justifies an untouched confirmation seed whose gates are frozen
before seed 2718 is observed. This run cannot erase the discovery screen's
formal failure and cannot directly promote to a larger-scale test.
