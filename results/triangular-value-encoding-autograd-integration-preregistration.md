# TVE custom-autograd full-model integration gate

## Purpose and ordering

This gate executes only if
`results/triangular-value-encoding-autograd-v2.json` reports that every frozen
correctness and isolated operator-cost gate passed. It asks whether the custom
backward preserves the established TVE optimization behavior in a complete
37.8M model while recovering material training overhead.

This is an implementation-admission gate, not new efficacy evidence. The
five-seed 100M-token direct-native experiment owns the primary architecture
endpoint and its seeds are not pooled here.

## Frozen arms and seeds

Three experiment arms share the exact scratch architecture and canonical
initial parameter state:

1. `canonical_baseline`: gauge-canonical ordinary attention, no value encoding.
2. `reference_tve`: exact serving forward plus the existing zero-forward
   PyTorch surrogate backward.
3. `custom_tve`: exact serving forward plus the custom Triton backward.

Untouched seeds are `17107`, `18211`, and `19319`. To balance sequential thermal
and order drift, their arm orders form this frozen Latin square:

```text
17107: canonical_baseline, reference_tve, custom_tve
18211: reference_tve, custom_tve, canonical_baseline
19319: custom_tve, canonical_baseline, reference_tve
```

Each arm trains for 305 optimizer steps at sequence length 512, micro-batch 32,
gradient accumulation 2, AdamW learning rate `3e-4`, weight decay `0.1`, 50
warmup steps, gradient clip `1.0`, and the pinned self-product corpus. This is
9,994,240 prediction tokens per arm. Evaluation uses the fixed 64 batches of 32
sequences at steps 0, 61, 152, and 305.

Before timing each arm, run three forward/backward no-update warmups and clear
gradients. Training step time includes data-ready model forward, backward,
clipping, optimizer step, zero-grad, and CUDA synchronization. Evaluation and
artifact serialization are excluded. Cost summaries use steps 11 through 305.

## Initial and first-backward gates

For every seed:

1. `reference_tve` and `custom_tve` initial state hashes, initial 64-batch
   per-batch NLL vectors, and first training-batch logits/loss are bit-exact.
2. The first-backward full-model gradient has relative L2 error at most `0.005`
   and cosine similarity at least `0.99998`; every gradient is finite.
3. Every layer has exactly 960 active output slots. Their gradients retain the
   isolated v2 relative-L2, cosine, and RMSE limits. Per-layer value-weight
   gradients retain the relative-L2 and cosine limits; their maximum absolute
   value is reported but is not compared to v2's `dV` maximum because a
   projection-weight gradient is a different quantity. The isolated v2 gate
   remains authoritative for the complete `dV` contract and may not be loosened.

The first-backward probe uses separately rebuilt models and performs no update.

## Optimization-equivalence gates

The seed is the statistical unit. All clustered intervals are two-sided 95%
Student-t intervals with 2 degrees of freedom; validation batches and training
steps are paired measurements, never independent replicates.

1. At steps 61, 152, and 305, the complete seed-clustered interval of
   `custom_tve - reference_tve` paired validation NLL lies inside plus or minus
   `0.02%` of mean reference NLL.
2. At terminal, the absolute relative difference of the two mean validation
   losses is at most `0.01%`.
3. Within every seed, the 305-step paired training-loss trajectory has RMSE no
   more than `0.05%` of mean reference loss and absolute mean bias no more than
   `0.02%`.
4. Both TVE arms retain the small-scale architecture signal: all three terminal
   seed means beat `canonical_baseline`, and each clustered upper bound is at
   most `-0.025%` of mean canonical NLL.

## Full-model cost gates

For each seed, use the median timed training step from steps 11 through 305.

1. `custom_tve / reference_tve <= 0.95` in every seed.
2. The upper 95% seed-clustered interval of log timing ratios is at most
   `log(0.95)`.
3. Define recovered overhead as

   ```text
   (T_reference - T_custom) / (T_reference - T_canonical).
   ```

   Reference overhead must be positive in every seed, recovered overhead must
   be at least 10% in every seed, and its mean must be at least 20%.
4. Custom peak allocated bytes may not exceed reference in any seed.

Custom TVE is not required to match canonical training time because it performs
real additional arithmetic.

## Structural and artifact gates

- All arms have equal parameter count, parameter tensors, buffers, serialized
  state values/bytes, optimizer state bytes, and zero runtime metadata.
- All metrics are finite. Both TVE arms end with nonzero bounded BF16
  coefficients and exact BF16 coefficient export.
- Both TVE arms pass the all-layer bit-exact serving bridge.
- Every arm uses a unique experiment-arm checkpoint name. Each TVE backend uses
  a unique BF16 attention export name. Paths, sizes, and SHA-256 hashes must
  validate; no pre-existing output or artifact is allowed.
- The integrity manifest binds this runner and preregistration; the pilot,
  attention, custom-autograd, and tests; the passing v2 result and manifest; the
  passing H100 v6 result and manifest; the data manifest; and runtime versions.

## Interpretation

A pass admits the custom backward to the 360M second-domain experiment. It
shows 37.8M/10M optimization equivalence plus material recovery of the old
training overhead. It does not establish 360M efficacy, second-domain efficacy,
native served latency, or a general fixed-compute capability gain.
