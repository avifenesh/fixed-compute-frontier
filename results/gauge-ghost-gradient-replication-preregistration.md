# Gauge Ghost Gradient five-seed replication

## Hypothesis formed before these seeds

The valid deterministic mechanism screen at seed `21017` found that a
training-only TVE Jacobian improved terminal validation NLL by `0.2567%`
relative to canonical training, captured `115.2%` of full TVE's gain, and
served as an ordinary transformer. That seed is development evidence only and
is not included here.

This replication tests the narrower claim on five untouched initialization
seeds: a richer virtual feature algebra in backward can improve a fixed dense
Transformer while the learned forward function and inference graph remain
ordinary.

## Frozen method

For attention values `v` and the TVE virtual transform `T(v; O)`, train with

```text
v_ghost = stopgrad(v) + T(v; O) - stopgrad(T(v; O)).
```

Numerically, `v_ghost == v` for every forward pass. Its VJP is the TVE VJP, so
the ordinary physical O weights receive an added signed-second-moment update
and, after those coordinates become nonzero, values receive triangular
feedback. The ghost expression is deleted at inference. There are no extra
parameters, buffers, KV values, cache metadata, or serving operators.

This changes the optimization vector field, not the served hypothesis class.
It cannot establish a higher approximation ceiling than an ordinary
Transformer; it can establish a better solution under the frozen finite
training budget.

## Frozen experiment

- New seeds: `22003, 23003, 24007, 25013, 26003`.
- Arms: `canonical_baseline` and `backward_only_tve`.
- Arm order alternates by seed, starting canonical on seeds 22003, 24007,
  26003 and ghost first on seeds 23003, 25013.
- 37,758,336 parameters; 12 layers; hidden 384; Q heads 6; KV heads 2;
  head width 64; FFN width 1,024.
- Pinned self-product corpus, identical data order, sequence length 512,
  micro-batch 32, accumulation 2, 305 optimizer steps = 9,994,240 prediction
  tokens per arm.
- AdamW learning rate `3e-4`, 50 warmup steps, weight decay `0.1`, gradient
  clip `1.0`.
- Fixed 64 x 32-sequence validation batches at steps 0, 61, 152, and 305.
- Deterministic CUDA algorithms; Flash, memory-efficient, and cuDNN SDPA
  disabled; math SDPA enabled; reduced-precision math-SDPA reductions disabled.

## Required validity checks

For every seed:

1. The exact intervention probe passes, including equal initial state, logits,
   loss, and every backward-only/full-TVE first-batch parameter gradient.
2. Canonical and ghost arms have identical initial state and initial
   per-batch evaluation losses.
3. Parameters, parameter tensors, buffers, serialized state, optimizer state,
   and metadata bits are equal; total parameters are exactly 37,758,336.
4. Ghost forward semantics are ordinary canonical and require no TVE cache
   encoding.
5. Reused physical coefficients become nonzero, remain bounded by 0.5 in BF16,
   and export bit-exactly.
6. Training is finite and the terminal ordinary-serving attention artifact
   exists at the exact path, size, and SHA-256 reported. The artifact is
   reloaded; seed, arm, ordinary-forward tag, 12-layer topology, tensor shapes,
   BF16 dtypes, finite tensors, and terminal all-layer ghost/ordinary forward
   identity are revalidated.
7. Every arm records exactly 305 step losses and 9,994,240 prediction tokens;
   every required evaluation records exactly 64 batches and 1,048,576 tokens.
   Every recorded and derived numerical value must be finite.

Any failed integrity, protocol, data, intervention, structural, numerical, or
artifact check invalidates the replication.

The result is checkpointed after each completed arm. Resume is allowed only
when schema, source, preregistration, manifest, protocol, data ledger, runtime,
deterministic backend, ordered seed prefix, ordered arm prefix, recomputed
intervention probes, realized accounting, artifacts, and hashes all revalidate.
Completed arms are never silently rerun or accepted from an unbound payload.
An arm journal is written before training. If the frozen helper finishes a
backward artifact but the parent result checkpoint is interrupted, only an
artifact with the exact bound journal and full reloaded serving validation may
be moved to a hash-recorded recoverable quarantine before deterministic
retraining. Unbound or malformed orphans abort the run.

## Frozen primary gates

Using paired terminal validation-batch mean differences within each seed and a
two-sided Student-t interval across the five seed means (`df=4`, critical value
`2.776445105`), all of the following must hold:

1. Every one of the five seed means favors ghost-gradient training.
2. The upper 95% seed-cluster bound clears a `0.01%` relative NLL improvement
   versus mean canonical NLL.
3. The improvement of mean terminal NLL is at least `0.05%` relative.

Validation batches are paired diagnostics within a seed; the five
initialization seeds are the architecture replicates. Training throughput is
reported but is not an efficacy gate. A pass supports an ordinary-inference
optimization edge at this model/data/token scale. It is not yet a deployment
breakthrough until an efficient backward implementation and a second
model/domain scale pass.
