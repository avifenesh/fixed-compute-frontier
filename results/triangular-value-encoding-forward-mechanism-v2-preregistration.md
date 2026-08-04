# TVE deterministic forward-mechanism falsifier v2

## Disclosed invalid v1

V1 stopped before training. Initial states, logits, and losses matched, but its
bit-exact gradient gate failed. A diagnostic rerun found identical tensor
values and strides yet differences across 102 parameter gradients. Repeating
the probe with deterministic CUDA algorithms and math SDPA made every
backward-only versus full-TVE parameter gradient bit-exact. The failure was
therefore an uncontrolled nondeterministic attention-backward comparison, not
an efficacy result. V1 remains an invalid screen and is hash-bound here.

V2 changes only the numerical execution contract needed to make the causal
intervention exact: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic PyTorch
algorithms, Flash and memory-efficient SDPA disabled, and math SDPA enabled.
cuDNN SDPA is also disabled, and reduced-precision BF16/FP16 math-SDPA
reductions are disabled so the selected backend and accumulation contract are
both explicit.
The architecture, data, seed, optimizer, token budget, and outcome thresholds
remain those of v1.

## Question

Does the small LM gain require TVE's nonlinear forward path, or can TVE's
altered gradient pathway produce the same gain while inference remains an
ordinary canonical transformer?

This is a one-seed diagnostic screen. It can falsify forward-path necessity for
this seed. It cannot positively isolate representational capacity, even after
replication.

## Frozen arms

All arms start from the same gauge-canonical parameters.

1. `canonical_baseline`: ordinary values in forward and backward.
2. `backward_only_tve`: exactly ordinary values in forward, TVE's reference
   VJP in backward, implemented as `stopgrad(v) + T(v) - stopgrad(T(v))`.
3. `full_tve`: exact serving TVE forward plus the same reference TVE VJP.

The backward-only and full arms must have bit-exact initial state, first-batch
logits/loss, and every first-batch parameter gradient under the frozen
deterministic execution contract. A nonzero-coefficient local BF16 probe must
show ordinary backward-only forward, distinct full forward, identical value
and coefficient VJPs, and nonzero active coefficient gradients.

## Frozen training protocol

- Untouched seed `21017`.
- Arm order: canonical, backward-only, full.
- 37,758,336 parameters; 12 layers; hidden 384; Q heads 6; KV heads 2;
  head width 64; FFN width 1,024.
- Pinned self-product corpus, sequence length 512, micro-batch 32,
  accumulation 2, 305 optimizer steps = 9,994,240 prediction tokens.
- AdamW learning rate `3e-4`, 50 warmup steps, weight decay `0.1`, gradient
  clip `1.0`.
- Fixed 64 x 32-sequence validation batches at steps 0, 61, 152, and 305.
- Same parameters, buffers, serialized state, optimizer state, and zero
  inference metadata in every arm.
- Deterministic CUDA algorithms and math SDPA for the entire probe and run.

## Frozen diagnostic classification

Let `G_full = L_canonical - L_full`,
`G_backward = L_canonical - L_backward`,
`M_forward = L_backward - L_full`, and
`capture = G_backward / G_full`, using terminal mean validation NLL.

- `forward_path_required_for_gain_replication` requires positive full gain, a
  relative full-over-canonical gain of at least `0.05%`, a relative
  full-over-backward-only margin of at least `0.05%`, `capture <= 0.50`, and
  full TVE below backward-only at all three post-training evaluations.
- `backward_signal_captures_gain_this_seed` requires a relative
  full-over-canonical gain of at least `0.05%`, `capture >= 0.80`, and a
  relative full-over-backward-only margin below `0.025%`.
- Every other valid result is `mixed_or_inconclusive`.
- Any failed determinism, intervention, structure, state, serving bridge,
  artifact, integrity, or data check yields only `invalid_screen`.

Paired validation-batch intervals are within-seed diagnostics, not independent
architecture replicates. If either causal classification fires, repeat on at
least three untouched seeds. A replicated forward-path requirement would
still establish forward/backward consistency or forward-path necessity for
the gain, not representational capacity by itself.
