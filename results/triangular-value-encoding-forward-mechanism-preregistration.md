# TVE forward-mechanism falsifier

## Question

Does the small LM gain require TVE's nonlinear forward path, or can TVE's
altered gradient pathway produce the same gain while inference remains an
ordinary canonical transformer?

This is a one-seed diagnostic screen. It can cheaply falsify forward-path
necessity for one seed if the backward-only arm captures the gain. It cannot
positively isolate representational capacity, even after replication.

## Frozen arms

At the same gauge-canonical initialization:

1. `canonical_baseline`: ordinary values in forward and backward.
2. `backward_only_tve`: exactly ordinary values in forward, TVE's reference VJP
   in backward, implemented as

   ```text
   stopgrad(v) + T(v) - stopgrad(T(v)).
   ```

3. `full_tve`: exact serving TVE forward plus the same reference TVE VJP.

The backward-only and full arms must have bit-exact initial state, first-batch
logits/loss, and first-batch gradients. The backward-only output must be
bit-exact to its input for a random BF16 integration probe, while coefficient
gradients remain nonzero.

## Frozen training protocol

- Untouched seed `21017`.
- Arm order: canonical, backward-only, full.
- 37.8M parameters; 12 layers; hidden 384; Q heads 6; KV heads 2; head width 64;
  FFN width 1,024.
- Pinned self-product corpus, sequence length 512, micro-batch 32, accumulation
  2, 305 optimizer steps = 9,994,240 prediction tokens.
- AdamW learning rate `3e-4`, 50 warmup steps, weight decay `0.1`, gradient clip
  `1.0`.
- Fixed 64 x 32-sequence validation batches at steps 0, 61, 152, and 305.
- Same parameters, parameter tensors, buffers, serialized state size, optimizer
  state size, and zero inference metadata in every arm.

## Frozen diagnostic classification

Let

```text
G_full = L_canonical - L_full
G_backward = L_canonical - L_backward
M_forward = L_backward - L_full
capture = G_backward / G_full
```

using terminal mean validation NLL.

- `forward_path_required_for_gain_replication` requires `G_full > 0`, a
  relative full-over-canonical gain of at least `0.05%`, a relative
  full-over-backward-only margin of at least `0.05%`, `capture <= 0.50`, and
  full TVE below backward-only at all three post-training evaluations.
- `backward_signal_captures_gain_this_seed` requires a relative
  full-over-canonical gain of at least `0.05%`, `capture >= 0.80`, and a relative
  full-over-backward-only margin below `0.025%`.
- Every other result is `mixed_or_inconclusive`.

An invalid probe, structural comparison, or artifact check yields only
`invalid_screen`, regardless of losses.

The paired 64-batch loss differences and intervals are reported only as
within-seed diagnostics; batches are not treated as independent architecture
replicates.

If the first classification fires, repeat on at least three untouched seeds.
Even after replication it establishes only that a forward-consistent TVE path
is required for the training benefit: the backward-only arm is a
straight-through, non-integrable surrogate, so full TVE can beat it through
forward/backward consistency rather than representational capacity alone. If
the second classification fires, forward-path necessity is falsified for this
seed; replication is required before generalizing that rejection across seeds
or scale.
