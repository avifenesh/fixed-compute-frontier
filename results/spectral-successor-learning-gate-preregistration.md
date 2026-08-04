# Spectral successor learning gate: frozen preregistration

Date frozen: 2026-07-30

## Status and scope

This is a **T1 mechanistic falsification**, not a breakthrough test. A pass only
permits a natural-language pilot. It is not evidence that the final research
bar has been met.

The candidate is training-only spectral successor supervision: a deleted
64-dimensional linear head predicts an ordered, discounted code of the future
suffix from every causal hidden state. The deployed Transformer is identical
to the control Transformer.

Frozen artifacts:

- experiment: `experiments/spectral_successor_learning_gate.py`
  SHA-256 `77f57beb65cde92a4fabff3ccb8b6e0aaf39f4f0b1510b0a7083fa2a2ee1c55b`
- tests: `tests/test_spectral_successor_learning_gate.py`
  SHA-256 `60119b0d4b689fbd70aca1b7090f904e2abe447aee79990b645502288263d729`
- Stage 0 result: `results/spectral-successor-stage0.json`
  SHA-256 `2f8042d7a053f39ce002ee7be62e0683ac64d4f2f14bdd0c8a774895cd45cf7a`
- hypothesis: `results/spectral-successor-training-hypothesis.md`
  SHA-256 `8e4fa680d5222bc9dfe358ce2abcc461d2dc506cba8acf449c59f2182cce1294`

No candidate result has been run or observed before freezing these hashes.

## Exact question

Does preserving ordered, multi-horizon information in one fixed-width target
teach the same causal model a latent sequence plan materially earlier than
next-token prediction, compressed multi-token prediction, unordered future
summary prediction, or an information-destroying spectral control?

## Fixed model and training

- Same deployed causal Transformer in every arm: vocabulary 512, width 128,
  3 layers, 4 heads, FFN width 512, tied embedding/unembedding.
- From-zero initialization, paired by seed and task.
- Seeds: 731 and 947.
- AdamW, learning rate 0.001, betas (0.9, 0.95), weight decay 0.1.
- Linear warmup 40 steps, cosine decay, 800 optimizer steps.
- Batch size 128, BF16 autocast, gradient norm clipped to 1.
- Checkpoints: 0, 10, 20, 40, 80, 160, 320, 640, 800.
- Candidate auxiliary loss weight: 1.0. No tuning after results.

## Fixed tasks

Each example is BOS, a class key, 48 random distractor tokens, SEP, then a
34-token plan. The structured plan is one of 64 affine permutations over a
17-token alphabet, repeated twice. Every class therefore has the same token
multiset but a distinct order.

Controls on the structured task:

1. `ntp`: next-token prediction only.
2. `mtp8`: eight future offsets compressed into 64 random-code dimensions.
3. `fsp_bow`: unordered 64-dimensional suffix summary.
4. `spectral_scrambled`: the spectral target assigned to a different batch
   row, preserving auxiliary compute and target statistics but destroying the
   example-to-target information.

Candidate: `spectral`.

Specificity controls compare `ntp` with `spectral` on:

- random class-to-permutation mappings;
- an order-insensitive task where every class has the same sorted plan.

## Resource accounting

Every comparison uses charged multiply-like work, not steps or wall clock.
The ledger charges three times forward MACs for forward plus backward, and
also charges the auxiliary head and target construction. Softmax,
normalization, activation scalar operations, optimizer scalar operations, and
memory traffic are reported as exclusions. Wall time is recorded separately.

The training-only 64x128 head is deleted. All arms must export the same model
graph and the same trainable parameter count.

## Frozen decision rule

Advance only if **all** conditions hold:

1. `spectral` reaches at least 90% exact accuracy on the first two plan tokens
   in both seeds.
2. Charged work to that threshold is at least 1.25x lower than the best of all
   four structured controls in both seeds.
3. Candidate terminal total NLL is within 1% of the best structured control in
   both seeds.
4. Order-insensitive terminal total NLL is within 2% of NTP in both seeds.
5. Random-mapping first-two exact accuracy is within two percentage points of
   NTP in both seeds.
6. Exported graph and parameter count are identical, with no auxiliary state.

The 1.25x threshold is only a cheap-screen admission threshold. It is **not**
the project success threshold. Eventual success still requires at least 1.5x
total-training compute-equivalent gain across scales and seeds, a confirmed
scaling-law break, or a qualitative capability that the control lacks even at
2x training compute.

## Interpretation before seeing results

- Any failed gate closes this formulation without hyperparameter rescue.
- A loss improvement below the stated acquisition threshold is a negative
  result, even if statistically clean.
- A pass means only that ordered future credit is worth testing on a fixed
  natural-language model under the stronger optimizer/baseline envelope.
- No result from this synthetic system may be called a breakthrough.
