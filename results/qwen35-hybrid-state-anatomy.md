# Qwen3.5 hybrid recurrent-state anatomy

Status: **diagnostic closed — measured heterogeneity, no candidate 004 admitted**  
Date: 2026-07-23

## Question

Qwen3.5-0.8B Base alternates three Gated DeltaNet layers with one full-attention
layer. Does its fixed-width recurrent state spend the same capacity on heads
that need very different amounts, and can unequal head growth recover memory at
unchanged model quality?

This is an architecture-state experiment. No weights were trained and no corpus
was tuned. Held-out text was used only to test whether a cache intervention
changes the frozen model's future distribution.

## Physical ledger

The 24-layer text stack contains 18 Gated DeltaNet layers and six full-attention
layers. Each recurrent cache tensor is FP32 with shape
`[batch, 16, 128, 128]`:

\[
16\cdot128\cdot128\cdot4 = 1\ \text{MiB/layer}.
\]

Per request:

| State | Size |
|---|---:|
| 18 recurrent matrices | 18 MiB |
| 18 convolution caches | 0.84375 MiB |
| Six attention KV caches | 12 KiB/token |
| Total at 2,048 prompt tokens | 42.84375 MiB |

A rank-`r` factorization stores `U`, singular values, and `V`, or
`r(128 + 1 + 128)` FP32 values per head. Rank 32 halves the recurrent state;
rank 64 is already slightly larger than the dense matrix.

## Causal probe

For every sequence:

1. prefill the frozen model and clone its pristine hybrid cache;
2. SVD all 288 recurrent head matrices;
3. replace every matrix with an oracle rank approximation while leaving
   convolution state and exact-attention KV untouched;
4. teacher-force the identical 32-token continuation;
5. measure full-vocabulary `KL(clean || changed)`, target-token delta NLL, and
   top-1 agreement.

Unequal variants keep the same total rank budget, allocate rank in multiples of
eight, and cap every head at 64 or 96. The spectral-energy allocator uses the
current sequence's state, making it a non-deployable diagnostic.

The pre-registered noninferiority gate required mean KL at most `0.001`, maximum
panel p95 KL at most `0.01`, every natural/structured panel mean at most `0.002`,
and an upper paired delta-NLL bound at most `0.01` nat/token. Rank zero also had
to be at least ten times worse, proving that the suffix really depended on the
prefix state.

## Result

Eight disjoint WikiText validation windows at 2,048 prompt tokens were the
held-out panel. Each also had a 64-token-block shuffle preserving its token
histogram. The table reports the natural held-out windows.

| State representation | Representable recurrent state | Representable total cache | Ideal total saving | Mean KL | Worst sequence p95 KL | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Dense control | 18.00 MiB | 42.84 MiB | — | 0 | 0 | reference |
| Uniform rank 32 | 9.04 MiB | 33.88 MiB | 20.9% | 0.00404 | 0.01379 | fail |
| Unequal avg 32, cap 64 | 9.04 MiB | 33.88 MiB | 20.9% | 0.00162 | 0.00479 | fail mean gate |
| Uniform rank 40 | 11.29 MiB | 36.14 MiB | 15.7% | 0.00241 | 0.00782 | fail |
| Unequal avg 40, cap 64 | 11.29 MiB | 36.14 MiB | 15.7% | 0.00115 | 0.00419 | fail mean gate |
| Uniform rank 48 | 13.55 MiB | 38.40 MiB | 10.4% | 0.00183 | 0.00644 | fail |
| Unequal avg 48, cap 64 | 13.55 MiB | 38.40 MiB | 10.4% | 0.00111 | 0.00337 | fail mean gate |
| Uniform rank 64 | dense is smaller | 42.84 MiB | 0% | 0.00112 | 0.00413 | no storage edge |

Deleting all recurrent state produced mean KL **0.7748**, over two orders of
magnitude worse than the compressed variants. The probe was sensitive.

Unequal allocation is a real effect: at the same average-rank-32 storage it
improved mean KL by about **2.5x** over equal ranks. Its median allocated rank
was roughly 22, while its 90th percentile hit the cap of 64. Some heads need to
grow while many can shrink.

That effect did not produce the required edge. The best sub-dense result still
missed the pre-registered mean-KL gate, and moving from average rank 40 to 48
barely helped. Raising the cap from 64 to 96 reduced Frobenius reconstruction
error but worsened causal KL from `0.00162` to `0.00206` at average rank 32.
Singular energy is therefore not a reliable importance measure.

Repeated handcrafted prose, code, binding, and periodic panels had made rank 32
look nearly lossless through 8K context. The disjoint held-out panel falsified
that optimistic result, which is why the repeated panels are calibration only.

## Structural boundary

This intervention performs a one-time oracle SVD. It is not an efficient
low-rank recurrence. Every DeltaNet token applies another rank-one correction,
so a permanently capped state requires truncation, orthogonalization, or a
different closed update rule. Those operations add compute, workspace, and a
new numerical path. Unequal ranks also require ragged or bucketed GPU kernels.

All byte reductions above are theoretically representable at the 2K prompt
boundary, not achieved GPU savings. The harness retains dense cache tensors,
the SVD itself adds work, and subsequent updates regrow rank.

The mechanism is not novel.
[*State Rank Dynamics in Linear Attention LLMs*](https://arxiv.org/abs/2602.02195)
independently reports stable rank stratification across linear-attention heads
and uses joint rank/norm pruning to reduce cache overhead. Our stricter causal
distribution test independently recovered the heterogeneity but did not find a
lossless sub-dense operating point.

## Evidence boundary

- Supported: Qwen3.5 recurrent heads use highly unequal effective state rank;
  equal allocation is not optimal under an oracle intervention.
- Not supported: a smaller recurrent state with unchanged frozen-model
  behavior, a fast permanently bounded-rank update, denser knowledge, or a new
  architecture frontier.
- Decision: retain this as anatomy evidence and close the branch. Do not call
  it candidate 004.

## Artifacts

- `experiments/qwen35_hybrid_state_anatomy.py`
- `results/qwen35-anatomy-512.json`
- `results/qwen35-anatomy-2048.json`
- `results/qwen35-anatomy-8192.json`
- `results/qwen35-heldout-2048.json`
- `results/qwen35-heldout-2048-hetero.json`
- `results/qwen35-heldout-2048-hetero-cap96.json`
- `results/qwen35-heldout-2048-rank-boundary.json`
