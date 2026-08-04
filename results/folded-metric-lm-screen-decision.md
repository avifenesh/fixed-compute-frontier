# Folded learned-metric LM screen — decision

Decision: **close this exact recipe; do not replicate or tune it.**

The candidate produced the best checkpoint and passed every integrity,
stability, statistical, adaptation, and serving-equivalence check. It failed the
one frozen effect-size gate that separated learned geometry from the strongest
cheap control.

## Terminal result

| Arm | Validation NLL |
|---|---:|
| Baseline AdamW | 5.528750636 |
| Per-matrix LR-matched AdamW | 5.518858504 |
| Fixed metric | 5.519534521 |
| Learned metric | **5.516058151** |

Learned metric improved NLL by 0.229572% over baseline, 0.062983% over fixed
metric, and 0.050742% over LR-matched. All three paired validation-batch 95%
intervals favored learned metric. The decisive learned-minus-LR interval was
`[-0.00335321, -0.00224750]` NLL.

The preregistered rule required at least 0.1% improvement over LR-matched.
Learned metric achieved 50.7% of that threshold and missed it by 0.0027185 NLL.
It therefore does not advance to another seed.

## What the result teaches

Static per-matrix LR scaling explains 77.9% of the learned arm's absolute NLL
improvement over baseline. Learning `A` did create a real additional trajectory:
learned beat fixed metric on every terminal validation batch, median `A` movement
reached 0.724, and median factor/base cosine reached 0.251 versus 0.181 for
fixed `A`.

That residual is not identified as superior matrix geometry. The median
effective/base update norm in the learned arm grew from 1.046 at step 1 to
1.088 at step 1,525, so an adaptive step-size ramp remains a fatal explanation.
The next optimizer-side proposal must control time-varying per-matrix update
norms or prove a directional effect that those norms cannot reproduce.

## Cost and claim boundary

Serving equivalence is exact. Every arm disk-serialized the same factor-free
37,758,336-parameter state dict, reloaded into the ordinary dense model, and
produced bitwise-equal BF16 logits and equal loss. Served parameter bytes,
dense MACs, branches, and KV state are unchanged.

Training cost is not unchanged. Learned metric took 199.27 seconds versus
174.47 for LR-matched, a 14.2% wall-time increase. The implementation retained
18,284,544 persistent extra bytes, including diagnostic initial-`A` copies, and
temporarily materialized 75,497,472 bytes of old dense products per optimizer
step. Those implementation costs are removable but real for this run.

The paired intervals quantify validation-batch uncertainty for one trained
checkpoint per arm; they do not establish seed robustness. Only language NLL
was measured. The permitted claim is therefore narrow:

> On this one-seed 50M-token screen, training-only learned factors selected a
> plain dense checkpoint with 0.2296% lower NLL than AdamW and 0.0507% lower NLL
> than a first-step norm-matched LR control, at 14.2% extra training wall time
> and zero serving overhead.

This closes rank 16, scale 1, equal factor LR, all-QKV/O/FFN targeting under the
frozen AdamW schedule. It does not establish smarter geometry, denser knowledge,
training-compute efficiency, downstream capability, replication, or novelty.

Result SHA-256:
`1994f3fe210017c01f6f363fcb5f83d4cd363d5adb2b4f790df91036ac7945dc`.
