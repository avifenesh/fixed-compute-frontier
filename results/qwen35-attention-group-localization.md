# Qwen3.5 full-attention KV-group localization — negative result

Status: **stopped at the frozen individual-unit gate**  
Date: 2026-07-23  
Calibration artifact SHA-256:
`415fc62415f4b352116a3a942b3a494bf4a66782874193a66b63ff9c855b1957`

## Result

Zero of the 12 physical `(full-attention layer, KV group)` units passed the
pre-registered individual localization score `s <= 1`. Nine passing units were
required before a simultaneous 9-of-12 oracle intervention was allowed, so the
experiment stops without running or selecting a combination.

| Unit | Best family | Score `s` | Limiting criterion | Natural mean KL | Shuffled mean KL | All-panel top-1 |
|---|---|---:|---|---:|---:|---:|
| L7 G1 | tail-mean 64 | 3.125 | top-1 | 0.000721 | 0.000680 | 0.968750 |
| L3 G0 | tail-mean 64 | 3.207 | top-1 | 0.000861 | 0.000506 | 0.967928 |
| L3 G1 | tail-mean 64 | 3.701 | top-1 | 0.000936 | 0.000887 | 0.962993 |
| L15 G0 | tail-mean 64 | 3.701 | top-1 | 0.001504 | 0.000869 | 0.962993 |
| L11 G1 | tail-mean 64 | 3.947 | top-1 | 0.001195 | 0.000755 | 0.960526 |
| L7 G0 | tail-mean 64 | 4.112 | top-1 | 0.001359 | 0.001142 | 0.958882 |
| L19 G1 | top-k 64 | 4.276 | top-1 | 0.003265 | 0.006000 | 0.957237 |
| L11 G0 | tail-mean 64 | 4.441 | top-1 | 0.001784 | 0.002298 | 0.955592 |
| L23 G0 | tail-mean 64 | 4.441 | top-1 | 0.000676 | 0.004357 | 0.955592 |
| L15 G1 | tail-mean 64 | 4.934 | top-1 | 0.000961 | 0.003574 | 0.950658 |
| L19 G0 | tail-mean 64 | 6.066 | shuffled KL | 0.001673 | 0.012132 | 0.956414 |
| L23 G1 | tail-mean 64 | 8.059 | top-1 | 0.001110 | 0.009278 | 0.919408 |

The score normalizes seven frozen constraints: natural, shuffled, and
structured mean KL; natural token-KL p95; natural delta-NLL upper bound;
all-panel top-1 disagreement; and worst-case mean KL. A score above one fails.

## Interpretation

The damage is distributed rather than isolated to three indispensable KV
groups. Several individual substitutions have low mean KL, but even the best
one changes `3.125%` of top predictions, versus the `1%` allowance. Keeping the
three worst groups exact therefore cannot meet the pre-registered resource
strategy by simply replacing the other nine.

`tailmean64` was the least damaging family for 11 of 12 groups. `topk64` won
only for L19 G1. Neither the uniform nor first-order moment was best for any
unit; their best normalized scores ranged far above the sparse families. This
supports the global result: one undifferentiated background statistic does not
preserve the omitted query-relevant values.

This does not prove that a newly trained architecture cannot organize its
state differently. It rejects post-hoc algebraic replacement of this frozen
Qwen hybrid under these families and thresholds. Continuing would require a
new jointly trained algebra with its own claim and budget, not threshold
repair or post-hoc selection on these calibration windows.

Raw artifacts:

- [`qwen35-group-localization-2048.json`](qwen35-group-localization-2048.json)
- [`qwen35-group-localization-2048.log`](qwen35-group-localization-2048.log)
- [`qwen35-attention-group-localization-preregistration.md`](qwen35-attention-group-localization-preregistration.md)

The one-unit hook was first checked on a short code-path smoke test; its exact
control remained identical and targeted perturbations were smaller than the
corresponding global intervention.
