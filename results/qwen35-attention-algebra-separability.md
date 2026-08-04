# Qwen3.5 full-attention algebra separability — negative result

Status: **global oracle families rejected**  
Date: 2026-07-23  
Model: Qwen3.5-0.8B Base, BF16  
Hardware: rented H100 PCIe 80 GB, Vast instance 45568493

## Result

None of the tested bounded-read algebra families preserved the frozen model's
output distribution at the pre-registered quality margin. This is a stronger
failure than a causal implementation failure: every candidate received exact
full-history scores, an exact prefix cache, and, where applicable, exact top-k
indices and tail mass.

| Oracle intervention | Natural mean KL | Natural worst-window p95 | Shuffled mean KL | Structured mean KL | Natural delta-NLL U95 | Overall top-1 |
|---|---:|---:|---:|---:|---:|---:|
| top-k 64 | 0.032384 | 0.237908 | 0.085013 | 0.000324 | 0.046809 | 0.8331 |
| top-k 64 + tail mean | 0.008641 | 0.055791 | 0.033581 | 0.012306 | 0.003728 | 0.9227 |
| uniform value moment | 1.252260 | 8.063667 | 0.584956 | 2.870938 | 1.473762 | 0.4910 |
| first-order KV moment | 0.939995 | 5.764158 | 0.288367 | 0.941171 | 1.144450 | 0.6225 |
| top-k 256 | 0.006327 | 0.041855 | 0.018019 | 0.000225 | 0.015626 | 0.9219 |

The limits were natural mean KL `0.001`, natural worst-window p95 `0.01`,
shuffled and structured mean KL `0.002`, delta-NLL upper bound `0.01`, and
overall top-1 agreement `0.99`. Only `k <= 64` survived the resource screen;
top-k 256 therefore fails on both quality and budget.

The worst individual case mean KL was `0.125098` for top-k 64, `0.055518` for
tail-mean 64, and `0.022983` for top-k 256. The global averages do not hide a
passing worst case.

## What the algebra measurement says

Across 912 captured `(case, layer, query-head)` cells:

- mean attention effective support was `232.3` tokens;
- the top 64 retained only `67.19%` of probability mass on average;
- renormalized top-64 changed the local attention output by `15.24%` in
  relative norm;
- adding one exact-mass uniform tail summary reduced that to `8.13%`, still
  far above the end-to-end tolerance;
- top 256 retained `85.83%` of mass, yet still failed the output gate.

So the missing information is not adequately represented by one background
average. The omitted values remain query-relevant and heterogeneous. A
zeroth- or first-order fixed moment is even less sufficient.

There is real internal heterogeneity, but it does not rescue the global claim.
For example, layer 7's second KV group retained about `87%` mass in top 64,
while several groups retained less than `60%`; layer 23 had relatively small
local output errors despite diffuse mass. A separately pre-registered
calibration-only KV-group localization follows this result. It is not a
candidate admission and cannot turn already-opened windows into held-out data.

## Harness validity and limits

- The custom exact hook matched the library attention with KL `0` and top-1
  agreement `1.0`.
- Zeroing full-attention outputs produced mean KL `1.6399` natural, `1.0102`
  shuffled, and `3.3220` structured, confirming that the measured layers
  matter.
- Bottom-64 attention was similarly destructive.
- Eight disjoint WikiText validation windows, eight corresponding block-
  shuffled controls, and three seeded structured cases used 2,048-token
  prompts with 64 teacher-forced continuation tokens.
- The reported delta-NLL interval treats continuation tokens as independent.
  A reviewer correctly flagged this as optimistic relative to a paired
  sequence bootstrap. The observed failures are tens to thousands of times
  too large for that correction to reverse the verdict.
- Prefix computation and caches remain exact. All variants retain full KV and
  compute full score matrices. Runtime and memory figures from these oracle
  kernels make no efficiency claim.

Raw artifact:
[`qwen35-algebra-heldout-2048.json`](qwen35-algebra-heldout-2048.json), SHA-256
`ae2d973f65c5770dcf0f0281c68d39ed0e5299ce6e96d7939e6016e9400668fb`.
The frozen gate is in
[`qwen35-attention-algebra-preregistration.md`](qwen35-attention-algebra-preregistration.md).

## Decision

Reject global tropical/top-k, top-k-plus-one-tail-summary, and first-order
moment replacement for this frozen hybrid. Do not spend a causal selector or
kernel implementation on these global forms. The only allowed follow-up is
localization of which physical KV groups cause the failure; any combined
assignment must be frozen on calibration data and evaluated simultaneously.
