# Fused Feedback-SwiGLU H100 decision

Status: **cross-feature feedback branch closed**.

The materially different fused Triton executor preserved zero-feedback output exactly and ran the full valid frozen grid, but it failed every latency gate:

| Tokens | Same width / baseline | Equal parameter / baseline |
|---:|---:|---:|
| 1 | 1.1121x | 1.1177x |
| 8 | 1.1121x | 1.1121x |
| 32 | 1.2048x | 1.2031x |
| 128 | 1.7432x | 1.7409x |

The group reductions and low-occupancy per-token/group programs cost far more than their scalar-MAC count suggests. The cost worsens with token batch, showing that this is not merely Python or kernel-launch overhead.

No language-quality evidence exists that could justify an 11% to 74% FFN latency increase. Retain the algebraic observation and exact endpoint, but do not train or optimize this branch further. The next branch removes cross-feature reduction entirely and tests per-neuron self-refinement.
