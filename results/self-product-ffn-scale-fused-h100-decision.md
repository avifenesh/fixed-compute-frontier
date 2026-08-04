# Self-product FFN fused H100 decision

## Decision

**Close the self-product FFN as a fixed-served-cost frontier. Do not run the
100M-token scale experiment and do not repair this branch under the same
claim.**

The candidate retained the small-model quality finding and achieved exact
static BF16 bytes, exhaustive finite-BF16 activation equivalence, and
bit-identical prompt/next-token logits and K/V caches. Fair activation fusion
also made it faster in four of five serving cells. It nevertheless failed the
precommitted all-cell latency and memory envelope.

## Frozen result

All arms contain exactly 102,247,040 parameters / 204,494,080 BF16 parameter
bytes. The self-product performs 2,688 SiLUs and pointwise products per layer
token versus 1,792 for either SwiGLU baseline.

Candidate median wall-time ratios:

| Cell | versus split SwiGLU | versus packed-input SwiGLU |
|---|---:|---:|
| decode 1 x 512 + 1 | 0.9603 | 0.9798 |
| decode 8 x 512 + 1 | 0.9604 | 0.9724 |
| prefill 1 x 512 | 0.9886 | 0.9727 |
| prefill 8 x 512 | 0.9460 | 0.9721 |
| prefill 32 x 512 | **1.0295** | **1.0424** |

CUDA-event ratios agreed. At batch-32 prefill the entire 95% bootstrap
interval was above 1.0 for both baselines, so this is not timing ambiguity.

The candidate used less incremental allocated memory than both baselines in
every measured cell, but the strict absolute/reserved envelope still failed
against split SwiGLU in small cells because its resident allocator floor was
higher. Peak allocation passed against both baselines at batch 32.

## Knowledge retained

- Replacing two width-M SwiGLU input banks with one width-1.5M generator can
  increase identifiable degree-two atoms and produced a repeated 0.43%-0.68%
  small-model NLL gain.
- The same shape gives a real decode/small-prefill speed advantage after fair
  fusion and reduces incremental activation memory.
- The extra 50% scalar nonlinear work becomes visible at large prefill even
  when dense weights/MACs are exactly matched. Equal dense MAC is therefore
  not equal served compute.
- Future algebraic candidates must be shaped so their new interaction is
  funded by an operation already present in the baseline, not by a wider
  pointwise domain hidden behind equal matmul accounting.

Frozen result SHA-256:
`fd8138554bbf78db72a957f192c3913511acab106626d246ad409aeea1566d56`.

