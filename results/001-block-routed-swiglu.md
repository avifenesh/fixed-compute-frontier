# 001 — Block-routed SwiGLU

Status: **closed negative result**  
Date: 2026-07-22  
Primary attempted edge: A/C/E

## Claim tested

Input-dependent pairings of existing SwiGLU gate/up channels might expose many
structured feature combinations while retaining the same dominant MLP matrix
multiplications and nearly the same serving latency.

## What survived

- A 64-channel cyclic-shift router was GPU-feasible on an H100 PCIe.
- BF16 correctness passed against PyTorch references.
- Interleaved full-block latency overhead averaged 0.73% at 512 tokens and
  0.83% at 4096 tokens.
- CUDA graph capture worked and peak allocation did not increase.

## What failed

In a corrected equal-parameter, five-seed structured-composition test:

- baseline held-out accuracy: 66.06%;
- routed held-out accuracy: 63.38%;
- paired mean change: -2.67 percentage points.

On the random-lookup control, routing raised familiar-pair training accuracy
from 73.13% to 81.54%, while held-out accuracy stayed at chance (50.28% versus
50.42%). It improved memorization, not reusable knowledge.

## Knowledge retained

1. A large count of route identities is not a large count of independently
   stored features.
2. A strict function-class superset does not imply that optimization will find
   a better generalizing solution.
3. Kernel feasibility and capability gain are independent claims and must be
   gated separately.
4. Identity containment is not protection against training degradation.
5. Structured holdouts plus random-lookup controls prevented a memorization
   gain from being misreported as knowledge density.
6. This branch must not be continued by changing routing rules; a materially
   different mechanism requires a new claim and a new numbered entry.

## Evidence

Implementation and raw artifacts remain in
`~/projects/virtual-feature-mlp/` (local, unpublished):

- `benchmark_interleaved_h100_b64.json`
- `structured_factor_results.json`
- `paired_swiglu.py`
