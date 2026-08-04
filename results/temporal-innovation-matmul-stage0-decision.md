# Temporal-innovation dense projection — Stage 0 decision

Status: **rejected as an exact stack and as a free frozen-model transform**  
Date: 2026-07-26  
Decision: `reject_exact_and_pretrained_slack`; no GPU rental

## What was gained

For an arbitrary dense learned matrix, the identity

\[
W x_t = W x_{t-1} + W(x_t-x_{t-1})
\]

is exact.  If the input innovation has `k` nonzeros, one can read `k` columns
of `W` rather than all `D`, without restricting the rank or learned degrees of
freedom of `W`.  The FP32 executable check used a 5.21%-sparse innovation and
matched dense recomputation at relative L2 error `6.91e-8`.

RMSNorm is not an obstruction.  Its changing global scale can be applied to
the cached old projection while only the sparse unnormalized innovation is
projected.  That form matched at relative error `1.32e-7`.

This is the useful algebraic remnant.  It is a conditional systems primitive,
not a new model architecture.

## Why the broad method fails

The sparsity does not compose through arbitrary dense learned maps.  For fixed
nonzero `delta` and continuously distributed dense `W`, each coordinate of
`W delta` is zero only on a measure-zero hyperplane.  Therefore `W delta` is
dense with probability one.  The executable witness turned five changed input
coordinates into changes in all 80 outputs.  A smooth nonlinearity such as
SiLU does not create exact zeros afterward.

An exact event-driven stack must consequently give up at least one desired
property: unrestricted dense weights, exactness, or sparse propagation.
Deadband/quantization is the least restrictive escape, so it received the
frozen-model measurement gate.

## Frozen-model measurement

The exact captured checkpoint was
`HuggingFaceTB/SmolLM2-360M@f8027fd0eaeea54caa13c31d31b9fdc459c38b49`;
`model.safetensors` matched the pre-existing SHA-256
`7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f`.
The CPU FP32 pass observed Q/K/V/O and gate/up/down projection inputs at layers
0, 8, 16, 23, and 31 over four fixed 192-token lanes.  There were 2,240
projection-transition records.

| representation | attention changed coordinates | FFN changed coordinates | attention projection distortion | FFN projection distortion |
|---|---:|---:|---:|---:|
| exact FP32 | 100.00% | 100.00% | 0 | 0 |
| oracle per-coordinate int8 | 98.96% | 98.75% | 0.499% | 0.812% |
| oracle per-coordinate int6 | 95.83% | 95.31% | 1.982% | 3.238% |
| oracle per-coordinate int4 | 82.19% | 80.73% | 9.357% | 12.743% |

The table reports medians.  Even int4, already causing large local distortion,
changes roughly four fifths of all codes instead of the required one tenth.
The int8 p90 changed-code fractions were 99.48% for attention and 99.37% for
FFN.  Quantization preserved values reasonably at eight bits but did not
produce events; lower bit widths reduced events too little and increased
error.

Directly keeping the largest 10% of raw activation changes was also far outside
the gate: median relative projected-delta error was 59.54% in attention and
63.33% in FFN, versus a 5% limit.  Keeping half the coordinates still left
18.97% and 21.75% median error.  The `down_proj` input was exactly dense, so the
FFN's dense expansion, gating, and nonlinearity did not preserve an exact
sparse event stream.

## Physical ledger

The weights are unchanged, but incremental evaluation needs previous inputs
and outputs.  Sharing identical Q/K/V and gate/up inputs, the naïve BF16 state
is 28,160 bytes per layer.  Across 32 layers that is 901,120 bytes per request
(`0.8594 MiB`).  Crediting K/V outputs already present in the normal KV cache
still leaves a lower bound of 860,160 bytes (`0.8203 MiB`).

Those figures omit changed-index encoding, dense output reads/writes, kernel
launches, and support divergence across requests.  Arbitrary changed columns
are gathers, not a dense Tensor-Core GEMM.  More importantly, the optimistic
logical weight-read fractions measured here are 81%-100%, before any gather
penalty.  There is no plausible H100 gate to run.

## Collision and evidence boundary

[Sigma-Delta Quantized Networks](https://arxiv.org/abs/1611.02024) already
propagates discretized changes between layers, and
[Delta Activation training](https://arxiv.org/abs/2107.07305) explicitly
promotes temporal activation sparsity.  The family is prior art; our useful
addition to this ledger is the exact RMSNorm form and the language-model
fan-out/slack measurement.

Supported:

- exact sparse-delta updates can reduce one arbitrary dense projection when a
  genuinely sparse innovation is supplied;
- RMSNorm can be folded into that exact update;
- ordinary frozen LLM activations contain no useful inherited temporal-event
  slack at the tested projections.

Not supported:

- an exact event-sparse stack of unrestricted dense maps;
- a no-retraining LLM speedup;
- a quality-preserving quantized-event path;
- H100 latency, throughput, energy, or cost improvement.

Explicit event-sparsity training is not mathematically disproved.  It would be
a different architecture that pays a sparsity objective and approximation
error, and it still needs a mechanism that prevents every dense map from
re-densifying the next event stream.  The present result gives no reason to
spend a training or GPU budget on that repair.

## Artifacts

- `experiments/temporal_innovation_matmul_gate.py`
- `tests/test_temporal_innovation_matmul_gate.py`
- `results/temporal-innovation-matmul-stage0-preregistration.md`
- `results/temporal-innovation-matmul-stage0-screen.json`

