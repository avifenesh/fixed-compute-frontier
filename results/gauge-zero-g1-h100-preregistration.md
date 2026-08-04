# Gauge-zero G1 H100 preregistration

Status: frozen before the formal run  
Date: 2026-07-27

## Claim under test

A norm-free RoPE/GQA K pair has a commuting rotation gauge. For RoPE pair
`p`, the deterministic pivot is physical K column `p`. The compiler rotates
that pivot to `(radius, 0)` and applies the same orthogonal rotation to every
sharing Q pair. A zero pivot already satisfies the constraint and is left
unchanged. The ordinary score function is unchanged in real arithmetic.

The odd pivot is therefore exactly zero at initialization. It remains an
ordinary dense K weight, but the candidate also reads that same physical BF16
weight `a` as `c = a / tau` and applies, before RoPE,

`odd <- odd + c * even^2`.

The shear target is the same row that contains `a`; the squared source does
not depend on `a`. Ordinary attention is exactly contained at `a = 0`. The
candidate has one new quadratic tangent direction while preserving the dense
Q/K tensor shapes, parameter values, optimizer tensors, and KV coordinates.

This gate tests only the serving executor. It does not test LM quality or
claim low-bit quantization support.

## Frozen implementation and controls

- GPU: NVIDIA H100 80GB HBM3.
- dtype: BF16 weights, activations, and outputs; FP32 epilogue arithmetic is
  rounded back to BF16 before RoPE storage.
- target shape: hidden 4096, 32 Q heads, 8 KV heads, head dimension 128,
  fused QKV width 6144.
- rows: 1, 8, 32, 128, 512, and 2048.
- timing: 400 randomized matched trials per cell after 20 warmups.
- cache condition: a 128 MiB buffer is touched immediately before every timed
  call, outside its CUDA-event interval, so the 50,331,648-byte QKV weight is
  cold for both arms.
- control: ordinary `torch.mm` with the same canonical dense QKV tensor,
  followed by the same fused RoPE kernel with the shear disabled.
- candidate: the identical `torch.mm`, followed by the fused RoPE+G1 kernel.
- simulated trained coefficients: deterministic BF16 Gaussian values with
  standard deviation 0.02 stored in the ordinary odd-pivot K weights. The
  control also consumes those values in its dense K GEMM; only the quadratic
  reuse is disabled.
- pivot rule: column equals RoPE pair index. It is derived from shape and
  needs zero model, GPU-buffer, or code-carried index bits.
- bootstrap: 5,000 matched resamples of the median candidate/control CUDA
  event ratio.

The development result is excluded from the formal decision.

The latency claim is deliberately limited to incremental overhead relative to
this exact minimal normal-GEMM plus one-pass RoPE executor. The control kernel
must also be bit-exact to an independent PyTorch RoPE reference. A pass is not
a claim that this executor is the fastest possible production baseline; a
full production-block comparison remains a later admission gate.

## Frozen gates

All gates must pass:

1. Candidate and control outputs are each bit-exact to their independent BF16
   PyTorch references in every cell.
2. Candidate/control median latency ratio is at most 1.02 in every cell.
3. Bootstrap 95% upper ratio is at most 1.03 in every cell.
4. Bootstrap 95% upper ratio is at most 1.02 for rows 1 and 8.
5. Both arms use the same 50,331,648-byte dense QKV tensor and the same output
   bytes; no coefficient sidecar or packed-matmul decoder is present.
6. Pivot metadata is explicitly reported as zero bits and the deterministic
   rule is recorded.

If any gate fails, zero-pivot G1 is not admitted to the LM quality pilot under
the no-served-cost claim. Thresholds and cells will not be changed after the
formal result is observed.

## Already-required algebra and serialization tests

The attention test suite must pass before execution, including split-half
RoPE/GQA containment, exact parameter/tensor counts, eager and SDPA parity,
BF16 cache equality, and the one-update fatal round trip. In that round trip,
all FP32/chart state is discarded; ordinary BF16 Q/K/V/O plus the fixed pivot
rule must reproduce the coefficient, full output, and BF16 cache exactly.

## Scope after a pass

A pass permits a matched 10M-token from-scratch LM pilot with raw baseline,
canonical dense bilinear control, and gauge-zero G1. It does not by itself
establish capability gain, novelty, low-bit quantization, tensor-parallel
compatibility, or a full production kernel.
