# Gauge-exposed partner-feedback SwiGLU — H100 preregistration

Status: **frozen before execution**.

This is a target-shape deployed-operator feasibility gate, not capability
evidence or production-runtime parity.

## Frozen paths

At BF16 `D=4096,M=14336`, both arms run the same full path with separate gate
and up GEMMs, a fused activation kernel, and the down GEMM:

1. ordinary SwiGLU;
2. partner feedback using adjacent-pair involution, with `lambda` decoded from
   `U[0,0]` inside the activation kernel.

The candidate kernel may recompute the partner's ordinary SwiGLU scalar.  It
may not allocate `z0`, a route table, a coefficient tensor, or another
activation-sized workspace.  It issues one common carrier-pivot load per
kernel program, not per feature.

## Protocol

- one H100; token batches `1,8,32,128`;
- at least 40 graph warmups and 200 randomized-order CUDA-event samples;
- captured path includes all three GEMMs and the activation kernel;
- nonzero correctness reference exercises negative-clipped, unclipped, and
  positive-clipped partner activations;
- report absolute timing distributions, coefficient decode error, logical
  added activation loads/operations, and workspace.

## Gates

1. At encoded `lambda=0`, maximum row-relative full-FFN error versus ordinary
   fused SwiGLU is at most `0.2%` in every cell.
2. Nonzero fused activation matches a Torch float32 reference within `1%`
   maximum row-relative error.
3. BF16 lambda decode absolute error is at most `0.002` at frozen
   `lambda=0.08`, `carrier_gain=4`.
4. No duplicate parameter tensor, route metadata, serial feature dependency,
   or additional activation workspace.
5. Candidate/baseline median is at most `1.02x` at batches 1 and 8 and at most
   `1.05x` over the full grid.

Failure closes this exact fused partner-feedback recipe before language
training.  Passing authorizes only a frozen matched LM screen.  Because both
arms use separate projection GEMMs, a pass still requires later validation
against a production coalesced gate/up executor.

