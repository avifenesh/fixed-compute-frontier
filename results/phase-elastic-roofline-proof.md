# Phase-elastic FFN: exact H100 roofline before the physical kernel

Status: **mathematical feasibility proof, not a latency result**  
Date: 2026-07-28

## Exact served objects

For one `D=384`, width-1,024 BF16 SwiGLU layer:

- matrix coordinates: `3*384*1024 = 1,179,648`;
- persistent weights: `2,359,296` bytes;
- matrix FLOPs per token: `2*1,179,648 = 2,359,296`.

For the phase-elastic layer:

- always-on base payload: `1,484,288` bytes;
- full base-plus-branch payload: `2,338,816` bytes;
- full matrix coordinates: `3*384*(1024+1472) = 2,875,392`;
- full matrix FLOPs per token: `5,750,784`;
- full coordinate-MAC ratio: `2.4375x`;
- persistent saving: `20,480` bytes/layer, or `245,760` bytes over 12 layers.

## H100 bound

Use 3.35 TB/s HBM bandwidth and 989 TFLOP/s dense BF16 Tensor Core throughput
(half NVIDIA's 1,979 sparse-marketing figure).  The ridge point is

`989e12 / 3.35e12 = 295.224 FLOP/byte`.

Ignoring the small activation/scaling terms, arithmetic intensity at `R`
simultaneous tokens is:

- dense BF16: `R` FLOP/byte;
- packed base: `1.5895 R` FLOP/byte;
- packed full path: `2.4589 R` FLOP/byte.

Therefore the ideal memory-bound regions end near:

- dense BF16: `R=295`;
- packed base: `R=186`;
- packed full path: `R=120`.

The optional branch can consequently add 2.4375x raw matrix arithmetic without
raising the *weight-only* lower bound below about 120 tokens.  Once the minimal
activation workspace store/read is included, the ideal full/dense ratios at
rows `{1,8,32,128,512,2048}` are approximately
`{0.99,1.01,1.07,1.24,1.90,2.44}`.  The defensible no-cost target is therefore
decode batch 1-8; batch 32 is a near-cost target, and 128+ is not no-cost even
ideally.  The precise resource exchange remains: **unused decode Tensor Core
capacity is converted into learned feature capacity while resident weight bytes
remain fixed.**

## Per-layer lower bounds

| Rows | Dense BF16 | Packed base | Packed full |
|---:|---:|---:|---:|
| 1 | 0.704 us | 0.443 us | 0.698 us |
| 8 | 0.704 us | 0.443 us | 0.698 us |
| 32 | 0.704 us | 0.443 us | 0.698 us |
| 128 | 0.704 us | 0.443 us | 0.744 us |
| 512 | 1.221 us | 1.221 us | 2.978 us |
| 2,048 | 4.886 us | 4.886 us | 11.908 us |

These are `max(weight_bytes/bandwidth, matrix_flops/throughput)`, not predicted
wall times.  They omit launch, dequantization, activation traffic, occupancy,
register pressure, and imperfect bandwidth/compute utilization.

## Workspace boundary

A combined base-plus-branch intermediate uses `2,496*R*2` bytes versus
`1,024*R*2` for dense, an extra `2,944R` bytes.  The 12-layer persistent saving
pays that difference through `floor(245,760/2,944)=83` simultaneous decode
tokens.  Thus rows 1, 8, and 32 remain no-larger even with the simple combined
workspace.  A sequential/reused branch workspace can extend the boundary.

The 18 logical code/scale tensors cannot remain separate CUDA allocations:
4-KiB allocator rounding consumes the whole 20,480-byte layer slack.  The
production layout must be one monolithic 128-byte-aligned blob.  Every natural
component size is already divisible by 128, so the exact 2,338,816-byte layout
needs no padding.  Padding the width-1,472 branch to 1,536 would add 36,864 code
bytes and is forbidden.

## Dense reference captured before the H100 exited

CUDA-event mean per replay for CUDA-graphed PyTorch BF16
`linear([gate;up]) -> SwiGLU -> down` was:

| Rows | Dense graph time |
|---:|---:|
| 1 | 13.933 us |
| 8 | 19.141 us |
| 32 | 19.596 us |
| 128 | 22.028 us |
| 512 | 24.747 us |
| 2,048 | 39.900 us |

These snapshot values are not a final fused-CUTLASS control; the candidate must
be remeasured beside the strongest same-clock dense kernel.

## What remains unproved

The roofline gives a nonempty feasible region; it does not prove an implementable
kernel reaches it.  The physical gate must:

1. reconstruct INT8-plus-ternary and signed W4 operands inside the GEMM
   mainloop without materializing dense weights.  INT8 and ternary must combine
   in registers before one BF16 MMA; separate GEMMs make the full ratio 3.4375x
   and invalidate the claim;
2. fuse gate, up, SwiGLU, and base/branch selection into no more launches than
   the strongest dense control;
3. reproduce the packed oracle numerically;
4. measure prefill, decode, combined workspace, peak VRAM, and a prompt-plus-
   generation service cell.

The retained Vast H100 exited before this physical gate completed, and a new
instance could not be rented because the Vast account had no credit.  No
latency pass is inferred from this proof.

Sources: [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/),
[CUTLASS mixed-input Hopper support](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html).
