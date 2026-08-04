# Phase-elastic v3 50M decision

Status: **exact-packed quality/storage candidate passed; physical cost pending**  
Date: 2026-07-28

## Outcome

Every frozen quality, phase-transition, branch-value, artifact, and storage gate
passed on fresh seed 9901 after 49,971,200 tokens per arm.

| Measure | Dense BF16 | Packed candidate | Relative candidate edge |
|---|---:|---:|---:|
| Full-sequence endpoint NLL | 5.445655 | 5.411612 full | +0.6251% |
| Base-only endpoint NLL | 5.445655 | 5.432432 base | +0.2428% |
| Five-boundary mixed suffix | paired reference | packed mixed | +0.6194% |
| Actual cached P256/P480 suffix | paired reference | packed mixed | +0.6288% |

The optional branch itself improved over the same packed base by:

- 0.3976% on the one-pass mixed suffix grid;
- 0.4039% on actual BF16 cached execution.

Every populated first-token and decode-offset bucket improved versus both dense
and the candidate's base-only path.  Thus the result is not an aggregate hiding
a local regression, and it cannot be attributed only to quantized-base
regularization.

## Exact served artifact

- BF16 dense FFN reference: 2,359,296 bytes/layer.
- Packed candidate: 2,338,816 bytes/layer.
- Saving: 20,480 bytes/layer.
- INT8-plus-ternary base: 1,484,288 bytes/layer.
- W4 branch: 854,528 bytes/layer.
- Training shadows/log scales after reload: none.
- Pre-export versus independent packed reload max bucket-loss difference: 0.0.
- Artifact SHA-256: `d7bb0164c71ee0c01cb80dd68711302a2776d86fc1fcd99bb25143c6c8965402`.

The candidate needed 244.03 seconds of training time versus 152.67 seconds for
dense in this prototype.  Offline training compute was not the protected
resource; served bytes, quality, and eventual phase latency are.

## Surviving claim

The result establishes a new resource allocation mechanism at this scale:

> Store a high-quality low-bit base used in every phase, and spend the remaining
> weight bits on a wider branch used only in memory-bound decode.  Train with a
> shared tokenwise phase boundary so a base-produced prompt cache is consumed by
> the full decoder.

This packs more learned features into slightly fewer FFN bytes and improves the
actual mixed-cache language objective across two seeds and a 50M horizon.

## Not yet established

The full decode path performs 2.4375x as many matrix coordinates.  A roofline
proof shows this can fit H100 decode slack below about 120 simultaneous tokens,
but only a fused physical kernel can establish equal serving cost.  The H100
instance exited before that gate, and Vast rejected a replacement for lack of
account credit.  Do not call this an end-to-end Pareto win until packed-kernel
latency, numerical agreement, workspace, and peak VRAM pass.

The pending physical protocol now makes the 12-layer weight stream primary,
tests warm and 128 MiB-flushed L2 conditions, interleaves paired trials, uses
the faster of fused Triton and packed cuBLAS dense controls, disqualifies a
dense denominator more than 5% slower than the prior H100 snapshot, allocates
and audits all 12 monolithic blobs, and rejects compiler spills/local memory.
It cannot run in decision mode until a resumed H100 passes compiler smoke and
the source, protocol, and imported helpers are hash-pinned.

Evidence:

- `results/phase-elastic-v3-50m-phase-faithful.json`
- `results/phase-elastic-roofline-proof.md`
- `experiments/phase_elastic_packed_h100.py`
