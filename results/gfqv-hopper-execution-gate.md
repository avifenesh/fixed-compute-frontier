# GFQV and K-reuse value gates — Hopper execution gate

Status: **one-kernel GFQV is feasible; K-reuse is preferred when an existing
K/V postprocess exposes pre-RoPE K**  
Date: 2026-07-26  
Decision now: **no rental until the learnability gate and target-engine fusion
boundary are proved**

## 1. GFQV exact logical ledger

For one value head of width `r`, choose a fixed contiguous pivot range `P` and
write `p=X_P`. The proposed value is

\[
V = X_{\bar P}B + p + p\odot(pG_{off}),
\qquad \operatorname{diag}(G_{off})=0.
\]

Per token/head:

| term | multiplications | additions | learned weights |
|---|---:|---:|---:|
| `X_rest B` | `(d-r)r` | `(d-r-1)r` | `(d-r)r` |
| `p G_off` | `r(r-1)` | `r(r-2)` | `r(r-1)` |
| `p * (...)` | `r` | 0 | 0 |
| sum base, identity, gate | 0 | `2r` | 0 |
| **total GFQV** | **`dr`** | **`(d-1)r`** | **`dr-r`** |
| dense `XW_V` | `dr` | `(d-1)r` | `dr` |

Thus the logical multiply/add count is exactly dense V, while `G=0` exactly
recovers the gauge-fixed linear baseline. The function family adds selected
quadratic monomials without deleting the original full-rank linear value path.

## 2. One-kernel Hopper schedule

GFQV cannot be implemented by a normal final GEMM epilogue, because the
`p G` partial must be gated before the `X_rest B` partial is added. It can be
implemented in one custom CuTe/CUTLASS kernel with one accumulator:

1. Put `P` in the first `r` hidden coordinates.
2. Store the value operand as the ordinary rectangular row sequence
   `[G_dense; B]`, with a physically stored zero diagonal in `G_dense`.
3. WGMMA the first `r` K slice into accumulator `Z`.
4. Drain the outstanding WGMMA group and apply, elementwise in the FP32
   accumulator, `Z <- fma(p, Z, p)`.
5. Continue WGMMA over the remaining `d-r` K slice, accumulating `X_rest B`
   into that same accumulator.

Q/K output-tile CTAs keep the standard full-K path; V CTAs take the boundary
transform. There is one launch, no global intermediate, one FP32 accumulator,
and the same rectangular physical weight bytes and WGMMA count as dense V.
TMA can prefetch B stages while the G WGMMA group drains, so this need not be a
second cold-started pipeline.

The physical overhead versus dense V is:

- one mid-mainloop WGMMA wait/drain;
- one accumulator FMA per output element;
- a shared/register read of `p` for every accumulator element;
- a branch/layout specialization for V output tiles.

The zero diagonal is not worth sparse execution. Dense WGMMA performs those
`r` zero products per head, just as a dense identity pivot would; trying to
skip individual diagonal entries destroys tensor-core regularity.

### Why the overhead is not automatically negligible

The scalar FLOP ratio is tiny—one FMA per value output versus roughly `d`
tensor-core MACs—but scalar issue and WGMMA issue are different resources. For
an `m64n128` FP32 accumulator, every thread owns many accumulator registers;
touching all of them can take tens of scalar instructions. The mandatory
boundary wait also interrupts a normally continuous K pipeline. Only a Hopper
measurement can establish the actual penalty.

Two alternatives are worse:

- a separate `p@G` kernel adds a launch and global intermediate;
- two simultaneous U/Z accumulators avoid the mid-loop transform but roughly
  double accumulator registers, threatening occupancy or spilling on Hopper.

Decode with small `M`, few KV heads, and split-K is the hardest case. A
split-K implementation must keep the nonlinear G contribution separate until
it has been gated; ordinary linear partial-sum reduction cannot apply the gate
to the combined G+B sum.

## 3. K-reuse refinement

The refined value is

\[
V = p + X_{\bar P}B + p\odot(\alpha\odot K_{pre}),
\]

where `K_pre` is the already-computed, pre-RoPE key for the matching KV head and
`alpha` is learned and initialized to zero.

Per token/head, relative to dense V:

| resource | dense V | K-reuse gate | saving |
|---|---:|---:|---:|
| learned weights | `dr` | `(d-r)r+r` | `r^2-r` |
| multiplications | `dr` | `(d-r)r+2r` | `r^2-2r` |
| V projection K extent | `d` | `d-r` | `r` |

`alpha=0` recovers the complete gauge-fixed linear baseline. At sequence length
one, attention returns V directly, so a nonzero gate supplies a strict
quadratic function without relying on softmax interactions.

### Preferred physical placement

The best implementation does **not** synchronize independent QKV GEMM CTAs. It
uses an already-existing postprojection kernel *only if that kernel already
consumes both pre-RoPE K and V* to apply RoPE and write the KV cache:

1. A jagged fused-QKV projection computes K with `K=d` and `V_base` with
   `K=d-r`.
2. The existing RoPE/cache-write kernel sees `K_pre` and `V_base`, reads the
   small `p` and `alpha`, applies the gate, then rotates K and writes both cache
   entries.

If this exact boundary exists, K-reuse adds no launch and no K/V global read
that the postprocess did not already require. It replaces an `r x r` V weight
slice with `r` alpha weights and a few elementwise operations. This is
physically better than GFQV's second tensor-core phase.

Current vLLM `main` exposes a nearby but not free boundary: Llama runs
`qkv_proj`, splits Q/K/V, calls `rotary_emb(positions, q, k)`, and only then
passes Q/K/V into attention. Extending that existing RoPE kernel to gate V
preserves the launch count, but baseline RoPE does not read or write V. The
extension therefore adds a V roundtrip plus reads of `p` and `alpha`. This is
small in one-token decode but material in prefill. A cache-write kernel that
already reads V would avoid that roundtrip, but it must still have access to
**pre-RoPE** K; post-RoPE K is a different architecture.

### Monolithic-QKV boundary

Independent GEMM CTAs cannot read another CTA's K accumulator. If the target
engine fuses projection, RoPE, and cache writing into one kernel, the options
are:

- assign one CTA/warpgroup a paired K+V head: run K-only output columns for the
  pivot slice, K+V columns for the rest, then gate V in a paired epilogue;
- store K to shared memory and synchronize paired warps;
- use a Hopper CTA cluster and distributed shared memory.

The first is the only credible option. It need not use a register-heavy
`N=256` tile. For `r=128`, assign a CTA 64 matching K coordinates and 64 V
coordinates in one `N=128` accumulator, using two CTAs per KV head. The pivot
slice issues a K-only `N=64` WGMMA; the remaining slice issues the combined
`N=128` K+V WGMMA. The epilogue gates the paired V half, rotates the K half,
and scatters them to their normal layouts. This preserves a standard-size
accumulator and increases the decode grid.

For the canonical Hopper `SM90_64x128x16_*` WGMMA atom, the required
register pairing is not speculative. CUTLASS defines its accumulator mapping as

```
(T128,V64) -> (M64,N128)
value shape  = (2,2,16)
value stride = (64,8,512)
```

where the flattened output coordinate is `m + 64n`. Moving from output column
`j` to `j+64` adds `4096 = 8*512`: it changes only the same thread's third
logical value index by eight. Therefore ordering the tile as
`[K_0..K_63, V_0..V_63]` puts every matching `K_j,V_j` pair in the same thread,
in different accumulator registers. The gate needs no warp shuffle, shared
memory exchange, or CTA synchronization. CUTLASS uses this same
`CLayout_64x128` for FP32 accumulation with FP16 and BF16 inputs.

What remains to validate by compilation is narrower: CuTe must let the
`N=64` pivot-slice atom alias the first half of the `N=128` accumulator without
a fragment repack, and the extra `p_j/alpha_j` scalar loads must not induce a
spill. Cluster/DSM synchronization remains rejected because it is more costly
than the tiny arithmetic being saved.

Therefore:

- **existing K/V postprocess:** K-reuse is the preferred hardware candidate;
- **paired-CTA monolithic kernel:** plausible, benchmark-required;
- **new kernel or cross-CTA/DSM handoff:** reject as slower than the saved work.

## 4. Decode, prefill, and state relevance

- **Decode:** the projection is often weight-bandwidth dominated. K-reuse can
  remove `g(r^2-r)` stored V weights, but at `d=4096,g=8,r=128` this is only
  130,048 BF16 weights, about 0.248 MiB/layer or 0.078 microseconds at ideal
  H100 HBM bandwidth. At `M=1`, extending an existing RoPE kernel adds only a
  few KiB of V/p/alpha traffic, so it can still be positive. Any new launch
  loses.
- **Prefill:** K-reuse removes roughly `r/d` of V tensor-core work while adding
  linear elementwise work; GFQV keeps the dense WGMMA count and adds a boundary
  transform. An extended RoPE kernel that newly reads and writes V adds
  `2Mgr` activation elements. For the 4K/8-head/128-wide example at `M=2048`,
  that is about 8 MiB in BF16 versus only 0.248 MiB of static V weights removed
  per invocation; use a paired projection epilogue or a true
  RoPE+gate+attention fusion instead. The full attention operation grows
  quadratically with context, so either value-projection change becomes a small
  share at long prefill.
- **KV state:** neither design shrinks K/V dimensions or cache bytes. Their
  quality claim is extra function class at matched projection work, not a
  FlashAttention or cache-bandwidth win.
- **GQA:** the few KV heads make saved bytes smaller and low-batch grids
  narrower; pairing each K/V head is natural, but occupancy is more fragile.
- **MHA:** more head pairs expose more output parallelism and larger aggregate
  savings, but full-RoPE still affects only the K gauge, not this V-side gate.

Forward equality is not a training-cost equality. Backpropagation through the
quadratic gate needs product-rule paths, and K-reuse sends value-loss gradients
into the key projection. Training FLOPs, saved tensors, and optimizer state must
be logged as a separate currency even if served inference passes.

## 5. Preregistered no-rent/rent gate

### No-rent gate

Do not rent a GPU unless all of these pass locally:

1. **Exact containment:** random-matrix tests show `G=0` or `alpha=0` matches a
   dense full-rank V/O pair after gauge transformation, including GQA output
   blocks.
2. **Exact ledger:** an executable counter confirms the formulas above for
   weights, multiplications, and additions; compiler FLOPs are reported
   separately from logical FLOPs.
3. **Learnability:** at matched train steps, data, initialization budget,
   learned scalars, and logical inference operations, the candidate beats
   dense V and an equal-FLOP nonlinear control on at least two tasks and three
   seeds. A synthetic task constructed exactly from the candidate is only a
   positive control, not sufficient evidence.
4. **Fusion boundary:** inspect one target serving engine separately for decode
   and prefill. Prove either an existing pre-RoPE K/V consumer, a decode-only
   extended-RoPE path whose added bytes stay below the removed weight bytes, a
   paired K/V projection epilogue, or a one-launch/one-accumulator CuTe schedule
   for GFQV. No new kernel or global intermediate is allowed.
5. **Numerical plan:** zero-gate inference reproduces baseline tolerance in
   BF16 and the transformed Q/O/V weights have a stated FP8 scaling plan.

### H100 rental gate

If all five pass, rent one H100 for a bounded kernel test. Compare against the
same tiling, weight order, CUDA Graph, and output fusion with the gate disabled.
Test at least:

- `d={4096,8192}`, `r=128`, `g={1,8,d/r}`;
- decode/projection token counts `M={1,8,32,128}`;
- prefill counts `M={512,2048,8192}`;
- BF16 first, then FP8 only if BF16 passes.

Measure warm median and p95 kernel time, HBM throughput, tensor-core throughput,
registers, spills, occupancy, launches, and numerical error.

Promote only if:

- geometric-mean fused-QKV/postprocess latency is at most `1.02x` baseline;
- no declared production-critical shape exceeds `1.05x`;
- projected end-to-end slowdown is at most `0.5%`;
- no extra launch, global intermediate, or register spill appears;
- the local learning gain remains large enough to dominate the measured
  serving penalty.

Otherwise close the branch. Test H200 only after H100 promotion; H200's higher
HBM bandwidth makes a fixed synchronization/scalar overhead relatively harder
to hide.

Current implementation boundary checked:

- [vLLM Llama attention on `main`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)
- [CUTLASS Hopper accumulator-layout derivation](https://docs.nvidia.com/cutlass/4.3.0/media/docs/cpp/cute/0t_mma_atom.html#accumulator-mapping-1)
- [CUTLASS SM90 WGMMA traits](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm90_gmma.hpp)
