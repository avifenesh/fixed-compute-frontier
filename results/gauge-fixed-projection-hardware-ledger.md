# Gauge-fixed K/V projections — served-resource ledger

Status: **algebraically real in a narrow regime; not a frontier model candidate**  
Date: 2026-07-26  
Hardware target: NVIDIA Hopper-class H100/H200; no GPU rented

## Verdict

An identity-minor gauge can remove actual stored weights and projection work
from ordinary attention, but only when the runtime stores a packed matrix and
executes a custom projection. Padding the removed block with zeros/ones and
calling an ordinary dense GEMM saves nothing.

Even with the right kernel, the available fraction is small for normal head
widths, it does not shrink the KV cache, and it does not reduce the
FlashAttention work. Full RoPE also removes the general query/key gauge, leaving
only the value/output side. In representative 4K-width layers, the ideal
whole-block saving is about `0.26%` for full-RoPE MHA and `0.074%` for
full-RoPE GQA before kernel-efficiency and quantization penalties.

The proposed full `c^2` identity-minor saving does **not** apply to DeepSeek MLA:
the compressed latent is passed through RMSNorm, which breaks `GL(c)` gauge
invariance. Direct K/V up-projections are already algebraically absorbed into
Q/O on the efficient inference path. The surviving orthogonal gauge can at
most triangularize a pivot block, and its physical saving is too small and too
awkward for Hopper tensor cores to justify this direction.

Decision: retain this as an exact quotient/packing result and hardware no-go.
Do not rent a GPU for it.

## 1. Exact ordinary-attention ledger

Let

- `M` be the tokens projected in one call;
- `d` be model width;
- `g` be the number of KV heads;
- `r` be their common width;
- `N = 2gr` be the combined K+V output width;
- `b_w` be bytes per stored weight.

Assume one common contiguous pivot set `P`, `|P|=r`, is valid for every K and V
head. Independent head gauges can then make every pivot block an identity. The
combined packed projection is

\[
  [K,V] = X_{\bar P}\,\bar W_{KV} + \operatorname{repeat}(X_P),
  \qquad \bar W_{KV}\in\mathbb R^{(d-r)\times 2gr}.
\]

The exact resource ledger is:

| resource | dense K+V | packed identity-minor K+V | reduction |
|---|---:|---:|---:|
| stored scalar weights | `dN` | `(d-r)N` | `rN = 2gr^2` |
| MACs per call | `MdN` | `M(d-r)N` | `MrN = 2Mgr^2` |
| identity contribution | included in dense MACs | `MN` fused adds | dense did `MrN` MACs |
| activation elements read | at least `Md` | `M(d-r)+Mr = Md` | none |
| output elements written | `MN` | `MN` | none |
| KV-cache bytes/token | unchanged | unchanged | **zero** |
| attention-score/value work | unchanged | unchanged | **zero** |

Counting a MAC as two FLOPs, the net arithmetic reduction after the identity
adds is `MN(2r-1)` FLOPs. For normal `r`, this is essentially the same `r/d`
fraction as the removed weights.

For head widths `{r_h}` rather than one width, each valid side removes
`sum_h r_h^2` weights. At fixed total KV width `D=sum_h r_h` and cap `c`,

\[
  \sum_h r_h^2 \le \lfloor D/c\rfloor c^2+(D\bmod c)^2 \le cD.
\]

Thus the maximum fraction removed from one projection is at most `c/d`.
The quadratic-looking gain is exactly capped by the head-width/model-width
ratio. Unequal widths also require one kernel group per width bucket; arbitrary
per-head widths fragment the GEMMs and can erase the saving.

### Positional-encoding boundary

- **V/O:** the full gauge is exact. For GQA, the inverse transform is applied to
  every output-projection block fed by the shared V head.
- **Q/K without multiplicative position rotations:** the full gauge is exact.
- **Full RoPE:** an arbitrary basis transform does not commute with the
  position-dependent rotations. The general `GL(r)` gauge is gone, so the
  `r^2` K identity block is unavailable. Only the V-side saving should be
  charged.
- **Partial or decoupled RoPE:** a full gauge can survive on a separate NoPE
  subspace. Charge its width, not the full key width.

## 2. When Hopper executes the saving

### Required packed kernel

The useful implementation is a single jagged-output QKV kernel:

1. CTAs assigned Q output tiles run the full `K=d` loop.
2. CTAs assigned packed K/V tiles run only `K=d-r`.
3. The K/V epilogue adds the token-dependent `X_P` value for each head/output.

This keeps one launch and preserves wide output tiling. It needs two weight
layouts/pointers and a custom epilogue; ordinary cuBLAS/cuBLASLt does not infer
the algebra from identity constants.

Hopper WGMMA is tile-dense: BF16/FP16 instructions have K extent 16; FP8 has K
extent 32, and the M extent is 64. Standard widths such as 128 or 512 align
cleanly. NVIDIA sparse Tensor Cores require two zeros in every contiguous group
of four across the *whole* operand. The isolated identity block is sparse, but
the dense remainder violates that global 2:4 layout, so leaving the full
rectangular matrix in place does not activate sparse Tensor Cores. Packing away
the complete identity K slice is the relevant route.

The implementation stops being credible when any of these are true:

- the pivot indices differ by head, forcing per-head gathers or packing;
- the identity add is a second kernel rather than a fused epilogue;
- a previously fused QKV projection is split into two launches;
- each head has a unique width, producing narrow/underoccupied GEMMs;
- the runtime still allocates the original rectangular tensor;
- Q/O gauge transforms worsen FP8/INT4 range or accuracy enough to require a
  wider format.

A fixed, common, contiguous pivot range trained into the architecture is the
only clean layout. A post-training compiler can search a common well-conditioned
minor, but arbitrary indices add gather cost and transformed weights can have
bad dynamic range.

Tensor parallelism makes the latency case weaker. With head/output-column
sharding, each GPU removes only its local heads' weights while paying the same
kernel-dispatch and scheduling floor. If a GQA runtime replicates its few KV
heads on every rank, the per-rank bytes remain removable but the model-wide
storage saving is duplicated rather than converted into new independent
capacity. The existing output all-reduce is unchanged in either case.

### Break-even model

Let `P_eff` and `B_eff` be measured effective arithmetic and HBM rates, and let
`T_over` include extra launches, gathers, an unfused add, occupancy loss, and
layout penalties. A useful first-order comparison is

\[
T_0 \simeq \max\left(\frac{2MdN}{P_{eff}},
                       \frac{b_w dN + B_{act}}{B_{eff}}\right),
\]

\[
T_1 \simeq \max\left(\frac{2M(d-r)N+MN}{P'_{eff}},
                       \frac{b_w(d-r)N+B'_{act}}{B'_{eff}}\right)
              +T_{over}.
\]

The packed form wins only when `T_0-T_1>0`. In the small-`M`, weight-bound
limit with equal efficiencies and a fused epilogue, the available time budget
for *all* overhead is only

\[
  T_{budget} \simeq \frac{b_w\,2gr^2}{B_{eff}}.
\]

NVIDIA documents H100 SXM memory bandwidth as `3.35 TB/s` and H200 as
`4.8 TB/s`. NVIDIA's CUDA Graph guidance gives `1–5 us` as a typical starting
range for per-kernel launch overhead. At peak bandwidth, even `1 us` requires
removing more than `3.35 MB` per H100 invocation or `4.8 MB` per H200
invocation. Most standard-head cases remove less. Therefore an extra launch is
normally fatal; CUDA Graphs reduce CPU submission cost but do not make a
second GPU kernel free.

For dense BF16, using the nonsparse half of NVIDIA's published H100 tensor-core
rate, the ideal roofline crossover is approximately

\[
  M_* \approx P_{dense}/B_{HBM} \approx 295 \text{ tokens on H100},
  \qquad 206 \text{ on H200}.
\]

Below this, projections tend toward weight bandwidth; above this, prefill tends
toward compute. Real crossover depends on shape and achieved efficiency.

## 3. Representative sizes

The table grants both K and V gauges; full-RoPE models receive half the listed
saving.

| shape | removed weights/layer | BF16 bytes/layer | ideal H100 HBM time | ideal H200 HBM time |
|---|---:|---:|---:|---:|
| MHA: `d=4096,g=32,r=128` | 1,048,576 | 2.0 MiB | 0.626 us | 0.437 us |
| GQA: `d=4096,g=8,r=128` | 262,144 | 0.5 MiB | 0.157 us | 0.109 us |
| MHA: `d=8192,g=64,r=128` | 2,097,152 | 4.0 MiB | 1.252 us | 0.874 us |
| GQA: `d=8192,g=8,r=128` | 262,144 | 0.5 MiB | 0.157 us | 0.109 us |

FP8 halves the byte and HBM-time columns. It does not improve the fractional
speedup, because the dense FP8 tensor-core rate rises with the narrower weight
format.

For an illustrative `d=4096`, `r=128`, SwiGLU width `f=11008` layer, count
`2d^2 + 2dgr + 3df` linear weights. The exact gauge removal is:

| attention | both K+V | V only under full RoPE |
|---|---:|---:|
| MHA (`g=32`) | 0.518% of layer weights | 0.259% |
| GQA (`g=8`) | 0.148% of layer weights | 0.074% |

These percentages are an optimistic upper bound on weight-bound whole-layer
latency and on fixed-VRAM capacity that can be reinvested. They fall further at
long context because no KV-cache bytes or FlashAttention operations disappear.
During long prefill, attention is quadratic in sequence length while this
projection saving is linear; during decode, cache reads grow with context while
the projection saving remains constant.

## 4. MLA collision and corrected bound

DeepSeek's efficient path forms

\[
z = XW^{DKV},\qquad c=\operatorname{RMSNorm}(z),
\]

then caches `c`; the K/V up-projections are absorbed into query/output-side
matrices. The official implementation has `wkv_a: d -> (512+64)`, applies
`kv_norm` to the 512-dimensional latent, and caches that normalized latent.

Consequences:

1. The full change of basis `z -> zB`, `B in GL(c)`, is not an invariance of
   RMSNorm. Therefore a `c x c` identity pivot cannot be imposed exactly.
2. Absorb the learned RMSNorm scale into the up-side matrices. Pure RMS
   normalization then permits only orthogonal `B`, because orthogonal maps
   preserve the norm.
3. With an orthogonal `B`, QR can make one invertible `c x c` pivot block
   triangular. This removes only `c(c-1)/2` entries, not `c^2`.
4. A triangular block needs a TRMM-like or custom block-skipping path. Dense
   WGMMA still executes its zeros; splitting it introduces another tiny kernel.

For DeepSeek-V3 dimensions `d=7168`, `c=512`, RoPE side width `64`, 61 layers:

| claim | weights removed/layer | share of `wkv_a` | total FP8 storage | physical status |
|---|---:|---:|---:|---|
| full identity minor | 262,144 | 6.35% | 16.0 MB | **invalid because RMSNorm** |
| orthogonal triangular form | 130,816 | 3.17% | 8.0 MB | exact algebra; poor tensor-core shape |

The triangular form is only about `0.07%` of the nominal DeepSeek-V3 attention
weights per layer, before counting active MoE work. It also leaves the MLA cache
and FlashMLA kernel unchanged. There is no credible served Pareto gain here.

## 5. Sources and falsification gate

Primary/current boundaries used:

- [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA H200 specifications](https://www.nvidia.com/en-gb/data-center/h200/)
- [NVIDIA Hopper WGMMA programming guide](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/wgmma_programming.html)
- [NVIDIA 2:4 structured sparsity requirements](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/)
- [NVIDIA CUDA Graph launch-overhead guidance](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/quantitative-benefits.html)
- [DeepSeek-V2 MLA paper](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 official inference implementation](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)
- [DeepSeek FlashMLA](https://github.com/deepseek-ai/FlashMLA)

A future GPU test is justified only if a later architecture makes the removed
fraction material (at least several percent of active layer bytes/work) while
retaining: one fused launch, common contiguous pivots, aligned width buckets,
unchanged quantized accuracy, and no regression in achieved bandwidth or
tensor-core utilization. This candidate does not meet that gate.
