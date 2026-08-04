# TVE full-attention native-cost gate design

This is a build design, not yet a frozen preregistration. It closes the main
scope hole in H100 v6, which timed packed QKV plus a cache-write epilogue but not
the strongest native KV-update/attention path.

## Backend and measured path

Pin FlashAttention main commit
`00756db9d921da0846453283ddfbeb7457abd09b` (reported version 2.8.4,
2026-07-24). Its inference API supports GQA, in-place KV update, and a paged
cache through an int32 block table with page size divisible by 256.

One CUDA graph covers every attention layer:

`hidden -> packed QKV GEMM -> RoPE/current KV update -> causal paged GQA -> O GEMM`

RMSNorm, MLP, residual, embeddings, and LM head stay outside the measurement;
including unchanged work would hide marginal TVE cost.

## Required arms

1. `native_canonical`: the backend's fastest legal fused KV-update/attention
   schedule.
2. `schedule_control`: the candidate schedule with TVE arithmetic disabled.
3. `tve_candidate`: reads coefficients from physical O slots inside the timed
   path, with no packed coefficients or persistent side state.

Candidate versus native is primary. Candidate versus schedule isolates the
arithmetic. A deliberately slower TVE-shaped control cannot admit the method.

## Exact geometries

- 37.8M pilot: 12 layers, hidden 384, Q6/KV2, head 64, packed QKV width 640,
  960 reused slots per layer.
- SmolLM2-360M: 32 layers, hidden 960, Q15/KV5, head 64, packed QKV width 1,600,
  2,400 reused slots per layer.

The 37.8M gate uses a real terminal candidate BF16 export. The 360M gate waits
for a real exact-shape trained export; synthetic coefficients may be used only
for implementation development.

## Frozen grid for the later preregistration

- Decode: `(batch,total_context)` = `(1,128)`, `(1,2048)`, `(1,8192)`, and
  `(8,2048)`, one new query token.
- Prefill: `(batch,tokens)` = `(1,128)` and `(1,2048)`.
- Page size 256 with deterministic noncontiguous block tables.
- Historical decode cache is built outside timing; current QKV, TVE/cache
  mutation, attention, and O projection are inside.

Correctness is per layer and cell: raw control caches, encoded candidate V,
unchanged Q/K, full attention, and O outputs against independently assembled
same-backend references. Candidate delta must be nonzero in every layer.

Timing is 50 warmups and 200 randomized paired single-graph trials, at least
128 MiB of arithmetic L2 disturbance before each arm, same-stream CUDA events,
5,000 paired bootstrap samples of log ratios, and no outlier removal. Every
cell must meet candidate/native median at most 1.005, upper 95 percent at most
1.010, and p95 ratio at most 1.025. The epilogue sidecar retains an upper 95
percent candidate/schedule bound of 1.025.

Graph nodes, weights, KV/page-table bytes, workspaces, temporaries, and metadata
are ledgered. Pre-encoding, coefficient packing, hot-cache repeated replay,
extra untimed copies, and dilution by unchanged decoder work are forbidden.

## Implementation fork

Start with the honest prewrite schedule to quantify the native gap. If it fails,
the only admissible optimization is to place the triangular transform inside
the native current-V cache-update path or the QKV GEMM epilogue. Long-context
attention may not be used to hide an extra launch.
