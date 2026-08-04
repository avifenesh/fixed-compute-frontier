# Real FFN heavy-hitter recovery — decision

Status: **closed**  
Date: 2026-07-27

## Decision

Do not continue the proposal to replace a dense SwiGLU evaluation with a
sublinear sparse-recovery sketch of its largest hidden activations.  The oracle
support itself missed the frozen output-error gates, so a better sketch,
selector, or CUDA kernel cannot rescue the claim.

## Frozen run

- checkpoint: `HuggingFaceTB/SmolLM2-135M` at revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;
- 4,096 WikiText-103 tokens, sequence length 256, batch 4;
- layers 0, 14, and 29;
- top `k=96` of 1,536 SwiGLU channels, or 6.25% active;
- H100 SXM executor;
- source and preregistration frozen before the full run.

The required oracle output-relative-error gate was median `<=5%`, p90 `<=10%`
at every layer.  A learned/sketched selector was allowed median `<=7.5%`, p90
`<=15%`.

| Layer | Oracle median | Oracle p90 | Selected median | Selected p90 |
|---:|---:|---:|---:|---:|
| 0 | 23.49% | 46.27% | 33.70% | 62.56% |
| 14 | 61.50% | 68.99% | 79.00% | 88.16% |
| 29 | 20.01% | 26.17% | 31.12% | 38.97% |

All three layers failed before accounting for sketch construction, recovery,
indexing, sparse gathers, or kernel efficiency.

The activation-energy view did not save the proposal.  Channels needed for 95%
of activation energy varied sharply by depth: median 167 (10.87%) at layer 0,
540 (35.16%) at layer 14, and 78 (5.08%) at layer 29.  Even the apparently
sparse last layer retained a distributed output because down-projection values
and cancellation, rather than activation magnitude alone, determine error.

## Retained knowledge

1. Native dense SwiGLU is not secretly a reliable top-6.25% sparse layer.
2. Activation compressibility is not output compressibility; the down matrix
   and cancellation must be in the oracle.
3. A learned sparse FFN can still work by changing training, but that enters the
   already-occupied Spark/MoC/memory-layer family and is not evidence for this
   post-hoc sketch claim.
4. Future FFN proposals must pass an oracle output gate before spending time on
   routers, sketches, or GPU kernels.

Evidence: [`real-ffn-heavy-hitter-gate.json`](real-ffn-heavy-hitter-gate.json)
and
[`real-ffn-heavy-hitter-preregistration.md`](real-ffn-heavy-hitter-preregistration.md).
