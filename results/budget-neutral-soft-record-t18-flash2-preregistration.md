# Budget-neutral soft record T18-Flash2 — operator/runtime preregistration

Status: frozen before execution  
Date: 2026-07-31

## Purpose

The one-head T18 operator passed algebra and equal-MAC gates but failed isolated
H100 allocation and batch-32 latency.  T18-Flash2 tests the sole runtime
successor: two 192-dimensional record heads executed by forced FlashAttention,
with exactly the same served parameters and matrix MACs.

No capability training is permitted in this gate.

## Frozen algebra and ledger

- hidden width: 384;
- heads: 2;
- head dimension: 192;
- slots per head: 2,406 including one null record;
- key storage: one physical `2,406 x 384` matrix;
- value storage: one physical `2,406 x 384` matrix;
- residual SwiGLU width: 444;
- candidate and two-SwiGLU baseline parameters/MACs: exactly 2,359,296 per
  replaced pair and per token.

For head `j`:

`p_j = softmax(h_j K_j^T / sqrt(192))`

`o_j = p_j V_j`.

Output is `concat(o_0,o_1)`.  Heads may route different record halves; this is
part of the frozen architecture, not claimed equivalent to one-head T18.

K/V batch dimensions are stride-zero expanded to the query batch before forced
Flash SDPA.  No key/value data copy is allowed.

## Frozen contracts

1. CPU FP32 forward and complete gradients match an explicit two-head dense
   reference within `2e-6`;
2. dominant null keys with zero null values suppress both output halves below
   `1e-5`;
3. full candidate model and block pair exactly match baseline parameter counts;
4. matrix MAC ledger remains exact;
5. H100 reports the forced Flash backend usable for expanded 192-dimensional
   Q/K/V;
6. storage pointers/strides prove K/V batch expansion is a view, not a copy.

## Frozen runtime protocol

Reuse T18 runtime v1 exactly:

- surfaces, trials, warmups, iterations, BF16 autocast, CUDA-event timing,
  interleaved order, peak incremental HBM procedure, seed 10,109, and 1.05x
  gates are unchanged;
- block-local surfaces: `(1,1), (1,128), (8,128), (32,128)`;
- whole-model surfaces: `(1,1), (1,128), (8,128)`;
- no CUDA graph, `torch.compile`, fallback backend, or custom kernel.

Every candidate CUDA forward is forced to `SDPBackend.FLASH_ATTENTION`; backend
failure aborts rather than silently falling back.

## Admission gates

All operator contracts must pass, then:

1. candidate block median latency <=1.05x baseline at every surface;
2. candidate whole-model median latency <=1.05x at every surface;
3. candidate block and whole-model peak incremental allocation <=1.05x at every
   surface;
4. all values are finite and hashes match.

Passing admits a from-scratch real-prose/natural-question training protocol.
It still does not show record usefulness, language preservation, QA gain, or a
strict Pareto result.

Failure closes dense 2,406-slot soft-record attention under the equal served
budget at this scale; training may not proceed by ignoring the runtime result.
