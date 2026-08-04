# Budget-neutral soft record T18 — unfused runtime decision

Status: **FAIL; fused two-head successor admitted**  
Date: 2026-07-31

## Decision

Do not train the materialized one-head 384-wide operator.  Its equal matrix-MAC
ledger hid a high-batch runtime and allocation penalty.

One successor is admitted: split each 384-dimensional entity key/value record
into two independently normalized 192-dimensional record heads and execute
them with the installed FlashAttention SDPA backend.  This preserves exact
parameters and matrix MACs while changing one 384-wide softmax into two
192-wide softmaxes.

## Frozen outcome

Whole-model median latency passed at every measured surface:

- batch 1, token 1: 1.0090x baseline;
- batch 1, 128 tokens: 1.0070x;
- batch 8, 128 tokens: 1.0068x.

Whole-model peak allocation was exactly 1.0x baseline at all three surfaces.

The isolated block exposed the hidden cost:

- batch 1, token 1: 0.8857x latency, 2.3537x peak allocation;
- batch 1, 128 tokens: 0.9615x latency, 2.5560x peak allocation;
- batch 8, 128 tokens: 1.0342x latency, 2.4996x peak allocation;
- batch 32, 128 tokens: **1.4776x latency, 2.7999x peak allocation**.

The result therefore failed block latency and block peak-allocation gates.
Result artifact SHA-256:
`099e2742108f66fb5dfd54a53e95a9dd09ee9c13d287f66d4f13be6f723a9bc9`.

## Backend diagnosis

On the installed PyTorch 2.11/CUDA 12.8 H100 runtime:

- Flash SDPA accepts head dimensions 64, 128, 192, and 256;
- a 384-wide head is rejected because fused Flash/CuDNN kernels cap head
  dimension at 256;
- the 384-wide default SDPA path was slower and used more peak allocation than
  the explicit materialized reference;
- a forced two-head 192-dimensional Flash SDPA exploratory measurement was
  valid when K/V batch views were explicitly expanded.

These observations choose executor topology, not a capability result.

## Sole runtime successor

For `h = concat(h_0,h_1)` and record rows
`K_e = concat(K_e0,K_e1)`, `V_e = concat(V_e0,V_e1)`, compute

`o_j = softmax(h_j K_j^T / sqrt(192)) V_j`, for `j in {0,1}`,

then concatenate `o_0,o_1`.

The matrix ledger remains:

`2 heads * 2 (QK and PV) * 2,406 slots * 192 = 1,847,808 MACs/token`.

Parameter storage is unchanged at two 2,406-by-384 matrices.  The nonlinear
ledger doubles softmax exponentials/reductions from 2,406 to 4,812 per token,
so only measured fusion—not MAC arithmetic—can admit it.
