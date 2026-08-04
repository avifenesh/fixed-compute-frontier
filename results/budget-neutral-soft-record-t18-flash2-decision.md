# Budget-neutral soft record T18-Flash2 — decision

Status: **FAIL; CLOSE dense soft-record serving operator**  
Date: 2026-07-31

## Decision

Close dense 2,406-slot soft-record attention under the equal served budget.
Do not train it and do not rescue it with pre-casting, more heads, custom
kernels, changed slots, relaxed peak accounting, or whole-model-only gates.

T18 Stage 0 remains a valid algebraic result: the operator can exactly replace
two SwiGLUs in parameter and matrix-MAC counts.  The runtime results show that
this equality is insufficient for equal serving cost.

## Result

All operator contracts passed:

- two-head FP32 reference maximum error was below `2e-6`;
- null output maximum was `6.37e-6`;
- forced FlashAttention was eligible and executed;
- expanded K/V tensors shared storage with their base tensors;
- block and whole-model parameter counts matched exactly;
- matrix MACs remained exactly 2,359,296 per replaced pair/token.

Whole-model latency remained inside the 1.05 gate:

- batch 1, token 1: 1.0178x;
- batch 1, 128 tokens: 1.0296x;
- batch 8, 128 tokens: 1.0265x.

But the isolated block failed:

- batch 1, token 1: 1.0277x latency, 4.7313x peak allocation;
- batch 1, 128 tokens: 1.0197x latency, 6.4369x peak allocation;
- batch 8, 128 tokens: 1.0443x latency, 4.1759x peak allocation;
- batch 32, 128 tokens: **1.2341x latency, 3.8021x peak allocation**.

Flash2 improved the batch-32 latency from the one-head materialized result's
1.4776x, but did not pass.  Result artifact SHA-256:
`bd8484ea15c6b84bcef82e1323b480c3c74252831978bb4af33041510e6a3527`.

## What the peak result exposed

K/V expansion was a stride-zero view before the call, but BF16 autocast of
FP32 parameters materialized expanded K/V operands for the fused backend.  The
mathematical parameter ledger did not price this dtype/layout conversion.

A serving-BF16 pre-cast implementation might avoid that allocation, but the
frozen protocol explicitly made Flash2 the sole runtime successor and closed
the architecture on failure.  It remains an untested implementation note, not
an admitted rescue.

## Retained knowledge

1. Equal parameters and MACs do not imply equal latency or workspace.
2. A memory-shaped matrix decomposition can disappear inside whole-model
   output-projection cost while still be 23% slower locally.
3. Broadcast, dtype conversion, and fused-kernel eligibility are part of model
   algebra once serving cost is the target.
4. Moving knowledge into a new served lookup operator has now failed both the
   exact-address interface and runtime boundaries.

## Next boundary

The served graph must remain the ordinary Transformer.  A new compiler may act
only during training and must merge its writes into existing dense weights
before serving.  Address and write/read compatibility must be co-trained from
zero; they cannot be inferred post-hoc from an ordinary LM checkpoint.
