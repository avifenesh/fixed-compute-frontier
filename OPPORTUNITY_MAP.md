# Opportunity Map

This file describes where an edge could originate. It deliberately contains no
active candidate and no ranking.

## 1. Learning edge

Same served graph, better weights: data quality and ordering, optimization,
initialization, loss allocation, auxiliary training objectives, anti-forgetting,
or training-only supervision. The required control is an equally tuned baseline
with equal data exposure and total training FLOPs. Any teacher, selector,
verifier, or synthetic generator is recorded in the training ledger.

## 2. Structural edge

Exploit real distributional structure through a better inductive bias or reuse
of factors. This can improve useful composition but cannot manufacture
independent information. Every test needs both structured generalization and a
random-mapping control.

## 3. Allocation edge

Remove a measured low-value consumer of bytes, FLOPs, KV, or traffic and
reinvest the exact saving into a higher-value component. The donor saving must
exist in the relevant prefill/decode regime and on the target hardware; nominal
parameter or FLOP arithmetic is insufficient.

## 4. Compression edge

Quantization, pruning, low-rank structure, shared dictionaries, coded weights,
and state compression may reclaim physical resources. Metadata, indices,
decompression, unsupported kernels, rare-feature loss, and packed-versus-
expanded execution all belong in the budget.

## 5. Conditional-compute edge

Spend a fixed average or peak budget where it has higher marginal value.
Routing, dispatch, imbalance, variable tail latency, batch fragmentation, and
hard-example undercomputation are part of the mechanism—not incidental costs.

## 6. Runtime edge

Fusion, IO reduction, layouts, scheduling, parallelism, speculative execution,
and caches may preserve model quality while lowering wall time or energy. This
is a B/D result by itself. It becomes a capability result only after a separately
measured reinvestment of the saved budget.

## 7. Request-compute edge

Reduce the total work needed to solve a task, including generated reasoning
tokens. Per-token speed is not sufficient. Quality, output length, total joules,
and latency per completed task must be compared together.

## 8. Transferred-compute edge

Retrieval, tools, external memories, teachers, helper models, caches, and
offline generation can be excellent systems engineering, but their resources
are external to the core checkpoint. They count as a trade unless the complete
system still wins the D/E ledger.
