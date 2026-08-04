# T32R streaming boundary scan — H100 preregistration

Status: **FROZEN BEFORE IMPLEMENTATION OR TIMING**  
Date: 2026-07-31

## Question

Can the exact T32R reader be inserted after block 3 while reducing all ten
SwiGLUs from width 1,024 to 940, with no p50/p95 or peak-memory regression on
the retained H100?

This is a physical block gate, not a learnability or quality experiment.

## Frozen environment and shapes

- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0;
- PyTorch `2.11.0+cu128`, Triton `3.6.0`, BF16 parameters/activations;
- hidden 384, ten residual SwiGLU blocks;
- baseline FFN width 1,024;
- candidate FFN width 940;
- record table `2405 x 381` BF16 amplitude cells;
- shared embedding `49152 x 384` BF16;
- scan rank 32, four heads of width eight;
- batch sizes `1, 8, 32`;
- prompt lengths `9, 12, 31`;
- seed `320320`.

Common attention and normalization are omitted from both arms. This makes the
gate conservative: common work cannot dilute reader overhead.

## Frozen baseline

For ten layers, execute an actual residual SwiGLU:

```text
x = x + down_l(silu(gate_l(x)) * up_l(x))
```

with three independent BF16 matrices at width 1,024 per layer.

## Frozen candidate

Execute the same ten residual SwiGLUs at width 940. Immediately after layer 3,
run the exact rank-32 reader on the final prompt position and add its gated
summary there before layers 4--10.

The reader timing includes:

1. gather two BF16 record rows by handle;
2. decode all 128 base-token IDs, including the final auxiliary token;
3. gather normal token embeddings;
4. project every embedding exactly once through `384 x 32`;
5. add fixed sinusoidal position codes;
6. apply the depthwise three-tap context filter;
7. update four-head attention by exact online softmax;
8. apply two role maps, output projection, scalar gate, and boundary residual.

## Frozen streaming schedule

Use eight 16-token chunks. Materialize at most one
`[batch,2,16,32]` projected chunk. Carry the last two projected token vectors
between chunks:

- at chunk 0, process positions 0--14 and retain 14--15;
- at later chunks, first finish the preceding pending position using the new
  first token, then process all new internal positions and retain the last two;
- after chunk 7, finish position 127 with zero right padding.

The online state per request/document/head is running maximum `m`, normalizer
`l`, and output numerator `y`. Updates use the exact safe-softmax rescaling
recurrence. No full `[128,32]` projection or `[128,384]` gathered embedding
tensor may exist.

## Frozen implementation path

- implement the two arms in PyTorch with fixed-shape Python loops;
- compile each specialized `(batch,length)` callable with
  `torch.compile(fullgraph=True, mode="reduce-overhead")`;
- compilation/autotuning time is excluded;
- use the compiled steady-state path, whose CUDA graph addresses/shapes remain
  fixed;
- no implementation edits are permitted after any timing cell is observed.

If full-graph compilation fails before timing, the physical gate fails. Do not
fall back to eager execution after observing the failure.

## Correctness gates before timing

1. GPU record decode exactly matches the seeded host token records;
2. streamed summaries match a full materialized FP32-safe-softmax Torch
   reference within maximum absolute error `5e-3`;
3. compiled and eager candidate outputs match within `5e-2` absolute error;
4. every output is finite;
5. the streamed implementation exposes no tensor with both record length 128
   and embedding width 384 or projection width 32;
6. baseline and candidate resident parameter bytes are exactly equal after
   adding the frozen candidate slack/baseline match tensors.

Any correctness failure stops before timing and closes the implementation.

## Timing protocol

For each of nine `(batch,length)` cells and each arm:

- 100 unmeasured compiled warmups;
- 300 CUDA-event measurements;
- synchronize only after event recording;
- alternate arm measurement order by cell;
- report p50, p95, mean, minimum, maximum, and candidate/baseline ratios;
- record total elapsed time, peak allocated bytes, peak reserved bytes, and
  final GPU state.

Reset peak memory statistics immediately before one compiled replay. Resident
parameters and static compiled graph pools remain charged. Report both total
peak and incremental peak above pre-replay allocation.

## Mandatory gates

All must pass:

1. environment, shape, source, test, and preregistration integrity;
2. every correctness gate;
3. candidate p50 no greater than baseline p50 in all nine cells;
4. candidate p95 no greater than baseline p95 in all nine cells;
5. candidate total peak allocated and reserved bytes no greater than baseline
   in all nine cells;
6. zero extra KV entries, prompt tokens, model bytes, network/storage reads, or
   auxiliary model work;
7. exactly one timing run and zero post-result implementation changes.

## Decision

- **Pass:** admit a separately frozen raw-only learnability preflight with
  dense, free-memory, learned-summary, and Engram-like controls.
- **Fail:** close this exact T32R physical composition. Retain the information
  result and algebra, but do not rescue timing with a new chunk size, rank,
  compiler mode, custom kernel, or relaxed cell after observing results.
