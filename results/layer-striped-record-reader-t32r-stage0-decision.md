# T32R handle and compute census — Stage-0 decision

Status: **PASS TO FROZEN RANK-32 SCAN ALGEBRA AND BLOCK MICROBENCH**  
Date: 2026-07-31

## Verdict

The exact-record reader refinement passes its raw routing, namespace,
parameter, token-count, and conservative multiplication gates. The next step
may specify and microbenchmark the bounded scan. Natural training remains
forbidden.

## Evidence

- 104/104 evaluator questions routed exactly two distinct titles using only
  `id` and `question`;
- replacing the two title spans by input-only handles saved 381 tokens total,
  3.6635 per question on average, and never increased a question's token count;
- candidate question lengths were 9--31 tokens, safely beyond the frozen
  four-token arithmetic break-even;
- the minimum conservative multiplication margin was 4,898,528, with mean
  7,617,721;
- all 2,405 records round-tripped exactly, used only base IDs 0--49,137, and
  remained disjoint from handle IDs 49,152--51,556;
- the 2,405-by-381 record table occupies 1,832,610 BF16 bytes;
- forward/reversed record digests matched at
  `3c7e83a9b4b3a6910957570a93707cd67f65e6a9bada7edaa5a3c20897e16549`;
- removing 84 SwiGLU channels per layer frees 967,680 entries; the frozen
  record/reader allocation consumes 947,025 and leaves 20,655 matched slack;
- 24/24 combined unit tests passed;
- the census took 0.992 seconds CPU wall time with CUDA hidden and zero model
  weight, label, support-value, or GPU reads.

## What remains unproved

The multiplication bound prices a reader envelope; it does not define a good
reader. A tokenizer ID has no semantics until it indexes the shared token
embedding, and rank 32 may discard the exact information that made the 9B
oracle succeed. Kernel launch and gather traffic may also erase the arithmetic
advantage.

The next artifact must freeze one scan, not search architectures on H100. It
must specify query formation, token/position projection, local context,
multihead selection, document-role separation, summary injection layer,
training signal, parameter use, and exact multiplication/memory traffic.

## Prior-art control boundary

Per-layer embeddings, token-specific value banks, hashed lookup transforms,
conditional memory, and document parameters are occupied. A later natural run
must include an Engram-like conditional-memory arm and a free learned document
summary at equal parameters/FLOPs. A win over dense-only is insufficient.

## Integrity

- result SHA-256:
  `e791d9a37e934e850a704b78cdc10c7f918feddc3aea78c8482ca9736671533e`
- preregistration SHA-256:
  `84a910d40f267a93cb9ac35edc874c6b0bf7f7d270b2d55beb8a4d035a0ec7ed`
- source SHA-256:
  `dbab287ff5dd98136aeba6c538e8336239f5da0891a45718fd8b51654034cc5a`
- tests SHA-256:
  `01129bb388e4113c5a0a642af84bc05ad6c5970f29af99a3b27d08dda209d78f`
