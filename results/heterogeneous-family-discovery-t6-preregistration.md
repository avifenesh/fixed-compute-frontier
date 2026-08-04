# Heterogeneous family discovery T6 — preregistration

Frozen before implementation or measurement: 2026-07-30

## Question

Can a training-time solver portfolio infer which discrete algebra explains a
charged task, compile only the supported programs into one fixed causal LM, and
abstain on unstructured data—without being told that every task is parity?

## Tasks and discovery

- 320 task IDs reuse token rows absent from both sealed FineWeb-Edu streams.
- Inputs contain 16 bits.  Every structured task depends on an unknown 8-bit
  support.
- There are 64 tasks from each of four families: parity, strict majority,
  exact-half count, and count modulo 3.  Another 64 tasks have independent
  random labels and serve as abstention controls.
- Every task has a sealed 96-example charged prefix.  Structured prefixes are
  generated until exactly one `(support, family)` program in the frozen search
  space fits all 96 examples.  Random prefixes are generated until no program
  fits.

The candidate compiler is given only task IDs, bit matrices, and labels.  It
enumerates all `C(16,8) = 12,870` supports and the four frozen count functions,
accepts exactly one zero-error program, and otherwise abstains.  It must recover
all 256 structured programs exactly and abstain on all 64 random tasks.

## Shared in-place interpreter

The model is T3's 36,577,152-parameter tied-embedding causal LM with ten
pre-RMSNorm blocks, six 64-wide heads, SwiGLU width 1,024, causal masking, and
four RoPE language heads.  Model shape, vocabulary 49,152, graph, and dense
FLOPs are identical in every arm.

Each accepted task stores a 16-value support sign vector and a four-value
one-hot family code in its existing task-token embedding row.  One shared
interpreter:

1. copies support and family to the query;
2. selects the eight supported positions by attention;
3. reduces their binary signs to one of nine integer sums;
4. applies one of four shared preregistered count tables; and
5. separately preserves a first-bit copy route.

Random tasks receive no compiled support or family code.  The compiler is
deleted after writing the ordinary dense weights.  There is no sidecar,
adapter, expert, retrieval store, added vocabulary row, or runtime solver.

## Arms and budgets

Two paired model seeds run:

- `baseline_1x`: ordinary gradient training on all 30,720 prefix examples,
  followed by 1,000 mixed steps;
- `compiler_1x`: portfolio discovery/compilation followed by the same 1,000
  mixed steps;
- `baseline_2x`: the same prefix gradient training followed by 2,000 mixed
  steps.

Every twentieth mixed step is algorithmic; the other 19 use the sealed natural
stream.  AdamW, BF16 forward arithmetic, FP32 losses, clipping at 1, paired
batches, and the frozen warmup/cosine schedule match T3.

## Frozen gates

Every gate must pass in both paired model seeds:

1. Identical pre-compilation initialization hashes; exact support and family
   recovery for all 256 structured tasks; zero accepted random tasks.
2. The BF16 quick test and candidate checkpoints 250, 500, and 1,000 have at
   least 99% mean and minimum accuracy within every structured family, plus at
   least 99% protected-copy accuracy.
3. Candidate random-task accuracy remains between 40% and 60% on independent
   labels, demonstrating abstention rather than leakage.
4. At step 2,000, ordinary-control parity and modulo-3 mean accuracy are each
   below 80%, while protected copy is at least 95%.
5. Candidate natural validation NLL is no more than 0.5% above its paired 1x
   control at steps 250, 500, and 1,000 and improves from 500 to 1,000.
6. All losses, gradients, parameters, and evaluations remain finite; no step is
   skipped; maximum loss is below 100.  Pre-clip norms are recorded.
7. Parameter count, vocabulary, graph, and inference FLOPs are identical across
   arms; interpreter width is independent of task count.

Passing establishes automatic family selection and abstention across four
algebras in the same fixed LM.  It remains a synthetic count-program result and
does not by itself prove broad natural-language reasoning gains.  Failure on
discovery, BF16 execution, coexistence, or finite training closes this exact
portfolio without post-measurement tuning.
