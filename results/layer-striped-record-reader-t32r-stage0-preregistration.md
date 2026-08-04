# T32R handle and compute census — Stage-0 preregistration

Status: **FROZEN BEFORE IMPLEMENTATION**  
Date: 2026-07-31

## Scope

This CPU-only gate checks title-handle routing, token-count effects, exact record
expansion, namespace separation, and the paper's parameter/multiplication
inequalities. It does not read evaluator answers or support values, load model
weights, implement the learned rank-32 scan, or use a GPU.

## Frozen inputs

- corpus:
  `data/hotpot-real-prose-t12/candidate-raw-prose.jsonl`, SHA-256
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`;
- sealed evaluator:
  `data/hotpot-real-prose-t12/sealed-evaluator.jsonl`, SHA-256
  `5a0413f53bfc0fc73b5e66c941c708077624908095934af740c84d234b9fea6b`;
- packing/base tokenizer: `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;
- exact T32 pack/unpack code from the passed Stage-0 result.

Evaluator access is limited to `id` and `question`. `answer`,
`supporting_titles`, and `supporting_sentence_ids` values are forbidden.

## Frozen handle tokenizer

Use the existing longest-nonoverlapping-title route. For token-count accounting,
tokenize each non-title question span independently with the base tokenizer and
emit one input-only handle for each of the two matched title spans. The two
handles are not output-vocabulary tokens and are not passed to the base
tokenizer.

Compare this length to ordinary whole-question base tokenization. Require no
question to become longer and require positive aggregate token saving. This is
only a routing/token-count census; no tokenizer model is modified.

## Frozen record namespace

- base token IDs: `0..49151`;
- input-only handle IDs: `49152..51556`;
- one handle per document, exactly 2,405;
- records are compiled with the base tokenizer before handle substitution;
- every record contains exactly 128 padded base-token slots and no handle ID;
- each table row stores 381 base-64 digits; the scale reference and two reserved
  coordinates are generated constants and are not table parameters.

Every corpus record must pass exact T32 pack/unpack. Report the observed token
domain and a deterministic record digest.

## Frozen parameter ledger

```text
removed FFN width per layer     84
removed dense entries          10 * 84 * 3 * 384 = 967,680
record digit entries           2,405 * 381       = 916,305
shared token/query projection  384 * 32          = 12,288
scan Q/K/V                     3 * 32 * 32       = 3,072
bounded local scan reserve     3 * 32 * 32       = 3,072
summary output                 32 * 384          = 12,288
candidate allocated entries                         947,025
matched slack                                        20,655
```

Require exact equality to these values and allocated entries no greater than
removed entries.

## Frozen multiplication bound

For candidate handle-question length `q`:

```text
scan upper bound = 3,700,000 + 12,288*q
dense saving     = 967,680*q
margin           = dense saving - scan upper bound
```

Require `q >= 4` and strictly positive margin for every routed question. No
title-token saving is included in this inequality.

## Mandatory gates

1. all input hashes, schemas, tokenizer name/revision, and CPU-only boundary
   match;
2. all evaluator questions route exactly two distinct titles without answer or
   support access;
3. handle tokenization never increases question length and saves tokens in
   aggregate;
4. all 2,405 records round-trip, contain only base IDs, and produce the same
   digest in forward/reversed input order;
5. all parameter-ledger values match and allocation stays within removed dense
   entries;
6. every question has at least four candidate tokens and positive conservative
   multiplication margin;
7. zero model-weight, answer, support-value, evaluator-label, or GPU reads.

## Decision

- **Pass:** admit a fully specified rank-32 scan algebra and isolated CPU/GPU
  block microbench; do not admit natural training.
- **Fail:** close this exact T32R allocation. Do not alter rank, FFN width,
  handle rule, record length, or compute equation after observing the census.
