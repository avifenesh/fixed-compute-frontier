# Packed token evidence plane T27 — decoded-prefix oracle preregistration

Status: **FROZEN BEFORE MODEL-WEIGHT DOWNLOAD OR SCORING**  
Date: 2026-07-31

## Question

Do the first 55 frozen SmolLM2 tokenizer IDs of each routed raw document retain
enough natural evidence for a strong fixed reader to answer title-disjoint
comparison questions, with a large causal advantage over omitted and shuffled
evidence?

This is deliberately stronger than the deployable candidate.  It decodes the
token tape perfectly into text and gives it to a 9B post-trained reader without
paying the candidate's packed in-model read cost.  Passing establishes only an
information ceiling.  The reader is never a teacher, compiler, candidate
component, or serving dependency.

## Frozen artifacts

- Raw corpus:
  `data/hotpot-real-prose-t12/candidate-raw-prose.jsonl`, SHA-256
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`.
- Evaluation rows:
  `data/hotpot-real-prose-t12/sealed-evaluator.jsonl`, SHA-256
  `5a0413f53bfc0fc73b5e66c941c708077624908095934af740c84d234b9fea6b`.
- T27 Stage-0 result:
  `results/packed-token-evidence-plane-t27-stage0.json`, SHA-256
  `6d24c18fbb21ec95e41fff87f547067f93cee4fe46dcdb2a5f499e564dbef0a8`.
- Packing tokenizer: `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Oracle reader: `Qwen/Qwen3.5-9B`, revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Reader runtime: Transformers `5.14.1`, PyTorch `2.11.0+cu128`, BF16 on the
  retained H100.

The model's tokenizer/config may be downloaded to define the prompt and verify
single-token labels before this freeze.  Model weight download and scoring
occur only after the preregistration hash is recorded.

## Frozen routing and tape

Use the existing label-blind longest-nonoverlapping-title rule over all 2,405
corpus titles.  Order the two titles by their occurrence in the question.
Exactly two must be found in every row.

For each routed document:

1. tokenize the entire raw `text` with the pinned SmolLM2 tokenizer and no
   special tokens;
2. take exactly the first 55 IDs, padding shorter documents with token ID zero;
3. decode only the unpadded IDs with cleanup disabled;
4. re-encode and require exact equality to the retained ID sequence.

The oracle never reads supporting titles, supporting sentence IDs, training
questions, answers while building prompts, or any semantic parser output.

## Entity anonymization

Replace the two title spans in the question, in occurrence order, with
`Entity A` and `Entity B`.  In each evidence excerpt, replace case-insensitive
occurrences of that excerpt's own source title with the corresponding alias.

This preserves the question-to-document binding while reducing the reader's
ability to answer from memorized named-entity facts.  Labels remain untouched
and are used only after predictions are frozen.

## Frozen conditions

1. `correct_prefix`: the two correctly routed decoded 55-token prefixes.
2. `shuffled_prefix`: replace every source title by the title 997 positions
   later in the lexically sorted 2,405-title list, modulo 2,405.  This fixed
   permutation has no fixed point and preserves distinctness.  Decode that
   wrong document's first 55 IDs and alias its own title as the requested
   entity.
3. `omitted`: show `[omitted]` for both evidence excerpts.
4. `full_document`: the two correctly routed complete raw documents, with the
   same aliasing.  This is a non-gating truncation reference except for finite
   execution and at least 80% accuracy.

No condition changes the question or label distribution.

## Frozen prompt and score

System message:

```
Use only the supplied evidence. Answer the comparison question with exactly
yes or no. Entity names have been replaced consistently. Do not use outside
knowledge and do not explain.
```

User message:

```
Question: {aliased_question}

Evidence for Entity A:
{evidence_a}

Evidence for Entity B:
{evidence_b}

Answer (yes or no):
```

Render the pinned Qwen chat template with `enable_thinking=False` and an
assistant generation prompt.  The candidate answer strings are lowercase
`yes` and `no`; the Qwen tokenizer must encode each as exactly one token
(`9405` and `2083` at freeze time).

Run one deterministic BF16 forward pass per prompt and compare the final
position logits for those two tokens.  Predict `yes` iff `logit_yes >
logit_no`; an exact tie is invalid.  No generation, sampling, calibration,
threshold fitting, retry, or answer parsing is allowed.

## Mandatory gates

All gates are required:

1. Every frozen artifact hash and package/model revision matches.
2. Exactly two label-blind titles route for every evaluation row; the fixed
   shuffle is a derangement and preserves the two-title distinction.
3. Every retained prefix has at most 55 IDs, every padded tape has exactly 55,
   and decoded text re-encodes to the identical unpadded IDs.
4. Prompt construction reads no answer or supporting field; both answer
   classes are present and labels enter only during final scoring.
5. The yes/no candidate strings remain the frozen single token IDs; every
   prompt length is within the reader context and all logits/margins are finite
   and nonzero.
6. `correct_prefix` accuracy is at least 80% and at least 15 absolute points
   above T19b's 60.5769%.
7. `correct_prefix` exceeds both `shuffled_prefix` and `omitted` by at least 15
   absolute percentage points.
8. `full_document` reaches at least 80% accuracy.  It is reported as a
   truncation ceiling but need not exceed the prefix.
9. Predictions, labels, condition accuracies, paired causal changes, margin
   summaries, hardware/package metadata, hashes, and zero retries are recorded.

## Kill boundary

Failure closes exact fixed-prefix evidence as T27's natural payload.  Do not
rescue it with 56 tokens, first-sentence parsing, question-conditioned
selection, another reader, prompt changes, generation, threshold fitting,
training labels, Huffman coding, or a larger model after observing the result.

Passing admits only a separate same-graph physical reader microbenchmark.  It
does not admit from-zero model training or a production claim.
