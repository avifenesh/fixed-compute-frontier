# T32 whole-record/local-window natural-information oracle — preregistration

Status: **FROZEN BEFORE EVALUATOR READ OR MODEL SCORING**  
Date: 2026-07-31

## Question

Do the two correctly routed 128-token T32 records contain enough causal
information for a strong fixed reader to answer title-disjoint natural
comparison questions, and can a deterministic raw-token selector retain that
information in two 48-token windows?

This is an information upper bound. The 9B reader is not a candidate
component, teacher, compiler, training target, or serving dependency. Passing
does not show that the unchanged 36.6M graph can execute the read.

## Frozen artifacts

- raw corpus:
  `data/hotpot-real-prose-t12/candidate-raw-prose.jsonl`, SHA-256
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`;
- sealed evaluator:
  `data/hotpot-real-prose-t12/sealed-evaluator.jsonl`, SHA-256
  `5a0413f53bfc0fc73b5e66c941c708077624908095934af740c84d234b9fea6b`;
- T32 Stage-0 result:
  `results/scale-referenced-whole-record-t32-stage0.json`, SHA-256
  `cc4882785b27ce7e19ea443783dbeadc22e859a06e286804199d0210f656e232`;
- packing/selection tokenizer: `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;
- oracle reader: `Qwen/Qwen3.5-9B`, revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`;
- reader runtime: Transformers `5.14.1`, PyTorch `2.11.0+cu128`, BF16 weights
  on the retained H100.

Reader tokenizer/config inspection is allowed before this freeze solely to
define the exact chat continuation. No evaluator row, answer, or support field
may be read before the preregistration is frozen.

## Frozen routing and record decode

Use the existing label-blind longest-nonoverlapping-title rule over all 2,405
raw corpus titles. Order the two titles by occurrence in the question. Exactly
two must be found in every evaluator row.

For each routed document, tokenize exactly `title + "\n" + text` with the
pinned packing tokenizer and no added special tokens. Retain the first 128 IDs,
pad shorter records with ID zero, pass through the frozen T32 pack/unpack
functions, and decode only the recovered unpadded IDs with cleanup disabled.
Re-encoding must reproduce the identical retained ID sequence.

The fixed shuffle maps every title to the title 997 positions later in the
lexically sorted 2,405-title list modulo 2,405. It must be a derangement and
must keep the two routed documents distinct.

## Frozen aliasing

Replace the two matched title spans in the question, in occurrence order, with
`Entity A` and `Entity B`. In each evidence string, replace case-insensitive
occurrences of that string's own source title with the corresponding alias.
For shuffled evidence, the wrong source title is replaced by the requested
alias. This retains binding while suppressing direct named-fact recall.

## Frozen raw-token local selector

The selector receives the original question, its two matched title spans, and
one recovered record at a time. It receives no answer, supporting title,
supporting sentence, parser output, embedding, pretrained-model state, or
reader score.

1. Replace the two title spans in the original question by empty strings and
   tokenize the remainder with the packing tokenizer.
2. From the 2,405 stored 128-token corpus records, compute document frequency
   `df(v)` for each token ID using presence/absence only.
3. Discard query token ID zero and every query token with
   `df(v) >= 0.25 * 2405`.
4. Let `Q` be the remaining unique query token IDs and
   `idf(v)=ln((2405+1)/(df(v)+1))`.
5. For every contiguous 48-token window `W_j` of the unpadded recovered
   record, compute

   \[
   s(j)=\sum_{v\in Q}idf(v)\,1[v\in W_j].
   \]

6. Select the maximum-score window, breaking ties by the smallest start
   position. If the record is shorter than 48 tokens, use it all. If `Q` is
   empty or every score is zero, the earliest window is selected.

This block is explainable as exact token equality, a frozen shared token
weight, a length-48 box sum, and argmax. It is not claimed to fit the physical
reader yet.

## Frozen conditions

For every evaluator question construct exactly five conditions:

1. `question_only`: both evidence fields contain `[omitted]`;
2. `correct_full128`: the two correct decoded T32 records;
3. `shuffled_full128`: decoded records from the fixed wrong documents;
4. `correct_local48`: the two deterministic selected windows from the correct
   records;
5. `shuffled_local48`: the same selector applied independently to the wrong
   records.

The question and answer distribution are identical across conditions.

## Frozen prompt

System message:

```text
Use only the supplied evidence. Answer the comparison question with exactly yes or no. Entity names have been replaced consistently. Do not use outside knowledge and do not explain.
```

User message:

```text
Question: {aliased_question}

Evidence for Entity A:
{evidence_a}

Evidence for Entity B:
{evidence_b}

Answer (yes or no):
```

Render with the pinned Qwen chat template, `enable_thinking=False`, and an
assistant generation prompt.

## Complete forced-string likelihood

Do not decide from one BF16 output logit. Teacher-force the complete rendered
assistant response for each candidate and sum FP32 log-softmax probabilities
over every continuation token. Tokenizer inspection fixed the two complete
continuations as:

```text
yes: [9405, 248046, 198]
no:  [2083, 248046, 198]
```

They decode to `yes<|im_end|>\n` and `no<|im_end|>\n`. The rendered prompt must
be an exact prefix of both full chats. Predict `yes` iff its total continuation
log likelihood is greater than `no`; predict `no` iff lower. A non-finite or
exactly tied score is an integrity failure. No generation, sampling,
calibration, threshold fit, retry, length normalization, or answer parsing is
allowed.

All prompts and scores are built before the source accesses answer values.
Labels are then read once for final scoring only.

## Mandatory gates

All are required:

1. artifact hashes, package versions, model revisions, exact schemas, and
   CUDA/H100 identity match;
2. every evaluator row routes exactly two distinct titles; the shuffle is a
   distinctness-preserving derangement;
3. every record pack/unpack and decode/re-encode is exact with at most 128
   unpadded IDs;
4. the selector uses only the frozen raw inputs; window lengths, starts,
   scores, zero-score rate, and selected-token digests are recorded;
5. both answer classes exist, but prompts and selector outputs are frozen
   before labels are accessed;
6. both continuation token sequences and prompt-prefix relations match the
   frozen values; all total log likelihoods and margins are finite and
   nonzero;
7. `correct_full128` accuracy is at least 80%;
8. `correct_full128` exceeds both `question_only` and `shuffled_full128` by at
   least 15 absolute percentage points;
9. `correct_local48` accuracy is at least 75% and differs from
   `correct_full128` by at most five absolute points;
10. `correct_local48` exceeds `shuffled_local48` by at least ten absolute
    percentage points;
11. predictions, labels, per-condition accuracies, paired causal changes,
    likelihood-margin summaries, prompt/token maxima, hashes, package/hardware
    metadata, elapsed time, energy, and zero retries are recorded.

## Decision rule

- **Pass:** admit only a new reader-placement theorem and same-graph physical
  microbenchmark. Do not admit training.
- **Fail:** close T32 as the active natural payload. Do not change record
  length, window length, selector, shuffle, prompt, reader, scoring rule, or
  thresholds after seeing the result.
