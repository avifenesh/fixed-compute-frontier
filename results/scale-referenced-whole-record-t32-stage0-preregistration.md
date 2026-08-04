# T32 scale-referenced whole-record codec — frozen Stage-0 preregistration

Status: **FROZEN BEFORE IMPLEMENTATION OR CORPUS CENSUS**  
Date: 2026-07-31

Pre-run correction: the source JSONL schema, checked before implementation,
contains the raw integrity key `document_id` in addition to `title` and `text`.
The identifier is admitted only for uniqueness and order-invariance checks. It
is forbidden from entering the packed record. No numerical or corpus result
was observed before this correction.

## Scope

This CPU-only gate tests the exact T32 codec, its BF16 common-scale boundary,
the frozen corpus domain, and the prospective write ledger. It does not read
questions, answers, support fields, model weights, or natural labels. It does
not train a model or authorize a GPU.

## Frozen inputs

- Corpus: `data/hotpot-real-prose-t12/candidate-raw-prose.jsonl`
- Expected corpus SHA-256:
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`
- Required source fields: `document_id`, `title`, `text`
- Codec inputs: token IDs derived only from `title + "\n" + text`
- Tokenizer: `HuggingFaceTB/SmolLM2-135M`
- Revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- Token domain: first 128 IDs without added special tokens, padded with ID 0

The sealed evaluator and every question/answer/support field are forbidden.

## Frozen record format

Use exactly the P2 construction in
`results/scale-referenced-whole-record-t32-paper.md`:

- 127 triples `(low6, middle6, 4*high4+aux2)` in coordinates `0..380`;
- `aux[0:8]` stores token 127 in little-endian two-bit chunks;
- `aux[8:12]` stores the unpadded length in little-endian two-bit chunks;
- all later auxiliary values are zero;
- coordinate 381 is the real scale reference `1`;
- coordinates 382 and 383 are zero;
- base-64 digit amplitude is `2*d-63` only in the numerical replay, not in the
  integer serialized record.

No checksum, second record, overflow path, entropy code, changed radix, or
token truncation other than the declared first-128 domain may be added after
seeing results.

## Mandatory unit worlds

1. token IDs `0`, `1`, `63`, `64`, `4095`, `4096`, `49151`, and `65535`;
2. lengths `0`, `1`, `127`, and `128`;
3. all-zero, all-maximum, alternating-boundary, and seeded random records;
4. every base-64 digit and all 63 decision boundaries;
5. reversed corpus order;
6. malformed records with nonzero reserved cells or nonzero auxiliary values
   after index 11 must be rejected.

## Numerical replay

For every digit `d=0..63`, every boundary, and scale grid

```text
2^-12, 2^-8, 2^-4, 2^-2, 1, 2^2, 2^4, 2^8, 2^12
```

round the scaled digit amplitude and scale reference to BF16, cast to FP32,
and decode by all fixed linear boundary comparisons. Require the exact digit.

Repeat with deterministic signed perturbations of magnitude `0.005*s` applied
independently to the rounded digit and reference values before FP32 boundary
comparisons. All four sign combinations must decode exactly. This is a
registered robustness probe, not a claim about arbitrary model residuals.

## Corpus census

Report:

- corpus/tokenizer/source/preregistration hashes;
- document count, min/mean/max unpadded length, and number truncated at 128;
- minimum/maximum token ID;
- exact pack/unpack count and failures;
- serialized integer-record SHA-256 in title-sorted order;
- reverse-order determinism;
- active and reserved cells per document;
- payload writes and prospective write ledger;
- wall time, CPU time, peak RSS, Python, NumPy, and Torch versions;
- explicit confirmation that CUDA is hidden/unavailable.

## Frozen prospective write ledger

Charge:

- 132,800 title-code entries;
- `2,405 * 32 = 76,960` gate-key entries;
- 2,405 thresholds;
- 2,405 up constants;
- `2,405 * 382 = 918,710` payload entries;
- 147,456 shared three-coordinate token codes for all 49,152 token rows;
- a fixed 400,000-entry shared-reader reserve.

The total is 1,680,736 entries, 4.5950% of the 36,577,152-entry model. The
reader reserve is charged even though no reader is implemented at Stage 0.

## Mandatory gates

All must pass:

1. input hash, exact source schema, codec-input field boundary, tokenizer
   identity/revision, and CPU-only
   boundary match;
2. every unit world and all 2,405 corpus records round-trip exactly;
3. all observed token IDs are below 49,152 and every stored length is at most
   128;
4. integer records use exactly 382 active plus two zero reserved coordinates;
5. every BF16 scale/boundary and registered perturbation replay decodes
   exactly;
6. title-sorted output is invariant to reversed input order;
7. the prospective ledger equals 1,680,736 and is below 5% of model entries;
8. zero forbidden-field reads, model-weight reads, evaluator reads, or GPU
   operations.

## Decision rule

- **Pass:** admit only a separately frozen full-record/local-window natural
  information oracle. No reader circuit, training, or GPU is automatically
  admitted.
- **Fail:** close T32. Do not change radix, width, perturbation, tokenizer,
  record count, ledger, or threshold after observing the result.
