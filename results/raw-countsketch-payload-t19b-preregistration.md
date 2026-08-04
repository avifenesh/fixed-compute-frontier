# Raw CountSketch payload T19b — sufficiency preregistration

Status: **frozen before result**  
Date: 2026-07-31

## Key shift

T19a showed that explicit title routing is learnable by both compiled and dense
training arms.  T19b therefore gives future arms the same address protocol and
asks whether a raw-only payload can retain useful document evidence inside the
existing FFN matrices.

This is a representation and physical-budget screen.  It does not train an LM
and cannot establish a smarter-model result.

## Exact physical ledger

For 2,405 documents, 4,150 active title-token rows, a 32-dimensional title
code, and a 220-dimensional document payload:

| write | entries |
|---|---:|
| title token codes | `4,150 * 32 = 132,800` |
| per-document FFN gate key | `2,405 * 32 = 76,960` |
| per-document gate threshold | `2,405` |
| per-document up constant | `2,405` |
| per-document down payload | `2,405 * 220 = 529,100` |
| **total** | **743,670** |

The total is 64 entries below T11's demonstrated 743,734-entry isolation
budget.  Documents are assigned deterministically to one channel across three
ordinary 1,024-channel FFNs.  No new parameter, tensor, operation, lookup,
adapter, or model class is allowed.

## Raw-only payload

The compiler reads only each document's `document_id`, `title`, and `text`.
It lowercases and tokenizes alphanumeric words, then emits:

- word unigrams;
- adjacent word bigrams;
- for every four-digit year from 1000 through 2099, an exact-year feature and
  its decade feature.

For each feature, term weight is `(1 + log(count)) * idf`, with
`idf = log((N + 1) / (document_frequency + 1)) + 1`.  SHA-256 independently
selects one of 220 buckets and a sign.  Signed weights are accumulated and the
document vector is L2-normalized.  No question, answer, supporting-page field,
parser, pretrained model, embedding model, or fitted semantic projection is
available to the compiler.

The vector is quantized through BF16 before the physical write.  Every payload
must remain finite and nonzero; minimum quantized cosine to its FP32 source
must be at least 0.9999.

## Fixed lightweight reader screen

The reader is deliberately separate from the compiler.  It receives the 146
existing development-training questions and their yes/no labels.  It finds
the two longest nonoverlapping raw titles occurring in each question; no
support annotations are used.  After removing those title spans, it sketches
the remaining question with the same frozen feature map and corpus IDF.

For question sketch `q` and document sketches `a,b`, the symmetric feature
vector is the concatenation of:

- `min(q*a, q*b)`;
- `max(q*a, q*b)`;
- `a*b`;
- `abs(a-b)`;
- the scalar dot products `q.a`, `q.b`, and `a.b`, plus the minimum and maximum
  of the first two.

Features are standardized using training statistics only.  A single
float64 logistic regressor is fitted by LBFGS with fixed L2 coefficient 0.01,
zero initialization, threshold 0.5, at most 200 iterations, and no sweep or
retry.  Evaluation uses T12's 104 development questions.  The evaluator is
development evidence, not a future final set.

## Gates

All gates are required:

1. Frozen corpus/evaluator/natural-file hashes match and compiler documents
   contain only the three raw fields.
2. Exactly two raw titles are found in every train and evaluation question
   without support annotations.
3. The physical write count is exactly 743,670 and at most 743,734.
4. Candidate and untouched control models have identical ordinary state-dict
   keys, shapes, parameter count, and class.
5. Every FP32 and BF16 payload is finite/nonzero, and minimum BF16 cosine is at
   least 0.9999.
6. The fixed reader reaches at least 75% evaluation accuracy and at least 10
   absolute points above T12's 60.5769% trained semantic probe.
7. Training and evaluation both contain yes and no answers; the optimizer is
   finite and uses no retry.

Failure closes this exact 220-dimensional unigram/bigram/year CountSketch as a
useful payload.  Do not rescue it by changing hashes, features, dimension,
classifier, regularization, threshold, or labels.  Passing admits an actual
from-zero standard-Transformer payload experiment with matched dense controls.
