# Exact raw-key oracle T19e — preregistration

Status: **frozen before result**  
Date: 2026-07-31

## Question

T19d proved that the raw corpus fits the aggregate digital-plane budget, but
an entropy stream is sequential rather than query-addressable.  A hash table,
Bloom filter, locally decodable code, or FFN associative table can improve
storage and lookup only after a useful key has been defined.

T19e asks the prior question with the strongest exact lexical control:

> If every raw unigram, adjacent bigram, year, and decade key is available
> exactly, with no collision, quantization, capacity limit, or retrieval
> error, can the fixed reader solve held-out questions?

This is an upper-bound sufficiency screen.  It is intentionally more
privileged than an admissible served model and cannot establish a production
claim.  Failure means that compacting the same lexical keys cannot solve the
semantic interface.

## Frozen raw key space

The compiler reads only the 2,405 records containing `document_id`, `title`,
and `text`.  It applies T19b's already frozen raw transformation:

- lowercase alphanumeric unigrams;
- adjacent lowercase alphanumeric bigrams;
- exact-year and decade keys for four-digit years in 1000 through 2099;
- `(1 + log(count)) * idf` weights;
- corpus IDF `log((N + 1) / (df + 1)) + 1`;
- per-document L2 normalization.

Unlike T19b, every distinct key gets its own coordinate.  There is no hashing,
collision, truncation, learned embedding, parser, support annotation,
pretrained model, question, or answer in the compiler.

## Exact symmetric reader

For each question, the reader finds the same two longest nonoverlapping raw
titles and removes their surface spans.  The remaining question is encoded in
the exact corpus vocabulary and L2-normalized.  Let `q` be this sparse query
vector and `a,b` the two exact document vectors.

The candidate vector is exactly the uncompressed analogue of T19b:

- `min(q*a, q*b)` for every raw key;
- `max(q*a, q*b)` for every raw key;
- `a*b` for every raw key;
- `abs(a-b)` for every raw key;
- scalar `q.a`, `q.b`, `a.b`, `min(q.a,q.b)`, and `max(q.a,q.b)`.

Only the 146 training questions and yes/no labels fit the reader.  The 104
development rows contribute only `question` and `answer`; their support titles
and supporting-sentence IDs are forbidden inputs.

Features are standardized with training statistics only.  One float64
logistic regressor is fitted from zeros with mean binary cross entropy, L2
coefficient 0.01, L-BFGS-B, at most 200 iterations, threshold 0.5, and no
sweep or retry.

Two causal controls are frozen:

1. **query-only:** the same optimizer sees only `q`, proving whether question
   wording alone predicts the label;
2. **document shuffle:** the fitted exact reader is evaluated after one fixed
   seed-19,005 permutation of development document payloads, without refit.

## Gates

All gates are required:

1. Frozen corpus, train-QA, evaluator, and preregistration hashes match.
2. The compiler sees exactly the three raw document fields and constructs a
   collision-free coordinate for every distinct frozen lexical key.
3. Exactly two raw titles are found in every training and development
   question without consulting support annotations.
4. Both answer classes occur in train and development; the one-shot optimizer
   and all logits/weights are finite.
5. Exact candidate training accuracy is at least 95%.
6. Exact candidate development accuracy is at least 75% and at least 10
   absolute points above T19b's 60.5769% compressed lexical result.
7. Exact candidate beats the query-only development control by at least 10
   absolute points.
8. Shuffling development document payloads reduces candidate accuracy by at
   least 10 absolute points.

## Decision rule

- **Fail:** close exact lexical keys as the semantic key system.  Perfect
  hashing, Bloom filters, wider lexical sketches, and locally decodable codes
  over the same keys are dominated because the collision-free oracle failed.
  The next admissible object must learn a shared relational key/value algebra
  from raw prediction, not merely improve lookup.
- **Pass:** lexical key sufficiency is established only as an oracle.  Admit a
  separate fixed-budget/BF16/ordinary-Transformer compression gate; do not
  claim a smarter model from T19e.

No feature change, stemming rule, regularization change, classifier change,
threshold change, sentence selection, or post-result rescue is allowed.
