# XOR functional record T24 — natural-QA sufficiency oracle preregistration

Status: **FROZEN BEFORE ORACLE EXECUTION**  
Date: 2026-07-31

## Question

Does the exact T24 raw quotient retain enough typed evidence to improve
title-disjoint natural comparison QA, even when storage, hashing, and decoding
are made perfect?

This is a fatal information gate before any learned language-to-record model.
It is deliberately stronger than the deployable candidate: the oracle decodes
every registered T24 edge to its raw anchor and target strings without paying
the 220-cell read or physical Transformer cost.  If this decoded representation
cannot beat prior lexical and query-only controls, a neural interface cannot
repair the missing information merely by learning the same table.

## Frozen representation

Rebuild the exact T24 quotient from the raw corpus only.  For every registered
edge `(skeleton, target)` in a document, emit:

- `anchor:word` once for every `(offset, word)` in the skeleton;
- `target:word` once for the target.

Counts use `1 + log(count)` and corpus IDF, followed by L2 normalization,
exactly as in T19e.  No sentence, parser, stemmer, synonym, QA label, support
annotation, embedding, or pretrained model enters the representation.

After the two longest nonoverlapping corpus titles are removed from a question,
each remaining raw query word emits both `anchor:word` and `target:word` when
that coordinate exists.  This gives the oracle both possible roles without
using the answer.

The symmetric reader is exactly T19e's fixed construction over the two routed
document vectors: query/document minimum and maximum interactions,
document-document products and absolute differences, and five scalar dots.
Use T19e's unchanged L-BFGS logistic reader, L2 coefficient `0.01`, 200-iteration
limit, 0.5 threshold, and seed-19,005 document shuffle.  Fit only the existing
146 QA-training rows and evaluate once on the existing 104 title-disjoint
development rows.

## Frozen controls and gates

Report candidate, query-only, and shuffled-document results.  All gates are
required:

1. raw corpus, training QA, evaluation QA, T24 Stage-0 result, and this
   preregistration match their frozen hashes;
2. the rebuilt quotient exactly has 3,392 skeletons and 14,146 edges;
3. every compiler input field is one of `document_id`, `title`, and `text`;
4. both QA classes and exactly two label-blind routed titles exist for every
   row;
5. optimization and all scores are finite;
6. candidate training accuracy is at least 95%;
7. candidate evaluation accuracy is at least 75%;
8. candidate is at least ten absolute points above T19e's collision-free raw
   lexical oracle (56.7308%), T19b (60.5769%), and its own query-only reader;
9. shuffling the routed document records reduces evaluation accuracy by at
   least ten absolute points.

Failure closes the current anchor-window quotient as the semantic payload for
natural comparison QA.  It does not close the XOR read algebra.  No feature,
normalization, stemmer, threshold, regularizer, classifier, or skeleton change
is allowed after seeing the result.

Passing admits one from-zero learned-interface experiment; it is still not a
serving or smarter-model result.
