# XOR functional record T24 — Stage 0 decision

Status: **PASS; ADMIT ONE LEARNED LANGUAGE-INTERFACE EXPERIMENT**  
Date: 2026-07-31

## Decision

The exact read-algebra wall exposed by T23 is repaired at Stage 0.  A fixed
220-cell record can represent every registered raw-derived functional edge in
the real-prose corpus, and the value of a query is determined by a constructive
GF(2) identity rather than a learned latent coordinate convention.

This admits one learned-interface experiment.  It does not admit a physical
export or a smarter-model claim.

## Exact result

The frozen quotient contained 3,392 relation skeletons and 14,146
document-skeleton edges across 2,248 documents.  Target vocabularies required
at most seven bits, and the largest document had 101 registered relations.

One global hash family mapped every skeleton to three of 109 logical slots.
Every document incidence matrix had full row rank; no per-document seed or
retry was used.  Solving `A_d Z_d = Y_d` over GF(2) produced:

- 14,146/14,146 exact total reads;
- 11,313/11,313 exact train-split reads;
- 2,833/2,833 exact evaluation-split reads;
- exact odd-integer 16-level indices through BF16.

The frozen evaluation controls were:

| record condition | accuracy |
|---|---:|
| correct | 100.0000% |
| shuffled document | 22.2732% |
| zero | 24.2146% |
| random | 0.6354% |

Every control was more than 20 points below the correct record.

## What changed

T22 and T23 asked an ordinary reader to discover what an arbitrary 220-scalar
vector meant.  Even optimized continuous vectors were ignored.  T24 instead
defines meaning algebraically:

`read(d,s) = Z_d[h1(s)] XOR Z_d[h2(s)] XOR Z_d[h3(s)]`.

Because the compiler solves the corresponding linear equations, exact recall
is guaranteed for every registered key.  The writer is a finite GF(2) solver,
not a semantic predictor.

The table uses 109 seven-bit logical variables, stored as 109 low-nibble and
109 high-nibble four-bit cells plus two reserved cells.  That is exactly 220
cells or 880 physical bits per document.  The most information-dense document
required 319 semantic bits; unused physical bits are not reported as a
compression gain.

## Remaining wall

The record currently receives an already constructed raw skeleton key and
returns a lexical target rank.  A useful model must still learn, from raw prose
only:

1. language to the three compiled relation indices;
2. `(relation, rank)` to the correct target surface;
3. transfer from raw cloze views to unseen natural questions and composition.

The next experiment must isolate exactly this interface.  Dense controls get
the identical raw-derived query/target examples.  Correct, zero, shuffled, and
random records remain mandatory, as do protected natural NLL and document-
disjoint natural QA.  Failure closes this raw quotient as a natural-language
knowledge interface; changing table dimensions or hashes would not address
that failure.

## Prior-art boundary

XOR/Bloomier retrieval structures are prior art.  The retained finding is not
the data structure by itself.  It is that the project's real raw-only
functional quotient fits exactly in the already established per-document
digital plane and has an exact fixed-depth read identity suitable for an
ordinary-graph integration attempt.

## Evidence hashes

- Result JSON SHA-256:
  `e31fb880785c49c1a4f972cc5d8b1c8f10f8aba5a990c707cf975def35b6c99e`
- Executed source SHA-256:
  `657428ea4914abd7532911da44375788d5b48466d6fe497aefca399c4ebd10ae`
- Tests SHA-256:
  `bfd6a02906c80b62fbc4b992620aa4bfe0eeccb6e057c04c1055c58a042046d1`
- Frozen preregistration SHA-256:
  `be07f9de6c7ebbafaa9f0d832ce79bc5af07c57b3ac0630de2231ceaa695060f`
