# XOR functional record T24 — natural-QA sufficiency decision

Status: **FAIL; CLOSE THE LOCAL ANCHOR-WINDOW QUOTIENT**  
Date: 2026-07-31

## Decision

Do not train a neural language interface for the current T24 payload.  Even an
uncompressed oracle that perfectly decoded every registered relation into its
raw anchor and target strings performed worse than query-only, shuffled
records, and the earlier lexical payload.

The XOR retrieval algebra remains a valid exact storage/read component.  What
is closed is the claim that four-word anchor-window skeletons and rare target
words form a useful sufficient statistic for natural comparison knowledge.

## Evidence

The fixed logistic reader fitted 145/146 training rows and then produced:

| condition | title-disjoint accuracy |
|---|---:|
| decoded T24 records | 45.1923% |
| same reader, shuffled records | 54.8077% |
| query only | 54.8077% |
| prior collision-free lexical oracle T19e | 56.7308% |
| prior lexical payload T19b | 60.5769% |

The correct records were 9.6154 points worse than both query-only and shuffled
records, and 15.3846 points worse than T19b.  They missed the 75% absolute gate
by 29.8077 points.  This is not a marginal reader or storage failure.

All integrity contracts passed:

- 3,392 skeletons and 14,146 edges were rebuilt exactly;
- compiler fields were only `document_id`, `title`, and `text`;
- all 250 questions routed exactly two titles without support labels;
- both classes were present and optimization was finite;
- no query had an empty feature representation.

## Interpretation

T24 Stage 0 established a constructive identity for storing any finite
query-to-seven-bit function in the fixed 220-cell plane.  This oracle now shows
that exact access is not enough: the chosen raw quotient mostly captures local
word-completion accidents, not occupation, type, chronology, membership,
medium, or other relations needed by the natural questions.

The next representation must change the source of semantic identifiability.
It cannot use a wider XOR table, another hash, a larger target vocabulary,
stemming, synonym expansion, a stronger classifier, or a neural wrapper over
the same local windows.  The remaining admissible architectural direction is
a from-zero relational learning contract whose state is equivariant to entity
renaming and whose relation program is invariant.  That contract must first
show that its extracted records beat a fully decoded information oracle before
physical storage is relevant.

## Retained result

Retain the exact XOR/Bloomier-style record as a possible future read substrate:
one fixed global family, 109 seven-bit variables stored in 220 four-bit cells,
101-relation maximum, zero retries, and 14,146/14,146 exact reads.  It may be
reused only if a genuinely different relation-bearing sufficient statistic
passes its own frozen natural-capability oracle.

## Evidence hashes

- Sufficiency result SHA-256:
  `a39559ca4f4ab21b2921330a2ddefaf853211050ac0b97b2669cae1c37546a20`
- Executed source SHA-256:
  `a95f5a30f8042133843f0be958f7cec47487131ee1b48c79f3c550f8eebd122f`
- Tests SHA-256:
  `e82b85a90b546c9ccd9dc2e393f7c0f5617ff2b2a957b79fdd2a12da2626e9d1`
- Frozen preregistration SHA-256:
  `71e5709e7d8d2da1f1fad7207dbca69e7dde5988b3fffec6390c4c860783b101`
- Retained Stage-0 result SHA-256:
  `e31fb880785c49c1a4f972cc5d8b1c8f10f8aba5a990c707cf975def35b6c99e`
