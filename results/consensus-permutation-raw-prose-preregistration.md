# Consensus-permutation raw-prose compiler — Stage 0 preregistration

Status: frozen before execution  
Date: 2026-07-31

## Claim under test

A training-only compiler can receive only raw token sequences, recover a
latent factual table from redundant noisy prose by global permutation
agreement, and derive held-out compositions without being given entity IDs,
relation IDs, value labels, template labels, slot positions, or clean facts.

This is not yet a production-model claim.  It is the cheapest fatal test of the
missing information path between prose and the already validated digital
relation plane.

## Why the compiler can exceed one model read

The compiler is not assumed to understand a sentence better than the deployed
model.  It receives many independently corrupted surface views of the same
latent table.  After literal frame induction, view `v` yields a table

```text
X_v = P_v X Q_v + noise,
```

where `P_v` permutes entity aliases and `Q_v` permutes relation paraphrases.
The compiler searches for the permutations that make complete row-signature
multisets agree.  For `R` independent 16-way relations, two distinct entities
collide in all relation values with probability `16^-R`; the union bound is

```text
Pr(any entity-signature collision) <= N(N-1) / (2 * 16^R).
```

With `N=128, R=6`, this is below `2.5e-4` per view.  Wrong relation
permutations must reproduce the entire multiset of 128 six-symbol signatures,
which is much stronger than matching marginal frequencies.

Each raw fact mention is independently corrupted with probability `epsilon`.
For an odd number `m` of repetitions and `epsilon < 1/2`, majority recovery of
one fact fails with probability at most

```text
exp(-2 m (1/2 - epsilon)^2).
```

The experiment reports the union bound over every entity, relation, view, and
paraphrase.  This is the exact source of the compiler's extra reliability:
global repeated evidence, not a pretrained teacher or hidden labels.

## Frozen world

- 128 latent entities;
- 6 independent categorical relations;
- 16 shared opaque value words;
- 3 mutually disjoint entity-alias vocabularies;
- 2 lexically disjoint prose frames per relation and alias view;
- 63 mentions of every fact in every frame;
- independent 15% contradictory value replacements;
- shuffled raw sentences supplied as strings.

The compiler may split punctuation and whitespace.  It receives no generator
objects or latent metadata.  Slot positions are inferred by masking every pair
of token positions and finding repeated literal frames.  The larger variable
vocabulary is inferred as the entity slot; the smaller is inferred as the
value slot.

## Frozen gates

All must pass:

1. Every raw sentence is assigned to exactly one induced frame, and the exact
   number of surface frames and alias views is recovered.
2. Direct fact accuracy, entity-alias alignment, and relation-paraphrase
   alignment are each exactly 100% against sealed generator metadata.
3. Bitwise-XOR answers for every entity and every ordered relation pair are
   exactly 100%, although no composition statement or answer appears in the
   corpus.
4. The two-channel SwiGLU identity used for deployed XOR,
   `(SiLU(beta*x)-SiLU(-beta*x))*y/beta = x*y`, has maximum numerical error
   below `1e-12` on bipolar bits.
5. A null corpus whose views contain independent tables is rejected rather
   than forced into an alignment.
6. A corpus with duplicate complete entity signatures is rejected as
   non-identifiable rather than resolved using hidden identity.
7. The registered majority-error union bound is below 1%.

## Kill interpretation

- Failure of exact recovery closes this concrete consensus-permutation
  extractor at Stage 0; no GPU is rented.
- Passing does **not** establish real-language extraction.  Atomic aliases,
  repeated literal frames, a shared value vocabulary, and independent noise
  remain simplifications.
- Passing admits only an integration test: write the recovered table into an
  unchanged Transformer, execute XOR inside its ordinary attention/SwiGLU
  graph, and compare against frontier gradient training given both the same raw
  corpus and the same compiler-derived labels.

## Anti-cheating controls for the admitted integration

The strongest gradient control must receive the compiler's recovered direct
facts and composition targets, rendered as training examples, with at least 2x
the candidate's charged work.  Thus the candidate cannot win merely because a
preprocessor exposed information that the baseline never saw.  The compiler,
all data transformation, and all failed alignment searches are charged as
training cost; the compiler is deleted at export.
