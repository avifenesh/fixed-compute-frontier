# Raw-prose equality plane T10 — representation preregistration

Status: frozen before GPU execution  
Date: 2026-07-31

## Exact claim under test

Starting only from a raw prose corpus, the Stage 0b compiler recovers opaque
facts and surface equivalences.  A deleted-at-export writer then places those
recovered facts into a standard 36.6M-parameter causal Transformer.  The
unchanged model must answer direct factual queries and execute a new equality
composition inside its ordinary attention and SwiGLU graph.

No generator-private entity, relation, value, template, or slot IDs may reach
the writer.  This is a representation integration gate, not yet a training
breakthrough test.

## Frozen corpus

- 384 latent entities;
- 8 independent 16-way categorical relations;
- 2 disjoint opaque alias vocabularies;
- 2 disjoint prose frames per relation and alias view;
- one clean mention per fact/frame, for 24,576 raw statements;
- all strings shuffled before compilation.

The reduced clean regime isolates integration.  Stage 0b already sealed exact
recovery under 15% contradictions and 63 repeats.

The protected natural train and validation streams are rebuilt from the pinned
SmolLM2 tokenizer and FineWeb-Edu-dedup revisions used by the earlier 36.6M
screens.  T10 must reject execution unless the resulting files are exactly
`100,195,056` and `4,202,496` bytes with SHA-256 values
`1871a8a790e2b2ae5273f1a46bc9fa2e39cbe95d5cb7537bde5a2b3e35cb464a`
and `889188f0e4c43cd49b2933ac462e8bd367520f15fe08d324f169f7eca962ce7b`.

## Surface-to-model rule

The model vocabulary is unchanged.  The 820 required alias, frame-anchor,
value, query, and answer symbols are assigned only to token rows absent from
the protected natural train and validation streams.  Ordinary frame words are
mapped to existing used rows for later gradient controls.

For each recovered relation, half of its recovered paraphrase anchors receive
role A and half receive role B.  A query is four tokens:

```text
[entity alias, role-A relation anchor, role-B relation anchor, operation]
```

The query protocol is intentionally minimal.  It tests whether recovered
surface equivalences can address and compose the plane; it does not yet prove
general natural-question understanding.

## In-graph algebra

Each fact is stored as four bipolar bits.  Block 0 uses the two existing
unrotated attention heads:

- head 0 copies the selected alias's 32-bit fact row;
- head 1 simultaneously copies the A and B three-bit relation codes into
  disjoint registers.

Block 1 selects two four-bit values using 64 shared SwiGLU channels.  Block 2
computes four bitwise products with two channels per bit:

```text
(SiLU(beta*x) - SiLU(-beta*x)) * y / beta = x*y.
```

For bipolar encodings, a bit product is `+1` exactly when the two bits agree.
The tied output rows classify equality by the signed margin

```text
sum(bit_products) - 3,
```

which is positive only when all four bits agree.  Direct value rows decode the
selected A register by Hamming dot product.

## Frozen gates

All must pass under BF16 autocast on the rented GPU:

1. The compiler receives raw strings only and exactly recovers the 3,072 facts,
   768 aliases, 32 relation paraphrases, and two alias views.
2. Deployed parameter count, checkpoint tensor shapes, inference graph, KV
   state, precision, and per-token serving FLOPs are unchanged.
3. Direct accuracy is 100% for every alias, relation, and role-A paraphrase,
   with zero errors and positive worst-case logit margin.
4. Equality accuracy is 100% for every alias, ordered relation pair, role-A
   paraphrase, and role-B paraphrase, with zero errors and positive worst-case
   logit margin.
5. At least two aliases and four surface anchors address every latent fact
   without consulting the compiler at inference.
6. No latent generator object is accepted by the weight writer.
7. All program coordinates remain frozen and exact after a destructive
   perturb-and-enforce self-check.

## Admission and kill rule

Any error closes this concrete read/compose construction before training.  A
pass admits matched training controls only.  Those controls must see the same
raw prose plus the compiler-derived direct and equality targets; otherwise a
candidate win could be only preprocessing privilege.
