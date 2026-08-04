# Extensional incidence quotient T28 — Stage-0 CPU census preregistration

Status: **FROZEN BEFORE IMPLEMENTATION OR CENSUS**  
Date: 2026-07-31

## Question

Does the frozen raw corpus contain a large, directly witnessed graph that
connects different surface patterns through repeated use on the exact same
entity--value tuple, strongly enough to support T28's shared relation codes?

This stage measures existence of the bridge.  It does not train a model,
inspect natural QA, claim semantic purity, or use a GPU.

## Frozen input boundary

- Only `data/hotpot-real-prose-t12/candidate-raw-prose.jsonl` may be read.
- Expected SHA-256:
  `a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a`.
- Only each row's `title` and `text` fields may be accessed.
- The sealed evaluator, answers, questions, supporting titles, supporting
  sentence IDs, pretrained models, parsers, taggers, embedding models, and
  external knowledge are forbidden.

## Frozen raw tokenizer and sentences

Casefold Unicode text.  Tokenize with the standard-library regular expression
`\w+|[^\w\s]`.  A token is lexical when it contains at least one Unicode word
character.  Split sentences after `.`, `?`, or `!` tokens and retain the
punctuation token.

Tokenize every document title by the same rule.  A sentence is eligible only
when it contains a casefold-exact contiguous occurrence of its own title token
sequence.  Replace every nonoverlapping occurrence of that own-title sequence
by one `<S>` token.  Do not infer implicit subjects or resolve pronouns.

## Frozen value lattice

Across all sentences, enumerate every contiguous lexical-token span of length
one through four that:

1. does not overlap an own-title occurrence;
2. starts and ends with lexical tokens;
3. contains no sentence-final punctuation;
4. has a normalized token tuple occurring in at least two and at most
   `floor(N/20)` distinct documents, where `N` is the raw document count.

Document frequency is computed before constructing pattern edges and uses only
the same raw corpus.  It is a deterministic rarity boundary, not a semantic
word list.

For each eligible subject sentence and value span, emit:

- tuple `u = (document_title, normalized_value_tokens)`;
- pattern `p` equal to the complete normalized sentence after replacing every
  own-title occurrence by `<S>` and the chosen value span by `<V>`;
- a witness containing document index, sentence index, original token bounds,
  subject bounds, and the exact source tokens.

Deduplicate identical `(u,p)` edges while retaining the lexicographically first
witness.  No score or selector chooses a likely relation span.

## Frozen bridge graph

Create one node for every pattern with at least two distinct tuple neighbors.
Connect distinct patterns `p` and `q` iff at least one exact tuple `u` is
incident to both.  Record every shared-tuple witness.  Relation candidates are
the undirected connected components.

This realizes the constructive theorem:

> If patterns of each latent relation form a connected co-incidence subgraph
> and no tuple joins patterns from different relations, graph components recover
> relation classes up to permutation.

The census can measure connectivity and recurrence.  It cannot prove the
no-cross-relation assumption on natural text; that remains an explicit oracle
boundary.

## Independent reference checks

The implementation must include:

1. an obvious set-based graph constructor;
2. an independent tuple-to-pattern inverted-index constructor;
3. exact agreement of node sets, edge sets, and connected components on
   exhaustive hand-written tiny worlds;
4. deterministic agreement under reversed document order;
5. exact witness regeneration back to source tokens;
6. adversarial worlds for singleton patterns, coextensive relations, one
   polysemous bridge, punctuation, repeated titles, and overlapping value
   spans.

## Frozen metrics

Report at minimum:

- document, sentence, token, and eligible-subject counts;
- distinct value spans and their document-frequency distribution;
- emitted and deduplicated edge counts;
- distinct tuples and patterns;
- fraction of tuples incident to at least two patterns;
- fraction of documents containing at least one bridged tuple;
- graph nodes, edges, components, and component-size distributions;
- number of components with at least two patterns, four subject titles, and
  four tuples;
- fraction of all documents covered by those qualifying components;
- retained fact counts per document if every qualifying component receives a
  deterministic ID and each document retains distinct `(component,value)`
  records;
- documents exceeding 24 records;
- relation-class and value-dictionary cardinalities against 4,096 and 65,536;
- source-regeneration failures and independent-constructor mismatches;
- peak RSS, CPU time, wall time, and serialized logical sizes of the edge and
  graph objects.

Component IDs are assigned by the lexicographically smallest pattern in each
component.  Within a document, facts are ordered by component ID then value;
the first 24 are retained only for the budget census.  This ordering is not an
admitted semantic selector.

## Mandatory admission gates

All must pass:

1. Input hash, field boundary, tokenizer, subject rule, value lattice, and
   deterministic IDs match this preregistration.
2. Zero witness-regeneration failures and zero disagreement between graph
   constructors or document-order runs.
3. At least 25% of distinct eligible tuples are incident to two or more surface
   patterns.
4. At least 50% of all documents contain at least one bridged tuple.
5. At least 64 components contain at least two patterns, four distinct subject
   titles, and four distinct tuples.
6. Those qualifying components cover at least 50% of all documents.
7. At most 4,096 qualifying components and at most 65,536 retained normalized
   value spans are required.
8. At least 95% of covered documents fit at most 24 distinct
   `(component,value)` facts; no overflow rescue or second record is allowed.
9. Every edge and graph object is finite; the complete census finishes on CPU
   without reading model weights or evaluator data.

These thresholds are deliberately breakthrough-sized.  A small but clean
relation graph is an information-extraction result, not the project's model
edge.

## Kill boundary

Failure closes the exact positive co-incidence bridge.  Do not rescue it by
lowering support, adding embeddings, lemmatizers, parsers, pretrained models,
evaluator queries, manual stopwords, fuzzy tuple matching, matrix completion,
negative sampling, a larger value span, or a different document-frequency
threshold after observing the census.

Passing admits only a separately preregistered semantic-purity and decoded
structured-record oracle.  It does not admit relation-encoder training, H100
work, physical packing, from-zero LM training, or a production claim.
