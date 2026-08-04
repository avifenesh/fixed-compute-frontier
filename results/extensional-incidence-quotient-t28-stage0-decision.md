# Extensional incidence quotient T28 — Stage-0 decision

Status: **FAIL — EXACT POSITIVE CO-INCIDENCE BRIDGE CLOSED**  
Date: 2026-07-31

## Verdict

The recovery theorem is valid, but its observable bridge is absent at the
required scale in the frozen raw corpus.  T28 stops before semantic scoring,
model training, physical packing, or GPU work.

## Decisive evidence

- 48,352 deduplicated candidate edges produced 46,233 distinct entity--value
  tuples and 48,124 surface patterns.
- Only 1,729 tuples, **3.7398%**, appeared under multiple patterns; the gate was
  25%.
- Only 620 of 2,405 documents, **25.7796%**, contained any bridged tuple; the
  gate was 50%.
- Requiring a surface pattern itself to occur on at least two tuples left 167
  nodes.  None shared a tuple with another recurring pattern: **zero graph
  edges** and 167 singleton components.
- Qualifying components: **0**, versus the required 64.
- Qualifying document coverage: **0%**, versus the required 50%.

Integrity passed: the corpus hash and field boundary matched, all source spans
regenerated, reverse document order produced the identical edge/component
objects, eight tiny-world tests passed, and no evaluator or model was read.

## Interpretation

Exact tuple equality is a genuine semantic bridge when independently worded
statements repeat the same facts.  This corpus mostly presents each
entity--value fact once.  The event lattice therefore contains many candidate
surfaces but no cross-instance equations capable of choosing a shared relation
coordinate.

This explains why Universal-Schema-style systems require matrix completion,
external schemas, embeddings, or much denser repeated evidence.  Adding any of
those now would introduce a new latent-identifiability hypothesis; it is not an
implementation refinement of T28.

## Retained result

Retain the exact and noisy extension-recovery theorems and the witnessed raw
event enumerator as diagnostics.  Do not retain T28 as a compiler candidate.
The next Stage--2 direction must work with one-shot facts or provide a different
raw-observable intervention; it cannot assume duplicate tuple evidence.
