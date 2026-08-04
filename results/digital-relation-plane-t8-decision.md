# Digital relation plane T8 — decision

Status: all 42 preregistered gates passed; breakthrough candidate, replicate at
a second model scale before final claim  
Date: 2026-07-31

## Result

Across three rotated model seeds, the same 110,776,960-parameter deployed causal
LM retained all 32,768 arbitrary 16-way entity–relation facts exactly after
1,000 mixed language/knowledge steps:

- accuracy 100.000% in every seed;
- zero errors among 98,304 total sealed fact evaluations;
- minimum correct-class logit margin 6.625 at every checkpoint;
- unseen random-query accuracy 6.616%, near 16-way chance (6.25%).

The twice-exposed controls received two independently shuffled complete table
passes plus 2,000 mixed steps.  Their terminal full-table accuracies were:

| seed | calibrated Muon 2x | AdamW 2x |
|---|---:|---:|
| 4,123 | 12.039% | 7.724% |
| 4,487 | 11.820% | 6.928% |
| 4,889 | 11.710% | 7.169% |

Muon 1x averaged 7.05%.  This is a qualitative memory-acquisition change, not a
sub-percent benchmark movement: exact retrieval replaces 28,823–30,498 errors
per control seed.

## Density result

The table has `16^32768 = 2^131072` possible states.  Any exact arbitrary-table
representation needs at least 131,072 logical payload bits.  The construction
uses 32,768 robust 16-level payload scalars—four logical bits each—for exactly
131,072 logical payload bits.

Thus the value payload meets the counting lower bound in 16-ary logical cells:
one scalar per fact.  Shared overhead is 5,443 additional nonzero entries,
including entity norm compensation, relation codes, routing, and one decoder;
it does not grow with the number of entity–relation facts at fixed dimensions.

This is not physical bit optimality.  The checkpoint scalars have more than
four physical precision bits, and the candidate freezes 20,361,151 entries,
mostly isolation zeros.  The important result is exact logical density inside
an unchanged ordinary Transformer graph.

## Language and resource result

Mean terminal metrics:

| arm | fact accuracy | natural NLL | wall time | energy estimate |
|---|---:|---:|---:|---:|
| digital plane + Muon 1x | 100.000% | 6.1221 | 84.0 s | 16.61 kJ |
| Muon 1x | 7.049% | 6.3178 | 105.4 s | 22.98 kJ |
| Muon 2x | 11.856% | 6.1351 | 207.0 s | 45.05 kJ |
| AdamW 2x | 7.273% | 6.1895 | 136.4 s | 30.34 kJ |

The candidate used 2.46x less wall time and 2.71x less estimated energy than
Muon 2x, while its mean natural NLL was also 0.21% lower.  Peak training HBM was
4.72 GB versus 4.16 GB for Muon, because the candidate retains masks/frozen
values during training.  Serving HBM, parameter count, graph, and FLOPs are
identical.

## Mechanism

The advance over T4's failed amplitude packing is exact norm compensation.
Every entity row is padded to the same squared norm, so RMSNorm preserves the
relative 16 amplitude levels.  A 6-bit relation query selects one of 64 scalar
cells through a shared SwiGLU router; a 48-channel shared triangular decoder
maps that cell to a four-bit output code.  Fact count changes payload rows, not
runtime decoder width.

## Claim boundary

FFNs as key-value memories and product-key/external memory layers are prior art.
Those broad ideas are not claimed.  T8's narrower demonstrated seam is:

> an arbitrary, counting-bound logical relation payload compiled directly into
> existing Transformer embedding/FFN weights, with exact BF16 retrieval,
> unchanged serving graph/resources, language coexistence, and twice-exposed
> gradient controls.

The current limitation is explicit schema.  The compiler receives parsed entity,
relation, and value IDs in token rows absent from the natural corpus.  T8 does
not prove extraction from ordinary prose, paraphrase invariance, or natural
compositional reasoning.

The result is therefore a breakthrough **candidate** for dense knowledge.  The
formal project contract still requires replication at another model scale.
After that, the next risk is not another optimizer run; it is whether a
from-zero extractor can map ordinary token sequences into the digital plane
without a teacher or hand-parsed table.

