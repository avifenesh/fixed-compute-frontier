# Heterogeneous family discovery T7 — decision

Status: all 81 preregistered gates passed; mechanism survives adjacent scale,
but broad-LLM breakthrough claim remains withheld  
Date: 2026-07-31

## What passed

The sealed 110,776,960-parameter run completed four arms in three rotated model
seeds.  Source, pilot, tests, preregistration, predecessor, and both data hashes
match.  There were no skipped/retried steps or discarded samples.

The candidate recovered all 256 new supports and algebraic families from the
30,720-example prefix and abstained on all 64 random decoys.  At steps 250,
500, and 1,000, every family and the worst individual task were 100% in every
seed.  Random-decoy accuracy was 49.91%, 50.52%, and 50.63%; protected copy was
100%.

At 2x mixed training, the stronger controls averaged:

| arm | parity | majority | exact-half | modulo-3 | natural NLL |
|---|---:|---:|---:|---:|---:|
| compiler + Muon, 1x | 100.00% | 100.00% | 100.00% | 100.00% | 6.0833 |
| calibrated Muon, 2x | 49.56% | 62.28% | 64.66% | 60.68% | 5.9878 |
| AdamW, 2x | 49.17% | 66.46% | 66.88% | 62.55% | 6.0919 |

Thus the capability separation survives a 3.03x model-size increase, three new
seeds, an untouched task world, a calibrated matrix optimizer, AdamW, and an
execution-order rotation.  It is not an artifact of a weak AdamW baseline.

## Resource result

Mean per-arm charged resources were:

| arm | wall seconds | issued-work estimate | energy estimate | run cost |
|---|---:|---:|---:|---:|
| compiler + Muon, 1x | 88.6 | 3.070 PFLOPs | 17.35 kJ | $0.098 |
| Muon, 1x | 109.1 | 3.963 PFLOPs | 22.69 kJ | $0.121 |
| Muon, 2x | 178.1 | 6.759 PFLOPs | 37.52 kJ | $0.198 |
| AdamW, 2x | 120.7 | 3.516 PFLOPs | 25.50 kJ | $0.134 |

Against Muon 2x, the candidate used 2.20x less estimated work and 2.01x less
wall time while changing the procedural result from partial/chance to exact.
Against AdamW 2x it used only 1.15x less estimated work, but still changed all
four families to exact.

The compiler itself took 2.04 CPU seconds, with an estimated 6.326 billion
integer support-search multiply-adds and 1.581 billion Boolean family
comparisons.  World construction took another 2.13 seconds and was charged to
every arm.

Peak allocated HBM was 15.82 GB for the candidate, driven by exhaustive
evaluation.  Every arm exports the same 110,776,960 parameters and inference
graph.  The candidate fixes 13,837,996 entries (mostly isolation zeros) and
only 6,189 nonzero entries; reducing that 12.49% capacity reservation is still
important.

## Protected language result

Candidate natural NLL was better than paired Muon 1x by 2.37%, 2.52%, and
2.14%.  This likely reflects avoiding 480 interfering prefix-gradient updates;
it is supporting noninterference evidence, not the breakthrough claim.

Muon 2x reached a 1.6% better mean natural NLL than the 1x candidate.  Therefore
T7 does **not** show a broad language-modeling compute advantage.  It shows a
qualitative acquisition advantage on the discovered discrete subproblem while
natural modeling remains protected.

## Claim boundary

The result now supports this narrow finding:

> A training-time discrete solver can install hundreds of automatically
> selected exact programs into an otherwise unchanged causal LM, at two model
> scales and three adjacent-scale seeds, when direct supervised gradient
> training fails with more than twice the charged work.

It does not yet support “a generally smarter LLM.”  The candidate searches four
human-specified count families, uses synthetic task tokens, and reserves a
hand-constructed interpreter.  Neural compilation, Transformer program
construction, and FFNs as key-value memories are existing broad ideas.

The correct next test is not another NLL polish run.  It is a dense-knowledge
extension: an arbitrary high-cardinality relation table compiled as a digital
memory plane inside the existing embedding/FFN graph, with capacity measured in
facts per deployed scalar and exact held-out retrieval.  That attacks the
user's light-model/dense-knowledge objective directly and can falsify whether
the mechanism scales beyond a small algebra library.

