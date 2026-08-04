# Consensus-permutation raw-prose compiler — Stage 0 decision

Status: **superseded — extraction valid, XOR composition gate invalid**  
Date: 2026-07-31

## Post-run audit correction

The direct fact, alias, relation, contradiction, and rejection results below
remain valid.  The XOR composition result does not.  Evaluation converted
opaque value words back to generator-private integer IDs before applying XOR.
Those integer meanings were never present in the raw prose and were not
recovered by the compiler.  The reported XOR accuracy therefore used hidden
semantic metadata and cannot admit GPU integration.

Stage 0b replaces XOR with categorical equality.  Equality is invariant under
any permutation of opaque value names, so it can be derived from the recovered
table without generator-private semantics.

## Result

The compiler received 290,304 shuffled prose strings containing 2,661,120 raw
tokens.  It was not passed entity IDs, relation IDs, value labels, template
labels, slot positions, clean facts, or the latent world object.

In 13.64 CPU seconds it induced all 36 literal surface frames, inferred three
disjoint alias views and six relation classes, and aligned 384 opaque aliases.
Despite 43,639 contradictory mentions (15.03% of the corpus), it recovered:

- 768/768 direct facts;
- 384/384 entity aliases;
- 36/36 relation paraphrases;
- 4,608/4,608 XOR calculations in the evaluator, now explicitly invalidated
  because their value-to-integer map came from sealed generator metadata.

The weakest winning categorical vote was 68.25%.  The registered union bound
on any majority error was `9.12e-4`, below the 1% gate.  The two-channel SwiGLU
bipolar product identity had zero floating-point error on all four Boolean
inputs.

## Fatal controls

The compiler rejected both cases it was required to reject:

- independently generated view tables: no admissible alignment;
- two entities with identical complete signatures: non-identifiable identity.

It therefore did not manufacture success by selecting an arbitrary
permutation when the observations lacked enough information.

## What was gained

The old compiler paradox has a concrete answer in this regime.  The compiler
does not need a stronger per-sentence semantic model.  Repeated views create a
global constraint system: majority cancels independent statement noise, table
invariants align relation paraphrases, and complete value signatures align
aliases.  The reliability comes from aggregating evidence unavailable in one
sentence, with all aggregation charged as training-only work.

## What is still missing

This is not yet a smarter model.  The surface language has atomic aliases,
literal repeated frames, a shared value vocabulary, and independent
contradictions.  More importantly, the recovered table and XOR answers have
not yet been executed by the deployed Transformer.

No integration is admitted by this artifact.  A corrected Stage 0b must first
pass.  Only then may the next test:

1. feed the same raw strings to the compiler and frontier controls;
2. compile only the recovered table, never generator metadata;
3. retrieve direct facts through held-out question forms and aliases;
4. execute XOR at inference inside ordinary attention and SwiGLU layers;
5. give the strongest gradient control the compiler-derived direct and XOR
   targets, so preprocessing information is not the candidate's advantage;
6. preserve identical parameter count, graph, precision, KV state, and serving
   FLOPs across arms.

The result advances one missing causal link.  The original production goal
remains open.
