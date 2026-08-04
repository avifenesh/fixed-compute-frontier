# Packed rule memory T4 — preregistration

Frozen before implementation or measurement: 2026-07-30

## Question

Can ordinary BF16 embedding scalars act as active multi-bit rule memory inside
a fixed causal LM, so one model stores more exact capabilities with fewer
rule-data scalars and no added inference operation?

T4 is not a retry of T3's failed control-gradient threshold.  T3 remains
closed.  T4 changes the representation and density question.

## Construction

- The model shape remains T3's tied-embedding 36M-class causal graph:
  vocabulary 49,152, hidden width 384, ten pre-RMSNorm blocks, six 64-wide
  heads, SwiGLU width 1,024, causal masking, and four RoPE language heads.
- All arms have identical parameter shapes, parameter count, graph, and random
  initialization before compilation.
- Natural train/validation streams and their hashes remain sealed.
- Algorithm symbols reuse vocabulary IDs absent from both natural streams.  No
  vocabulary row or parameter is added.
- There are 768 independently sampled 16-of-32 parity rules.  A 64-example
  charged prefix is sealed for every rule.

Each 32-bit support is packed into eight base-16 digits.  The digits are stored
as eight ordinary BF16-safe integer values in the rule-token embedding row.
Thus the complete rule table occupies `768 * 8 = 6,144` scalar entries—fewer
than T3's 8,192 entries for only 256 rules.

One shared block decodes all eight nibbles into 32 support signs.  The decoder
uses a piecewise-linear representation shared across the four bits of each
nibble: one constant pair, one linear pair per digit, and fourteen shared
hinges per digit, at most 130 active SwiGLU channels.  Later shared blocks copy
the decoded rule, select its 16 positions, reduce their signs to an integer,
and decode parity.  BF16 margins are one full integer between adjacent
breakpoints.

Compiler isolation is temporal rather than permanent: it protects the program
state only through the four blocks that decode and execute it.  After the
result sign is written, later blocks protect only that single result
coordinate; all other coordinates return to the language path.  The exported
checkpoint contains only the original dense weights—no unpacking kernel,
sidecar, adapter, expert, retrieval store, or runtime solver.

## Arms and budgets

Two paired model seeds run:

- `baseline_1x`: gradient training on all 49,152 prefix examples, then 1,000
  mixed steps;
- `compiler_1x`: exact GF(2) recovery, nibble packing, in-place compilation,
  then the same 1,000 mixed steps;
- `baseline_2x`: the same prefix gradient training, then 2,000 mixed steps.

Mixed training uses one algorithm batch every 20 steps and FineWeb-Edu on the
other 19.  Optimizer, BF16 forward arithmetic, FP32 losses, global clipping at
1, paired batches, and the 50-step-warmup/cosine schedule are unchanged from
T3.

## Frozen gates

Every gate must pass in both paired seeds:

1. Identical pre-compilation initialization hashes; exact recovery of all 768
   supports; exactly 6,144 rule-data entries.
2. The sealed BF16 quick test and candidate checkpoints 250, 500, and 1,000
   have at least 99% parity accuracy for every rule and at least 99% protected
   copy accuracy.
3. The 2x ordinary control has mean parity below 80% and protected copy at
   least 95% at step 2,000.
4. Candidate natural validation NLL is no more than 0.5% above its paired 1x
   control at steps 250, 500, and 1,000, and improves from 500 to 1,000.
5. All losses, gradients, parameters, and evaluations remain finite; no step is
   skipped; maximum loss is below 100.  Pre-clip norms are recorded but T3's
   arbitrary norm-100 diagnostic is not reused as a T4 acceptance threshold.
6. Vocabulary, parameter count, graph, and forward FLOPs are identical across
   arms.  The shared decoder width is independent of the number of rules.

## Interpretation boundary

Passing establishes a 12x rule-density increase over T3's one-bit-per-scalar
table (`3x` as many rules with `0.75x` as many description scalars), at the same
model size and inference graph.  It remains a synthetic GF(2) capability test;
it does not establish automatic family discovery or broad language reasoning.
Failure on BF16 decoding, language coexistence, or finite training closes this
packed-nibble construction without post-measurement threshold tuning.
