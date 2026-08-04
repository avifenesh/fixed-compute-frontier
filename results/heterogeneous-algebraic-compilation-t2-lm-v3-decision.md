# Heterogeneous algebraic compilation — 37M LM coexistence v3 decision

Status: preregistered protocol failed; mechanism retained  
Date: 2026-07-30

V3 completed all six paired arms and emitted the sealed result
`heterogeneous-algebraic-compilation-t2-lm.json`.  The aggregate gate is false.
This exact no-normalization substrate is closed; it must not receive another
optimizer or threshold refinement.

## Substantive result

The compiled arm recovered the hidden support from 256 charged examples and
held 100% parity and 100% protected-copy accuracy through 1,000 mixed steps in
both worlds.  The ordinary gradient-trained control remained at 49.90% and
49.63% parity after 2,000 mixed steps.  Thus twice the gradient-training budget
did not recover the qualitative capability.

At the terminal 1x checkpoint, compiled-vs-control natural validation NLL was:

| world | control 1x | compiled 1x | relative delta |
|---|---:|---:|---:|
| 731 | 6.61837 | 6.63003 | +0.176% |
| 947 | 6.64266 | 6.65669 | +0.211% |

The model shape and deployed graph were identical across arms: 36,619,776
parameters, 10 causal blocks, hidden width 384, six attention heads, and
SwiGLU width 1,024.  The compiled circuits reserved 27 and 26 hidden
coordinates respectively.  They froze 2,706,918 and 2,604,764 parameter
entries, so this experiment demonstrates coexistence but not yet dense use of
capacity.

## Why the preregistered claim fails

Two frozen hygiene gates failed:

1. World 947 missed the 0.5% language-NLL tolerance at step 250: +0.616%.
   Its step-500 and step-1,000 comparisons passed.
2. The maximum pre-clip gradient norm exceeded 100 in every world.  World 947
   showed severe transients (compiled maximum 6.71e7; control maximum 4.29e12)
   despite finite terminal measurements.  This is a property of the common
   no-normalization training substrate, not evidence for a stable frontier.

The v3 all-gates result therefore remains false.  Partial gate success may not
be relabeled as a preregistered win.

## Finding retained

This is a qualitative optimization separation, not a sub-percent quality
improvement: a cheap GF(2) solver plus weight compiler installs an exact
capability into an otherwise ordinary same-size causal LM that gradient
training fails to acquire with twice the examples and steps.  The final
language cost is small in these two worlds, but the task is synthetic, the
compiler is handed the algebraic hypothesis family, and the circuit reserves
far too much trainable capacity for one rule.  Those three limitations prevent
a general breakthrough claim.

## Next admissible step

Do not tune this substrate again.  A successor must change the research
question, not polish v3:

- use a stable normalized causal architecture;
- compile a shared programmable primitive rather than one dedicated circuit
  per rule;
- measure exact capabilities per constrained degree of freedom;
- discover the applicable rule family from held-out behavior rather than being
  told `GF(2)`;
- preserve the same exported parameter count and inference graph.

The breakthrough target is many independently queryable exact rules sharing
one compiled interpreter, with ordinary 2x training failing, natural NLL
noninferior, and no numerical-instability exception.
