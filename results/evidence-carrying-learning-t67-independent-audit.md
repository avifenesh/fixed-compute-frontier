# T67 independent audit — evidence governance is useful, but the intelligence claim is occupied

Date: 2026-08-01  
Status: **INDEPENDENT NO-RUN VERDICT; TRANSACTION CONTRACT RETAINED**

## Verdict

T67 should not run as a new intelligence mechanism. Current work already covers
its principal conjunction:

- TrustMem verifies persistent memory transitions and trains the updater;
- Self-Trained Verification trains a generator inside a verifier loop and
  retains gains without the verifier;
- self-proving models attach formally checkable evidence to answers;
- verifier-driven test-time training makes verified outputs durable through
  parameter updates;
- Supersede trains the memory-update policy; and
- current memory research explicitly recommends preserving raw episodes and
  gating consolidation.

MemoPilot further closes the central `train U, not only f` wording by optimizing
a separate memory-update model end to end. Kumiho closes most of the proposed
versioned graph, dependency, immutable evidence, and belief-revision machinery.

The narrow retained systems hypothesis is:

> Durable cognitive-state changes should be atomic, versioned transactions.
> Evidence remains attached; tentative state cannot influence normal behavior;
> provenance and transition semantics are checked independently; and later
> counterevidence revokes rather than erases history.

This is not a distinct learning principle unless evidence-rich training causes
standalone transfer beyond verdict-only feedback, runtime gating, ordinary
training at equal compute, and current verifier-trained memory controls.

## Mathematical audit

The propositions in the main T67 note are scoped correctly:

- T67.1 explicitly accounts for false rejection of correct items and missed
  wrong items. Its posterior threshold `p_z > c/(q+c)` follows directly from
  the declared fix and corruption probabilities.
- T67.2 correctly distinguishes false-accept rate from posterior contamination
  among accepted updates. This distinction must be preserved in every result.
- T67.3 is a valid Jensen/value-of-information control. It does not establish a
  source of informative evidence.
- T67.4 is valid only for candidates fixed before fresh independent tests and a
  frozen probe distribution, as the note states.

The second audit of the exact file required and confirmed four corrections:

- `q+c>0` and signal-dependent `q_z,c_z` were made explicit in T67.1;
- T67.2 now defines gain and damage conditional on gate acceptance;
- T67.3 now charges the evidence cost and is labeled a free-information result;
  and
- T67.4 now gives the exact logarithmic denominator and conditions adaptive
  candidates on the complete prior history.

Additional mandatory boundaries are:

1. Over adaptive commits, a per-step error bound must hold conditional on the
   complete prior history before it can be union-bounded.
2. Searching many candidates creates selection bias; offline verifier false
   acceptance does not bound the selected candidate's error.
3. Multiple learned verifiers cannot have their error rates multiplied without
   conditional independence.
4. Calibration must be measured after the updater has optimized against the
   verifier.
5. A per-update bound does not bound downstream damage from a central false
   belief.
6. Verifier-free capability is an empirical distribution-shift question, not a
   consequence of verifier-in-loop training.

## What a certificate means in the open world

A certificate can separately establish:

1. provenance — a record genuinely came from a declared source;
2. entailment — the update faithfully represents that record;
3. authority and currency — the source governs the current value under declared
   rules; and sometimes
4. world truth — the claim is actually correct.

The first two are often mechanically checkable. The fourth usually is not.
Calling provenance a truth certificate would recreate the same circularity T66
identified.

## Decisive future falsifier

If this line is ever reopened, compare runtime certification, verdict-only
feedback, evidence-rich diagnostic feedback, TrustMem/STV-style controls, raw
episodes, and equal-compute ordinary training. Remove the verifier at final
evaluation. The evidence representation earns a learning claim only if it
improves held-out composed updates and downstream current-world reasoning over
verdict-only feedback while preserving old capability.

A lower bad-write rate alone is reliability engineering. It does not establish
a smarter model.
