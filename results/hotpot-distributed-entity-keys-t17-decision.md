# Hotpot distributed entity keys T17 — decision

Status: **FAIL; CLOSE post-hoc analog contextual addressing**  
Date: 2026-07-31

## Decision

Close the entire post-hoc analog entity-address route represented by T15-T17.
Do not continue it with more contexts, layers, steps, metrics, projections,
classifiers, or relaxed exactness.

The digital-plane capacity result from T11 is retained.  What is now rejected
is the assumption that an ordinary LM's unmodified contextual geometry can be
made into an exact multi-token entity address by compiler-side analog keys.

## Evidence

- All eight contracts passed; physical allocation was 805, 836, and 764 rows
  in blocks 7, 8, and 9, each below the 1,024-row limit.
- The exact key ledger was 923,520 scalars, 2.5249% of model parameters.
- Key training completed 800 finite updates with no retry.
- Canonical retrieval was only 1,918/2,405 (79.7505%).
- Training raw-context retrieval was 9,090/9,620 (94.4906%).
- Held raw-context retrieval was 1,827/2,405 (75.9667%).
- Natural-question retrieval was 225/292 (77.0548%) on train surfaces and
  166/208 (79.8077%) on development surfaces.
- The served checkpoint and state were unchanged; zero answers were read.
- Key artifact SHA-256:
  `a19f576027a81e5aa8788ee50ecf946374a9f0d5fd5eeef10ef6a98473ea21a5`.
- Result artifact SHA-256:
  `123eaf9d08fbdf601eb79f08ef0fcde5460182e95598156ff557938cc843d9bb`.

## What the sequence established

| interface | development result |
|---|---:|
| deterministic external compositional title code (T12) | 100.00% |
| raw contextual last-title vector (T15) | 81.25% |
| raw-prose-trained shared folded projection (T16) | 96.15% |
| physically distributed class-specific gate rows (T17) | 79.81% |

T16 proves that raw-prose-only invariance training can recover much of entity
identity with no served projection.  T17 proves that distributing independent
analog rows across physical layer gauges does not turn that signal into a
reliable in-model address.  The result is not a near-threshold failure.

## Wall boundary

This is a real wall for the current implementation family:

`ordinary contextual state -> post-hoc analog similarity/key -> exact record`.

It is not a theorem that a Transformer cannot carry entity identity.  It says
identity must be part of the architecture's trained/discrete protocol, not
inferred after ordinary LM training from a geometry the LM objective never
promised.

## Retained path

Do not require exact top-1 identity before testing usefulness.  A standard FFN
already computes a soft mixture of channel values.  T16's 96.15% signal can be
retained only as initialization for a budget-neutral soft record operator,
where near-neighbor mixtures are allowed and success is judged directly by
held-out QA plus language quality.

The next architecture must replace served computation rather than add it.  It
must also expose its parameter, MAC, nonlinear-operation, memory-traffic, and
kernel ledger before training.
