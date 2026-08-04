# T70 operator smoke v1 decision — state channel passes, writer-credit claim fails

Date: 2026-08-02  
Status: **PRIMARY CAUSAL-OBJECTIVE TEST FAILED; OPERATOR CHANNEL VALIDATED; NO DECISIVE OR RENTED RUN**

## Frozen development test

The tiny causal Transformer used segments ordered as

```text
[memory read] [event] [memory write]
```

Only suffix write outputs persisted. A random key/value was observed in one
segment; after hard erasure, a later segment queried the same key. The candidate
received query loss through the boundary. The `no-lifetime-credit` model
preserved the forward state but detached it at the boundary, so no gradient
reached the earlier write.

CPU configuration: one Transformer layer, width 16, two heads, two state slots,
16 keys, four random values, batch 64, 200 steps, one development seed. This is
an operator smoke, not the preregistered cross-family screen.

## Result

| condition | accuracy |
|---|---:|
| lifetime-credit candidate | 100.00% |
| detached boundary | 92.81% |
| candidate with reset state | 26.09% |
| candidate with within-batch permuted state | 26.25% |
| chance | 25.00% |

The frozen smoke required the detached model to remain at or below 33%. It did
not.
The result is therefore a **fail**, even though the causal state channel itself
worked exactly as intended.

Machine-readable result:
[`bounded-persistent-meta-learner-t70-operator-smoke-v1.json`](bounded-persistent-meta-learner-t70-operator-smoke-v1.json).

## Why the detached path worked anyway

Detaching state blocks the direct gradient path through the observation-time
write, but it does not freeze the whole writer. Observation and query passes
share Transformer parameters, and query-time gradients continue to update
those shared parameters. Fixed random label-bearing features pass through this
shared trained operator, producing decodable finite lookup codes without
direct cross-boundary writer credit.

This random-feature account is a plausible existence explanation, not a
uniquely identified mechanism. A finite set can receive distinct random codes
under suitable nondegeneracy assumptions, and a downstream decoder can learn
the finite map. That says nothing about robustness under finite precision or
noise, and it does not establish learned compression, abstraction, or
systematic transfer.

The state corruption was a random within-batch permutation, not a strictly
independent-environment intervention: fixed points are allowed. At batch size
64 with four labels its expected accuracy is 26.17%, which explains the
observed 26.25%. A future decisive manifest must use a derangement or a truly
independent environment state.

## What survives

1. Appended write slots correctly carry environment information across a hard
   causal boundary; reset and within-batch permutation remove its useful
   alignment with the query.
2. Direct cross-boundary gradient credit into the observation-time write is
   **not necessary** for the lookup capability under this shared-parameter
   implementation.
3. Any protected benefit of learned lifetime writes must come from finite-state
   compression, conditioning, systematic transfer, uncertainty, active
   acquisition, or revision—not mere state transport.

## Decision

Do not run more lookup seeds and do not change the v1 acceptance threshold.
The failed writer-credit claim is closed on this witness. T70 remains only as a
broader stateful-system/evaluation candidate: the rotated cross-family screen
must show that the complete bounded learner does more than expose a random
episodic code to a trained decoder.

No decisive local run is admitted until the concrete cross-family
implementation and manifest are frozen. No GPU rental is justified.

Independent review:
[`bounded-persistent-meta-learner-t70-operator-smoke-v1-independent-audit.md`](bounded-persistent-meta-learner-t70-operator-smoke-v1-independent-audit.md).
