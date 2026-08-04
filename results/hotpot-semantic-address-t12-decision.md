# Hotpot semantic address T12 — decision

Status: **CLOSED NEGATIVE for the static-input-embedding semantic interface**  
Date: 2026-07-31

## Decision

Do not compile real-prose property channels from T12's static token-embedding
geometry.  The checkpoint and the compositional entity address are retained;
the failed semantic interface is not.

## Frozen evidence

- The 36,577,152-parameter shared base completed 20,000 steps and 40,960,000
  token presentations with zero failed or retried updates.
- Protected natural NLL fell from 10.865491 to 5.436326, passing the 5.5 gate.
- All 2,405 canonical entity-title addresses were distinct.
- All 208 sealed natural title surfaces retrieved their canonical title, with
  a worst-case cosine margin of 0.090759.
- The trained static-input-embedding semantic diagnostic reached 60.5769% on
  104 evaluation questions, below the frozen 70% gate.
- It exceeded the privileged lexical reference by only 4.8077 percentage
  points, below the frozen 10-point gate.
- The result artifact reports `all_gates_pass: false` and SHA-256
  `e7c260ec7cbf1aef33e1f8e15213a6ee53f58741ab6971b1446e8e6186b0853a`.
- The sealed pre-diagnostic checkpoint is retained at SHA-256
  `a2900585a6e9afe4a9fba4f55afca30df5bd79c335d646cda5b9e940b72d693b`.

## Interpretation boundary

T12 rejects the assumption that the input embedding table is already a
reliable semantic key space at this scale and training budget.  It does not
reject the digital plane: natural multi-token addressing passed, and T12 never
tested the contextual states that the ordinary model computes at an FFN gate.

No post-result stopword, pooling, threshold, code-width, or step-count change
belongs to T12.  More training of this same static interface would reopen a
closed lane without identifying the failed assumption.

## Direct refinement

T13 may change exactly one architectural interface: replace static input rows
with the contextual state already presented to the final block's FFN gate.
The model, checkpoint, data, scoring rule, train-fitted scalar threshold, and
accuracy gates remain fixed.  No weights are trained and no layer or pooling
sweep is allowed.

Because T12's aggregate evaluation result is known, T13 is a development
diagnostic, not fresh held-out evidence.  Passing T13 can admit a compiler, but
any capability claim must use a newly frozen holdout.
