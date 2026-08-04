# T30 natural-information bridge audit

Status: **FAIL ON PAPER; CLOSE AS ACTIVE BREAKTHROUGH DIRECTION**  
Date: 2026-07-31

## Verdict

T30 is a correct compact operator, but there is no defensible path from its
proved noncommutative routing advantage to a breakthrough-size raw-prose or
production-language-model gain under the current design.

Do not train it merely to reproduce permutation state tracking.  Retain its
exact compiler, codec, and lazy-frame constructions as reusable primitives.

## What T30 actually adds

Relative to a one-layer diagonal-affine recurrence, T30 can move an arbitrary
existing coordinate to another coordinate using a group action.  The
corrected preflight proves that separation exactly.

It does not add:

- a semantic address for a fact;
- more state or checkpoint entropy;
- selective independent movement of many facts;
- a relational join;
- a proof that a whole matched Transformer/SSM hybrid cannot construct the
  same downstream statistic;
- a measured serving-cost advantage.

## The algebra of an ordinary factual write

Let memory be a finite key-indexed array `M`.  The natural update

```text
write(k, v): replace M[k] by v and leave every other key unchanged
```

is a partial overwrite.  For a one-hot selector `e_k`, it is already an affine
diagonal update:

\[
M'=(1-e_k)\odot M+e_k\odot v.
\]

Writes to distinct keys commute; two writes to the same key obey last-write
wins.  This is an overwrite monoid, not a global permutation group.

Therefore, once the semantic key `k` and value `v` are available, T29's
diagonal scale and offset can perform the correct write.  T30's extra global
rotation/reflection is not needed for this operation.

## Where the unsolved work lives

Raw prose does not supply `e_k` directly.  A writer must infer that differently
worded mentions refer to the same entity/relation, separate distinct mentions,
resolve scope and negation, and decide whether a sentence inserts, overwrites,
or qualifies a record.

That map is

\[
\text{raw token context}\longrightarrow(k,v,\text{update type}).
\]

T30 changes only the subsequent state action.  It supplies no identifiability
result, supervision signal, or constructive algorithm for this semantic
address.  Earlier failed semantic-code lanes in this project encountered the
same missing bridge.

Relational reasoning has a second mismatch.  A query such as “who is the
parent of the author of X?” requires key matching and composition of partial
relations.  Algebraically this is closer to map lookup and relational join
than to applying one global element of `D_109` to every stored coordinate.

## Why the whole-model separation disappears

The exact lower bound applies to one diagonal-affine map acting on arbitrary
incoming state.  In a hybrid language model, the offset presented to that head
can already be context-dependent because earlier attention or nonlinear
layers have read the prefix.  Such a hybrid may calculate a routed result in
its contextual input and write it through the offset.  This can cost depth or
training efficiency, but it removes a whole-model function-class theorem.

The remaining claim would therefore be an inductive-bias or resource trade,
which requires natural and physical evidence.  No current argument predicts
the project's required large margin.

## Information and capacity boundary

The T30 recurrent state is 109 BF16 values plus an eight-bit orientation in a
220-byte envelope.  T29 has 110 BF16 values in the same envelope.  The group
frame changes how existing information is transformed; it does not increase
the number of representable bit patterns.

Likewise, an 880-bit document record cannot losslessly encode more than 880
independent bits.  T30 may be a better inductive bias for structured actions,
but it has no information-theoretic route to denser arbitrary factual
knowledge.

## Prior-art boundary

[PD-SSM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/77b830c18836a9b2e1395a4936dd687a-Abstract-Conference.html)
already gives permutation-diagonal transitions with linear-cost scans,
provably optimal finite-state tracking, and a hybrid experiment where
transitions are expressed by variable-length English sentences.

[Learning State-Tracking from Code](https://arxiv.org/abs/2602.14814) further
shows that permutation composition is now a standard architecture diagnostic
and studies how state visibility affects learnability under next-token-like
training.

[Why Are Linear RNNs More Parallelizable?](https://arxiv.org/abs/2603.03612)
places permutation-diagonal LRNNs in a known expressivity class and identifies
diagonal-plus-low-rank recurrences as strictly richer in its complexity model.

Consequently, a successful T30 state-tracking run would validate an
implementation detail, not discover the requested frontier.

## Closure decision

The first synthetic learnability screen was withdrawn before execution.  The
corrected arbitrary-state theorem and exhaustive preflight passed.  This is
enough to distinguish “the algebra is wrong” from “the algebra is useful for
the target.”

The wall is **problem selection**, not an implementation bug:

```text
proved capability: move existing features by a compact group action
needed capability: infer semantic addresses and perform keyed updates/joins
```

T30 is therefore closed as the active breakthrough direction without a GPU
training run.  Reopening requires a new, explainable natural statistic for
which global dihedral routing is necessary against a full matched hybrid, plus
a predicted breakthrough-size oracle margin.

## Key shift for the next search

Derive the semantic operation before choosing the cheap matrix family.  A new
candidate must begin with all four typed maps:

```text
address: raw context -> stable key
update:  (key, value, old memory) -> new memory
query:   (question, memory) -> selected records
reason:  selected records -> answer-bearing state
```

For each map, name the conserved resource, prove a matched-baseline witness,
and isolate one unknown learnability link.  A new state mixer with no account
of `address` is not admitted.
