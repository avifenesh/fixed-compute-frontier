# Equivariant coded operator T25 — paper audit

Status: **NOT ADMITTED; DO NOT IMPLEMENT OR TRAIN**  
Date: 2026-07-31

## Proposed construction

The sketch assigns every answer token `y` a fixed bipolar code

\[
c(y)\in\{-1,+1\}^{16},
\]

maps a question or document relation to a key

\[
k\in\mathbb R^{13},
\]

and writes a per-document operator

\[
C_d\in\mathbb R^{16\times 13}.
\]

The read is

\[
r=C_d k(q),
\]

followed by nearest-code decoding.  Entity-renaming interventions would train
the relation key to be invariant to entity identity while the writer and query
encoder are learned from raw prose.

The 208 operator coefficients fit the proposed 220-cell logical plane.  A
SwiGLU realization could represent signed products with two channels per
coefficient, requiring 416 of 1,024 intermediate channels.  Those are useful
layout facts, not a capability proof.

## What can be proved

### Fixed-code identity

There are `2^16 = 65,536` distinct 16-bit codes, enough to assign unique codes
to a 49,152-token vocabulary.  With bipolar inner-product decoding, the
correct code scores 16 and a code at Hamming distance one scores 14, giving a
minimum score margin of two for any unique codebook.  This proves exact
noiseless identification, not robust semantic decoding.

### Interpolation theorem up to key dimension

Let

\[
K=[k_1,\ldots,k_m]\in\mathbb R^{13\times m},\qquad
Y=[c(y_1),\ldots,c(y_m)]\in\mathbb R^{16\times m}.
\]

If `m <= 13` and `K` has full column rank, then

\[
C_d=Y K^+
\]

satisfies `C_d K = Y`.  Thus the operator can exactly store at most thirteen
linearly independent arbitrary key-to-code associations by construction.

### Rank obstruction beyond thirteen associations

For `m > 13`, `K` has a nonzero null vector `a` with `Ka = 0`.  Any linear
operator must obey

\[
C_d K a = 0.
\]

Exact interpolation therefore requires `Ya = 0` for every `a` in the null
space of `K`, equivalently

\[
\ker K\subseteq\ker Y.
\]

Arbitrary target assignments do not satisfy this constraint.  Increasing
training time, changing the optimizer, or improving the implementation cannot
remove this limit without changing the key dimension, operator family, or
structural assumption on the data.

### Equivariance is not sufficiency

Entity-renaming augmentation can enforce a consistency condition such as

\[
k(\pi x,\pi q)=k(x,q)
\]

for an entity permutation `pi`.  This removes one nuisance degree of freedom.
It does not prove that occupation, chronology, membership, comparison, and
other relations become distinguishable, nor that natural paraphrases map to
the same key.  Many useless constant or lossy maps satisfy the same invariance.

## Failed paper gates

### P1: no baseline failure witness unique to this operator

A fixed-size Transformer can implement the same 16-by-13 linear map inside its
ordinary layers.  The proposal does not enlarge the served function class.
Any gain would have to come from learnability or allocation, but no minimal
family yet shows why ordinary training cannot learn the same map at equal
resources.

### P2/P3: the central semantic block has no constructive contract

The storage and read maps have clear meanings.  The learned map from raw prose
to relation keys does not.  “Make it invariant under entity renaming” is a
constraint, not a construction of a relation-bearing sufficient statistic.
There is no proof or oracle evidence that its output preserves the facts needed
by held-out natural questions.

### P4: the known obstruction is already close to the proposed workload

The operator has only thirteen independent key dimensions.  To proceed, the
candidate would need a predeclared structural theorem that all required
per-document answer mappings factor through at most thirteen relation degrees
of freedom.  Natural documents are not currently shown to satisfy that
condition.

### P5: matched-class inclusion removes the expressivity claim

Because the served graph and parameter budget can embed this operator, the
only admissible claim is that the write/read organization makes useful weights
substantially easier to learn from the same raw data.  No sample-complexity,
optimization, or identifiability argument currently establishes such an edge.

### P7: no breakthrough-sized prediction follows from the algebra

The exact code margin and 208-cell fit establish feasibility.  They do not
predict a qualitative capability gain, a scaling-law change, or a large
compute-equivalent improvement.  The main unknown is still the entire semantic
learning problem rather than one composition interaction.

## Decision

Do not write a Stage-0 implementation and do not use the H100 for T25.

Retain only these proved components:

1. the 16-bit fixed-code noiseless identity and its small margin;
2. the exact pseudoinverse construction for at most thirteen independent keys;
3. the null-space obstruction beyond thirteen arbitrary associations;
4. entity-renaming equivariance as one necessary control, never as evidence of
   sufficiency;
5. the possible 208-cell / 416-product-channel physical layout.

T25 may be reconsidered only if a new paper supplies both:

- a constructive, explainable language-to-relation statistic; and
- a theorem or stronger decoded-information oracle showing that the target
  capability factors through its bounded operator family.

Changing code width, key width, augmentations, optimizer, or decoder alone is
not re-admission.  Those changes leave the missing semantic sufficiency claim
untouched.
