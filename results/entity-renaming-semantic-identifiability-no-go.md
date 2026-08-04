# Entity renaming does not identify semantic relations

Status: **GENERAL PRE-RUN NO-GO; CLOSE RENAMING-ONLY SUCCESSORS**  
Date: 2026-07-31

## Question

Can a from-zero compiler obtain a canonical relation key merely by training an
encoder to be invariant/equivariant under consistent entity renaming, possibly
with two augmented views, a discrete bottleneck, and raw reconstruction?

No.  Those ingredients can remove entity identity, but they do not create the
cross-surface equivalence relation needed to identify paraphrases or logical
relations.

This result does not say that entity renaming is useless.  It remains a strong
shortcut control.  It says that renaming cannot be the source of semantic
identifiability.

## Setup

Let `X` be the set of raw entity-bearing strings.  Let a finite group `G` act
on `X` by consistently permuting only entity identities.  A relation encoder

\[
f:X\to Z
\]

is renaming-invariant when

\[
f(gx)=f(x)\qquad\text{for every }g\in G.
\]

Let `[x] = {gx : g in G}` be the orbit of `x`.  Let `rho(x)` denote the desired
semantic relation, where two different surface forms may have the same
relation.

## Proposition 1: invariance identifies at most the orbit quotient

Every invariant `f` factors through the quotient map

\[
q:X\to X/G.
\]

That is, there is a function `h` with

\[
f=h\circ q.
\]

### Proof

Define `h([x]) = f(x)`.  This is well-defined because any two members of the
same orbit differ by a group element and invariance makes their encodings
equal.  Conversely, every function on `X/G` produces an invariant encoder when
composed with `q`.

The quotient retains every non-entity token, its order, punctuation, syntax,
and document-specific wording.  Therefore the canonicalized-string map

\[
f_{surface}(x)=\text{replace entities by canonical placeholders}

\]

is invariant and can be injective over `X/G`.  It distinguishes paraphrases
that should share `rho`.  Renaming invariance alone imposes no equation between
different orbits `[x]` and `[x']`, even when `rho(x)=rho(x')`.

## Proposition 2: same-instance views recover their support component, not a
relation by default

Let two stochastic augmentations produce `(V_1,V_2)` from one raw instance
`x`.  Form the bipartite support graph whose left vertices are possible values
of `V_1`, whose right vertices are possible values of `V_2`, and whose edges
are pairs with positive joint probability.

Any deterministic zero-error common code

\[
z_1(V_1)=z_2(V_2)
\]

must be constant on each connected component of this graph.  The maximal
deterministic common variable is the component identity.

If augmentations of different canonicalized sentences have disjoint support,
then each component identifies a sentence orbit, not a semantic relation.  To
make the components equal relation classes, the view process must satisfy both:

1. it connects independently worded instances of the same relation; and
2. it does not connect instances of different relations.

Entity renaming and independent token masking do not provide the first
condition.  Pairing two masks from the same sentence tells the learner which
sentence they came from, but not which other sentences are paraphrases.

This is the concrete boundary behind common-information and multi-view
identifiability results: those theorems recover a shared latent only when the
view-generating process actually makes that latent the shared source.  They do
not prove that arbitrary augmentations of prose make “relation” the shared
source.  Relevant general results include
[multi-view nonlinear ICA](https://proceedings.mlr.press/v115/gresele20a.html)
and [multi-view causal representation identifiability](https://arxiv.org/abs/2311.04056).

## Proposition 3: a finite bottleneck does not select the semantic quotient

Suppose `Z` has `K` values.  Renaming invariance permits every partition of
`X/G` into at most `K` cells.  Balance, entropy, or code-usage constraints may
choose a partition of a desired size, but they do not say which orbits belong
together.

A reconstruction loss creates the opposite pressure: it rewards retaining
surface distinctions.  A masked-token loss rewards distinctions useful for
frequent token prediction, which T20 already showed need not be relation truth.
Thus invariance plus bottleneck plus reconstruction does not construct the
desired semantic quotient.

## Corollary for T25

The condition

\[
k(\pi x,\pi q)=k(x,q)
\]

can prevent the relation key from using entity names.  It cannot show that two
wordings of occupation, chronology, membership, nationality, or comparison
map to the same key.  The constant map, a topic map, and an injective
canonical-surface map can all satisfy the same renaming condition.

Therefore entity-renaming equivariance does not repair T25's missing
language-to-relation construction.  A renamed-view contrastive loss, more
augmentations, code balancing, or a reconstruction head is not re-admission.

## What would provide a real semantic bridge

A raw-only successor must name an observable relation that connects distinct
surface orbits.  Examples of admissible evidence sources are:

1. **Cross-instance incidence:** different patterns repeatedly occur with the
   same entity pairs or the same typed outcomes, with a factorization theorem
   showing recovery up to a harmless permutation.
2. **Independent natural views:** genuinely different statements of the same
   fact are linked by raw identity/equality evidence that does not already use
   a semantic parser.
3. **Identifiable generative structure:** a grammar or latent-variable family
   for which raw moments or minimum-description-length selection recover the
   relation program under explicit conditions.
4. **Sequence-derived supervision:** a transformation whose target is
   computable from the same raw sequence and whose equivalence classes are
   proved to match the relation needed downstream.

Each source has a failure boundary.  Repeated entity pairs may express several
relations; natural views may be absent; latent factorizations may be rank
deficient or non-identifiable; and masked-token targets may capture surface
prediction rather than relation truth.

## Required pre-run diagnostic

Before training a semantic encoder, report the proposed bridge graph or moment
object and measure:

- coverage: how many candidate facts participate in a cross-instance bridge;
- multiplicity: independent entities and surface forms per putative relation;
- ambiguity: how often one bridge key carries several relations;
- identifiability: rank, separation, anchor, or support-connectivity margin;
- query reachability: whether held-out query surfaces enter the same recovered
  component/basis without labels;
- decoded-information ceiling: whether perfect recovered records would produce
  a breakthrough-sized natural capability gain.

If the bridge is absent, no encoder implementation or GPU training follows.

## Decision

Close the broad family

```
entity renaming + same-instance augmentation + bottleneck/reconstruction
    -> semantic relation key
```

as an unsupported identifiability claim.

Retain entity renaming only as an adversarial control inside a candidate whose
cross-surface semantic bridge is independently defined and tested.
