# Task-agnostic useful-structure extraction — no-go and escape boundary

Status: **GENERAL PRE-RUN THEOREM**  
Date: 2026-07-31

## Question

Can a lossy compiler read raw prose, receive no task or query information, and
be guaranteed to retain whatever structure will be useful for arbitrary future
knowledge and reasoning questions?

No.  “Useful” is not identifiable from the corpus alone without a declared
relationship between the raw-data distribution and the future task family.

This does not rule out autonomous compilation.  It separates its three valid
forms:

1. lossless structural compilation;
2. lossy compilation under an explicit generative/task assumption;
3. learned compilation whose task transfer remains the one empirical
   hypothesis.

## Theorem 1: a universal lossy compiler cannot preserve all future tasks

Let `X` be a finite set of possible raw corpora, let

\[
C:X\to S
\]

be a compiler, and suppose `C` is non-injective.  Then there is a binary
downstream task that no decoder using only `C(x)` can answer correctly for every
`x in X`.

### Proof

Because `C` is non-injective, there are distinct corpora `x` and `x'` with
`C(x)=C(x')`.  Define a task `f` with `f(x)=0` and `f(x')=1`; assign arbitrary
values elsewhere.  Every decoder `D(C(x))` returns the same result on `x` and
`x'`, so it is wrong on at least one.  QED.

The task can be written as a question asking for the exact corpus feature on
which `x` and `x'` differ.  Therefore this is directly a knowledge-recall
boundary, not merely an abstract classification result.

## Corollary 1: exact unpredictable recall requires an injective plane

If future questions may request any token or fact distinction in the corpus,
the persistent representation must be injective on the admitted corpus family.
It may be compressed, indexed, grammatical, or graph-structured, but it must
retain enough information to reconstruct every relevant distinction.

The T32 exact record plane is valid for this reason.  A learned 32-dimensional
summary is not a universal substitute for it.

## Theorem 2: no label-free criterion uniquely determines semantic usefulness

Fix one observed raw corpus `x`.  Let `T_1(x)` and `T_2(x)` be two different
statistics.  There exist downstream task distributions for which `T_1` is
sufficient and `T_2` is useless, and other distributions for which the roles
reverse.

### Construction

Choose the first task family to ask only functions of `T_1` that vary inside
level sets of `T_2`; choose the second symmetrically.  Since the compiler sees
the same raw corpus under both future task distributions, no corpus-only rule
can certify which statistic is “the” useful one.

Consequently an MDL score, reconstruction loss, next-token loss, invariance,
or multi-view agreement becomes meaningful only after stating why its induced
equivalence classes align with the future queries.

## Valid escape A: lossless useful structure

A compiler may transform raw text into an invertible object that also supports
useful operations:

```text
raw corpus <-> compressed self-index / grammar / succinct graph
                              |
                              +-> exact search, rank, select, locate, expand
```

Invertibility removes the task-identifiability problem.  Compression and
indexing claims can be proved independently.  Semantic usefulness of the
operations remains empirical, but no future fact was discarded.

This is the proper comparison class for a typed digital plane.  A direct
`uint16` table is only the simplest lossless control.  A compressed full-text
self-index is a stronger control because it can simultaneously reduce bytes
and support exact substring search.  Current systems already demonstrate this
class: [Infini-gram mini](https://arxiv.org/abs/2506.12229) reports an FM-index
at 44% of corpus size, while
[Autoregressive Search Engines](https://openreview.net/forum?id=Z4kZxAjg8Y)
uses an FM-index to constrain generated corpus substrings.  Therefore “use an
FM-index” is not a novelty claim here; it is a mandatory control.

## Valid escape B: identified task family

Lossy extraction is defensible when a declared generative family makes a
statistic sufficient.  For example, if

\[
X=G(F,N),\qquad Y\perp X\mid F,
\]

and a raw-computable algorithm constructively recovers `F`, then storing `F`
is sufficient for `Y`.  The paper must prove both recovery and conditional
sufficiency under the stated assumptions, then measure whether natural prose
satisfies them.

This is where earlier relation-bridge candidates failed: they named a useful
latent relation but did not expose the raw observable that identifies it on the
natural corpus.

## Valid escape C: one empirical transfer link

A compiler may optimize a universal proxy such as next-token prediction,
description length, or masked reconstruction and leave this claim empirical:

> the proxy-selected statistic transfers to the frozen future question
> distribution.

That is admissible only if every other block is proved or independently
microbenchmarked, the proxy uses raw data only, and the transfer link receives
zero/shuffle/random and strongest-control comparisons.  A large teacher or QA
labels cannot be hidden inside the compiler.

## Consequence for the active goal

The project cannot require all three simultaneously without assumptions:

1. lossy semantic extraction;
2. no task/query signal of any kind;
3. guaranteed sufficiency for arbitrary future reasoning.

The non-vacuous target is:

- use a lossless typed plane as the task-agnostic base;
- let raw-only training construct routing/reading structure;
- state the exact query domain, beginning with autonomous exact-title routing;
- compare against direct tokens and a compressed self-index;
- leave only proxy-to-natural-task transfer as the learned uncertainty;
- demand a breakthrough-sized natural gain and unchanged complete serving
  cost before calling the model smarter.

