# T38 local COPY upper-bound pre-run audit

Date: 2026-08-01  
Decision: **REJECTED BEFORE CORPUS ACCESS; ZERO CENSUS RUNS**

## What is mathematically valid

For a fixed tokenized corpus `D`, fixed literal set `L`, and event position
`i >= 8`, any correct T38 `COPY` prediction obeys

\[
1[\text{correct COPY at }i]
\leq
1[y_i\notin L\ \land\ \exists j\in[\max(0,i-32),i):x_j=y_i].
\]

The target must be an eligible token already present in the selected head's
context.  Summing the right-hand side and dividing by every event therefore
gives an optimistic within-corpus ceiling: it grants a perfect compiler,
unlimited records, perfect arbitration, and no purity or support failure.

This lemma is retained.  It is sufficient to close the 3% COPY gate only when
evaluated on the exact frozen adjudication corpus with the exact literal set
derived from its frozen training split.  A partition-independent but looser
ceiling may omit `y_i notin L`, but it must still use that exact adjudication
corpus.

## Fatal inference error

The proposed audit substituted project-local markdown and code for the frozen
OpenWebMath and CodeParrot adjudication documents.  No deterministic or
stochastic dominance relation was proved between those distributions.  A
local ceiling below 3% therefore does not imply that the remote ceiling is
below 3%.

A direct counterexample is enough.  Let every local eligible target be new in
its preceding 32-token window, so the local ceiling is zero.  Let the remote
corpus repeatedly emit one eligible token already in that window, so its
ceiling approaches one.  Both corpora satisfy the audit's syntactic rules.
Thus the desired implication is false.

The proposed audit also chose the 8,192 literals by frequency over the complete
local corpus.  T38 freezes them from the remote training split only.  Equal set
cardinality gives neither inclusion nor preservation of eligible repeated
events, so this substitution can undercount the events relevant to T38.

## Additional integrity failures

The review also found that the prototype:

1. relaxed 16- and 32-token heads at early positions rather than reporting
   exact per-head availability;
2. had no exclusive opening/final seal binding the implementation, tests,
   tokenizer, and an explicit raw-byte input manifest before corpus access;
3. did not itself verify and bind the preregistered pytest receipt;
4. allowed a mutable glob universe and replacement UTF-8 decoding; and
5. mapped an empty decisive stratum to zero and used floating-point threshold
   comparison where exact integer arithmetic was available.

The toy expected unrestricted-repeat count was corrected from three to two,
after which the three prototype unit tests passed.  That repair does not cure
the fatal corpus/partition inference and does not authorize a run.

## Research decision

Do not execute or rescue-tune this local proxy.  It could only produce a local
descriptive statistic, while the claimed closure requires the exact frozen
T38 train/adjudication pair.  T38 remains algebraically valid but empirically
and physically unadmitted.  Its two sealed remote attempts remain
infrastructure-inconclusive, not positive or negative scientific results.

The only admissible reuse of the lemma is a newly preregistered audit over the
exact frozen corpus and train-derived literal set, or the unrestricted-repeat
ceiling over the exact adjudication corpus.  No third remote fetch retry is
authorized here.
