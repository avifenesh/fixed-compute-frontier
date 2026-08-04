# Typed weight-page relational machine T34 — independent paper audit

Date: 2026-07-31  
Decision: **REJECT; ADMIT NO MICROBENCHMARK OR GPU RUN**

## Outcome

T34's intended gain is large enough, and several local arithmetic facts are
correct.  The proposed method nevertheless has no proved causal path to that
gain.  Its semantic recovery theorem is circular, its global coverage
arithmetic is not an accuracy theorem, and its strongest declared control
contains the complete typed function.

Running the comparator, packed table, compiler, model, or H100 kernel would
therefore measure components already known to work without deciding whether
the architecture creates a smarter fixed-cost model.

## Fatal issue 1 — the MDL theorem assumes recovery

The paper assumes a positive gap `gamma` between the true grammar and every
competing grammar.  Formally, for the true grammar `G*`, the assumption is

\[
L(G)+L(D\mid G)
\ge
L(G^*)+L(D\mid G^*)+\gamma
\quad\text{for every }G\not\sim G^*.
\]

This is already the conclusion that `G*` is the unique minimizer.  The proof
restates the assumption; the preceding structural assumptions do not derive
the gap.

The honest object is an MDL minimizer certificate containing the selected and
runner-up code lengths.  It certifies only the objective, not semantic truth.
A successor requires an identifiable stochastic source and a derived
finite-sample separation bound.

Equal MDL length also proves only that this objective cannot choose.  It is not
by itself a theorem that every possible raw-derived statistic is identical.

## Fatal issue 2 — coverage is not repair

The oracle gap and threshold arithmetic are numerically correct:

- baseline: 56.7308%;
- explicit-record oracle: 89.4231%;
- gap: 32.6923 points;
- target: 80%;
- required fraction of oracle gap: 0.7117639.

The reported independent and Bonferroni `p,q` thresholds are also arithmetically
correct.  But global typed-path validity does not say that the path fires on
examples the baseline gets wrong, and it does not charge harm on examples the
baseline gets right.

For typed-path selection `S`, typed correctness `T`, and baseline correctness
`B`, the exact accuracy change is

\[
A_{cand}-A_0
=
P(S\cap T\cap\neg B)
-
P(S\cap\neg T\cap B).
\]

The breakthrough gate is therefore

\[
P(S\cap T\cap\neg B)
-
P(S\cap\neg T\cap B)
\ge 0.232692.
\]

A successor must preregister repair-set coverage and baseline-correct harm,
not infer gain from global compiler and parser accuracy.

## Fatal issue 3 — the strongest control contains the method

The complete local T34 function can be written

\[
f(h,P)=J\!\left(
  \operatorname{CMP}_{c(h)}
  \left(P[a_1(h)],P[a_2(h)]\right)
\right).
\]

A conditional-memory control granted the same router, packed page, comparator,
and residual injection `J` computes exactly the same function.  Therefore

\[
\mathcal F_{T34}\subseteq\mathcal F_{control}.
\]

T34 cannot require a ten-point function-class advantage over this control.  If
the control is denied typing or comparison, it is no longer the strongest
declared control.  The only possible residual claim is an acquisition or
inductive-bias advantage; that is exactly the unproved compiler claim.

## Correct representation bounds

For a total direct-address table over `N*K` keys and `M` non-absent values, the
payload lower bound is

\[
\left\lceil NK\log_2 M\right\rceil\text{ bits}.
\]

For a partial direct-address table whose cells may also be absent, it is

\[
\left\lceil NK\log_2(M+1)\right\rceil\text{ bits}.
\]

For exactly `F` sparse entries, an information lower bound before indexes and
alignment is

\[
\log_2 {NK\choose F}+F\log_2 M.
\]

The original paper conflated direct addressing, sparse hashing, and ordered
lookup.  Direct addressing provides worst-case `O(1)` reads while paying for
the complete universe.  Sparse hashing offers expected `O(1)` probes but must
store keys and indexing state.

The following local arithmetic survives:

- raw-byte payload versus T8's robust 16-level BF16 payload has a 4x ceiling;
- removing 30 SwiGLU channels from every width-384 layer across ten layers
  frees 691,200 bytes;
- the 683,148-byte typed record leaves 8,052 bytes;
- the width reduction is 2.9297%;
- removed dense work is 345,600 MAC per processed token.

## Frozen architecture problem

The paper alternates between two non-equivalent graphs:

1. each token chooses either dense or typed execution for a page;
2. every FFN is permanently narrowed from 1,024 to 994 to fund one global
   conditional-memory sidecar.

The second is not a routed page substitution.  A routed design needs its own
routed dense/MoE control and ledger.  In either case, parser, relation metadata,
router, injection, scratch, and alignment bytes remain unallocated.  Natural
compilation, query alignment, safe selection, and residual use are four open
composition hypotheses, not one.

## Current prior-art boundary

The architecture components are already represented by current work:

- Engram reallocates a fixed parameter/active-compute budget from MoE to
  deterministic conditional lookup;
- Memory Grafting performs offline memory construction plus exact lookup and
  learned projection/gating, albeit with a stronger pretrained constructor;
- UltraMemV2 studies the learned-memory-versus-compute allocation;
- Relational Memory-Augmented Language Models retrieve explicit relation
  triples into a language model;
- knowledge compilation formalizes the expensive-offline/cheap-online exchange;
- explicit-memory and differentiable-computer models already combine token
  banks or exact programs with learned language interfaces.

The remaining possible seam is autonomous raw-to-semantic recovery under the
complete fixed serving ledger.  T34 did not supply that recovery.

## Retain and close

Retain:

- heterogeneous page types as a design vocabulary;
- packed token/relation bytes;
- exact finite comparison;
- the corrected byte/MAC ledger;
- the repair-minus-harm end-gain equation.

Close:

- T34 as a breakthrough candidate;
- its MDL recovery theorem;
- any B1--B3 implementation justified by this paper;
- any claim of novelty or capability from typed pages alone;
- any H100 work on this branch.

A successor enters a microbenchmark only after it provides:

1. a non-circular identifiability theorem;
2. repair-minus-harm effect arithmetic;
3. one frozen architecture/resource graph;
4. a claim not contained by its strongest matched control.
