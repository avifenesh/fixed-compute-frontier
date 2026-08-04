# Counterfactual interaction spectrum — paper decision

Status: **REJECT AS ARCHITECTURE; RETAIN AS COMPILER DIAGNOSTIC**  
Date: 2026-08-01

## Proposed object

For an enumerated ordered span pair in raw context `c`, substitute fixed probes
`a,b` and score the resulting text with a from-zero language model energy
`E_c(a,b)`.  Relative to reference probes `a0,b0`, form

\[
J_c(a,b)=E_c(a,b)-E_c(a,b_0)-E_c(a_0,b)+E_c(a_0,b_0).
\]

The hope was that this double difference would cancel context and entity main
effects, leaving an observable relation signature for a compact triple plane.

## Valid conditional theorem

Assume the energy really has the form

\[
E_c(a,b)=\alpha_c+\beta_c(a)+\gamma_c(b)
          +\phi(a)^\top R_{r(c)}\psi(b)+\eta_c(a,b).
\]

Let probe-difference rows be `P_i=phi(a_i)-phi(a0)` and
`Q_j=psi(b_j)-psi(b0)`.  Then

\[
J_c=P R_{r(c)}Q^\top+N_c.
\]

If `P,Q` are known and full column rank,

\[
\widehat R=P^\dagger J_c(Q^\dagger)^\top,
\qquad
\|\widehat R-R\|_F
\le
\frac{\|N_c\|_F}
{\sigma_{\min}(P)\sigma_{\min}(Q)}.
\]

If each energy residual is at most `epsilon`, each double-difference residual
is at most `4 epsilon`, hence for a `p x q` probe grid

\[
\|N_c\|_F\le4\epsilon\sqrt{pq}=B.
\]

Known-prototype classification needs observable separation greater than `2B`;
unsupervised distance clustering needs greater than `4B`, plus representation
of every type and a known noise bound.

If `P,Q` are unknown, the numeric relation matrix is not identifiable up to
permutation.  For invertible `A,B`, the transformation

\[
P'=PA,\quad Q'=QB,\quad R'=A^{-1}RB^{-\top}
\]

leaves `J` unchanged.  Only shared observable `J` prototypes can be clustered.

## Fatal semantic counterexamples

The theorem does not imply that nonzero interaction means relation truth.

- `salt and pepper` can have strong non-additive language-model energy from
  collocation without expressing the intended knowledge relation.
- `Alice is Bob's mother` can have additive energy
  `u(Alice)+v(Bob)+constant`, yielding `J=0` despite a true relation.

Agreement, type compatibility, token length, memorized co-occurrence,
negation, modality, and multiple simultaneous relations all contaminate the
same observable.  Thus “non-event is additive” is the semantic conclusion,
not a consequence of next-token training.

## Enumeration and query walls

An `L`-token record has `T=L(L+1)/2` spans and `O(L^4)` ordered span pairs.  At
`L=128`, exhaustive enumeration has 68,153,280 ordered pairs before excluding
overlap, or 22,717,760 ordered disjoint pairs.  Each then requires a probe grid
of model evaluations.  Pruning needs the entity/span selector the proposal was
meant to discover, while millions of tests require correspondingly stringent
family-wise false-positive control.

Declarative and interrogative clusters also admit independent label
permutations unless a cross-surface anchor or a literally shared observable
prototype is supplied.  The current task requires a program such as

```text
lookup birthDate(A); lookup birthDate(B); compare <
```

rather than one relation lookup.  The interaction spectrum supplies neither
that program nor argument binding.

## Control and effect-size audit

Offline counterfactual compilation can legitimately amortize work that one
online pass cannot perform, so “the scorer already knows it” is not a no-go.
But a matched relation extractor or knowledge-graph compiler receives the same
corpus, model, offline budget, triple plane, and fixed lookup/join/comparison
operators.  CIS has no proved acquisition or serving-function advantage over
that control.

The full serving ledger must still fund entity IDs, relation codes, indices,
query parsing, integer operations and traffic, residual injection, and the FFN
capacity removed to make room.  Nothing in the conditional theorem bounds
natural extraction coverage, query-program accuracy, or harm on examples the
baseline already answers.  It therefore cannot derive the required move from
56.7308% to 80%.

## Decision

Do not implement or run CIS.  Retain `J` and its noise bound as a diagnostic
for comparing raw-only compilers after a separately valid event selector is
available.  It is not itself the semantic bridge.
