# T41 group-action executable memory — paper derivation

Date: 2026-08-01  
Status: **CLOSED BY INDEPENDENT PAPER AUDIT; RETAIN LEMMAS ONLY; ZERO RUNS**

## Independent audit verdict

T41 is not an E3 separation from the strongest matched learner.  A learner
given the identical canonicalizer, pooled counts, compiler, and interpreter
ties it exactly.  The concrete lexical dictionary bound is correct but applies
only to a weaker learner that is denied the quotient.  Post-T35 absorption
therefore remains fatal.

The proposed natural uncertainty in P8 is also `Delta A` itself.  It bundles
parsing, source-law fit, opcode purity, admission, displaced-memory harm,
gating, and downstream use, which L6 forbids.  The finite-field affine theorem
does not define signed-decimal token parsing or causal multi-token
serialization, and the paper supplies no complete resource exchange.

Consequently no exhaustive finite-field implementation, corpus census, model
run, local GPU work, or rental is admitted.

## Claim in one sentence

Replace some surface-specific conditional-memory records with orbit keys and
small covariant opcodes, so one raw-observed program serves many renamed or
numerically transformed bindings while total table bytes and active lookup
count remain fixed.

T38 is the identifier-permutation special case.  T41 changes the research
object from one `COPY` trick to a typed family of exact group actions and asks
whether executable records can offload both lexical reconstruction and simple
symbolic reasoning from the dense backbone.

## P1. General orbit-program theorem

Let a group `G` act on contexts `X` and outputs `Y`.  Suppose a constructive
canonicalizer returns

\[
\kappa(x)=(\tau(x),b(x))
\]

with orbit label `tau` and binding `b`, satisfying

\[
\tau(gx)=\tau(x),\qquad b(gx)=g\,b(x).
\]

Let an opcode `u(tau)` be interpreted by `psi(u,b)`, and require

\[
\psi(u,g b)=g\,\psi(u,b).
\]

Then

\[
F(x)=\psi(u(\tau(x)),b(x))
\]

is exactly `G`-equivariant:

\[
F(gx)=gF(x).
\]

The proof is substitution.  The key is invariant, so the same opcode is read;
the covariant binding and interpreter transform its result by `g`.

This theorem does not discover `G`, prove that natural text follows the group
law, or make an arbitrary semantic relation equivariant.  Every admitted type
must provide its own raw parser, canonical form, interpreter, and failure
surface.

## P2. Two exact typed instances

### Identifier permutations

Eligible symbols are renamed by `Sym(E)`.  The key records literal tokens and
first-occurrence equality classes; the binding records concrete symbols.
Valid opcodes include `LITERAL(a)` and `COPY(j)`.  This is exactly the retained
T38 construction.

### Affine numeric bindings

Work first over a finite field `F_p`.  For a context containing two distinct
numeric binders `z0,z1`, define the binding

\[
b=(z_0,z_1)
\]

and canonical coordinate

\[
c(z)=\frac{z-z_0}{z_1-z_0}.
\]

For an affine action `g(z)=az+d`, `a != 0`,

\[
c(gz)=\frac{az+d-(az_0+d)}{az_1+d-(az_0+d)}=c(z).
\]

The opcode `AFFINE(c)` is interpreted as

\[
\psi(\operatorname{AFFINE}(c),(z_0,z_1))
=z_0+c(z_1-z_0).
\]

It is exactly covariant because applying `g` to both binders applies `g` to
the result.  One record can therefore represent an entire affine orbit, such
as a constant-step sequence, rather than storing every concrete number tuple.

Natural decimal integers are not silently identified with a finite field.  A
production type needs an exact signed-decimal parser, overflow/range rules,
and either integer/rational partial actions or an explicit modular semantics.
Those costs and abstentions are part of the method.

## P3. Finite-sample separation from concrete lexical memory

Fix one orbit template with `k` interchangeable bindings and orbit size

\[
P=(M)_k.
\]

Assume bindings are uniform over the orbit and the correct opcode is fixed.
In the noiseless case, one observed binding identifies the opcode for the
orbit-factored estimator and it is correct on all `P` bindings.

A concrete dictionary that can answer only an observed binding has expected
accuracy on a fresh binding

\[
A_{lex}(n)=1-\left(1-\frac1P\right)^n
\]

after `n` independent observations.  To make its unseen-binding error at most
`epsilon`, it needs

\[
n\ge
\frac{\log \epsilon}{\log(1-1/P)}
\approx P\log(1/\epsilon).
\]

For symmetric independent opcode noise `nu < 1/2`, let
`gamma=1/2-nu`.  Majority compilation from `n` orbit-pooled occurrences has

\[
\Pr[\hat u\ne u]\le \exp(-2n\gamma^2)
\]

by Hoeffding, so

\[
n\ge \frac{\log(1/\delta)}{2\gamma^2}
\]

suffices independently of `P`.

This is a real E3 separation against the declared concrete-key learner.  It is
not a lower bound on an unrestricted Transformer, a Symbol-Invariant network,
or any control already given the same canonicalizer and interpreter.  Such a
control must tie T41 and is included to expose attribution.

## P4. What can become better

The proposed gain is not faster inference by itself.  Under a fixed
conditional-memory allocation, orbit records can buy:

1. higher held-out accuracy on unseen identifier and numeric bindings;
2. more distinct reusable programs at the same table bytes;
3. exact extrapolation outside the concrete values seen during training;
4. fewer dense layers spent reconstructing local symbolic regularities; and
5. a hybrid split in which lexical records retain entity facts while
   executable records handle reusable operations.

The fifth item is the production hypothesis.  Engram already shows that
conditional memory can trade static reconstruction for effective reasoning
depth.  T41 asks whether storing covariant operations rather than only static
vectors increases the useful work obtained from that same memory allocation.

## P5. Typed raw-only compiler

For every raw next-token event, the compiler may use only frozen tokenization,
exact token equality, and exact parsers belonging to declared types.  It:

1. enumerates valid typed windows;
2. canonicalizes each window;
3. converts the observed next token to a valid opcode or bottom;
4. pools opcode counts over the orbit;
5. retains only records satisfying frozen support, diversity, and confidence
   bounds; and
6. abstains on parse, range, collision, ambiguity, or purity failure.

Questions, answers, support spans, pretrained parsers, teacher hidden states,
and evaluator feedback are forbidden.  This directly avoids the stronger
offline-teacher assumption used by Memory Grafting.

The compiler is explainable: every record exposes its canonical key, concrete
bindings, opcode counts, confidence interval, and counterexamples.

## P6. Strongest controls

At equal persistent bytes, active lookups, backbone work, and training token
budget, a later test must include:

1. dense or MoE frontier baseline;
2. lexical Engram with static learned vectors;
3. lexical records with the same compact opcode payload but concrete keys;
4. invariant orbit keys with static outputs and no covariant interpreter;
5. identifier-only T38;
6. a matched symbolic arithmetic unit with a learned gate;
7. a compute/state-matched Symbol-Invariant Transformer;
8. T41 with shuffled opcodes, bindings, and type tags; and
9. an identical canonicalizer/interpreter control, which must tie T41.

Control 9 prevents a false function-class claim.  The candidate contribution
is the explicit group-action parameterization and its finite-budget acquisition
behavior, not an instruction withheld from the control.

## P7. Resource ledger still required

For window bound `w`, type count `T`, and `H` active memory heads, charge:

- literal/type tests and exact numeric parsing;
- at most `w(w-1)/2` equality comparisons for identifier canonicalization;
- modular or checked integer arithmetic for numeric canonicalization;
- collision-free lookup or fingerprint plus equality verification;
- opcode, confidence, support, type, and range metadata;
- binding-vector workspace;
- opcode execution and token re-encoding;
- any residual injection, normalization, projection, and gate; and
- divergence, traffic, p50/p95/p99 latency, and energy.

The candidate replaces memory heads; it is not added to them.  A logit-only
opcode path and a residual path are different served graphs and receive
different ledgers.

No statement here proves that canonicalization is latency-neutral on an H100
or H200.

## P8. End-effect equation and materiality gate

Let `S` be the event that T41 fires, `T` that its interpreted prediction is
correct, and `B` that the matched frontier prediction is correct.  With
baseline fallback outside `S`, the exact accuracy gain is

\[
\Delta A=P(S\cap T\cap\neg B)-P(S\cap\neg T\cap B).
\]

The algebra proves correctness only inside declared orbit laws.  The sole
natural-corpus question may eventually be the signed mass of this equation
after the raw compiler is frozen.  A next-token coverage statistic alone is
not a production-model bridge.

Before model training, a decoded oracle at the exact deployed types must show:

- at least ten absolute points on held-out, unseen-binding symbolic reasoning;
- a causal loss of at least 90% of that advantage under opcode/binding
  shuffles;
- no more than 0.5 points of harm on nonmatching/protected events; and
- an optimistic production-slice mass large enough that a ten-point target is
  not arithmetically impossible.

If the valid orbit mass is small, T41 closes even if its conditional accuracy
is perfect.

## P9. Prior-art boundary

The components are known:

- Engram provides the strongest current lexical conditional-memory baseline;
- Symbol-Invariant Transformer provides a direct renaming-equivariant neural
  control;
- NALU and OccamLLM provide arithmetic-module controls; and
- parameterized matching and group canonicalization are classical.

The research object is their fixed-budget conjunction: raw-only count
compilation, exact orbit keys, covariant executable payloads, and replacement
of lexical memory records under one complete serving ledger.  No novelty or
capability claim is made until the conjunction beats all controls.

## Final paper decision

Close T41 as an active breakthrough candidate.  Retain only:

1. the conditional group-action equivariance theorem;
2. T38's identifier-permutation canonicalizer and covariant opcodes;
3. the distinct-binder affine invariant/interpreter over `F_p`;
4. the concrete-dictionary expected-coverage formula, explicitly not a bound
   on a quotient-aware learner;
5. the fixed-template Hoeffding bound under iid, constant-opcode assumptions;
   and
6. the exact tie with the identical canonicalizer/interpreter control.

No execution follows from those lemmas.
