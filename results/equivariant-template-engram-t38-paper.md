# T38 equivariant template Engram — paper derivation

Date: 2026-08-01  
Status: **ALGEBRA RETAINED; FROZEN EMPIRICAL PROTOCOL CLOSED; NO MODEL OR GPU RUN ADMITTED**

The two sealed attempts produced no corpus statistic.  A later provenance
audit found that the frozen CodeParrot row count names a partial viewer
derivative while the protocol requires `partial=false`; direct-shard sampling
would change the population and all offsets.  The empirical protocol is closed
in
[`equivariant-template-engram-t38-protocol-closure.md`](equivariant-template-engram-t38-protocol-closure.md).
The orbit/covariance algebra is retained and generalized under an explicit E3
claim in
[`group-action-executable-memory-t41-paper.md`](group-action-executable-memory-t41-paper.md).

## Claim in one sentence

Replace a fixed fraction of lexical conditional-memory heads with an exact
renaming-equivariant table whose keys are token-equality templates and whose
values are relative copy/literal operations, so one learned raw-text pattern can
serve combinatorially many concrete names at the same table bytes and active
lookup count.

This is not semantic hashing.  The address is an exact quotient under a declared
permutation group, and the value co-transforms with the renamed input.

## P1. Frozen algebra

Partition the tokenizer vocabulary into disjoint sets

\[
V=L\sqcup E,
\]

where `L` contains literal tokens fixed by renaming and `E` contains eligible
interchangeable tokens.  The group `G=Sym(E)` permutes `E` and fixes every token
in `L`.  The natural implementation must derive the partition from a frozen
raw-only rule, such as a training-frequency boundary; the theorem is conditional
on the declared partition and does not claim that every rare token is
semantically interchangeable.

For a length-`n` context `x=(x_1,...,x_n)`, canonicalize left to right:

1. emit a tagged literal `LIT(x_i)` when `x_i in L`;
2. on the first occurrence of an eligible token, append it to binding vector
   `beta=(beta_0,...,beta_{k-1})` and emit `VAR(j)`;
3. on a repeated eligible token, emit the existing `VAR(j)`.

Write

\[
\kappa(x)=(\tau(x),\beta(x)),
\]

where `tau` is the literal/equality template and `beta` lists eligible tokens in
order of first occurrence.

### Canonicalization theorem

For every `g in G`,

\[
\tau(gx)=\tau(x),\qquad \beta(gx)=g\beta(x).
\]

Proof: `g` preserves literal identities and is a bijection on eligible tokens,
so it preserves exactly the first-occurrence and equality relation.  The `j`th
new eligible token remains the `j`th new token after renaming, with identity
`g beta_j`.  Induction over context positions proves both statements.

The converse also holds: two contexts have the same template exactly when a
bijection of their encountered eligible tokens, fixing literals, maps one
context to the other.  That finite bijection extends to an element of
`Sym(E)`.  Thus `tau` is a complete canonical **orbit label/normal form**, not
a probabilistic collision.  It is not literally a representative in the
original context space unless the tagged `VAR` alphabet is embedded back into
that space.

## P2. Covariant payload

For next token `y`, define the representable operation

\[
\eta_x(y)=
\begin{cases}
\operatorname{LITERAL}(y), & y\in L,\\
\operatorname{COPY}(j), & y=\beta_j,\\
\bot, & y\in E\setminus\{\beta_0,\ldots,\beta_{k-1}\}.
\end{cases}
\]

The interpreter is

\[
\psi(\operatorname{LITERAL}(a),\beta)=a,
\qquad
\psi(\operatorname{COPY}(j),\beta)=\beta_j.
\]

For a template with `k(tau)` variables, constrain its table entry to

\[
T[\tau]\in
\{\operatorname{LITERAL}(a):a\in L\}
\cup
\{\operatorname{COPY}(j):0\leq j<k(\tau)\}.
\]

The discrete expert predicts

\[
F_T(x)=\psi(T[\tau(x)],\beta(x)).
\]

### Exact equivariance theorem

For every `g in G` and every represented context,

\[
F_T(gx)=gF_T(x).
\]

Proof: the key is unchanged by P1.  A literal payload is fixed by `g`; a copy
payload selects `g beta_j` after renaming.  Applying the interpreter gives the
identity directly.

This is the critical difference from an invariant key with a static vector
value.  The relative payload preserves which concrete symbol fills each role.

## P3. Raw-only compiler

The compiler needs no question, answer, parser, pretrained model, or benchmark
label.  For every raw training token position:

1. canonicalize the preceding `n` tokens;
2. convert the observed next token to `LITERAL`, `COPY`, or bottom;
3. count operations under the template; and
4. retain a template only when its non-bottom support and operation purity pass
   thresholds frozen before held-out evaluation.

The compiled plane stores a winning operation, support, and confidence, or a
small quantized operation distribution.  Ambiguous templates abstain.  This
makes the claimed extraction observable: each cell is explained by its raw
occurrences, equality pattern, relative target, and empirical purity.

End-to-end training may replace counts with learned operation logits later, but
the count compiler is the strongest decoded-information oracle and must pass
first.  A neural model is not allowed to rescue an impure quotient.

## P4. Density and representation-sharing separation

Suppose a retained template contains `k` distinct eligible variables and the
eligible vocabulary has size `M=|E|`.  Its orbit contains

\[
(M)_k=M(M-1)\cdots(M-k+1)=\frac{M!}{(M-k)!}
\]

concrete injective renamings.  One template cell plus a relative operation is
exact on the entire orbit.

Under a frozen concrete-key dictionary that stores one represented key record
per admitted concrete `n`-gram, an exact lexical table whose payload is a
static token ID needs a distinct key record for each concrete renaming on which
the target changes.  Payload IDs may be interned, but without an independent
equivariant address/interpreter one lexical cell cannot emit both `beta_j` and
`g beta_j`.  A structured hash or circuit that does share these entries is not
excluded; it is itself a quotient/interpreter control and must be charged as
such.  Therefore `(M)_k` is an exact orbit size and a possible key-record
sharing factor against the declared concrete dictionary, **not** an
information-theoretic storage lower bound over arbitrary static circuits.

On an orbit-equivariant target law, the template operation is correct on every
binding in the orbit, including unseen bindings.  Observed empirical purity by
itself does not prove that the natural target law has this symmetry.  T38 has
therefore proved a representation/parameter-sharing separation, not a
statistical sample-complexity theorem; the latter would require a distribution,
learner, noise model, and finite-sample bound.

The theorem is deliberately narrow:

- it compares exact key/value memory parameterizations, not an unrestricted
  Transformer that might learn the same symmetry internally;
- it applies only where the correct output is a fixed literal or a bound input
  variable; and
- it says nothing yet about how much natural prose or held-out reasoning lies
  in that family.

Those boundaries isolate the empirical uncertainty instead of hiding it.

## P5. Serving design at matched allocation

Use T38 as one branch inside a conditional-memory hybrid, not as an added
module.  Starting from an Engram-like allocation with `K` active hash heads:

- keep total table bytes fixed;
- replace, rather than add, `K_eq` lexical heads with template heads;
- keep the number of active table reads fixed;
- keep the MoE/dense backbone parameter and active-FLOP budgets fixed; and
- admit a deployment only after complete traffic, workspace, throughput,
  p50/p95/p99 latency, and energy are noninferior.

For fixed `n`, canonicalization can be an unrolled comparison network.  The
equality relation alone uses at most `n(n-1)/2` token-equality comparisons.
The complete ledger must additionally charge literal-mask tests,
first-occurrence selection, distinct-rank/`VAR(j)` formation, binding-vector
routing, template hashing, and all transient workspace.  The binding values
are token IDs already present in the context, but any separately materialized
binding vector is still charged.  Exactness also requires either a
collision-free dictionary/minimal perfect hash over retained templates, or a
stored fingerprint followed by equality verification and abstention on
mismatch; an unchecked hash is not the theorem.  The table payload can fit in
a few bytes:

\[
\lceil\log_2(|L|+n+1)\rceil
\]

bits for literal/copy/bottom identity, plus frozen confidence/count fields.

Two integration levels must be kept separate:

1. **logit expert:** scatter a calibrated bias to the interpreted token ID;
   cheapest, but only a next-token expert;
2. **residual expert:** gather the interpreted token's shared embedding and
   inject it before later layers through a charged scalar/vector gate; capable
   of changing downstream computation but more expensive.

The residual path is not granted the logit path's ledger.  Its embedding read,
gate, normalization, and downstream interference must be measured.  Saved
table-vector bytes or projection work may fund it, but only under the complete
resource vector.

## P6. Why this is potentially larger than a copy trick

Lexical conditional memory stores surface-specific regularity.  T38 stores a
small executable rule over bindings.  Examples in its exact family include:

```text
A ... B ... A       -> COPY(0)
A gave B to C; B    -> COPY(1)
x = y; return x     -> COPY(0)
```

The same cell applies after every eligible-symbol permutation.  This can
offload equality tracking, variable reuse, copying, and name-independent
templates from continuous layers.  In a hybrid, lexical heads remain
responsible for entity-specific facts while equivariant heads target reusable
program structure.  The research hypothesis is that separating **surface
memory** from **binding programs** frees more backbone depth for reasoning than
spending the same inactive bytes only on lexical N-grams or MoE experts.

This is not yet a production-model gain.  Natural names are often multi-token,
many rare tokens are not interchangeable, long binding distances exceed small
`n`, and most factual answers introduce a token absent from the query context.
Those are measurable kill conditions, not details to tune after a result.

## P7. Strongest controls

Any eventual experiment must include, at the same total and active allocation:

1. pure MoE/dense frontier baseline;
2. current lexical Engram with all `K` lexical heads;
3. lexical Engram with the same few-byte token/opcode payload but no quotient;
4. equality-template key with a static learned vector but no relative
   interpreter;
5. relative copy interpreter keyed by concrete lexical N-grams;
6. random or frequency-matched template assignments; and
7. T38 with both the canonical key and covariant payload.

Controls 3--5 are mandatory: they isolate whether any gain comes from compact
payloads, key sharing, or the equivariant read rather than bundling all three.
At the later model stage, add two matched architecture controls: an
Engram-backed model with the same equivariant-head allocation but a learned
static payload, and a compute/state-matched Symbol-Invariant/Renamer-style
network.  The latter asks whether a table is a better fixed-budget realization
of the already-known function family, not whether T38 invented that family.

## P8. Prior-art boundary

- Parameterized string matching already studies equality under alphabet
  bijections; for a current reference see
  [Algorithms for Parameterized String Matching with Mismatches](https://arxiv.org/abs/2412.00222).
- [Interchangeable Token Embeddings](https://arxiv.org/abs/2410.17161) and the
  ICML 2026 [Symbol-Invariant Transformer](https://arxiv.org/abs/2601.23169)
  directly establish alpha-equivalence/open-vocabulary neural inductive biases.
- Copy/pointer networks and induction mechanisms already provide relative
  token reuse.
- [Engram](https://arxiv.org/abs/2601.07372) already provides iso-parameter,
  iso-active-FLOP multi-head hashed N-gram memory with context-aware gating.

No novelty is claimed for canonical renaming, copying, N-gram lookup, or
invariant neural networks separately.  The candidate contribution is the
specific **conditional-memory quotient plus covariant opcode payload**, trained
or compiled from ordinary raw next-token events and substituted inside a fixed
Engram/MoE allocation.  The search so far found close components but not this
combination; that is a provisional collision statement, not proof of novelty.
In particular, the Symbol-Invariant Transformer already contains the relevant
equivariant function family.  T38 can only win by realizing a useful subset of
that family with better fixed-budget memory, serving, or learning efficiency;
it cannot claim functional expressivity unavailable to that architecture.

## P9. Pre-run admission gates

Before any model training or GPU use, a frozen CPU opportunity audit must answer:

1. What fraction of raw next-token events is representable as literal/copy
   under each predeclared `n` and raw-only partition?
2. How many templates have enough support and held-document purity to survive?
3. What is the effective orbit reuse: distinct concrete bindings per retained
   template, not the theoretical `(M)_k` maximum?
4. On untouched natural reasoning/code text, how often does a retained
   operation remain correct on a document-disjoint concrete binding never seen
   under that template during compilation?  Mechanical renaming of a
   context/target pair is only an implementation metamorphic test and cannot
   count as evidence that natural language obeys the symmetry.  A semantic
   renaming test would require a separately frozen lexer/entity rule and is not
   silently inferred from token equality.
5. What is the optimistic aggregate repair ceiling after lexical heads removed
   for `K_eq` are charged?
6. Do shuffled bindings, shuffled operations, and concrete-key controls destroy
   the apparent gain as predicted?

The audit is admitted only after an independent theorem/resource review.  It
must use a small local corpus and finish cheaply on CPU.  It may not train a
model, tune on downstream labels, or select a partition after seeing benchmark
accuracy.

Advance beyond the census only if all of the following preregistered numerical
gates pass on a document-disjoint evaluation set:

- at the same total table bytes and lookup count, the conservative net decoded
  top-1 event advantage over the strongest concrete-key lexical control is at
  least **3.0 absolute percentage points** on the frozen natural
  prose/reasoning/code mixture, with a paired 95% bootstrap lower bound above
  **2.0 points**;
- every test-active admitted template with at least **20** held-document
  occurrences has at least **95%** operation purity, the aggregate
  retained-event purity is at least **97%**, and the same aggregate threshold
  holds when restricted to concrete binding tuples unseen under that template
  during compilation;
- the median admitted template has at least **8** distinct training binding
  tuples and its 75th percentile has at least **16**, excluding duplicate
  documents and exact repeated contexts;
- correct non-literal `COPY` coverage is at least **3% of all held-out token
  events** on the natural reasoning/code slice, outside synthetic
  alpha-renaming;
- a credible equal-allocation ledger against lexical Engram; and
- one remaining uncertainty: whether the from-zero hybrid uses the proven
  quotient without harming its lexical path.

The net event advantage counts `+1` when T38 is correct and the concrete-key
control is wrong or abstains, `-1` for the reverse, and `0` otherwise, divided
by **all** held-out next-token events.  Both memories use the same frozen
capacity, deterministic admission order, and abstain-to-no-prediction fallback;
wrong non-abstaining predictions are charged.  This prevents a tiny selected
tail from passing through conditional accuracy and makes the census a decoded
memory comparison rather than a coverage/purity product.  These are
opportunity gates, not a claim that three next-token points automatically
become three benchmark-score points.  Equal bytes and reads at this stage are
design constraints only; full traffic, workspace, latency, and energy parity
remain unproved until a physical implementation.

## Current decision

T38 survives the algebraic screen and has a specific representation/parameter-
sharing separation over a frozen concrete lexical key/value dictionary.  It
does **not** earn a model run.  The independent review found no fatal algebraic
error but required the restrictions above.  The original sealed corpus fetch
and one separately reviewed rate-limited transport retry both stopped before
corpus materialization because the remote source did not satisfy the frozen
transport/provenance contract.  Neither attempt produced a manifest, finalized
seal, census result, or statistic, so they are infrastructure-inconclusive.

A proposed local structure-rich COPY-opportunity proxy was then
[rejected before corpus access](equivariant-template-engram-t38-local-upper-bound-pre-run-audit.md).
Its within-corpus oracle inequality is valid, but neither its document
distribution nor its full-corpus literal partition upper-bounds the frozen
remote adjudication corpus with its train-derived partition.  Zero local census
runs were made, and no local result exists.

T38 is therefore retained as an exact algebraic candidate with unresolved
natural coverage, reuse, harm, control, and physical-serving questions.  A new
empirical attempt would require fresh authorization and a new preregistration
over the exact frozen corpus; no automatic third remote retry, local GPU, or
rental is admitted.
