# Stage -2 semantic-bridge search contract

Status: **ACTIVE SEARCH SPECIFICATION; NO CANDIDATE ADMITTED**  
Date: 2026-07-31

## Why this is the search object

The current evidence has localized the frontier:

- finite digital storage is sufficient for the tested record budget;
- exact fixed-cost write/read identities exist;
- entity addressing can be made explicit;
- generic continuous reads, next-token gradient energy, local lexical
  quotients, and renaming-only invariance do not expose natural relation truth.

The missing function is not another storage table.  It is a raw-computable
**semantic bridge** that connects different surface forms because they express
the same usable relation.

Let `S` be raw surface instances and `R` latent relation programs.  A bridge is
an observable object `B(D)` built from raw corpus `D` for which a constructive
algorithm recovers

\[
\hat r:S\to\{1,\ldots,K\}

\]

up to a global permutation, with the recovered code sufficient for both
document writes and unseen query reads.  Global permutation is harmless: the
writer and reader use the same recovered basis.

## The one function allowed to vary

The exact record substrate, destructive controls, served budget, and final
evaluation remain fixed.  Candidate generation may vary only

```
bridge(raw corpus) -> relation basis + raw-surface encoder
```

until one bridge passes its decoded-information oracle.  This prevents a failed
semantic extractor from being hidden by simultaneous changes to storage,
reader width, decoder, model size, or runtime.

## Candidate interface

Every proposal supplies four typed functions:

```
observe(document_id, title, raw_tokens) -> raw events
fit(multiset[raw event]) -> frozen bridge state
encode(frozen bridge state, raw surface) -> discrete relation code
explain(code) -> observable witnesses and confidence
```

`observe` may use equality, token positions, counts, and other declared raw
operations.  It may not use questions, answers, support labels, a pretrained
model, parser, embedding service, or hidden benchmark feedback.

`explain` is not prose generated after the fact.  It returns the raw incidence
edges, anchors, moment components, grammar productions, or other evidence that
caused the assignment.

## Mandatory mathematical packet

### S0. Generative family

State a family `P(D | R,N)` separating the intended relation program `R` from
nuisance `N`.  Name which assumptions are structural and which are expected to
hold only approximately in natural language.

### S1. Recovery theorem

Prove that the algorithm recovers `R` up to declared harmless ambiguities under
the family.  Accepted forms include tensor/moment identifiability, separable
factorization, support-component recovery, grammar identifiability, or another
constructive result.

### S2. Counterexample theorem

Give the smallest corpus on which recovery is impossible or aliases two
relations: missing cross-instance bridges, rank deficiency, multiple relations
per entity pair, disconnected query vocabulary, or an equivalent obstruction.

### S3. Query theorem

Specify why an unseen question surface is encoded in the same basis as the
declarative surface.  “The Transformer should generalize” is an empirical
claim, not a theorem.  If exact alignment is not provable, it must be the sole
composition hypothesis and receive an isolated raw-to-query microbenchmark.

### S4. Information and compute ledger

Count bridge state, compiler work, derived training targets, plane bits,
physical weight writes, and every served operation.  Training-only state is
deleted at export but remains charged to the training frontier.

### S5. Breakthrough-size derivation

Propagate primitive error rates through the intended reasoning operation.  For
example, if a two-record conjunction requires two independent correct relation
reads with probability `p`, its ceiling is at most `p^2`; a 75% task target
therefore requires `p >= sqrt(0.75) = 0.8660` before other errors.  Reject a
bridge whose measured margins cannot support the final effect size.

## Evaluator tiers

### E0. Exhaustive identifiable worlds

Generate small raw-language worlds with hidden relation programs, random entity
renamings, independent surface forms, and known nuisance variables.  Candidate
code sees only raw strings.  Exhaustively score:

- relation recovery up to optimal label permutation;
- unseen-entity and unseen-surface encoding;
- exact record write/read;
- one-hop and composed reasoning;
- state bits and operation counts.

This tier checks the theorem implementation; passing it does not validate the
natural-language assumptions.

### E1. Adversarial assumption worlds

Include at least:

- one entity pair expressing multiple relations;
- relations with disjoint vocabularies across train and query;
- style or topic perfectly correlated with relation during fitting and broken
  at evaluation;
- singleton facts with no cross-instance bridge;
- imbalanced relations and near-rank-deficient moments;
- random labels with identical raw statistics.

The implementation must fail or lower confidence exactly where its theorem
predicts.  Confident success on an information-theoretically ambiguous world
is a rejection.

### E2. Natural bridge audit

On the development corpus, before any model training, report:

- event and bridge coverage;
- cross-entity and cross-surface multiplicity;
- ambiguity/impurity;
- rank, singular gap, anchor margin, or the candidate's corresponding
  identifiability statistic;
- fraction of question predicates reachable from the frozen raw basis;
- sensitivity to document, entity, and surface shuffles.

Development labels may be used only by the evaluator after the bridge is
frozen.  The bridge implementation never receives them.

### E3. Stronger decoded-information oracle

Decode recovered codes and values without quantization, hashing, physical FFN
placement, or learned record reads.  Use the strongest fixed reader allowed by
the preregistration.  The candidate representation must show a
breakthrough-sized natural gain and a large correct-versus-shuffled causal
drop.  Failure closes the bridge before a from-zero LM run.

### E4. From-zero composition

Only after E0-E3 pass, co-train the unchanged served model from the same random
initialization and raw token pool.  Frontier controls receive the same derived
views and charged compute.  The sole new uncertainty is whether the model can
use the already validated bridge/plane composition.

## Search scoring

An LLM or program search may propose bridge algorithms, but the score is
lexicographic rather than one averaged metric:

1. mathematical/integrity validity;
2. no confident violations on E1;
3. natural bridge coverage and identifiability margin;
4. decoded natural sufficiency and causal control gap;
5. training and served resource ledger;
6. simplicity and independent explainability;
7. only then, model capability.

Invalid candidates receive no gradient from downstream accuracy.  This avoids
evolving hidden label leaks or increasingly elaborate readers around an
insufficient representation.

## Candidate populations

Maintain diversity by separating evidence sources rather than architecture
names:

1. cross-instance incidence and moment factorizations;
2. naturally repeated independent descriptions;
3. identifiable grammar/MDL programs;
4. sequence-derived predictive equivalence classes;
5. other raw-computable bridges with a new recovery theorem.

Entity-renaming-only, local-window expansion, post-hoc hidden geometry, generic
latent vectors, and stronger readers over the closed T24 quotient are excluded
populations.

## Current decision

No semantic bridge is admitted yet.  The first attempted family—renaming plus
same-instance multi-view consistency—fails S1 because its support components
identify sentence orbits unless an additional cross-paraphrase bridge is
already available.  The proof is in
[`entity-renaming-semantic-identifiability-no-go.md`](entity-renaming-semantic-identifiability-no-go.md).

The second attempted family—cross-instance three-view moment factorization—has
a valid conditional identifiability theorem but fails before implementation:
raw prose does not supply relation-event boundaries or conditionally
relation-specific views, and natural questions are not a fitted view.  The
paper audit is in
[`cross-instance-moment-bridge-t26-paper-audit.md`](cross-instance-moment-bridge-t26-paper-audit.md).

The next paper candidate must instantiate one of the remaining evidence
sources and survive S0-S5 before reference code or GPU use.

T31 instantiated sequence-derived predictive equivalence. Exact future-law
equivalence is a canonical right congruence and minimal sufficient state, but
its product coordinates are not observationally identifiable and unrestricted
emissions restore the full joint-state interaction cost. It is closed on
paper in
[`predictive-congruence-factor-plane-t31-paper-audit.md`](predictive-congruence-factor-plane-t31-paper-audit.md).

The retained question is narrower than “find a compact predictive state”:
does raw language expose a recoverable factorization whose predictive
interaction degree remains bounded as useful knowledge grows? No architecture
or GPU run is admitted until that law has an observable witness and a matched
distributed-state baseline.

T28 instantiated cross-instance incidence with an exhaustive raw event
lattice rather than selected relation spans.  Its
[paper gate](extensional-incidence-quotient-t28-paper.md) proves exact and
noisy recovery under extension separation, but the frozen raw-only census
found 3.7398% multiply witnessed tuples and zero edges between recurring
pattern nodes.  Exact positive co-incidence is closed before model or GPU work.
The next bridge must support one-shot facts or expose a different raw
intervention; it may not assume duplicate tuple evidence.

T29 takes the one-shot route without a semantic quotient.  Its
[title-triggered affine prefix operator](title-triggered-affine-prefix-t29-paper.md)
stores the exact transition of a dedicated input-driven recurrence, so applying
the record is provably equivalent to scanning the raw document in that memory
stream.  Its
[frozen exact CPU gate passed](title-triggered-affine-prefix-t29-stage0-decision.md),
including the BF16 codec, finite-precision bound, and unchanged-state ledger.
That result admits only a separately frozen unquantized information oracle;
quantized natural training and physical GPU work remain forbidden.

T30 is a direct structural refinement rather than a new semantic quotient.  Its
[dihedral-monomial operator](dihedral-monomial-prefix-t30-paper.md) keeps the
same executable-prefix interpretation and 220-cell document record while
adding noncommutative cross-coordinate routing inside the same 220-byte state
envelope.  It is admitted only to its frozen exact CPU gate.
That gate subsequently passed.  The first learnability screen was withdrawn
before execution because it weakened the diagonal-affine control and did not
exercise noncommutativity.  A corrected arbitrary-state theorem passed its
exhaustive preflight, after which the natural-information bridge failed on
paper: T30 moves existing features but does not infer semantic addresses or
perform keyed relational joins.  T30 is retained as a primitive and closed as
the active breakthrough direction without training.

One orthogonal escape was admitted to cheap gates: preserve raw ordered
evidence rather than identifying a semantic quotient.  The
[T27 packed-token evidence plane](packed-token-evidence-plane-t27-algebra.md)
has a fixed 55-token/220-cell code and a constructive three-SwiGLU content-read
law.  Its primitives passed, but its frozen decoded-prefix oracle failed on an
exact BF16 decision tie.  The fixed-prefix payload is closed before physical
implementation.  The digital identities may be reused only inside a new
Stage--2 candidate with its own semantic bridge and gates.

T32 instantiates that new raw-evidence candidate without claiming a semantic
quotient. Its scale-referenced base-64 layout stores the full frozen 128-token
record in 382 width-384 cells and permits direct access to each regular token.
The frozen CPU codec gate passed every exactness, BF16, order, domain, and
resource check. The separately frozen decoded full-record/local-window oracle
then passed: correct evidence reached 89.4231%/88.4615%, while question-only
and shuffled controls stayed at 46.1538%/48.0769%. T32 has therefore passed the
natural-information bridge with more than 40 points of causal gain. The active
uncertainty is no longer payload sufficiency; it is whether a fixed-budget
small graph can transport and interpret the record without sacrificing its
language path. Only reader algebra and physical microbenchmarks are admitted
before a from-zero composition run.

The subsequent T32R algebra supplied exact token-embedding expansion and a
rank-32 query-conditioned scan, but the v2 proof-first audit withdrew its H100
implementation before code or timing.  Two corrections change the admission
order.  First, because T32R reads a persistent table directly rather than
transporting it through RMSNorm, direct `uint16` IDs strictly improve the
resident-byte and decode-work ledgers over BF16 radix amplitudes.  Second, the
implementation derives candidates by exact case-folded title matching on the
raw question without reading evaluator support fields.  A later frozen audit
found only 102/104 supporting-pair matches because title-shaped predicate and
category mentions can displace an argument.  This proves raw-only candidate
generation, not autonomous route correctness, aliases, or implicit retrieval.  The natural 9B
explicit-text oracle also does not prove sufficiency of the proposed rank-32
small-model bottleneck.  T32's information result, literal candidate generator, and
reader primitives are retained, while exact-bottleneck sufficiency and the
strongest conditional-memory control return ahead of physical timing.  See
[`t32r-proof-first-capability-audit.md`](t32r-proof-first-capability-audit.md).

An attempted list-decoded knowledge syndrome then asked the base model to
propose top-`K` answers and used raw-record membership as a cheap verifier.  Its
paper audit closes the general direction: the apparent `log K` state is per
selected answer, while a record with `n` possible answers needs
`Omega(n log(K/delta))` approximate-membership bits; exact arbitrary substring
support recovers the full record entropy.  More importantly, membership cannot
distinguish two candidate strings that both occur under different entity-role
bindings, and cannot derive the current yes/no comparison answers at all.  The
extractive list-rescue lemma is retained only as a control; no implementation
or GPU run is admitted.

Counterfactual interaction spectra were then considered as an observable
semantic bridge.  Double finite differences do cancel additive context/entity
terms and recover a bilinear interaction operator under a separated generative
law, with an explicit singular-value/noise bound.  The natural claim fails:
collocation can create nonzero interaction without a fact, a true relation can
have additive model energy, exhaustive span-pair probing is quartic, and
statement/query codes are independently permutable without a cross-surface
anchor.  The spectrum is retained only as a compiler diagnostic; its
architecture and any GPU run are closed in
[`counterfactual-interaction-spectrum-paper-decision.md`](counterfactual-interaction-spectrum-paper-decision.md).
