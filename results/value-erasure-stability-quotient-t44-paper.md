# T44 value-erasure stability quotient — paper packet

Date: 2026-08-01  
Status: **WITHDRAWN AFTER GOAL CORRECTION; NARROW SEMANTIC-COMPILER OBJECT WAS NOT THE RESEARCH GOAL; ZERO RUNS**

> This packet is retained as a scoped mathematical idea, not as the active
> direction. The owner clarified that raw-prose compilation is only one
> possible mechanism. The actual goal is substantial progress toward real
> intelligence: better learning, continual learning, reasoning, abstraction,
> memory, self-correction, or another model-level capability gain at a
> defensible complete cost. No adverse empirical result is claimed for T44.

## 0. Claim and the exact change of direction

The historical target for this withdrawn packet was a from-zero model that
compiled raw prose into a digital plane. That object no longer defines the
active research mission.

T44 does not invent another record reader. T32 already showed that full raw
records contain enough natural information, and T33 showed that a sentence
index preserves that information more cheaply and completely than an
ambiguous exact one-hole grammar. T35 then proved that a sidecar plus an
ordinary reader is absorbed by an identical-memory control.

T44 therefore tests one narrower `E2` escape from that closure:

> In a declared noisy slot-source family, deleting the true value is the only
> short deletion that makes independently instantiated relation surfaces
> stable. Recover that value-erasure quotient from raw strings, store only the
> task-sufficient `(document, relation, value)` state, and spend the nuisance
> bytes it removes on more facts rather than on another reader.

The possible edge is a **sufficient-state rate advantage**, not a new function
class. A generic conditional memory given the same recovered records must tie
the local read and is a mandatory control.

## 1. Raw observable

For each titled raw document, replace exact nonoverlapping occurrences of its
title by the typed symbol `SUBJ`. Split the remaining raw text into reversible
sentence byte spans. For every contiguous candidate token span `u=[a,b)`,
form

\[
s(x,u)=\operatorname{replace}(x[u],\textsf{HOLE}).
\]

The compiler records `(document, sentence offsets, u, s(x,u))`. It does not
choose a value, relation, entity type, answer, support sentence, parser output,
embedding, or model score.

For a fixed edit radius `rho`, connect candidates from different documents
only when their skeletons have token edit distance at most `2rho`. An
admissible cluster must have:

1. at least `m` distinct documents;
2. at most one candidate from each raw sentence;
3. a witnessed center `tau` within distance `rho` of every member;
4. no member within `rho` of a second admitted center;
5. a unique shortest deleted span in each covered sentence; and
6. positive complete two-part code saving after the center, member offsets,
   deleted token payloads, lengths, and exceptions are charged.

The relation code is a deterministic ordering of admitted centers. The value
is the exact deleted token span. Every output therefore has a replayable raw
witness; the natural uncertainty is whether the stable hole is semantically
useful.

This statistic differs from T28 and T33:

- it does not require the same entity--value tuple to recur;
- it permits bounded surface variation instead of exact anchors; and
- it asks whether **erasing one varying span collapses cross-document surface
  variation**, rather than retaining every recurring interval.

## 2. S0 — separated noisy slot source

Let `Sigma` be a finite token alphabet. Each relation `r in [K]` has a
canonical skeleton

\[
\tau_r\in(\Sigma\cup\{\textsf{SUBJ},\textsf{HOLE}\})^*.
\]

An instance chooses a fresh document subject `e`, a value string `v`, and
nuisance noise `n`, fills the two typed slots, and applies at most `rho`
slot-respecting token edits outside the value:

\[
x=G(\tau_r,e,v,n).
\]

The family has five structural conditions.

1. **Observable subject.** `e` is the exact titled-container identity and its
   raw occurrence is recoverable without a semantic parser.
2. **Slot-respecting noise.** Replacing the complete generated value span by
   `HOLE` produces a skeleton within edit distance `rho` of `tau_r`.
3. **Relation separation.** For `r != r'`,
   `d_edit(tau_r,tau_r') > 4rho`.
4. **Value scattering.** An unsafe deletion—one that fails to cover the full
   value—leaves at least `lambda` independently generated value symbols in the
   skeleton. Those symbols have conditional min-entropy at least
   `lambda log_2 A` for an effective alphabet size `A`.
5. **Sufficient task law.** Future admitted questions depend on the raw
   sentence only through its subject, relation, and value:

   \[
   Y\perp X\mid(Q,E,R,V).
   \]

Conditions 1--4 are source assumptions. Condition 5 declares the task family;
without it, the general task-agnostic lossy-compilation no-go applies.

The natural corpus is not assumed to satisfy this family. The point of the
theorem is to derive a raw statistic and a falsifiable margin, not to rename
natural semantics as an assumption.

## 3. S1 — recovery theorem

### Theorem 1: exact cluster recovery under deterministic scattering

Assume conditions 1--3. Strengthen condition 4 for a finite corpus as follows:
every pair of unsafe candidates belonging to different instances has skeleton
distance greater than `2rho`, while every deletion that covers a complete
value but also removes fixed context is strictly longer than the exact value
deletion.

Then the admissible complete-link edit clusters recover every relation
partition with at least `m` instances, up to permutation of relation names,
and the unique-shortest rule recovers every exact value span.

**Proof.** Correct value deletion leaves every instance within `rho` of its
relation center, so any two correct skeletons are within `2rho` and the center
witness exists. Relation separation and the triangle inequality imply that
correct skeletons from different relations are farther than `2rho`, so they
cannot merge. Deterministic scattering prevents every unsafe candidate from
joining another-instance cluster. A deletion that covers the value plus fixed
context can form the same relation cluster, but is longer than the exact value
deletion; the unique-shortest rule removes it. Thus each admitted cluster is
exactly one relation's value deletions, modulo the arbitrary numeric name of
the center. QED.

This does not assume that the true grammar is the unique MDL minimizer. It
derives recovery from observable edit separation, value variation, and a
typed title anchor.

### Theorem 2: finite false-cluster bound

Let an unsafe skeleton retain an aligned `lambda`-symbol leakage string whose
symbols are independent and uniform over an alphabet of size `A`. Let

\[
V_A(\lambda,t)=\sum_{j=0}^{t}{\lambda\choose j}(A-1)^j
\]

be the Hamming-ball volume. For one fixed set of `m` unsafe candidates, the
probability that all leakage strings lie within distance `2rho` of the first is
at most

\[
\beta^{m-1},\qquad
\beta={V_A(\lambda,2\rho)\over A^\lambda}.
\]

If the exhaustive lattice contains at most `P` unsafe candidates, a union
bound gives

\[
P(\text{some size-}m\text{ unsafe star})
\le {P\choose m}\beta^{m-1}.
\]

**Proof.** Conditional on the first leakage string, each other independent
string lies in its radius-`2rho` Hamming ball with probability `beta`.
Multiply the `m-1` probabilities and union-bound over candidate subsets. QED.

The bound is deliberately conservative: edit alignment, nonuniform values,
and dependent boilerplate can make it vacuous. That is a kill result, not a
reason to tune `rho` after inspecting answers.

### Plain witness

Raw instances such as

```text
SUBJ was released in 1997 .
SUBJ was released in 2004 .
SUBJ was released during 2011 .
```

can share a small-radius center after erasing the year. The years need never
repeat and the entity--value tuples are one-shot. Deleting `released` instead
leaves the varying years in the skeleton, so value scattering prevents a
stable cross-document cluster in the theorem family.

## 4. S2 — exact counterexamples

T44 must abstain in each of these worlds.

1. **Co-surface relations.** `SUBJ joined HOLE` denotes a band in one corpus
   region and an employer in another. Raw edit stability merges them unless an
   additional observable separates the senses.
2. **Constant value.** If every instance has the same value, deleting the
   relation words or deleting the value can be equally stable; scattering is
   absent.
3. **Multiple facts in one stable frame.** `SUBJ was born in V1 in V2` creates
   two plausible holes unless the source supplies another separation.
4. **Boilerplate hole.** A repeated navigation or citation frame can have a
   stable variable span without expressing task-relevant knowledge.
5. **Paraphrase distance.** Semantically identical surfaces outside the
   radius become different codes. The theorem provides no synonym oracle.
6. **Question disconnect.** Declarative clusters and interrogative strings can
   be independently permuted if raw training provides no bridge between them.

These are semantic boundaries, not implementation corner cases.

## 5. S3 — query boundary

The recovered relation permutation is harmless for writes and raw cloze reads:
delete the witnessed value from an admitted statement and the same cluster ID
is obtained exactly.

Natural questions are not generated by the declared edit channel. T44 makes
no theorem that `When was X released?` lies near `X was released in HOLE`.
Instead, after the raw compiler is frozen, its exact statement-to-code pairs
may create a from-zero auxiliary task:

```text
masked raw statement -> frozen relation code
raw title + relation code -> exact value span
```

Transfer from ordinary language pretraining plus this raw-derived task to an
unseen natural question is the **single learned composition hypothesis**. It
must be isolated in a held-out query-to-code test. No model run is allowed
until the decoded records themselves pass the natural information oracle.

## 6. S4 — information and resource separation

Let `F=(E,R,V)` be the typed fact state and let `N` be all nuisance wording,
formatting, and unrelated sentence content. Under condition 5, `F` is
sufficient for the admitted task. If `X` is reversibly generated from `(F,N)`,
any lossless raw representation needs at least

\[
H(X)=H(F)+H(N\mid F)
\]

bits on average, while an entropy-coded sufficient quotient approaches
`H(F)`. The available rate edge is exactly `H(N|F)`; T44 has no advantage when
that quantity is smaller than relation IDs, indexes, lengths, integrity bits,
and alignment.

For a concrete tokenizer-ID layout with `K<=65,535`, store each fact as:

```text
relation_id : uint16
value_len   : uint8 or escaped uint16
value_ids   : uint16[value_len]
```

Document offsets use a separate monotone index. The exact serving bytes are

\[
B_{T44}=2F+B_{length}+2\sum_{j=1}^{F}|v_j|+B_{doc-index}
          +B_{integrity}+B_{alignment}+B_{query-interface}.
\]

Compiler candidates, edit-distance indexes, centers, rejected spans, and
witnesses are training-only state and are deleted at export, but their bytes,
work, energy, and wall time remain reported as the alternative currency.

The first comparison is not against an unindexed token array. It is against:

- T32/direct `uint16` raw records;
- the 48.5-KiB sentence index from T33;
- a compressed full-text self-index;
- a generic conditional memory given the same T44 records; and
- the best byte-matched vector memory.

The generic memory must tie exact local outputs. T44 can advance only if its
raw-derived quotient is naturally sufficient and its packed state buys a
strict rate/coverage point; no function-class superiority is claimed.

## 7. S5 — end-effect and breakthrough gate

The current frozen reference numbers are:

```text
question-only / dense reference     56.7308%
full raw-record information oracle  89.4231%
material target                     80.0000%
```

For a typed-path selection event `S`, typed correctness `T`, and baseline
correctness `B`, the exact gain is

\[
A_{cand}-A_0
=P(S\cap T\cap\neg B)-P(S\cap\neg T\cap B).
\]

Reaching 80% requires the repair-minus-harm term to be at least `0.232692`.
Global rule coverage, compiler purity, or `q p^2` alone does not establish
that overlap.

The decoded-information gate is therefore joint:

1. at least 80% title-disjoint natural accuracy;
2. no more than one absolute point below the full-record oracle under the same
   reader and route;
3. at least 15 points correct-versus-shuffled record causality;
4. measured repair-minus-harm at least `+23.2692` points over the frozen dense
   reference;
5. at most 50% of the complete resident bytes of the matched raw-record plus
   sentence-index control; and
6. a fixed-byte population test where the quotient's additional facts improve
   held-out knowledge, rather than merely leaving unused bytes.

If two independent fact reads are required, joint availability is measured
directly. Independence may be reported as a diagnostic but cannot replace the
joint repair/harm count.

These gates make the potential gain explicit: **roughly raw-record oracle
quality at half the bytes, then use the freed half for additional knowledge at
the same serving allocation.** A low-coverage extractor or a small NLL polish
cannot pass.

## 8. Strongest controls and attribution

Every eventual comparison receives the same raw corpus, tokenizer, derived
statement-to-code examples, training-token budget, and declared compiler-work
allowance.

1. unchanged dense/MoE frontier baseline;
2. direct raw-token plane plus the T33 sentence index;
3. compressed self-index plus the strongest legal reader;
4. exact T33 rules and the complete ambiguous T33 span lattice;
5. random and frequency-matched edit clusters;
6. T44 records with correct, zero, shuffled-document, shuffled-relation, and
   wrong-value conditions;
7. generic packed conditional memory containing the identical T44 records and
   query interface; and
8. byte-matched learned vector memory/Engram/Lngram-style controls.

Control 7 must tie T44's local function. Any production claim is relative to
the current frontier systems at the complete resource point, not relative to a
copy of the same method with a different name.

## 9. Block cards and admission order

| block | proof or microbench | fatal result |
|---|---|---|
| edit metric | two independent exact implementations; exhaustive tiny strings | any distance/center disagreement |
| exhaustive erasure lattice | exact byte/token replay and reversed-order invariance | hidden selector, missing span, or irreproducible witness |
| theorem worlds | exhaustive relation, noise, over-deletion, co-surface, constant-value, and multi-fact worlds | failure inside the theorem family or confidence on a declared ambiguity |
| natural raw census | frozen `rho,m,length` and code before evaluator access | no large unambiguous stable-hole population, vacuous collision bound, or byte rate above 50% |
| decoded information | strongest fixed reader over decoded facts and all causal controls | below 80%, more than one point below full raw records, or repair-minus-harm below 23.2692 points |
| query mapper | from-zero raw-derived examples; unseen titles and question surfaces | relation-code transfer below the end-effect requirement |
| physical plane | only after all semantic gates; full same-cost graph | any byte, state, work, p95 latency, or protected-quality regression |
| from-zero model | multi-seed candidate and all matched controls | no large natural gain or failure to use extra fixed-budget fact capacity |

No CPU census is admitted by this draft. An independent paper audit must first
decide whether the recovery theorem, natural falsifier, control boundary, and
effect path are valid. A pass would authorize only tiny theorem-world tests and
one frozen CPU raw census. It would not authorize a model download or GPU.

## 10. Prior boundary checked on 2026-08-01

- Unsupervised relation extraction, slot-schema induction, grammar induction,
  MDL template learning, and open information extraction are established
  fields; T44 claims none of those names as new.
- T33 already tested exact reversible one-hole rules and found the complete
  ambiguous grammar less compact and less complete than a sentence index.
  T44 lives or dies on the new stable-erasure statistic and the stricter typed
  sufficient-state rate gate.
- T28's exact co-incidence quotient required duplicate entity--value facts.
  T44 requires repeated relation mechanisms but permits one-shot facts.
- T26's tensor theorem began after privileged event selection. T44 enumerates
  every deletion and includes unsafe-deletion separation in its source theorem.
- Hidden-schema and unsupervised-RE systems commonly use pretrained encoders,
  parsers, contrastive objectives, or external schemas. T44's first census
  permits none of them.
- Current explicit-memory systems remain mandatory end controls. Even a valid
  compiler theorem does not establish a smarter production model.

## Paper decision

T44 is neither admitted nor rejected by its authoring pass. It supplies the
missing object requested after T43: a raw-observable relation/value statistic,
a constructive recovery theorem that does not assume an MDL optimum, an exact
counterexample boundary, an explicit query gap, a sufficient-state rate law,
and a repair-minus-harm end gate.

The next action is independent adversarial review. No local or rented compute
is justified before that review.
