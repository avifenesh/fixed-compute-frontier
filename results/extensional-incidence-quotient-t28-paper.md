# Extensional incidence quotient T28 — paper gate

Status: **CPU CENSUS FAIL; T28 CLOSED BEFORE MODEL OR GPU WORK**  
Date: 2026-07-31

## P0. One-sentence edge

At unchanged served checkpoint bytes, Transformer graph, BF16 precision, KV
state, and per-token matrix shapes, compile recurring raw entity--value
incidence into globally shared discrete relation records so a fixed-size model
can retain substantially more one-shot facts and compose them on held-out
queries than matched ordinary from-zero training.

The different currency is corpus-wide training-time incidence computation.  It
is charged.  It is not a serving model, teacher, parser, retrieval index, or
extra inference artifact.

## P1. Exact baseline obstruction

T21--T24 exposed a per-document gauge.  If document `d` owns an arbitrary code
`c_d` and a shared reader is fitted only on observed document/query pairs, then
any bijection of the document codes accompanied by its inverse in the fitted
reader preserves training behavior.  Unseen document/query edges are
unconstrained.

Entity renaming removes entity identity but leaves every canonicalized surface
orbit distinct.  It therefore cannot force two independently worded instances
of one relation to share a code.

T28 supplies the missing cross-instance equation.  A surface pattern is
identified by what entity--value pairs it is observed to relate, not by a free
latent coordinate.

## P2. Raw event lattice and declared meanings

Let `E` be the title inventory available in the raw corpus.  Construction is
label blind and exhaustive:

1. find exact longest nonoverlapping occurrences of corpus titles;
2. treat a document's own title as an available subject mention;
3. within each sentence, enumerate every bounded contiguous candidate value
   span and every ordered combination of one or two entity slots;
4. replace the entity and value spans by typed placeholders while retaining
   every other token and its order;
5. emit a witnessed incidence edge `(u, p)`, where `u` is the concrete
   entity--value tuple and `p` is the canonical surface pattern.

No candidate is selected semantically at this stage.  The true event, if it
exists in this family, is one member of an explicit finite lattice.  Every edge
must carry byte offsets sufficient to reproduce its exact source spans.

Define the Boolean incidence matrix

\[
B_{u,p}=1 \quad\Longleftrightarrow\quad
\text{witnessed tuple }u\text{ occurs with surface pattern }p.
\]

The blocks and meanings are:

```text
enumerate(raw title, raw sentence) -> witnessed tuple-pattern edges
quotient(pattern columns of B)      -> globally shared relation IDs
assign(entity, relation, value)     -> typed finite fact records
pack(records)                       -> radix-16 cells in the digital plane
map(query surface)                  -> relation ID
read(entity, relation ID)           -> exact value ID or absent
reason(values)                      -> ordinary model answer
```

The first four outputs have direct observable meanings.  `map(query surface)`
is the one remaining learned composition hypothesis after the raw quotient,
packing, and reading blocks pass.

## P3. Recovery theorem

### Noise-free theorem

Suppose every retained pattern `p` expresses one latent relation `r(p)`, and
each relation `r` has an extension vector

\[
z_r\in\{0,1\}^{|U|}
\]

over the same finite tuple universe `U`.  Assume complete observation

\[
B_{:,p}=z_{r(p)}
\]

and distinct extensions: `z_r != z_s` whenever `r != s`.

Then equality of observable incidence columns recovers the latent relation
partition exactly, up to a harmless permutation of relation names:

\[
p\sim p' \iff B_{:,p}=B_{:,p'}
           \iff r(p)=r(p').
\]

**Proof.** Patterns of the same relation equal the same extension vector by
construction.  Patterns of different relations have unequal extension vectors
by assumption.  Therefore the equivalence classes of equal columns are exactly
the inverse images of `r`.  Naming each class is arbitrary, but writer and
reader share that permutation.  QED.

This is the cross-surface bridge missing from entity-renaming invariance.  Two
different sentences are equated only through repeated observable action on
tuples.

### Noisy finite-sample theorem

Let each observed bit be independently flipped with probability `eta < 1/2`.
For two patterns of the same relation, the expected normalized Hamming distance
is

\[
\mu_{same}=2\eta(1-\eta).
\]

If every pair of distinct relation extensions differs on at least a `delta`
fraction of `m` shared tuple probes, then the expected different-relation
distance exceeds the same-relation distance by

\[
g=\delta(1-2\eta)^2.
\]

Thresholding at `mu_same + g/2` and applying Hoeffding's inequality gives, for
each compared pair,

\[
P(\text{wrong merge or split})
\le \exp\left[-m\delta^2(1-2\eta)^4/2\right].
\]

For `P` surface patterns, a union bound gives total recovery failure at most

\[
P^2\exp\left[-m\delta^2(1-2\eta)^4/2\right].
\]

The theorem names the exact natural-data quantities the CPU census must
measure: shared probe count, extension separation, noise, and pattern count.
No optimizer can repair a failed separation bound.

## P4. Failure boundary

The construction is not universal semantic induction.

1. **Coextension:** two distinct relations true of exactly the same observed
   tuples are information-theoretically indistinguishable.
2. **Singletons:** a one-off surface with no shared tuple probes supplies no
   cross-instance bridge.
3. **Polysemy:** one surface pattern used for several relations violates the
   single-component assumption and must remain split by additional observable
   context or be rejected.
4. **Enumeration noise:** if distractor masks have comparable support and
   separation to true events, the raw lattice does not identify facts.
5. **Open-world negatives:** absence of an incidence edge is not automatically
   a false fact.  The first census must compare positive co-incidence and may
   not silently treat every missing edge as a negative label.
6. **Unseen query language:** a natural question with no learned or observable
   path to a recovered surface class cannot be assigned a relation ID by this
   theorem.  This is the one composition hypothesis, not a proved block.
7. **Independent facts:** arbitrary nonrecurring facts still require their
   information-theoretic record bits.  T28 can organize them; it cannot compress
   random truth below entropy.

These are kill conditions, not tuning suggestions.

## P5. Constructive fixed-cost reader

Assume at most 4,096 recovered relation IDs and 65,536 value IDs.  Encode one
fact as:

- three radix-16 cells for a 12-bit relation ID;
- four radix-16 cells for a 16-bit value ID;
- one radix-16 flag cell for polarity, type, or absence convention.

One fact costs eight cells or 32 logical bits.  A 220-cell document plane holds
24 complete facts in 192 cells, leaving 28 cells for header, occupancy, or
error-detection state.

The retained T27 paired-SiLU identity implements exact real-arithmetic matching
of one 12-bit relation key across 24 fact slots with

\[
2\text{ signs}\times3\text{ nibbles}\times24=144
\]

SwiGLU channels.  Unique-slot value selection needs

\[
24\times4=96
\]

products, and the retained four-nibble token decoder needs at most 192
channels.  All are below width 1,024.  No T27 natural-payload result is
inherited; only its exact digital identities are reused.

The query mapper outputs a shared relation code.  The digital read is then
fixed and document independent; there is no per-document coordinate gauge.

## P6. Resource equation

For `N` documents, `F <= 24N` retained facts, `K <= 4096` relations, and one
65,536-entry value dictionary:

\[
\text{logical payload bits}=32F+O(N),
\]

with a hard per-document ceiling of 880 bits.  Corpus enumeration, incidence
bitsets, clustering, discarded candidates, and relation-code supervision are
training-only resources and must be reported in bytes, CPU/GPU work, and wall
time.

The served candidate may overwrite only already budgeted BF16 checkpoint
entries.  It may not add a relation table, dictionary file, memory token,
retrieval call, KV slot, longer prompt, hidden buffer, or extra operator.  The
exact write ledger and physical layer placement are deferred until the raw
census fixes `F`, `K`, and value encoding.  No model run is admitted before
that ledger exists.

Matched controls receive the same raw text, tuple witnesses, recovered cluster
IDs, value dictionary, auxiliary relation targets, training work, and reserved
capacity.  They include:

1. ordinary dense next-token training;
2. dense training plus the same compiler-derived auxiliary targets;
3. equal-capacity free document records;
4. random and frequency-matched pattern partitions;
5. the exact digital record with correct, zero, shuffled-document, and
   shuffled-relation ablations.

Because an ordinary Transformer can represent the same function, the claim is
an acquisition and organization advantage, not a larger function class.

## P7. Why the effect could be large

For a relation-bearing sentence of `L` tokenizer IDs, literal storage costs
roughly `16L` raw token-ID bits before compression.  A typed fact costs 32 bits
under the declared finite dictionaries.  At `L=20`, that is a tenfold logical
rate difference; at `L=50`, it is 25-fold.  This is only achievable when the
raw incidence quotient correctly removes repeated surface form.

The candidate can therefore cross the breakthrough bar only if the CPU census
finds all of the following simultaneously:

- enough recurring extension-separated patterns to cover a large natural fact
  fraction;
- no more than 24 retained facts per addressed document after a frozen
  priority rule;
- at most 4,096 relation classes and 65,536 values;
- a decoded structured-record oracle above the natural capability gate with
  large zero/shuffle causal drops;
- a query mapper that generalizes to held-out surface forms and entities;
- no protected language regression after equal-capacity reallocation.

A low-coverage extractor or a sub-percent language metric cannot promote T28.

## Prior-art and novelty boundary

The broad entity-pair by textual-pattern matrix is established by
[Universal Schema](https://aclanthology.org/N13-1008/), including matrix
factorization over surface relations.  Anchor/separability and multi-view
factorization provide known identifiability tools.  T28 therefore claims no
novelty for incidence relation extraction, clustering, or digital lookup in
isolation.

The unverified conjunction is narrower: exhaustive raw-only witnessed event
lattices, an extension-separated discrete quotient used as supervision from
zero, exact finite fact records compiled into already-paid Transformer weights,
and a substantial title-disjoint reasoning gain with no served-resource
increase.  A prior-art collision on that complete conjunction would close the
novelty claim but not alter the mathematical gate.

Predictive-state/Hankel methods are another observable-state route, but they do
not remove the independent-fact storage lower bound.  A corpus containing `n`
independently distinguishable entity continuations contains an `n x n`
identity submatrix in its Hankel matrix, hence has rank at least `n`; every
rank-`r<n` approximation retains spectral error one on that witness.  T28
factors repeated relation structure while leaving independent values explicit.

## Block microbench plan

No block inherits a pass from the theorem.

| block | first implementation | mandatory pass | kill result |
|---|---|---|---|
| event lattice | obvious CPU enumerator with byte-offset witnesses | exact source regeneration; no answer/support/parser/model fields; exhaustive tiny-world agreement | any hidden semantic selector or irreproducible edge |
| incidence quotient | independent bitset and sparse-edge implementations | exact planted recovery; empirical margins clear the noisy theorem; coextension/polysemy adversaries rejected | insufficient shared probes, separation, or raw coverage |
| relation targets | deterministic cluster numbering and dictionary | invariant to entity renaming and input order; held-out surface/entity split fixed | cluster IDs track document or surface identity rather than extension |
| digital pack/read | retained reference plus independent implementation | exhaustive codebook, corruption margin, zero/shuffle/duplicate controls | any collision, ambiguous slot, or width overflow |
| information oracle | decoded typed records shown to a frozen strong reader | preregistered absolute capability and at least 15-point zero/shuffle causal gaps | structured records are naturally insufficient |
| query mapper | smallest from-zero matched screen | unseen patterns and entities map to the right arbitrary relation permutation | training-template memorization or query-only shortcut |
| physical plane | same-graph H100 microbench only after all above pass | numerical agreement and full service-envelope noninferiority | added graph/state/bytes or latency/quality regression |

## Paper decision

T28 passes only to a CPU raw-evidence census because it now has:

- a raw-computable event lattice rather than privileged relation spans;
- an observable cross-instance equivalence relation;
- exact and noisy recovery theorems;
- explicit counterexamples;
- a constructive fixed-width digital reader;
- one named composition hypothesis: unseen natural query to recovered relation
  code.

It does **not** admit a model download, representation training, H100 work,
physical integration, or production claim.  The CPU census must be frozen
before looking at natural answers and must close the branch if recurrence,
separation, coverage, or the 24-fact budget is inadequate.

## Observed Stage-0 decision

The frozen census found only 3.7398% multiply witnessed tuples, 25.7796%
bridged documents, and zero edges between recurring surface-pattern nodes.
There were no qualifying components.  T28 is therefore closed under its kill
boundary.  See
[`extensional-incidence-quotient-t28-stage0-decision.md`](extensional-incidence-quotient-t28-stage0-decision.md).
