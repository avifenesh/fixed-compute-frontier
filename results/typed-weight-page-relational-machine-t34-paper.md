# Typed weight-page relational machine T34 — proof-first paper

Date: 2026-07-31  
Status: **PAPER FAIL AFTER INDEPENDENT AUDIT; NO IMPLEMENTATION OR GPU RUN**

## Claim in plain language

A Transformer currently interprets nearly every persistent byte as a real
number that must participate in a matrix multiplication.  T34 gives a fixed
weight page a second, typed interpretation:

- a **dense page** is an ordinary BF16 FFN tile;
- a **record page** is a packed token/span table;
- a **relation page** is a packed finite relation;
- a **program page** is a small exact interpreter table.

The complete model has the same persistent-byte ceiling.  A token activates
either the dense interpretation or one typed interpretation, and the active
byte/work/latency ceiling may not exceed the dense page it replaces.  The
training-only compiler may spend global corpus passes and discrete search, but
is deleted at export and its cost is charged.

This is not yet a claim that natural prose can be compiled correctly.  It is a
new resource hypothesis:

> use **page type** rather than more matrix width as the scaling variable, so
> exact knowledge and exact finite reasoning can use their native algebra
> while uncertain language remains in the real-valued network.

The target is not a one-percent kernel gain.  On the current two-record
knowledge witness, T34 must move held-out accuracy from the 56.7308% dense
reference to at least 80% without increasing the served byte, KV, token, or
latency ceilings and without regressing protected language quality.

## Why this is a different object

MoE gives every expert the same algebra: usually two or three dense matrices.
Engram and memory layers add conditional vector lookup.  T34 instead treats a
weight allocation as a tagged union of executable representations:

\[
P_j\in
\{\textsf{DenseBF16},\textsf{Token16},\textsf{RelationBits},
  \textsf{FiniteProgram}\}.
\]

For a token state `h`, a router produces a page type and address.  A dense page
contributes the usual FFN tile.  A typed page returns an exact token pointer,
typed value, or program result.  The result is embedded through the model's
existing token/value basis and re-enters the residual stream.

The all-dense allocation is a member of the architecture family.  At a fixed
deployment point, converting pages is a real trade: those bytes cease to be
dense weights.  T34 wins only if the typed pages buy more capability than the
lost dense width.

## Minimal admitted witness

The first witness is deliberately smaller than a general virtual machine.
Let

\[
T:E\times R\to V
\]

be a finite partial relation table.  A query is

\[
q=(e_1,e_2,r,c),
\]

where `c` is one of `EQ`, `NE`, `LT`, or `GT`.  The exact answer is

\[
c(T(e_1,r),T(e_2,r)).
\]

This covers the current two-document comparison shape without claiming graph
reachability, arbitrary code execution, or open-domain theorem proving.

The typed execution is

```text
v1 = LOOKUP(e1, r)
v2 = LOOKUP(e2, r)
z  = CMP[c](v1, v2)
```

`LOOKUP` is a packed associative-array read.  `CMP` is an exact integer,
ordinal, equality, or token-identity operation.  Neither operation requires a
dense Q/K/V or FFN projection once the addresses and opcode are known.

## Local theorem 1 — exact execution and error factorization

Assume:

1. both required facts are present and each is correctly compiled with
   probability `p`;
2. the query parser emits the correct two entities, relation, and comparator
   with probability `q`;
3. the typed interpreter is exact on valid records;
4. the parser event and two fact-compilation events are mutually independent
   for the optimistic planning model.

Then the typed path is correct with probability

\[
q p^2.
\]

The interpreter contributes no additional semantic error.  Without any
independence assumption, if all three marginal probabilities are measured on
the same population, the valid Bonferroni lower bound is instead

\[
\max(0,q+2p-2)
\]

so both quantities must be reported.

These are path-availability probabilities, not end-task accuracy.  To turn an
available fraction `s` into the optimistic mixture

\[
A_{cand}=A_0+s(A_{oracle}-A_0),
\]

the system must detect every unavailable or ambiguous typed path and fall back
to the baseline with no regression.  Without safe abstention, `q p^2` is only
an upper bound on the fraction of requests that may benefit, and wrong typed
paths must be charged separately.

On the current slice, the explicit-record oracle supplies at most

\[
89.4231-56.7308=32.6923
\]

accuracy points.  Reaching 80% requires recovering

\[
\eta=\frac{80-56.7308}{32.6923}=0.7118
\]

of that gap.  Under the optimistic independent model, exact comparison, and
perfect non-regressing fallback,

\[
q p^2\ge 0.7118.
\]

For `q=0.95`, this requires `p >= 0.8656`; for `q=0.90`, it requires
`p >= 0.8893`.  Under the conservative dependence-free bound and `q=0.95`, it
requires `p >= 0.8809`.  These are pre-run extraction gates, not fitted
thresholds.

## Local theorem 2 — information and packing boundary

For an arbitrary total table with `|E|=N`, `|R|=K`, and `|V|=M`, exact storage
requires at least

\[
NK\log_2 M
\]

bits by counting the `M^(NK)` possible tables.  A packed value array reaches
this payload bound up to indexing, missing-value, alignment, and error-detection
overhead.

T8 already reached the logical value bound using one robust 16-level BF16
scalar per fact: four logical bits in two physical bytes, or two logical bits
per resident byte.  A raw typed page can store eight payload bits per resident
byte.  On the same four-bit-value witness it therefore offers at most a **4x
payload-density ceiling**, not 16x, before indexes and alignment.  Larger claims
are forbidden.

This is a physical representation gain, not a universal neural-capacity lower
bound.  An arbitrary BF16 weight tensor contains sixteen physical bits per
scalar and could encode a digital table if given an appropriate conditional
decoder.  The comparison is therefore against the complete read operator, not
against parameter count alone.

## Local theorem 3 — exact finite comparison

Let values use a canonical `b`-bit encoding preserving equality and, for
ordered values, numeric order.  Equality is the zero test of bitwise XOR;
unsigned less-than is a fixed `O(b)` Boolean circuit, or `O(ceil(b/w))`
machine-word operations with a native comparison on `w`-bit words.

Thus `CMP` is exact for all `2^b` values and its learned parameter count is
zero.  A neural decoder is needed only to map the exact result back into the
language residual stream.  Exhaustive enumeration over small `b` and an
independent arbitrary-precision reference can verify the implementation.

This theorem does not imply that raw prose exposes the right typed value or
that a natural question exposes the right opcode.

## Local theorem 4 — the scoped dense-work separation

Consider the restricted dense key-value implementation that assigns one
independent FFN channel to each of `F` arbitrary facts and evaluates an ordinary
dense FFN for every query.  Its gate/up/down work and weight traffic scale as
`Omega(Fd)` for residual width `d`, even when only one fact is needed.  A packed
direct-address table uses `Theta(F log M)` resident bits and reads one value in
`O(1)` expected probes or `O(log F)` ordered probes.

This is a separation from an always-evaluated dense key-value bank.  It is **not
a lower bound against all Transformers**, and it does not separate T34 from a
general conditional-memory layer granted the same keys, packed values, and
comparator.  That strongest control can implement the same local function.

Therefore the research claim is narrower:

> Can a from-zero, raw-derived training process organize fixed model bytes into
> semantically addressed typed records more reliably than ordinary dense,
> MoE, n-gram memory, and free learned conditional-memory controls at the same
> complete serving cost?

If a free conditional-memory control matches T34, the architecture is absorbed
and the branch closes.

## The semantic compiler

The compiler may use only raw documents, document titles/containers already
present in the corpus, tokenizer output, and the same from-zero objectives
available to every legal control.  It may not use support spans, evaluator
relations, a parser, a pretrained embedding model, or a stronger pretrained
teacher.

The compiler is a latent-program learner with a finite declared language:

```text
document title -> subject candidate
sentence index -> evidence sentence
raw span lattice -> value candidate
surface context -> latent relation candidate
raw token/type tests -> value type candidate
```

It selects a graph and decoder minimizing a complete two-part code:

\[
L(G)+L(D\mid G),
\]

subject to exact source reconstruction, entity-renaming equivariance, and
counterfactual value-swap consistency.  The whole candidate lattice remains
visible to the evaluator; the compiler may abstain rather than force a fact.

### Restricted recovery theorem

Let the corpus be generated by a finite prefix-free relational grammar with:

1. raw-observable document-subject anchors;
2. one factual production per admitted sentence;
3. uniquely decodable value boundaries;
4. a finite relation-template dictionary;
5. relation templates reused across at least two independently named subjects;
6. no noise production with an equal or shorter complete two-part code;
7. a positive code-length gap `gamma` between the true grammar and every
   competing grammar in the declared finite candidate class.

Then exact minimization of `L(G)+L(D|G)` recovers the true typed records and
relation templates up to a global permutation of relation IDs.

**Proof.**  Prefix-freeness and unique decodability make the true parse
feasible and exact.  By assumption 6 no noise production ties or beats a true
production locally.  By assumption 7 every competing complete grammar has
description length at least `gamma` larger.  Therefore the true grammar is the
unique minimizer, modulo renaming latent relation symbols, which leaves both
writer and reader unchanged.  QED.

This theorem is intentionally conditional.  It does not claim that natural
prose satisfies unique boundaries, finite templates, or a positive MDL gap.
T33 is evidence that a local one-hole grammar does not: retaining its complete
surface lattice produced high coverage but about 4.49 competing edges per
covered support sentence, and a sentence index dominated the representation.
T34 changes the downstream object and global objective, but it must measure a
real natural code gap rather than assume one.

### Counterexample theorem

If two candidate parses reconstruct every observed string with equal total
code length but assign different relation/value records, raw data cannot choose
between them.  Likewise, if declarative relation forms and question forms lie
in disconnected raw components, no raw-only learner can identify their common
relation label.

Consequently the compiler must abstain on tied parses, and question alignment
must be either:

- identified by raw-observable cross-form evidence; or
- the single explicitly isolated learned composition hypothesis.

It may not be hidden inside an opaque end-to-end score.

## Why an offline compiler is not automatically a stronger serving model

The compiler receives a different resource: repeated non-causal passes over
the whole training corpus and discrete global search.  Serving receives only
the compiled pages and a bounded parser.  Using the compiler itself for every
query would repeat corpus-scale search and violate the latency/work contract.

This is the same valid amortization pattern as building a database index or
compiling a logical theory, but it is not free.  Let compiler cost be `C_train`
and serving save `Delta C` per request.  A total-cost claim requires an explicit
break-even volume

\[
Q_{break}=C_{train}/\Delta C.
\]

The primary T34 claim is a serving-frontier claim; training cost is reported
separately and never erased.

## Complete block graph

```text
raw prose
  -> sentence/title/span lattice
  -> latent typed-record compiler
  -> packed typed weight pages

natural query
  -> entity/relation/opcode parser
  -> two exact page lookups
  -> exact typed comparison
  -> shared result embedding
  -> ordinary language decoder
```

The page allocator and interpreter are explainable.  The natural compiler and
query parser are the only semantic learners.  They must be tested separately
so an end-to-end run does not ask two miracles at once.

## Strongest controls

Every control receives the same raw corpus, titles, tokenization, auxiliary
raw-derived views, training token count, compiler-work allowance, and total
served byte/work ceilings.

1. full-width dense Transformer;
2. byte-matched narrower Transformer with unused slack returned as trainable
   dense width;
3. sparse MoE under the same resident and active ceilings;
4. Engram-style exact n-gram lookup;
5. free learned conditional memory with the same page bytes and router;
6. direct `uint16` raw records plus sentence index and the strongest legal
   learned reader;
7. T34 with correct, zero, shuffled, wrong-relation, and wrong-value pages.

Control 5 is fatal: if it matches the candidate, typed relational structure
has not produced a capability edge beyond generic conditional memory.

## Fixed resource equations

For residual width `d`, FFN width `m`, `L` layers, and BF16 weights, the full
SwiGLU FFN store is

\[
B_{ffn}=6Ldm\text{ bytes}.
\]

Removing `delta_m` FFN channels from every layer frees

\[
B_{page}=6Ld\,\delta_m\text{ bytes}
\]

and removes

\[
C_{dense}=3Ld\,\delta_m
\]

dense multiply-accumulates per processed token.

At `d=384`, `L=10`, one removed channel per layer funds 23,040 bytes.  The
existing direct typed-record construction costs 683,148 bytes, so it fits in
30 channels per layer with 8,052 bytes remaining.  That reduces FFN width from
1,024 to 994, or 2.93%.  A T34 relation sidecar must fit inside the remaining
slack or declare further removed width before seeing quality results.

The physical gate additionally charges:

- router weights/MACs;
- page address and bounds checks;
- bytes read from HBM/L2;
- typed scratch and alignment;
- result embedding/injection;
- branch divergence and synchronization;
- compile-time metadata resident at serving;
- p50 and p95 prefill/decode latency.

Nominally skipped GEMM MACs are not a latency proof.

## Block cards and microbench order

No end-to-end model or target-GPU run is admitted until these cards exist and
pass in order.

### B1 — typed comparator

- **Type:** two `b`-bit values plus opcode -> one exact result.
- **Invariant:** equals an independent big-integer reference.
- **Microbench:** exhaustive all pairs for small `b`; random and boundary pairs
  at deployment width; invalid-type abstention.
- **Kill:** any semantic mismatch or resource overflow.

### B2 — packed relation page

- **Type:** finite partial table -> byte image; key -> value/absent.
- **Invariant:** exact round trip, deterministic layout, no alias.
- **Microbench:** exhaustive small tables, random large tables, collision and
  absent-key adversaries, two independent implementations.
- **Kill:** less dense than the strongest direct typed control or more bytes
  than the frozen page allocation.

### B3 — restricted grammar compiler

- **Type:** raw synthetic corpus -> relation/value records plus witnesses.
- **Invariant:** recovers the grammar up to relation permutation under the
  theorem assumptions; abstains on the counterexample family.
- **Microbench:** exhaustive small identifiable worlds, equal-code ambiguous
  worlds, entity renaming, value swaps, noise, and template collisions.
- **Kill:** confident output on an ambiguous world or failure under the exact
  theorem family.

### B4 — frozen natural compiler audit

- **Type:** raw development corpus only -> frozen typed records.
- **Invariant:** every record includes the raw spans and code-length evidence
  that caused it.
- **Microbench:** only after freezing, open support metadata for scoring;
  report per-record `p`, two-record joint coverage, ambiguity, abstention, and
  shuffled-corpus controls.
- **Kill:** cannot support `q p^2 >= 0.7118` even with `q=1`, or is dominated by
  raw record plus sentence index.

### B5 — query parser

- **Type:** raw question -> entity pair, relation code, comparator, confidence.
- **Invariant:** no support field or privileged record handle.
- **Microbench:** held-out titles, title-shaped distractors, relation
  paraphrases, reversed comparator, absent relation, entity permutation.
- **Kill:** measured joint `q,p` cannot cross the effect-size inequality, or
  shuffled relation IDs do not cause the predicted collapse.

### B6 — exact composition oracle

- **Type:** frozen parser output plus frozen pages -> answer code.
- **Invariant:** equals direct execution of the decoded records.
- **Microbench:** correct/zero/shuffled/wrong pages and strongest free-memory
  control; no learned language decoder yet.
- **Kill:** less than 80% decoded accuracy or less than a ten-point advantage
  over the strongest byte/work-matched control.

### B7 — physical page operator

- **Type:** target-shape pages and parser state -> injected residual.
- **Invariant:** bit-exact typed result and bounded numerical injection error.
- **Microbench:** H100 only after B1--B6 pass; complete dense-replacement graph,
  not an isolated cache hit.
- **Kill:** resident, active-work, workspace, or p95 latency exceeds the dense
  page it replaces.

### B8 — from-zero composition

- **Single remaining hypothesis:** the ordinary language path can learn to use
  the already validated exact result injection without losing protected
  language quality.
- **Run:** multi-seed matched candidate/control training.
- **Kill:** target accuracy below 80%, less than ten points over the strongest
  control, any protected regression outside its frozen interval, or any served
  resource regression.

## Prior-art boundary

- Engram proves that deterministic conditional lookup can beat iso-parameter,
  iso-FLOP MoE and can free backbone capacity, but its key is a local n-gram
  and its value is a learned vector.
- Memory Grafting uses a stronger pretrained model offline to construct latent
  n-gram values.  T34 forbids that teacher in the claimed autonomous path.
- UltraMemV2 and product-key memory scale sparse learned vector tables.  They
  are mandatory controls, not evidence that a typed relation/compiler works.
- Knowledge compilation establishes the general offline-compile/fast-query
  resource exchange.  T34's open question is whether raw language exposes a
  recoverable typed theory inside a fixed LLM byte and latency budget.
- T8 already established exact logical payloads in unchanged Transformer
  weights when given an explicit schema.  T34 is valuable only if it removes
  that schema assumption and survives the stronger typed-memory controls.

The individual ingredients are not claimed as new.  The possible novelty seam
is the conjunction of typed weight-page allocation, raw/from-zero latent
compilation, exact native finite operations, and strict complete serving
substitution for dense FFN pages.

## Original paper decision

T34 has a real alternative resource, an exact local operator, a scoped
separating witness, a quantitative breakthrough path, and fatal controls.  It
does **not** yet pass to implementation because the restricted compiler theorem
relies on a positive natural MDL gap that T33 warns may be absent.

The next action is paper-only: adversarially audit the recovery theorem and
derive a raw-observable statistic capable of certifying the natural code gap.
No CPU corpus census, model training, or H100 microbenchmark is admitted until
that audit leaves only one empirical semantic hypothesis.

## Superseding audit disposition

The independent audit rejected T34 before implementation.  Its decisive
findings are recorded in
[`typed-weight-page-relational-machine-t34-paper-audit.md`](typed-weight-page-relational-machine-t34-paper-audit.md).
The restricted MDL argument assumes the unique code-length gap it claims to
prove, global compiler/parser coverage does not imply repair of baseline
errors, and the declared free conditional-memory control contains the complete
local typed function.  Typed pages, packed records, and exact comparison remain
reusable primitives; T34 is closed as a breakthrough candidate.
