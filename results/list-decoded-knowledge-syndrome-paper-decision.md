# List-decoded knowledge syndrome — paper decision

Status: **REJECT AS A GENERAL KNOWLEDGE PLANE; RETAIN EXTRACTIVE LEMMA**  
Date: 2026-08-01

## Proposed edge

Let the language model supply a ranked list of `K` candidate answers.  Store a
short universal hash or approximate-membership structure over the raw record,
then return the first candidate accepted by that structure.  The intended
claim was that the model supplies semantic side information while the record
needs only about `log2(K/delta)` verification bits.

## The valid narrow lemma

Let the true answer `A` have rank `j <= K`.  Suppose `A` belongs to a stored set
`S`, every higher-ranked candidate is outside `S`, the filter has no false
negatives, and each fixed nonmember has false-positive probability at most
`epsilon`, independently of the candidate list and secret hash choice.  First
accepted-candidate decoding then returns `A` with probability at least

\[
1-(j-1)\epsilon.
\]

If the true answer appears in the top `K` with probability `p_K`, aggregate
accuracy is therefore at least

\[
p_K\bigl(1-(K-1)\epsilon\bigr).
\]

This is a useful extractive list-rescue control.  It is not a compact encoding
of a general query-to-answer function.

## Why the `log K` storage claim fails

The short fingerprint verifies one already selected answer against one fixed,
nonadaptive list.  A record containing `n` possible answers needs an
approximate-membership structure for all `n` items.  A one-sided structure
requires

\[
m \ge n\log_2(1/\epsilon)-O(n)
\]

bits.  Setting `epsilon <= delta/(K-1)` gives

\[
m \ge n\log_2((K-1)/\delta)-O(n),
\]

not `log K` total bits.  A standard Bloom filter uses approximately
`1.4427 n log2((K-1)/delta)` bits.

For exact membership in an arbitrary `n`-element subset of a universe of size
`U`, the state needs at least

\[
\log_2 {U \choose n}
\]

bits.  Exact arbitrary-substring membership for an `L`-token record over a
vocabulary of size `V` needs at least `L log2 V` bits in the worst case: the
single length-`L` substring identifies the entire record.  A public `b`-bit
hash also has an adversarial collision whenever `b < log2 U`.

## Fatal semantic counterexample

Consider the record:

```text
Alice was born in Paris. Bob was born in Rome.
```

For the question "Where was Alice born?", let the language model rank
`[Rome, Paris]`.  Both strings are exact members of the record.  First-positive
membership returns `Rome` with certainty even with a collision-free index.
The missing information is the relation binding `(Alice, born-in, Paris)`, not
answer-string membership.

The active natural witness is harder still: its answers are `yes` and `no`,
which are conclusions of a two-record comparison rather than evidence
substrings.  A substring filter cannot execute that comparison.  Hashing
query-relation-answer propositions would require the writer to have already
performed the semantic extraction the proposal was meant to avoid.

## Resource and control audit

- Single-token top-`K` can reuse already-computed logits, but multi-token lists
  require additional decoding work.
- Hash probes, index bytes, and memory traffic remain served costs.
- Exact tries, suffix-constrained decoding, pointer/copy readers, and generic
  proposal-and-filter rerankers are stronger matched controls; Bloom storage
  only adds false positives.
- Slepian-Wolf/list-decoding intuition compresses a selected message given
  correlated side information.  It does not encode every future
  query-to-answer mapping in a raw record.

## Decision

Do not implement or run LDKS.  Retain only the extractive lemma as a control
for a future architecture in which a relation-aware reader has already reduced
the record to a unique support set.  The next candidate must change the
observable semantic algebra, not the membership codec.
