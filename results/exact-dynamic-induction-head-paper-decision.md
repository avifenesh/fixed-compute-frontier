# Exact dynamic induction head: paper decision

Date: 2026-08-01  
Status: **REJECT AS A BREAKTHROUGH; HOLD ONE NARROW KV-GROUP SUBSTITUTION; NO CODE OR GPU**

## Proposed object

The exact dynamic induction head (EDIH) would replace an attention component
with an online suffix data structure over token IDs.  At step `t`, it finds the
longest suffix of `x[1:t]` that occurred earlier and emits either an earlier
continuation or an empirical continuation distribution.  The analog trunk
would retain semantic processing while the digital head handled exact
repetition, copying, code, and template continuation without Q/K/V projection.

## Decision

The exact-token specialization is real, but the broad claim fails for three
separate reasons:

1. exact arbitrary-length recall necessarily uses growing state;
2. an ordinary suffix automaton does not maintain exact continuation counts in
   amortized constant time;
3. suffix/cache/induction mechanisms already cover the algorithmic object, so
   novelty could only come from a fixed-budget architectural substitution with
   material natural-language evidence.

The defensible gain is narrower: more literal context per state byte and exact
copy/repetition behavior.  There is no derived gain in general reasoning or
stored parametric knowledge.

## 1. Information lower bound

Suppose the head must later reproduce any `L`-token string over a vocabulary
of size `V`.  If two histories lead to the same `B`-bit state, a common future
query cannot distinguish them.  Injectivity therefore requires

\[
2^B\ge V^L
\quad\Longrightarrow\quad
B\ge L\log_2 V.
\]

EDIH does not evade the finite-state obstruction of a recurrent model.  It
pays for exact recall with state that grows with context.  A stronger recent
result proves an `Omega(N)`-bit lower bound for deterministic online longest
repeating suffix computation even over a constant alphabet; see the
[online LRS lower bound](https://arxiv.org/abs/2607.05004).

This may still be a favorable *representation exchange* relative to BF16 KV,
but it is not free memory or unbounded intelligence at capped state.

## 2. The O(1) claim splits into two different operations

An online suffix automaton has amortized linear construction and can maintain
longest-match structure.  A stored position can support cheap drafting, as in
[SAM Decoding](https://aclanthology.org/2025.acl-long.595.pdf).  That establishes
neither an exact empirical distribution nor constant-time frequency updates.

For context `u`, exact next-token counts are

\[
C_t(u,a)=\#\{i<t:x_{i-|u|+1:i}=u,\ x_{i+1}=a\}.
\]

Appending a token changes the occurrence count of every suffix represented on
the suffix-link path from the new terminal state.  On `a^t`, that path can
contain `Theta(t)` relevant suffixes.  The honest options are:

- store one occurrence/continuation pointer, which is not a distribution;
- cap the context order at `r`, giving `O(r)` update work;
- use lazy dynamic-tree machinery, generally introducing logarithmic work and
  substantial mutable metadata; or
- periodically rebuild/propagate counts and charge the amortized work and
  synchronization.

Thus “amortized O(1) automaton update” and “O(1) exact distribution update” are
not interchangeable claims.

## 3. State ledger

A suffix automaton for a length-`N` sequence has at most `2N-1` states and
`3N-4` transitions.  Even a compact design needs state fields such as suffix
link, maximum length, occurrence position/count, adjacency offsets, transition
labels, and transition destinations.

A lower-level packed ledger is roughly 30–40 bytes per token before dynamic
hash/table slack.  A mutable batched implementation can plausibly reach
80–120+ bytes/token.  By comparison, one BF16 K/V head of dimension 128 stores

\[
2\cdot128\cdot2=512\ \text{bytes/token}.
\]

So a useful 4–10x state advantage is plausible for exact suffix queries.  It
must be compared at concrete shapes.  In grouped-query attention, deleting one
query head saves no K/V state; the substitution must remove an entire KV group
to fund the digital structure.

A capped explicit context trie can instead reach `Theta(rN)`
context/continuation entries, so it cannot inherit the suffix automaton's
linear-state claim.

## 4. Why Bayesian mixing is not a no-harm theorem

For sequence experts `p_i` and prior weights `pi_i`, the sequence mixture

\[
q(x_{1:T})=\sum_i\pi_i p_i(x_{1:T})
\]

satisfies

\[
L_q\le \min_i\bigl(L_i-\log\pi_i\bigr).
\]

For a uniform prior this is cumulative teacher-forced log-loss regret of at
most `log M` against the best of `M` static experts.  It is not a per-token
correctness guarantee, a downstream benchmark guarantee, or a free-generation
no-harm result.  If the original LM is included as an expert, its logits still
have to be computed, removing the claimed replacement saving.  Classical
[context-tree weighting](https://pure.tue.nl/ws/portalfiles/portal/1383848/Metis122608.pdf)
already supplies a principled variable-order context mixture.

## 5. Natural capability boundary and prior art

Exact matching is useful for copying names, repeated key/value pairs, code,
JSON, and templates.  It misses paraphrase, aliases, coreference, and latent
relations.  MQAR is close to the data structure's native operation and cannot
stand in for production language evidence.

The object also collides heavily with existing work:

- suffix-automaton and suffix-tree drafting in
  [SAM Decoding](https://arxiv.org/abs/2411.10666) and
  [SuffixDecoding](https://proceedings.neurips.cc/paper_files/paper/2025/file/b7aea253ab34a773967f1e4cdea9e4fb-Paper-Conference.pdf);
- neural and explicit cache language models;
- exact/fuzzy induction models such as
  [Induction-Gram](https://arxiv.org/abs/2411.00066);
- conditional n-gram memory in
  [Engram](https://arxiv.org/abs/2601.07372); and
- exact longest-match lookup into latent memory in
  [Memory Grafting](https://arxiv.org/abs/2605.20948).

The suffix lookup itself is therefore not a novelty claim.  Only a
fixed-serving-budget substitution that improves a trained model could become a
new result.

## Retained experiment, not yet admitted

The only surviving experiment replaces one *whole KV group* in a small hybrid
with a capped exact-context expert.  It must freeze:

- total parameters and peak persistent/mutable state bytes;
- served MACs, memory traffic, batch, throughput, and p50/p95/p99 latency;
- training tokens and compute;
- natural held-out NLL, exact-copy/code slices, semantic reasoning, and
  long-context tasks.

Required controls are the unmodified model, the same KV group removed,
equal-byte local attention, rolling n-gram/trie, SAM single-pointer, CTW, and a
neural cache.  Before training, the data structure itself needs a CPU/GPU
microbenchmark of update/query traffic, cache misses, and batch divergence.

This experiment is admitted only if a frozen optimistic effect equation shows
that the exact-match slice can produce a material aggregate gain after all
semantic regressions and the removed KV group are charged.  If the only win is
MQAR or speculative-decoding speed, the lane closes as non-breakthrough.

## Bottom line

The implementation needed rethinking, and the general claim hit a wall.  EDIH
can buy much cheaper literal memory than BF16 KV; it cannot turn that literal
memory into general semantic intelligence, eliminate growing state, or claim
no-harm at no cost.
