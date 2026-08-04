# Self-indexed sparse latent — Stage-0 preregistration

Status: **algebra and fatal controls frozen before execution**  
Date: 2026-07-30  
GPU: **not admitted**

## Claim being tested

An ordinary dense layer forgets which latent features produced its state and
rediscovers the next features with a global matrix or router.  The candidate
keeps feature identity in the state itself:

\[
S_t=\{(a_i,v_i)\}_{i=1}^{k},\qquad a_i\in\{0,\ldots,N-1\}.
\]

Every address owns a contiguous degree-`d` block of learned edge records.
One transition reads only the blocks owned by active addresses, emits their
messages, merges equal destinations, and retains at most `k` events.  The
state therefore routes itself; it does not score all `N` features.

For fixed word width, active count `k`, degree `d`, and transition depth `L`:

\[
P=\Theta(Nd)\ \text{resident records},\qquad
F=O(Lkd)\ \text{edge operations},\qquad
R=O(Lkd)\ \text{record reads}.
\]

Thus resident feature capacity can grow with `N` while active work remains
bounded.  This is a scaling separation from dense global matvec, not yet a
language-model result.

## What is different from an atlas or ordinary MoE

A local low-rank atlas still computes a dense router from the current hidden
vector and is algebraically a routed expert bank.  Here the active addresses
are first-class recurrent state.  Later transitions follow stored edges from
those addresses, so a global selection is not repeated at every step.

The changed currency is explicit: **word-addressed learned memory and discrete
state**.  A matched RAM/table-lookup controller receives the same advantage and
must be an equal control.  No superiority over that control is claimed.

## Frozen Stage-0 gates

The executable must pass every gate below.

1. On small random labeled graphs, indexed pointer walks exactly match dense
   one-hot transition matrices for every tested start and label sequence.
2. A direct table-lookup control exactly matches the candidate and has the same
   logical record-read count.  This prevents renaming random access as new
   information.
3. Across at least four graph sizes, active record reads remain exactly
   `L*k*d` while resident records grow as `N*d`.
4. The report includes a fixed 2-GiB ledger with 8-byte edge records,
   `d=16`, `k=256`, and `L=32`, including pointer width, resident node count,
   active bytes, merge volume, and the equal-byte dense-BF16 comparison.
5. The locality wall is reported: from one address, at most `d^L` paths and at
   most `1+d+...+d^L` distinct nodes are reachable in `L` steps.  Reaching an
   arbitrary one of `N` nodes therefore requires `L >= ceil(log_d N)` unless
   degree, a global router, or a direct externally supplied address grows.
6. All exact results and the preregistration hash are emitted as JSON.

## Interpretation frozen in advance

Passing proves a real compute/capacity separation for locally addressable
transition programs.  It does **not** prove better language modeling,
trainable routing, novelty, or GPU efficiency.

The branch is killed before GPU work if any later learnability screen needs:

- dense `N`-way edge logits or a global `N`-way router;
- active count or degree proportional to `N`;
- a dense shadow model at inference;
- uncounted tokenizer, retrieval, CPU, or host-memory work; or
- a quality claim based only on a task that directly supplies the correct
  internal address.

GPU admission requires a trainable model to beat a compute- and byte-matched
dense/PEER-style control by a **qualitative capability threshold or a large
compute-equivalent margin**, not a sub-percent language-loss change.

