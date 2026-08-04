# Quotient-group memory — correct classical primitive, architecture claim closed

Status: **retain only as a classical typed-RAM primitive**  
Date: 2026-07-26  
Candidate number: **none**

## Outcome

The algebra is correct, but the architecture claim does not survive. A weighted
disjoint-set forest maintains a quotient of entities and exact group-valued
relations inside each equivalence class. That is a useful typed data structure,
not a new model algebra: a recurrent controller with the same indirect-address
RAM executes the identical algorithm, and the exact mechanism already appears
in prior work.

## Operator

Let `G` be a fixed, word-sized finite group and let each active entity `i` have
an unknown group coordinate `z_i`. The head exposes:

```text
LINK(a, b, g): assert z_b = g * z_a
REL(a, b):     return g such that z_b = g * z_a, or DISCONNECTED
```

If a new `LINK` closes a cycle, the head returns `CONSISTENT` or
`CONTRADICTION`; a contradiction does not mutate the semantic relation state.
Aliases are the trivial-group case. Nontrivial groups add exact relative
offsets, permutations, unit transforms, or coordinate-frame relations.

The implementation is a weighted disjoint-set forest. It stores a parent
pointer, rank, and potential `w_i` with invariant

\[
z_i=w_i z_{parent(i)}.
\]

After `FIND(i)` compresses the path to root `r_i`, it returns `P_i` satisfying

\[
z_i=P_i z_{r_i}.
\]

For connected `a,b`, the answer is therefore

\[
REL(a,b)=P_bP_a^{-1}.
\]

For a constraint `z_b=g z_a`, if root `r_b` is attached under `r_a`, its stored
potential is

\[
w_{r_b}=P_b^{-1}gP_a.
\]

Substitution gives `z_b=g z_a` immediately. The reverse-rank attachment uses
`w_{r_a}=P_a^{-1}g^{-1}P_b`. Path compression replaces adjacent potentials by
their ordered group product, so the invariant is preserved even for a
non-commutative group. This is the correctness proof by induction over links
and compressions.

## Information theorem

For `n` entities and group order `q`, the number of distinct query-answer states
is

\[
C_n(q)=\sum_{k=1}^{n}{n\brace k}q^{n-k},
\]

where `{n brace k}` is a Stirling number of the second kind. A partition with
`k` components has one free group coordinate for every non-root entity.

Any exact state must therefore carry at least

\[
\lceil\log_2 C_n(q)\rceil
\]

bits: if two distinct labeled partitions shared a machine state, some `REL`
query would require different answers from that same state.

Two simpler subfamilies expose the asymptotics:

- the trivial group contains all `B_n` Bell partitions and requires
  `Omega(n log n)` bits;
- one connected component has `q^(n-1)` relative labelings and requires
  `Omega(n log q)` bits.

The time claim uses an **online word-RAM/cell-probe model** with
`Theta(log n)`-bit words, one parent or fixed-group element fitting in a word,
and every memory probe and group operation charged. With `G` fixed, the forest
uses

\[
O(n(\log n+\log q))
\]

bits, matching the combined logical-state lower bound within a constant factor.
With union-by-rank and path compression, `m` operations cost
`O(m alpha(m,n))` word probes and fixed-group operations. The ordinary
Union-Find problem is the `q=1` subcase, so the standard online cell-probe lower
bound applies to this broader problem as well. This is a data-structure bound,
not a neural-architecture separation.

If `q` scales, this ledger is incomplete: a Cayley table can cost
`Theta(q^2 log q)` bits, while a compact group representation may make
multiplication and inversion nonconstant. A variable-group experiment must
charge that representation and its actual operations.

## What this buys

For a stream of correct monotone group operations over already assigned dense
entity slots, the structure stores exact transitive relations without retaining
or rescanning the event history. This is a workload advantage over an unindexed
history scan. It is not an advantage over the strongest matched controller,
which simply uses the same data structure.

The resource being introduced is precisely:

> mutable, indirectly addressed, typed combinatorial state.

For `n=64` and the fixed six-element non-commutative group `S3`, the exact
distinguishing-state lower bound is 339 bits. The simple logical fields use 768
bits (96 bytes if truly bit-packed); a more convenient hypothetical
`uint32/uint8/uint8` array layout uses 384 bytes. Neither number is the Python
implementation's measured memory.

The theorem assumes 64 predeclared dense slots. It excludes the name-to-slot
map, active/free-slot and generation metadata, overflow/eviction/deletion,
operation evidence and provenance, Python object/allocator overhead, executable
code, and packed-kernel implementation. Every applied system must charge them.

## Executed proof checks

The CPU gate uses `S3`, so incorrect multiplication order cannot hide behind a
commutative group. A weighted forest and an independently structured
breadth-first graph traversal, sharing only the tested group primitives, agreed
across three deterministic traces on:

- 12,288 `LINK` operations, including consistent cycles and contradictions;
- 49,152 independently resolved `REL` queries;
- preservation of existing answers after rejected contradictions.

The focused tests separately check group inverses and associativity,
non-commutativity, path composition, contradiction rejection, the state-count
formula reducing to the first five Bell numbers when `q=1`, and independent
brute-force enumeration for small `q>1`.

Artifacts:

- [Executable Stage-0 gate](../experiments/quotient_group_memory.py)
- [Machine-readable result](quotient-group-memory-stage0.json)
- [Focused tests](../tests/test_quotient_group_memory.py)

## Collision and failure boundary

The late collision is direct. ED-Batch Algorithm 5 stores a parent and a
possibly non-commutative permutation transform to that parent, returns a root
and root-relative transform, joins roots with composed transforms, and rejects
incompatible cycles. Library Checker's named task also exposes non-commutative
potential union-find. ED-Batch uses it for neural-system memory planning rather
than as an LLM request-memory head, but the claimed operator and proof mechanism
are already established.

RAM-Net additionally covers learned sparse indirect addressing at the broader
substrate level. Neural Turing machines and generic RAM-augmented recurrence can
simulate the operator. What remains is an engineering hypothesis—whether a
typed classical data structure is worth exposing to a model—not a new
architecture theorem.

The head cannot retain evidence, chronology, uncertainty, deletions, or
non-group relations. A wrong learned merge is destructive. Resolving natural
language into stable entity IDs and correct group elements may be harder than
maintaining the relations after resolution. Pointer chasing and scattered
writes may also lose to dense GPU kernels despite superior logical complexity.

Primary references:

- [Tarjan's set-union analysis](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1974/28764.html)
- [Fredman–Saks cell-probe lower bound](https://doi.org/10.1145/73007.73040)
- [ED-Batch, especially Algorithm 5](https://proceedings.mlr.press/v202/chen23g/chen23g.pdf)
- [Library Checker non-commutative potential union-find](https://judge.yosupo.jp/problem/unionfind_with_potential_non_commutative_group)
- [Concurrent DSU and lower-bound discussion](https://arxiv.org/abs/2003.01203)
- [RAM-Net selective-address memory](https://arxiv.org/abs/2602.11958)
- [Neural Algorithmic Reasoning](https://arxiv.org/abs/2105.02761)

## Decision

The implementation and theorem work under a correct typed operation stream.
They re-demonstrate a classical weighted union-find result; they do not prove a
new model method. The architecture branch is closed, no candidate 004 is
admitted, and neither learned-interface training nor GPU rental is justified by
this claim. The proof is retained so we do not rediscover or overclaim it later.
