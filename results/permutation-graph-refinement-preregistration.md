# Permutation-graph logical in-place refinement — pre-candidate preregistration

Status: **mechanism, workload families, accounting rules, and kill gates frozen**  
Date: 2026-07-25  
Candidate number: **none**  
GPU authorization: **none for this gate**

Frozen machine-plan SHA-256:
`db9218e7542ccfe20cef81f557c9df498a218bc2b91d035afae7ccffaba3c882`.

Post-freeze outcome: [closed at Stage 0 by the fatal exact
control](permutation-graph-refinement-stage0.md); no GPU or training run.

Exact implementation shapes, typed-operation ceilings, and every trained
artifact must be frozen in a Stage 0 child manifest before Stage 1 may start.
This document does not authorize training or a candidate claim.

## Question and possible gain

Can a bounded working state built from exact discrete labels, exact graph
relations, reversible permutations, and a small shared iterative rule improve
relational execution or corruption recovery at the same served-resource
envelope as strong flat, recurrent, and Transformer controls?

The possible edge is narrow: exact variable binding, mutable relational state,
multi-hop traversal, consistency repair, and synthetic operation-count
extrapolation. There is no claim of denser arbitrary factual knowledge,
unbounded recall, natural-language generality, or a general Transformer
replacement.

This is a pre-candidate falsification. It cannot admit candidate 004, support
an A–E claim, authorize a larger pretrain, or claim physical in-place
execution. A full pass permits only a later G0 proposal with captured
artifacts, workload hashes, numeric resource ceilings, and hardware.

## Streaming machine

Commands arrive one at a time. At step `t`, the system may read the current
command and current bounded state, execute one controller call, mutate the
state, and discard the command. It may not reread old commands or retain
encoder activations outside the declared state. A query arrives after the
command stream and is handled once. Any bit readable by a later step is
persistent state and counts.

For one `N = 64` episode, the primary state is

\[
\mathcal S_t=(X_t,Z_t,\Pi_t,p_t,c_t,status_t).
\]

- `X_t` is `BF16[64,16]`: 2,048 bytes of continuous local belief.
- `Z_t` is `uint16[64]`: 8 value bits, occupancy and condition flags, and six
  reserved zero bits: 128 bytes.
- `Pi_t={pi_0,pi_1}` is two `uint16[64]` involutions with fixed points or
  disjoint swaps: 256 bytes.
- `p_t` is `uint16[64]`: one exact parent or the `65535` root sentinel: 128
  bytes.
- `c_t` is exactly `BF16[32]`: 64 bytes of controller state.
- `status_t` is `uint16[1]`: bit 0 is the exact command-rejection flag and
  bits 1–15 remain zero: 2 bytes.

These views occupy 2,626 typed-storage bytes in one contiguous 4,096-byte
per-episode arena. The remaining 1,470 bytes are counted padding and are
zeroed, masked, and unreadable. No side tensor, Python object, host cache, or
encoder activation may carry episode state. Causal ablations use the identical
arena and offsets. Envelope competitors may use any layout up to the same
4,096-byte persistent-state cap.

An exact relation rewrite is a matching-preserving two-switch on four distinct
nodes:

\[
(a,b),(c,d)\longrightarrow(a,c),(b,d).
\]

Invalid endpoints cause a declared no-op plus error flag; they never silently
change the relation.

### Frozen refinement semantics

There are eight sweeps, with relation schedule

\[
(\pi_0,\pi_1,\pi_0,\pi_1,\pi_0,\pi_1,\pi_0,\pi_1).
\]

Within one sweep, every disjoint pair is read before that pair is written.
Because pairs do not overlap, ascending-minimum-node pair order is exactly
equivalent to synchronous Jacobi semantics and requires no second persistent
state buffer. Fixed points receive the frozen unary form of the same rule:
their continuous value is unchanged, and their discrete choice is restricted
to `NOOP`, `TOGGLE_I`, or `SET_I_FROM_CURRENT`.

For pair `(i,j)`, a shared local network emits one gate per feature,
`alpha in [0,1]^16`, and a categorical choice from the frozen local opcode
bank. At evaluation, categorical choices use deterministic argmax. Continuous
transport is

\[
\begin{bmatrix}x_i'\\x_j'\end{bmatrix}
=
\begin{bmatrix}1-\alpha&\alpha\\\alpha&1-\alpha\end{bmatrix}
\begin{bmatrix}x_i\\x_j\end{bmatrix},
\]

applied feature-wise. The exact channel may select only `NOOP`, `SWAP_VALUE`,
`MOVE_I_TO_J`, `MOVE_J_TO_I`, `TOGGLE_I`, `TOGGLE_J`,
`SET_I_FROM_CURRENT`, or `SET_J_FROM_CURRENT`; it cannot directly synthesize
arbitrary `Z` bits. Command execution and relation rewiring use the separately
frozen global opcode semantics below.

The exact operation count is computed per evaluated pair:

\[
F_{refine}=\sum_{k=0}^{7}\sum_{(i,j)\in M_k}C_{pair}(i,j,k),
\]

not the ambiguous `8NrC_rule` shorthand. MACs, comparisons, integer
operations, gathers, scatters, branches, and writes remain separate columns.
The eight sweeps do not make an 8-hop or 32-hop query free; traversal work is
executed and counted separately.

“In-place” here means only logical mutation of one bounded state. Physical
in-place inference, allocator behavior, register use, and synchronization are
unproven until Stage 4. Training/autograd workspace is disclosed separately
and cannot support a serving claim.

## Resource comparison

The primary `N=64` cap is 4,096 allocated persistent state bytes per episode
and 262,144 learned parameters. The Stage 0 child manifest must additionally
freeze actual learned bytes, inference workspace bytes, and numeric ceilings
for every typed operation before any training result exists.

Two comparison lanes prevent dummy work and crippled controls:

1. **Causal ablations:** identical tensors, allocation, controller, pair
   schedule, accesses, and issued work; the tested signal is masked, shuffled,
   or replaced while all else stays fixed.
2. **Envelope competitors:** each system may optimize its architecture but
   must stay at or below the same learned-byte, persistent-state, workspace,
   and separately typed operation ceilings. No dummy work is added to force
   equality.

Logical, allocated, allocator-reserved, and peak transient bytes are distinct.
Likewise, BF16/FP32 MACs, integer arithmetic, hashes, comparisons,
gathers/scatters, atomics, and communication are never converted to one
fictional equivalent operation. Every baseline configuration, training run,
checkpoint, and generator is individually hashed.

The strict lane requires complete task cost, not only graph-rule cost, to fit
the frozen envelope. A lower-state/higher-compute result is a **frontier
trade** unless a later service ledger establishes noninferior latency,
goodput, GPU-seconds, or joules.

## Ancestry memory–compute axis

This is a separate scaling diagnostic at `N in {64,256,1024}`. The 4,096-byte
primary cap does not apply to it. Each `N` gets a Pareto ledger with semantic
payload, tensor-storage allocation before framework allocator reservation,
allocator reservation, transient workspace,
logical reads/writes, and eventual HBM/L2 traffic. It cannot support a
candidate claim by itself.

For the single-parent forest `p`, define the strict-ancestor materialized
closure row

\[
a_v[u]=1\iff u\in Anc(v),\qquad u\ne v.
\]

The closure costs `N^2` bits, but that is **not** an exact-memory lower bound.
There are `(N+1)^(N-1)` labeled rooted forests, whose state entropy is

\[
(N-1)\log_2(N+1)\text{ bits}.
\]

An exact packed parent array is already within a small constant of that bound
and trades memory for traversal work.

| `N` | forest entropy | packed direct-ID parents + sentinel | frozen `uint16` parents | packed closure only | 64-bit sketch only | dedicated BF16x16 only |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 47.43 B | 56 B | 128 B | 512 B | 512 B | 2,048 B |
| 256 | 255.18 B | 288 B | 512 B | 8,192 B | 2,048 B | 8,192 B |
| 1,024 | 1,278.93 B | 1,408 B | 2,048 B | 131,072 B | 8,192 B | 32,768 B |

All cached modes also retain the exact `uint16` parent array so updates and
cycle checks are comparable. The frozen implementations are:

| Mode | Batch-one allocated state | Membership query | Mutation behavior | Guarantee |
|---|---:|---:|---:|---|
| `parent_recompute` | `2N` B | parent chase, `O(depth)` | `CUT` is `O(1)`; reparent includes `O(depth)` cycle check | exact |
| `exact_interval_rebuild` | `6N` B | preorder interval test, `O(1)` | cycle check plus full `O(N)` DFS interval rebuild | exact for the forest |
| `materialized_closure_bitset` | `2N + 8N ceil(N/64)` B | indexed packed bit, `O(1)` | cycle check plus full packed-closure rebuild | exact membership |
| `ancestor_bloom64_rebuild` | `10N + 32` B | four hashes and bit tests | cycle check plus full parent-chase recomputation of all rows | no false negatives after rebuild; false positives possible |
| `continuous_ancestor_bf16x16` | `34N` B plus learned weights | learned vector read | learned update/rebuild, fully counted | no exact identity guarantee |

The resulting batch-one tensor-storage allocations are frozen as follows;
learned weights and framework allocator reservation are separate ledger rows.

| `N` | parents | exact intervals | packed closure | Bloom64 | BF16x16 |
|---:|---:|---:|---:|---:|---:|
| 64 | 128 B | 384 B | 640 B | 672 B | 2,176 B |
| 256 | 512 B | 1,536 B | 8,704 B | 2,592 B | 8,704 B |
| 1,024 | 2,048 B | 6,144 B | 133,120 B | 10,272 B | 34,816 B |

`ancestor_bloom64_rebuild` stores one `uint64` per node. Four fixed `uint64`
seeds are committed before evaluation; bit positions are
`splitmix64(node_id xor seed_j) & 63`. A membership answer is positive only if
all four bits are set. There are no counters or hidden child indexes. After a
cut or reparent, every row is rebuilt by parent chasing, so deletion does not
leave stale bits. Hash work, parent reads, writes, the 32 seed bytes, and all
temporary storage count.

The continuous ancestry matrix is dedicated state; it may not alias primary
`X`. Neither the Bloom signature nor the continuous vector may claim arbitrary
modular path aggregates. Those remain a separate primary-graph query family.

At `N=64`, the Bloom payload is exactly as large as the packed closure before
metadata. At `N=1024`, Bloom state is smaller than a materialized closure but
larger than exact parents and exact preorder intervals. Therefore “2x smaller
than closure” is only a cache-compression diagnostic, not a memory edge over
the strongest exact representation.

The ancestry workload crosses membership-query-to-mutation ratios
`{1:1,8:1,64:1}`. Depth bins are `{1,8,32}` for `N=64`, add `128` for
`N=256`, and add `512` for `N=1024`. Mutations are 25% cuts, 50% valid
reparents, and 25% new-leaf attachments. Every mode receives the identical
forest and operation trace.

## Frozen workload semantics

Every episode begins from a randomized node renaming and a randomized valid
state. Values are bytes; path sums use arithmetic modulo 257. Global opcodes
are:

- `SET(i,v)`: set byte `v` and occupancy at `i`;
- `SWAP_VALUE(i,j)`: swap value and occupancy, leaving conditions attached to
  nodes;
- `MOVE_VALUE(i,j)`: move value and occupancy to `j` and clear them at `i`;
- `TOGGLE(i)`: flip the condition bit;
- `TWO_SWITCH(r,a,b,c,d)`: apply the valid four-distinct-node relation rewrite;
- `SET_PARENT(child,parent)`: reject self-links and cycles, otherwise reparent;
- `CUT_PARENT(child)`: replace its parent with the root sentinel.

Rejected operations set the controller error bit and leave all task state
unchanged. Complete-state and complete-trajectory equality cover `Z`, both
permutations, `p`, and the error bit; learned beliefs are scored separately.

Training uses 16 or 32 active nodes and at most 32 mutations. Evaluation uses
64 active nodes and `{64,256,1024}` mutations. At least half of commands address
an operand relationally rather than by literal node ID.

Query families are complete state, complete trajectory, one-hop, 8-hop,
32-hop, ancestor membership, and modular path aggregate. Hop queries use
reduced relation words with no adjacent identical involution. Generated
8-hop/32-hop paths must visit at least 9/33 distinct nodes; short cycles and
involution cancellations cannot satisfy those cells. Query execution and every
pointer read count against each system.

Natural-language routing is a later synthetic gate. Its locked set uses unseen
command paraphrase families, entity renamings, clause order, distractors, and
counterfactual changes. Structured commands and text commands never share
templates across train and evaluation. Passing this gate means only held-out
synthetic-command routing.

## Controls

Every envelope control receives the same command stream, task information,
exact global opcode bank, relation/pointer primitives, query interface,
training examples, optimizer-update budget, and tuning-trial budget.

1. **Flat exact registers:** exact state and primitives, but no learned pairwise
   graph refinement. Saved work may be spent elsewhere under the same typed
   ceilings.
2. **Continuous recurrent graph:** same exact relation/pointer primitives and
   resource envelope, but mutable payload is continuous and has no exact `Z`
   channel.
3. **One-pass discrete graph:** exact task state with one learned pass; it may
   use the full operation ceiling but cannot reread intermediate node state.
4. **Tiny Transformer or recurrent scratchpad:** strongest dev-selected member
   of a frozen shape grid under the same state, learned-byte, workspace, and
   typed-operation ceilings, with the same exact primitive bank.

Causal ablations retain identical execution while replacing meaningful
permutations by random valid involutions, disabling rewiring, masking the exact
channel, or shuffling refinement evidence. A Stage 0 child manifest freezes all
shape grids, trial counts, training compute, selection rules, and per-variant
hash requirements before Stage 1.

## Frozen stage order and decisions

### Stage 0 — exact executor, generator, and accounting

Implement oracle commands and node bindings only. Exact executors must preserve
all invariants and reach 100% bitwise task-state/query accuracy through 1,024
operations. Reconcile the primary arena, every ancestry allocation, workspace,
and typed operation ledger. Freeze generator, corruption traces, baseline shape
grid, numeric resource ceilings, training budget, and artifact schema in a
hashed child manifest.

Gold traces come from a separately implemented reference interpreter that the
tested executor and generator may not import or call. The child manifest hashes
that interpreter. Before randomized traces, exhaustively enumerate small-`N`
opcode validity classes, involutions, and forests for `N <= 6` with value
alphabet `{0,1,255}`. Metamorphic tests cover `pi(pi(i))=i`, two-switch
invertibility, forest acyclicity, rejected-operation no-op behavior, and
equivariance under arbitrary node renaming.

Failure is a harness/executor failure. Passing authorizes Stage 1 only; it is
not evidence for the mechanism.

### Stage 1 — oracle-bound frozen correction

Use oracle global opcodes and node bindings so router or language errors cannot
explain the result. All systems receive identical corrupted current-command
evidence. The frozen
grid crosses operand-evidence erasure probabilities `{0.05,0.10,0.20}` with
candidate-set sizes `{2,4}` and local evidence-feature sign-flip probabilities
`{0.01,0.05,0.10}`; severity zero is the clean control. No cell may be selected
after results, and corruption never changes the hidden source-of-truth task
state.

At `N=64,T=1024`, the simultaneous one-sided lower bound for improvement over
both the strongest one-pass control and strongest envelope competitor must be
at least five points on exact complete-trajectory recovery. Clean accuracy may
regress by at most one point, no query family may regress by more than one
point, and every invariant violation fails. A gain obtained by exceeding any
typed-operation, state, learned-byte, or workspace ceiling fails rather than
becoming a win.

### Stage 2 — learned canonical-command routing

Run only if Stage 1 establishes that refinement itself is useful. Learn opcode
and argument routing from canonical structured commands under the streaming
contract. Every seed must reach at least 99% exact complete-trajectory accuracy
at 1,024 operations. The candidate must be noninferior to the strongest matched
control: the simultaneous one-sided lower confidence bound for every
candidate-minus-control query/length cell must be at least `-1` percentage
point.

There is deliberately no five-point graph-advantage gate on clean routing. With
clean canonical commands and shared exact primitives, such a gap would more
likely expose an unequal baseline than useful refinement.

### Stage 3 — held-out synthetic-language routing

Run only after Stage 2 passes, without changing architecture, thresholds, or
templates. Every seed must retain at least 99% exact complete-trajectory
accuracy at 1,024 operations. Accuracy may decline by at most one point per
sequence-length doubling from 64 to 1,024, and every protected query family
retains the Stage 2 noninferiority bound.

### Stage 4 — hardware feasibility

This stage is not authorized. It becomes eligible only after Stages 0–3 pass.
A later preregistration captures one exact H100/H200 stratum and measures
allocated/reserved state, issued work, HBM/L2 traffic, synchronization,
registers, occupancy, p50/p95 latency, goodput, and energy. Logical-byte or
nominal-FLOP savings alone cannot promote the branch.

## Statistics and stopping

- Training seeds: `17`, `29`, `43`.
- Each evaluation cell has at least 256 independent latent graph worlds and
  10,000 episodes.
- Comparisons are paired by world and trace. Bootstrap latent worlds with
  10,000 resamples using RNG seed `20260725`.
- Simultaneous one-sided 95% bounds use the bootstrap max statistic over all
  registered seed, length, query-family, and corruption cells. Superiority
  uses a lower bound; nonregression uses the corresponding upper loss bound.
- Report each seed, every cell, and the worst seed. Opcode, synthetic-language,
  clean, and each corruption severity remain separate.
- Generator code, all configuration hashes, and locked world-seed commitments
  are published before training. Every trained per-variant/per-seed artifact is
  hashed after training and before any locked Stage 1 evaluation.

Stop when a required stage fails. Do not repair a failed threshold by adding
state, rounds, controller width, hidden exact copies, training trials, or
test-time work under the same claim.

## Collision and claim boundary

[Discrete graph diffusion](https://arxiv.org/abs/2209.14734),
[graph cellular automata](https://arxiv.org/abs/2110.14237),
[latent diffusion reasoning](https://arxiv.org/abs/2510.04573), and
[masked diffusion language models](https://arxiv.org/abs/2406.07524) establish
nearby components. The provisional conjunction tested here is narrower:
bounded exact permutation transport, dual discrete/continuous graph state, and
iterative logical correction under a served-resource envelope. Novelty is not
claimed by this preregistration.
