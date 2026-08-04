# Thinking Table — Architecture Frontier

Status: **gauge-funded G1 retained as a pre-candidate; no candidate 004 admitted**  
Date: 2026-07-27

The research target is the model architecture itself: Transformer versus
recurrent/SSM models, hybrids, and sparse conditional computation. Corpus
selection, curriculum, distillation, and other weight-improvement recipes are
not candidate mechanisms here. Training data is held fixed when architectures
are compared.

The latest outward pass is consolidated in
[`results/outward-mathematics-reset.md`](results/outward-mathematics-reset.md).
Its matched-machine theorem is now an admission gate: a finite-precision
operator is not an architecture edge merely because it is written in a new
algebra. The proposal must first identify randomness/error, physical hardware
asymmetry, distributional structure, or learnability as the changed currency.

The serving envelope remains fixed:

\[
W_{\mathrm{resident}} + B K(n) + X \le M_0, \qquad
F_{\mathrm{active}}(n) \le F_0, \qquad
T_{\mathrm{decode}} \le T_0.
\]

The newest result is a narrower pre-candidate, not an admission. A full frozen
package failed 3/5 because a centered scale-vs-rotation conditioning claim was
not stable. Inside that package, an uncentered one-curvature key passed every
direct teacher-free content-addressing gate in 5/5 worlds at the same 40
learned scalars and serialized bytes as its controls. Details and closure are
in
[`results/gauge-embedded-curved-keys-energy-decision.md`](results/gauge-embedded-curved-keys-energy-decision.md).

The retained key shift is one quadratic content feature per RoPE pair funded
by an existing gauge coordinate—not G2, and not a claim that the scale chart is
intrinsically better. Its next gate uses new unseen worlds, adds an ordinary
signed-linear no-harm task, and keeps H100/LM admission closed until then.

`W` is resident learned state, `K(n)` is per-request KV/recurrent state at
context length `n`, `B` is frozen concurrency, and `X` is workspace. Prefill,
energy, and training cost are recorded separately. A changed mathematical
operation is counted even if a GPU bottleneck hides its latency.

## What the existing primitives actually buy

| Primitive | Resource it buys | Price it pays |
|---|---|---|
| Full attention | Exact content-addressable token memory and fresh layer-specific reads | KV grows with context and layer count; scans and traffic grow with context |
| Mamba / recurrent state | Constant-size context state and constant decode work | History is compressed; arbitrary exact recall eventually aliases |
| Linear / delta attention | Query-dependent reads from a fixed sufficient statistic | State rank limits the attention maps it can represent |
| Dense MLP | A reusable nonlinear transform and parametric knowledge | Every token reads and executes the whole transform |
| MoE | More conditional learned functions at similar active FLOPs | All experts still consume resident bytes; routing and dispatch add costs |
| Static hybrid | A fixed compromise between exact recall and compressed state | The same compromise is imposed on every token and sequence |

The key separation is:

1. **parametric memory** — facts and procedures stored in weights;
2. **exact context memory** — independently addressable token records;
3. **compressed context state** — bounded recurrent summaries;
4. **computation topology** — which earlier representations later computation
   is allowed to reuse.

Transformer, Mamba, and MoE are not competitors on one scalar axis. They spend
different resources on these four jobs.

## Hard boundaries

- A fixed `S`-bit recurrent state cannot distinguish more than `2^S` histories.
  It cannot promise exact recall of unbounded random information.
- A linear-attention map with feature width `r` has rank at most `r`; arbitrary
  rank-`n` attention requires `r >= n` in the worst case.
- MoE does not remove model bytes. At fixed VRAM, additional experts must be
  paid for by smaller experts, fewer weights elsewhere, lower precision, or
  reclaimed request state.
- Route count, expert-path count, or state-slot identity is not independent
  learned information. Result 001 already falsified that shortcut.
- A broad win is possible only by exploiting structure in real workloads:
  compressible histories, conditional feature use, redundant layer memories,
  or reusable computation.

## Candidate 003: bounded semantic feedback hybrid — closed

The three-seed probe found that top-written feedback improves structured state
tracking, but the recurrent summary is unnecessary and exact token-level
feedback creates an `nL` nonlinear prefill critical path. See
[`results/003-bounded-semantic-feedback.md`](results/003-bounded-semantic-feedback.md).

The design below is retained as the audited hypothesis, not an active candidate.

### The architectural change

Ordinary decoder layers cache their own pre-attention representations. A high
level representation computed for token `t` is not directly available to the
low layers processing token `t+1`.

The candidate uses one globally shared, strictly bounded memory written from
the completed high-level representation of each token:

1. a small exact KV ring with `C` semantic token records;
2. a fixed recurrent summary `(S, z)` updated from records evicted from that
   ring;
3. every layer reads both tiers using its own query, with no other long-history
   layer-local KV caches.

The exact tier preserves recent or selected discrete bindings. The recurrent
tier retains compressible older structure. Writing high-level representations
also creates a feedback path from late computation at token `t` to early
computation at future tokens.

This is a **memory-topology hypothesis**, not a claim that compression beats the
information bound.

### Why capability could improve

- **Greater effective depth across time.** A belief or partial computation
  formed at the top of token `t` can be refined from the bottom at token `t+1`
  instead of being reconstructed from low-level cached features.
- **More useful bytes.** One semantic record is shared across read depths rather
  than storing a separate KV view at every layer.
- **Two kinds of history.** Exact slots handle arbitrary bindings while the
  recurrent state handles aggregates, repeated structure, and state tracking.
- **Reinvestment option.** If shared memory removes projection and cache bytes,
  those bytes can later fund wider dense transforms or MoE experts while the
  total deployment envelope remains fixed.

The predicted gains are state tracking, variable binding, iterative/algorithmic
computation across generated tokens, and mixed exact-plus-structured recall.
There is no claim of universal long-context recall.

### State and compute ledger

For `L` ordinary attention layers with KV width `d_kv`, element width `b_s`, and
context `n`:

\[
K_{\mathrm{ordinary}} = 2 B L n d_{kv} b_s.
\]

For a shared exact ring of capacity `C`, exact-key width `d_k`, recurrent
feature width `r`, and value width `d_v`:

\[
K_{\mathrm{candidate}}
= B b_s \left(C(d_k + d_v) + r d_v + r\right),
\]

which is independent of `n`. The candidate writes one shared K/V projection
per token instead of one per attention layer. Exact scans and summary reads
still occur at every reading layer; `C` and `r` must therefore be chosen so the
measured full block, not nominal FLOPs alone, fits `F_0` and `T_0`.

Removed layer-local K/V projection bytes are not silently called a win. In the
equal-parameter capability test they are reallocated to model width; in the
smaller-model test they are removed. Both comparisons keep their own ledgers.

### The different cost

Top-output feedback creates a token-serial dependency. It can make training and
prompt prefill less parallel even though autoregressive decode is already
serial. That is the explicit alternative currency. If exact wavefront/chunk
execution cannot keep prefill inside the frozen serving envelope, the candidate
fails; training difficulty does not excuse a serving regression.

Other risks are a shared-memory bottleneck, recurrent-summary collisions,
unstable long feedback loops, loss of useful layer-specific views, and extra
register/traffic pressure from reading two memory types.

## Cheapest causal test

Train small equal-parameter/equal-operation models on private generators that
randomly mix:

1. delayed random key-value recall, which requires exact storage;
2. counters, parity, finite-state transitions, and running aggregates, which
   reward recurrent state;
3. variable-order multi-step programs, which test effective temporal depth;
4. mixed episodes whose future query is unknown when memory is written.

The matched causal controls keep the same global memory, learned modules,
buffering, eviction schedule, reads, writes, state bytes, and commit timing:

- write the input-level rather than completed top-level token representation;
- mask the exact-read contribution while still executing and updating it;
- mask the summary-read contribution while still executing and updating it.

An ordinary layer-local two-tier hybrid is retained only as a stronger
diagnostic. It uses `L` times the persistent state and `L` writer commits, so a
candidate win against it would be strong, but a loss would not isolate the
cause or by itself falsify the fixed-budget architecture.

The candidate survives the causal screen only if top-level writing beats the
matched input-level writer and both memory tiers are independently necessary
across at least three seeds. Metrics are reported separately for recent random
bindings, random bindings older than the exact ring, and structured state
updates; no aggregate may hide a failed slice. A fixed-data small
language-model A/B and the protected capability suite follow before scaling.

## Hardware gate

Before a larger pretrain, benchmark the complete block on the reserved H100:

- prefill and decode at the frozen context/concurrency cells;
- state and workspace bytes;
- HBM/L2 traffic, registers, occupancy, spills, and energy;
- p50/p95 latency and throughput;
- exact accounting for the feedback dependency and any wavefront schedule.

Kill the branch if it materializes hidden per-layer memories, increases the
frozen state/weight budget, misses decode noninferiority, or makes prompt
prefill incompatible with the service contract.

## Late collision boundary

The components are established; the precise intersection is the research
claim, so novelty remains provisional:

- [Feedback Transformer](https://arxiv.org/abs/2002.09402) and
  [LCKV](https://arxiv.org/abs/2405.10637) provide shared high-level feedback
  KV, but the exact history grows or is simply evicted.
- [Recurrent Transformer](https://arxiv.org/abs/2604.21215) adds temporal depth
  with per-layer memories, retaining `O(Ln)` exact state.
- [YOCO](https://arxiv.org/abs/2405.05254) shares a global cache without the
  high-to-low temporal feedback edge.
- IndexMem and Tensor Cache combine exact and compressed tiers, but retain
  layer-local memory rather than one global semantic feedback bank.

That was the narrow claim tested. It did not survive: the summary was
unnecessary, and the exact feedback topology missed the prefill-latency gate.

## Return to zero after 003

No next candidate is admitted yet. The following are retained as collision
checks, not renamed as new work:

| Tempting direction | Why it is not the next frontier |
|---|---|
| Route the same token to a parameter expert and its private persistent state | [Mixture-of-Memories](https://arxiv.org/abs/2502.13685) already leaves inactive memories unchanged and gives each memory specific projections and kernel parameters; [Sparse Delta Memory](https://arxiv.org/abs/2607.07386) scales sparse mutable state further |
| Route Mamba/SSM parameters as experts | [Routing Mamba](https://arxiv.org/abs/2506.18145) and [MossNet](https://arxiv.org/abs/2510.26182) already cover this family |
| Share fewer exact KV banks and spend the bytes on MoE | YOCO/LCKV-style cache sharing and modern compressed-KV MoE hybrids already occupy the basic exchange |
| Compose a small operator-expert bank in sequential paths | Ordinary multilayer MoE already creates paths, while Cartesian/product expert work covers the obvious combinatorial factorization; logical path count is not new learned information |
| Share a learned recurrent initial state and copy only each request's modified entries | Exact copy-on-write saves request memory only while the state remains physically sparse. For state size `D`, value bytes `b`, index bytes `i`, batch `B`, and `u_t` dirty entries, it requires `u_t/D < (1-1/B)b/(b+i)`; with BF16 values and 32-bit indices the large-batch limit is below one third. Dense recurrent updates fail immediately. When the inequality holds this is a useful implementation optimization, not a source of additional learned information or capability. |
| Bank compute on easy tokens, then spend it on attention/depth/MoE for hard tokens | If the heavy route fits the frozen per-token cap, this is ordinary conditional routing. If it needs saved credit, hard-token latency exceeds that cap. A causal bank also cannot help an early difficult token using future savings. This can preserve average FLOPs or worst-case latency, not both while granting extra serial work; CoLT5, Mixture-of-Depths, Taipan, AdaMoE, and TriRoute already occupy the modeling family. |
| Decouple many narrow address heads from fewer wide value/payload heads | At fixed cached width `S`, a learned `A x B` address-to-payload crossbar adds both `AB` parameters and `AB` work per history token; a nontrivial crossbar therefore cannot simultaneously match parameters, KV state, and attention FLOPs. The transported features still lie in the span of the `B` payload maps. Talking-Heads, Collaborative MHA, Compositional Attention/DCMHA, DHA, MFA, and IHA already cover static and dynamic versions of this factorization. |
| Add a tiny exact digital register beside attention and recurrent state | Exact XOR, modular counters, and finite-group updates are cheap, but the learned operator router remains approximate: a trajectory survives `T` decisions with probability `(1-epsilon)^T`. State-dependent routing is token-serial; scan-compatible routing must be restricted to a compact closed operator family. Arbitrary `b`-bit automata still require exponential transition information in the worst case. Neural RAM/Programmer, PD-SSM, and Symbolic Neural CPU already occupy the mechanism; it is a useful specialized executor, not a general density or knowledge edge. |
| Give useful DeltaNet state heads more rank and shrink the others under one fixed state budget | Direct Qwen3.5 anatomy confirmed unequal need: a spectral-energy allocation improved frozen-model KL about 2.5x over uniform rank 32 at identical representable bytes. It still missed the pre-registered noninferiority gate; the first passing neighborhood was rank 64, where factor storage is no smaller than dense. A one-time SVD also does not provide a closed capped-rank update or fast kernel. [State Rank Dynamics](https://arxiv.org/abs/2602.02195) independently occupies the broad stratification/pruning claim. |
| Replace full-attention reads with top-k, one tail summary, or low-order moments | On Qwen3.5-0.8B, exact-oracle top-k 64 missed the natural KL limit by about 32x; top-k plus an exact-mass tail mean missed by about 8.6x and failed shifted/structured panels. Even top-k 256 failed and exceeded the resource screen. A physical KV-group localization found 0 of 12 groups passed individually, so no 9-of-12 combination was allowed. The failure precedes selector, state, and kernel costs. |

### Constraint-basin algebra gate — closed

The nature-inspired proposal used a corrupted continuous state, sparse parity
constraints, and a fixed number of in-place energy-descent steps. Algebraically,
its leave-one-out factor products are ordinary factor-to-variable messages and
its clipped update is a tied recurrent hypergraph layer. A matched generic
recurrent control embeds it exactly under the same state, steps, and arithmetic,
so it cannot claim a distinct best-achievable capability algebra over that
control.

At binary states the first step is threshold bit-flipping. Replacing constraints
with a log-sum-exp bank of learned prototypes makes one unit energy step exactly
a softmax attention read. Exact MAP remains the source-recovery ceiling, and a
direct Hamming syndrome decoder reaches it on the frozen toy clean/single-error
panel. The CPU evidence is recorded in
[`results/constraint-basin-no-go.md`](results/constraint-basin-no-go.md).

This does not establish resource dominance over every basin implementation. An
analog-confidence decoder with node rather than per-edge dynamic state may
still occupy a useful engineering tradeoff against belief propagation. It is
not admitted as a distinct language-model capability algebra.

### Quotient-group memory — algebra valid, architecture claim closed

The next pass changed the state algebra rather than inventing another vector
mixer. A weighted disjoint-set forest stores a mutable quotient of entities and
an exact finite-group relation from each entity to its component root. `LINK`
adds a relation, `REL` returns its transitive composition, and a cycle detects a
contradiction. For `n` entities and fixed word-sized group order `q`, exact query
behavior has `sum_k S(n,k) q^(n-k)` possible states; the forest's
`O(n(log n + log q))` bits match the resulting information lower bound within a
constant factor and support inverse-Ackermann amortized access. Scaling `q`
must additionally charge the group representation and operation cost.

This is a workload advantage over an unindexed event-history scan, not an
architecture separation. Give a recurrent controller identical mutable
indirect-address RAM and it executes the same structure. More decisively,
ED-Batch Algorithm 5 already uses parent-relative non-commutative permutation
transforms and rejects incompatible cycles in a neural-systems pipeline. The
proof remains useful, but this is a classical typed data structure rather than
candidate 004. The audited closure is in
[`results/quotient-group-memory-stage0.md`](results/quotient-group-memory-stage0.md).

### Finite-field fingerprint — valid epsilon trade, not an architecture

A fixed private polynomial fingerprint bypasses the earlier learned-router
failure because every token receives the same exact Horner update. For two
explicitly delimited length-`n` strings it has zero false negatives and false
positive probability at most `(n-1)/p`, while deterministic exact one-pass
equality requires `ceil(n log2 |Sigma|)` bits. The frozen million-byte ledger
uses 234 logical bits and bounds error below `5.5e-14`.

The gained resource is accepted error plus private randomness. A randomized
digital recurrent cell performs the identical update, and Karp–Rabin already
establishes polynomial fingerprinting. It cannot replace general KV because it
retains only the declared equality invariant. The executable closure is in
[`results/epsilon-fingerprint-register-stage0.md`](results/epsilon-fingerprint-register-stage0.md).

### Measured hybrid anatomy

The [Qwen3.5 recurrent-state diagnostic](results/qwen35-hybrid-state-anatomy.md)
started from an actual hybrid rather than another invented mechanism. At a 2K
prompt boundary the frozen model used 18 MiB of FP32 DeltaNet state, 0.844 MiB
of convolution state, and 24 MiB of exact-attention KV per request. Unequal
rank allocation was causally better than equal allocation at the same
representable bytes, but every sub-dense configuration failed the fixed
`mean KL <= 0.001` gate. The byte figures were oracle representations, not
achieved GPU savings; the harness retained dense tensors and paid for SVD.

The next admission must identify a different conserved physical resource and a
capability that the above mechanisms cannot already express under the same
ledger.

### Permutation-graph screen — closed at Stage 0

The screen was executed through its CPU correctness gate and closed before GPU
rental; it is not candidate 004. Independent implementations agreed bit-for-bit
on 3,102 transitions and 13,440 queries across three seeds.

The ancestry subproblem already produced a useful negative correction before
compute: for a single-parent forest, a materialized `N^2` ancestor vector is a
cache, not the minimum exact state. Parent pointers preserve exact identity in
`O(N log N)` bits, while preorder intervals provide constant-time exact
membership by paying an `O(N)` rebuild after mutation. A 64-bit Bloom vector is
smaller than the closure at large `N`, but larger than both exact controls and
adds false positives or rebuild work. It cannot be called a free memory win.

The empirical question did not survive its fatal control. The oracle-bound
stage exposes the exact current task state and complete command: opcode, bound
arguments, and every transition payload such as the `SET` byte. A
zero-parameter flat executor therefore applies the deterministic transition
directly and achieves 100% whole-trajectory accuracy using 514 logical
task-state bytes. The learned eight-sweep candidate is upper-bounded by the
same 100%; it cannot clear a `+5` point gate. For the proposed reduced-word and
local-candidate repairs, hiding bindings either leaves an exact pointer/MAP
solution on the degree-two path/cycle topology or makes the target
information-theoretically ambiguous. This does not close arbitrary global
latent-variable tasks.

- [Stage 0 closure result](results/permutation-graph-refinement-stage0.md)
- [Frozen pre-candidate preregistration](results/permutation-graph-refinement-preregistration.md)
- [Machine-readable pre-candidate plan](manifests/permutation-graph-refinement.pre004.json)

### Frozen attention-algebra diagnostic

The six full-attention layers were also tested through the exact identity

\[
\operatorname{Attn}(q)=\nabla_s\left[
\tau\log\sum_i e^{(q^T k_i+s^T v_i)/\tau}
\right]_{s=0}
\]

and three cheaper read families: tropical/top-k selection, top-k plus one
additive tail statistic, and bounded zeroth/first-order moments. The most
optimistic variants retained full KV, exact scores, exact top-k indices, and
exact tail mass. None passed. Top-64 retained only `67.19%` mean probability
mass; its local output error averaged `15.24%`, reduced to `8.13%` by one tail
mean but still far outside the end-to-end gate.

The follow-up intervened on each of the 12 shared KV groups separately. Zero
groups passed the frozen score; even the best changed `3.125%` of top
predictions against a `1%` limit. This closes post-hoc replacement by these
algebras. A future candidate must train representations into a genuinely
different sufficient statistic and pay for its information loss explicitly;
it cannot inherit a claim from this frozen-model approximation.

### Retained permutation and switch primitives

The algebraically sound pieces from the closed permutation-graph branch have
been moved into a primitive registry rather than promoted into candidate 004.
Two-involution factorization is exact but trades parallel matchings for a
second dense transport stage; when fixed points are skipped, it ties ideal
cycle-following only on one- and two-cycles and moves more bytes on longer
cycles. It is also dominated in optimistic logical workspace by ideal
cycle-following.
A Beneš network makes every route structurally valid but cannot distinguish a
semantically correct permutation; nonlinear pair coupling reduces to the
already-established conditional butterfly/sparse-mixer family.

The reusable result is the admission boundary: learned proposal plus exact
commit can amplify capability only when the verifier checks a gold-relevant
semantic certificate more cheaply than the strongest same-information solver.
Structural validity alone is a safety mechanism.

- [Retained primitive registry and exact ledgers](RETAINED_PRIMITIVES.md)

### Feature-DAG matrix replacement — fixed ancestry rejected

A CPU-only return-to-zero pass attacked the learned dense maps in QKV/O and
FFN directly.  A five-stage nonlinear feature DAG and five controls each used
exactly 1,024 learned scalars.  The DAG reproduced a full-rank prefix transform
and won all five matched hierarchy worlds, establishing that circuit
complexity rather than rank can govern a structured transform's work.

The architecture claim failed its mixed control.  With only 25% independently
rotated Haar content, the DAG was 1.94x to 2.87x worse than the strongest
control and passed mixed noninferiority in zero of five worlds.  On pure Haar
its median NRMSE was 58.86x dense linear.  No GPU was used or justified.  The
fixed ancestry graph is closed rather than repaired with learned topology or
periodic dense layers:

- [Feature-DAG Stage-0 decision](results/feature-dag-matmul-stage0-decision.md)
- [Frozen preregistration](results/feature-dag-matmul-stage0-preregistration.md)

### Temporal-innovation dense projection — exact stack rejected

The next algebra retained an unrestricted dense matrix and cached its previous
output:

\[
W x_t = W x_{t-1} + W(x_t-x_{t-1}).
\]

This is exact and also admits an exact RMSNorm update.  It buys reduced weight
reads only when the innovation is sparse.  The composition theorem kills the
broad architecture: for a generic dense `W` and any fixed nonzero innovation,
`W Delta x` is dense with probability one.  Exact event sparsity cannot pass
through consecutive unrestricted dense maps.

The frozen checkpoint gate then tested the approximate escape.  Across five
SmolLM2-360M layers, exact changes occupied 100% of every Q/K/V/O and FFN
projection input.  Oracle int8 codes still changed about 99%; int4 reduced the
median only to 82.19% in attention and 80.73% in FFN while introducing 9.36%
and 12.74% median projected-output distortion.  Retaining the largest 10% of
raw changes produced 59.54% and 63.33% median projected-delta error.

The lower-bound additional BF16 state was `0.8203 MiB` per request at 32
layers after crediting existing K/V.  Column gathers and divergent request
supports were not charged.  Since the optimistic logical reads remain
81%-100%, no GPU kernel gate is justified.

- [Temporal-innovation decision](results/temporal-innovation-matmul-stage0-decision.md)
- [Frozen preregistration](results/temporal-innovation-matmul-stage0-preregistration.md)

### Capped dense exceptions — scaling law, not candidate

The retained shift is now a measurable law rather than a mechanism name.  For
an `O(D log D)` bulk and rank-`r(D)` dense correction, total logical work is
`O(D log D + D r(D))`.  A constant rank fraction silently restores quadratic
scaling.  Token- or layer-level full-dense fallbacks similarly need a fraction
that vanishes with width; `O(D log D)` total work requires only
`O(log D / D)` full fallbacks.

The decisive variable is the activation-weighted exception spectrum.  With
singular values `sigma_i proportional to i^-s`, squared tail energy changes
regime at `s=1/2`.  Below it the dense exception stays proportional; at the
boundary it shrinks only as a near-dense power of width; above it, fixed-error
rank can approach a width-independent cap.  A 25% flat/Haar exception
at width 4,096 needs rank 3,994 to hold total relative error to 5%, costing
1.95x dense under the ideal low-rank ledger.  At exponent 1, the same finite
gate needs rank 24 and 1.46% of dense logical work.

The broad family is already occupied by diagonal/low-rank, folded
sparse-plus-low-rank, learned structured FFNs, hierarchical matrices, and
sparse-plus-low-rank attention.  The next admission must explain why language
training produces `s>1/2` and confirm a decreasing exception fraction across
at least three widths under causal quality—not choose another structure first.

- [Dense-exception scaling result](results/dense-exception-scaling-law.md)
- [Deterministic phase table](results/dense-exception-scaling-law.json)

## Closed branches retained

- [001 — Block-routed SwiGLU](results/001-block-routed-swiglu.md)
- [002 — Attention evidence scaling](results/002-attention-evidence-scaling.md)
- [003 — Bounded semantic feedback](results/003-bounded-semantic-feedback.md)
- [Hybrid recurrent-state anatomy — no candidate admitted](results/qwen35-hybrid-state-anatomy.md)
- [Full-attention algebra separability — no family admitted](results/qwen35-attention-algebra-separability.md)
- [KV-group localization — stopped at individual-unit gate](results/qwen35-attention-group-localization.md)
- [Permutation-graph logical refinement — fatal exact control at Stage 0](results/permutation-graph-refinement-stage0.md)
- [Temporal-innovation dense projection — exact composition and frozen slack rejected](results/temporal-innovation-matmul-stage0-decision.md)
- [Dense-exception scaling law — admission criterion, not a candidate](results/dense-exception-scaling-law.md)
- [Retained permutation/switch primitives — systems library, not a candidate](RETAINED_PRIMITIVES.md)
- [Block-triangular microdepth FFN — learned one-hop correction but lost to SwiGLU](results/triangular-microdepth-lm-screen-decision.md)
- [Orbit-activated carrier chain — exact zero-byte carrier, failed H100 latency](results/orbit-activated-swiglu-h100-decision.md)
- [Gauge-exposed partner feedback — served cheaply, rejected by matched LM](results/partner-feedback-swiglu-lm-decision.md)
- [Projection-coalesced attention/FFN — beats sequential, dominated by independent parallel control](results/coalesced-attention-ffn-lm-decision.md)
- [Generator-edge FFN — full functional rank, rejected by unfused H100](results/generator-edge-ffn-h100-decision.md)
- [Self-product FFN — repeated small-model gain, rejected by fair split/packed fused H100 envelope](results/self-product-ffn-scale-fused-h100-decision.md)

No branch is to be repaired under its old claim.
