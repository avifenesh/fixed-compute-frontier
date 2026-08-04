# Cache-resident recurrent microcore — paper screen

Date: 2026-07-31  
Decision: **REJECT AS THE ACTIVE GOAL CANDIDATE; ADMIT NO MODEL OR GPU RUN**

## Question

Can a small nonlinear Transformer/SSM core remain resident in an H100 cache,
be applied repeatedly, and turn otherwise idle decode arithmetic into greater
reasoning depth without increasing the complete serving-cost vector?

There is a real conditional systems opportunity, but it is not the requested
research candidate.  The construction is weight-tied recurrent depth plus
cache-aware execution.  It can improve the parameter frontier relative to an
equal-unrolled-depth untied model; a matched looped model, however, has the same
function.  Additional visits are potentially latency-neutral only in measured
memory-bound decode cells.  Parity in prefill, sufficiently large batches, and
cache-contended concurrency cells is unproved.  The construction also contains
no raw-prose writer, persistent digital plane, or knowledge-acquisition path.

## Clean algebra

Let a physical block with a visit input `e_r` be

\[
F_\theta:\mathbb R^d\times\mathbb R^m\times\mathbb R^u
\rightarrow\mathbb R^d,
\qquad h_{r+1}=F_\theta(h_r,x,e_r),
\]

and apply it for `R` visits.  If one untied block has `P` parameter bytes and
costs `G` arithmetic operations per visit, then

| construction | unique block bytes | active block work |
|---|---:|---:|
| `R` untied blocks | `R P` | `R G` |
| one block visited `R` times | `P` | `R G` |

Thus tying can divide the physical parameter storage of this unrolled path by
`R` without reducing its sequential depth or arithmetic.  This is a genuine
parameter-efficiency identity.  Loop embeddings, controllers, state, and their
operations must be added to the second row; they are not free.  This comparison
is different from adding `R-1` visits to a one-visit baseline, which adds
`(R-1)G` work unless some other executed component is removed.

For a decode cell, a minimal roofline lower envelope is

\[
t_{decode}\;\gtrsim\;
\max\!\left(\frac{B_{HBM}}{\beta_{HBM}},
             \frac{B_{L2}}{\beta_{L2}},
             \frac{F}{\pi_{compute}},
             D_{launch/sync}\right).
\]

If a reused core remains in L2, another visit can increase `F` while adding
little HBM traffic.  Its latency is hidden only while the HBM term remains the
maximum and while L2 traffic, synchronization, and occupancy stay below their
own ceilings.  This is a conditional roofline statement, not a zero-cost
identity.

NVIDIA documents 50 MB of total L2 on H100, 64K 32-bit registers per SM, and
228 KB of shared memory per SM (at most 227 KB for one thread block).  These
figures do not imply that a model-wide core must live only in L2: a persistent
kernel may shard a core across SM-local registers/shared memory or thread-block
clusters.  They also do not guarantee that all 50 MB of L2 can be made
persistent.  The usable persisting set-aside is device- and runtime-dependent
and must be queried, while residency competes with the rest of the serving
working set.

Sources:

- NVIDIA Hopper tuning guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/
- CUDA persisting-L2 controls: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#control-l2-cache-set-aside-size-for-persisting-memory-access
- Mamba-3: https://arxiv.org/abs/2603.15569

## Small positive witness

Choose `F(h,x)=Ah+Bx` and visit it four times.  One stored pair `(A,B)` computes

\[
h_4=A^4h_0+(A^3+A^2+A+I)Bx
\]

with the same sequential depth as four untied affine layers while storing one
quarter as many affine matrices.  If `(A,B)` is already in a fast cache, visits
two through four need not reload those weights from HBM.

This proves the local storage and reuse opportunity.  It does not prove that
the tied iteration is as expressive as four independently learned layers, that
natural-language training selects a useful algorithm, or that the added visits
are latency-neutral.

## Strongest-control result

The complete candidate function, including visit inputs, is

\[
f_{cand}(x)=
F_{\theta,e_R,x}\circ\cdots\circ F_{\theta,e_1,x}(E(x)).
\]

A matched middle-cycle or looped Transformer with the same physical block,
visit count, loop embeddings, state, and execution schedule computes exactly
the same function.  Therefore

\[
\mathcal F_{cand}=\mathcal F_{matched\ looped\ control}.
\]

Cache residency changes the physical schedule, not this equality.  This does
not erase the real `P` versus `RP` parameter edge against an untied
equal-unrolled-depth baseline.  It does erase a claim that this proposal is a
new function class beyond the strongest looped control.

This is now a crowded measured family rather than an unexplored object:

- Hyperloop uses a begin/middle/end organization, loops only the middle block,
  and adds loop-level hyper-connections.  At matched unrolled depth it reports
  competitive or better perplexity and downstream accuracy with roughly half
  the parameters; its table also shows nearly matched training throughput for
  the looped and ordinary Transformer paths.  https://arxiv.org/html/2604.21254v3
- DeepLoop derives a tied-visit stability condition
  `M kappa_R (beta/alpha)^2 = O(1)` and changes the residual-scaling exponent
  for aligned recurrent visits.  https://arxiv.org/html/2607.13491v1
- Recent controlled group-word experiments show that weight tying can select a
  serial algorithmic frontier, but this is an inductive-bias result on a
  declared task family, not a free-compute theorem.
  https://arxiv.org/abs/2607.20594
- Cache-resident inference has already been developed as an execution model on
  GB-scale CPU last-level caches; its gains shrink as batch amortizes baseline
  weight loading.  https://arxiv.org/abs/2606.25353

## Resource boundary

Consider two frozen workload cells using the same served artifact.

1. **Decode, batch 1:** baseline time may be dominated by streaming unique weights
   from HBM.  A small reused block can add arithmetic before compute becomes
   binding.  A conditional latency-neutral window may exist.
2. **Prefill or high-batch decode:** one weight load is more strongly amortized across many
   tokens, while the block's matrix work is performed for every token and
   visit.  Compute is therefore more likely to bind.  Whether one extra visit
   changes latency in a finite frozen cell requires a target-shape bound or
   measurement; the roofline alone does not prove a slowdown in every cell.

Against a one-visit baseline, the active-operation coordinate in the current
serving vector rises by exactly `(R-1)G` before controller work unless another
component is removed.  That already fails componentwise parity even if wall
latency happens to be flat.  Against an equal-unrolled-depth untied baseline,
active block work can match and bytes can fall; that is a legitimate
parameter-efficiency comparison, but its full latency/traffic parity across
the frozen workload remains an empirical boundary.  Neither comparison adds
the raw-prose/digital-plane acquisition path required by the active objective.

## Goal-alignment audit

| Active requirement | Recurrent microcore |
|---|---|
| starts from raw prose | ordinary LM input only |
| autonomously extracts useful structure | no writer or identifiable statistic |
| writes a persistent digital plane | absent |
| materially improves held-out knowledge | unproved: recurrence may improve use of parametric knowledge, but it adds no stored digital knowledge and supplies no acquisition path |
| materially improves reasoning | plausible recurrent-depth inductive bias, already represented by matched looped models |
| identical complete serving cost | possible against an equal-depth untied baseline, but the complete frozen ledger is absent; extra visits fail the operation cap against a one-visit baseline unless funded |
| one local empirical uncertainty | no; natural algorithm selection and every physical workload boundary are both unresolved |

The idea fails admission to the active objective before implementation even if
its low-batch decode roofline is favorable.  This is not a theorem that looped
models cannot improve the parameter frontier; it is a scope and evidence
decision for the explicitly frozen raw-prose-to-digital-plane goal.

## Retain

Retain one design law:

> Arithmetic-intensity slack is workload-conditional **latency** slack only.
> It does not fund the independently capped operation coordinate.  Extra work
> is admissible only when equal work is removed elsewhere or the frozen
> resource contract explicitly permits it; even then, the slack ends when a
> compute, cache, synchronization, or tail-latency boundary becomes active.

This law is useful when comparing two already goal-aligned operators.  It is
not itself a raw-prose acquisition mechanism and does not authorize a GPU
microbenchmark.

## Close

Close as active candidates:

- cache-resident recurrent microcore;
- extra loop visits described as globally free;
- recurrent depth without a digital-plane acquisition path;
- any H100 run justified only by the batch-1 roofline intuition.
