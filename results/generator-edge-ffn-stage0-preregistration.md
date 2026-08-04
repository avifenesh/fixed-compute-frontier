# Generator-edge FFN — Stage 0 preregistration

## Admission

Projection coalescing failed because it forced attention selection, token
transport, and local memory to share columns.  This branch keeps the strong
independent parallel attention control unchanged and reuses generators only
inside the FFN, where every constructed interaction receives its own down
column.

## Frozen algebra

For `M` learned generators and two fixed directed permutations `p1,p2`:

```
a = X A^T                                      # width M
f1_i = SiLU(a_i) * a_p1(i)
f2_i = SiLU(a_i) * a_p2(i)
Y = [f1, f2] B^T                               # 2M independent readouts
```

Freeze

```
p1(i) = i + 1 mod M
p2(i) = 5 i + 3 mod M.
```

For the power-of-two widths used here, both are permutations and the two
outgoing edges never coincide.  Every generator is used twice as a gate and
twice as a value.  Feature order is fixed; there is no router, learned graph,
edge coefficient, bias, or auxiliary loss.

## Exact FFN ledger

Ordinary width-`M` SwiGLU:

```
Wg: M D, Wu: M D, Wdown: D M       total 3 D M
```

Generator-edge:

```
A: M D, B: D (2M)                   total 3 D M
```

Both require the same dense learned weights and dense projection MACs per
token.  Both evaluate `M` SiLUs.  Generator-edge performs `2M` rather than `M`
elementwise products and gathers two fixed permutations.  It materializes a
`2M` feature tensor in the reference path; an eventual served kernel must
generate edges inside down-GEMM tiles to avoid claiming that workspace is
free.

At `D=384,M=1024`, both use 1,179,648 FFN projection weights/MACs per layer.
Attention remains the independent parallel baseline, so the whole dense block
ledger is 1,572,864.

## What capability is being purchased

SwiGLU creates `M` independently read interaction atoms from `2M` independently
projected linear factors.  Generator-edge creates `2M` independently read
interaction atoms from `M` reusable factors.  It trades arbitrary factor pairs
for twice as many value/readout vectors when useful concepts participate in
multiple relations.

This also removes SwiGLU's exact per-unit `up/down` scale redundancy.  Because
each generator appears through non-homogeneous SiLU in outgoing edges and
linearly in incoming edges, generic generator rescaling cannot be absorbed by
the readouts.  The claim must be checked by sampled functional-Jacobian rank;
raw parameter count alone is insufficient.

## Exact-budget controls

All controls run inside the same one-norm parallel attention/FFN block.

1. `parallel_swiglu`: ordinary independent width-1,024 SwiGLU.
2. `parallel_self_product`: width 1,536 with
   `f_i=SiLU(a_i)*a_i`.  Its one input and one output projection also total
   `3DM`.  This tests whether simply buying more identifiable self-gated atoms
   beats relational reuse.
3. `parallel_duplicate_edge`: the generator-edge parameterization, but both
   routes use `p1`.  Its paired down columns collapse functionally to their
   sum; it tests whether two distinct relations, not a wider optimizer tensor,
   create the gain.
4. `parallel_generator_edge`: the frozen two-permutation candidate.

## Prior-art boundary

Multiplicative/polynomial networks, FFNs as key-value memories, SwiGLU, and
Masked GLUs are populated directions.  Masked GLU is especially adjacent: it
derives several gate/value streams from a shared weight matrix using learned
binary masks and a specialized kernel.  The present hypothesis is narrower:
one full dense generator bank, no mask bits or learned routing, a fixed
degree-two relation graph, and an independent output vector for every edge.

No novelty claim is allowed from a positive screen alone.

## Stage 0 gates

Before language training:

1. Exact parameter/MAC equality is reproduced by code.
2. The graph has exactly `2M` distinct directed edges and in/out degree two.
3. Forward, backward, and a separate reference implementation agree.
4. At a deterministic small double-precision probe, candidate functional
   Jacobian rank must exceed ordinary SwiGLU rank at the same 3DM raw
   parameters.  Report the self-product and duplicate controls too.
5. Unfused H100 whole-model latency must be no more than 1.05x the parallel
   SwiGLU control at batch/token cells `(1,1)`, `(1,512)`, `(8,512)`, and
   `(32,512)`.  This is a feasibility gate, not a production claim.

## Capability gate if Stage 0 passes

Use the frozen 50M-token stream and optimizer protocol from the coalesced
screen, with seed 1907 and evaluations at 10M/50M tokens.  Promotion requires:

- candidate terminal NLL at least 0.05% below parallel SwiGLU, with a wholly
  favorable paired 95% interval;
- candidate terminal NLL at least 0.025% below self-product and duplicate-edge
  controls;
- candidate early NLL noninferior to parallel SwiGLU within 0.05%;
- finite training and activation diagnostics;
- replacing `p2` by `p1` after training hurts by at least 0.01%, with a wholly
  favorable paired interval.

One seed can reject or justify replication; it cannot prove A-E dominance.

## Fatal interpretation

- If self-product wins, identifiable atom count matters but graph reuse does
  not; close this graph claim.
- If duplicate-edge matches the candidate, distinct relations are not causal.
- If candidate improves NLL but fails H100, retain the algebra only and do not
  call it fixed served cost.
- No learned topology, more degree, edge scalars, or distillation rescue is
  allowed under this branch.
