# Permutation-graph logical refinement — closed at Stage 0

Status: **closed before GPU or training**  
Date: 2026-07-26  
Candidate number: **none**

## Outcome

The exact harness passed, but its mandatory flat exact control makes the
refinement superiority gate mathematically unreachable.

No GPU was rented. No learned result was observed. This is a pre-training
algebraic rejection, not a failed optimization run.

## Executed Stage 0 evidence

Two independently implemented transition systems agreed bit-for-bit across:

- 3,072 randomized state transitions: 1,024 for each seed `17`, `29`, and `43`;
- 30 directed semantic transitions covering every opcode, including accepted
  two-switches, their inverses, and rejected forms;
- 13,440 ancestry, 1/8/32-hop, and path-aggregate queries.

The exact 4,096-byte arena and frozen ancestry allocation table were validated
separately by their focused accounting tests.

The repository suite passed 48 tests, including all involutions through
`N=6`, all parent assignments through `N=4`, two-switch inverses, forest
acyclicity, node-renaming equivariance, Bloom no-false-negative checks, and
independent interpreter agreement.

Evidence:

- [Exact gate report](permutation-graph-stage0-exact-gate.json)
- [Machine-readable decision](permutation-graph-refinement-stage0-decision.json)
- [Tested executor](../experiments/permutation_graph_stage0.py)
- [Independent reference](../experiments/permutation_graph_reference.py)
- [Ancestry representations](../experiments/permutation_graph_ancestry.py)

## Fatal control

The proposed oracle-bound refinement stage exposes the exact current task
state and a complete oracle command `C_t`: opcode, bound node arguments, and
every transition-determining payload such as `SET`'s byte value. The gold
transition is deterministic:

\[
S_t=F(S_{t-1},C_t).
\]

The required flat exact control is therefore:

```text
state = exact_initial_state
for oracle_command in stream:
    state = exact_transition(state, oracle_command)
    emit exact_snapshot(state)
```

It ignores corrupted `X`, candidate sets, and all refinement evidence. By
induction it produces every gold snapshot exactly. Stage 0 executed this
control against the independent interpreter and observed 100% whole-trajectory
agreement.

The control uses zero learned parameters and only 514 logical task-state
bytes:

| State | Bytes |
|---|---:|
| `Z: uint16[64]` | 128 |
| two `uint16[64]` involutions | 256 |
| `parent: uint16[64]` | 128 |
| `status: uint16[1]` | 2 |
| **Total** | **514** |

The candidate's accuracy is upper-bounded by 100%. Consequently,

\[
Acc(candidate)-Acc(exact\ control)\le 0,
\]

while the frozen gate requires at least `+5` percentage points. Training cannot
change this inequality.

## Hiding the final binding does not rescue the claim

Suppose the final node binding is a hidden target rather than visible oracle
input.

1. For the proposed reduced 8-hop-word repair, the visible anchor and relation
   labels determine the target with eight pointer gathers—less work than eight
   full matching sweeps.
2. For at most four operand roles with four candidates each, the same-information
   exact/MAP control enumerates at most `4^4 = 256` assignments. Whether its
   complete typed-operation cost fits a future envelope remains unproven and
   would require a separate frozen ledger.
3. If the visible information does not uniquely determine the binding, two episodes can
   have identical visible tensors and different gold bindings. Conditional
   entropy is positive, so universal exact recovery is impossible for every
   model.

After treating fixed points as unary and ignoring their self-loops, the union
of two involutions has maximum degree two: it is a collection of paths and even
alternating cycles. These concrete repaired tasks do not create a computation
unavailable to a flat control with the same state and primitives. This result
does not close arbitrary global latent-variable tasks; those would require a
new workload, resource proof, and preregistration.

## Ancestry result retained

The ancestry analysis remains useful independently. At `N=1024`, exact parent
traversal uses 2,048 bytes and exact DFS intervals use 6,144 bytes, while the
approximate Bloom64 representation uses 10,272 bytes. An ancestor vector is
smaller than a 133,120-byte materialized closure cache, but not smaller than
the strongest exact representations. It trades query/update work and error;
it is not a free memory gain.

## Preregistration gaps found

- The JSON's corruption arrays omit an explicit zero-severity cell even though
  the prose requires a clean control.
- “Complete trajectory accuracy” must mean all 1,024 snapshots correct for an
  episode, not average per-snapshot accuracy. A 99% whole-episode target
  requires per-step reliability on the order of `99.999%`.

Neither correction can repair the fatal exact-control dominance.

## Decision

Close the branch before GPU rental. Do not implement or train the eight-sweep
refiner under this claim. Return to zero and require the next algebra to expose
a capability that a cheaper same-information exact or recurrent control cannot
already compute.
