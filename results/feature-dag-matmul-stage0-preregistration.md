# Feature-DAG matrix replacement — Stage 0 preregistration

Status: **frozen before observing learned results**  
Date: 2026-07-26  
Stage: CPU-only algebra and learnability gate; no GPU rental

## Claim under test

A dense learned projection computes every output feature independently.  A
feature DAG can instead compute recurring partial features once and reuse them
through later nodes.  This can improve capability per active scalar operation
only when the target transform has reusable compositional structure.

This gate does **not** claim a universal replacement for dense matrices.  An
arbitrary `D x D` linear map has `D^2` continuous degrees of freedom.  The Haar
slice below is the mandatory boundary showing the price of restricting the
computation circuit.

The candidate node is

\[
z'_i=a_i z_i+b_i z_{i-s}
     +c_i\tanh(d_i z_i+e_i z_{i-s}+f_i),
\]

where the five stages use offsets `(1, 2, 4, 8, 16)`.  Missing predecessors
are exactly zero.  A learned diagonal affine readout follows the fifth stage.
Q, K, V, O, and FFN roles are not separately simulated here; the gate tests
the common channel-transform primitive that would replace their learned dense
maps.  Attention's dynamic `Q K^T` and `A V` products are outside this claim.

## Exact algebra control

For the inclusive-prefix matrix `P`, `P_ij = 1[j <= i]`, the recurrence

\[
p_0=x_0,\qquad p_i=p_{i-1}+x_i
\]

uses `D-1` additions although `P` is full rank and contains `D(D+1)/2`
nonzero entries.  The executable self-test must recover the same prefix with
the five offset stages at `D=32` to absolute error below `1e-6`.

This is a witness that matrix rank is not the relevant lower bound; repeated
subexpression or straight-line-program complexity is.  It is not evidence
that trained language-model matrices have low circuit complexity.

## Frozen dimensions and data

- width `D=32`;
- train/validation/test examples: `4096 / 1024 / 2048`;
- independent Gaussian inputs for every split;
- five worlds: `31013, 31019, 31033, 31039, 31051`;
- one deterministic initialization per `(world, target, variant)` in the
  screen; initialization variance is required only after a pass;
- Adam, `1,200` steps, batch `256`, validation every `40` steps;
- validation-selected checkpoint; no test-selected step or hyperparameter.

Each world contains four separately trained target transforms:

1. `prefix`: normalized inclusive prefix sums;
2. `hierarchical`: a frozen random instance of the five-stage nonlinear
   feature-DAG family;
3. `mixed`: a standardized `0.75 * hierarchical + 0.25 * Haar` target;
4. `haar`: an independently sampled Haar-orthogonal linear map.

Hierarchy coefficients and Haar matrices vary independently by world.  Target
normalization is fitted on the training split and applied unchanged to
validation and test.

## Equal learned-scalar controls

Every variant has exactly `1,024` trainable scalar parameters:

| variant | transform | learned scalars |
|---|---|---:|
| `dense_linear` | unrestricted `32 x 32` linear map | 1,024 |
| `dense_ffn` | `32 -> 16 -> 32` with `tanh` | 1,024 |
| `low_rank` | rank-16 linear factorization | 1,024 |
| `butterfly` | sixteen learned `2 x 2` butterfly stages | 1,024 |
| `lookup` | four 3-bit address tables, eight `32`-vectors each | 1,024 |
| `feature_dag` | five nonlinear offset stages plus diagonal affine readout | 1,024 |

Parameter equality is not operation equality.  The result must separately
report scalar parameter reads, multiplications, additions, comparisons, table
lookups, and transcendental evaluations per example.  Fixed wiring indices
and implicit schedules are also reported.  Logical operation counts are not a
GPU-speed claim.

## Frozen primary metric and decision

The metric is test normalized root-mean-square error (`NRMSE`), reported for
every target and world.  Lower is better.  The strongest control for a target
is selected only after each control's own validation-selected checkpoint.

The reuse mechanism passes its Stage-0 preference gate only if all hold:

1. prefix median NRMSE is at most `0.05`;
2. `feature_dag` beats every control on hierarchical test NRMSE in at least
   four of five worlds;
3. its median relative hierarchical NRMSE improvement over the strongest
   control is at least `10%`;
4. on `mixed`, it is within `5%` relative NRMSE of the strongest control in at
   least four of five worlds.

Promotion remains blocked unless its median Haar NRMSE is also within `10%` of
`dense_linear`.  Therefore the possible decisions are:

- `reject`: the reuse mechanism misses any structured preference gate;
- `mechanism_only`: it passes the structured gates but fails the Haar boundary;
- `pre_candidate`: it passes both structured and Haar gates.

Neither `mechanism_only` nor a single-initialization `pre_candidate` justifies
an H100.  The next gate would first need initialization confirmation and a
real jointly trained language-model proxy.  Any later serving claim requires
a fused implementation and direct H100/H200 traffic and latency measurement.
