# T73 guarded mechanism worlds — exact active discovery with unknown local dynamics

Date: 2026-08-02  
Status: **INDEPENDENTLY REVISED; COMPLETE DEPTH-ONE THEOREM RETAINED; MULTI-LEVEL AND ALL RUNS HELD**

## 0. Result and scope

T72 proves that pure affine dynamics cannot test active learning because a
fixed basis schedule is minimax. T73 adds one hidden ordered guard. Unlike the
initial known-branch witness, the complete depth-one class with two unknown
distinct affine branches still has a constructive order-one adaptive advantage:

\[
q_{adapt}\le 2(d+1)+\lceil\log_2(m-1)\rceil,
\qquad
q_{nonadapt}=m(d+1).
\]

At `m=33,d=1`, adaptive acquisition needs at most `9` probes while every exact
fixed schedule needs `66`.

This is a theorem about active system identification, not a model result and
not component novelty. It admits only a future depth-one CPU calibration after
the remaining finite ledger is frozen. It does not establish the ratio for
multi-level trees, admit neural training, or earn a natural pilot.

## 1. Complete depth-one world

Fix `m>=3`, a prime field `F_p`, and payload width `d>=1`. A world samples:

- threshold `t in {1,...,m-1}`;
- two distinct affine maps
  `f_0(x)=A_0x+b_0` and `f_1(x)=A_1x+b_1` on `F_p^d`; and
- a later canonical-region change process defined in Section 5.

The transition is

\[
P(u,x)=
\begin{cases}
(u,f_0(x)),&u<t,\\
(u,f_1(x)),&u\ge t,
\end{cases}
\qquad u\in\{0,\ldots,m-1\}.
\]

Selector rank is an explicitly ordered typed value. A lifetime encoding is

\[
S(u,x)=(u,S_xx),
\qquad
\widetilde P=S\circ P\circ S^{-1},
\]

where one invertible affine-monomial `S_x` is used on payload input and output.
Therefore

\[
\widetilde P^h=S\circ P^h\circ S^{-1}.
\]

Ordered selector values may not be arbitrarily permuted; that would remove the
order required by binary search. A multi-selector successor may permute
selector coordinate names but not their public rank semantics.

The interface is:

```text
PROBE u,x       -> FEEDBACK P_s(u,x)
PREDICT u,x     -> private scored P_s(u,x)
ROLLOUT u,x,h   -> private scored static P_s^h(u,x)
```

Probes are chosen and charged. No change occurs inside one rollout. Repeating
`PROBE` can simulate a rollout in `h` environment actions, so interaction and
execution work are both charged.

## 2. Observable quotient

A dynamic world includes its current transition, change time, and complete
change kernel. Define

\[
w\sim_{obs}w'
\Longleftrightarrow
\forall\pi,\quad
\mathcal L(\tau_{obs}\mid w,\pi)=
\mathcal L(\tau_{obs}\mid w',\pi),
\]

and `w ~score w'` when every legal private query has the same target. The
finite protocol is valid only after it proves

\[
w\sim_{obs}w'\Longrightarrow w\sim_{score}w'.
\]

Coincident branch maps make the threshold behaviorally absent and are excluded
from the complete depth-one theorem. In successor trees, coincident adjacent
regions and redundant guards must be quotiented before splitting or scoring.
Syntax and leaf names are never targets.

## 3. T73.1 — known-branch threshold separation

With two known distinguishable branch outcomes, there are `m-1` possible
thresholds. Binary search identifies `t` in

\[
\lceil\log_2(m-1)\rceil
\]

adaptive probes. Adjacent thresholds `t,t+1` differ only at selector `u=t`, so
an exact nonadaptive schedule must query every `u=1,...,m-2`, and those probes
suffice. Its count is `m-2`.

For `m=33`, the ratio is `31/5=6.2`. This subtheorem is useful but not the
complete unknown-mechanism result.

## 4. T73.2 — complete depth-one unknown-branch separation

### Adaptive construction

For every legal threshold, `u=0` selects `f_0` and `u=m-1` selects `f_1`.
Probe the affine basis

\[
0,e_1,\ldots,e_d
\]

at each endpoint. This identifies both maps in `2(d+1)` probes. Because the
maps are distinct, their difference is a nonzero affine map, so the learner can
compute a witness `x_*` with

\[
f_0(x_*)\ne f_1(x_*).
\]

Binary-search `t` using that witness. Hence

\[
q_{adapt}\le
2(d+1)+\lceil\log_2(m-1)\rceil.
\]

The computation used to find `x_*` is charged even though it requires no
additional environment probe.

### Exact nonadaptive count

Let `X_u` be the payloads scheduled at selector `u`. If an interior `X_t` does
not affinely span `F_p^d`, choose two distinct affine maps that agree on every
point in `X_t`. The adjacent-threshold worlds with thresholds `t` and `t+1`
then agree on the entire schedule. Thus each interior selector needs at least
`d+1` affinely independent payloads.

The endpoints also need `d+1`: when `t=1`, `f_0` is observable only at `u=0`;
when `t=m-1`, `f_1` is observable only at `u=m-1`. Therefore every exact fixed
schedule needs at least

\[
m(d+1)
\]

probes. Querying an affine basis at every selector achieves the bound, so

\[
q_{nonadapt}=m(d+1).
\]

For `m=33`, this exceeds twice the adaptive upper bound for every `d>=1`.

This theorem does not extend automatically to multi-level trees; their
behavioral quotient, map calibration, redundant guards, and minimax decision
trees remain separate work.

## 5. T73.3 — quotient-invariant change and protected behavior

The depth-one behavioral partition has two nonempty canonical regions:

\[
R_0=\{u:u<t\},
\qquad
R_1=\{u:u\ge t\}.
\]

The change kernel selects a canonical region `R_c` and replaces its affine map
with a behaviorally different affine map. It does not select a syntax leaf.
The other region remains unchanged. Null replacements are excluded.

For old/new changed-region maps define

\[
\Delta(x)=f_c'(x)-f_c(x).
\]

Freeze a payload battery `X` containing a point where `Delta(x)!=0`, and freeze
selector representatives from both regions. Then

\[
Q_{changed}=\{(u,x):u\in R_c, x\in X, \Delta(x)\ne0\},
\]

while queries using the unchanged region form `Q_protected`. Both are nonempty
by construction. Score acquisition regret, calibration around the first
contradiction, recovery on `Q_changed`, maximum loss on `Q_protected`, and
restoration of the old map.

A paired complete-state swap between worlds differing only in the changed
canonical region is candidate-neutral. Component-level latent edits are not.

## 6. Exact execution control

For fixed selector `u`, let its selected map be `(A,b)`. Then

\[
P^h(u,x)=
\left(u,A^h x+\sum_{i=0}^{h-1}A^i b\right).
\]

An augmented `(d+1)x(d+1)` matrix evaluates this in `O(d^3 log h)` field
operations. A learned model is compared with this exact solver, not a weak
unrolled affine baseline. `10x` private horizons add execution depth but no new
identification information.

## 7. Frozen depth-one CPU-calibration requirements

No enumerator exists yet. A future paper freeze may choose a candidate such as
`m=33,p=3,d=1`, for which the static unbound world count before surface/change
quotienting is

\[
(m-1)M(M-1),
\qquad M=p^{d(d+1)},
\]

or `2304` worlds at those values. Before code, record exact class count,
reachable-version-space count, enumeration/DP memory and time upper bounds,
and the complete change-world count.

The CPU artifact must:

1. prove observable equivalence refines score equivalence for the complete
   canonical change process;
2. enumerate quotient-disjoint development and decisive partitions;
3. reproduce the T73.2 adaptive construction and exact nonadaptive count;
4. verify frozen changed/protected battery sizes for every allowed change;
5. compute conditional sufficient-state or rate-distortion bits, transcript
   bits, canonical guarded-map bits, binding bits, and served precision;
6. report—not gate on—the code/parameter/storage/work ledger for a naive table,
   a minimized decision DAG under a declared compressor, the shortest supplied
   classical solver, and later the neural model; and
7. implement exact basis-plus-binary-search, fixed-basis-every-selector,
   version-space, full-context, and random-query controls with frozen outputs
   and budgets.

Compact procedural recognition is a legitimate solution. The goal is not to
distinguish an algorithm compiled in weights from “reasoning” by definition.
This depth-one calibration still cannot admit neural training by itself because
it lacks held-out composition.

## 8. Successor composition gate

A multi-level guarded tree is admitted only after a separate paper proves or an
exact feasible DP computes its full adaptive/nonadaptive minimax ratio after
quotienting redundant regions. Splits are made over behavioral dynamic worlds,
not raw ASTs, and hold out tree shapes, selector combinations, coefficients,
bindings, horizons, and canonical-region changes.

Passing depth one earns work on this successor. It does not imply the successor
ratio or transfer.

## 9. Factorial neural causal design

Only a passing depth-one calibration plus a non-collapsing successor grammar
can freeze neural work. The first learned comparison is factorial:

1. one recurrent backbone trained on fixed off-policy lifetimes with per-step
   answer/action losses and full BPTT;
2. the identical condition with state detached at declared boundaries;
3. full BPTT plus the additional regret, action-cost, calibration, change, and
   retention terms of the lifetime objective; and
4. behavior cloning from the exact minimax policy, with oracle generation calls
   charged.

Match tokens, supervised targets, optimizer steps, unroll length, gradient
work, data order, tuning trials, and seeds. This off-policy factorial isolates
loss and credit interventions. Then separately report on-policy end-to-end
learning, where policies necessarily generate different data.

Full context, byte-matched memory/retrieval, test-time gradients,
self-consolidation, exact guarded-system identification, and minimized
decision-DAG controls remain required. A structured factor state enters only
after a frozen generic-state failure identifies a representation or execution
defect.

## 10. Resulting-model boundary

T73 is classical binary search plus piecewise-affine system identification.
Its legitimate empirical question is whether lifetime training makes one
neural backbone amortize this combined algorithm across held-out compositions
better than matched controls.

A positive guarded-affine result earns a second non-isomorphic synthetic
grammar, not a natural pilot. A natural pilot is considered only if the same
frozen updater transfers without family-specific retraining across that second
grammar and still wins acquisition, revision, retention, and complete cost.

No CPU enumeration, neural training, GPU use, or rental is admitted in this
round.

Independent review:
[`guarded-mechanism-worlds-t73-independent-audit.md`](guarded-mechanism-worlds-t73-independent-audit.md).
