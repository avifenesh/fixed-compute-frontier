# T73 guarded mechanism worlds — independent audit

Date: 2026-08-02  
Status: **THRESHOLD SUBTHEOREM PASSES; COMPLETE GRAMMAR REQUIRES REVISION BEFORE ENUMERATION; NO NEURAL OR NATURAL-PILOT ADMISSION**

## Verdict

T73 repairs T72's surface-rollout error and contains a correct adaptive threshold
primitive. It is not yet a frozen complete-class test. Unknown/coincident leaf
maps can erase the threshold signal, syntax-level leaf changes are not invariant
under the claimed behavioral quotient, and Stage A criterion 6 is circular and
cannot rule out a compressed memorized policy. The same-backbone comparison also
does not yet isolate a lifetime objective from ordinary full-lifetime BPTT,
extra action supervision, or changed on-policy data.

Keep the no-code/no-run status. After the corrections below, exact enumeration
may decide whether a *specific tiny composite class* has the required `2x` gap.
The known-branch theorem alone does not admit it.

## 1. T73.1 is correct but narrowly scoped

For `m>=3`, there are `m-1` thresholds. A query at `u` returns the comparison
`t<=u`. Binary search therefore needs

\[
q_{adapt}=\lceil\log_2(m-1)\rceil.
\]

Adjacent thresholds `t,t+1` differ only at `u=t`, so an exact nonadaptive set
must contain every `u=1,...,m-2`; those points suffice. Hence

\[
q_{nonadapt}=m-2.
\]

At `m=33`, `31/5=6.2` is correct. State `m>=3`; at `m=2` both counts are zero
and the ratio is undefined.

This proof assumes two known branch outputs, or at least one frozen payload
witness whose two known outcomes label the branches. With unknown affine leaf
maps, one observed vector does not say “left” or “right” until the maps are
calibrated. Equal leaf maps make the threshold behaviorally absent. Composite
trees add map calibration, leaf matching, redundant guards, and structure
uncertainty. Therefore T73.1 neither lower-bounds nor proves a ratio for the
complete class.

There is, however, a valid constructive repair for the **complete depth-one
class with two unknown distinct full-affine branches**. Probe an affine basis
`0,e_1,...,e_d` at `u=0` to identify `f_0` and at `u=m-1` to identify `f_1`.
Because the maps are distinct, compute an `x_*` with
`f_0(x_*) != f_1(x_*)`, then binary-search the threshold. Hence

\[
q_{adapt}\le 2(d+1)+\lceil\log_2(m-1)\rceil.
\]

For a nonadaptive schedule, let `X_u` be the payloads probed at selector `u`.
If an interior `X_t` does not affinely span `F_p^d`, two distinct affine maps
can agree on all of `X_t`; the adjacent-threshold worlds `(t,f_0,f_1)` and
`(t+1,f_0,f_1)` then agree on the complete schedule. Thus each interior selector
needs at least `d+1` affinely independent payloads. The endpoints need the same:
when `t=1`, `f_0` occurs only at `u=0`, and when `t=m-1`, `f_1` occurs only at
`u=m-1`. Therefore

\[
q_{nonadapt}=m(d+1),
\]

with equality achieved by probing an affine basis at every selector. This gives
a proved `>2` gap at `m=33` for all `d>=1`. It does not extend automatically to
unknown multi-level trees.

Exact correction:

> T73.1 proves a strict separation only for the known-distinguishable-branch
> threshold subfamily. In the complete class, first quotient coincident adjacent
> leaves and redundant guards, charge all payload-map calibration, and compute
> adaptive and nonadaptive minimax counts over the resulting behavioral
> hypotheses. Failure to retain a `2x` ratio closes that candidate size.

If the threshold primitive is meant to survive by construction, freeze a public
calibration set `X_cal` on which every adjacent behavioral leaf is distinguishable
and charge the probes needed to identify its outcomes. Otherwise do not describe
T73 as a “provable active-learning successor” before the composite enumeration.

## 2. Surface and rollout closure pass after one clarification

One shared lifetime state encoding is the correct repair. Formally use

\[
S(u,x)=(\pi u,S_xx),\qquad \widetilde P=S\circ P\circ S^{-1},
\]

where `pi` permutes selector coordinates but not ordered selector values, and
`S_x` is one invertible affine-monomial payload map used on input and output.
Then

\[
\widetilde P^h=S\circ P^h\circ S^{-1}
\]

and the displayed affine rollout formula is correct because `u` and its selected
leaf remain fixed and no change occurs inside the rollout. Add this equation to
remove any residual input/output-binding ambiguity.

Sequential `PROBE`s can reproduce an `h`-step rollout in `h` charged actions.
The interaction budget must prevent this from becoming an unreported evaluation
advantage; the logarithmic augmented-matrix solver remains the proper execution
control.

## 3. Behavioral quotient and change process need a canonical object

Static learnability is straightforward on the finite domain: complete one-step
behavior determines every static rollout under the shared conjugacy. The change
law is the unresolved part. “Replace one leaf” is syntax-dependent. Two trees
can compute the same piecewise-affine function but contain different duplicated
or redundant leaves; sampling a syntax leaf can then induce different future
change distributions. That contradicts both “equivalent trees receive identical
score” and “the identifiable object is the piecewise-affine transition.”

Use one of these exact fixes:

- reduce every tree to a frozen canonical behavioral partition and target one
  canonical region; or
- define a world as the current transition **plus its complete change kernel**,
  then quotient and split these dynamic worlds rather than static functions.

In either case, require observational equivalence over the complete pre/post
change transcript to refine private-score equivalence. Tree-shape and factorial
holdouts are valid only after quotienting; affine maps, redundant guards, and
different trees will otherwise leak across splits.

## 4. Protected change is constructible with a stronger contract

The central construction is sound. If a changed canonical region `R_*` and an
unchanged nonempty region `R_j` exist, freeze selector representatives and a
payload battery `X`. Define

\[
Q_{changed}=\{(u,x):u\in R_*,\ \Delta(x)\ne0\},
\]

and

\[
Q_{protected}=\{(u,x):u\in R_j,\ j\ne *,\ x\in X\}.
\]

At least two **nonempty canonical regions**, not merely two syntax leaves,
guarantee protected selectors. Stage A must freeze minimum region and query-set
sizes, require `Delta` to be nonzero on the changed battery, and define the
allowed change generator before enumeration. Changing a leaf into an equivalent
map is a null change and must be quotiented out. A paired complete-state swap is
the correct candidate-neutral intervention.

## 5. The seven Stage A criteria do not all pass as written

1. **Observable refines score:** achievable for static worlds under the shared
   conjugacy; for changed worlds it requires the canonical/dynamic-world fix.
2. **Quotient-disjoint splits:** achievable, but syntax holdouts may become empty
   or much smaller after reduction. Freeze partitions of quotient classes, not
   raw trees.
3. **Composite `2x` gap:** not implied by T73.1. Unknown-map calibration can
   reduce the ratio below two. Exact DP must decide it.
4. **Changed/protected sizes:** achievable with the canonical-region and frozen
   change-generator conditions above.
5. **`10x` private rollouts:** achievable; charge any sequential-probe simulation
   and prove no change occurs inside a target rollout.
6. **Table exceeds ledger:** circular and non-diagnostic. The neural parameter
   count/precision is not frozen before Stage A, a naive table can exceed the
   ledger while a compressed decision DAG or binary-search/system-ID program
   fits easily, and no finite test can distinguish that program from a learned
   “reasoning” algorithm. Replace this pass/fail gate with an explicit ledger for
   a naive table, minimized decision DAG under a declared compressor, and the
   shortest supplied classical solver. Freeze the neural ledger before comparing
   them. Treat compact procedural recognition as legitimate, not memorization.
7. **Controls executable:** too weak to be a gate. Name exact algorithms,
   implementations, budgets, and required evaluation outputs. “Program
   induction” is not one reproducible control.

There is also a practical tension: exact minimax DP over a complete class is
exponential in the number of behavioral hypotheses, while making a verbatim
table exceed even a tiny neural parameter ledger pushes that class upward. T73
must report the class count, reachable version-space count, DP memory/time upper
bound, and all table/solver code sizes *before* declaring a size enumerable. Do
not let the enumerator choose the model ledger after seeing these quantities.

## 6. Same-backbone Stage B is not yet a one-variable causal test

An ordinary recurrent learner trained with per-step losses over an unrolled
lifetime and full BPTT already receives future gradients through earlier state.
Conversely, a detached “next-step” learner tests a credit-path intervention, not
merely a different training unit. Active query learning may also give the
lifetime system RL rewards or exact-policy information absent from the
next-step system. On-policy policies generate different interaction data, so
“same interaction distribution” cannot simultaneously hold without an
off-policy corpus.

Freeze a factorial comparison:

1. same lifetime trajectories, per-step answer/action losses, and full BPTT;
2. the same condition with state detached at declared boundaries;
3. full BPTT plus the additional regret, action-cost, calibration, change, and
   retention terms called the lifetime objective; and
4. behavior cloning from the exact minimax query policy as a strong supervision
   control, with its oracle calls charged.

Use fixed off-policy training lifetimes to isolate the loss/credit intervention,
then separately report on-policy end-to-end learning. Match tokens, targets,
optimizer steps, unroll length, gradient work, tuning budget, and seeds. Without
this design, a positive result belongs to an objective/data/supervision package,
not specifically to future-competence training.

## 7. Prior-art boundary and resulting-model path

The threshold separation is binary search. The composite task is exact learning
from membership queries, adaptive decision-tree learning, and piecewise-affine
system identification. Direct primary controls include
[generalized binary search](https://arxiv.org/abs/0910.4397),
[adaptive exact decision-tree learning](https://proceedings.mlr.press/v98/bshouty19a.html),
and classical
[piecewise-affine/hybrid identification](https://doi.org/10.1016/S0005-1098(02)00224-8).
Program-library induction such as
[DreamCoder](https://arxiv.org/abs/2006.08381) is also a direct structural
control. The 2026
[agentic-automata benchmark](https://arxiv.org/abs/2606.16576) already uses
membership/equivalence interaction and classical learners to expose LLM planning,
evidence-integration, and hypothesis failures.

T73 acknowledges this classical status, so novelty is not the defect. The only
new empirical claim available is that a particular lifetime-training
intervention makes one neural backbone amortize and transfer this combined
algorithm better than its matched controls.

A positive tiny result inside one known guarded grammar does **not** by itself
earn a natural-domain pilot. It can earn a second, non-isomorphic synthetic
grammar. A natural pilot is justified only after the same frozen updater, without
family-specific retraining, transfers across that successor grammar; the causal
objective comparison above survives; and the system beats the strongest matched
adaptive/context/retrieval/test-time-training controls on acquisition, revision,
retention, and complete cost. Even then the pilot is external validity, not an
intelligence claim.

## Exact disposition

- Accept the T73.1 counts for the known-branch subfamily, with `m>=3`.
- Keep all enumeration and training held.
- Canonicalize behavioral regions or quotient complete change kernels.
- Strengthen the protected-change generator and split definitions.
- Replace circular Stage A criterion 6 and operationalize criterion 7.
- Require exact composite minimax enumeration; the subtheorem is insufficient.
- Replace the two-arm neural comparison with the factorial causal design.
- Do not admit a natural pilot from guarded-affine Stage B alone.
