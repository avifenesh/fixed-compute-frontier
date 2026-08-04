# T72 finite mechanism grammar — independent audit

Date: 2026-08-02  
Status: **REJECT V0 AS AN ADMISSION WORLD; RETAIN ONLY AS AN AFFINE CALIBRATION AFTER CORRECTION; NO CODE OR RUN**

## Verdict

T72 has the right proof-first discipline, and the affine rollout identity and
finite-class query bounds are useful controls. The present v0 is not yet a
closed, identifiable admission world for developmental learning.

Four issues are blocking:

1. Independent input and output bindings are not a conjugacy. One-step probes
   then need not identify rollout targets.
2. The behavioral quotient includes private scored targets rather than only
   learner-observable evidence. Observational indistinguishability and scoring
   equivalence must be defined separately.
3. `Q_alias` cannot detect a behaviorally equivalent reparameterization; T72.1
   proves that no behavioral query can do so.
4. For the full affine class, a fixed basis schedule is already minimax: `d+1`
   probes identify `A,b`. Adaptivity has no acquisition advantage, let alone
   `2x`. A bounded-AST subset might have an adaptive advantage, but T72 supplies
   no construction or proof that it does.

The Stage A enumerator should remain held until these definitions are rewritten.

## 1. Surface closure and identifiability

Let `S_in,S_out` denote the proposed independent affine-monomial input and output
encodings. A one-step surface map is

\[
F=S_{out}\circ P\circ S_{in}^{-1}.
\]

But its surface iterate is

\[
F^2=S_{out}P(S_{in}^{-1}S_{out})P S_{in}^{-1},
\]

which is generally not `S_out P^2 S_in^{-1}`. Thus even complete knowledge of
all `PROBE` answers need not determine a `ROLLOUT x,2` target. This is an
information failure, not a model limitation.

Use one shared state encoding `S` on both sides:

\[
\widetilde P=S\circ P\circ S^{-1},\qquad
\widetilde P^h=S\circ P^h\circ S^{-1}=(\widetilde P)^h.
\]

Alternatively expose and charge an explicit output-to-input bridge. Independent
input/output bindings without that bridge must be removed.

The quotient also needs two relations:

\[
w\sim_{obs}w'\iff
\forall\pi\;\mathcal L(\tau_{obs}\mid w,\pi)
=\mathcal L(\tau_{obs}\mid w',\pi),
\]

and `w ~score w'` when every legal scored query has the same target. Private
targets are not part of `tau_obs`. The protocol is learnable only if
`w ~obs w'` implies `w ~score w'`. Score the quotient by `~score` only after
that refinement condition is proved. The world definition must also include the
change-point time and change kernel; a static `P` alone does not define the
claimed lifetime behavior.

## 2. The v0 primitive library is underspecified

As written, vector-valued `id`, `add`, and scalar `scale_c` generate only scalar
linear combinations of the whole input vector, not an arbitrary matrix `A`.
`const_c` is also typed as a scalar while `P` is vector-valued. The claim that v0
generates general affine programs therefore does not follow.

Choose and freeze one of these exact interpretations:

- scalar nodes with leaves `input_i`, scalar constants, scalar add/scale, and
  `d` ordered output roots; this can represent `P(x)=Ax+b`; or
- vector nodes, in which case state the much smaller reachable canonical class
  and stop using arbitrary-`A` controls.

The schema library size must exclude field parameters if those parameters are
also charged separately. Specify arity, node types, output roots, sharing, and
whether the object is an AST or a DAG. Affine-monomial surface conjugation keeps
the *function class* affine, but it need not preserve a fixed latent AST-size
bound. Enumerate program-plus-binding worlds rather than claiming syntactic
closure under rendering.

## 3. Mathematical boundary corrections

### T72.1 — quotient

The impossibility statement is correct only for the observable relation above.
The current relation, which includes unobserved `Y`, can separate worlds the
learner can never distinguish. Replace T72.1 with the two-relation definition
and the refinement test.

### T72.2 — Fano and state bits

The displayed Fano inequality is correct conditional on a uniform class index
`H` and a decoder that identifies that index from `M`. It does not automatically
give a bit floor for prediction accuracy. Add all of:

1. a discrete or quantized `B`-bit state channel, so
   `I(H;M) <= H(M) <= B`; an unconstrained real latent has no such bit bound;
2. uniform sampling over quotient classes, or the nonuniform entropy/prior form;
3. a proof that the actual scored-query performance decodes `H`, or a
   rate-distortion bound for the declared prediction loss; and
4. conditioning on the complete transcript available before erasure.

The `ceil(log2 N)` index is a storage optimum only after the class is exactly
known and the codebook is shared. Call it an optimal class-index code, not an
unqualified information ceiling.

### T72.3 — active identification

The `bar_alpha` greedy bound and `ceil(log_K N)` leaf-count lower bound are
valid for noiseless exact identification of a finite class, with `K` a uniform
upper bound on probe outcomes. For vector feedback here, `K <= p^d`. They are
generic upper/lower bounds, not “exact active-identification bounds”; only the
stated minimax dynamic program is exact. Define the nonadaptive comparator as
the optimal set/sequence fixed before outcomes, not merely a “predeclared” weak
schedule.

For the complete affine class

\[
\mathcal H=\{x\mapsto Ax+b:A\in\mathbb F_p^{d\times d},
b\in\mathbb F_p^d\},
\]

`|H|=p^{d(d+1)}` and each probe has at most `p^d` outcomes, so every exact
method needs at least `d+1` probes. The fixed nonadaptive schedule
`0,e_1,...,e_d` recovers `b` and every column of `A` in exactly `d+1` probes.
Therefore adaptive and nonadaptive minimax counts are equal. The scalar-affine
case similarly has a fixed two-probe solution.

A size-bounded sparse/union-of-subspaces affine class can in principle behave
differently, but a `2x` advantage must be exhibited by the frozen quotient and
exact decision trees. If enumeration does not show it, affine v0 cannot test
active experiment choice and must remain a calibration task.

### T72.4 — code bound and rollout

The AST/DAG expression is a loose latent-program upper bound, but it omits the
node count, exact arities/padding, `d` output roots, types, and surface binding.
A valid bound must include, at minimum,

\[
B_{out}\le d\lceil\log_2(n+d)\rceil
\]

plus a prefix/topological encoding and `B_bind`. For one shared coordinate
permutation plus per-coordinate invertible affine value maps, a loose binding
charge is

\[
B_{bind}\le\lceil\log_2(d!)\rceil+
d\left(\lceil\log_2(p-1)\rceil+\lceil\log_2p\rceil\right).
\]

If the stored code is the observable canonical affine map instead, charge
`d(d+1) ceil(log2 p)` bits for `A,b`; do not also claim that its hidden AST was
recovered. For tiny settings the claimed exponential separation must be
reported numerically, not inferred from asymptotics against the truth table for
an unrestricted function class.

The formula

\[
P^h(x)=A^h x+\sum_{i=0}^{h-1}A^i b
\]

and augmented-matrix `O(d^3 log h)` control are correct for a static affine map.
They remain valid on the surface only under the shared conjugate binding above.
Specify whether a hidden change can occur inside a rollout; if so, this formula
does not cover that rollout.

## 4. Revision and retention are not yet constructible as stated

For pre/post affine maps, define

\[
\Delta(x)=(A'-A)x+(b'-b).
\]

Then `Q_changed` may contain points with `Delta(x) != 0`, while
`Q_protected` may contain points solving `Delta(x)=0`. A nontrivial change does
not guarantee a protected point: for example, `A'=A` and `b'!=b` changes every
full-vector output. Stage A must select a change family for which both sets are
nonempty with preregistered minimum sizes, not select convenient points after a
result. Coordinate-level scoring is another valid design, but it is a different
contract.

`Q_alias` must be deleted. If a reparameterization is behaviorally equivalent,
T72.1 says no legal behavioral query can detect whether it was “accepted.” Such
an alias should receive identical score. An internal-state diagnostic would
require a new, explicitly exposed and charged interface.

Candidate-state swaps should be defined as paired-world interventions: swap
states between worlds that differ only in the changed behavioral mechanism,
then require the frozen affected/protected query pattern. An opaque state has no
intrinsic “corresponding node” to swap.

## 5. What Stage A can and cannot establish

AST, graph-shape, and larger-AST holdouts do not ensure behavioral
compositionality in affine v0. Every composition collapses to `A,b`; larger or
different graphs can occupy the same quotient class. The required post-quotient
disjointness test may therefore empty the advertised splits. Likewise, `10x`
rollouts test whether the learner has acquired affine exponentiation, not
whether it preserves an internal multi-node composition.

Fresh bindings and a committed generator prevent literal test-instance reuse,
but they do not rule out task recognition. With fixed `p,d` and one affine
family, recognizing the task and invoking the fixed basis estimator plus matrix
exponentiation is the optimal solution. A tiny exhaustively enumerated world
also admits a memorized decision tree. Stage A item 7 needs an operational
parameter/table-bit accounting for those controls; a hash commitment alone is
irrelevant.

This is classical exact concept learning with membership queries
([Angluin](https://doi.org/10.1023/A%3A1022821128753)), version-space splitting
or generalized binary search
([Nowak](https://arxiv.org/abs/0910.4397)), affine system identification, and
program induction. Learned reusable program libraries and neural search are
already represented by systems such as
[DreamCoder](https://arxiv.org/abs/2006.08381). The 2026 agentic-automata study
also compares LLM discovery directly with classical automata learners
([primary paper](https://arxiv.org/abs/2606.16576)).

That is not fatal, because T72 need not claim theoretical novelty. It does mean
the legitimate result is narrow: whether lifetime-objective training causes the
same neural backbone to amortize a known identification/update algorithm better
than next-step training and strong classical/program-induction controls.

Affine v0 can calibrate that comparison, persistence, and leakage. It cannot
earn a natural pilot merely by passing Stage B. A resulting-model path requires
a later preregistered grammar in which behavior does not collapse to one affine
map, the exact adaptive advantage is nonzero, protected changes are guaranteed,
and the same updater transfers without family-specific retraining. Only that
result could admit the natural-domain pilot proposed by T71.

## Exact disposition

- Keep the no-code/no-run status.
- Replace independent bindings with one conjugate state binding.
- Separate observational and scoring equivalence and include the change process.
- Correct the Fano claim and AST/binding ledger.
- Delete `Q_alias`; construct changed/protected sets algebraically.
- Prove the exact adaptive/nonadaptive ratio before calling v0 an active-learning
  test. For the full affine class the ratio is `1`, not `2`.
- Retain affine v0 only as a model-free calibration. Do not let it admit a neural
  run or natural pilot without a non-collapsing successor grammar.
