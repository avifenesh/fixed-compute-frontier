# Independent adversarial audit: value-relative abstraction boundary (T59)

Date: 2026-08-01  
Scope: exact theorem assumptions, literature occupancy, and experiment gate; no experiments run

## Verdict

**RETAIN THE NO-STATE-MERGE BOUNDARY; RETAIN `W + Z_G` ONLY AS AN OCCUPIED DESIGN CONTRACT; NO DISTINCT UPDATER; NO RUN.**

T59.1 and T59.2 express valid no-free-lunch facts after tightening their
assumptions. They establish that arbitrary point-separating rewards defeat every
nontrivial exact state aggregation, and that dynamics/action structure alone
does not determine a policy. They do not select a representation, memory update,
goal compiler, exploration rule, or learning algorithm.

The proposed persistent broad world state `W` plus goal-conditioned working
state `Z_G` is sensible, but is already the broad contract of goal-conditioned
bisimulation, successor-feature/task-transfer methods, value-aware abstraction,
abstract world models, rate-distortion abstraction, and task-conditioned
planning. Cross-session persistence does not turn this conjunction into a new
updater. T59 contains no learned compiler and no acquisition or served-compute
separation, so no experiment follows.

## T59.1: exact scope and corrections

For a deterministic encoder `phi:S -> Z`, the one-step counterexample is valid
under all of the following:

- the task family contains every bounded **measurable** one-step reward, or at
  least a reward family that separates every pair of distinct states;
- the state measurable space is point-separating (standard Borel is sufficient),
  so an admissible reward can distinguish `s` and `s'`;
- the selected action `a_0` is legal in both states, or state-only one-step
  rewards are admitted;
- exact preservation includes the immediate reward and is required from every
  initial state, for the relevant action/policy; and
- `phi` is fixed independently of the reward/query.

Then if `phi(s)=phi(s')`, choose an admissible one-step reward with different
values at `(s,a_0)` and `(s',a_0)`. No single abstract reward can preserve both.
Thus exact preservation for the point-separating reward family implies
injectivity **on the states distinguishable by that family**.

The exact conclusion is narrower than “no compression”:

> Universal exact value preservation forbids lossy state aggregation across
> query-distinguishable states.

An injective representation can still losslessly re-encode observations,
exploit algorithmic structure, share a generative dynamics program, or reduce
computation for a restricted query. The theorem gives no bit, sample, or compute
lower bound. Replace every claim of “no compression” by “no nontrivial exact
state merging” unless a coding/computational lower bound is separately proved.

For a randomized encoder, ordinary injectivity is not the right statement.
Exact preservation of all separating rewards requires the latent distributions
for distinguishable states to permit almost-sure recovery (equivalently, no
information-destroying overlap under the chosen formalization). T59 should
either remain deterministic or state this extension precisely.

The aggregated-transition condition in section 1 is the standard exact MDP
homomorphism/bisimulation condition for a fixed reward and common action set.
With state-dependent legal actions, legal-action equivalence must also be
required. In a POMDP/history setting, the object being aggregated must be a
belief or predictive/history state; the fully observed Markov theorem does not
transfer automatically.

## T59.2: exact scope

The self-loop counterexample is correct for a one-step task or discounted
infinite horizon: identical transitions with swapped, strictly ordered action
rewards yield opposite unique optimal actions. Calling the generated Lie algebra
identical is harmless but unnecessary—the zero/self-loop algebra makes the same
point.

The result proves only:

\[
\text{action/dynamics structure alone}\not\Rightarrow\text{policy transfer}.
\]

It does not refute transfer conditioned on a goal/reward descriptor. Nor does it
show that structural knowledge is useless: shared dynamics can reduce planning
or transition acquisition once the new reward is supplied. Any stronger no-gain
claim would need a sample- or compute-complexity lower bound.

## T59.3 is not yet a theorem or algorithm

The rate-distortion display

\[
\min_{q(z\mid h,G)} I(H;Z\mid G)
\quad\text{subject to}\quad
\sup_{\pi\in\Pi_G}|V_G^\pi(h)-\bar V_G^\pi(z)|\le\epsilon
\]

is only a sketch. It lacks:

- a joint distribution over `(H,G)` for the conditional mutual information;
- a horizon or discount, action-legality convention, and task distribution;
- a definition of the stochastic-encoder constraint (expectation,
  high-probability, or every `z` in the encoder support);
- a relationship between history policies and abstract policies, plus a lifting
  map that makes `bar V` well-defined;
- a definition of `W`, its update, its memory/compute budget, and the compiler
  `C(W,G)=Z_G`; and
- a learnability, optimization, acquisition, or served-compute guarantee.

The supremum over **all** policies is much stronger than preserving the optimal
decision and often approaches model/bisimulation equivalence. If the goal is
decision sufficiency, specify the regret of the lifted abstract optimal policy.
If the goal is policy-uniform evaluation, retain the supremum but accept the
stronger representation requirement. These are different contracts.

There is also an unresolved retention problem. If future goals are unrestricted
and `W` must later compile an exact sufficient `Z_G` for all of them, T59.1
applies to `W`: it must retain all query-distinguishable information. A bounded
`W` therefore needs a declared future-goal distribution/family and permitted
distortion. “Broader than any one working abstraction” is not a preservation
rule.

## Decisive counterexample to an automatic `W -> Z_G` gain

Let `W` contain a complete finite MDP and let each goal specify an arbitrary
bounded state-action reward. Constructing the smallest exact quotient for each
goal can require reading the entire reward and transition description, and the
quotient may be injective for every goal. In that family, `Z_G` is neither
smaller nor cheaper than `W`; compilation adds cost.

Conversely, if goals depend on one supplied feature and dynamics factor cleanly,
`Z_G` can be tiny. The gain comes from the restricted goal/dynamics factorization,
not from the two-level diagram itself. T59 must state and exploit that restriction
before claiming a computational advantage.

## Current primary-literature occupancy

The broad contract is already occupied:

- [Goal-conditioned bisimulation](https://proceedings.mlr.press/v162/hansen-estruch22a.html)
  defines abstractions over goal-conditioned tasks, proves sufficiency under its
  reward family, and learns a metric representation for goal transfer.
- [State Abstractions for Lifelong Reinforcement Learning](https://proceedings.mlr.press/v80/abel18a.html)
  directly studies reusable abstraction across a sequence of tasks.
- [Learning Abstract World Models for Value-preserving Planning](https://arxiv.org/abs/2406.15850)
  learns abstract MDPs from sensorimotor experience with bounded planning-value
  loss for temporally extended actions.
- [Adaptive state-action abstractions via rate-distortion](https://arxiv.org/abs/2606.06123)
  dynamically changes abstraction resolution using a Bellman-residual plus
  bisimulation-error certificate.
- [Learning to Perceive the World Through Control](https://openreview.net/forum?id=ZAZb1ZGnFH)
  studies representations that retain control-relevant features while rejecting
  nuisance information.
- [VPSD-RL](https://arxiv.org/abs/2605.06500) makes value preservation depend on
  both controlled dynamics and the reward functional.
- [Behavior Consistency in Text-Based World Models](https://arxiv.org/abs/2604.13824)
  explicitly trains/evaluates world models by downstream behavior rather than
  surface-state fidelity.

Successor representations/features and universal goal-conditioned value
functions further occupy the general pattern “retain reusable predictive
dynamics, supply a new reward/goal, derive a task-specific decision object.”
T59's `W + G -> Z_G` diagram is a clear synthesis of this literature, not a new
computational primitive.

What may remain unimplemented in current LLM agents—a single persistent state
updated across sessions and a certified cheap compiler—is a systems capability
conjunction. It becomes a distinct updater only after defining the update and
compiler and proving or demonstrating an advantage over these exact controls.

## Exact corrections

1. Replace “all bounded rewards” by a measurable, point-separating reward family
   and state the common-legal-action/one-step assumptions.
2. Replace “no compression” by “no lossy state aggregation”; do not infer coding
   or compute lower bounds from injectivity.
3. Scope the abstraction theorem to fully observed MDPs, or define the belief/
   history-state analogue and legal-action preservation.
4. Keep T59.2 as a reward-necessity counterexample, not a claim that shared
   dynamics cannot reduce learning cost.
5. Fully define the distribution, policy lifting, distortion, horizon, `W`
   update, compiler, budgets, and future-goal family in T59.3.
6. Choose optimal-decision regret or policy-uniform value preservation; do not
   mix them.
7. Add a retention contract for `W`; unrestricted exact future goals force it
   to retain all query-distinguishable state information.
8. Delete the claim of a missing architectural primitive. At present this is an
   occupied desideratum plus an unspecified persistent implementation.

## Experiment disposition

**No experiment is earned.** T59 is a boundary note, not a candidate updater.
Its theorems rule out an overbroad objective but do not select a mechanism. The
next work is paper-only: specify one restricted goal family, one bounded update
for `W`, one compiler to `Z_G`, and an exact prediction of lower acquisition or
served compute than goal-conditioned bisimulation/value-aware world-model
controls. Only a proved separation or concrete algorithm with a preregistered
falsifier could admit a later small local test.
