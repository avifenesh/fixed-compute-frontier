# T59 value-relative abstraction boundary — why structure alone cannot transfer intelligence

Date: 2026-08-01  
Status: **EXACT NO-STATE-MERGE BOUNDARY RETAINED; OCCUPIED DESIGN CONTRACT ONLY; NO RUN**

## 0. Result in one sentence

An action algebra, causal graph, predictive state, or reusable program is not a
task solution by itself. To transfer decisions, an abstraction must preserve
both controlled consequences and the current reward/query. If a measurable
future reward family separates every pair of states, every nontrivial exact
state merge can be defeated by a query that distinguishes the merged states.
This forbids lossy aggregation; it does not forbid lossless or algorithmic
compression.

## 1. Exact controlled abstraction

Let a fully observed controlled Markov process have state `s in S`, common
legal action set `A`, reward
`r(s,a)`, and kernel `P(.|s,a)`. A deterministic abstraction is

\[
\phi:S\to Z.
\]

For an exact Markov abstraction under a fixed task, every pair `s,s'` with
`phi(s)=phi(s')` must satisfy

\[
r(s,a)=r(s',a)
\]

and, for every abstract cell `z'`,

\[
\sum_{u:\phi(u)=z'}P(u\mid s,a)
=\sum_{u:\phi(u)=z'}P(u\mid s',a)
\]

for every legal action. These reward and aggregated-transition conditions make
the abstract process well defined and preserve policy values.

The important word is **fixed**. Which states may be merged depends on what the
agent will be asked to predict or optimize.

## 2. T59.1 — arbitrary queries destroy nontrivial exact compression

Assume the measurable state space is point-separating and the future task
family contains every bounded measurable one-step reward
`r:S x A -> [0,1]`, or at least separates every distinct state pair. Require
exact immediate-reward preservation from every initial state, and fix `phi`
independently of the query. If `phi` merges two reward-distinguishable states
`s != s'`, choose one action `a_0` legal in both and the reward

\[
r(s,a_0)=1,\qquad r(s',a_0)=0,
\]

with arbitrary values elsewhere. The abstract reward for `phi(s)=phi(s')`
cannot equal both values. Therefore `phi` fails to preserve this one-step task.

Hence:

\[
\boxed{\text{Exact value preservation for a point-separating reward family implies }
\phi\text{ is injective on reward-distinguishable states.}}
\]

No transition, architecture, or training assumption can remove this state-merge
result; the adversarial query itself makes the distinction relevant. It does
not provide a bit, sample, or compute lower bound: an injective representation
may still re-encode losslessly or exploit a compact generative program.
Nontrivial lossy aggregation requires a restricted task/query family or
accepted epsilon error.

## 3. T59.2 — identical action structure does not imply policy transfer

Consider one state with two self-looping actions. The action dynamics, generated
monoid, Lie algebra, observations, and transition kernel are identical in two
tasks. In task one,

\[
r_1(a_0)=1,\qquad r_1(a_1)=0,
\]

while task two swaps the rewards. Their optimal policies are opposite despite
identical action structure.

The same obstruction survives in rich worlds: an abstract action algebra does
not determine its representation on state, and even a complete dynamics model
does not determine the goal. T58 can still reduce transition acquisition in a
restricted shared-dynamics family after the new reward is supplied, but algebra
matching alone cannot certify policy transfer.

Likewise, a reusable theory in T57 earns decision value only when its raw
binding, task readout, and downstream planner are correct. Predictive
compression and symbolic elegance are not substitutes for task sufficiency.

## 4. T59.3 — the useful object is query-relative sufficiency

Let `G` denote the current goal/query. A working representation `Z_G` should
retain the history information needed for future decisions under `G`, while a
slower shared world state `W` may retain broader reusable dynamics. An idealized
rate-distortion form is

\[
\min_{q(z\mid h,G)} I(H;Z\mid G)
\]

subject to a declared value or prediction distortion such as

\[
\sup_{\pi\in\Pi_G}
\left|V_G^\pi(h)-\bar V_G^\pi(z)\right|\le\epsilon.
\]

This motivates an already-known design contract:

```text
persistent broad world state W
        + current goal/query G
        -> compact task-sufficient working state Z_G
        -> experiment / plan / answer
        -> outcome updates W and recompiles Z_G
```

The distinction is essential:

- `W` supports continual learning and reuse across goals;
- `Z_G` prevents every decision from processing all accumulated knowledge;
- the experiment policy seeks information valuable for `G`, not generic
  surprise; and
- a changed goal can make a previously discarded distinction relevant, so `W`
  must be broader than any one working abstraction.

## 5. What current systems are missing

Current LLM deployments normally collapse these roles into one fixed parameter
state plus a transient token context. They have no trained, persistent world
update across requests and no explicit contract that compiles accumulated
experience into a goal-sufficient working model. Retrieval adds old content;
reasoning tokens repeatedly reconstruct relevance; online fine-tuning mutates a
global prior without a task-sufficiency certificate.

The desired systems capability is therefore more precise than "memory" or
"causality":

> maintain a broad, revisable model learned over a lifetime, and cheaply compile
> from it the smallest currently sufficient model for the goal, uncertainty,
> and legal actions at hand.

This is not yet a distinct method. Goal-conditioned bisimulation, successor
features, value-aware and abstract world models, rate-distortion abstraction,
and lifelong state abstraction occupy the broad contract. Cross-session
persistence alone does not supply a new update or compiler.

Current primary boundaries include:

- Goal-conditioned bisimulation:
  https://proceedings.mlr.press/v162/hansen-estruch22a.html
- State abstractions for lifelong reinforcement learning:
  https://proceedings.mlr.press/v80/abel18a.html
- Learning abstract world models for value-preserving planning:
  https://arxiv.org/abs/2406.15850
- Adaptive state-action abstractions via rate-distortion:
  https://arxiv.org/abs/2606.06123
- Operator-Guided Invariance Learning / VPSD-RL:
  https://arxiv.org/abs/2605.06500
- Behavior Consistency in text world models:
  https://arxiv.org/abs/2604.13824
- Learning to Perceive the World Through Control:
  https://openreview.net/forum?id=ZAZb1ZGnFH

## 6. Substantial model claim that would count

A method would count only if, with one frozen deployed updater and a complete
cost ledger, it demonstrates all of:

1. a persistent broad state that improves from real outcomes across sessions;
2. goal-conditioned working states that are materially smaller/cheaper than the
   broad state and change when the goal changes;
3. protected decision quality within a declared epsilon against a full-state
   control;
4. at least 30% lower cumulative regret or at least 4x lower acquisition cost
   on held-out families and 10x training horizons; and
5. a gain over matched recurrent, retrieval, full-context, program-memory,
   world-model, and value-aware abstraction controls.

Static benchmark accuracy, a prettier latent space, or recovery of a known
algebra does not count.

## 7. Experiment disposition

No experiment is admitted from T59 alone. The theorem kills universal exact
lossy state aggregation; it does not select how to represent or update `W`,
compile `Z_G`, assign long-horizon credit, or preserve information that a later
goal may need. If future goals remain unrestricted and exact, the same theorem
forces `W` to retain every query-distinguishable state distinction. A bounded
`W` therefore needs a declared future-goal family and distortion contract.

The next paper object must specify one **goal-conditioned compiler/updater** and
show either:

- a representation/sample/served-compute separation from a matched recurrent
  world model; or
- a restricted family with an exact sufficiency and horizon-extrapolation
  theorem.

Only then is a small local symbolic falsifier earned.

The independent audit gives the exact theorem scope and literature boundary:

- [T59 independent audit](value-relative-abstraction-boundary-t59-independent-audit.md)
