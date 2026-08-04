# T49 counterexample-guided state birth — structural learning instead of hidden-state drift

Date: 2026-08-01  
Status: **CLASSICAL LEMMAS RETAINED; ARCHITECTURE NO-GO; T50 BOUNDARY ONLY; NO EXPERIMENT**

## 0. Key shift

T45-T48 show that once a latent factor, intervention dictionary, or controlled
test basis is supplied, Bayesian filters, PSRs, recurrence, and system
identification already solve the update algebra. The missing operation is
earlier:

> create a new internal distinction when two histories currently treated as
> the same predict different consequences under some legal experiment.

T49 treats representation as a structure that can split and grow, not only a
fixed-width vector whose values move.

## 1. Exact finite controlled world

Let a deterministic Mealy system have reachable latent states `S`, legal
actions `A`, outputs `O`, transition `delta`, and output map `lambda`:

\[
s_{t+1}=\delta(s_t,a_t),
\qquad o_{t+1}=\lambda(s_t,a_t).
\]

Two reachable states are behaviorally equivalent when every future experiment
produces the same output trace:

\[
s\sim s'
\Longleftrightarrow
\forall w\in A^*,\quad
\lambda^*(s,w)=\lambda^*(s',w).
\]

The quotient is the minimal exact predictive state machine. Histories are
equivalent when the states they reach are equivalent; the learner need not see
the latent state names.

## 2. T49.1 — state birth from a valid counterexample

Suppose the learner currently maps representative histories `h,h'` to one
abstract node, but a legal action word `w` satisfies

\[
\lambda^*(s(h),w)\ne\lambda^*(s(h'),w).
\]

Then `w` is a certificate that the node is too coarse. Split the node by the
observable signature

\[
c_w(h)=\lambda^*(s(h),w).
\]

This split is sound: it never separates two truly behaviorally equivalent
states. It is progressive: the witnessed block becomes at least two nonempty
blocks.

### Conditional termination theorem

Assume the partition covers the entire reachable state set; the learner can
reset or replay representative histories; its routing and transition table are
total and stable during refinement; a complete oracle both finds a valid
counterexample for every nonminimal partition and localizes it to the
responsible state pair; and the true minimal quotient has `n` reachable
classes. If the learner starts with one block and performs sound split-only
refinement, then at most `n-1` successful splits produce the exact quotient.

**Proof.** Each counterexample soundly refines at least one block, so the number
of blocks strictly increases. Soundness prevents the partition from becoming
finer than the `n` true behavioral classes. If no counterexample exists, every
pair in a block agrees on all action words and is behaviorally equivalent. QED.

This is classical partition refinement/active automata learning, not a novelty
claim or an acquisition theorem over raw histories. Access sequences, replay,
counterexample search and localization, stable routing, and reachable-state
coverage are assumed resources. Its role is only to specify what “invent a
state variable” means operationally.

## 3. T49.2 — fixed test horizons cannot certify sufficiency

For every finite horizon `L`, there are two states that agree on every action
word of length at most `L` but differ at length `L+1`.

**Construction.** Use one action `a` and chain states `c_0,...,c_{L+1}` with

\[
\lambda(c_0,a)=1,\quad
\lambda(c_i,a)=0\ (i>0),\quad
\delta(c_0,a)=c_0,\quad
\delta(c_i,a)=c_{i-1}\ (i>0).
\]

Starts `c_L` and `c_{L+1}` emit identical zero traces for the first `L`
actions. On action `L+1`, the first emits one and the second emits zero.

Therefore an auxiliary loss or fixed panel of short future tests can declare a
state sufficient while missing a real longer-horizon distinction. The machine
size in this construction grows with `L`; when a valid finite state bound is
known, a finite distinguishing-depth bound exists. Test generation or a
declared complexity-dependent horizon bound is the necessary resource.

## 4. T49.3 — the verifier is not free intelligence

An equivalence oracle that always finds `w` when the abstraction is wrong has
already solved the hard search problem. In a known finite machine, exhaustive
product-graph search can find a shortest distinguishing word. In an unknown
black-box environment, an equivalence query is an additional information
interface, not ordinary passive feedback.

Charge separately:

\[
C_{cex}=(N_{membership},N_{equivalence},N_{resets},
L_{executed},F_{search},B_{table},R_{sacrificed}).
\]

This vector is only shorthand. Complete cost must also include verifier and
specification construction, counterexample bits/length, snapshot and replay
instrumentation, latency/energy/traffic/safety exposure, global-to-local
counterexample attribution, word enumeration and legality checks, access
sequence/provenance retrieval, meta-training and its deployment amortization,
and merge-certificate construction.

If the verifier is a larger model, simulator, theorem prover, test suite, or
human, its construction and calls are teacher compute. If it is only the same
learner searching the real environment, failed probes, risk, and horizon are
interaction cost.

## 5. Candidate architecture contract

The possible learned system has five typed parts:

```text
encode(raw history) -> current abstract node
propose(node pair, uncertainty) -> legal distinguishing experiment
execute(experiment) -> paired output traces
refine(valid counterexample) -> split/birth + provenance edge
consolidate(nodes) -> merge only under a certified equivalence condition
```

Only `propose` may initially remain learned **after** every method receives the
same frozen encoder, typed action compiler, stable state routing, replay access,
and counterexample localizer. Without that restriction, the apparent one-block
candidate actually contains at least three learned maps: raw history to node,
raw interface to legal experiment, and delayed outcome to responsible node
pair. Counterexample validity, split soundness, state-table update, destructive
controls, and cost accounting remain exact. This addresses the “stronger
compiler” objection only in the frozen-interface setting: the learned model
does not declare itself correct; the environment or verifier supplies a
falsifiable consequence.

The state representation should be a dynamically growing graph or table of
predictive distinctions and their witness experiments. Sparse routing may keep
active work bounded while stored structure grows, but no such physical edge is
claimed until its memory, retrieval, and traffic are explicit.

## 6. Why this may improve a result model

The potential gain is not lower perplexity. It is a model that can:

1. form a new state/hypothesis distinction after deployment;
2. retain the exact experiment that justified the distinction;
3. revise only the affected node rather than globally perturbing old knowledge;
4. ask targeted questions when its current abstraction aliases two worlds; and
5. reuse learned test-proposal strategy across renamed systems.

In finite worlds with a real equivalence interface, the exact control reaches a
correct world model after finitely many structural births. A fixed short-test
representation has no such guarantee.

This does not yet beat an RNN, Transformer, or meta-RL agent in function class;
each can emulate the table. The claimed edge must be acquisition reliability,
cross-world transfer, noninterference, or physical sparsity.

## 7. Strongest controls and current evidence

Mandatory controls include:

- Angluin L-star and modern active DFA/Mealy learners;
- exact bisimulation/partition refinement and CEGAR;
- a spectral PSR/OOM learner with the same legal tests;
- recurrent/meta-RL and ORBIT-style cross-episode learning;
- a Transformer agent retaining the full transcript;
- external episodic memory with the same bytes and retrieval work; and
- adaptive recurrent/fixed-point reasoners for test proposal.

[Agentic automata learning](https://arxiv.org/abs/2606.16576) reports that
frontier LLM agents degrade with DFA size through failures in query planning,
evidence integration, and hypothesis construction. Conversely,
[Counterexample Guided Learning in the Large](https://arxiv.org/abs/2606.11521)
reports large success gains for LLM regex induction when a verifier returns
structured counterexamples. Those results make this a live capability seam,
but they also establish counterexample-guided LLM learning as current prior art.

## 8. Fatal controls

Close T49 before a model run if:

1. the exact classical learner already uses fewer interactions and less total
   work on every admitted world;
2. the learned proposer merely paraphrases breadth-first distinguishing-word
   search;
3. gains vanish when full-transcript and exact-table controls receive equal
   state and verifier access;
4. state births degenerate into one node per history with no cross-world reuse;
5. merge/consolidation lacks a sound condition and reintroduces forgetting;
6. natural tasks do not expose an executable counterexample or outcome
   interface;
7. verifier/search cost exceeds the capability obtained; or
8. one fixed synthetic automaton family is the only positive domain.

The independent audit triggers the architecture control already: active
automata learning, CEGAR, PSR/OOM adaptation, recurrent/meta-RL controllers,
and exact or episodic tables all contain dynamic state refinement. Append-only
table noninterference is not an end-to-end edge because learned routing,
retrieval, finite memory, and consolidation can still interfere.

## 9. Former substantial gate — withdrawn

The earlier numerical gate is withdrawn rather than used to admit a run. It
mixed exact quotient recovery with approximate accuracy, used a generic
recurrent proposer instead of the best cost-aware classical search as primary
control, denied controls the candidate's raw encoder, and treated append-only
table safety as end-to-end noninterference. Interaction savings also do not
imply decision-regret savings without a frozen task horizon and diagnostic
action cost.

Any future gate needs one shared typed test language, identical encoder and
verifier access, exact versus approximate targets separated, complete search
cost, and routing/retrieval/consolidation included in protected behavior.

## 10. Disposition and next proof obligation

The conditional refinement lemma and fixed-horizon counterexample are retained.
Novelty, architecture gain, and experiment admission are no-go. The only
admitted continuation of this branch is:

> an amortized counterexample-ranking theorem containing both a no-shared-prior
> lower bound and a shared-prior achievability/cost result, with identical legal
> information and fully charged meta-training.

That object is T50. No CPU, local GPU, or rented GPU run is admitted by T49.
