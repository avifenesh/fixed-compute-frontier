# T49 counterexample-guided state birth — independent audit

Date: 2026-08-01  
Verdict: **HOLD THE CLASSICAL REFINEMENT LEMMAS; NO-GO FOR AN ARCHITECTURE OR EXPERIMENT**

## 1. Outcome

T49 gives a correct operational description of one way to refine an exact
finite-state abstraction: a witnessed future-output difference soundly splits a
coarse block. It does not identify a new capability mechanism. State creation
from counterexamples is the central operation in active automata learning,
partition refinement, CEGAR, and adaptive predictive-state construction, and a
recurrent controller or external table can implement the same refinement.

The only potentially open edge is narrower than “dynamic state birth”:

> across a distribution of related systems, can a learned ranker reduce the
> complete expected cost of finding a valid counterexample relative to the best
> classical search with the same prior and legal information?

T49 does not yet prove that edge. No CPU, model, local-GPU, or rental experiment
is admitted.

## 2. Mealy refinement theorem

### What is mathematically valid

For a deterministic Mealy machine, behavioral equivalence

\[
s\sim s'\iff \forall w\in A^*,\;
\lambda^*(s,w)=\lambda^*(s',w)
\]

is the usual right behavioral equivalence. If a partition is defined over the
entire reachable state set and a word `w` gives different traces for two states
in one block, refining that block by

\[
c_w(s)=\lambda^*(s,w)
\]

is sound: equivalent states cannot be separated. Starting from one block,
split-only sound refinement can perform at most `n-1` successful refinement
operations before reaching `n` blocks, because each operation increases the
block count by at least one and no true equivalence class is split.

### Hidden assumptions required for the stated conclusion

The theorem becomes an exact-quotient result only after adding all of the
following.

1. **The partition covers every reachable state class.** A learner that stores
   only observed representative histories can terminate while an undiscovered
   reachable class has no representative. Counterexamples must also provide or
   induce access sequences for new states, as observation-table and
   discrimination-tree algorithms do.
2. **The oracle is complete.** “Every nonminimal partition receives a valid
   counterexample” is precisely an equivalence oracle over all legal words and
   all relevant state pairs. The termination proof assumes the hard search
   problem rather than solving it.
3. **Histories can be replayed to the same state.** Evaluating `c_w(h)` requires
   reset/replay access to `s(h)`. Determinism alone does not supply reset,
   snapshot, or reproducible access.
4. **The current object is a state partition, not merely an encoder collision.**
   A learned encoder may map one true state to different nodes on different
   histories, or change earlier routes after training. In that case the block
   count is not bounded by `n`, and the split theorem does not describe the
   implemented learner.
5. **The learned hypothesis has total, stable transitions.** Standard active
   automata learners maintain closure/consistency or a discrimination tree so
   that a counterexample to the whole machine can be converted into a specific
   state/suffix refinement. T49 assumes the responsible pair `(h,h')` is
   already localized.

The “never merges without an equivalence proof” clause is unnecessary for the
split-only theorem: sound refinement from one block never creates duplicate
blocks belonging to the same true equivalence class. Once learned routing,
approximation, or merging is allowed, a finite equivalence proof is itself an
oracle/search cost and the `n-1` split bound no longer follows automatically.

The corrected result is therefore a conditional partition-refinement lemma,
not a finite-query acquisition theorem over raw histories.

## 3. Fixed-horizon construction

The chain construction and its output timing are correct. Mealy output is
emitted from the pre-transition state. Starting from `c_L`, the first `L`
actions emit zero and leave the machine at `c_0`; action `L+1` emits one.
Starting from `c_{L+1}`, actions `1,...,L+1` emit zero. With the one-action
alphabet, the starts therefore agree on every word of length at most `L` and
differ at length `L+1`.

The scope must remain explicit. The construction grows the machine with `L`.
It proves that no fixed horizon independent of system complexity certifies all
finite machines. It does not prove that adaptive test generation is required
when a valid state bound is known: in an `n`-state deterministic Mealy machine,
a finite distinguishing-depth bound exists. T49's alternative “or a declared
horizon bound” is the correct boundary.

## 4. The verifier and exact blocks contain most of the intelligence

The proposed vector

\[
C_{cex}=(N_{membership},N_{equivalence},N_{resets},L_{executed},
F_{search},B_{table},R_{sacrificed})
\]

is a useful start but is not complete. It must also charge:

- construction and maintenance of the target specification, simulator,
  theorem prover, test suite, or human oracle;
- the number of information bits and length delivered by each counterexample,
  not merely one equivalence-call count;
- state snapshot/replay instrumentation and failed attempts to reproduce both
  representative histories;
- execution latency, energy, traffic, safety exposure, and side effects for
  every prefix and suffix;
- localization of a global hypothesis counterexample to the responsible node
  pair and distinguishing suffix;
- candidate-word enumeration, legality checking, deduplication, and adaptive
  search state;
- storage and retrieval of access sequences, discrimination witnesses,
  transition tables, and provenance;
- meta-training systems, trajectories, labels/verifiers, and the number of
  deployment systems across which that cost is amortized; and
- construction and verification of any merge/consolidation certificate.

An equivalence query is strictly stronger than ordinary sampled feedback. In a
black-box environment it may encode an unbounded search over words and return a
high-information witness. Counts of membership and equivalence queries are not
commensurate without their complete generation and execution cost.

Most importantly, T49 says only `propose` remains learned, but its raw contract
also contains unresolved learned maps:

```text
raw history -> stable abstract node
raw interface -> legal executable action word
delayed outcome -> responsible node pair
```

The exact table can provide the first map only when histories are already
symbolic action/output strings and the discrimination tree is authoritative.
The substantial gate later relies on raw renamed interfaces that the classical
learner allegedly lacks; that reintroduces an encoder/compiler as a second
learned edge. Either give every control the identical frozen encoder, or T49 is
testing encoder plus proposer plus verifier composition rather than one learned
counterexample-ranking edge.

## 5. Novelty and strongest-control containment

Dynamic state creation is not an uncovered function-class mechanism.

- Angluin-style active learning creates hypothesis states and refines them from
  membership/equivalence counterexamples; modern symbolic Mealy learners also
  provide query-complexity bounds ([active symbolic Mealy learning](https://arxiv.org/abs/2509.14694)).
- Partition refinement and CEGAR create distinctions or predicates exactly
  when an abstract behavior admits a counterexample.
- PSR/OOM algorithms enlarge or change predictive tests/bases when the current
  predictive state is insufficient.
- Recurrent/meta-RL and a transcript-conditioned Transformer can simulate the
  table and proposal policy; episodic memory can append nodes, witnesses, and
  provenance with the same bytes and retrieval operations.
- Growing structure is already an explicit continual-learning mechanism, for
  example [Bayesian depth/width adaptation](https://proceedings.mlr.press/v235/thapa24b.html),
  [self-composing growable continual-RL policies](https://proceedings.mlr.press/v235/malagon24a.html),
  and current [network expansion for deep RL](https://proceedings.mlr.press/v267/kang25c.html).
- Counterexample-guided LLM refinement is itself current prior art:
  [Counterexample Guided Learning in the Large](https://arxiv.org/abs/2606.11521)
  reports verifier-driven regex induction, while
  [agentic automata learning](https://arxiv.org/abs/2606.16576) directly studies
  experiment selection, evidence integration, and hypothesis construction.

Consequently the exact graph/table is a strong control, not an architecture
advantage. The residual claim can only concern amortized acquisition cost,
noninterference, or physical implementation. Noninterference in the symbolic
table is not enough: learned routing can still alter which node is reached, and
an append-only episodic-memory control obtains the same local-write property.

## 6. Substantial-gate audit

The proposed gates are not yet a coherent preregistration.

1. **The wrong primary control.** Beating a generic recurrent/meta-RL proposer
   by `2x` is insufficient if an exact active automata learner with the same raw
   encoder and verifier is cheaper. The primary comparator must be the best
   cost-aware classical counterexample search, followed by learned controls.
2. **Exact and approximate targets are mixed.** “Reach the same exact quotient”
   and “90% of the exact learner's final model accuracy” are different tasks.
   Exact recovery is binary; an approximate gate needs a frozen horizon-weighted
   behavioral loss or policy-regret metric.
3. **The raw-interface comparison is asymmetric.** A classical learner cannot
   be denied the candidate's encoder. Encoder training/inference must either be
   common or separately attributed.
4. **Interaction savings do not imply regret savings.** A `2x` query reduction
   supplies no `30%` decision-regret result without a frozen relation between
   diagnostic actions, their opportunity cost, exploitation decisions, and the
   task horizon.
5. **The protected-state gate is too easy in the exact table.** Append-only
   disjoint state births give zero table regression by construction. End-to-end
   protected behavior must include routing, retrieval collisions, memory
   pressure, consolidation, and finite-capacity eviction.
6. **The shuffle control conflicts with exact validation.** A counterexample
   shuffled to the wrong node pair should be rejected by the exact validity
   check, producing no causal collapse. The useful ablation is random versus
   learned ordering among *valid* candidate tests, or removal of the ranking
   score while keeping validation identical.
7. **One rule across three families is undefined.** Automata words, causal
   interventions, and software-state experiments need a shared typed test
   language. Family-specific compilers would contain additional intelligence;
   without them “one shared rule” is not measurable.

The gates are simultaneously too easy where the symbolic table guarantees the
result and potentially impossible where they demand exact cross-domain recovery
against an optimal algorithm. They must be rebuilt after the next theorem
freezes the actual comparison class and cost objective.

## 7. Exactly one next theorem object

The sole next object should be an **amortized counterexample-ranking theorem
with a no-shared-prior lower bound**.

Freeze a finite legal candidate set `W`, execution costs `c(w)`, exact validator,
and a context `X` available to every method. For each refinement problem let
exactly one candidate `W*` be the first valid distinguishing test. Compare total
cost

\[
C_{total}=
{C_{meta}\over N_{deploy}}
+\mathbb E\,C_{search}
\]

for learned and classical rankings.

The theorem should have two parts within this one object.

1. **No-shared-prior boundary.** If `W*` is uniform and independent of `X`
   after arbitrary renaming, every ranking has expected position `(\lvert
   W\rvert+1)/2`; meta-training cannot improve expected search cost except by
   changing the legal information interface.
2. **Shared-prior achievability.** If `p(w\mid X)` is stable across systems, the
   Bayes cost-optimal ordering is decreasing `p(w\mid X)/c(w)`. Bound the learned
   ranker's excess expected search cost in terms of its estimation error and add
   `C_meta/N_deploy`. The strongest control receives the same `X`, prior family,
   candidate set, validator, and costs.

This theorem isolates exactly one learned edge—ranking already legal,
enumerated, exactly checkable tests. If it cannot show a strict amortized cost
advantage over the Bayes/classical ranking after meta-training is charged,
close T49 without implementation.

## 8. Disposition

- **Mathematical validity — HOLD:** retain the corrected sound-refinement lemma
  and the fixed-horizon chain counterexample.
- **Novelty — NO-GO:** counterexample-driven state refinement and growing
  structure are established mechanisms.
- **Architecture gain — NO-GO:** exact tables, L-star/CEGAR/PSR, recurrent
  controllers, and episodic memory contain the proposed function; no acquisition
  or physical separation is proved.
- **Experiment admissibility — NO-GO:** zero CPU/model/GPU runs. Admit only the
  single amortized-ranking theorem above.
