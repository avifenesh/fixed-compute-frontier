# T57 falsification-gated theory memory — abstraction acceptance discipline

Date: 2026-08-01  
Status: **RETAINED AS CONTROL / ACCEPTANCE PROTOCOL; DISTINCT UPDATER REJECTED; NO RUN**

## 0. Result in one sentence

A useful acceptance discipline can freeze a proposed executable edit, evaluate
it only on future randomized interventional data, charge a uniquely decodable
description, and retain provenance and rollback. This does not define a
distinct updater. Only the fixed-class Bayesian mixture inequality is exact as
initially stated; the proposal and reuse results require narrower conditions,
while raw symbol grounding remains unsolved.

## 1. The computational object

The persistent state is not a free latent vector and not prose. It is

\[
Z_t=(G_t,\mathcal H_t,\mu_t,\mathcal E_t),
\]

where:

- `G_t` is a typed grammar of primitive and invented predicates;
- `H_t` is a finite active set of executable transition/reward theories;
- `mu_t` is a posterior over those theories; and
- `E_t` is provenance: observations and interventions that support or refute
  each theory and invented predicate.

For raw observation `o_t`, action `a_t`, and next outcome `y_t=(o_{t+1},r_t)`,
the read path is the Bayesian predictive mixture

\[
p_t(y\mid o_t,a_t)
=\sum_{h\in\mathcal H_t}\mu_t(h)
 p_h(y\mid b_t(o_t),a_t),
\]

where `b_t` is the current probabilistic binding from surface features to the
primitive symbols consumed by the theories. The posterior update is

\[
\mu_{t+1}(h)
\propto \mu_t(h)p_h(y_t\mid b_t(o_t),a_t).
\]

The deployed neural model has three bounded roles:

\[
(C_t,Q_t)=P_\phi(o_{\le t},a_{<t},y_{<t},Z_t),
\]

where `C_t` is a small set of proposed grammar/theory edits and `Q_t` is a
distribution over legal diagnostic actions. The executor, Bayesian update,
provenance graph, and acceptance test are not declared by the proposer.

## 2. Exact write rule

An edit `c` may introduce a reusable subprogram `g`, alter a binding, split a
predicate, or replace one local theory fragment. It is evaluated on future
prequential data, not the residual that proposed it.

Let `L_G(D)` be the cumulative predictive codelength of data `D` under the
Bayesian mixture induced by grammar `G`. Freeze the edit, binding, fitted
parameters, diagnostic policy, and test before collecting the audit stream
`D^+`. A schematic score is

\[
\Delta_t(g)
=L_{G_t}(D^+)-L_{G_t\oplus g}(D^+)-\ell(g)-\ell(\text{binding})
>\tau_t.
\]

This score is not itself a lifetime false-accept guarantee. Repeated adaptive
proposals, diagnostic actions, optional stopping, and quarantine returns require
an anytime-valid e-process, a predeclared summable alpha schedule, or another
explicit online multiple-testing construction. The test likelihood must be
conditional on the registered adaptive design.

Randomized surface contexts are a stress test only after their sampling
distribution, independence unit, and per-context evidence are defined. A causal
promotion requires an observed randomized-intervention outcome effect under a
declared estimand, positivity, consistency, control, and replication—not merely
a model prediction moving in the expected direction. Correlational compression
alone may remain an episodic predictor but cannot enter the causal mechanism
library.

Rejected edits remain in a bounded quarantine until their sequential evidence
threshold expires. Accepted theories are versioned; a new edit changes only
the dependency cone reachable from that edit. Global weight mutation is not
part of the online update.

This is an operational definition of an abstraction:

> a reusable subcomputation whose definition costs fewer bits than it saves in
> future interventional prediction across multiple surface contexts.

## 3. T57.1 — exact mixture-regret certificate

For a fixed countable theory class with prior `pi_G(h)>0`, define

\[
P_G(D)=\sum_h\pi_G(h)P_h(D).
\]

For every `h*` and every sequence `D`,

\[
-\log P_G(D)+\log P_{h^*}(D)
\le -\log \pi_G(h^*).
\]

**Proof.** `P_G(D)>=pi_G(h*)P_h*(D)`. Take negative logarithms. QED.

This theorem applies only to one fixed countable class, fixed proper prior, and
joint sequence likelihood. It does not automatically cover changing grammars,
finite active-set pruning, data-adaptive candidates, bindings, or switching.

For a separately normalized description prior

\[
\pi_G(h)=\frac{e^{-\ell_G(h)}}{Z_G},
\qquad
-\log\pi_G(h)=\ell_G(h)+\log Z_G.
\]

If `G'=G plus g` shortens the same comparator by `Delta`, the new-minus-old
penalty after separately charging the edit is

\[
-\Delta+(\log Z_{G'}-\log Z_G)
+\ell(g)+\Delta\ell(\text{binding/version}).
\]

The normalizer can erase the apparent saving. A clean dynamic formulation uses
one prefix-free hierarchical code over complete grammar histories and theories,

\[
L(G,h)=L(G)+L(h\mid G),
\qquad
\sum_{G,h}e^{-L(G,h)}\le1,
\]

and pays edit, binding, version, switching, and duplicate-program equivalence
costs once. Even then, a shorter comparator only lowers a term in an upper
bound on future mixture regret; it does not certify realized predictive gain.

The statement is not a reward/regret theorem. To obtain decision regret, model
error must be related to value under a frozen horizon, policy, diagnostic
action cost, and concentrability or simulation-lemma assumptions.

## 4. T57.2 — when a learned proposer can beat its own raw answer

Let `T` count proposal opportunities up to and including the first correct
proposal. If, conditional on the complete past and on reaching opportunity `i`,

\[
P(\text{correct proposal at }i\mid\mathcal F_{i-1},T\ge i)\ge q>0,
\]

then `T` is stochastically dominated by a `Geometric(q)` random variable:

\[
\mathbb E[T]\le 1/q,
\qquad
P(T>n)\le(1-q)^n.
\]

Exact geometric distribution additionally requires iid Bernoulli trials with a
fixed success probability. Nonzero recall is insufficient for usefulness: the
claim needs a uniform conditional `q`, verifier power, observable and safe
distinguishing consequences, bounded proposal/audit/rollback cost, lifetime
false-accept control, and positive expected value after discovery delay.

This answers the earlier "if the compiler is stronger, use it as the model"
objection only under those conditions. There is no stronger compiler here: a
fallible model proposes and external consequences provide new evidence. The
full bill includes proposal opportunities, diagnostic interactions, delayed
audit data, verifier power, and contamination from falsely accepted edits.

## 5. T57.3 — compositional transfer is a code opportunity, not a theorem

`log binom(M,k)` encodes only an unordered subset of distinct entries. Real
reuse must encode ordered or repeated roles, a self-delimiting `k`, wiring,
types, parameters, versions, interface maps, and raw binding. A valid lifetime
comparison is

\[
\Delta_{code}=L_{scratch}(h)
-\left[L(\mathcal L)
+L(S,\Gamma,\theta,\beta\mid\mathcal L)\right],
\]

where `L` is the acquired library, `S` selection with roles/repetitions,
`Gamma` wiring, `theta` parameters, and `beta` binding. Construction, storage,
search, retrieval, and execution must be amortized over actual reuse. A scratch
lower bound of `kL` is valid only when each mechanism has conditional minimum
prefix length `L`; it is not generally true.

Under one valid hierarchical code, reusable composition may reduce the oracle
comparator penalty by a positive net `Delta_code`. This is an opportunity for
lower future mixture regret, not a guarantee of realized log-loss or reward
gain. Evidence must still identify the binding and distinguish novelty.

## 6. The raw-grounding problem is still fatal

The updater consumes primitive predicates through `b_t`. If those predicates,
their types, action legality, or error localization are supplied by the
benchmark, most of the claimed intelligence has been handed to the learner.

Current primary results make the boundary sharp:

- online dynamic predicate invention already performs prediction-driven model
  repair and reports orders-of-magnitude sample-efficiency gains, but assumes
  deterministic full observability, ground atoms, user-selected metarules,
  hand-typed primary predicates, and background knowledge;
- finite-sample causal representation learning can recover latent causal
  variables and unknown intervention targets under explicit mixing and
  intervention assumptions;
- rate-distortion state-action abstraction already supplies a Bellman-residual
  versus bisimulation-error rule for changing granularity; and
- DreamCoder, Bayesian program learning, active automata learning, CEGAR,
  SkillRise, ALMA, and learned memory systems occupy the remaining individual
  blocks.

Thus replacing the symbolic front end with a VLM/LLM encoder is not a solution.
It relocates the unproved compiler into `b_t`. A valid continuation needs an
observable raw-input condition under which binding and new primitive creation
are identifiable, together with an online algorithm and acquisition bound.

Primary references:

- https://arxiv.org/abs/2602.17217
- https://arxiv.org/abs/2603.25796
- https://arxiv.org/abs/2606.06123
- https://arxiv.org/abs/2602.16612

## 7. Strong controls

Any pilot must give identical raw observations, interventions, audit delay,
state bytes, and theory-execution budget to:

1. the exact online predicate-invention/program learner;
2. a Bayesian program mixture with a fixed grammar;
3. a generic recurrent/meta-RL learner with the same persistent bytes and
   train-time surface randomization;
4. a full-history transformer plus textual/program memory; and
5. an oracle-predicate variant that measures the entire grounding penalty.

The decisive ablations are `no invention`, `no cross-context acceptance`, `no
intervention`, `random proposals at matched q`, and `reset the persistent
library`. A gain over PPO or a reset LLM alone is not evidence.

## 8. Substantial-result gate

The method would count only if one frozen updater, trained before the test
families are revealed, achieves all of:

- at least 30% lower cumulative decision regret than the strongest matched
  recurrent/program-memory control on held-out families;
- at least 4x fewer real environment interactions to a fixed success level;
- positive transfer under arbitrary surface-symbol and action renaming;
- positive transfer at 10x the training horizon and after raw history has left
  context;
- no more than 2% protected static-capability loss; and
- total cost including proposals, verification, planning, memory, and
  retrieval below the matched control at the attained capability.

The logarithmic codelength theorem alone does not admit an experiment. A raw
binding theorem or a deliberately restricted raw-input family with no hidden
parser is required first.

## 9. Current disposition

Independent audit rejects T57 as a distinct updater. It is a synthesis of
Bayesian/MDL program induction, DreamCoder-style proposal/library learning,
online predicate invention, active experimentation, and CEGAR/CEGIS repair.
Retain its future-only prequential, randomized-context, interventional,
code-cost, provenance, and rollback requirements as an acceptance/control
protocol.

No CPU/GPU experiment is admitted. Reopening would require a global dynamic
mixture/switching theorem, anytime-valid lifetime acceptance, a complete code
and cost ledger, and either an explicit symbolic-oracle claim or a restricted
raw family with a binding identifiability/acquisition theorem. See the
[independent audit](falsification-gated-theory-memory-t57-independent-audit.md).
