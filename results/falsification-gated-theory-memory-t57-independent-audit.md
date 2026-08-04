# Independent adversarial audit: falsification-gated theory memory (T57)

Date: 2026-08-01  
Scope: theorem, resource, distinctness, and experiment-gate audit; no experiments or resources used

## Verdict

**RETAIN AS A CONTROL / ACCEPTANCE PROTOCOL; NO-GO AS A DISTINCT UPDATER; NO RUN.**

The useful part of T57 is procedural: freeze a proposed symbolic edit before
future data, test it across randomized contexts, require an intervention, charge
its description cost, protect regressions, and retain provenance. That is good
research hygiene. It is not a new learning operator. The updater is a synthesis
of Bayesian/MDL program induction, DreamCoder-style neural proposal and library
learning, online predicate invention, active experimentation, and
CEGAR/CEGIS-style propose-check-counterexample refinement.

The conjunction may be a worthwhile **governance protocol** if formalized, but
prequential cross-context interventional acceptance does not create a theoretical
separation from those families. Current work makes the collision especially
direct: [dynamic predicate invention](https://arxiv.org/abs/2602.17217) already
uses online Predict-Verify-Refine and reusable invented predicates;
[Kintsugi](https://arxiv.org/abs/2605.09487) performs localized typed executable
KB edits behind deterministic validation and protected-regression gates; and
[ESBM](https://arxiv.org/abs/2606.07127) combines typed predicates and mechanism
memory with adaptive questions and action-intervention probes.

Only the first line of T57.1 is an exact theorem as written. T57.2 is a
stochastic-domination result, not the stated geometric-distribution result.
T57.3 is a codelength opportunity under missing code assumptions, not a theorem
and not a realized predictive advantage. The raw binding `b_t` remains an oracle
containing most of the hard problem.

## Theorem scorecard

| Claim | Status | Required disposition |
|---|---|---|
| Fixed-class Bayesian mixture log-loss bound | **Correct**, for one fixed countable class, fixed proper prior, and joint sequence likelihood | Retain, but state that it does not yet cover a changing grammar, finite active-set pruning, or data-adaptive edit selection |
| A grammar shortening by `Delta` lowers the oracle-relative ceiling by approximately `Delta` | **Not established** | Replace with the exact prior-penalty difference, including grammar normalizer or use one global prefix-free hierarchical prior |
| Failures until first correct proposal are geometric when success probability is at least `q` | **False as stated** | Replace “geometrically distributed” by “stochastically dominated by a Geometric(`q`) random variable”; define whether the count includes the successful attempt |
| Nonzero proposer recall plus an `alpha` verifier is enough | **False for lifetime usefulness and generally false for lifetime false-accept control** | Require conditional recall bounded away from zero, test power, bounded costs, and an anytime-valid multiple-testing construction |
| Scratch code is at least `kL`; reuse code is at most `log binom(M,k)+binding+O(k)` | **Unsupported except for an unordered subset under strong assumptions** | Encode order, repetition, wiring, types, parameters, library construction, and binding explicitly |
| Mixture certificate translates directly into large sequential log-loss advantage | **False** | Say only that the comparator-complexity term in an upper bound may fall by the net codelength saving |
| One expected-direction action response is a causal witness | **Insufficient** | Define the intervention estimand, randomization, positivity, outcome test, context invariance, and adaptive-policy correction |

## T57.1: exact correction to the prior and grammar claim

The proof

\[
-\log P_G(D)+\log P_{h^*}(D)\le -\log \pi_G(h^*)
\]

is valid because `P_G(D) >= pi_G(h*) P_h*(D)`. It applies to a fixed class and
a fixed prior. It does not by itself certify a learner that changes `G_t`,
constructs candidates from prior failures, prunes to a finite active set, or
re-estimates a binding.

For the paper's stated normalized description-length prior, write

\[
\pi_G(h)=\frac{e^{-\ell_G(h)}}{Z_G},
\qquad
Z_G=\sum_{h\in\mathcal H_G}e^{-\ell_G(h)}.
\]

Then

\[
-\log\pi_G(h)=\ell_G(h)+\log Z_G.
\]

If `G' = G plus g` shortens the same comparator by `Delta`, the new-minus-old
comparator penalty, after separately charging the grammar edit, is exactly

\[
-\Delta
+\bigl(\log Z_{G'}-\log Z_G\bigr)
+\ell(g)
+\Delta\ell(\text{binding/version}).
\]

Therefore an improvement is certified only when the shortening exceeds the edit,
binding/version, and normalizer penalties. The unknown normalizer can erase or
reverse the claimed gain. Adding many duplicate or nearly equivalent programs
can also dilute useful prior mass while making some textual descriptions short.
Canonicalization or an explicit treatment of semantic equivalence is necessary.

The clean repair is not to call the normalizer “approximately” harmless. Define
one fixed, prefix-free hierarchical code over complete grammar/edit histories and
theories before seeing the sequence:

\[
L(G,h)=L(G)+L(h\mid G),
\qquad
\sum_{G,h}e^{-L(G,h)}\le 1.
\]

Allocate any leftover Kraft mass to a dummy outcome. Then the global mixture
bound pays `L(G) + L(h|G)` exactly, without comparing separately renormalized
grammar priors. If multiple edit histories denote the same executable theory,
either use a canonical history or pay/sum over them explicitly. If lengths are
in bits, use `2^{-L}`; if they are in nats, use `e^{-L}` consistently.

Further corrections are mandatory:

- Choose one-part Bayes coding or two-part MDL. `-log P_G(D)` already integrates
  theory identity through the prior. Adding another theory-identification code
  double-counts it.
- The runtime object is a finite active set `H_t`, whereas the theorem uses a
  countable full mixture. Pruning and renormalizing change the prior. Retain
  inactive mass, or charge deletion, resurrection, and switching in a global
  edit-history code.
- A newly accepted grammar can improve only future conditional loss. It cannot
  retroactively reduce the already incurred prequential loss. A valid dynamic
  guarantee needs a global mixture over grammar histories or an explicit
  switching/selection-regret theorem.
- Freeze the candidate, its binding, and all fitted parameters before `D+`.
  Revising any of them after inspecting the audit stream is leakage.
- The plug-in prediction `p_h(y | b_t(o),a)` discards uncertainty about `b_t`.
  The honest model is a joint posterior or mixture over bindings and theories;
  otherwise both confidence and codelength are overstated.

## The acceptance rule does not yet control false acceptance

Future prequential evaluation is necessary, but it is not sufficient for the
paper's `alpha` language. The proposer may emit many candidates after many
failures, retain them in quarantine, choose diagnostic actions adaptively, and
stop when one passes. A per-candidate or per-look level `alpha` does not control
the lifetime probability of any false acceptance. With `m` independent tests it
is `1-(1-alpha)^m`, approximately `m alpha` for small `alpha`; adaptive dependence
and optional stopping require an explicit sequential construction.

One repair is to define, conditional on the frozen past, a null predictive law
and an alternative code-weighted family whose likelihood-ratio process is an
e-process. Candidate codes must satisfy a Kraft inequality, and promotion can
occur only when the aggregate e-value crosses `1/alpha`. Equivalently, use a
predeclared summable `alpha_i` schedule or an appropriate online FWER/FDR method.
The proof must include candidate selection, repeated looks, quarantine returns,
and optional stopping. Merely subtracting `ell(g)` in an MDL score does not
establish this unless the score is explicitly derived as such a valid
likelihood-ratio/e-process.

The audit data distribution also depends on `Q_t`. Register or randomize the
diagnostic policy before outcomes and use the conditional likelihood under that
adaptive design, including action propensities where required. Otherwise the
proposer can search for a context/action slice on which a wrong edit compresses.
Diagnostic actions, resets, unsafe failures, opportunity regret, and delayed
outcomes belong in the cost ledger.

“Two independently randomized surface contexts” is a useful stress test, not a
generalization theorem. The paper must define the meta-distribution over
contexts, independence unit, per-context sample size, minimum contribution, and
desired confidence for a new context. Two contexts can share the same latent
confounder or benchmark generator.

Finally, “an action changes a prediction in the expected direction” is not a
causal witness. At minimum the protocol needs a legal randomized intervention,
positivity, consistency/no-interference assumptions, an observed outcome shift
relative to a control, and replication across held-out contexts. The test should
compare causal and correlational alternatives; changing the model's prediction
without the predicted outcome change proves nothing.

## T57.2: exact proposer correction

Let `T` be the number of relevant proposal opportunities up to and including the
first correct proposal. If

\[
\Pr(\text{correct proposal at }i\mid \mathcal F_{i-1},T\ge i)\ge q,
\]

then `T` is stochastically dominated by a Geometric(`q`) random variable:

\[
\Pr(T>n)\le (1-q)^n,
\qquad
\mathbb E[T]\le 1/q.
\]

It is exactly geometric only for iid Bernoulli trials with one fixed success
probability. If `N` means failures **before** success, then
`E[N] <= (1-q)/q`; the paper currently mixes these conventions.

“Nonzero measurable recall” is not enough for an advantage. It supplies eventual
discovery only in an infinite sequence of reachable, informative opportunities,
with no useful resource bound. A tiny or decaying `q` can produce prohibitive or
infinite expected wall time, environment actions, proposer calls, and audit cost.
A useful statement also needs:

- a uniform lower bound on conditional recall over reachable failure states;
- verifier power, not only false-positive control;
- observable consequences and enough safe interventions to distinguish edits;
- bounded proposal, execution, audit, and rollback costs; and
- positive expected net lifetime value after discovery delay and wrong-edit
  contamination.

Versioning makes rollback possible; it does not prevent a falsely accepted edit
from changing routing, future data collection, or dependent theories before the
regression suite detects it.

## T57.3: the compositional codelength bound is not valid as written

The scratch lower bound `kL` holds only if every one of the `k` mechanisms has
minimum conditional prefix codelength `L` given all previous mechanisms and no
shared structure can compress them jointly. If `L` is an average or a convenient
per-mechanism code, the expression is an encoding cost, not a lower bound.

Likewise, `log binom(M,k)` encodes only an unordered subset of `k` distinct
library entries. Actual composition normally requires more:

- unordered distinct selection: `log binom(M,k)`;
- ordered distinct roles: `log (M)_k`;
- ordered selection with repetition: `k log M`;
- plus a self-delimiting code for `k`, the composition graph/wiring, types,
  parameters, versions, interfaces, and the raw-to-symbol binding.

Arbitrary wiring is not `O(k)` in general; it can require `O(k log k)` or more.
The binding term cancels only if scratch and reuse literally use the same binding
code. Library reuse can instead increase alignment and interface cost.

The honest lifetime comparison is

\[
\Delta_{\text{code}}
=L_{\text{scratch}}(h)
-\Bigl[L(\mathcal L)
+L(S,\Gamma,\theta,\beta\mid\mathcal L)\Bigr],
\]

where `L` is the stored library, `S` the selection with roles/repetitions,
`Gamma` the composition graph, `theta` parameters, and `beta` the binding. The
library construction, storage, search, retrieval, and execution costs must be
amortized over the environments that use it. If the library is treated as fixed
side information for a transfer task, every matched control must receive the
same side information and its acquisition cost must still appear in the lifetime
ledger.

Even a positive `Delta_code` changes only the **comparator-complexity term** in a
mixture upper bound. It is not a direct realized log-loss gain. The new grammar
may make identical predictions, dilute prior mass, be discovered too late, or
increase mixture loss. Replace the last sentence of T57.3 with:

> Under a single valid hierarchical code, reusable composition may reduce the
> oracle comparator penalty by its net codelength saving. This is an opportunity
> for lower future mixture regret, not a guarantee of realized predictive gain.

## Raw binding and primitive invention remain the load-bearing oracle

Adding a predicate name is not primitive invention. The learner must determine
an observation equivalence class or encoder, arity and types, object identity
through time, action applicability, intervention semantics, and alignment of the
new primitive across contexts. Under arbitrary partially observed raw processes,
these objects are not identifiable: observationally equivalent latent models can
assign different primitive meanings and intervention effects.

`b_t` therefore cannot remain a side function while the updater is credited with
raw abstraction formation. The paper must do one of two things:

1. label binding, primitive vocabulary, object persistence, action legality, and
   error localization as oracle inputs and restrict the claim to symbolic model
   revision; or
2. specify a restricted raw-input family—such as a finite object-centric POMDP
   with identifiable emissions, separation margins, known intervention channel,
   and bounded aliasing—and provide a joint binding/theory algorithm with an
   identifiability and acquisition bound.

[Beyond identifiability](https://arxiv.org/abs/2603.25796) does not fill this
gap. Its finite-sample recovery concerns latent causal factor models under
specific mixing, sub-Gaussian observation, and multiple-intervention-environment
assumptions. It cannot be imported as a theorem for arbitrary online raw POMDP
binding. Conversely, the assumptions in [dynamic predicate
invention](https://arxiv.org/abs/2602.17217)—deterministic fully observable
ground atoms, supplied metarules/types and background predicates—show exactly
how much structure current symbolic demonstrations receive.

## Hidden resource and oracle bill

The present cost ledger omits or underspecifies:

- perception, segmentation, tracking, entity persistence, ontology alignment,
  and calibration of `b_t`;
- primitive/action invention, typing, legality, and executable semantics;
- residual localization, candidate synthesis, type checking, compilation, and
  equivalence/canonicalization of duplicate programs;
- likelihood evaluation, posterior normalization, full-mixture approximation,
  active-set pruning, quarantine, resurrection, and switching;
- intervention choice, real actions, resets, safety checks, opportunity cost,
  delayed effects, and context generation;
- lifetime sequential-test calibration, null/alternative specification,
  multiple testing, optional stopping, and protected-regression evaluation;
- library indexing, storage, retrieval, wiring, parameter fitting, and execution;
- planner/control compute, neural proposer calls, outer-loop training data and
  compute, state bytes, provenance bytes, and read/write traffic.

These are not implementation trivia. Several are precisely the operations that
could make T57 outperform a generic learner. All must be equalized against the
strongest matched symbolic and neural controls.

## Distinctness against current primary work

The exact combination is not the same artifact as any single cited system, but
combining occupied blocks is not an updater-level novelty claim:

- [DreamCoder](https://arxiv.org/abs/2006.08381) already combines Bayesian
  program learning, a learned proposal/search policy, library abstraction, and
  compositional reuse. [Bayesian Program Learning by Decompiling Amortized
  Knowledge](https://arxiv.org/abs/2306.07856) makes the neural-proposal/library
  coupling still more direct.
- [Dynamic predicate invention](https://arxiv.org/abs/2602.17217) already puts
  prediction, verification, local repair, and reusable predicate invention in
  an online loop.
- [Counterexample Guided Learning in the
  Large](https://arxiv.org/abs/2606.11521) and [property-guided LLM program
  synthesis](https://arxiv.org/abs/2605.16142) use fallible neural proposals,
  external verification/counterexamples, and iterative symbolic repair.
- [Kintsugi](https://arxiv.org/abs/2605.09487) covers typed local executable edits,
  deterministic gates, dependency-like localization, and protected regression;
  [ESBM](https://arxiv.org/abs/2606.07127) adds explicit mechanism prediction and
  active world-model intervention probes.
- [Self-Revising Discovery Systems](https://arxiv.org/abs/2606.01444) explicitly
  describes typed schema revision, provenance, verification, and an MDL gate.
  [Adaptive state-action abstractions](https://arxiv.org/abs/2606.06123) already
  gives a certificate-driven granularity switch, while [causal and compositional
  abstraction](https://arxiv.org/abs/2602.16612) formalizes what it means for
  interventions and mechanisms to survive abstraction.
- [SkillRise](https://arxiv.org/abs/2607.26784) further occupies learned
  cross-task extraction, revision, and reuse of an explicit abstraction-like
  artifact, though without T57's verifier.

The defensible claim is therefore narrow:

> T57 specifies a potentially useful **acceptance discipline** for symbolic
> library edits: future-only prequential evidence, randomized surface contexts,
> interventional testing, explicit code cost, provenance, and rollback.

That discipline could improve reliability over ungated DreamCoder- or
predicate-invention-style systems. It does not yet prove better learning,
different asymptotics, raw abstraction formation, or lower full cost. Treat it
as a strong control or systems-safety layer, not the mechanism proposed to close
the intelligence gap.

## Exact corrections required in T57

1. Replace the normalized-prior commentary in T57.1 with the exact normalizer
   equation above, preferably using one fixed hierarchical Kraft-valid prior over
   grammar histories and theories.
2. State separately the fixed full-mixture theorem and any finite active-set or
   switching approximation. Remove the suggestion that a changed grammar
   automatically inherits the fixed-class certificate.
3. Specify one-part versus two-part coding and jointly integrate binding
   uncertainty; freeze the whole candidate before future audit data.
4. Derive the promotion threshold from an anytime-valid e-process, summable
   alpha schedule, or other explicit lifetime multiple-testing rule. Include
   adaptive proposal and diagnostic-action selection.
5. Replace “geometrically distributed” with stochastic domination, define the
   proposal count, and delete “nonzero recall is enough.” State a uniform `q`,
   verifier power, and complete expected-cost condition.
6. Replace T57.3 with a prefix code for the library, ordered/repeated selection,
   wiring, types, parameters, version, and binding. Describe only a possible
   reduction in the comparator penalty, not direct actual log-loss advantage.
7. Define causal acceptance by an estimand and observed randomized-intervention
   evidence, not a prediction moving in the expected direction.
8. Define the surface-context sampling distribution and a real cross-context
   generalization test; two contexts alone receive no theorem status.
9. Either expose `b_t` and primitive semantics as oracle inputs or give a
   restricted raw family plus identifiability/acquisition theorem.
10. Complete the resource ledger and name an exact matched online
    predicate-invention/Bayesian program learner, not only broad baselines.

## Experiment gate

**No experiment is earned.** The current paper correctly says no run, and this
audit strengthens that gate. A neural pilot would primarily test the quality of
the hidden binding/compiler, candidate generator, and benchmark scaffolding; it
would not test the three claimed theoretical gains.

The next earned work is paper-only:

1. prove the hierarchical dynamic-mixture/switching statement;
2. prove lifetime false-accept control under adaptive candidates and actions;
3. state the restricted raw family and either prove binding identifiability or
   explicitly mark it oracle-assisted; and
4. replace the reuse opportunity with a complete uniquely decodable code and a
   full resource comparison.

Only after those survive another audit would a tiny finite symbolic sanity check
be admissible. It should exhaustively enumerate theories, grammars, bindings, and
audit streams so the code identities and false-accept bound can be checked. It
should not start with a neural proposer, raw pixels, an LLM benchmark, rented
hardware, or a capability claim.

## Final classification

- **Useful updater discipline:** yes.
- **Distinct updater architecture:** no.
- **Exact theorem currently valid:** fixed-class Bayesian mixture inequality
  only.
- **Normalized description-length grammar gain:** not established.
- **Proposer theorem:** correctable to stochastic domination; usefulness claim
  overreaches.
- **Compositional transfer theorem:** not established; opportunity calculation
  only after a complete code.
- **Raw primitive invention:** unsolved and currently hidden in `b_t`.
- **Any experiment earned now:** no.
