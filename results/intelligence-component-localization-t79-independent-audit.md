# T79 independent audit: intelligence-component localization

**Verdict: REVISE, then RETAIN as a project admission gate.**

The proposed diagnostic is useful because it forces a candidate to say what intelligence failure it changes and to demonstrate a controlled, material effect. It is not yet safe as written. The one-step identity needs explicit information-order and decision-class assumptions; it does not by itself decompose sequential return under policy-induced distribution shift; and several proposed interventions do not uniquely identify the component named in the table. The literature also already occupies both the knowing-versus-using distinction and the general causal-intervention methodology. T79 should therefore be retained as an internal discipline after the corrections below, not presented as a novel theorem or diagnostic framework.

## 1. Exact decomposition and its validity conditions

Let \(H\) be all information available at the audited decision, \(Y\) the prediction or action target, \(K\) an encoded state, \(d\) the deployed rule, and \(\ell\) an integrable loss. For a common class of measurable decision rules, define

\[
R^*(X)=\inf_f \mathbb E[\ell(f(X),Y)].
\]

Whenever these quantities are finite,

\[
\mathbb E[\ell(d(K),Y)]-R^*(H)
=\bigl(R^*(K)-R^*(H)\bigr)
+\bigl(\mathbb E[\ell(d(K),Y)]-R^*(K)\bigr).
\]

The equality is algebraic. The two terms are nonnegative only under additional conditions:

1. **Information order.** It is sufficient that \(K\) is a Blackwell garbling of \(H\): \(K\) is sampled from a kernel \(Q(dk\mid H)\) with \(Y\perp K\mid H\). Deterministic \(K=E(H)\) is a special case. Stochastic encoders are therefore allowed, but their randomness must not carry target information unavailable in \(H\).
2. **Complete source information.** Fixed weights, retrieved external state, prior lifetime state, tool results, and any other side information used to produce \(K\) must be included in \(H\), or the comparison must condition on them. Otherwise \(R^*(K)-R^*(H)\) can be negative. For example, if balanced \(Y\) is independent of \(H\), but a leaked side channel sets \(K=Y\), then under 0--1 loss \(R^*(H)=1/2\) and \(R^*(K)=0\).
3. **Matched decision classes.** The infimum defining \(R^*(K)\) must range over a class that contains the deployed rule \(d\). If the probe and deployed policy use different function, data, or compute classes, the second term is not guaranteed nonnegative and does not have the claimed meaning.
4. **Well-defined risk.** Bounded loss is sufficient but not necessary; finite integrable risks suffice. The infimum need not be attained.

Under these assumptions, Blackwell monotonicity gives \(R^*(H)\le R^*(K)\), and the definition of the infimum gives \(R^*(K)\le \mathbb E\ell(d(K),Y)\). The current proof, which relies only on deterministic composition, is too narrow for stochastic \(K\) and silent about target-correlated side channels.

Empirical probes do not observe \(R^*\). They provide model-class- and sample-dependent upper bounds with optimization and generalization error. A measured negative residual is therefore possible even when the population terms are nonnegative. T79 should require held-out estimation, uncertainty intervals, matched probe capacity, and explicit wording that the reported quantities estimate rather than equal the ideal decomposition.

## 2. One-step risk does not decompose sequential intelligence

The identity is valid on a fixed reference distribution over \((H,K,Y)\). In a sequential environment, replacing a planner or policy changes future actions, state visitation, observations, and targets. The learned and intervened systems then induce different occupancy measures. Differences in return combine component quality with coverage, compounding error, control stability, and distribution shift; they are not the two algebraic terms above.

T79 should split its claims into two levels:

- **Fixed-distribution audit:** choose an exogenous audit distribution \(\mu\) over reset states, histories, and queries. Freeze the checkpoint and compare all one-step components on the same \(\mu\).
- **Closed-loop causal audit:** from matched initial-state distributions and paired environment seeds, intervene on one mechanism and measure total return. Call this a total causal effect unless a policy-indexed performance-difference or occupancy decomposition supports a narrower attribution.

The current statement “static pass, closed-loop replacement fail implies planning or distribution shift” is only a hypothesis list, not a localization. A policy change can also expose control instability, action-interface errors, compounding approximation error, or missing coverage.

## 3. The present state variable conflates distinct failures

A single \(K\) cannot distinguish initial formation, retention, retrieval, and use. A safer nested model is

\[
H \longrightarrow M \longrightarrow Z \longrightarrow A,
\]

where \(M\) is retained memory/state, \(Z=G(M,Q)\) is the information retrieved or exposed for the current query \(Q\), and \(A=d(Z)\) is the deployed output. Under the corresponding Markov/garbling conditions,

\[
\begin{aligned}
\mathbb E[\ell(A,Y)]-R^*(H)
={}&[R^*(M)-R^*(H)] \\
&+[R^*(Z)-R^*(M)] \\
&+[\mathbb E\ell(A,Y)-R^*(Z)].
\end{aligned}
\]

These terms can be interpreted, with care, as formation/retention, retrieval/exposure, and use gaps. This still does not algebraically separate planning from readout: an unconstrained optimal decoder from \(Z\) absorbs both. That distinction requires explicit planner state and surgical interventions.

## 4. Corrected causal intervention matrix

The interventions should be interface-matched and crossed rather than interpreted one at a time.

| Contrast | Held fixed | Changed | Supported inference |
|---|---|---|---|
| Learned \(M\), normal retrieval, learned planner | Audit distribution and full system | Nothing | Baseline only |
| Same learned \(M\), forced retrieval of the verified relevant item | Planner, readout, memory contents, input interface | Router/exposure | Retrieval effect conditional on the item being retained |
| Same retrieved \(Z\), oracle planner using **only \(Z\)** | Representation, retrieval, action decoder/executor | Planning computation | Planning effect conditional on available information |
| Same frozen plan, oracle readout/executor | Representation, retrieval, planner output | Readout/execution | Readout or execution effect |
| Oracle-correct \(M\) encoded through the same interface | Retrieval and downstream computation | Stored representation | Representation effect, subject to interface compatibility |
| Immediate versus delayed audit on identical replayed \(\mu\), plus restored-checkpoint control | Query/task distribution and audit procedure | Intervening learning/time | Retention effect rather than ordinary task or coordinate drift |
| Fixed-\(\mu\) versus paired closed-loop versions of every row | Component intervention | Occupancy feedback | Distribution/compounding interaction, not a pure component |

Required interpretation corrections:

- A strong probe establishes decodability by that probe, not natural use or causal relevance. Probe capacity, supervision, data, and compute must be matched and reported; control tasks and multiple probe classes are required.
- Oracle-policy success is not evidence of a pure use gap if the oracle sees true environment state, transitions, or rewards absent from \(K\). For isolation it must be Bayes-optimal conditional only on the same \(K\) or \(Z\).
- Oracle-knowledge injection can change format, salience, scale, and planner interface. Use the native interface and include sham or distribution-matched injection controls.
- An immediate pass followed by a delayed fail is not sufficient evidence of forgetting. Repeat on the identical audit distribution and train a fresh probe at each checkpoint. Frozen-probe failure with fresh-probe success indicates representational reparameterization, not necessarily information loss. Restoring the earlier checkpoint supplies a causal control.
- Probing during training changes the representation. Diagnostic probes should ordinarily be fit on frozen checkpoints.
- Factorial interactions may be real. Without structural assumptions, intervention order or an explicitly chosen attribution convention, the total gain need not have a unique additive component allocation.

## 5. Direct prior art and novelty boundary

The broad diagnostic territory is already occupied:

- Blackwell's comparison of experiments supplies the classical information-order result needed for the nonnegative representation term: [Equivalent Comparisons of Experiments](https://doi.org/10.1214/aoms/1177730497).
- [Probing Representation Forgetting](https://arxiv.org/abs/2203.13381) directly uses optimal probes before and after continual tasks to distinguish representational loss from downstream forgetting.
- [The World Model Remembers, The Actor Forgets](https://arxiv.org/abs/2607.19749) is an especially close sequential precedent: world-model probes retain knowledge while the actor collapses, and frozen-model self-imitation can restore behavior.
- [Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in LLM Finetuning](https://arxiv.org/abs/2607.08393) explicitly formalizes a “Knowing--Using Gap” and uses activation self-patching to recover a large fraction of oracle headroom. This is a direct 2026 collision with T79's framing, not merely adjacent work.
- [Amnesic Probing](https://aclanthology.org/2021.tacl-1.10/) shows why ordinary probe accuracy does not establish causal task use and introduces information-removal interventions; [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/) addresses probe memorization and selectivity.
- [Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://www.jmlr.org/papers/v26/23-0058.html) already provides a general formal account of mechanism interventions, patching, mediation, scrubbing, tracing, and erasure. [How Reliable are Causal Probing Interventions?](https://aclanthology.org/2025.ijcnlp-long.47/) further documents reliability and selectivity/completeness tradeoffs.
- [Rethinking Continual Experience Internalization](https://arxiv.org/abs/2606.04703) already studies progressive experience loss and stepwise corrective injection. [Mechanistic World Models](https://arxiv.org/abs/2607.12474) is relevant conceptual context, but it is a blueprint rather than the strongest empirical localization precedent.

Accordingly, the algebra, the knowing-versus-using distinction, probe-based retention diagnosis, and the general idea of causal component replacement are not novelty. A future research contribution would need a new validated identification theorem, a demonstrably more discriminating intervention protocol, or a substantive intelligence mechanism discovered through the gate. T79 itself can honestly remain an internal admission rule.

## 6. Does the substantial gate remain intelligence-first?

Mostly, but not completely. Requiring an order-one capability effect on unseen world families, causal ablation, protected-capability checks, persistence, and a complete cost ledger is consistent with the charter's intelligence-first objective. Cost accounting acts as a control rather than the optimized object.

Two clauses drift toward efficiency-first selection:

1. Making “at least \(2\times\) faster acquisition” mandatory rejects a learner that acquires a genuinely new, durable capability at the same speed. Acquisition efficiency should be an alternative route or a secondary claim-specific requirement, not a universal intelligence gate.
2. The later aspiration that a smaller learner beat a larger frozen system is a useful scale-efficiency stress test, not a necessary definition of intelligence progress.

The requirement that every candidate modify a single term also risks false purity. Coupled mechanisms may genuinely improve representation, retrieval, and planning together. Replace it with: predeclare the affected channels; run the strongest feasible factorial controls; show causal mediation of the end-to-end capability gain; and report unresolved interactions rather than forcing a unique label.

Recommended substantial admission rule:

> Admit a candidate only if it produces a preregistered, durable, causally supported capability gain on held-out world families: normally at least 10 absolute points or 30% regret reduction against the strongest complete-cost-matched learned control, outside uncertainty; with no material protected-capability loss; and with a complete cost ledger. Faster acquisition may independently qualify an efficiency claim, but is not required for a capability claim. The candidate must predeclare affected representation, retention, retrieval, planning, and readout channels and localize the gain as far as the intervention design permits.

## 7. Required revision checklist

Before T79 governs another run:

1. State the Blackwell/Markov condition for stochastic \(K\), include all side information in \(H\), and align decision classes.
2. Distinguish ideal Bayes risks from empirical probe estimates and require uncertainty plus probe controls.
3. Separate fixed-distribution localization from closed-loop total effects and add an occupancy/distribution-shift analysis.
4. Replace the single \(K\) with at least retained \(M\) and retrieved \(Z\), and distinguish planning from readout by intervention rather than algebra.
5. Replace the current outcome table with the crossed, interface-matched intervention matrix above.
6. Cite the direct knowing--using, representation-forgetting, causal-probing, and causal-abstraction precedents; make no novelty claim for T79 itself.
7. Make faster acquisition optional for a capability claim, and permit coupled mechanisms with explicit mediation and interaction accounting.

No experiment should be admitted from T79 in its current form. After these revisions, retain it as an intelligence-first project gate.
