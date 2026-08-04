# Independent audit: intelligence-gap reset (T53)

Date: 2026-08-01  
Scope: theorem and research-direction audit; no experiment  
Audited artifact: intelligence-gap-reset-t53.md

## Verdict

**REVISE, then retain as a research reset. No architecture or run is admitted.**

T53 correctly identifies a hard conjunction—raw representation discovery, active disambiguation, reusable mechanisms, local revision, and retention—but it overstates this as an absent seam. Several current systems already combine large subsets. The defensible gap is narrower: no demonstrated general system jointly solves the full lifecycle under partial raw observations, weak assumptions, complete-cost accounting, and matched guarantees.

## 1. Passive causal non-identifiability

T53.1 is valid under its binary Bayes experiment. Let \(E\) be uniform on \(\{0,1\}\), and assume

\[
P(O_{1:T}\mid E=0)=P(O_{1:T}\mid E=1)
\]

for every history. Then \(I(E;O_{1:T})=0\), the posterior remains uniform, and every environment-identity decision rule has average error \(1/2\).

An explicit witness would strengthen the paper. Let hidden \(U\sim\mathrm{Bernoulli}(1/2)\). In \(E_0\), set \(X=U,\ Y=X\). In \(E_1\), set \(X=U,\ Y=U\). Passive observations of \((X,Y)\) are identical, but under \(do(X=0)\), \(E_0\) has \(Y=0\), while \(E_1\) has \(Y=U\).

Required scope corrections:

- The theorem assumes no environment-correlated side information beyond the passive history.
- If the query asks for a distribution rather than which world is present, the Bayes-optimal answer is the posterior mixture; “error \(1/2\)” applies to binary identification.
- Passive text may contain reports of experiments, multi-environment changes, or causal assumptions. Such data are passive to the model but not observational-only in the theorem's sense.
- The result rules out a universal guarantee from purely observationally equivalent data. It does not rule out next-token models representing causal laws, using externally collected intervention data, or operating inside an interactive agent.

Replace “closes passive next-token prediction alone” with the precise claim that no learner can universally identify intervention laws from an observational channel that assigns identical distributions to causally different worlds without extra assumptions or variation.

## 2. Is the lifecycle seam absent?

Not literally. The closest current collisions include:

- [Mechanistic World Models](https://arxiv.org/abs/2607.12474), which explicitly organizes world models around reusable mechanisms and autonomous discovery.
- [Explicit Symbolic Behavioral Models](https://arxiv.org/abs/2606.07127), which combines typed executable mechanism memory, adaptive questions, active probes, local edits, and recovery after mechanism changes in Atari-style protocols.
- [Self-Revising Discovery Systems](https://arxiv.org/abs/2606.01444), which formalizes provenance-preserving schema revision and MDL-gated regime changes.
- [Controlled World Model Identifiability](https://arxiv.org/abs/2607.22430), which jointly identifies latent state and controlled dynamics under nonlinear observations and action-excitation assumptions.
- [Finite-sample causal representation learning](https://arxiv.org/abs/2603.25796), which recovers latent representations, graphs, and unknown intervention targets from few environments.
- [Continual RL with Online World Models](https://arxiv.org/abs/2507.09177), which supplies online updates, planning, a regret bound, and forgetting resistance.

None establishes T53's full conjunction across raw partial observations, changing ontologies, active intervention selection, mechanism recombination, local repair, and complete cost. The novelty target is therefore an integration theorem and matched evaluation boundary—not the broad “self-maintaining executable model” concept.

## 3. Description complexity and substantial gain

The comparison is conditional and should be rewritten.

For a vector transition \(\{0,1\}^r\to\{0,1\}^r\), a literal truth table uses \(2^r\) entries but \(r2^r\) output bits. A modular code has conditional length approximately

\[
|\mathcal L|
+\sum_{j=1}^r
\left[
\log_2 M+\log_2(r)_k
\right],
\]

plus types, ports, observation maps, variable encoder, action semantics, bindings, noise parameters, and the interpreter. If \(\mathcal L\) is shared, state the number of environments over which its cost is amortized. If its mechanisms are arbitrary \(k\)-bit Boolean functions, their own truth-table cost can be \(M2^k\) bits.

This is a hypothesis-class description advantage, not an architecture advantage. An equally informed generic learner may encode and update the same factorization.

The local-repair claim also needs assumptions. Preserving \(d-s\) mechanisms exactly requires immutable mechanisms and unchanged encoders, bindings, router, planner, and interfaces. Otherwise a “local” parameter update can alter every component's effective input or selection. Description savings do not by themselves imply fewer interactions, lower regret, or better planning.

The proposed 2x interaction and 30% regret gates are empirical policy choices, not theorem-derived substantiality thresholds. Freeze matched accuracy, confidence, horizon, total compute, and memory first. Replace the ambiguous “one-point confidence bound” with a simultaneous noninferiority test over protected tasks and repeated update times.

## 4. Next proof object

The current object is too broad. It does not define the observation-map class, stochastic process, action coverage, reset ability, confounding, library status, equivalence target, deployment query, or loss. “Add the weakest assumption” is also undefined without an ordered assumption family.

Use one finite controlled model:

\[
z_{t+1,j}\sim
p_{\theta_{e,j}}
(\cdot\mid z_{t,\mathrm{Pa}_{e,j}},a_t),
\qquad
o_t\sim g_e(\cdot\mid z_t),
\]

with finite alphabets, bounded indegree, an explicit shared library, at most \(s\) changed mechanisms, a declared observation-map family, and a fixed training action policy.

The impossibility pair must be indistinguishable under the training policy but differ on a declared deployment action/query. If two worlds are equivalent under every allowed policy and reward observation, they cannot require observably different optimal plans within that same interface.

The sufficiency theorem must choose one added condition—such as a quantified action-excitation margin—and recover the model only up to action-observation bisimulation, latent permutation, or another explicit equivalence. State sample, computation, interaction, and persistent-memory bounds against an equally informed generic Bayesian/PSR/POMDP learner.

## 5. Hidden compiler and oracle bill

Charge all of the following:

- pretrained perception/LLM parameters and calls;
- variable, mechanism, and program proposal search;
- intervention-target metadata, resets, checkpoints, and action safety;
- executable sandbox, compiler, verifier, and success oracle;
- graph matching, binding, posterior inference, and planning;
- change-point localization and attribution search;
- replay/protected-task evaluation and multiple-comparison correction;
- consolidation, deduplication, semantic-equivalence checks, provenance, and migrations;
- persistent bytes, retrieval, active mechanism depth, and amortization horizon.

Any supplied segmentation, ontology, mechanism boundary, world-family identifier, or “responsible component” label is an oracle. Give every control identical compiler outputs and foundation-model access. If the compiler performs the difficult inference, it is part of the candidate, not free preprocessing.

## Final disposition

T53 is a sound reset after these corrections. Its passive theorem is a narrow but exact non-identifiability result. Its research seam is unclosed, not absent. The next admissible contribution is a finite, matched joint-identification theorem showing that learned reusable structure is cheaper than an equally informed generic learner after the entire compiler, interaction, revision, and memory bill is included.
