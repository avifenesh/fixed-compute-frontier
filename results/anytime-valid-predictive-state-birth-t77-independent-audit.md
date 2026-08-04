# T77 independent audit: anytime-valid predictive-state birth

**Audit date:** 2026-08-02  
**Artifact audited:** `results/anytime-valid-predictive-state-birth-t77.md`  
**Mode:** theorem and literature audit only; no implementation, CPU run, GPU run, or empirical result is admitted.

## Verdict

**T77 is not admitted as a model-level intelligence result or as an experimental lane.** Its central Type-I-error argument is a recognizable and potentially useful assembly of likelihood-ratio e-processes, Ville's inequality, and a Kraft/MDL prior. The fixed-codebook result can be made correct under causal adaptive actions. T77.2's displayed Azuma constant is also correct under one precise bounded-martingale-difference convention.

The document is nevertheless not ready to support its stronger claims. Four blockers are fatal until repaired:

1. **The guarantee is only for an exact, simple, correctly supported predictive null.** It does not control false *structural* births under a composite, fitted, drifting, or misspecified baseline. An arbitrarily small persistent misspecification can make eventual acceptance almost certain.
2. **Candidate generation and validation accounting are underspecified.** A fixed countable coded family may share one validation stream, but an adaptively fitted, uncoded law may not score the data that fitted it. Repeated proposals, restarts, families, and post-birth baseline changes require one explicit wealth/alpha ledger.
3. **The code-length claim is not operational.** Parameters, precision, bindings, learned tables, interpreter/DSL choice, state initialization, runtime dependencies, and any proposal-side artifacts must be charged or integrated under a fixed prior. Otherwise the Kraft premise is false or vacuous.
4. **The three-family and anti-prose controls can establish a scoped executable-abstraction result, not substantial model-level intelligence.** Predictive evidence does not identify causal ontology, semantic novelty, minimality, or model-mediated discovery.

The strongest defensible claim after theorem repair would be: *a pre-specified system can propose bounded executable predictors and admit them with anytime-valid predictive evidence relative to a stated null, while paying an explicit complexity and sequential-testing budget.* That is valuable, but narrower than “new latent structure” or substantial intelligence.

## 1. Correct sequential result under adaptive causal actions

### 1.1 Filtration and timing that T77 must state

Let \(\mathcal F_t\) be the information available after outcome \(Y_t\). Let

\[
\mathcal G_t = \mathcal F_{t-1}\vee\sigma(A_t)
\]

be the information available after choosing action \(A_t\) but before seeing \(Y_t\). Randomized actions are allowed when their fresh random coins do not reveal the current outcome, future outcomes, or a hidden environment state unavailable to the null model. The action and the next predictive distribution must be \(\mathcal G_t\)-measurable.

Under a simple exact null, define the conditional kernel

\[
p_t(y)=P_0(Y_t=y\mid \mathcal G_t).
\]

For candidate \(g\), let \(q_{g,t}(y)\) be a normalized, \(\mathcal G_t\)-measurable predictive kernel. If \(Q_g\ll P_0\) conditionally, then

\[
R_{g,t}=\frac{q_{g,t}(Y_t)}{p_t(Y_t)},\qquad
\Lambda_{g,n}=\prod_{t=1}^n R_{g,t}
\]

satisfies

\[
\mathbb E_{P_0}[R_{g,t}\mid\mathcal G_t]=1.
\]

Thus \((\Lambda_{g,n})_{n\ge 0}\) is a nonnegative \(P_0\)-martingale. Conditional subprobability alternatives give a supermartingale, which is also sufficient. Causal adaptive action selection does not damage the proof: the action is conditioned on, and the likelihood ratio concerns the next outcome kernel. The result fails if the action timing leaks \(Y_t\), the environment uses hidden action-selection information absent from \(p_t\), or the recorded action is post-outcome.

### 1.2 Replacement theorem for T77.1

**Theorem A (fixed coded family, shared stream).** Let \(\{Q_g:g\in\mathcal C\}\) be a fixed countable family of fully specified predictable kernels. Let deterministic weights \(w_g\ge0\) satisfy \(\sum_gw_g\le1\). Under the timing and support conditions above,

\[
M_n=\sum_{g\in\mathcal C} w_g\Lambda_{g,n}
\]

is a nonnegative supermartingale with \(M_0\le1\). Therefore, by Ville's inequality,

\[
P_0\!\left(\exists n,\exists g:\;w_g\Lambda_{g,n}\ge\alpha^{-1}\right)
\le
P_0\!\left(\sup_n M_n\ge\alpha^{-1}\right)
\le\alpha.
\]

When \(w_g=2^{-L(g)}\), the required weight condition follows from a genuinely prefix-free fixed code. This is the clean version of T77.1. A union bound plus per-candidate Ville bounds gives the same conclusion, but the mixture proof makes the wealth interpretation explicit.

Important corrections:

- **Candidates do not need independent validation streams.** Every fixed candidate may be evaluated on the same outcomes; independence among candidates is irrelevant.
- **Data-dependent selection among the fixed family is allowed.** Even retrospective selection is safe because the selected-crossing event is a subset of the union over all already coded \(g\). The fixed code/weight has already paid for selection.
- “Fresh validation” must mean that a factor used for \(Y_t\) was fixed before \(Y_t\), not necessarily a physically separate environment or stream.
- A recursively updating predictor is allowed when its update algorithm is already part of \(g\) and its next kernel is predictable. That is prequential prediction, not forbidden fitting.

These facts make T77.1 broader than T77 currently suggests, but only for an honestly fixed family.

### 1.3 Adaptively proposed or fitted candidates

Suppose proposal \(j\) is created at a stopping time \(\tau_j\) from \(\mathcal F_{\tau_j}\). Conditional on that history, a future-only process

\[
\Lambda_{j,n}^{\text{post}}
=\prod_{t=\tau_j+1}^{n}
\frac{q_{j,t}(Y_t)}{p_t(Y_t)}
\]

is valid if every \(q_{j,t}\) is normalized and chosen before \(Y_t\). A separate stream is unnecessary; the same physical stream can host many such processes. What is forbidden is retroactively setting \(q_j\) using outcomes and then multiplying likelihood ratios over those same outcomes, unless the realized option and its full multiplicity were already in a fixed countable codebook.

For an open-ended sequence of fitted proposals, assign predictable candidate wealth \(a_j\ge0\) satisfying

\[
\sum_j a_j\le\alpha\quad\text{almost surely},
\]

and accept proposal \(j\) only if \(\Lambda_{j,n}^{\text{post}}\ge1/a_j\). Equivalently, use a single online e-wealth construction. If proposal \(j\) contains a fixed within-proposal coded family, use weights \(w_{j,g}\) with \(\sum_gw_{j,g}\le1\) and threshold \(1/(a_jw_{j,g})\). This explicitly distinguishes the candidate code penalty from the budget spent on opening a new data-dependent proposal episode.

The following cases must not be conflated:

| Candidate construction | May use shared stream? | May score pre-fit outcomes? | Valid route |
|---|---:|---:|---|
| Fixed, fully specified countable candidate | Yes | Yes | Kraft-weighted union/mixture |
| Candidate fit on an independent proposal split and frozen | Yes, after fit | No | Condition on proposal data; test future/holdout data |
| Candidate fit at a stopping time from past outcomes | Yes, prospectively | No | Start its e-process after the stopping time and spend new wealth |
| Prequential updater fixed in advance | Yes | Yes, through its fixed update rule | Each next factor must be predictable |
| Point estimate fit on the entire validation block | No naive reuse | No | Split/universal inference, or pre-code/integrate all parameter choices |
| Continuous uncoded parameter chosen after seeing validation | No | No | Quantized fixed code, prior mixture, or a separately proved e-process |

Previous validation outcomes may become proposal data for a later version. They may not be rescored retroactively by the newly fitted version. Optional stopping or resuming one existing e-process is safe; resetting its wealth to one after an unfavorable path and testing again is a new attempt and spends new alpha.

T77 should provide a single lifetime ledger covering candidates, versions, task families, resets, and the sequence of newly committed baselines. Kraft weights already implement \(\alpha_g=\alpha w_g\) inside a fixed family; a second alpha schedule must be hierarchical rather than an unexplained duplicate charge.

## 2. T77.2: Azuma audit

Define

\[
X_t=\log\frac{q_t(Y_t)}{p_t(Y_t)},\qquad
\mu_t=\mathbb E_Q[X_t\mid\mathcal G_t],\qquad
D_t=X_t-\mu_t.
\]

The needed drift assumption is predictable and pathwise:

\[
\sum_{t=1}^{N}\mu_t\ge Nd\quad Q\text{-a.s.},
\]

or the stronger \(\mu_t\ge d\) almost surely for every \(t\). An unconditional statement such as \(\mathbb E_Q[\sum X_t]\ge Nd\) is not enough for the displayed tail bound.

**Theorem B (fixed-horizon power bound).** If \((D_t)\) is a martingale-difference sequence under \(Q\), \(|D_t|\le c\) almost surely, and the predictable drift condition holds, then for

\[
T=L(g)\ln2+\ln(1/\alpha)
\]

and \(Nd>T\),

\[
Q\!\left(\log\Lambda_N<T\right)
\le
\exp\!\left[-\frac{(Nd-T)^2}{2Nc^2}\right].
\]

The constant in T77.2 is therefore correct **only under its stated centered-difference bound** \(|X_t-\mu_t|\le c\). If T77 intends only \(|X_t|\le c\), it must state the concentration convention carefully: applying the same absolute-difference Azuma step after centering gives \(|D_t|\le2c\), hence denominator \(8Nc^2\); a Hoeffding range argument for \(X_t\in[-c,c]\) can recover denominator \(2Nc^2\). More generally, if \(X_t\in[a_t,b_t]\), the Hoeffding-Azuma form is

\[
Q\!\left(\sum_tD_t\le-s\right)
\le \exp\!\left[-\frac{2s^2}{\sum_t(b_t-a_t)^2}\right].
\]

The threshold is in nats and is correct. A sufficient explicit integer horizon for power at least \(1-\beta\) is

\[
N\ge
\left\lceil
\left(
\frac{\kappa+\sqrt{\kappa^2+4dT}}{2d}
\right)^2
\right\rceil,
\qquad
\kappa=c\sqrt{2\ln(1/\beta)}.
\]

Although this is a fixed-\(N\) bound, it also upper-bounds failure to cross by \(N\), because “no crossing by \(N\)” implies \(\log\Lambda_N<T\).

If the actual conditional environment is \(R_t\), the candidate need not equal truth. Its drift is

\[
\mu_t=
D_{\mathrm{KL}}(R_t\|P_{0,t})
-D_{\mathrm{KL}}(R_t\|Q_{g,t}).
\]

Thus T77 may assume a positive candidate advantage rather than exact truth, but it must establish the conditional lower bound under the action policy.

Fitting \(Q\) on proposal data does not by itself invalidate Theorem B: condition on that data, freeze the candidate, and verify that the *realized* candidate satisfies the drift and increment bounds. Fitting on current or future validation outcomes invalidates adaptedness. Prequential fitting from past validation outcomes remains allowed if every next distribution is fixed before the next outcome and the drift/bound conditions hold.

Finally, log likelihood ratios are not generally bounded. T77 needs frozen support floors or an explicit tail/variance assumption. If bounded increments are unrealistic, it needs a Freedman/Bernstein or other time-uniform e-process argument with the actual conditions, not an informal substitution.

## 3. Exact-null, composite-null, and misspecification failure modes

### 3.1 What the theorem actually controls

T77.1 controls false rejection of one exact conditional distribution \(P_0\). It does **not** control false declaration of a new state variable, mechanism, causal abstraction, or ontology.

If the true environment is \(R\ne P_0\) and some ordinary correction \(Q_g\) has positive long-run conditional log-loss advantage, its likelihood ratio tends to grow. With a finite threshold, eventual crossing can then have probability near one. The accepted candidate may merely express:

- parameter recalibration or ordinary refitting;
- overdispersion, temporal drift, or a nuisance covariate;
- a different but observationally equivalent factorization;
- a lookup table or memorizer;
- wrapper leakage, action-policy leakage, or simulator-version mismatch.

No amount of Ville/Kraft bookkeeping converts predictive superiority into semantic novelty, causal correctness, minimality, locality, or a “birth” of a real latent entity.

A proposal-fitted and then frozen \(P_0\) yields a guarantee relative to that realized fitted baseline, not relative to the best member of a composite old-model class. A Bayesian mixture denominator gives a guarantee under the stated Bayesian mixture construction; it is not automatically uniform frequentist validity for each null member. A generalized likelihood ratio or maximized fitted-null denominator is also not automatically an e-process.

Every commit changes the baseline. Lifetime validity over a sequence of births therefore requires conditional online wealth accounting under each new null. The one-time T77.1 statement does not supply that lifecycle result.

### 3.2 Safe repairs

1. **Composite conditional e-factors.** Require factors \(E_t\ge0\) satisfying

   \[
   \sup_{P\in\mathcal H_0}\mathbb E_P[E_t\mid\mathcal G_t]\le1.
   \]

   Then \(\prod_tE_t\) is a supermartingale under every null. For a discrete outcome space, the conservative choice \(E_t=q_t(Y_t)/\bar p_t(Y_t)\), where \(\bar p_t(y)=\sup_{P\in\mathcal H_0}p_t(y)\), is valid because \(p_t(y)\le\bar p_t(y)\), though it may have poor power. Safe-testing projections or least-favorable null constructions can be less conservative.

2. **Universal inference.** Fit an alternative on one split and evaluate it on a holdout against the supremum null likelihood. This supplies a finite-sample holdout e-value in broad settings. Sequential use needs valid non-overlapping blocks or a separately proved conditional e-process; naive overlapping refits do not inherit validity.

3. **A material predictive-null rather than an exact log-loss null.** Let \(X_t=s(P_0,Y_t)-s(Q,Y_t)\) be a bounded proper-score gain and test \(\mathbb E[X_t\mid\mathcal G_t]\le\delta\). If \(X_t\in[a,b]\), then for fixed \(\lambda\ge0\),

   \[
   \exp\!\left(
   \lambda\sum_{t=1}^n(X_t-\delta)
   -\frac{\lambda^2(b-a)^2n}{8}
   \right)
   \]

   is a null e-process. This tests improvement beyond a predeclared practical tolerance. It still does not prove structural truth. Log score requires support floors or another tail treatment.

4. **Robust null neighborhoods.** Predeclare a contamination, total-variation, calibration, or nuisance-parameter neighborhood and construct factors uniformly safe over it.

5. **Structural controls outside the e-gate.** Require the candidate to beat a refittable same-class baseline, transfer under held-out interventions, survive state lesions, and mediate a local revision. These are additional empirical identification criteria, not consequences of the likelihood-ratio theorem.

### 3.3 Zero-support outcomes

For martingale equality, conditional absolute continuity is required. If \(p_t(Y_t)=0\) under an exact null, that observation has zero null probability and an extended infinite likelihood ratio can reject without Type-I cost. In real systems this is dangerously brittle evidence of model misspecification, not automatically a deep discovery.

“Use a dominating distribution” is not a generic repair: replacing \(p_t\) by an arbitrary denominator changes the hypothesis and generally loses the e-factor expectation bound. Safe options are to specify a smoothed/contamination/composite null in advance and prove validity for it, or impose a support floor as part of the null. Post hoc clipping likewise changes the test and needs a new proof.

## 4. Prefix code and double-use audit

The code-length term is legitimate only if the codebook is fixed before the scored validation outcomes and is genuinely uniquely decodable/prefix-free. The serialized object must include, directly or through charged references:

- executable syntax and all constants;
- parameter values and quantization precision;
- variable/observation/action bindings;
- bounded-state layout, initial state, reset and update semantics;
- hyperparameters and frozen training/proposal settings that determine the law;
- library, parent-model, interpreter, DSL, and runtime versions;
- tables, generator seeds/identifiers, external calls, and proposal-produced artifacts used at prediction time.

Continuous point parameters do not have finite prefix lengths without quantization; use a fixed discretization or integrate them under a prior. A universal interpreter constant can be shared within one predeclared experiment, but comparing or tuning multiple DSLs/interpreters must pay for that choice. A data-dependent compressor or codebook trained on validation is not covered by Kraft merely because its final strings are prefix-free.

For a committed parent memory \(M\), conditional lengths \(L(g\mid M)\) are legitimate only when a separate conditional Kraft inequality holds for each frozen parent. The system also needs hierarchical alpha/wealth allocation across parents. Accepted library routines may be referenced cheaply only if their identities and versions are frozen and their earlier cost was honestly paid.

Multiple programs with the same semantics are safe but waste prior mass. Choosing the shortest description from a fixed family is safe because the entire family was already charged. Learning the shortest description from validation, hiding a validation lookup table in a short generator ID, or letting executable code call an LLM or validation data at runtime defeats the interpretation. The manifest therefore needs a side-channel and dependency audit, not just a byte count.

The two uses of data must be distinguished precisely:

- Past data may update a predeclared executable predictor and may influence the next action and next prediction.
- The same outcome may not both choose an otherwise uncoded current likelihood factor and be scored by that factor.

## 5. Active evidence gathering

Adaptive actions preserve validity under the filtration above, but T77's evidence-growth objective is not yet well defined. “Expected evidence” must state the distribution under which the expectation is taken: candidate, null, design prior, robust mixture, or worst case. Maximizing expectation only under one candidate can choose self-confirming actions; a sound design objective should explicitly separate the null from relevant alternatives while keeping the action policy inside the tested conditional kernels.

Both \(P_0\) and \(Q_g\) must predict the action-conditioned outcome distribution. If taking an action changes simulator mode, observation granularity, censoring, or policy-dependent state in a way not represented by those kernels, the likelihood ratio no longer has the advertised conditional expectation.

Any imported T75 numeric statement should also be labeled according to its actual provenance. An analytic identity cannot “prove” empirical compression, entropy reduction, or realized transfer numbers. T77 should link the frozen T75 artifact and distinguish theorem, deterministic calculation, and measured result.

## 6. Novelty and direct prior-art collisions

### 6.1 What is standard

The statistical core is not a new theorem. Prior-weighted likelihood-ratio martingales are simultaneously:

- e-processes/safe tests with Ville-style optional-stopping validity ([Safe Testing](https://arxiv.org/abs/1906.07801), [E-values](https://arxiv.org/abs/1912.06116), [time-uniform concentration](https://arxiv.org/abs/1808.03204));
- Bayes factors or alternative-mixture likelihood ratios against a simple null; and
- an MDL/Occam construction when \(-\log w_g=L(g)\ln2\) ([MDL tutorial](https://pubmed.ncbi.nlm.nih.gov/10733861/)).

Universal inference already supplies split-fit likelihood-ratio-style finite-sample validity without regular parametric assumptions ([Wasserman, Ramdas, and Balakrishnan](https://arxiv.org/abs/1912.11436)). Composite-null e-process construction is an active established area, not a free corollary of the simple-null proof ([robust likelihood-ratio e-processes](https://arxiv.org/abs/2408.14015)).

### 6.2 State, causal, and program-discovery lineages

Predictive-state construction and splitting precede T77 in causal-state reconstruction ([CSSR](https://arxiv.org/abs/cs/0406011)), predictive state representations ([PSRs](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html)), and learned causal-state representation ([Zhang et al.](https://arxiv.org/abs/1906.10437)). Adaptive interventions for mechanism discrimination overlap active causal experimental design ([ABC3](https://www.nature.com/articles/s42256-023-00719-0)).

Executable abstraction discovery and library growth overlap Bayesian program synthesis ([Bayesian synthesis of probabilistic programs](https://doi.org/10.1145/3290350)), DreamCoder's learned proposal plus library abstraction ([DreamCoder](https://arxiv.org/abs/2006.08381)), and more recent theory/program synthesis ([TheoryCoder-2](https://arxiv.org/abs/2602.00929)). LLM predicate invention is a direct functional neighbor ([ADVENT](https://arxiv.org/abs/2607.01585)); transactional memory commit/revision and uncertainty-aware memory overlap [MemTX](https://arxiv.org/abs/2607.23929) and [BeliefMem](https://arxiv.org/abs/2605.05583).

### 6.3 RCE and PatchWorld are closer collisions

[Recursive Concept Evolution (RCE)](https://arxiv.org/abs/2602.15725) explicitly claims dynamic concept spawning, low-rank concept subspaces, MDL-based admission, merging, and persistence, with reported 8--18 point benchmark gains. This is a direct collision with the broad “detect inadequacy, birth a compact abstraction, complexity-gate it, and commit it” story. Those reported gains and theoretical claims should be treated as paper claims, not as independently validated evidence.

RCE also illustrates why T77 should not inherit nearby theory uncritically. Its Proposition 1 says the injection

\[
h'=(I+gP)h
\]

strictly expands covariance rank when the concept basis projects into the covariance null space. For constant \(g>0\) and an orthogonal projector \(P\), \(I+gP\) is invertible, so

\[
\operatorname{rank}\!\left((I+gP)\Sigma(I+gP)^\top\right)
=\operatorname{rank}(\Sigma).
\]

Its proof retains a positive-semidefinite \(g^2P\Sigma P\) term but omits the cross terms \(gP\Sigma+g\Sigma P\). If \(P\) lies wholly in the null space, then \(Ph=0\) almost surely and nothing changes. An input-dependent nonlinear gate can change covariance structure in some cases, and thresholded “effective rank” can change under scaling, but neither yields the proposition's universal strict-rank conclusion. This does not invalidate T77's own e-process theorem; it weakens any novelty or intelligence argument borrowed from the RCE framing.

[PatchWorld](https://arxiv.org/abs/2605.30880) is an even more direct systems collision: it induces executable symbolic belief-state programs from offline trajectories and improves them through counterexample-guided local code repair, with planning evaluated in partially observable agent environments. T77 may still differ by requiring an anytime-valid prospective e-gate, a prefix complexity budget, causal action selection, and a transactional fresh-process commitment protocol. It cannot claim novelty for executable belief-state induction or local counterexample repair themselves.

### 6.4 Residual distinctness

The potentially distinct contribution is the *conjunction* of:

1. a neural proposer;
2. a fully executable bounded predictive-state candidate;
3. prospective anytime-valid complexity-weighted admission;
4. causal evidence-seeking actions;
5. fresh-process persistence; and
6. cross-family transfer plus lesion/local-revision tests.

That is a plausible systems contribution, not a new statistical theorem. T77 does not yet specify or validate the conjunction tightly enough to establish distinctness. The literature section should use a component-by-component claim table and avoid “first” language absent a systematic search.

## 7. Can three families and anti-prose controls support an intelligence claim?

Only a limited claim is supportable. Success could show that one frozen proposer/updater discovers useful executable predictive abstractions across three predeclared task families. It would not alone show a substantial increase in general intelligence.

Required corrections to the evaluation logic:

- “Non-isomorphic” must have a formal criterion. Three surface generators can share one latent DSL trick or family-specific adapter.
- The universal interface, parser, simulator bindings, and DSL must be frozen and charged. Otherwise family knowledge is hidden in scaffolding.
- One naturalistic family is insufficient to rule out pretraining contamination or wrapper leakage.
- Equal prose/code bytes are not equal information or compute. Executable code can contain a lookup table, compressed seed, external model call, or far denser task information.
- A hand-coded enumerator, symbolic search, or MDL system could pass the same predictive gate. The learned proposer may only amortize search. That is a meaningful search-efficiency claim, but not evidence that the model itself formed a new concept.
- Evidence must separate proposal coverage/search efficiency from test validity. A perfect e-gate can safely reject almost everything while discovering nothing.
- A model-level claim requires complete-system Pareto improvement at fixed served compute on untouched tasks, after transcript/state purge, with strong non-neural and search baselines.
- Lesion, state swap, cross-domain reuse, and local repair should establish causal mediation: the admitted artifact must be necessary, selectively sufficient, and used by the model rather than merely correlated with success.
- Report absolute useful competence, not only deltas from a weak baseline, and correct across the full family/task/seed selection process.

At most, three strong families can be a pre-registered proof of breadth for one mechanism. “Substantial model-level intelligence” requires broader transfer, mechanistic mediation, contamination control, and a complete-system fixed-compute Pareto win.

## 8. Required edits before any implementation admission

1. Replace T77.1 with Theorem A and its filtration, support, fixed-codebook, and shared-stream conditions.
2. Add an adaptive-proposal theorem with stopping-time start, prospective factors, and a single lifetime alpha/e-wealth ledger.
3. State whether each candidate is fixed, split-fit, or prequential; forbid retrospective point-fit scoring unless a fixed code/prior mixture pays for all choices.
4. Replace the T77.2 assumptions with conditional predictable drift and a precise centered-difference or bounded-range condition; include the explicit horizon formula.
5. Define the null: exact simple, composite, or material-tolerance. Do not call a simple-null rejection a structural discovery.
6. Specify support handling and prove any smoothing/clipping construction rather than describing it informally.
7. Freeze a canonical serialization and enumerate everything charged by \(L(g)\), including bindings and runtime dependencies.
8. Define conditional code lengths and alpha allocation after a baseline commit.
9. Add same-class refitting, intervention transfer, lesion/swap, side-channel, and fresh-process checks as separate requirements.
10. Rewrite novelty as an integration claim and add RCE and PatchWorld to the direct-collision table.
11. Narrow the intelligence claim to what the predeclared evaluation can identify.

## Admission decision

- **Mathematical kernel:** conditionally admissible after the exact corrections above.
- **Novel theorem:** no; the core is standard Ville/e-process plus Kraft/MDL weighting.
- **Structural-birth guarantee:** not established.
- **Novel systems conjunction:** plausible but currently only a specification.
- **Substantial model-level intelligence claim:** not supported by three families and anti-prose controls alone.
- **Implementation admitted:** **no**.
- **CPU run admitted:** **no**.
- **GPU run admitted:** **no**.
- **Next admissible action:** revise the paper/manifest only; then obtain a new independent theorem audit before any implementation or run.

## Primary sources consulted

- Grünwald, [A tutorial introduction to the minimum description length principle](https://pubmed.ncbi.nlm.nih.gov/10733861/).
- Grünwald, de Heide, and Koolen, [Safe Testing](https://arxiv.org/abs/1906.07801).
- Vovk and Wang, [E-values: Calibration, combination and applications](https://arxiv.org/abs/1912.06116).
- Howard et al., [Time-uniform, nonparametric, nonasymptotic confidence sequences](https://arxiv.org/abs/1808.03204).
- Wasserman, Ramdas, and Balakrishnan, [Universal inference](https://arxiv.org/abs/1912.11436).
- Shalizi and Shalizi, [Blind construction of optimal nonlinear recursive predictors for discrete sequences](https://arxiv.org/abs/cs/0406011).
- Littman, Sutton, and Singh, [Predictive representations of state](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html).
- Zhang et al., [Learning causal state representations of partially observable environments](https://arxiv.org/abs/1906.10437).
- Tigas et al., [Interventions, where and how? Experimental design for causal models at scale](https://www.nature.com/articles/s42256-023-00719-0).
- Ellis et al., [DreamCoder](https://arxiv.org/abs/2006.08381).
- [TheoryCoder-2](https://arxiv.org/abs/2602.00929).
- [ADVENT](https://arxiv.org/abs/2607.01585), [MemTX](https://arxiv.org/abs/2607.23929), and [BeliefMem](https://arxiv.org/abs/2605.05583).
- Chaudhry, [Recursive Concept Evolution](https://arxiv.org/abs/2602.15725).
- Bai et al., [PatchWorld](https://arxiv.org/abs/2605.30880).

