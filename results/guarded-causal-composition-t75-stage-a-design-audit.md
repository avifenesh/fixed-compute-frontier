# T75 Stage A model-free design audit

Date: 2026-08-02  
Status: pre-code mathematical audit  
Scope: stationary, model-free T75 Stage A only; no learned-policy training, monitoring, change detection, or restoration

## Decision

T75 Stage A is mathematically feasible on CPU if “exact Bayesian” means an exact posterior over the 64 hidden worlds plus a completely frozen action policy. It is not currently credible to promise an exact finite-horizon Bayes-optimal policy or an exact Bayes/minimax cost frontier at `m=33` and the horizon needed for a 95% identification gate. The belief filter is small; the reachable belief-state planning problem is not.

Admission is therefore conditional:

1. Freeze the finite world model, prior, likelihood, losses, policies, budgets, and comparators below.
2. Rename any m=33 “exact Bayesian optimum/frontier” claim to “exact-posterior reference policy/frontier.”
3. Use exact dynamic programming only as a small-instance calibration oracle, unless a separately reviewed sufficient-state reduction makes the m=33 calculation certifiably finite and tractable.
4. Repair the numeric gates so error, cost, confidence, and comparator class are unambiguous.

No neural T75 run should be admitted from this audit.

## 1. Frozen finite world and observation model

Let

\[
W=\{(t,s):t\in\{1,\ldots,m-1\},\ s\in\{0,1\}\},\qquad m=33,
\]

so `|W| = 64`. The prior is uniform unless the manifest is versioned and changed before any results are seen:

\[
\pi_0(t,s)=1/64.
\]

For selector `u in {0,...,32}`, define the hidden orientation

\[
o_{t,s}(u)=s\oplus \mathbf 1[u\ge t].
\]

Under the only active probe, `do(X=0)`, the binary observation `Y` has

\[
q_{t,s}(u)=P(Y=1\mid t,s,u)=
\begin{cases}
\epsilon,&o_{t,s}(u)=0,\\
1/2,&o_{t,s}(u)=1,
\end{cases}
\qquad \epsilon=1/10.
\]

Thus

\[
\ell_{t,s}(y\mid u)=q_{t,s}(u)^y(1-q_{t,s}(u))^{1-y},\quad y\in\{0,1\}.
\]

Passive observations have the same distribution in every world. They have zero mutual information about `(t,s)`, leave the posterior unchanged, and are excluded from the active decision problem. If they are admitted as actions with positive cost, they are strictly dominated.

## 2. Exact posterior update

For belief `pi` and action `u`, the predictive probability and posterior are

\[
Z_y(\pi,u)=\sum_{w\in W}\pi(w)\ell_w(y\mid u),
\]

\[
T(\pi,u,y)(w)=\frac{\pi(w)\ell_w(y\mid u)}{Z_y(\pi,u)}.
\]

All finite histories have positive probability because both Bernoulli parameters lie strictly between zero and one. The update therefore has no zero-denominator branch. With `epsilon=1/10`, integer counts and rational arithmetic can define a bit-exact test oracle; production calculations may use normalized log weights but must agree with that oracle on small cases.

For counts `k_u` successes among `r_u` probes at each selector,

\[
\pi(w\mid D)\propto \pi_0(w)\prod_u q_w(u)^{k_u}(1-q_w(u))^{r_u-k_u}.
\]

This exact 64-entry filter is easy at `m=33`. It must be shared by every reference policy and comparator so policy differences are not confounded with inference differences.

## 3. The exact finite-horizon Bayesian DP

The mathematically exact recursion exists. Freeze a terminal loss and an action cost before using it.

Primary terminal loss:

\[
R_{01}(\pi)=1-\max_{w\in W}\pi(w),
\]

the posterior Bayes risk for world identification under 0-1 loss. Unit intervention cost is the natural Stage A resource. For a scalar Lagrange multiplier `lambda > 0`, remaining horizon `h`, and optional early stopping,

\[
V_0(\pi)=R_{01}(\pi),
\]

\[
V_h(\pi)=\min\left\{
R_{01}(\pi),
\min_{u\in\{0,\ldots,32\}}
\left[\lambda+\sum_{y\in\{0,1\}}Z_y(\pi,u)V_{h-1}(T(\pi,u,y))\right]
\right\}.
\]

This recursion exactly defines a Bayes-optimal policy for the frozen prior, loss, cost, and horizon. It does not define a minimax policy. A minimax claim would require a different objective and proof.

Recommended primary reporting is not a single arbitrary `lambda`: report the error/intervention Pareto set. A preregistered cost grid may be used to expose supported points, but unsupported Pareto points may require a constrained formulation. Secondary posterior-quality losses may include, with conventions frozen:

- log score Bayes risk `H(pi) = -sum_w pi(w) log pi(w)`;
- multiclass Brier Bayes risk `1 - ||pi||_2^2`.

Neither secondary score replaces block world-identification error.

## 4. Why the full exact m=33 policy is presently infeasible

The belief is continuous even though the hidden state is finite. A history can be summarized by the `2m=66` action/outcome counts `n_(u,y)`. At depth `h`, the number of nonnegative count vectors alone is bounded above by

\[
{h+65\choose 65},
\]

before merging count vectors that happen to induce the same belief. A history-tree implementation has branching from every action and observation, while exact finite-horizon POMDP alpha-vector representations also grow rapidly with horizon. At the constructive horizons relevant to a 95% block-identification guarantee, naive enumeration is astronomical.

The existence of the recursion is therefore not evidence of computational tractability. No audited sufficient statistic, symmetry quotient, alpha-vector bound, or structural threshold theorem currently turns the full `m=33` stopping problem into a manageable exact calculation. General exact finite-horizon POMDP belief-space results establish the representation, not a practical solver at this horizon; point-based methods are explicitly approximations. See the finite-horizon POMDP discussion and Smallwood-Sondik lineage in [Ross et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC2748358/) and the finite-horizon point-based approximation literature summarized by [Walraven and Spaan](https://www.st.ewi.tudelft.nl/mtjspaan/publications/b2hd-Walraven19jair.html).

Permitted exact-DP use:

- enumerate small calibration cases such as `m <= 5`, with a horizon chosen only after measuring and documenting state growth;
- compare the exact recursion with brute-force history enumeration on still smaller cases;
- use those cases to test the filter, stopping rule, tie handling, and approximate/reference policies.

Prohibited claim without a new proof:

- “exact Bayes-optimal m=33 policy,” “exact m=33 Bayes frontier,” or “exact m=33 minimax frontier.”

### 4.1 Audited information lower bound

Each charged intervention returns one binary result through the fixed channel

\[
P(Y=1\mid O=0)=0.1,\qquad P(Y=1\mid O=1)=0.5,
\]

where `O=o_W(u)`. Conditional on the preceding transcript and chosen action, let `q=P(O=1)`. The mutual information in that result is

\[
I(O;Y)=h_2(0.1+0.4q)-(1-q)h_2(0.1)-q.
\]

Consequently its capacity is

\[
C=\max_{0\le q\le1}\left[h_2(0.1+0.4q)-(1-q)h_2(0.1)-q\right].
\]

The stationary point is uniquely determined by

\[
p^*=\frac{1}{1+2^{(1-h_2(0.1))/0.4}},\qquad
q^*=\frac{p^*-0.1}{0.4},
\]

giving `q* approximately 0.462313` and `C approximately 0.147589` bits per binary result. These values should be recomputed with outward-rounded interval arithmetic in the model-free oracle before becoming machine-enforced constants. The underlying capacity and Fano facts are reviewed in the official [MIT 6.441 information-theory notes](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/) and the explicit Fano statement in [Stanford EE276 notes](https://web.stanford.edu/class/ee276/files/lectures/lecture_10.pdf).

For uniform `W` over 64 worlds and actual average decoding error `P_e <= 0.05`, Fano's inequality gives

\[
I(W;\widehat W)\ge
6-h_2(0.05)-0.05\log_2 63.
\]

For a fixed `B`-intervention transcript, the chain rule therefore yields

\[
B\ge
\frac{6-h_2(0.05)-0.05\log_2 63}{C}
\mathrel{\approx}36.687.
\]

Thus an integer fixed budget must be at least 37 interventions. For a variable stopping time, the correct statement is `E[N] >= 36.687...`, not that every episode uses 37 and not that the real-valued expectation must be rounded up.

Adaptivity does not invalidate the bound under the frozen Stage A protocol. If `H_i` is the transcript before action `A_i`, then a policy generated only from `H_i` and fresh world-independent random coins satisfies

\[
I(W;A_i\mid H_i)=0.
\]

Conditioned on `(H_i,A_i)`, `W -> O_i -> Y_i` is a Markov chain through the same binary channel, so

\[
I(W;Y_i\mid H_i,A_i)=I(O_i;Y_i\mid H_i,A_i)\le C.
\]

The action can look informative marginally because it summarizes earlier observations, but it contributes no new conditional information. For random stopping, pad the transcript with null action/result symbols after stopping; the stop decision is a function of the existing transcript, and summing the surviving increments gives `I(W;transcript) <= C E[N]` under the usual finite-expectation condition.

The lower bound is invalid if any of the following is allowed:

- initial policy/model state, random coins, binding, or episode selection is correlated with the freshly sampled world;
- the action selector has oracle access to the world or private target;
- action identity or timing is chosen by the environment as an additional world-dependent signal;
- an intervention returns side information beyond the single binary `Y`, or a nominally single charged intervention returns multiple conditionally independent results;
- passive observations, metadata, transcript remnants, caches, or a hidden-state side channel carry world information;
- the channel parameters vary with selector, history, or world beyond `O`;
- the claimed 5% quantity is merely posterior confidence rather than actual uniform-prior decoding error.

This bound belongs in Stage A as a mandatory sanity check and a universal lower-bound row. It is not an achievable policy, an exact frontier, or evidence that a policy near 37 interventions is optimal. The query family may be unable to realize the capacity-achieving `q` at every posterior, so the true problem-specific lower bound can be larger.

## 5. Frozen exact-posterior adaptive references

Stage A should contain two reference algorithms, neither labeled globally optimal.

### 5.1 Constructive fixed-repeat binary search

There are `L = ceil(log2(64)) = 6` binary decisions in a balanced search over the 64 worlds. At each decision, choose a selector that partitions the currently admissible worlds according to the frozen search tree, repeat that selector `r` times, and use the exact likelihood-ratio decision with deterministic tie-breaking.

For one local decision between `Bernoulli(0.1)` and `Bernoulli(0.5)` under equal prior, its exact Bayes error is

\[
e_r=\frac12\sum_{k=0}^{r}
\min\{\operatorname{Bin}(r,k;0.1),\operatorname{Bin}(r,k;0.5)\}.
\]

Freeze

\[
r_A=\min\{r\in\mathbb N:6e_r\le 0.05\},\qquad H_A=6r_A.
\]

The union bound makes this a transparent Bayes-average block-error upper bound for the frozen local-decision construction. The search tree, selector at every node, and tie rule must be serialized in the manifest; “binary search” alone is underspecified under noise. Noisy binary-search theory is relevant context, not a substitute for specifying this policy; see [Ben-Or and Hassidim, FOCS 2008](https://doi.org/10.1109/FOCS.2008.58) and its [preprint](https://arxiv.org/abs/quant-ph/0703231).

### 5.2 Exact-filter greedy reference

At each step, use the exact posterior and choose one frozen one-step objective, for example minimum expected next-step 0-1 Bayes risk:

\[
u^*(\pi)=\operatorname*{argmin}_u\sum_y Z_y(\pi,u)R_{01}(T(\pi,u,y)).
\]

Use the smallest `u` as the deterministic tie-break. Stop at the first time `max_w pi(w) >= 0.95` or at frozen `H_max`. This is a one-step Bayes-risk greedy policy, not the finite-horizon optimum. If expected entropy is used instead, it is a different preregistered policy and must not be selected post hoc.

Posterior confidence is a Bayesian stopping condition. It does not by itself prove a 5% error bound for every world. Report both uniform-prior Bayes-average error and worst-world error.

## 6. Exact nonadaptive comparator

### 6.1 Selector locations

The smallest noiseless selector-location set for exact identification has `m-1=32` locations, not `m-2`. Freeze

\[
U_{NA}=\{0,1,\ldots,m-2\}=\{0,1,\ldots,31\}.
\]

Why it is sufficient: the interior selectors expose every adjacent threshold boundary, while endpoint `u=0` anchors `s`. Why 32 locations are necessary: omitting an interior selector merges an adjacent pair of thresholds for a fixed orientation, and at least one endpoint anchor is needed to resolve the global orientation. The equivalent reflected set `{1,...,32}` is not used in this manifest.

### 6.2 Repetitions and decision rule

Use equal fixed repetition, chosen analytically before data:

\[
r_{NA}=\min\{r\in\mathbb N:32e_r\le0.05\},\qquad H_{NA}=32r_{NA}.
\]

Probe every `u in U_NA` exactly `r_NA` times, compute the exact posterior above, and return its MAP world with lexicographic `(t,s)` tie-breaking. This is the exact fixed nonadaptive comparator. Equal allocation plus a union-bound-derived repetition count is intentionally simple and reproducible; it is not claimed to be the globally optimal noisy nonadaptive allocation.

For any fixed allocation vector `r_vec`, its exact uniform-prior Bayes block error is

\[
R_{NA}(\mathbf r)=1-\sum_D\max_{w\in W}\left[\pi_0(w)P(D\mid w,\mathbf r)\right],
\]

where `D` ranges over all per-selector success counts and `P(D|w,r_vec)` includes binomial coefficients. This equation defines the exact risk, but direct enumeration over `product_u(r_u+1)` count vectors is itself generally impractical. Do not label an estimated risk or a union bound “exact expected risk.” A structure-aware exact computation would require its own proof and audit.

## 7. Required invariant and leakage checks

All must pass before any learned T75 candidate is evaluated.

1. **World cardinality and injectivity:** enumerate exactly 64 `(t,s)` worlds and show their noiseless orientation strings are distinct.
2. **Likelihood normalization:** for every world and selector, both observation probabilities are positive and sum to one; `epsilon` is exactly the manifest value.
3. **Passive zero information:** passive likelihood is identical across all worlds; applying a passive event leaves every posterior coordinate unchanged.
4. **Filter normalization:** after every active event, posterior entries are nonnegative and sum to one; rational-oracle and log-space results agree on frozen small cases.
5. **Simulator/likelihood identity:** empirical samples are not the test; the simulator's enumerated one-step probability table must equal the inference likelihood table exactly.
6. **Allowed symmetries only:** name and pair swaps must give the declared posterior permutation. Ordered selector values may not be arbitrarily permuted. Any selector reflection must include the audited transformation of `(u,t,s)`.
7. **Generator commitment:** generator source, manifest, serialized search tree, and private evaluation corpus are hash-committed before candidate scoring.
8. **No conjunction leakage:** no T75 selector-plus-intervention conjunction occurs in T73/T74 component training, development, prompts, exemplars, or candidate-tuning data.
9. **Checkpoint freeze:** base, single-family, router, modular-composer, and candidate checkpoints/configurations are frozen before private T75 evaluation.
10. **Target privacy:** hidden worlds, posterior targets, scores, and stop outcomes are not available to the candidate through tools, filenames, metadata, logs, or callbacks.
11. **Erasure integrity:** after the declared erasure only state `M` remains; transcript, KV cache, optimizer state, temporary files, binding tables, and process-global caches cannot carry the answer.
12. **Counterfactual ablations:** reset-state, independent-world, binding-corruption, and state-permutation controls must behave according to their preregistered nulls.
13. **Comparator evidence equality:** learned, router, modular, and candidate controls receive identical observations and action budgets; no control gets a privileged posterior or hidden state unless it is explicitly an oracle reference.
14. **Direct T75 ceiling quarantine:** a T75-trained ceiling is generated/tuned only after candidate freeze and can never inform candidate selection or thresholds.
15. **Randomness separation:** world, binding, observation, policy, and evaluation-bootstrap random streams have independently derived frozen seeds.
16. **Full ledger:** every policy reports all active probes, failures, stopped/censored episodes, wall time, CPU/memory, and any model-side served-state resource measures.

## 8. Numeric gate repairs

The current conceptual gates are not yet operational enough for a go/no-go result. Freeze these replacements.

### Gate A: reference validity

The exact-posterior reference must achieve:

- uniform-prior Bayes-average world-identification accuracy at least 0.95 at its frozen budget; and
- a separately declared worst-world target and simultaneous confidence procedure.

Do not silently convert a Bayes-average analytic guarantee into a worst-world claim. For simulated learned-policy evaluations, freeze episode count and a simultaneous binomial interval before generation.

### Gate B: recovered compositional gain

Use

\[
G=\frac{A_{candidate}-A_{reset}}{A_{ceiling}-A_{reset}},\qquad A_{reset}=1/64,
\]

with `A_ceiling` naming one frozen ceiling. Report a simultaneous lower confidence bound for `G`; reject the metric if the denominator is too small under a frozen minimum-separation rule. Keep the exact-posterior algorithmic reference, a direct-T75-trained neural ceiling, and any full-information oracle as separate rows.

### Gate C: strongest learned control

“30% lower regret” is undefined unless regret's reference and scalar loss are frozen. Use either:

- at least 30% lower frozen primary cost/loss than the strongest **non-compositional** matched learned/router control; and
- noninferiority to the matched explicit modular composer within a frozen margin,

or state a stricter Pareto requirement. Do not require 30% improvement over a nearly exact modular oracle merely by calling it a learned control.

### Gate D: action efficiency

At matched Bayes-average and worst-world error criteria, require candidate interventions no more than `1.5x` the frozen exact-posterior reference policy. The denominator is a reference-policy cost, not an exact optimum. Report the audited Fano/channel-capacity floor separately: 37 for an integer fixed budget, or `E[N] >= 36.687...` for variable stopping under its assumptions. Do not infer optimality from proximity to either a loose lower bound or a heuristic reference.

### Gate E: posterior quality and retention

Freeze log base, clipping for learned probabilities, Brier convention, aggregation unit, noninferiority margin, bootstrap unit, and multiple-comparison correction. Retention must be evaluated against the frozen pre-T75 component baseline with a simultaneous confidence rule.

### Gate F: final admission

Admission requires a Pareto improvement under the complete ledger: no worse on every frozen primary resource/quality coordinate and strictly better on at least one, with the declared confidence rule. Secondary exploratory metrics cannot rescue a failed primary gate.

## 9. Minimal frozen manifest

```yaml
manifest_id: t75-stage-a-v0
frozen_at: BEFORE_ANY_T75_RESULT
scope: stationary_model_free

world:
  m: 33
  epsilon: 1/10
  states: "(t,s), t=1..32, s in {0,1}"
  prior: uniform_1_over_64
  hidden_change: disabled

action_observation:
  selectors: [0, 32]
  active_probe: do_X_0_then_observe_Y
  passive_actions: excluded_zero_information
  observation: binary_Y
  intervention_cost: 1

inference:
  posterior: exact_64_world_likelihood
  rational_oracle: required_for_small_cases
  map_tie_break: lexicographic_t_then_s

objective:
  primary_terminal_loss: "1 - max_w posterior[w]"
  primary_report: error_intervention_pareto
  bayes_average_error_target: 0.05
  worst_world_target: MUST_FREEZE
  confidence_method: MUST_FREEZE
  posterior_stop: 0.95

adaptive_references:
  constructive:
    policy: serialized_balanced_six_decision_tree
    local_decision: exact_binomial_likelihood_ratio
    repetitions: "min r such that 6*e_r <= 0.05"
    tie_break: MUST_FREEZE
  greedy:
    policy: one_step_expected_01_bayes_risk
    action_tie_break: smallest_u
    horizon: "6*r_A"
  exact_dp:
    use: small_m_calibration_only
    maximum_m_and_horizon: MUST_FREEZE_AFTER_STATE_GROWTH_PILOT

nonadaptive_reference:
  selectors: [0, 31]
  allocation: equal_fixed
  repetitions: "min r such that 32*e_r <= 0.05"
  inference: exact_posterior_map
  claim: fixed_exact_algorithm_not_globally_optimal_allocation

evaluation:
  worlds: all_64_balanced
  information_lower_bound:
    channel_capacity_bits: interval_certified_approximately_0.147589
    fixed_integer_budget_floor_at_average_error_0_05: 37
    variable_stopping_expected_floor: interval_certified_approximately_36.687
    assumptions_checked: required
  world_binding_observation_policy_seeds: independent_and_frozen
  episode_count: MUST_FREEZE
  simultaneous_intervals: MUST_FREEZE
  stopped_or_censored_handling: MUST_FREEZE
  failed_runs_included: true
  full_resource_ledger: true

leakage:
  component_data_excludes_t75_conjunction: required
  candidates_and_controls_frozen_before_private_eval: required
  generator_manifest_corpus_hashes: required
  transcript_cache_binding_erasure_checks: required
  reset_independent_world_corruption_ablations: required

ceilings:
  exact_posterior_algorithmic_reference: separate
  direct_t75_trained_neural_ceiling: post_freeze_quarantined
  full_information_oracle: separate_if_used
```

Every `MUST_FREEZE` is a blocker, not an implementation default.

## 10. Blockers and Stage A boundary

### Blocking mathematical claims

1. There is no demonstrated tractable exact Bayes-optimal DP for `m=33` at the required horizon.
2. There is no demonstrated exact m=33 Bayes/minimax cost frontier.
3. The equal-allocation nonadaptive comparator is reproducible but is not proved globally optimal under noise.
4. Posterior stopping confidence, Bayes-average error, and worst-world frequentist error are not interchangeable.

### Blocking manifest choices

1. Worst-world target and simultaneous confidence method.
2. Episode count and stopped/censored-episode treatment.
3. Constructive-tree serialization and all tie rules.
4. Learned-probability clipping, score conventions, retention margin, and multiplicity correction.
5. Exact small-instance calibration limit after a model-free state-growth pilot.

### Admissible first implementation slice after those choices

Only the CPU/model-free layer: frozen generator and hashes, exact 64-world posterior oracle, invariant/leak tests, constructive adaptive policy, one-step greedy policy, fixed nonadaptive comparator, exact analytic `e_r` repetition selection, and small-instance DP calibration. This slice may validate the benchmark mechanics and reference algorithms. It cannot establish learned composition, neural efficiency, monitoring, restoration, or an exact m=33 optimum.

That is the narrow credible Stage A contract.
