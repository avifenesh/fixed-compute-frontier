# Independent audit — T70 bounded persistent meta-learner

Date: 2026-08-02  
Audited file: `results/bounded-persistent-meta-learner-t70.md` (including the affine-control, full-context-oracle, and added-control corrections made during audit)  
Scope: theorem correctness, causal interpretation, full resource/control accounting, current collisions, and admission under the goal of substantial deployed-model intelligence improvement. No experiment was run during this audit.

## Verdict

**RUN — but only the cheap local falsifier after the protocol edits below. HOLD every scaled or rented run.**

T70.1, the affine algebra, T70.2, and the narrow T70.3 channel claim are mathematically sound after small precision fixes. Unlike the closed preceding mechanisms, T70 directly targets the corrected goal and has both order-one synthetic effects and current positive evidence that lifetime/meta-RL training can create large online-acquisition gains. Novelty is neither present nor required.

The admitted hypothesis is not “latent slots beat recurrence.” It is: **one bounded, hard-erased, lifetime-trained deployed operator converts interaction into retained competence across a predeclared task meta-distribution at a better capability/resource frontier than the best bounded text, retrieval, fast-weight, and recurrent alternatives.** A GRU/SSM/recurrent meta-learner with the same lifetime objective is an implementation of that hypothesis, not a fatal control. The cheap screen can kill a concrete implementation/training recipe; it cannot disprove all meta-learning by one failed seed or undersized backbone.

## Formal audit and exact edits

### T70.1 — correct expected-error separation

For integers `n>=1`, `K>=2`, and `R>=1`, with `Theta` sampled after training, a query schedule independent of `Theta`, no cross-query state, and input restricted to `i`, every reset answer is independent of `Theta_i`; Bayes success is `1/K`. Linearity of expectation gives `nR(1-1/K)` errors without requiring independence among queries.

A persistent construction can initialize each addressed cell to one fixed guess, overwrite it with revealed feedback after its first occurrence, and thereafter retrieve it. It needs no separate seen bit: when the true symbol equals the default, the same output is correct before and after the write. Its expected error is exactly `n(1-1/K)` under this construction, so the ratio of **expected error counts** is `1/R`.

Required edits: state the integer/range assumptions; state that feedback arrives before the next occurrence; say “ratio of expected errors,” not an expected random ratio; and make explicit that query timing/order and every other current input are independent of `Theta`. The theorem separates persistence from reset, not latent memory from an equally provisioned text store, retrieval system, fast weights, or recurrent learner. The candidate already acknowledges that essential limitation.

### Affine witness — algebra correct, abstraction claim now appropriately narrow

For prime `p` and distinct `x_1,x_2`, the inverse exists and the displayed solution uniquely identifies `(a,b)`. Encoding coefficients in separate fixed-width fields uses **at most** `2 ceil(log2 p)` bits; the exact information count for the pair is `ceil(log2(p^2))` bits.

The corrected candidate rightly concedes that two defining examples are also constant memory. Sharpen this further: if the two `x` values are fixed or recoverable from the protocol, storing only their two `y` values uses essentially the same information as `(a,b)`. Thus this witness proves finite sufficient coding and systematic extrapolation, not an asymptotic memory advantage over optimal example storage. A valid test must separately score:

1. whether the state retains information after erasure;
2. whether the processor performs the finite-field computation on held-out `x`, renamed surfaces, and longer horizons; and
3. whether the same learned writer, without a family-specific update, produces a useful sufficient code.

The hard-coded affine solver remains an oracle ceiling, not a baseline the learned model is expected to beat.

### T70.2 — exact bound correct; Fano bound needs decoder/time conventions

After all symbols have been revealed, let `M` be the sole cross-boundary state and let the answer decoder receive only `(M,i)` plus randomness independent of `Theta`. Exact recovery of every environment requires an injection from `K^n` possibilities into at most `2^B` states, hence the integer statement is

\[
B\ge \lceil n\log_2 K\rceil.
\]

For approximate recall, let `p_i` be the minimum/MAP error for decoding `Theta_i` from `M`, let `p_e=n^{-1}\sum_i p_i`, and assume `0<=p_e<=1-1/K`. Then

\[
\begin{aligned}
I(\Theta;M)
&=n\log_2K-H(\Theta\mid M)\\
&\ge n\log_2K-\sum_i H(\Theta_i\mid M)\\
&\ge n\!\left[\log_2K-h_2(p_e)-p_e\log_2(K-1)\right],
\end{aligned}
\]

where conditional subadditivity gives the first inequality, coordinate-wise Fano gives the second, and concavity combines the errors. Since `I(Theta;M)<=H(M)<=B`, the displayed bound follows. Use MAP/minimal error; an arbitrarily degraded decoder error is not the relevant memory characterization. State that logs and entropy are base two and that the bound is for uniform independent coordinates and average error after the reveal phase.

### T70.3 — valid narrow intervention theorem

In the lookup world, if every variable surviving the boundary other than `M` is conditionally independent of `Theta_i` given the fixed current query `i`, then setting `M` to a constant independent of `Theta`, or replacing it with state from an independent environment, makes any answer independent of `Theta_i`; accuracy is exactly `1/K`. Above-chance accuracy therefore implies `I(Theta_i;M|i)>0`, and an in-distribution independent-state shuffle establishes that the declared state channel is necessary for the observed gain.

Required edits: evaluate the identical fixed future queries under factual and ablated memory; shuffle across independently sampled `Theta`, not within one correlated lifetime; include timing, boundary position, sampler state, caches, retrieval, environment metadata, and runtime RNG in the no-other-channel assumption; and keep the current observation/feedback from re-revealing the answer. This proves acquired information crossed through `M`. It does not prove abstraction, safe editing, calibration, or retention.

## Operator and resource ledger

The operator needs a noncircular time order. At step `t`: observe `o_t` and prior feedback, form bounded scratch, compute the policy/action and write proposal, execute `a_t`, receive `o_{t+1},f_t`, then update `m_{t+1}`. The present wording says `c_t` contains the “present action and feedback” before `P` produces the policy; index those as prior data or move them after the environment step. Declare whether `o_{t+1}` enters `U`, when writes commit, and whether writer randomness is stored or independently resampled.

The bit formula is valid only for the stored slot tensor. The deployment ledger must also count per-layer persistent KV/state, double buffers needed for atomic writes, routing/age/confidence metadata, error correction, host and device copies, serialization, and any state shared across replicas. Report effective served precision, not training dtype alone.

For self-attention memory tokens, quote both total and incremental work: `(C+s)^2` versus baseline `C^2`, an attention-score increment `2Cs+s^2`, plus projections, MLPs, KV traffic, and the writer. Cross-attention has `Cs` score work but still pays projections, memory reads, writes, and synchronization. Measure latency, bandwidth, peak memory, energy if available, and write frequency—not FLOPs alone.

Training must charge lifetime rollout/simulator calls, all generated tokens, verifier rewards, BPTT/recomputation length, activation memory, backward traffic, and hyperparameter search. The headline “30% lower regret” and “2x faster acquisition” need predeclared definitions: regret comparator, acquisition threshold, confidence interval, seeds, lifetime horizon, and aggregation across families. A later small-versus-large-model result must charge the small learner’s interaction history, persistent bytes, update latency, and offline lifetime-training bill.

## Collision and baseline audit

The current collision list is accurate and makes novelty claims untenable but the capability hypothesis still live:

- [ORBIT](https://arxiv.org/abs/2602.04089) and [LaMer](https://arxiv.org/abs/2512.16848) are strong positive controls for cross-episode meta-RL and online acquisition, already reporting large gains on unseen/harder interactive environments. Their adaptation uses growing in-context experience/reflection rather than the exact bounded latent/erasure contract.
- [State commitment learning](https://arxiv.org/abs/2606.05201) occupies counterfactual erasure and persistent-state sufficiency.
- [Trained Persistent Memory](https://arxiv.org/abs/2603.22329) directly occupies gradient-free latent write/read adapters for a frozen decoder-only model and shows strong capacity/inductive-bias sensitivity; its GPT-2/LoCoMo scope is not evidence of production-level general learning.
- [In-Place TTT](https://arxiv.org/abs/2604.06169) is the direct fast-weight, deployment-gradient alternative and must be compared at total state plus optimizer/backward cost.
- [NextLat](https://arxiv.org/abs/2511.05963) and [continual experiential latent memories](https://arxiv.org/abs/2606.17803) occupy compact predictive and modular latent adaptation; the latter uses test-time gradient steps and retrieval, so it is a resource-matched competitor rather than the same mechanism.
- [Unsupervised meta-RL](https://arxiv.org/abs/1806.04640) states the core limitation: a meta-learner inherits its task distribution, shifting design burden to meta-training. Cross-family claims require a declared shared meta-distribution, not an arbitrary unseen problem.
- [Replay-MAML PTW](https://proceedings.mlr.press/v330/koop26a.html) is a strong task-agnostic change-point/continual control in piecewise-stationary classification. Its `O(log T)` memory and computation grow with lifetime and its base learners use gradients, so it is not the same bounded deployment contract; charge those differences rather than excluding it.

Fair comparison is a Pareto comparison, because bytes, FLOPs, latency, interactions, and offline training cannot always all be exactly equal. Every method gets the same observations, feedback, boundary markers, task IDs (preferably none), and interaction budget. Report frontiers over total static parameters, all persistent/current storage, update and forward latency, environment calls, and training cost. Give text/retrieval controls freedom to use their budget optimally; do not force them into T70's slot format.

Most importantly, “the strongest same-state recurrent control” cannot be both outside and inside the hypothesis. If a lifetime-trained GRU/SSM wins, T70 has found an implementation. The causal ablation is lifetime credit/hard-erased persistence versus the same operator trained without cross-boundary future loss; architecture variants select the best realization. External text/retrieval, In-Place TTT, ORBIT/LaMer-style context, Replay-MAML PTW, and a full-context information oracle are comparative controls.

## Is cross-family transfer one coherent hypothesis?

**Conditionally yes.** It is coherent if all families share one typed serialized interaction interface and a predeclared meta-distribution whose common problem is “infer a compact controllable belief/rule from feedback, retain it, revise it at change points.” It is not coherent as unrestricted transfer to an arbitrary new action space or task; no meta-learner can guarantee that without shared structure.

Use the newly added full-context oracle gate to separate processor competence from writing. For a clean held-out-family test:

1. train/verify the backbone to solve every family when the full transcript is present;
2. freeze that competence or recheck it after lifetime training;
3. meta-train the bounded writer/update on only the training families, with no family ID and randomized surface forms;
4. test its retained-state recovery of the full-context-oracle gain on a held-out family; and
5. add factorial held-out combinations of familiar primitives, which are a stronger identifiable composition test than one unrelated family.

Without step 1, failure may be absent algebra/planning competence. Without steps 3–5, success may be routing among memorized mini-algorithms. Report per-family results; a favorable mean cannot hide a failed family.

## Why a cheap local run is admitted

The lookup witness guarantees an order-one signal for channel integrity, and the target thresholds are large enough that a tiny model need not resolve a sub-percent effect. The affine correction means there is no theorem guaranteeing a win over optimal storage, so the executable question is genuinely empirical: can one learned writer preserve and systematically use a sufficient code, retain unrelated rules, handle change, and transfer its writing behavior?

The local screen should be predeclared, short, and staged:

1. require the full-context oracle to pass before interpreting memory results;
2. require post-erasure recovery of a large fixed fraction of the oracle improvement, with constant and independent-shuffle states at theorem chance;
3. require the stated `>=30%` regret and `>=2x` acquisition improvements against the best bounded non-lifetime/text/retrieval baselines across seeds—not merely reset;
4. require unchanged-rule performance within the declared one-point margin after a change, plus horizon extrapolation;
5. require held-out-family or held-out-composition transfer under the protocol above; and
6. stop immediately if leakage, oracle failure, or a strong bounded baseline ties.

Passing this screen authorizes only a natural-domain pilot. Failure closes the tested composition/implementation unless the failure is the predeclared full-context-oracle failure; no rented multi-hour run is justified from synthetic recall alone.
