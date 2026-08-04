# Independent audit — T69 counterfactual-fork lifetime training

Date: 2026-08-02  
Audited file: `results/counterfactual-fork-lifetime-training-t69.md`  
Scope: theorem/proof correctness, fork statistics and oracle accounting, named collisions and controls, and admission under the goal of substantial real-intelligence improvement. No experiment was run.

## Verdict

**REVISE THE FORMAL RESULTS; CLOSE THE BROAD MECHANISM; NO RUN.**

The closure decision is correct, but not all retained mathematics is. T69.1 is false under its four stated assumptions: strict propriety does not prevent an insufficient bottleneck from merging histories. T69.2 is a valid experiment-*coverage* union bound, not an exposure/detection or learning bound. T69.3 is correct as an objective-permutation statement, while its Shannon-information sentence needs an explicit recoverability condition. The paired common-random-number estimator and variance formula are correct for equal-cost, correctly coupled scalar rollouts, but the claimed edge applies to action contrasts, requires positive covariance, and is fully available to a matched control.

None of these corrections revives the intelligence mechanism. A model trained on the identical action-conditioned tuples, branch metadata, paired seeds, and nonseparable losses receives the same evidence and can implement the same update. Forking is a potentially useful data-acquisition oracle, not a protected architecture or lifetime-learning advantage.

## Exact theorem audit and required edits

### T69.1 — false as written

Strict propriety identifies

\[
P(Y\mid \phi(H),E),
\]

not automatically `P(Y|H,E)`. If `phi` merges histories with different laws, the population-optimal decoder is their conditional mixture.

A counterexample satisfies all four stated assumptions. Let `H` be equiprobable in `{0,1}`, let there be one always-supported experiment, let `Y=H` deterministically, and constrain `phi` to one state. A Bernoulli decoder can represent either true law, optimization reaches the population optimum, support is positive, and the decoder sees only `(phi(H),E)`. The optimal log-loss decoder predicts `Bernoulli(1/2)`. Thus `phi(0)=phi(1)` at the population optimum although the histories are not predictively equivalent.

**Required replacement assumptions:**

1. optimization is over both `phi` and `q`, with no regularizer or auxiliary objective that trades predictive risk against compression;
2. the representation class has enough states to realize the entire declared predictive quotient, and one decoder can **jointly** realize every corresponding conditional law;
3. the learned pair attains the full-history Bayes risk (equivalently zero excess proper-score risk relative to conditioning on `(H,E)`); and
4. the discrete support assumptions have `P(H=h)>0` and `mu(E|h)>0` for every claimed `(h,E)`.

Under these assumptions, proper-score regret is an expected nonnegative strict divergence between `P(Y|H,E)` and `q(Y|phi(H),E)`. Equality with the full-history Bayes risk makes that divergence zero at every positive-mass pair, so a collision can occur only between histories with identical declared future laws. This proves the intended conclusion without the mixture error.

If `E` is continuous or uncountable, pointwise positive probability is impossible. The result then gives equality only `mu`-almost everywhere; extending it to **every** legal experiment needs continuity/dominance assumptions. Integrability/existence of the selected proper-score risk is also required. The converse and minimum-discrete-state statement are correct once restricted to a finite quotient, deterministic encoders, exact sufficiency, and positive-mass classes.

### T69.2 — correct coverage algebra, overstated interpretation

For a fixed finite set of `M` bad pairs, define for each pair a distribution over experiments legal from both histories, and let an experiment be “distinguishing” when the two **true** outcome laws differ. If each of `K` draws is independent for that pair and has distinguishing probability at least `rho`, then

\[
P(\text{some pair receives no distinguishing experiment})
\le M(1-\rho)^K\le Me^{-\rho K}.
\]

No independence across pairs is needed for the union bound. For `0<rho<1`, the exact integer condition is

\[
K\ge\left\lceil\frac{\log(M/\delta)}{-\log(1-\rho)}\right\rceil,
\]

with the displayed `log(M/delta)/rho` condition a conservative simplification. If `rho=1`, one test per pair suffices.

**Required wording change:** replace “expose every bad merge” with “sample at least one truly distinguishing experiment for every bad merge.” One stochastic outcome does not reveal that two laws differ. A detection theorem additionally needs a declared separation (for example, total variation, mean effect relative to variance, or a test-class discrepancy), repeated rollouts per `(history, experiment)`, a calibrated test, and family-wise error/power control. If the bad-pair set, experiment policy, or `rho` is selected from the same data, the bound must be conditional on a fresh split or uniform over that adaptive selection. Unknown or exponentially small `rho` remains fatal.

### T69.3 — objective invariance correct; information claim conditional

The empirical sum is invariant to permutation. With identical example weights, a fixed model, an exactly separable loss, and identical grouping-independent regularization, shuffled and fork-grouped presentation has the same objective and global minimizers. Batch order can change an optimizer trajectory, but a matched schedule removes that comparison.

“Grouping adds zero Shannon information” is not well-defined without naming a random variable of interest. The precise recoverability statement is

\[
H(G\mid D)=0,
\]

where `G` is sibling membership, **if** every tuple retains a stable exact history/snapshot identifier from which sibling groups are a deterministic function. If histories were anonymized, aliased, augmented, or deduplicated, or if grouping reveals a shared seed/snapshot not present in the tuples, `H(G|D)` need not be zero. Group-aware contrastive losses, shared-seed differences, batch normalization, cross-example attention, or stateful within-group updates are nonseparable algorithms; their metadata and loss must be given to the matched control.

## Common-random-number result and oracle ledger

For scalar outcomes `Y(a)=F(h,a,U)`, iid base variables `U_i`, and correct marginals under both actions,

\[
\widehat\Delta=\frac1n\sum_i(F(h,a,U_i)-F(h,b,U_i))
\]

is unbiased for `E[Y(a)]-E[Y(b)]`. Independence across seeds gives the stated variance

\[
\frac{\sigma_a^2+\sigma_b^2-2\operatorname{Cov}(Y(a),Y(b))}{n}.
\]

The unpaired formula assumes `n` independent samples **per action**. This is a fair `2n`-simulator-call comparison with `n` paired seeds, which also require two calls. Positive covariance lowers contrast variance; zero covariance gives no gain, and negative covariance makes pairing worse. The additive-noise zero-variance example is correct.

Three boundaries must be added:

1. pairing reduces variance of the **contrast**. It does not reduce the marginal sampling variance for learning each complete future law, so it does not by itself establish controlled predictive sufficiency;
2. restoring one PRNG state is not automatically a structural counterfactual coupling. Different actions may consume random draws differently. The implementation must define aligned exogenous variables or accept that it has only a common-random-number coupling, not a unit-level causal effect; and
3. a learned simulator supplies no new real-world information. Its model-training cost and counterfactual bias must be charged, and conclusions must be validated against real held-out interaction.

The fair cost expression is at least

\[
C_{snapshot}+K C_{restore}+K L(C_{env}+C_{model})+C_{update},
\]

plus full latent-state/RNG capture, snapshot storage, failed branches, active experiment selection, and simulator construction/validation. Paired-effect training must charge two branches per pair. A matched ordinary world model must receive the same branches, seed/sibling identifiers, and paired-difference or other nonseparable objective; then it gets the identical variance reduction.

## Collision and control audit

The cited sources are current and directionally relevant, but the collision strengths should be separated:

- [Recurrent Predictive State Policy Networks](https://proceedings.mlr.press/v80/hefny18a.html) is a direct collision with action-conditioned predictive-state learning and downstream control.
- [Action-conditional self-predictive RL](https://proceedings.mlr.press/v258/khetarpal25a.html) directly studies future-action-conditioned latent prediction, but not arbitrary forked outcome laws; its reported performance is task-dependent.
- [Action-Sufficient State Representations](https://proceedings.mlr.press/v162/huang22f.html) is a direct compact-state/control collision, although action sufficiency can be coarser than the full controlled predictive quotient.
- [NextLat](https://arxiv.org/abs/2511.05963) is adjacent evidence that predictive auxiliary losses can induce compact belief-like state; it is not a fork-oracle method.
- [ORBIT](https://arxiv.org/abs/2602.04089) is evidence for cross-episode in-context online learning on unseen environments, not a direct counterfactual-fork collision.
- [State commitment learning](https://arxiv.org/abs/2606.05201) is the closest modern counterfactual-training collision: same-prefix keep/erase branches enforce persistent-state sufficiency through a nonseparable criterion. It shows that the actual candidate is the branch relation and objective, which a fair control must share.

The direct impossibility result does not depend on claiming that every adjacent block is identical. For fixed tuples and a separable objective, fork naming changes nothing. For paired or nonseparable training, the control can consume the same oracle and loss. A control denied grouping or common seeds is intentionally weaker and cannot establish a mechanism gain.

## Admission decision under the corrected goal

**NO RUN is correct.** A local toy can demonstrate coverage, CRN variance reduction, or a batching effect, but those are already mathematical/statistical facts and cannot show a smarter production model. To reopen the direction, the candidate must identify evidence obtainable per real interaction, or an update/abstraction that cannot be reproduced by a matched action-conditioned learner given the same fork oracle, metadata, loss, compute, and memory. It must then predict a substantial held-out gain in long-horizon adaptation, planning, or transfer across genuinely new raw, stochastic environments—not merely lower loss on simulator siblings.

T69 should therefore be retained only after correcting T69.1, narrowing T69.2 and T69.3, and labeling common-random-number pairing as a contrast-estimation oracle. The broad counterfactual-fork lifetime-training mechanism remains closed.
