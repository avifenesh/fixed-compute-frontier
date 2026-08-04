# Independent audit — T78 prospective counterfactual abstraction credit

Date: 2026-08-02  
Verdict: **METHOD CLOSED; POSITIVE/NEGATIVE INFORMATION BOUNDARY RETAINED; NO RUN**

## 0. Bottom line

T78.1 and T78.2 are correct as narrow information separations. They do not
establish an abstraction-formation advantage. The prospective policy receives
information about a persistent future phase that the restricted current/past
selector ignores; a matched recurrent meta-learner can use the same information.

The proposed paired score is a useful admission measurement, but with a common
no-candidate baseline it selects exactly the same candidate as ordinary
cost-sensitive held-out empirical risk. It supplies an effect size, threshold,
and possibly lower variance—not a new learning objective.

Direct current work closes the method claim. Prospective Compression already
formalizes future-sensitive online library learning, ALMA selects executable
memory programs using deployment tasks, and SkillMaster explicitly trains
candidate skill edits by counterfactual utility on related probe tasks.

## 1. Correct resource-adjusted objective

Let `M_t` be the current library and `z` a candidate. Define complete
resource-adjusted loss

\[
R_j(M)=\ell_j(M)
+\lambda_{tok}C_{tok,j}(M)
+\lambda_{time}C_{time,j}(M)
+\lambda_{energy}C_{energy,j}(M)
+\lambda_{update}C_{update,j}(M).
\]

The discounted horizon value is

\[
U_{t,H}(z)=
\mathbb E\left[
\sum_{h=1}^{H}\gamma^{h-1}
\left(R_{t+h}(M_t)-R_{t+h}(M_t\cup\{z\})\right)
\middle|\mathcal H_t
\right]
-C_{birth}(z).
\]

With a bounded full library, the correct comparison is usually candidate `z`
against the best feasible replacement, not against a free empty slot. If the
library continues learning, the two arms become separate downstream state
trajectories. Freezing all other updates estimates the direct artifact effect;
allowing updates estimates the total policy-plus-learning-path effect.

If the no-`z` baseline is common across candidates,

\[
\arg\max_z\sum_j
\left[R_j(M_t)-R_j(M_t\cup\{z\})\right]-C(z)
=
\arg\min_z\sum_jR_j(M_t\cup\{z\})+C(z).
\]

Thus paired causal language does not change which candidate is selected. The
intervention identifies the effect of making the artifact available under the
evaluation policy. It does not establish semantic abstraction or causal truth.

## 2. Positive result and why it is insufficient

The strongest clean construction uses a persistent latent curriculum bit
`Theta~Bernoulli(1/2)` and two helpers

\[
z_b(x)=x\oplus b.
\]

Past/current evidence is independent of `Theta`; future labels satisfy

\[
X_j\sim Bernoulli(1/2),\qquad Y_j=X_j\oplus\Theta.
\]

Every past-only selector has expected zero-one loss `1/2`. One labeled future
probe identifies `Theta=X_j xor Y_j` exactly, after which the correct helper has
zero loss. For `H` deployment tasks and validation cost `c_val`, the gross
advantage is

\[
\frac H2-c_{val}.
\]

This is order-one, but it is an information separation rather than an
abstraction theorem. A generic recurrent learner sees the first labeled future
task, stores one bit, and is then perfect. Prospective prevalidation avoids at
most its first expected half-error while spending validation calls. It helps
only when early mistakes are unusually expensive or validation is much cheaper
than deployment.

## 3. Standard fixed-candidate guarantee

For `K` fixed candidates with conditionally independent bounded paired gains
`D_j(z) in [-B,B]`, Hoeffding plus a union bound gives, with probability at
least `1-delta`,

\[
\sup_z|\widehat U_m(z)-U(z)|
\le
B\sqrt{\frac{2\log(2K/\delta)}{m}}.
\]

The empirical maximizer therefore has regret at most

\[
2B\sqrt{\frac{2\log(2K/\delta)}{m}}.
\]

This is ordinary finite-class held-out selection. Candidate formation on the
same validation tasks invalidates the guarantee; proposal, admission, and final
meta-test streams must be separated or covered by an adaptive complexity result.

## 4. Fatal boundaries

### No persistent relatedness

If each future task draws an independent latent phase, previous probes contain
no information about the next task. Each fixed helper has loss `1/2`, and any
positive library cost makes prospective admission worse.

### Distribution reversal

If validation uses one phase and deployment switches to the opposite phase,
validation chooses the exactly wrong helper. Nonstationarity must be modeled,
not merely named.

### Complementarity

Let

\[
R(\varnothing)=R(\{a\})=R(\{b\})=1,
\qquad R(\{a,b\})=0.
\]

Singleton admission rejects both useful candidates. Conversely, with

\[
R(\varnothing)=1,
\qquad R(\{a\})=R(\{b\})=R(\{a,b\})=0,
\]

the first candidate receives all marginal credit. Individual intrinsic utility
is undefined without a library reference distribution, coalition evaluation,
or an assumption such as monotone submodularity. Real skill libraries can be
complementary, redundant, and order-dependent, so the usual greedy guarantee
cannot be assumed.

### Proposal coverage

Typing does not create useful candidates. If one independent proposal has
probability `p_epsilon` of being epsilon-optimal, `n` attempts cover such a
candidate with probability

\[
1-(1-p_\epsilon)^n.
\]

A typed proposer helps only if it improves this probability at matched proposal
tokens, compute, and wall time. T78 has no formation or search-complexity theorem.

## 5. Direct collision audit

- [Prospective Compression in Human Abstraction Learning](https://arxiv.org/abs/2605.09985)
  directly owns future-corpus abstraction selection under latent nonstationary
  curricula.
- [ALMA](https://arxiv.org/abs/2602.07755) searches executable memory update,
  storage, and retrieval code, then evaluates it with a fixed agent on later
  deployment tasks.
- [SkillMaster](https://arxiv.org/abs/2605.08693) is the decisive method
  collision: candidate skill edits are evaluated by counterfactual utility on
  related probe tasks; create/refine/select decisions are trained jointly with
  task behavior. It reports `8.8` and `9.3` absolute success-rate gains on
  ALFWorld and WebShop.
- [AgentCL](https://arxiv.org/abs/2606.02461) supplies compositional stream,
  stability, plasticity, and held-out negative-transfer controls.
- [RLAD](https://arxiv.org/abs/2510.02263) trains an abstraction proposer through
  an abstraction-conditioned solver, although on the same problem.
- [SCoL](https://arxiv.org/abs/2605.07076) meta-trains persistent sparse weight
  updates over evolving model state.
- [DreamCoder](https://arxiv.org/abs/2006.08381) remains the executable
  retrospective-library control, while a prospective version over held-out
  search/compression is the closest objective control.
- [Inducing Reasoning Primitives from Agent Traces](https://arxiv.org/abs/2606.02994)
  already creates typed pseudo-tool libraries and evaluates held-out transfer.
- [Decision-Aware Memory Cards](https://arxiv.org/abs/2606.08151) scores typed
  memory units using outcome uplift, necessity, action shift, and negative risk.

The tightly specified conjunction may differ in implementation, but no
unoccupied learning principle remains.

## 6. Cost and controls

Naive evaluation costs at least `2*K*m*r` full solver runs for `K` candidates,
`m` future tasks, `r` repeats, and two arms. It must additionally charge proposal
calls, task generation and verification, compilation/type checking, rejected
candidates, branched state replay, delayed credit, and untouched final
meta-testing.

Deployment must charge serialized bytes, indexes/embeddings, retrieval tokens,
routing latency, execution/sandbox overhead, fallback, cache effects, version
migration, garbage collection, and the opportunity to spend saved no-candidate
work on more ordinary solving.

The decisive controls are plain held-out ERM on the identical candidates,
random same-size libraries, retrospective and prospective compression, RLAD,
SkillMaster, ALMA, AgentCL memory families, SCoL, exact finite-DSL enumeration,
and a generic recurrent meta-RL learner with identical information, persistent
bits, and total compute.

## 7. Verdict

Retain the positive order-one information example, the independence no-go, and
the resource-adjusted admission equations. Close the method. It is not a new
route to real intelligence, and reproducing current skill-learning work cannot
earn a CPU or GPU experiment.

