# T46 interventional predictive-state boundary — information before architecture

Date: 2026-08-01  
Status: **AUDITED HOLD: CLASSICAL PASSIVE-VERSUS-INTERVENTION BOUNDARY; NO LEARNED MECHANISM OR EXPERIMENT ADMITTED**

## 0. Question

T45 began after the important decision had already been made: it supplied the
correct hidden factor and likelihood, leaving only evidence accumulation. T46
moves one step earlier:

> What can interaction identify that passive prediction cannot, what is the
> minimal sufficient controlled state, and how many informative interventions
> are unavoidable?

The construction below is a theorem world and an oracle control. It does not
claim that a new neural architecture has been found.

## 1. Controlled predictive equivalence

For positive-probability histories `h,h'` at the same time and with the same
legal action/observation interface, define

\[
h\sim_I h'
\Longleftrightarrow
\forall k,\forall a_{1:k},\quad
P(o_{1:k},r_{1:k}\mid h,\operatorname{do}(a_{1:k}))
=P(o_{1:k},r_{1:k}\mid h',\operatorname{do}(a_{1:k})).
\]

This is the controlled analogue of an ordinary predictive quotient. Two
histories are merged only when no future experiment or policy can expose a
behavioral difference. Adaptive tests are equivalent only when the same map
from *future* observation/reward suffixes to actions is used after both starting
histories; a policy may not inspect the identity of the starting history and
choose different actions merely because `h` and `h'` are different strings.
Predictive-state representations, causal states,
bisimulation, and Bayes-adaptive control already occupy neighboring algebra;
the quotient itself is a specification, not a novelty claim.

## 2. Hidden-parent intervention family

Let `J` be uniform on `{1,...,K}`. `J` names which one of `K` candidate causes
actually controls a binary outcome. At intervention `t`, the agent chooses a
subset `S_t` and receives

\[
O_t=\mathbf 1\{J\in S_t\}\oplus N_t,
\qquad N_t\sim\operatorname{Bernoulli}(\eta),
\qquad 0\le\eta<1/2,
\]

with independent noise. A passive observation is `S_t=emptyset`; its output is
only `N_t` and is independent of `J`.

**Concrete meaning.** Imagine `K` switches but only one is wired to a lamp. An
experiment may toggle a chosen group. The noisy lamp tells whether the live
switch was in that group.

The family deliberately separates four resources:

1. passive observations contain no causal orientation information;
2. the legal intervention names which alternatives it separates;
3. the action must remain associated with its outcome; and
4. state must accumulate the resulting posterior across trials.

## 3. T46.1 — passive impossibility

For any number of passive samples,

\[
P(O_{1:n}\mid J=j,S_{1:n}=\varnothing)
=P(O_{1:n}\mid J=j',S_{1:n}=\varnothing)
\]

for all `j,j'`. Therefore

\[
I(J;O_{1:n}\mid S_{1:n}=\varnothing)=0.
\]

No predictor, parameter count, context length, recurrence depth, or optimizer
can identify `J` from that channel better than the prior. This is missing
information, not weak computation.

## 4. T46.2 — exact controlled state and closed update

Let

\[
p_t(j)=P(J=j\mid S_{1:t},O_{1:t}).
\]

For `b_j(S)=1{j in S}`, define

\[
q_\eta(o\mid b)=
\begin{cases}
1-\eta,&o=b,\\
\eta,&o\ne b.
\end{cases}
\]

Then the exact update is

\[
p_{t+1}(j)=
{p_t(j)q_\eta(O_{t+1}\mid b_j(S_{t+1}))
 \over
 \sum_i p_t(i)q_\eta(O_{t+1}\mid b_i(S_{t+1}))}.
\]

The posterior is sufficient for every future intervention because the future
law is a mixture over `J`. For known `K,eta`, static `J`, homogeneous independent
noise, positive-probability histories, and a legal singleton action for every
`j`, it is also minimal for exact prediction of the full controlled future law:
singleton interventions give

\[
P(O=1\mid S=\{j\},h)=\eta+(1-2\eta)p_h(j),
\]

so two histories with identical controlled futures must have the same value of
every `p_h(j)`. Thus the exact interventional predictive quotient is the
reachable posterior state, up to a one-to-one recoding.

This is not a minimal-storage theorem: the posterior has `K-1` real degrees of
freedom and no finite-bit encoding is established. If the legal action family
cannot expose singleton coordinates, the quotient may be strictly coarser.

This establishes semantics and update closure. It does not show that a neural
learner can discover either one from raw interaction surfaces.

## 5. T46.3 — unavoidable intervention information

Every binary noisy intervention is a binary-symmetric channel whose mutual
information is at most

\[
C_\eta=1-h_2(\eta)
\]

bits. Let the fixed-length transcript be
`H_n=(S_1,O_1,...,S_n,O_n)`, with any policy randomness independent of `J`
conditional on prior history. If a decoder identifies `J` after `n`
interventions with error at most `epsilon`, Fano and the chain rule give

\[
n\ge\left\lceil
{[\log_2K-h_2(\epsilon)-\epsilon\log_2(K-1)]_+
 \over 1-h_2(\eta)}
\right\rceil.
\]

**Proof sketch.** Fano lower-bounds `I(J;H_n)` by the numerator. Conditional on
the past and selected subset, the selected action contributes no new
information because it is a function of existing history, while one
binary-symmetric observation contributes at most its channel capacity:

\[
I(J;H_n)=\sum_t I(J;O_t\mid H_{t-1},S_t)
\le n[1-h_2(\eta)].
\]

QED. This is a fixed-length converse; adaptive stopping times and expected
sample counts need a variable-length bound.

In the noiseless case, choose a binary code for the `K` hypotheses and let
intervention `r` contain exactly the hypotheses whose `r`-th code bit is one.
The `ceil(log2 K)` outcomes recover `J`, meeting the zero-error information
bound.

For noisy outcomes, let `d=ceil(log2 K)` and repeat each code-bit intervention
an odd number `m` of times. Majority vote and a union bound give

\[
P(\widehat J\ne J)
\le d\exp\{-2m(1/2-\eta)^2\}.
\]

It is sufficient to choose

\[
m=\operatorname{nextOdd}\left(
\left\lceil{\ln(d/\epsilon)\over2(1/2-\eta)^2}\right\rceil
\right),
\qquad n=dm.
\]

This constructive upper bound is not claimed optimal for noise. Posterior
splits can adapt to nonuniform beliefs and heterogeneous test costs.

## 6. T46.4 — informative action rule

Let `q_t(S)=sum_{j in S} p_t(j)`. The next outcome has probability

\[
P(O=1\mid p_t,S)=\eta+(1-2\eta)q_t(S).
\]

Its conditional information about `J` is

\[
I(J;O\mid p_t,S)
=h_2(\eta+(1-2\eta)q_t(S))-h_2(\eta),
\]

maximized when the intervention splits posterior mass as close to one half as
the legal action family allows. This is the exact **one-step entropy-gain** rule,
not a generally optimal policy under unequal action costs, risks, rewards,
finite horizons, or restricted future actions. Those cases require a charged
objective and potentially dynamic programming.

The policy is an oracle. A candidate must learn which real actions implement
useful partitions and must charge their cost and risk.

## 7. T46.5 — lossy action-outcome pairing counterexample

Correct outcome accumulation is insufficient if the responsible intervention
is lost. Let `K=4` and perform

\[
S_1=\{1,2\},\qquad S_2=\{2,3\}.
\]

In the noiseless subcase, suppose the system retains the unordered action
multiset and unordered outcome multiset but discards which outcome followed
which action. Equivalently for this pair, it retains only the aggregate bit

\[
O=\mathbf1\{J\in S_1\}\oplus\mathbf1\{J\in S_2\}.
\]

For `J=1`, the paired outcomes are `(1,0)`; for `J=3`, they are `(0,1)`. Both
produce the same action multiset, outcome multiset, and aggregate bit, yet they
require different answers to the future singleton intervention `{1}`. A state
that stores outcomes without the action pairing needed to interpret them is not
a sufficient learning state.

This proves loss from a one-bit/pairing-destroying summary, not a delayed-credit
theorem: the construction has immediate outcomes and no latent transition.
Delayed stochastic feedback needs a separate pair of mechanisms whose
likelihoods differ only through causal assignment to earlier actions. Any later
candidate must preserve or reconstruct that routing edge rather than assuming
this example already proves it.

## 8. What the theorem actually buys

Inside this deliberately simple hidden-index membership family, the gap is
qualitative:

- passive prediction has zero information about `J`, even with unlimited
  samples from the passive channel;
- legal interventions identify `J` with logarithmic experiments in the
  noiseless family; and
- a posterior-mass split is the exact information-seeking action.

This illustrates one thing passive data alone cannot guarantee: a channel may
leave action-relevant mechanisms unidentified. It is not yet a structural
causal-model discovery theorem—the passive channel was defined as pure noise,
and the family contains no confounding, intervention side effects, unknown
action semantics, or observational Markov-equivalence class.

It does **not** prove an architecture edge. A Transformer with action/outcome
context, an RNN/SSM, a PSR, a Bayes filter, episodic memory, or a meta-RL agent
can all implement this family. The missing neural edge is learning a reusable
controlled quotient and provenance-aware update across unseen surfaces and
world structures more reliably or cheaply than those controls.

## 9. Prior-art and strongest-control boundary

- Predictive-state representations already define state through controlled
  future tests.
- Recurrent Predictive State Policy networks already combine predictive state
  updates with a learned policy.
- RL-squared, VariBAD, Algorithm Distillation, and ORBIT cover implicit
  recurrent/meta-learned update algorithms.
- Predictive-coding meta-RL reports better acquisition of Bayes-like belief
  representations than ordinary meta-RL in partially observable tasks.
- Active causal discovery and group testing already provide information-aware
  intervention design.
- PABU and Unified Memory Agent cover compact learned belief/memory management
  in LLM agents.

Noisy twenty-questions and posterior matching also contain the subset-selection
and entropy-splitting result. Therefore neither “use interventions,” “store a
posterior,” “use a predictive state,” nor their simple combination is a novelty
claim.

## 10. Hidden oracle assumptions

T46 hands the oracle the known hypothesis count and prior, known homogeneous
noise, a static hidden index, arbitrary free and riskless subset interventions,
known action-to-partition semantics, immediate correctly paired outcomes, exact
posterior arithmetic, and no exploration cost. It tests none of surface
grounding, factor discovery, delayed credit, change points, abstraction,
consolidation, or finite-state approximation.

## 11. Admission result

T46 admits no experiment. Its retained result is a sharper target:

```text
learned surface encoder
    -> intervention-distinguishable predictive tests
    -> provenance-aware closed update
    -> calibrated state
    -> information-seeking action
```

The next paper object is the interventional-signature kernel

\[
G^\star(a,j,o)=P(O=o\mid\operatorname{do}(a),J=j),
\]

with renamed action surfaces and latent factors across episodes. It must derive
necessary and sufficient observable anchor/rank/separation conditions for
recovering this kernel up to control-equivalent permutation. Given the kernel,
posterior update and information-seeking control remain exact oracle blocks, so
only the action-surface-to-intervention likelihood map is empirical. If no raw
anchor makes it identifiable, a permutation/no-feedback theorem closes the
direction.

No theorem-world, CPU, model, or GPU run is admitted by T46.
