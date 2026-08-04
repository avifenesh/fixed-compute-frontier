# T47 operator-signature intervention kernel — grounding actions by effects

Date: 2026-08-01  
Status: **AUDITED NO-GO AS A TRANSFER CANDIDATE; DETERMINISTIC ORACLE ALGEBRA RETAINED; NO EXPERIMENT**

## 0. Why this is the next object

T46 assumed the complete intervention kernel: every action surface already
named a known partition of the hidden hypotheses. That is where most of the
intelligence was hidden.

T47 asks whether action meaning can instead be recovered from consequences
under arbitrary renaming:

\[
G^\star(a,j,o)=P(O=o\mid\operatorname{do}(a),J=j).
\]

The first tractable realization treats actions as state-transition operators.
It is a different conjugate linear-system family, not an instantiation of
T46's hidden-index stochastic kernel: it has no latent hypothesis `J`,
likelihood update, or posterior.
The intended bridge is

```text
renamed action surface + observed effects
    -> coordinate-invariant operator signature
    -> shared intervention type
    -> reusable likelihood/update/control law
```

This is system identification and invariant algebra before it is a neural
architecture. Its purpose is to expose exactly when consequence-based grounding
can and cannot replace a language label.

## 1. Renamed operator world

Let the canonical latent state be `x in R^d` and let the canonical action
library be matrices

\[
\mathcal A=\{A_1,\ldots,A_m\}.
\]

Episode `e` changes both coordinate names and action surface names:

\[
y=P_e x,
\qquad
u=\rho_e(a),
\]

where `P_e` is unknown and invertible and `rho_e` is an unknown permutation of
the action library. Applying surface action `a` produces

\[
y'=B_{e,a}y,
\qquad
B_{e,a}=P_eA_{\rho_e(a)}P_e^{-1}.
\]

Thus raw coordinates and action strings may be unrelated across episodes, but
the underlying operator algebra is shared. Full-state deterministic observation
is deliberately strong; later partial/stochastic worlds must earn their own
identifiability result.

## 2. T47.1 — exact within-episode operator recovery

Suppose an oracle can choose, reset, and legally reach `d` probe states and form

\[
Y=[y_1|\cdots|y_d],
\qquad
Y'_a=[y'_1|\cdots|y'_d].
\]

If `Y` is nonsingular and each transition is paired with action `a`, then

\[
B_{e,a}=Y'_aY^{-1}.
\]

This is a strong system-identification oracle, not an ordinary full-state
observation assumption. If probes arise endogenously, the admissible condition
is a bounded empirical excitation Gramian for the legally generated design,
with every state-preparation action and rejected trajectory charged.

The identity is also a boundary. If the reachable probe span is a
proper subspace `U`, two operators that agree on `U` but differ on its
complement generate exactly the same probe data. Full operator recovery is then
impossible without a structural restriction.

The conditioning quantity `sigma_min(Y)` and every reset/probe action are part
of the interaction ledger. An ill-conditioned probe basis may make the exact
identity statistically useless.

## 3. T47.2 — coordinate-invariant action signatures

For integer `L>=1`, define the power-trace signature

\[
s_L(A)=\big(\operatorname{tr}(A),\operatorname{tr}(A^2),\ldots,
\operatorname{tr}(A^L)\big).
\]

Similarity leaves it unchanged:

\[
s_L(PAP^{-1})=s_L(A).
\]

Let the finite canonical library have separation

\[
\delta_L=\min_{u\ne v}\|s_L(A_u)-s_L(A_v)\|_\infty>0.
\]

Then exact recovery of `B_{e,a}` identifies its canonical type by matching
`s_L(B_{e,a})` to the library. This aligns action meaning across arbitrary
invertible coordinate changes and arbitrary surface-label permutations.

The theorem is only relative to the declared finite library and assumes the
separation it uses. Power traces recover eigenvalue-multiset information, not
Jordan structure. For `d` by `d` matrices, higher powers add no generic spectral
information beyond Newton/Cayley-Hamilton relations, while simultaneous
conjugacy of an action tuple may require mixed-word invariants such as
`tr(A_u A_v A_w)`. Action types outside the library remain open-set unknowns.

## 4. T47.3 — epsilon-stable signature recovery

Assume

\[
\|B\|_2\le M,\qquad \|\widehat B\|_2\le M,
\qquad \|\widehat B-B\|_2\le\epsilon_B.
\]

The telescoping identity

\[
\widehat B^k-B^k
=\sum_{r=0}^{k-1}\widehat B^r(\widehat B-B)B^{k-1-r}
\]

gives

\[
|\operatorname{tr}(\widehat B^k)-\operatorname{tr}(B^k)|
\le d k M^{k-1}\epsilon_B.
\]

Therefore nearest-signature matching against an exact prototype library is
exact whenever

\[
2\max_{1\le k\le L} d k M^{k-1}\epsilon_B<\delta_L.
\]

This propagates an operator-estimation error into a discrete action-grounding
guarantee. It does not yet derive `epsilon_B` from noisy trajectories; that
requires a frozen observation-noise law, number of probes, and lower bound on
`sigma_min(Y)`.

The robustness statement is not invariant under unrestricted `P_e in GL(d)`:
similarity preserves traces but not Euclidean operator norms, probe condition,
observation-noise norms, or commutator margins. A statistical theorem must
bound `condition(P_e)`, specify covariant noise, or abandon latent-coordinate
norms for observable controlled-test probabilities. If both query and learned
prototype signatures have the same worst-case coordinate error `Q`, the safe
separation condition becomes `4Q<delta_L`, not `2Q<delta_L`.

## 5. T47.4 — unavoidable automorphism ambiguity

Define an automorphism of the action system as a pair `(C,sigma)` satisfying

\[
CA_uC^{-1}=A_{\sigma(u)}\quad\text{for every }u,
\]

while also preserving, under the same recoding, the initial-state distribution,
legal reset/probe set, process and observation noise, observation map,
reward/cost readout, and surface-action dictionary. Only under this complete
symmetry do two action dictionaries induce the same controlled trajectory law.
No learner can choose between them.

Hence action types are identifiable at best modulo the automorphism group of
the controlled system. A global permutation is harmless only when every writer,
reader, posterior update, and policy uses the same recovered basis. If two
library actions have the same signature but different downstream value, the
simple trace bridge is insufficient even if some stronger joint invariant
could distinguish them.

## 6. T47.5 — provenance from noncommutativity

Suppose feedback is delayed until after two actions. If the system retains only
the unordered action multiset, order is unidentifiable whenever

\[
A_uA_v=A_vA_u.
\]

Both orders give the same endpoint for every starting state.

If instead the commutator is nonzero,

\[
[A_u,A_v]=A_uA_v-A_vA_u\ne0,
\]

then some mathematical probe `x` has

\[
\|(A_uA_v-A_vA_u)x\|_2>0.
\]

This does not imply that a separating probe is legally reachable. The relevant
observable margin is

\[
\mu_{uv}=\sup_{x\in\mathcal X_{legal},\|x\|\le R}
\|[B_u,B_v]x\|_2.
\]

Order is observable only if this legal observed-coordinate margin is positive.
If two exact candidate endpoints are known and the unknown endpoint has noise
at most `epsilon_o`, nearest-endpoint discrimination is guaranteed when

\[
\|(A_uA_v-A_vA_u)x\|_2>2\epsilon_o.
\]

This is a small but genuine provenance theorem: noncommuting consequences can
orient an otherwise renamed two-action sequence. It does not solve arbitrary
delayed credit. Commuting dynamics, hidden states, long sequences, stochastic
effects, and multiple simultaneous causes require additional information.
If both stored endpoint prototypes and the target have error `epsilon_o`, use
the conservative margin `mu_uv>4 epsilon_o`. A reward or observation map may
also erase the commutator direction entirely.

## 7. What this object could buy

If all oracle assumptions survive, a system can ground an action-library index
by
what the action *does*, not what its string is called. A canonical transition,
belief-update, or control law learned in earlier episodes can then be reused in
a new episode whose coordinates and action labels are permuted.

But action labeling does not recover `P_e`. A canonical state-dependent policy
`pi(x)` cannot generally act on `y=P_e x` merely because the action operators
have been named. State canonicalization or a proved `GL(d)`-equivariant policy
is a second missing edge. Therefore T47 does not establish the claimed transfer.

That is a potential abstraction/transfer edge:

- a name-conditioned learner must relearn the renamed dictionary or memorize
  all renamings;
- an effect-signature learner canonicalizes the dictionary from probes and
  reuses one shared operator library; and
- noncommutative probes can sometimes recover order/provenance from delayed
  endpoints.

No neural advantage follows yet. A Transformer, RNN/SSM, graph network, or
meta-RL system can learn or emulate the same invariant. The eventual comparison
must include explicit system identification and a generic learner trained with
the same permutation augmentation, probes, state bytes, and interaction cost.

## 8. Prior-art boundary

Similarity invariants, system identification, realization theory, Koopman
operators, PSRs, bisimulation, group representation learning, and
noncommutative operator algebra are established fields. T47 does not claim any
of its individual identities as new.

Two current controls are especially close:

- [Holonomy Grid Codes for Generalisation Under Directed Actions](https://openreview.net/forum?id=RJBvW7ZdhG)
  already develops gauge-invariant transfer and noncommutative directed-action
  operators with exact block structure; and
- [MetaKoopman](https://arxiv.org/abs/2607.26345) meta-learns a prior over
  Koopman operators and performs closed-form Bayesian adaptation under dynamics
  shift.

These substantially narrow any novelty claim based on invariant action
operators or closed-form operator update alone.

The strongest matched control is explicit system identification or a
permutation/equivariance-augmented meta-learner receiving the same probes. It
can implement the same recovery and traces, potentially with fewer diagnostic
probes than full `d^2` operator estimation. T47 supplies no acquisition-cost or
sample-complexity separation from that control.

## 9. Kill conditions before code

Close this realization without a run if any of the following holds:

1. realistic legal probes cannot excite a full or declared sufficient state
   subspace;
2. canonical action signatures collide or their separation vanishes with
   system size;
3. estimating the signature costs as many interactions as relearning the task;
4. action effects are nonstationary across episodes rather than conjugate;
5. partial observations destroy operator identifiability;
6. a permutation-augmented generic meta-learner has the same acquisition law;
7. the natural interface provides no consequence anchor connecting action
   surfaces to the operator basis; or
8. the only gains occur in a synthetic family that directly manufactures the
   similarity law.

## 10. Decision and sole next paper obligation

T47 is closed as an intervention-kernel or transfer candidate. Retain only the
conditional recovery, trace perturbation, complete-automorphism, and legal
commutator-margin identities.

Before any theorem-world implementation, the candidate must supply exactly one
extension:

> an observable controlled-Hankel action signature defined solely through
> finite future-test probabilities obtainable from legal endogenous histories,
> with a finite-sample identification bound and a minimax interaction-cost
> comparison to spectral system identification and permutation-augmented
> meta-RL.

This replaces privileged latent-state probes with raw-observable PSR/OOM
quantities. If it collapses to the matched spectral/system-identification
control without an acquisition advantage, close the whole operator-signature
lane. No CPU, model, local GPU, or rented GPU run is admitted by T47.
