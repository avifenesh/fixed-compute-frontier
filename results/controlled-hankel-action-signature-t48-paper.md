# T48 controlled-Hankel action signature — observable correction and lane closure

Date: 2026-08-01  
Status: **CLOSED ON PAPER: VALID OBSERVABLE SIGNATURE, NO EDGE OVER SPECTRAL PSR/SYSTEM IDENTIFICATION; ZERO RUNS**

## 0. Purpose

T47 used privileged full-state basis probes. T48 makes every quantity observable
by defining action meaning only through probabilities of legal controlled future
tests. This repairs the latent-coordinate flaw and simultaneously reveals that
the construction is classical predictive-state/spectral system identification.

The result is useful but negative: it supplies a correct strongest control, not
a new learner.

## 1. Controlled Hankel object

A finite test uses joint controlled outputs `z_t=(o_t,r_t)`, so actions with
identical observation dynamics but different reward laws do not collapse:

\[
\tau=(a_1,z_1,\ldots,a_k,z_k).
\]

For a positive-probability history `h`, define

\[
H(h,\tau)=
P(z_{1:k}\mid h,\operatorname{do}(a_{1:k})).
\]

Choose a frozen finite set of legal core histories `Q={h_1,...,h_p}` and future
tests `T={tau_1,...,tau_q}`. For immediate action `a` and joint output `z`, define
the observable slice

\[
H_{a,z}(i,j)=
P(z, z(\tau_j)\mid h_i,
\operatorname{do}(a,a(\tau_j))).
\]

Here `a(tau)` and `z(tau)` are the action and joint-output sequences in the
suffix test. Every entry is the success probability of a declared experiment;
no latent state basis, action operator matrix, or coordinate transform appears.

If the infinite controlled Hankel matrix has finite rank `r`, a rank-`r` core
supports a predictive-state/observable-operator realization. T48 assumes no
such rank until it is measured or declared as a world-family condition.

## 2. Action signature under observable relabeling

For a finite joint-output alphabet `Z`, define the output-label-invariant
multiset signature

\[
\sigma_{Q,T}(a)=
\operatorname{sort}_{lex}
\left\{\operatorname{sv}(H_{a,z}):z\in Z\right\},
\]

where `sv` is the sorted singular-value vector, padded with zeros to a fixed
length, and the outer sort is lexicographic over those vectors. Without the
outer quotient, relabeling outputs merely reorders concatenated blocks.

If two episodes differ only by permutations of the frozen history rows, test
columns, joint-output labels, and action surfaces, the corresponding slice is
left/right multiplied by permutation matrices. Its singular values are
unchanged. Thus an action can be aligned to a finite canonical library whenever

\[
\delta=\min_{u\ne v}
\|\sigma(u)-\sigma(v)\|_\infty>0.
\]

The separation is computed only after quotienting output-label permutations;
this extra quotient can create collisions that a label-ordered concatenation
would hide.

This is only a sufficient finite-core signature. Singular values discard joint
orientation between slices; two actions may have equal signatures but different
controlled future laws. Adding the complete slice or mixed action-test words
recovers information at correspondingly larger acquisition cost.

## 3. Finite-sample identification

Suppose each of the

\[
M=m|Z|pq
\]

action/outcome/history/test entries is estimated from `n` conditionally
independent rollouts of the exact declared experiment. Hoeffding and a union
bound give

\[
P\left(\max_{a,z,i,j}
|\widehat H_{a,z}(i,j)-H_{a,z}(i,j)|>\epsilon\right)
\le 2M e^{-2n\epsilon^2}.
\]

Therefore confidence `1-alpha` is obtained with

\[
n\ge\left\lceil{\ln(2M/\alpha)\over2\epsilon^2}\right\rceil.
\]

This assumes `Q,T` and their cross-episode correspondence were frozen
independently of these samples. Adaptive core selection requires sample
splitting or a uniform bound over the selection family. Enumerating all short
histories and tests instead can grow exponentially with horizon.

For every slice,

\[
\|\widehat H_{a,z}-H_{a,z}\|_2
\le\|\widehat H_{a,z}-H_{a,z}\|_F
\le\sqrt{pq}\epsilon.
\]

Weyl's inequality gives the same bound on each singular value. Let
`Q_e=sqrt(pq) epsilon`. Nearest-signature matching succeeds when

- `delta>2Q_e` if the canonical prototypes are exact; or
- `delta>4Q_e` if both the new and canonical signatures are independently
  estimated with worst-case error `Q_e`.

If canonical prototypes are also estimated, `M` and `alpha` must cover their
entries too. These are entrywise-rollout bounds, not an efficient estimator
claim. Low-rank
matrix methods, shared rollouts, martingale concentration, or adaptive designs
may improve constants or dependence on `p,q`; they are mandatory controls.

## 4. The hidden interaction bill

The nominal `M n` rollouts undercount acquisition unless the ledger includes:

1. cost and probability of reaching each history `h_i`;
2. resets, failed/rejected trajectories, and action side effects;
3. complete suffix-test horizon and all intermediate observations;
4. dependence between rollouts in one continuing environment;
5. nonstationarity while the signature is being estimated; and
6. risks or rewards sacrificed to perform diagnostic interventions; and
7. search for a rank-revealing core and correspondence of its histories/tests
   across renamed episodes.

If a history occurs passively with probability `rho_i`, rejection acquisition
alone costs `1/rho_i` trials in expectation. More explicitly, the nominal bill
is lower-bounded by a term of the form

\[
\sum_{a,z,i,j}n\,
{\operatorname{cost}(h_i,\tau_j)\over
 P(\text{legally reach }h_i)},
\]

before resets, failures, and core search. Rare distinguishing histories can
therefore erase an apparent transfer benefit.

## 5. Minimax information boundary

Consider two action types whose complete controlled laws differ only in one
legal Bernoulli test, with success probabilities separated by `g`. Binary
hypothesis testing requires

\[
n=\Omega\left({\log(1/\alpha)\over g^2}\right)
\]

successful samples of that distinguishing test to attain error `alpha` in the
ordinary bounded-away-from-zero/one regime. This follows from the Bernoulli KL
divergence and Le Cam/Bretagnolle-Huber testing bounds. It excludes attempts
needed to reach the conditioning history; near probability boundaries, KL need
not scale as `g^2`. A structured cross-task prior can reduce Bayes-average cost,
but that prior is then the charged distinguishing anchor.

No neural architecture, meta-learner, or signature can evade this information
cost without a prior anchor that already distinguishes the actions. T48's
uniform entrywise estimator instead pays for all `pq` entries; it is generally
not minimax optimal.

## 6. Exact collision with the strongest control

The matrices `H` and `H_{a,z}` are exactly the controlled Hankel objects used by
predictive-state representations and observable-operator/spectral system
identification. A spectral PSR control receiving the same legal trajectories
can:

1. estimate the same slices;
2. compute the same signature;
3. retain the full low-rank predictive state rather than discard orientation
   through singular values; and
4. when `Q,T` form a sufficient rank-`r` core with a valid normalized
   realization, update and plan in that state.

A full-slice control unconditionally contains the proposed signature; PSR
update/planning additionally needs the stated rank/core condition. A
permutation-augmented recurrent/meta-RL control can also approximate the same
map. T48 supplies no function-class, information, sample-complexity, state, or
served-compute separation from these controls.

The correction from latent operator traces to observable future-test spectra
therefore collapses the proposed novelty into its strongest baseline.

## 7. Decision

Close the operator-signature lane without CPU or GPU work.

Retain:

- controlled future equivalence as the semantic specification;
- legal excitation and action-outcome pairing as mandatory resources;
- automorphism/gauge ambiguity as the exact identifiability boundary;
- the finite-sample signature margin as a diagnostic; and
- the minimax cost of acquiring a genuinely distinguishing consequence.

Do not retain:

- trace or Hankel signatures as a new architecture;
- a claim of zero-shot action grounding without an observable anchor;
- a delayed-credit claim from two-action order discrimination; or
- a transfer claim that omits state/policy alignment and interaction cost.

Reopening requires a new observable object with a proved minimax acquisition or
physical-resource advantage over spectral PSR/system identification—not a new
invariant computed from the same controlled Hankel matrix.

## 8. What this teaches the intelligence program

The hard missing operation is not Bayesian normalization, state persistence,
or a clever invariant. It is **creating the right causal variables and legal
distinguishing experiments from raw, changing interfaces**. Once a finite
controlled test basis is supplied, classical algorithms already provide state,
update, and identification.

The next research branch must attack hypothesis/variable formation or adaptive
structure growth directly. It may not hand the learner a fixed latent index,
known intervention dictionary, full state basis, or frozen Hankel core and then
claim to have learned intelligence.
