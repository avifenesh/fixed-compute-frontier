# T81 intervention-operator algebra — learn mechanisms through their action, not their names

Date: 2026-08-02  
Status: **BROAD METHOD AND CODED MATRIX EXECUTION CLOSED; STAR-CODE THEOREM RETAINED; NO GPU**

## 0. Research question

T65 leaves one central problem unresolved: raw observations do not arrive with
the correct variables, factor boundaries, or bindings. A neural grounder that
simply reconstructs pixels may preserve nuisance and invent a representation
whose coordinates are convenient for prediction but unstable across worlds.

T81 asks whether a learner can instead recover a reusable **algebra of
interventions**. The object to preserve is not the name or coordinate of a
latent variable, but how controlled changes compose, commute, and alter future
decisions.

The hoped-for edge is broad: if a small set of learned generators explains many
long action sequences, the same representation could improve long-horizon
prediction, causal experiment selection, cross-surface transfer, and local
adaptation. The user's magnitude gate remains mandatory: less than a 20%
improvement on a preregistered major end-to-end capability is not a success.

## 1. Smooth controlled setting

Let the relevant world state lie on a smooth manifold `Z`. For each locally
continuous intervention `i`, let `V_i` be its vector field and

\[
\Phi_i^t=\exp(tV_i)
\]

its local flow. The learner observes `x=g(z)` through a diffeomorphic change of
coordinates `g:Z->X`. The observable response field is the pushforward

\[
W_i(g(z))=Dg_z V_i(z)=g_*V_i.
\]

### Lemma T81.1 — coordinate naturality of local order sensitivity

For smooth vector fields `V_i,V_j` and a diffeomorphism `g`,

\[
[g_*V_i,g_*V_j]=g_*[V_i,V_j].
\]

Therefore zero versus nonzero commutator, involutivity of a distribution, and
the Lie-algebra relations among controlled mechanisms are preserved by a
smooth invertible observation reparameterization.

A zero bracket means the specified flows are locally order-independent under
the usual domain conditions. It does **not** certify statistical, causal,
physical, or parameter independence. A nonzero bracket is a signature of local
order sensitivity, not a unique semantic explanation for that sensitivity.

This is the standard naturality of the Lie bracket. It is not an identifiability
theorem for human concepts. It applies only where the observation map is a
valid chart and the fields can be estimated. Non-injective observations,
unobserved state, aliasing, discontinuities, unknown action mixtures, and noise
can make distinct latent fields induce the same visible response.

For an explicit failure, let the latent state be `(x,z)`, observe only
`h(x,z)=x`, and take `X=partial_z`, `Y=z partial_x`. Latent states with the same
visible `x` but different `z` induce different visible effects for `Y`, so no
single pushed-forward field on observation space exists, while
`[X,Y]=partial_x`. Naturality becomes useful only after an injective or
action-projectable state representation has already been identified. It does
not solve raw grounding by itself.

### Lemma T81.2 — order swaps expose the bracket

For sufficiently small `delta`, comparing two controlled paths gives

\[
\Phi_j^{-\delta}\Phi_i^{-\delta}
\Phi_j^{\delta}\Phi_i^{\delta}(z)
=z+\delta^2[V_i,V_j](z)+O(\delta^3),
\]

up to the sign convention for the bracket and composition order. Thus a short
four-action loop estimates their local order sensitivity without needing a
named latent variable for that observable effect.

This is a paid active-information channel. A passive trajectory does not in
general contain the counterfactual order swap, inverse interventions may be
illegal, and finite differences can be dominated by observation noise or
integration error. When inverses are unavailable, paired forward sequences
estimate the same second-order difference only under stronger reset/coupling
conditions.

## 2. New residual: coded commutator tomography

The direct prior art already occupies Lie-structured world models and
bracket-based causal screening. A more specific untested operator may remain:
use the bilinearity of the bracket to screen many candidate pairwise
order-sensitive relations in one controlled experiment.

Assume locally control-affine dynamics without an unmodeled drift,

\[
\dot z=\sum_{i=1}^{k}u_iV_i(z),
\]

and an interface that can apply signed simultaneous control mixtures. Choose
two coefficient vectors `a,b` and define

\[
A=\sum_i a_iV_i,\qquad B=\sum_i b_iV_i.
\]

Bilinearity and antisymmetry give

\[
[A,B]
=\sum_{i<j}(a_i b_j-a_j b_i)[V_i,V_j].
\]

At a fixed reset state and visible coordinate, one small order-swap experiment
therefore gives a linear sketch of all `N=k(k-1)/2` pairwise bracket vectors.
Write `q_ell in R^N` for one visible coordinate of those bracket vectors and

\[
M_{r,(i,j)}=a_{ri}b_{rj}-a_{rj}b_{ri}.
\]

After dividing out the squared step size, the idealized observation is

\[
y_{r,ell}=M_rq_ell+e_{r,ell},
\]

where `e` contains measurement noise and finite-step truncation. If the union
of interacting pairs is `s`-sparse and the random wedge-measurement matrix `M`
satisfies an appropriate restricted eigenvalue or RIP condition, standard
sparse recovery gives stable reconstruction using on the order of

\[
R=O(s\log(N/s))
\]

coded experiments rather than `N` separate pair tests. This complexity is
conditional on a measurement theorem for the chosen `a,b` distribution; the
dependent rank-one wedge rows are not silently treated as i.i.d. Gaussian
entries. For Gaussian `a,b`, `a^TQb` is isotropic over a fixed skew-symmetric
interaction matrix `Q`, which is encouraging but not by itself a uniform RIP
proof.

More precisely, place `q_ij` in the upper triangle of a skew-symmetric `Q`.
Then

\[
M_rq=a_r^TQb_r.
\]

For independent zero-mean unit-variance coordinates in `a_r,b_r`, direct
expansion gives

\[
\mathbb E[M_r^TM_r]=2I_N,
\qquad
\mathbb E[(a_r^TQb_r)^2]=\|Q\|_F^2=2\|q\|_2^2.
\]

The row entries are dependent products, so this exact isotropy calculation is
not yet the required uniform sparse-recovery theorem.

For example, `k=32` gives `N=496`. If only `s=32` interaction pairs matter,
the information-scale term `s log2(N/s)` is about `127`, roughly four times
smaller than exhaustive pair testing before constants, repeats, and noise. The
numerical ratio is a prediction, not a result.

### What this does and does not buy

- It could make **pairwise order-sensitivity screening** a substantially
  cheaper acquisition phase. That is not itself a grounded-mechanism or
  intelligence result. It counts toward the user's threshold only if the
  frozen complete discovery or control phase improves by at least 20% and
  downstream behavior is not harmed.
- It does not recover the individual vector fields, causal direction, semantic
  variable names, or a planner by itself.
- The action interface is stronger than one-at-a-time interventions. Every
  control receives the same mixture interface and the physical cost of a
  multi-control pulse is charged.
- Drift adds `[V_0,V_i]` nuisance terms unless it is canceled, included as a
  generator, or estimated. Signed inverse pulses, exact resets, and arbitrary
  simultaneous mixtures may be impossible or unsafe.
- The `delta^2` signal shrinks faster than first-order measurement noise as
  `delta` decreases, while truncation grows as `delta` increases. The optimal
  finite step and replicate count are central, not tuning details.
- Sparse brackets are an assumption. Dense interactions remove the claimed
  experiment advantage, and a poor observation chart can make visible support
  dense even when latent mechanisms are sparse.

### Control-energy boundary

Let `m=a wedge b` denote the wedge measurement row. Its squared Euclidean norm
is

\[
\|m\|_2^2
=\|a\|_2^2\|b\|_2^2-(a^Tb)^2
\le \|a\|_2^2\|b\|_2^2.
\]

If both mixture controls have fixed total norm at most one, every coded row has
norm at most one. It cannot simultaneously deliver unit-amplitude measurements
of all `N` pairs. Dense isotropic sensing spreads its signal over the ambient
interaction space, so additive observation noise can erase the noise-free
compressed-sensing advantage. Conversely, choosing order-one coefficients on
all `k` controls makes `||a||` and `||b||` grow like `sqrt(k)` and spends
order-`k` simultaneous actuation per pulse.

For independent Gaussian endpoint noise of variance `sigma^2`, the idealized
linear sketch has Fisher information

\[
\mathcal I(q)=\sigma^{-2}M^TM,
\qquad
\operatorname{tr}\mathcal I(q)
=\sigma^{-2}\sum_{r=1}^{R}\|M_r\|_2^2
\le R/\sigma^2
\]

under the unit-control-norm constraint. Coded interventions do not manufacture
total information; they reallocate a fixed information budget using the sparse
support assumption. Uniform dense estimation therefore cannot receive the
claimed compression, while sparse support recovery still requires a declared
minimum signal and finite-noise theorem.

The correct comparison reports at least two separate resources:

1. number of environment resets or experimental batches; and
2. integrated control energy, number of simultaneously perturbed channels, or
   the domain's actual intervention cost.

Coded tomography can dominate when a batch/reset is expensive and multiplexed
control is physically cheap. It is not a universal information gain at fixed
control energy and fixed additive noise. The 4x idealized experiment-count
example is inadmissible unless its complete signal-to-noise and actuation
ledger remains favorable.

### Proposition T81.3 — constructive star-coded recovery

The fully dense wedge ensemble above needs a specialized recovery theorem. A
less symmetric construction reduces each block to ordinary compressed sensing.

Fix one generator `V_p` and choose a partner mixture

\[
B_r=\sum_{j>p} b_{rj}V_j,
\qquad b_{rj}\in\{-1,+1\}.
\]

Then the exact bracket identity is

\[
[V_p,B_r]=\sum_{j>p}b_{rj}[V_p,V_j].
\]

At a fixed reset state, write `n_p=k-p`, let visible state width be `m`, and
define

\[
S_p\in\{-1,+1\}^{R_p\times n_p},\qquad
Q_p\in\mathbb R^{n_p\times m}.
\]

Row `j-p` of `Q_p` is `[V_p,V_j](x_0)^T`. If at most `d_p` rows are
nonzero, `R_p` independent sign-coded loops give

\[
Y_p=S_pQ_p+E_p,
\qquad Y_p,E_p\in\mathbb R^{R_p\times m}.
\]

Fix a target RIP constant `0<rho<sqrt(2)-1`, an overall failure probability
`eta in (0,1)`, and let `A_p=S_p/sqrt(R_p)`. The standard subgaussian RIP
theorem gives a constant `C_rho` such that `A_p` has `2d_p`-RIP constant at
most `rho`, simultaneously over all anchors with probability at least
`1-eta`, when for every tested `p` with `1<=d_p<=n_p`,

\[
R_p\ge C_\rho\left[
d_p\log\frac{e n_p}{d_p}
+\log\frac{2(k-1)}{\eta}
\right],
\]

using a union bound with per-anchor failure `eta/(k-1)`. Normalize
`Z_p=Y_p/sqrt(R_p)=A_pQ_p+F_p`, where `F_p=E_p/sqrt(R_p)`. The row-group
decoder

\[
\widehat Q_p=\arg\min_X\|X\|_{2,1}
\quad\text{subject to}\quad
\|A_pX-Z_p\|_F\le\epsilon_p
\]

obeys the standard block-sparse stability form

\[
\|\widehat Q_p-Q_p\|_F
\le C_0\frac{\sigma_{d_p}(Q_p)_{2,1}}{\sqrt{d_p}}
+C_1\epsilon_p.
\]

It exactly recovers noiseless exactly row-sparse `Q_p`. Exact support recovery
under noise additionally requires a declared beta-min separation between every
nonzero row norm and the recovery-error floor. Coordinatewise basis pursuit is
also valid; group decoding can exploit shared support but the RIP theorem alone
does not reduce its measurement order.

If the true degrees `d_p` are supplied by an oracle, the total identity-cost
loop count is bounded at the scaling level by

\[
R_{total}
=O\left(
\sum_{p:d_p>0}d_p\log\frac{e n_p}{d_p}
+|P_{tested}|\log\frac{k}{\eta}
\right),
\]

instead of testing every one of `N=k(k-1)/2` pairs. This is a direct
application of the standard subgaussian compressed-sensing theorem after the
bracket identity; it does not prove the same saving for the fully dense wedge
design.

This oracle allocation is not implementable from unknown realized degrees. An
honest non-oracle design must either declare known upper bounds
`d_p<=bar_d_p`, prove and charge an adaptive doubling/validation rule, or use a
worst-case bound `d` for every star, costing

\[
O\left(kd\log\frac{ek}{d}+k\log\frac{k}{\eta}\right).
\]

Every star is capped by its `n_p` direct pair tests when the compressed bound
is larger. A sparsity-adaptive decoder does not by itself choose the number of
physical measurements. More importantly, each sign-coded loop can actuate
`O(k)` partner channels.
The theorem is about **experimental-batch identity cost**. It makes no claim of
lower component-count or energy cost, and its stable-noise constant must be
translated back through the `delta^2` commutator signal before any physical
comparison.

The independent addendum admits one **matrix-only measurement falsifier**, not
a learned world model: verify conditioning and noise scaling of the star and
wedge designs, compare exact support recovery against exhaustive pairs and the
strongest generic sparse experimental-design control, and close the seam if it
fails to deliver at least `2x` fewer cost-equivalent loops in its declared
identity-cost regime or any claimed gain falls below 20%.

## 3. Conditional compositional advantage

The Baker--Campbell--Hausdorff expansion gives locally

\[
\log(\exp(A)\exp(B))
=A+B+\tfrac12[A,B]
+\tfrac1{12}[A,[A,B]]+\tfrac1{12}[B,[B,A]]+\cdots.
\]

Suppose the intervention generators span an `m`-dimensional Lie algebra with
structure constants

\[
[V_i,V_j]=\sum_{\ell=1}^{m}c_{ij}^{\ell}V_\ell.
\]

If the algebra is nilpotent of step `s`, BCH terminates after finitely many
nested brackets. The local effect of arbitrarily many sufficiently small action
compositions can then be represented using the `m` generators, their structure
constants, and the action program rather than a separate learned transition
for every one of `k^L` length-`L` strings.

This is a conditional representation theorem, not free execution. The action
program still carries `L log2(k)` bits and exact simulation still processes its
content. For a general non-nilpotent algebra, BCH is infinite and truncation
error grows with step size, commutator norms, and horizon. State-dependent
fields require evaluating generators along the path. Discrete, irreversible,
contact-rich, or topology-changing events are not captured by a local Lie
group without a separate hybrid mechanism.

## 4. Conditional localization property

If the observation chart `g` and every generator except `V_j` remain fixed,
then a change to `V_j` leaves all pairwise brackets `[V_a,V_b]` with
`a,b != j` unchanged. A library with independently versioned generators can in
principle update `V_j` and the relations incident to it rather than relearn a
global trajectory model.

This is not a causal repair guarantee. A changed observation map pushes forward
every visible generator; shared encoder parameters can entangle all fields;
finite data can attribute a residual to the wrong generator; and changing one
field can alter reachable state distributions on which other fields are
estimated. Protected behavior must be measured end to end.

## 5. Rejected broader architecture wrapper

The proposed broad system would contain:

1. a perceptual chart encoder and decoder with explicit uncertainty;
2. a bank of action-conditioned local vector fields;
3. a numerical flow integrator;
4. a coded commutator experiment selector plus a commutator consistency loss;
5. a sparse closure loss that fits structure constants or introduces a new
   generator only when existing brackets do not close;
6. an operator library that versions generators, relations, provenance, and
   confidence; and
7. a planner that composes generators while charging integration and search.

This list does not define a new grounding architecture. It assumes an
action-compatible chart, stable generator identities, finite closure, reward
grounding, and a planner—the difficult interfaces it was meant to learn. A
generic controlled world model, Lie-action model, Koopman learner, or causal
representation learner can instantiate the same roles. The independent audit
therefore closes the broad wrapper; only the coded screening operator in
Section 2 remains under analysis.

## 6. What a real gain would be

Reopening the broad architecture would require one learned operator interface
to substantially improve all of the following relative to the strongest
matched control:

- prediction and control at at least `10x` the training composition horizon;
- transfer across independently generated observation charts or visual skins;
- active identification of consequential local order sensitivities;
- adaptation after one hidden generator changes while preserving queries that
  do not depend on it; and
- performance on at least one discrete/hybrid family through a declared
  extension rather than silently excluding real discontinuities.

No present theorem predicts these gains. If a later mechanism does, the frozen
primary statistic is normalized lifetime regret averaged over these
major phases, with family-wise floors. Success requires at least 20% relative
reduction, a reported absolute change and fraction of available headroom, a
simultaneous confidence interval excluding 20%, and no protected-family loss
greater than one absolute point. A 10% to less than 20% result is provisional;
a single-digit result closes the candidate as the main research outcome.

## 7. Required controls and cost ledger

Controls must include:

1. a parameter- and data-matched recurrent or Transformer world model;
2. an action-conditioned neural ODE/CDE without bracket closure;
3. a Lie-action or equivariant world model;
4. a Koopman/operator model;
5. an object-centric or causal-factor world model;
6. an oracle-coordinate operator model; and
7. direct trajectory retrieval or full-context prediction where feasible.

Charge action labels, environment resets, paired order swaps, inverse actions,
all trajectories, encoder/decoder parameters, field evaluations, integrator
steps, commutator Jacobian-vector products, planning/search, persistent
operator state, and tuning trials. A geometric prior does not make its
experimental interface free.

## 8. Direct prior-art boundary

The algebra is established mathematics and the application surface is already
substantially occupied.

- [Latent Confounded Causal Discovery via Lie Bracket
  Geometry](https://arxiv.org/abs/2606.19610) already estimates interventional
  response fields and uses bracket residuals for causal screening under latent
  confounding. Its own strongest conclusion is diagnostic/candidate-generation
  rather than general causal identification.
- This project's earlier
  [T58 audit](action-lie-algebra-variable-formation-t58-independent-audit.md)
  already closed action vector fields, commutator experiments, Frobenius-style
  variable formation, algebra transfer, and localized acquisition as a broad
  grounding method. T81's coded query is a refinement of that closed lane.
- [Inter-environmental World Modeling for Continuous and Compositional
  Dynamics](https://arxiv.org/abs/2503.09911)
  already learns continuous, compositional latent actions with Lie-group
  structure across environments. Its ICLR 2025 submission was withdrawn, so
  the arXiv record is the correct citation.
- [Lie Group Symmetry Discovery and Enforcement Using Vector
  Fields](https://arxiv.org/abs/2505.08219) directly occupies vector-field
  generators for learned continuous symmetries.
- [Log Neural Controlled Differential Equations](https://arxiv.org/abs/2402.18512)
  uses Lie-bracket information to improve learned controlled dynamics.
- [HOWM](https://proceedings.mlr.press/v162/zhao22b.html) gives an algebraic,
  homomorphic route to compositional generalization in object-oriented world
  models.
- Recent causal-representation work already proves recovery under paid
  intervention or multi-environment assumptions, while mechanistic-world-model
  work explicitly advocates reusable mechanisms over predictive mappings.

The broad conjunction of commutator closure, long-horizon composition, active
discovery, and local repair is not an admitted novelty claim. Its grounding
step fails under partial observation, and its remaining roles are occupied.

The narrower coded-tomography residual was not found in the targeted search as
of 2026-08-02. Group and stochastic intervention design, separating systems,
finite-sample causal testing, sparse causal discovery, and compressed sensing
are mature adjacent fields. Absence from a targeted search is not a novelty
proof; the audit must search this exact bilinear-wedge construction and compare
it with group-testing intervention designs.

## 9. Pre-run decision rule

No experiment is admitted merely to show that Lie structure can fit rotations
or other group-generated toy data. That is already known. The independent
audit rejected the broad method and retained only coded sparse screening as a
paper seam. Before even a matrix-only local falsifier, its addendum must answer
all of the following positively:

1. Is Proposition T81.3's star-coded RIP reduction correct, and does a
   finite-noise regime retain at least `2x` fewer cost-equivalent loops than
   exhaustive pairs and the strongest equal-interface group design?
2. Does that family include observation changes and at least one failure mode
   beyond smooth invertible dynamics?
3. Can the screen isolate the coded-query delta at matched reset, component,
   energy, noise, and decoding cost?
4. Does the predicted advantage exceed the 20% end-to-end threshold rather
   than only improving graph recovery or one-step error?
5. Is the remaining claim materially distinct from the direct prior art?

Passing a matrix-only falsifier would validate an experimental-design seam, not
raw grounding or a smarter model.

## 10. Matrix falsifier outcome and closure

The frozen matrix screen was run locally on CPU at `k=32` with 64 seeds,
degrees `{1,2,4}`, and noise `{0,0.05,0.10}`. The constructive star code did
recover sparse supports, but it failed against the strongest equal-interface
baseline: known-degree adaptive pair testing with early stopping.

Across the eight cells where both methods passed the exact-global-support
criterion, the median ratio

\[
\frac{\text{adaptive-pair identity loops}}
     {\text{star-code identity loops}}
=0.9053
\]

favored adaptive pair testing. Star coding reached `2x` in zero of nine cells,
lost by more than 20% in three numeric cells, and did not pass at all in the
remaining high-degree/high-noise cell. Its only numeric win was `2.7%`, an
explicitly rejected single-digit result. Unnormalized coding also spent far
more control energy; equal-energy coding did not recover a dominance result.

The [decision](coded-bracket-screen-t81-matrix-v1-decision.md) and
[independent result audit](coded-bracket-screen-t81-matrix-v1-independent-audit.md)
therefore close dense-wedge diagnostics, `k=64`, neural scaling, local GPU, and
rental GPU work. The audit found one denominator bug that had favored the
candidate and no defect capable of reversing the negative conclusion.

The retained result is narrow: Proposition T81.3 is a correct idealized
compressed-sensing reduction. It is not a practical acquisition advantage in
the frozen family and is not an intelligence operator. Reopening this lane
requires a new theorem that beats the strongest adaptive pair policy under the
same information, reset, energy, and noise budget; a different decoder or code
is not a repair.
