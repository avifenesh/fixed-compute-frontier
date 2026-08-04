# T81 independent audit: intervention-operator algebra

Date: 2026-08-02

Verdict: **REVISE: RETAIN CODED-MIXTURE SPARSE SCREENING AS A DISTINCT PAPER SEAM; NO RUN YET; REJECT THE BROADER GROUNDING CLAIM**

The mathematical primitive is valid in a narrow smooth-control setting. Lie
brackets are natural under diffeomorphic changes of state coordinates, local
commutator loops expose bracket directions, and BCH describes short flow
composition. Those facts do not make brackets identifiable from arbitrary raw
observations, do not equate commutation with causal independence, and do not
make a learned finite generator table a grounded mechanism model. The proposed
direction's original learn/store/compose/localize wrapper also repeats this
project's T58 action-Lie-algebra lane, which was already closed without a run,
and now collides directly with a June/July 2026 paper that learns interventional
response fields and uses bracket residuals as causal-discovery diagnostics.

The refined coded-mixture proposal does add one technically distinct seam:
random simultaneous control mixtures turn a sparse table of pairwise brackets
into structured linear sketches. That can plausibly reduce ideal loop count
from \(\binom{k}{2}\) to sparse-recovery scale. It is not yet a proved physical
resource saving once actuation energy, drift, reset error, and endpoint noise
are matched, and it remains an interaction-screening method rather than raw
variable grounding. The seam deserves paper analysis, but no experiment is yet
earned.

## 1. Exact mathematical core and required conditions

The clean setting is a smooth control-affine system on a manifold \(M\):

\[
\dot x=f_0(x)+\sum_{i=1}^{k}u_i X_i(x),
\]

where \(X_i\) are sufficiently smooth controlled vector fields. If a change of
coordinates \(h:M\to N\) is a diffeomorphism, then the pushforward is natural:

\[
h_*[X_i,X_j]=[h_*X_i,h_*X_j].
\]

Thus exact zero/nonzero brackets, involutivity of a distribution, and the
abstract Lie-algebra isomorphism class are coordinate-independent under that
paired transformation. Euclidean bracket norms, margins, condition numbers,
time scales, and learned coefficient errors are not invariant under an
arbitrarily ill-conditioned diffeomorphism.

For one standard convention, a short commutator word satisfies locally

\[
\Phi_Y^{-t}\circ\Phi_X^{-t}\circ\Phi_Y^t\circ\Phi_X^t(x)
=x+t^2[X,Y](x)+O(t^3),
\]

with the sign reversed if the field or composition convention is reversed.
This requires local flows, sufficient derivatives on a common neighborhood,
small \(t\), and executable inverse flows. A uniform error statement requires
uniform derivative bounds and a domain that contains every intermediate point.

The BCH expression

\[
\log(\exp(tX)\exp(tY))
=t(X+Y)+\frac{t^2}{2}[X,Y]
+\frac{t^3}{12}\bigl([X,[X,Y]]+[Y,[Y,X]]\bigr)+\cdots
\]

is a local/formal composition tool. Convergence and truncation control require
a suitable finite-dimensional Lie group or a normed local setting plus size
and regularity bounds. Generic smooth vector fields form an infinite-dimensional
Lie algebra; a finite BCH table is not automatically an exact simulator.

These constructions do not transfer without modification to:

- irreversible, dissipative, hybrid, reset-free, or constrained actions;
- control-affine systems where reversing \(u_i\) does not reverse the drift
  \(f_0\);
- stochastic dynamics, where diffusion generators and their commutators need a
  stochastic-semigroup treatment rather than deterministic path reversal; or
- hard SCM interventions, which replace structural assignments and need not be
  smooth flows at all.

The proposed direction must state whether its “interventions” are actuator
flows, soft distributional regimes, or graph-surgery \(\operatorname{do}\)
operations. They are not interchangeable mathematical objects.

## 2. A bracket is not a mechanism-independence oracle

For complete vector fields, \([X,Y]=0\) is closely tied to local/global flow
commutation under the usual domain conditions. It means that these two state
transformations are locally order-independent. That is not equivalent to
physical, statistical, modular, or causal independence.

- Two mechanisms may share hidden state or parameters yet induce commuting
  flows on the measured state.
- Two independently actuated mechanisms may fail to commute because they act on
  a shared configuration manifold; rotations are the standard example.
- A nonzero bracket can arise from state-dependent action effects, drift,
  saturation, observation projection, transport choice, target mismatch, or
  estimation error. It does not uniquely identify a causal interaction.
- Distinct hard interventions on separate SCM assignments commute as graph
  surgery even when the affected variables are causally connected. Learned
  transports between the resulting regime distributions can nevertheless have
  nonzero brackets.

The correct claim is:

> A bracket is a coordinate-natural signature of local order sensitivity for
> specified vector fields, not a certificate of mechanism dependence or a
> semantic label for the cause of that order sensitivity.

Pairwise commutation is also stronger than Frobenius closure. A distribution
\(D=\operatorname{span}\{X_i\}\) is involutive when brackets remain in \(D\);
they need not vanish. Frobenius then gives local integral leaves under constant
rank, not independent semantic variables. If

\[
[X_i,X_j]=\sum_k c_{ij}^{k}X_k
\]

uses constant scalars on an open set, the fields realize a finite-dimensional
Lie algebra. State-dependent \(c_{ij}^{k}(x)\) are structure functions of a
frame/distribution, not a finite constant relation table of the proposed kind.

## 3. Raw observations break the claimed invariance-to-identification step

Let the raw observation be \(y=h(x)\). A unique observed field \(h_*X\) exists
automatically when \(h\) is a local diffeomorphism, or on an embedded image with
an appropriate restriction. For a noninjective sensor it exists only if \(X\)
is projectable: latent states in the same observation fiber must induce the
same observed velocity.

A two-dimensional witness makes the obstruction explicit. Let
\(h(x,z)=x\), \(X=\partial_z\), and \(Y=z\partial_x\). States with the same
raw observation \(x\) but different \(z\) give different observed effects for
\(Y\), so no single vector field on observation space represents it. Meanwhile
\([X,Y]=\partial_x\). The raw sensor has hidden the state needed even to define
the proposed generators consistently.

Learning a latent encoder does not remove this issue. Naturality applies only
after an injective/action-compatible latent chart has been identified. A lossy
encoder can destroy bracket directions; a state-dependent change of generator
basis introduces derivative terms; and an arbitrary latent representation can
move all apparent localization boundaries.

Distributional regime transports have another non-identifiability. Many flows
can push one distribution to another; composing with a measure-preserving map
or choosing a different off-support extension leaves the endpoint regime laws
unchanged but can alter the local vector fields and their brackets. A selected
normalizing flow or minimum-energy path is a modeling convention until a causal
identification assumption privileges it.

Therefore raw-observation identification requires, at minimum, a declared set
of conditions such as:

1. known intervention/action semantics and target identity;
2. randomized or otherwise unconfounded action assignment;
3. persistent excitation, overlap, reset/initial-state coverage, and calibrated
   action timing;
4. full-state observability, a sufficient history state, or a proved
   projectability/identifiability condition for the learned encoder;
5. a noise and hidden-confounding model under which the response fields are
   unique up to a stated equivalence; and
6. a finite, separated field family with uniform approximation and derivative
   control.

Without these, T81 is fitting a convenient response geometry, not recovering
grounded intervention mechanisms.

## 4. Generators, relations, composition, and localized revision

“Store generators and relations” assumes the hard part.

First, a finite list of starting fields usually generates infinitely many
independent nested brackets. Finite closure must be assumed or proved over an
open set, with a rank margin and a stopping/error theorem. Finite noisy samples
cannot prove global equality of vector fields or exclude a new direction at
greater bracket depth.

Second, a generator table is gauge-dependent. Under a constant basis change
\(E'_a=A_a{}^iE_i\), structure constants transform as

\[
c'_{ab}{}^c
=A_a{}^iA_b{}^j(A^{-1})_k{}^c c_{ij}{}^k.
\]

The individual generator names and coefficients are not canonical. Unknown
action permutations, scaling, time calibration, stabilizers, and representation
gauge must be aligned across environments and across updates. A local Lie
algebra also omits the global group, its action realization, orbit/stabilizer
geometry, discrete components, boundaries, passive state, and topology.

Third, BCH composition predicts the controlled state flow only within its local
validity regime. It does not predict rewards, observations, uncertainty,
constraints, or policy value. An abstractly identical algebra can act on two
state spaces in inequivalent ways.

Fourth, “localize a change to one generator” is not guaranteed. Changing
\(E_i\) changes every relation \([E_i,E_j]\); a rotating gauge can make one
physical change look dense; and a sensor, action calibration, confounder, or
reward change can mimic a generator change. Local revision needs stable modular
parameters and an intervention design proving which module changed. It is a
causal-identification result, not a consequence of storing a Lie table.

## 5. Coded-mixture bracket sketches

Let the coefficients be constant and let the action interface realize signed
simultaneous mixtures

\[
A=\sum_i a_iV_i,\qquad B=\sum_i b_iV_i.
\]

Bilinearity and antisymmetry give the exact identity

\[
[A,B]
=\sum_{i<j}(a_ib_j-a_jb_i)[V_i,V_j].
\]

At one reset state \(x_0\), index the \(N=k(k-1)/2\) pairs by \(e\), define
the unknown vector columns \(C_e=[V_i,V_j](x_0)\), and set
\(w_e=a_ib_j-a_jb_i\). An ideal commutator loop returns the multi-output
linear sketch

\[
q=Cw.
\]

If only \(s\) columns of \(C\) are nonzero, this is a group-sparse recovery
problem. Multiple state coordinates are multiple measurement vectors with
shared support; in the worst case the bracket vectors are collinear, so the
scalar sparse-recovery rate remains the safe comparison.

### What is exact and genuinely distinct

For independent unnormalized Gaussian \(a,b\sim\mathcal N(0,I_k)\), the wedge
row is isotropic up to scale:

\[
\mathbb E[w_e]=0,\qquad \mathbb E[w_ew_f]=2\,\mathbf 1\{e=f\}.
\]

Every scalar measurement of a skew coefficient matrix is the bilinear chaos
\(a^\top Kb\). It has nonzero small-ball probability for every nonzero \(K\).
These facts make an \(O(s\log(N/s))\)-type recovery theorem plausible.
However, the coordinates of a wedge row are dependent and every sensing matrix
row is a decomposable bivector, equivalently a rank-at-most-two skew matrix.
The ordinary iid-subgaussian compressed-sensing theorem cannot simply be cited.
A uniform restricted-eigenvalue/nullspace or small-ball proof for this wedge
ensemble, including noise and vector-valued group sparsity, is still required.

This coded design was not the subject of T58's pairwise-loop audit. It is a
real residual algorithmic seam: **compress sparse pairwise bracket discovery by
legal mixed-control queries**. Its possible novelty is this wedge-constrained
measurement and physical realization, not Lie brackets, sparse recovery, pooled
interventions, or group testing separately.

### Realizability conditions

The linear sketch is valid only if:

1. the system is control-affine over the pulse range and \(a_i,b_i\) are
   constant, so mixture coefficients introduce no derivative terms;
2. the actuator exposes signed simultaneous mixtures without clipping,
   cross-channel nonlinearities, delays, or a hidden mixer;
3. the loop can execute \(\pm A,\pm B\), return to the same local chart, and be
   repeated from a sufficiently accurate common reset state;
4. drift is absent, cancelled, or separately identified; otherwise commutators
   of \(f_0+A\) and \(f_0+B\) contain \([f_0,B]-[f_0,A]\) as well as \([A,B]\);
5. raw observations define the bracket vector through a diffeomorphic or
   projectable interface; and
6. pair sparsity is defined in a privileged physical control basis. An action-
   basis rotation can turn a sparse pair table into a dense one.

The observed endpoint has the form

\[
\frac{\Delta x}{t^2}=Cw+O(t)+\frac{\eta_{\mathrm{endpoint}}+\eta_{\mathrm{reset}}}{t^2},
\]

before drift, integration, and process-noise terms. Small \(t\) reduces BCH
bias but amplifies endpoint/reset noise. Sparse recovery also needs a minimum
nonzero bracket norm and must prevent cancellations among vector brackets.

### Loop count is not complete cost

Unnormalized Gaussian or Rademacher codes excite \(\Theta(k)\) total squared
control amplitude per pulse. They keep each wedge coefficient order one but
make a coded loop more energetic than a pairwise loop. If \(a,b\) are normalized
to unit \(\ell_2\) energy, a typical coefficient has scale \(1/k\) and variance
\(\Theta(1/k^2)=\Theta(1/N)\). Recovering an individual bracket then amplifies
fixed endpoint noise by order \(k\); repetitions can erase the nominal sketch-
count saving.

The relevant comparisons are therefore distinct:

- **identity loop cost:** every mixed or pairwise loop costs one, regardless of
  the number and amplitude of actuators;
- **linear/component cost:** charge total actuator-time or number of active
  channels; and
- **energy/noise cost:** fix \(\int\|u(t)\|_2^2dt\), sensor noise, reset error,
  and safety envelope.

An \(O(s\log(N/s))\) loop theorem under the first model does not imply the same
improvement under the latter two. The project must report all three rather than
selecting the favorable accounting after observing results.

## 6. It does not yet attack the grounding bottleneck

T81 does use action-conditioned evidence, which is more grounded than passive
sequence prediction. But the current proposal starts after the crucial
grounding choices: it presumes a controlled intervention family, a smooth state
or latent manifold, identifiable vector fields, stable generator identity, and
a relation family. Brackets then diagnose order sensitivity and help organize
composition.

The passive-factor counterexample remains decisive. Let the true state be
\(M=G\times W\), let every learned generator act only on \(G\), and let the
arbitrary hidden factor \(W\) determine reward or future observations. Every
generator, bracket, relation, and commutator experiment on \(G\) can match
across two worlds while their optimal answers or actions are opposite. The
algebra has not formed the missing variable, grounded the reward-relevant state,
or closed the knowledge-to-policy loop.

To attack grounding rather than diagnostics, a positive theorem would need to
recover a minimal controlled-predictive quotient from raw histories, prove that
the quotient and its intervention representation are identifiable up to an
explicit action-compatible gauge, and show that generator-local revision
reduces end-to-end learning regret or sample complexity against a matched
generic world model. T81 supplies none of those steps.

## 7. Direct literature and project-history collision

The novelty lane is already occupied from several directions:

- This repository's [T58 independent audit](action-lie-algebra-variable-formation-t58-independent-audit.md)
  evaluated the same core proposal: recover action vector fields through
  commutator experiments, use brackets/Frobenius for variable formation and
  structural decomposition, transfer the algebra, and localize acquisition. It
  proved the partial-observation, finite-closure, global-action, reward, and
  passive-factor objections above and concluded **diagnostic retained; no run**.
  The coded-mixture sketch is a new intervention-design refinement, but it does
  not repair the audit's raw-observation, reward, or grounding objections.
- [Latent-Confounded Causal Discovery via Lie-Bracket Geometry](https://arxiv.org/abs/2606.19610),
  currently arXiv v2 dated 25 July 2026, estimates response fields from
  observational/interventional regimes, computes Lie/Frobenius residuals,
  amortizes the fields, and uses the geometry to screen causal candidates. Its
  main experiments assume known single-node intervention targets. The paper
  explicitly concludes that brackets are nonspecific diagnostics: target
  mismatch, domain shift, weak overlap, omitted coordinates, transport choice,
  and estimation error can all cause nonclosure. Final identification is
  delegated to a downstream learner. This directly occupies T81's most novel-
  sounding diagnostic and documents its identification boundary.
- [Inter-environmental World Modeling for Continuous and Compositional
  Dynamics](https://arxiv.org/abs/2503.09911) introduces WLA, which learns
  latent continuous Lie actions from video with an object-centric autoencoder,
  composes them through exponentiated generators, and adapts a controller to
  new action sets. It assumes an abelian transition group and user-chosen latent
  structure, so it does not occupy T81's noncommutative bracket diagnostic
  exactly; it does occupy the latent-generator, flow-composition, and raw-video
  world-model design. Venue correction: its ICLR 2025 submission was withdrawn;
  the current citable result is the arXiv paper, not an accepted ICLR 2025 paper.
- [Learning Group Actions on Latent Representations](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e63309e532688c722177f81e99f94f32-Abstract-Conference.html)
  already learns group actions inside autoencoder latents. [Explicit Discovery
  of Nonlinear Symmetries from Dynamic Data](https://proceedings.mlr.press/v267/hu25o.html)
  explicitly recovers nonlinear infinitesimal generator bases from dynamics.
- [Efficient Dynamics Modeling in Interactive Environments with Koopman
  Theory](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f074a994e062146561db9cdc63999efa-Abstract-Conference.html)
  learns action-conditioned latent operators for long-range prediction and RL.
  Koopman and neural-ODE/control literature broadly occupy learned generators,
  latent flows, operator composition, and planning from controlled dynamics.
- Pooled experimental design is itself established. [Sparse Polynomial
  Learning and Graph Sketching](https://papers.neurips.cc/paper/5426-sparse-polynomial-learning-and-graph-sketching.pdf)
  recovers sparse edge/hyperedge structure from random aggregate queries;
  [compressed Perturb-seq](https://www.nature.com/articles/s41587-023-01964-9)
  uses compressed-sensing ideas for sparse single and combinatorial perturbation
  screens; and [Efficient Intervention Design for Causal Discovery with
  Latents](https://arxiv.org/abs/2005.11736) shows why intervention count and
  intervention size/linear cost are different objectives. These are not the
  same wedge-bracket measurement, but they occupy the high-level claim that
  sparse interactions can be discovered by coded group interventions.

The exact collision for the original generator/bracket diagnostic is T58
locally and the 2026 bracket-geometry paper externally. The searched primary
literature did not reveal the exact combination “legal control-mixture
commutator loops plus wedge-ensemble sparse recovery.” That narrow theorem may
be distinct. WLA, latent group-action learning, LieNLSD, Koopman, graph
sketching, and compressed perturbation design ensure that a contribution must
be claimed at that narrow level, not as a new operator algebra or grounding
architecture.

## 8. Complete resource bill

A fair ledger must include:

- real interventions, failed/safety-limited actions, environment resets,
  trajectory horizon, timing resolution, and coverage of initial states;
- paired observational/interventional regimes or action-labeled trajectories,
  including target metadata and overlap diagnostics;
- encoder/state-estimator, decoder, world model, and planner parameters and
  training, not only the relation table;
- \(k\) learned vector fields plus Jacobian-vector products for \(O(k^2)\)
  pairwise brackets; higher nested brackets require higher derivatives and a
  rapidly growing Hall/free-Lie word family;
- numerical ODE integration steps and tolerances for every flow/BCH rollout,
  truncation error, reset and inverse-action approximation, and stochastic
  uncertainty propagation;
- storage for field networks, replay/anchors, and change history; for a
  finite \(r\)-generator algebra, a dense structure tensor alone is
  \(O(r^3)\), before state-dependent coefficient functions; and
- gauge alignment, rank/eigenspace conditioning, relation re-estimation after
  every generator edit, reward learning, planning/search, and served latency.

For the coded variant also charge code amplitude and active-channel count,
mixture calibration, pulse energy, group-sparse decoding, repeated loops needed
to reach the same coefficient SNR, and the number of reset states needed to
avoid mistaking a bracket that vanishes at one state for a globally absent
interaction.

The bracket signal is second order in a small pulse duration, while higher
nested signals appear at still higher order. Noise, derivative estimation, and
irreversible control can make the intervention bill dominate. “Compact algebra”
does not imply a compact or cheap grounded learner.

## 9. Substantial gate and run disposition

The 20% rule applies to a broad preregistered end-to-end capability or learning
measure. It is not satisfied by bracket recovery, field MSE, graph-screen
compression, relation accuracy, rollout reconstruction, or a hand-built
noncommutative toy. Even a 20% gain in one of those narrow submetrics would not
cover a substantial part of the learning loop.

The decision bands should remain frozen: retain only at least 20% on that
broad measure, treat 10--20% as provisional and requiring an independent
replication, and reject a single-digit gain. A reduced intervention count can
be reported as mechanism evidence, but cannot silently replace the broad
measure.

There is no matched-control separation for the broad T81 architecture. A
generic recurrent world model,
neural ODE/control model, action-conditioned Koopman learner, causal
representation learner, or WLA-style latent action model can consume the same
trajectories and interventions. T81 gives no theorem predicting a 20% advantage
in active world discovery, cross-composition planning, continual retention, or
localized repair against those controls.

The coded screen does have a plausible **discovery-efficiency** path. In the
ideal identity-cost model, it beats exhaustive pairwise testing by at least 20%
whenever a proved constant \(C\) gives

\[
C s\log(N/s)\leq 0.8N.
\]

That is not yet the user's success metric. To count, a fixed intervention budget
must yield at least 20% better end-to-end world-model/graph recovery, planning,
or active-discovery regret across unseen families—not merely 20% fewer loops.
If claimed as acquisition efficiency, the charter's stronger \(2\times\)
interaction criterion should also be reported. The effect must survive equal
energy/noise and the strongest adaptive pairwise, random-group, direct-response,
and generic learned-world-model controls.

**No cheap local run is earned yet.** The direction first needs paper-only work:

1. define the raw observation, action/intervention, reset, noise, reward, and
   hidden-confounding interface;
2. state the unavoidable equivalence class and give two worlds with identical
   allowed data/bracket signatures but different target behavior;
3. add the weakest projectability/observability, reversibility, finite-closure,
   excitation, and separation assumptions that restore identification;
4. prove a finite-sample generator/relation recovery and BCH rollout bound;
5. prove generator-local change identification under a stable gauge; and
6. derive an end-to-end regret, acquisition, transfer, or repair advantage over
   the strongest matched recurrent/Koopman/causal-world-model control, with a
   plausible path to at least 20%.

For the coded seam specifically, the paper precheck must additionally prove or
falsify restricted recovery for the decomposable-wedge ensemble and compare
pairwise versus coded SNR under identity, linear, and energy cost. If those
calculations leave viable constants, one **matrix-only CPU falsifier** would be
appropriate: freeze \((k,s)\), code distribution and normalization, common
support/vector-bracket distribution, BCH/reset/noise model, decoder, and equal-
cost pairwise/group-query controls; then measure support recovery and coefficient
error. It should be killed unless it shows a large, replicated phase-transition
advantage, preferably at least \(2\times\) fewer cost-equivalent loops.

That future falsifier would test the compressed-screen lemma only. It would not
validate grounding or satisfy the 20% intelligence gate. Until the precheck is
written, preserve T58's architecture closure while retaining this one narrow
paper seam: **coded bracket screening may be a new experimental-design result;
Lie brackets remain a diagnostic, not T81's missing intelligence operator.**

## Addendum: audit of Proposition T81.3

Date: 2026-08-02

Verdict: **THE FIXED-ANCHOR REDUCTION IS CORRECT; REVISE THE THEOREM
STATEMENT; ADMIT ONLY THE MATRIX-LEVEL FALSIFIER.** Proposition T81.3 replaces
the unresolved dense-wedge sensing ensemble with an ordinary Rademacher design
inside each star. That is enough restricted-recovery structure to admit the
previously specified CPU falsifier. It does not establish a physical
intervention saving, and its displayed total-loop bound currently assumes
information about the unknown star degrees.

### Dimensions and exact reduction

For anchor \(p\in\{1,\ldots,k-1\}\), put \(n_p=k-p\) and let \(m\) be the
visible state dimension. With rows indexed by \(j=p+1,\ldots,k\), the correct
dimensions are

\[
S_p\in\{-1,+1\}^{R_p\times n_p},\quad
Q_p\in\mathbb R^{n_p\times m},\quad
Y_p,E_p\in\mathbb R^{R_p\times m}.
\]

Here row \(j-p\) of \(Q_p\) is the visible vector
\([V_p,V_j](x_0)^\top\). Then \(Y_p=S_pQ_p+E_p\) is dimensionally and
algebraically correct. Each unordered pair occurs exactly once because only
\(j>p\) is used, so \(s=\sum_p d_p\) is the number of nonzero oriented upper-
triangle pairs. Renaming the sign matrix from \(B_p\) to \(S_p\) would avoid
confusion with the partner vector field \(B_r\).

### Precise RIP and recovery claim

The probability parameter should not reuse the pulse duration \(\delta\). Fix a
target RIP constant \(0<\rho<\sqrt2-1\), let the overall failure probability
be \(\eta\in(0,1)\), and set \(A_p=S_p/\sqrt{R_p}\). The standard subgaussian
RIP theorem gives, for a constant \(C_\rho\),

\[
R_p\ge C_\rho\left[
d_p\log\frac{e n_p}{d_p}+\log\frac{2(k-1)}{\eta}
\right]
\]

for \(1\le d_p\le n_p\), with failure at most
\(\eta/(k-1)\). Equivalently one may expose the usual \(\rho^{-2}\)
dependence rather than absorbing a fixed recovery threshold into \(C_\rho\).
Merely saying that a \(2d_p\)-RIP holds is insufficient: its constant must be
below a stated basis-pursuit recovery threshold. This is the standard
Bernoulli/subgaussian result, not a new wedge-ensemble theorem; see Theorems
5.1--5.2 in [Rudelson and Vershynin's survey](https://public.websites.umich.edu/~rudelson/papers/rv-ICM2010.pdf).

Normalize the full observation equation:

\[
Z_p=Y_p/\sqrt{R_p}=A_pQ_p+F_p,
\qquad F_p=E_p/\sqrt{R_p}.
\]

For the row-group decoder

\[
\widehat Q_p=\arg\min_X\|X\|_{2,1}
\quad\text{subject to}\quad
\|A_pX-Z_p\|_F\le\epsilon_p,
\]

the vector RIP of \(A_p\) is also the row-block RIP of
\(A_p\otimes I_m\). Standard recovery therefore has the form

\[
\|\widehat Q_p-Q_p\|_F
\le C_0\frac{\sigma_{d_p}(Q_p)_{2,1}}{\sqrt{d_p}}
   +C_1\epsilon_p.
\]

It is exact for noiseless exactly row-sparse \(Q_p\). Coordinatewise basis
pursuit is also valid because every coordinate has support contained in the
same \(d_p\)-row set. No extra union bound over the \(m\) output coordinates is
needed: RIP is uniform over all sparse coefficient vectors. Group decoding can
use shared support, but the displayed RIP theorem alone does not promise a
better measurement order than coordinatewise recovery.

A union bound over the \(k-1\) anchors does not require independence and gives
overall success at least \(1-\eta\). Stable coefficient error is not exact
support recovery: the latter additionally needs a beta-min condition on every
nonzero row norm relative to \(C_1\epsilon_p\). The proposal's planned support-
recovery metric must state that separation.

### The loop bound is oracle-qualified

With the true \(d_p\) supplied in advance, summing the block bounds gives

\[
R_{\mathrm{total}}
=O\!\left(
\sum_{p:d_p>0}d_p\log\frac{e n_p}{d_p}
+|P_{\mathrm{tested}}|\log\frac{k}{\eta}
\right).
\]

The proposal's \(k\log(k/\eta)\) version is an honest upper bound if every
anchor is probed and a convention is supplied for \(d_p=0\). It is not an
implementable sparsity-adaptive theorem as written. Choosing \(R_p\) from the
unknown realized \(d_p\) is oracle allocation; a "universal decoder" removes
the decoder's need to know sparsity but does not choose the number of physical
measurements. The paper must instead do one of the following:

1. assume known upper bounds \(d_p\le\bar d_p\) and charge those bounds;
2. prove an adaptive doubling-and-validation rule, including its noisy stopping
   cost; or
3. use one worst-case bound \(d\) for every star, which costs
   \(O(kd\log(ek/d)+k\log(k/\eta))\).

It should also cap each star by its \(n_p\) direct pair tests when the compressed
bound is larger. In particular, T81.3 proves neither the dense wedge design's
\(O(s\log(N/s))\) rate nor an unconditional total rate depending only on \(s\).
It proves a blockwise, degree-sensitive identity-loop rate. Each loop still has
four flow legs, but that is the same constant factor as the pairwise baseline.

### Finite noise and control cost remain open

For unit, approximately orthogonal action channels, an unnormalized star loop
has partner energy proportional to \(n_p\). Its four-leg energy is approximately
\(2h(1+n_p)\), versus \(4h\) for a pair loop of duration \(h\), a ratio
\((1+n_p)/2\). Active-channel cost has the same linear growth. Moreover, BCH
remainders contain terms nonlinear in the mixture field and can grow with
\(n_p\); they are code-dependent bias, not automatically benign iid entries of
\(E_p\).

Normalizing the partner as
\(B_r=n_p^{-1/2}\sum_j b_{rj}V_j\) restores comparable quadratic control energy,
but attenuates every target bracket coefficient by \(1/\sqrt{n_p}\). Rescaling
to estimate the original \(Q_p\) amplifies endpoint/reset noise by
\(\sqrt{n_p}\), and equal-variance recovery can require order \(n_p\) more
repetitions. Dividing a loop displacement by \(h^2\) still amplifies endpoint
noise by \(h^{-2}\), while increasing \(h\) increases the finite-BCH bias.
Drift cancellation, legal inverse mixtures, saturation, reset error, and a
minimum bracket norm remain assumptions rather than consequences of RIP.

Thus the proposition supports a batch/reset-identity advantage only. Its
finite-noise guarantee is conditional on a correctly normalized bound for the
entire \(F_p\), and it does not discharge the component-count or energy-cost
objection.

### Run admission

The constructive star design clears the narrow mathematical prerequisite for
one matrix-only CPU falsifier **after the statement above is frozen**. The run
should compare oracle-degree, known-upper-bound, and implementable adaptive
allocations; coordinatewise and \(\ell_{2,1}\) decoding; unnormalized and equal-
energy codes; and identity-loop, active-channel, and energy/noise accounting
against direct pair tests and a generic sparse group-design control. It should
predeclare a beta-min/noise model and report both coefficient error and exact
row-support recovery.

Admission is not endorsement: the ideal identity-count result is already a
standard-CS corollary, so the run tests finite constants and cost survival. A
gain confined to unnormalized identity-loop count does not satisfy T81's 20%
broad-capability gate. Kill the experimental seam if the advantage falls below
20% in its declared cost model; keep 10--20% provisional only; and treat a
single-digit gain as rejection. No claim about grounding, compositional
intelligence, or physical intervention efficiency is reopened.
