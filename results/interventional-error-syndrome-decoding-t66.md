# T66 interventional error-syndrome decoding — sparse diagnosis before local repair

Date: 2026-08-01  
Status: **INDEPENDENTLY REJECTED AS A DISTINCT INTELLIGENCE MECHANISM; FANO/NSP DIAGNOSTIC CONTROLS RETAINED; NO RUN**

## 0. Claim

The unresolved T65 interface contains two different problems:

1. initially discovering a useful ontology from raw experience; and
2. diagnosing which part of an existing ontology became wrong.

T66 addresses only the second. It imports an idea from error-correcting codes,
compressed sensing, and active fault diagnosis:

> Treat action-conditioned prediction failures as a syndrome. Construct a
> fingerprint for each possible observation, binding, or mechanism edit;
> choose interventions that separate sparse combinations of those
> fingerprints; decode the responsible edits; then verify and commit only the
> localized repair.

This is not a stronger compiler rewriting a weaker model. The current model
proposes counterfactual edit fingerprints, while environment outcomes provide
the new information that selects among them. The method can repair sparse local
drift efficiently when most of the model is already right. It cannot bootstrap
a globally wrong ontology.

## 1. Finite diagnosis model

Let `p` edit atoms cover all currently admitted local explanations:

- observation-map atoms `O_j`;
- entity/binding swap atoms `B_j`; and
- factor/mechanism atoms `F_j`.

For each legal diagnostic probe `a_i`, record an outcome residual feature. In
the binary idealization, candidate edit `j` has fingerprint bit `A_ij`; in a
real-valued local model it has fingerprint column

\[
A_{:,j}=\widehat y(a_{1:m};M\oplus e_j)-\widehat y(a_{1:m};M).
\]

Here `M` is the current executable model and `M op e_j` is a counterfactual
version containing candidate edit `e_j`. After probing the environment, the
observed residual is

\[
r=y_{world}(a_{1:m})-\widehat y_M(a_{1:m}).
\]

The local sparse hypothesis is

\[
r=Ax+\xi,
\qquad \|x\|_0\le s\ll p,
\]

where nonzero entries of `x` name and size the responsible edits and `xi`
contains observation noise, model linearization error, and unmodelled effects.
Binding swaps can instead be represented as finite atoms; they need not be
infinitesimal parameters.

## 2. Information lower bound

Let the changed support `S` be uniform among the `N=binom(p,s)` supports of
exactly size `s`. For an at-most-`s` prior, replace `N` by
`sum_{k=0}^s binom(p,k)`. Suppose each diagnostic result is one noisy bit through a
binary-symmetric channel with error `eta<1/2`. Any fixed-length procedure that
identifies `S` with error at most `epsilon` requires

\[
m\ge
\frac{\log_2 N-h_2(\epsilon)-\epsilon\log_2(N-1)}
{1-h_2(\eta)}.
\]

The proof is the same Fano/channel-capacity argument as T46: the transcript
must convey almost `log_2 N` bits about the changed support, while one noisy
binary outcome conveys at most `1-h_2(eta)` bits. In the sparse regime,

\[
\log_2\binom{p}{s}=\Theta\!\left(s\log_2\frac ps\right).
\]

Therefore every method using this channel—including an optimal Bayesian
diagnoser—needs order `s log(p/s)` bits in the sparse regime. This is a
necessary support-identification bound, not an achievability result and not an
advantage for T66. It says nothing about amplitudes, signs, repair success,
fault-class semantics, or out-of-class rejection. It also does not apply to
unquantized infinite-precision measurements; outcome precision, reset
semantics, statefulness, and noise must be declared.

## 3. Exact sparse-decoding theorem

### Definition — null-space property

A probe matrix `A` has the order-`s` null-space property (NSP) when, for every
nonzero `h` in `ker(A)` and every index set `S` with `|S|<=s`,

\[
\|h_S\|_1<\|h_{S^c}\|_1.
\]

### Proposition T66.1

If `A` has the order-`s` NSP, every `s`-sparse edit vector `x` is the unique
solution of

\[
\min_z\|z\|_1\quad\text{subject to}\quad Az=Ax.
\]

**Proof.** Let `S=supp(x)` and let another feasible vector be `x+h` with
nonzero `h in ker(A)`. By the triangle inequality,

\[
\begin{aligned}
\|x+h\|_1
&=\|x_S+h_S\|_1+\|h_{S^c}\|_1\\
&\ge \|x_S\|_1-\|h_S\|_1+\|h_{S^c}\|_1\\
&>\|x\|_1.
\end{aligned}
\]

Thus no other feasible vector has equal or smaller `l1` norm. QED.

Robust NSP or a suitable restricted-isometry condition gives stable recovery
under bounded `xi`. Standard random sub-Gaussian linear-measurement designs can
obtain the required separation with `m=O(s log(p/s))` rows with high
probability, but that is a different channel from the binary Fano model above.
The two rates are not matching bounds unless precision, noise, reset,
adaptivity, and probe cost are made identical. More importantly, legal actions
select structured rows from a model-generated dictionary; they do not create
independent sub-Gaussian entries. A realized legal-probe matrix needs a robust
NSP/RIP, restricted singular-value, or coherence certificate before T66.1
applies.

## 4. Nonlinear boundary

Continuous and categorical edits require different models. For a
differentiable prediction vector `Y(theta)`, let small continuous edits be
collected in `Delta`. If the Jacobian is `L`-Lipschitz in operator norm along
the complete segment, Taylor's theorem gives

\[
Y(\theta+\Delta)-Y(\theta)=J_\theta\Delta+R,
\qquad
\|R\|_2\le\frac L2\|\Delta\|_2^2.
\]

Let `A` be the current model's Jacobian and `A_star=A+E` the unknown
environment-valid fingerprint matrix. The actual residual is

\[
r=(A+E)x+\xi.
\]

Stable support recovery requires a robust separation property for `A`, a bound
on `E` over sparse vectors, bounded noise and remainder, normalized columns,
and a minimum nonzero edit magnitude. T66 provides no mechanism that guarantees
these conditions.

Finite binding swaps, program replacements, and discrete mechanism edits do
not permit arbitrary fractional `l1` combinations. Multiple finite edits also
introduce cross-edit interactions. They require a constrained finite-hypothesis
or Bayesian decoder with explicit compatibility and behavioral-equivalence
semantics, not Proposition T66.1. Feedback-induced state drift additionally
requires matched checkpoints or a declared stateful measurement model.

## 5. Distinguishing error classes

Partition the dictionary as

\[
A=[A_O\;A_B\;A_F],
\]

for observation, binding, and factor edits. T66.1 applies to mixed faults only
if the combined dictionary has the separation property. This supplies an exact
identifiability boundary:

- if two class atoms have identical fingerprints under every legal probe,
  their causes are behaviorally indistinguishable and no learner can attribute
  the error correctly;
- high cross-class coherence makes diagnosis unstable under noise;
- interventions should be selected to increase the minimum separation or
  restricted singular value over still-plausible edit supports.

This directly answers the factor-versus-binding problem: it is solvable only to
the quotient induced by legal experiments, not by inspecting residual magnitude
alone.

## 6. Model-class rejection, not automatic variable invention

Define the best admitted sparse explanation residual

\[
\rho_s(r)=\min_{\|z\|_0\le s}\|Az-r\|_2.
\]

If `rho_s(r)` exceeds a threshold that includes observation noise,
finite-sample uncertainty, nonlinear remainder, and a validated fingerprint
uncertainty bound, the admitted local edit class is inconsistent with that
uncertainty set. Without a bound on `E=A_star-A`, a large residual may only mean
bad fingerprints and a small residual may be a spurious fit by wrong atoms;
`rho_s` is not a model-class rejection certificate.

Even a valid inconsistency certificate does not identify the missing variable.
That remains the raw-grounding problem from T65.

### Central circularity

The current model generates `A`, its edit vocabulary, and often the residual
feature channel, even though that model is known to be wrong. Its dictionary
can omit the true repair, distort every fingerprint through a bad observation
map or binding, and choose probes that are separating only inside its own false
world. NSP of `A` says nothing about agreement with `A_star`.

Breaking this loop requires a paid epistemically independent channel: raw or
redundant sensors outside the suspected component, executable environment
checks, an external verifier, or real calibration probes that estimate
fingerprint error. Probe design must be robust to a posterior or uncertainty
set over `A_star`, and candidate coverage must be measured before decoding.
Without this bridge, T66 assumes successful self-diagnosis rather than causing
it.

## 7. Repair transaction

A safe T66 cycle is:

1. detect a calibrated prediction failure;
2. enumerate typed edit atoms from the versioned model;
3. choose legal probes that separate the posterior candidate supports;
4. decode a sparse edit and its uncertainty;
5. apply it to a new content-addressed model version;
6. test held-out diagnostic probes and protected old query slices; and
7. commit only if the new version explains the failure without a protected
   regression, otherwise rollback and expand the hypothesis class.

The held-out verification probes cannot be reused to choose the edit. Exact
T65 noninterference remains conditional on a complete dependency certificate.

## 8. Potential gain and complete bill

For `s<<p`, a correctly calibrated support-separating diagnosis channel can
reduce probes relative to serial tests and localize retraining to implicated
components. T66 does not establish that a learned world model supplies that
channel, and the information lower bound provides no advantage over an equally
informed Bayesian or group-testing control. Normal serving need not run the
diagnosis procedure, so any extra computation would occur during
surprise-driven learning rather than every token.

The complete bill includes:

- real-world probe cost, latency, risk, resets, and opportunity cost;
- generating or storing an `m by p` fingerprint dictionary;
- counterfactual model executions for candidate edits;
- combinatorial probe selection or its approximation;
- sparse/Bayesian decoding and uncertainty calibration;
- held-out verification, protected-query replay, version storage, and rollback;
- failures of the sparse-additive assumption; and
- the cost of proposing new structure when the admitted class is rejected.

An equally informed optimal Bayesian diagnosis system can match or outperform
the decoder. The claimed advantage is not over that oracle control; it is the
architectural benefit of giving a learned world model an explicit diagnostic
interface instead of global undirected updating.

## 9. Direct collisions

The mathematical ingredients are established outside this application:

- compressed sensing and group testing provide sparse recovery and
  `s log(p/s)`-scale diagnosis;
- active fault isolation designs inputs whose residual sets separate known
  faulty configurations;
- Adaptive Interventional Debugging combines causal analysis, fault injection,
  and group testing to identify software root causes;
- Bayesian experimental design chooses interventions to reduce causal
  uncertainty;
- mechanistic localization improves the robustness and side effects of model
  editing and unlearning.

Current AI/world-model work also overlaps strongly:

- CSR learns causal representations from raw observations, uses prediction
  error to distinguish compact parameter shifts from state-space expansion,
  and updates a few parameters or adds variables;
- ESBM actively probes and locally edits executable symbolic mechanism memory;
- WM-SAR traces repeated error amplification backward to a compact planning
  subgraph and sends that region for repair;
- mechanism-sparsity representation learning provides identifiability results
  for sparse actions, interventions, and temporal dependencies.

T66 is therefore not a new sparse-recovery theorem or a claim that active
diagnosis is undiscovered. The only narrower hypothesis is whether a learned
world model can expose **calibrated, uncertainty-aware interventional edit
fingerprints** across observation, binding, and mechanism faults. T66 does not
provide that calibration mechanism, while direct fault-diagnosis and current
world-model repair work strongly occupy the remaining loop.

Primary sources:

- https://epubs.siam.org/doi/abs/10.1137/15M1046046
- https://arxiv.org/abs/2003.09539
- https://www.jmlr.org/papers/v9/seeger08a.html
- https://proceedings.mlr.press/v267/guo25k.html
- https://proceedings.iclr.cc/paper_files/paper/2025/file/83c230118e9f6688ba8f20bfef99e6da-Paper-Conference.pdf
- https://jmlr.org/beta/papers/v27/24-0771.html
- https://arxiv.org/abs/2606.07127
- https://arxiv.org/abs/2607.01767

## 10. Admission test

No current run is admitted. A future CPU-scale diagnostic-subsystem falsifier
would have to hide component
IDs, fault class, changed support, bindings, and raw-state permutation in a
family of modular controlled systems. It should compare:

- serial one-component testing;
- random probes plus sparse decoding;
- T66 active fingerprint design;
- an optimal or tightly approximated Bayesian active-diagnosis control with
  the same hypothesis set;
- prediction-error thresholding in the style of CSR;
- graph-backtrace repair in the style of WM-SAR; and
- global refitting and oracle-support upper bounds.

It must separately report oracle versus model-generated dictionaries,
candidate coverage, fingerprint calibration error, identifiable-support
coverage, decoder error conditional on coverage, continuous versus discrete
edits, and full probe/reset/verification cost. Only then could the subsystem
earn a larger local-model test if, across observation,
binding, mechanism, and mixed sparse faults, it achieves all of:

- at least `4x` fewer real probes than serial isolation at `p>=256,s<=3`;
- at least `95%` correct fault-class and support recovery under frozen noise;
- calibrated rejection of new-variable/out-of-class failures;
- no more than `2%` protected-query loss after repeated repair and rollback;
- a decisive gain over random probes and CSR-style residual thresholding; and
- competitive performance with the equally informed Bayesian control at lower
  complete computation or a clearly documented approximation trade.

These are empirical admission gates, not consequences of T66.1. Even passing
them would validate only a diagnostic subsystem. A production intelligence
claim separately requires raw learned factors and bindings, naturally withheld
changes, consequential tasks, a production-scale backbone, an end-to-end
reduction in interaction/regret or recovery time, protected capabilities, and
complete-cost Pareto evidence against AID/Bayesian/CSR/ESBM/WM-SAR-class
controls.

## 11. Decision

T66 retains two useful conditional controls: the Fano lower bound for a frozen
binary diagnosis channel and the NSP theorem for an independently valid linear
fingerprint matrix. It does not establish that a wrong learned model can build
that matrix or observe an independent residual.

The proposed intelligence mechanism is therefore rejected without a run.
Initial raw grounding, arbitrary global change, and new-variable invention
remain unsolved. The durable design rule is that self-correction needs an error
channel independent of the suspected representation; otherwise sparse decoding
can confidently diagnose the model's own fiction.
