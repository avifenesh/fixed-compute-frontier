# Independent adversarial audit: interventional error-syndrome decoding (T66)

Date: 2026-08-01  
Scope: information bound, sparse-recovery theorem, nonlinear and mixed faults,
fingerprint circularity, prior-art collision, and experiment sufficiency; no
experiments or GPU used

## Verdict

**NO RUN AS WRITTEN. RETAIN FANO AND NSP ONLY AS CONDITIONAL REFERENCE
THEOREMS; REJECT THE INFERENCE THAT THE MODEL-GENERATED FINGERPRINTS SATISFY
THEIR CHANNEL; TREAT T66 AS A DIAGNOSTIC-SUBSYSTEM HYPOTHESIS, NOT A
SUBSTANTIAL-INTELLIGENCE CANDIDATE.**

T66 identifies a useful engineering interface: actively localize a sparse fault
before editing a model. Its two mathematical results are standard and correct
inside their stated abstract models. They do not establish that a continually
learned world model supplies the required measurements. The decisive missing
object is a calibrated relationship between:

1. the possibly wrong model's counterfactual edit fingerprints;
2. the fingerprints the real environment would exhibit under the true repair;
3. the residual feature channel, which may itself use the wrong observation
   map; and
4. legal, state-changing probes with unequal cost and incomplete reset access.

Without that bridge, `r=Ax+xi` assumes the successful diagnosis. An admission
test with an externally correct atom dictionary could validate a sparse fault
localizer, but not model self-diagnosis and not a smarter production model.

## 1. Fano: valid necessary bound, wrong conclusion

Let `S` be uniform over exactly `s`-element supports, let every chosen probe have
a deterministic binary answer given `(S,a_i)`, and let that answer pass through
an independent memoryless binary-symmetric channel with crossover `eta`. For a
fixed-length adaptive transcript, Fano's inequality and the BSC capacity do give

\[
m\ge
\frac{\log_2\binom ps-h_2(\epsilon)-
\epsilon\log_2(\binom ps-1)}{1-h_2(\eta)}.
\]

Adaptivity does not evade this bound: conditioned on the past transcript, the
next chosen action contains no new information about `S`, and its one noisy bit
contributes at most the BSC capacity.

The scope is much narrower than T66 implies:

- For support size **at most** `s`, the hypothesis count is
  `sum_{k=0}^s binom(p,k)`, not `binom(p,s)`.
- The result identifies support only. It says nothing about edit amplitudes,
  signs, fault class, causal correctness, repair success, or out-of-class
  rejection.
- Correlated noise, stateful probes without reset, latent context, unknown
  amplitudes, or a model-dependent outcome map are not a memoryless BSC.
- A vector-valued or real-valued probe is not one noisy bit. Its information
  must be bounded under a declared quantizer/noise/SNR model.
- The bound applies to **every** method using that channel, including optimal
  Bayesian diagnosis. It supplies no advantage for T66.

Most importantly, a lower bound does not prove that logarithmic diagnosis is
possible. Replace “therefore ... possible” with “therefore ... necessary in
this channel.” An `O(s log(p/s))` achievability claim needs a legal probe family
whose induced binary tests are support-separating, or a separately declared
linear-measurement channel and recovery theorem. The BSC Fano lower bound and a
sub-Gaussian compressed-sensing upper bound cannot be presented as matching
bounds unless they refer to the same observations, precision, reset model, and
probe cost.

## 2. T66.1 is correct but does not apply to the proposed fingerprints

The order-`s` null-space property is necessary and sufficient for uniform exact
recovery of every real `s`-sparse vector by noiseless basis pursuit. T66.1 and
its proof are correct.

Nothing in the fingerprint construction implies NSP. Legal actions select rows
from a structured, model-generated response dictionary; they do not generate
independent isotropic sub-Gaussian entries. A standard random-design theorem is
therefore only an existence reference. T66 must prove a property of the
realized legal-probe matrix or use a checkable sufficient surrogate such as a
restricted singular-value or coherence certificate. Active row selection
cannot create separation when all legal probes give two atoms the same column.

There is also a model mismatch between the displayed matrix and the theorem:

- For continuous small edits, the linear matrix is the Jacobian `J_theta`.
  Unit finite differences are not Jacobian columns unless their step tends to
  zero with controlled error.
- For finite categorical atoms, such as binding swaps or replacing one
  mechanism program, simultaneous edits generally do not superpose. Fractional
  `l1` combinations may not even be legal models. These cases require a
  constrained finite-hypothesis/group-testing or Bayesian decoder.
- A column encodes one proposed edit value. Unknown edit magnitudes or alternate
  replacements enlarge the hypothesis class; a scalar coefficient does not
  make a nonlinear finite replacement scale linearly.
- Exact support recovery under noise requires more than stable value recovery:
  normalized columns, a robust NSP/RIP condition, a bounded perturbation, and a
  minimum nonzero signal are needed.

The correct statement is conditional: **if** a declared continuous local edit
parameterization yields `r=Jx+xi` and the selected legal probes make `J` robustly
separating, then sparse recovery is stable. T66 currently has no theorem that
its counterfactual finite-edit matrix has either property.

## 3. Nonlinearity and mixed faults are larger than the stated remainder

The Taylor bound is valid when `J` is `L`-Lipschitz in operator norm along the
entire segment from `theta` to `theta+Delta`. But it analyzes `J Delta`, while
T66 defines columns using finite differences. For binary finite edits,

\[
A x=\sum_{j:x_j=1}[Y(\theta+e_j)-Y(\theta)]
\]

and the error relative to `Y(theta+sum e_j)-Y(theta)` contains all mixed
cross-edit interactions. Observation changes alter inferred state; bindings
alter which entity a factor receives; factor changes alter later states. Their
joint effect is normally compositional or multiplicative, not additive. A
single quadratic remainder in a common Euclidean parameter space does not cover
discrete permutations, program replacements, routing changes, or policy-
environment feedback.

T66 needs two separate formalisms:

1. continuous local parameters with a Jacobian, trust region, matrix-
   perturbation bound, and robust sparse decoder; and
2. discrete typed edit hypotheses with compatibility constraints and a
   likelihood/Bayesian or combinatorial decoder.

For the continuous case, distinguish the model matrix `A` from the unknown
environment-valid matrix `A_star` and analyze

\[
r=(A+E)x+\xi,
\]

where `E=A_star-A` is fingerprint error. Support recovery needs a bound on
`E`, not merely on observation noise and Taylor remainder. For discrete mixed
faults, report recovery only modulo the behavioral equivalence relation induced
by legal interventions. T66 already notes indistinguishable atoms, but its
admission gate incorrectly demands 95% exact class/support recovery without
restricting to identifiable supports or allowing calibrated abstention.

Every active comparison must also begin from a matched checkpoint or specify
how state drift is included. Otherwise different probes observe different
world states, so rows do not form one measurement system.

## 4. The central circularity: a wrong model designs its own syndrome code

T66's `A` is not measured from the environment. It is generated by running the
current model and counterfactual edits of that model. Yet `M` is known to be
wrong. Four failure modes follow:

- the true repair atom may be absent from the model's vocabulary;
- a wrong observation map or binding may distort every candidate column;
- probe selection may confidently choose actions that are separating only in
  the wrong model; and
- the residual `y_world-y_M` may be encoded through the same faulty observation
  map, making observation faults invisible or self-confirming.

Environment outcomes select among hypotheses only if the hypothesized
fingerprints are calibrated enough to predict those outcomes. NSP of the
model's `A` says nothing about this calibration. A small `rho_s` can be a
spurious fit by wrong atoms; a large `rho_s` can reflect fingerprint error
rather than a missing variable. Therefore `rho_s` is not a model-class rejection
certificate until dictionary uncertainty is included.

Mandatory remedies are an outcome feature channel independent of the suspected
component (raw/redundant sensors or an external verifier), held-out real probes
that estimate fingerprint error, a posterior or uncertainty set over `A`, and
probe design robust to that uncertainty. Candidate coverage and fingerprint
calibration must be measured before decoding accuracy. Held-out verification
may reject a bad edit, but it does not retroactively justify a claimed
near-information-limit diagnosis cost.

## 5. Prior-art collision

T66's high-level loop is directly occupied:

- [Active fault isolation](https://epubs.siam.org/doi/abs/10.1137/15M1046046)
  already designs inputs to separate modeled fault configurations and studies
  the conditioning/optimization problem.
- [Adaptive Interventional Debugging](https://arxiv.org/abs/2003.09539) combines
  causal analysis, fault injection, adaptive interventions, and group testing
  to identify root causes in real software. This is the closest direct ancestor
  of the syndrome-decoding loop.
- [Bayesian sparse-model diagnosis and optimal design](https://www.jmlr.org/papers/v9/seeger08a.html)
  already couples sparse inference with experimental design and naturally
  represents uncertainty that T66's point fingerprint matrix omits.
- [CSR (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/83c230118e9f6688ba8f20bfef99e6da-Paper-Conference.pdf)
  learns causal latent world models from raw observations, uses prediction
  error to distinguish compact distribution/parameter shifts from state-space
  expansion, locally updates parameters or adds variables, and demonstrates
  low-sample policy adaptation. It substantially occupies T66's compact-repair
  versus model-class-expansion decision.
- [ESBM](https://arxiv.org/abs/2606.07127) generates active checkpoint probes
  from current failures and uncertainty, makes typed local mechanism edits,
  verifies them against task and world-model criteria, versions accepted
  models, and measures recovery after mechanism changes.
- [WM-SAR](https://arxiv.org/abs/2607.01767) localizes error amplification to a
  compact planning subgraph and repairs that region with an LLM, directly
  testing compact root-cause repair under token budgets and downstream rollout
  stabilization.

Consequently, T66 is not novel as active diagnosis, group testing, sparse
recovery, local world-model adaptation, or compact graph repair. The narrower
unimplemented hypothesis is:

> Can a learned world model expose calibrated, uncertainty-aware
> counterfactual edit fingerprints across observation, binding, and mechanism
> faults, such that legal interventions achieve better end-to-end repair cost
> than AID/Bayesian/CSR/ESBM/WM-SAR-class controls?

That is a legitimate experiment question only after solving the fingerprint
circularity above.

## 6. Admission test: diagnostic subsystem only

The proposed CPU test can establish that an engineered sparse diagnostic
subsystem recovers hidden atoms efficiently in a controlled modular family. It
cannot establish that a model learned the atoms, generated world-valid
fingerprints, invented missing structure, or became substantially more
intelligent.

The current `4x` versus serial and `95%` support thresholds are also
insufficient by themselves. Serial tests and group probes must have matched
reset, risk, duration, outcome precision, and intervention cost. Results must
separate:

- oracle dictionary versus model-generated dictionary;
- fingerprint calibration error versus decoder error;
- identifiable support coverage versus exact recovery conditional on coverage;
- continuous small edits versus discrete/mixed edits;
- known sparse faults versus dense, interacting, and out-of-class faults; and
- probe count versus total model executions, selection compute, verification,
  rollback, and failed-probe cost.

An “optimal or tightly approximated” Bayesian control needs an explicit
approximation and quality bound; otherwise losing to or beating it is not
interpretable. AID-style adaptive group testing must be a direct control, not
only serial testing and random sparse decoding.

A substantial intelligent-model claim requires a separate integrated gate:
raw learned factors and bindings, naturally occurring or withheld changes, a
production-scale backbone, consequential downstream tasks, and an end-to-end
reduction in interactions/cumulative regret or recovery time while preserving
old capabilities. It must charge candidate generation, all counterfactual
runs, real probes, verifier calls, memory, repair training, and rollbacks, and
must beat the direct prior-art controls under the same information and tools.
Support localization alone is an intermediate metric.

## Mandatory corrections before any run

1. Restate Fano as a necessary support-identification bound for a uniform,
   fixed-cardinality, memoryless BSC channel. Use `sum_{k<=s} binom(p,k)` for an
   at-most-`s` prior and remove the inference of achievability.
2. Do not combine the BSC lower bound and sub-Gaussian linear-measurement upper
   bound as one near-optimal scale without matching their channel, precision,
   adaptivity, reset, and cost assumptions.
3. Keep T66.1 as a conditional basis-pursuit theorem. Add a separate theorem or
   empirical certificate connecting legal model fingerprints to robust NSP/RIP
   or an appropriate finite-hypothesis separation property.
4. Separate continuous Jacobian edits from finite categorical edits. Do not use
   unconstrained `l1` combinations for binding swaps or program replacements.
5. Introduce the environment-valid matrix `A_star`, fingerprint perturbation
   `E=A_star-A`, minimum signal, column normalization, and a robust support-
   recovery/abstention condition.
6. Model cross-edit interactions, mixed observation-binding-factor faults, and
   matched-checkpoint requirements explicitly. Evaluate only identifiable
   supports or score the correct behavioral equivalence class.
7. Break the self-generated-fingerprint circularity with an independent outcome
   channel, real fingerprint calibration, uncertainty-aware robust probe
   selection, and candidate-coverage reporting. Include dictionary uncertainty
   in `rho_s`; otherwise it is not a rejection certificate.
8. Rewrite novelty around calibrated edit fingerprints in learned world models
   and add direct AID, active fault isolation, Bayesian, CSR, ESBM, and WM-SAR
   comparisons.
9. Relabel the admission test as a diagnostic-subsystem falsifier and add the
   separate full-cost downstream production gate required for any substantial
   intelligence claim.

After those corrections, a CPU-only oracle-versus-model-fingerprint falsifier
could be justified. The present protocol should not run because it can pass
with a correct externally supplied syndrome dictionary while leaving T66's
central learned self-diagnosis claim untested.
