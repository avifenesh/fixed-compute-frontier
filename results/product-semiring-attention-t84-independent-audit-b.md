# T84 product-semiring attention — independent adversarial audit B

Date: 2026-08-02  
Reviewed artifact: `product-semiring-attention-t84-paper-screen.md`, including
the revision that separates unnormalized `B_beta` from normalized `A_beta` and
uses `B_infinity` for the proposed tropical branch.  
Decision: **revise before any learning run**.

## Bottom line

The revised paper screen fixes the largest algebraic error in its earlier
form: row-normalized `A_infinity` is no longer claimed to implement the same
multi-step Bellman recurrence as unnormalized `B_infinity`. Equations 1--4,
the finite-temperature error bound, and the fixed-`S` Bellman proposition are
mathematically sound with the qualifications below.

That does not yet make the proposed model a defined experiment. The actual
soft branch is signed Euclidean attention, not the positive/log-domain
operation used in the product-semiring identity. The tropical value map is
unspecified. Branch normalization, residual mixing, changing scores, and the
MLP interrupt the exact recurrence after one local aggregation. Exact max also
creates a severe winner-only training path. A CPU result obtained before these
choices are frozen could be explained by the value transform, privileged graph
layout, branch scaling, or a weak control rather than by mixed aggregation.

The idea is not closed. A shared `QK` score followed by channel-wise soft and
max-plus reductions is a precise operator that standard attention does not
natively expose. But the broad conceptual territory is already occupied by
Tropical Attention's max-plus aggregation and explicit hybrid-semiring future
direction, and by HydraHead's heterogeneous attention branches with separate
normalization. The remaining novelty is a narrow implementation and empirical
claim, not a new semiring theorem.

No cheap CPU test can establish the model-level claim under the project's hard
gate. A revised CPU test can reject the operator or earn a later model-scale
test; it cannot establish simultaneous >=20% improvement, protected language
and retrieval non-inferiority, and same served cost.

## Claim-by-claim falsification

| Claim | Audit result | Consequence |
|---|---|---|
| `B_beta`, `A_beta`, and the beta=1 identity | Correct, but Equation 3 is a positive-value identity. For general beta, `exp(beta A_beta) = sum softmax(beta s) exp(beta u)`. | State the general identity and do not identify it with ordinary signed attention. |
| Finite-temperature bound | Correct for a nonempty set of finite, unmasked entries. The tight row-wise count is `n_i`, not global sequence length. | Treat `log(n_i)/beta` as a worst-case uniform bound, not a prediction that observed error must grow that way. |
| Product-semiring construction | The Cartesian product of two semirings is a semiring. The deployed branch pair as written is not literally that product, because signed softmax averaging is not log-semiring aggregation of log-positive values. | Rename it a heterogeneous/product-inspired attention block, or define a true log-domain soft branch and its signed representation. |
| Exact Bellman feature update | Correct for raw `B_infinity`, fixed `S`, max-plus state coordinates, and exactly `L` edges after `L` products. | The theorem proves a local primitive, not that the proposed Transformer stack executes the recurrence. |
| Shared-score compression | The direct construction uses one relation score and permits different predecessors per channel. The `r+C` versus `C(r+1)` count is not a lower bound for Transformers or hardware cost. | Keep it only as a construction; test multi-query/narrow-head and branch-specific-score controls. |
| Signed-value handling | Not defined. "Transform to finite log-positive potentials" does not specify forward map, inverse/map-back, epsilon, clipping, or semantics. | No run is interpretable until this is frozen and costed. |
| Trainability | Exact max sends gradient only through the winning predecessor of each channel. | Winner starvation and branch collapse must be tested before capability. |
| Causal model relevance | A causal mask gives only order-respecting token paths. It does not expose an arbitrary problem graph or preserve a fixed relation matrix across layers. | The CPU witness must be decoder-causal without privileged adjacency or DP state. |
| Novelty | Broad max-plus attention, trainable semiring modules, heterogeneous attention branches, and branch normalization are prior art. The within-head shared-score conjunction was not found in the searched primary literature. | At most a narrow architectural novelty remains, conditional on a broader search and a decisive result. |
| >=20% capability prior | Tropical Attention supplies evidence that max-plus bias can yield large synthetic OOD gains, but not for this mixed causal language model or against this gate's controls. | It justifies one kill test after revision, not admission now. |

## Algebra: what survives and what does not

### The normalized identity and error bound survive

For a query row with `n_i >= 1` finite entries,

\[
\exp(\beta A_\beta(s,u))
=\sum_j \operatorname{softmax}_j(\beta s)\exp(\beta u_j).
\]

The stated beta=1 equation is its special case. Applying
`max(x) <= LSE_beta(x) <= max(x)+log(n_i)/beta` separately to `s+u` and
`s` proves

\[
|A_\beta-A_\infty|\leq \log(n_i)/\beta.
\]

This is close to worst-case tight: make all `s_j` equal so its log-sum-exp gap
is `log(n_i)/beta`, while making one coordinate of `s+u` dominant so its gap
approaches zero. With a causal mask, `n_i` is the number of visible keys for
query `i`; a fully masked row is undefined. Masked entries must be `-infinity`
before both reductions.

The bound says what temperature suffices uniformly over adversarial inputs. It
does not imply that a learned finite-temperature branch will exhibit that
worst case, nor does it provide an accuracy advantage for exact max.

### The revised Bellman statement is locally correct

`B_infinity(S,U)=max_j(S_ij+U_jc)` is max-plus matrix multiplication. With
the same fixed `S` at every step, induction yields maximum scores of exactly
length-`L` walks. "Up to `L`" would additionally require identity/self-loop
edges. The revised screen correctly no longer assigns this theorem to
row-normalized `A_infinity`; that correction removes the earlier fatal
normalization mismatch.

However, the proposed Transformer does not stack this recurrence exactly:

- each layer normally recomputes `Q`, `K`, and hence `S` from a changed hidden
  state with different learned projections;
- branch RMS/scalar normalization changes the raw potentials;
- the output projection, residual addition, and MLP mix tropical coordinates
  with ordinary real coordinates; and
- a causal mask only permits paths consistent with token order.

Therefore the theorem certifies one operator call. To claim an exact
multi-layer dynamic program, T84 would need a persistent tropical state lane,
an explicit encode/decode invariant, and either tied/frozen transition scores
or a theorem for state-dependent transitions. Otherwise the honest claim is
only that the operator supplies a useful max-plus inductive bias.

### The "product semiring" name is ahead of the deployed algebra

If a score is embedded diagonally as `(s,s)` and values are pairs of
log-domain and max-plus-domain elements, componentwise matrix multiplication
over the Cartesian product is valid. T84's proposed soft branch instead
computes `sum softmax(s) v` for arbitrary signed `v`, while its tropical branch
computes `max(s+u)`. The first operation is not the log-semiring component
unless `v=exp(u)>0` and the result is kept or mapped consistently in log space.

This does not invalidate the heterogeneous operator. It invalidates the claim
that the exact deployed block follows merely from the Cartesian-product
identity. The paper should either:

1. define both components in their semiring domains, including a reversible or
   explicitly lossy signed representation; or
2. call the architecture shared-score heterogeneous aggregation and present
   the product semiring as motivation, not as its literal algebra.

## The missing signed-value contract

The phrase "finite log-positive potentials" leaves materially different models
under one name:

- `u = W_V h` treats every real coordinate directly as a log potential. This
  needs no positivity transform, but the output is a log-score rather than an
  ordinary value and its gauge/scale must be specified.
- `u = log(softplus(W_V h)+epsilon)` followed by exponentiation defines a
  positive-value interpretation, but adds nonlinear work, clipping choices,
  overflow risk, and potentially saturated gradients.
- sign splitting `v=v_pos-v_neg` doubles channels or compute and does not
  commute with max, so it is not an exact signed max-plus value update.

Max-plus computations are also basis dependent: an arbitrary rotation or
output mixing of feature coordinates does not preserve coordinate-wise maxima.
T84 needs a frozen equation for tropical encode, aggregation, branch
normalization, decode, and residual fusion. The all-softmax control must receive
the same encode/decode and normalization budget, or any win is confounded by
those additions.

## Gradient and optimization falsifier

For a unique winner `j* = argmax_j(s_j+u_jc)`,

\[
\frac{\partial B_\infty}{\partial s_j}
=\frac{\partial B_\infty}{\partial u_{jc}}
=\mathbf{1}[j=j^*].
\]

Thus every losing key and value route receives zero gradient from that channel;
ties receive an implementation-selected subgradient. Across channels, score
gradients are counts weighted by downstream channel gradients, so early shared
winners can concentrate learning on a few tokens. The soft channel group can
train the shared `QK` score, but that is not evidence that the tropical branch
learned a useful relation. It may merely consume a graph learned for soft
retrieval, or be ignored after fusion.

This is not hypothetical. *Semiring Activation in Neural Networks* reports
special initialization/learning-rate sensitivity and the zero-gradient
winner problem for max-like semiring layers. *Untangling Component Imbalance
in Hybrid Linear Attention Conversion Methods* independently shows that a
hybrid can score well while one branch is bypassed, and uses branch suppression
to expose/fix that failure. T84 therefore needs, per layer and seed:

- winner entropy and fraction of keys/values receiving any gradient;
- dead tropical-channel fraction and winner concentration;
- tropical versus soft branch output norm and residual contribution;
- branch ablation at inference, not only learned scale inspection; and
- exact-max versus finite-`beta`, annealed-`beta`, and straight-through
  training, with all surrogate and train/inference mismatch costs charged.

Backward workspace for per-query, per-channel argmax provenance must also be
included in the training ledger.

## Causal masking can invalidate the proposed witness

A full-attention graph task can hand the model exactly the topology that the
Bellman operator expects. That would not test a causal decoder language model.
Under a decoder mask, every selected predecessor must be earlier in token
order. Cycles, backward edges, and arbitrary graph walks cannot be represented
as repeated attention paths unless the input is topologically serialized,
unrolled, or supplied with special masks. Any of those can leak the algorithm.

The first witness must therefore use the same causal mask and positional scheme
as the intended model, hide adjacency/DP tables/operator labels, and prove that
all task information is recoverable from ordinary tokens. It should include a
permuted serialization control. If performance disappears when a convenient
topological order is removed, the result is a representation shortcut, not a
general reasoning improvement.

## Novelty boundary after 2025--2026 primary-literature check

- [Tropical Attention](https://arxiv.org/abs/2505.17190) already implements
  tropical projections, a tropical Hilbert score, and channel-wise max-plus
  aggregation; proves tropical-circuit/transitive-closure expressivity; and
  reports large synthetic combinatorial OOD effects. It explicitly names
  hybrid semiring architectures as future work and leaves autoregressive
  generation and production runtime untested. That makes T84 plausible, but
  also makes "soft plus tropical" an anticipated extension.
- [HydraHead](https://arxiv.org/abs/2606.20097) establishes static
  heterogeneous attention at head granularity, causal head selection, and
  separately normalized branch fusion. Its operators and purpose differ, but
  head-level hybridization and scale reconciliation are not novel.
- [Semiring Activation in Neural Networks](https://arxiv.org/abs/2405.18805)
  establishes trainable semiring neural modules and documents their
  optimization pathologies.
- [Simulating Hard Attention Using Soft Attention](https://aclanthology.org/2026.tacl-1.8/)
  weakens broad claims that hard selection alone creates an unreachable
  function class. It does not directly simulate T84's channel-dependent
  `argmax_j(s_j+u_jc)` with a single standard head, so it is a control
  requirement rather than a refutation.
- [Untangling Component Imbalance in Hybrid Linear Attention Conversion Methods](https://arxiv.org/abs/2510.05901)
  shows why good hybrid-model scores do not prove both branches are useful.

The independent search did not find the exact within-head construction that
reuses one dot-product score matrix for signed softmax channels and unnormalized
max-plus channels. That is the defensible novelty target. It remains a narrow
operator/layout claim until a broader formal search and an end-to-end result.

The candidate's 45.99% aggregate-error calculation from Tropical Attention is
arithmetically plausible from the published classification rows, but it is not
a transferable effect size. It chooses the best baseline separately per task,
sums errors across heterogeneous tasks, omits the regression rows, and has no
simultaneous interval. More importantly, those are task-specific noncausal
tropical models rather than one mixed causal learner. It supports "worth one
falsifier," not a 20% prior for T84.

## Fatal same-cost controls

The current list is directionally strong but still needs these frozen controls:

1. **Published Tropical Attention**, faithfully implemented, versus T84's
   ordinary-dot-product tropical branch. "All tropical" is otherwise
   ambiguous and could omit the strongest prior method.
2. **Shared-score versus branch-specific-score hybrid**, with total parameters,
   active width, and actual runtime matched. This isolates the central
   relation-sharing hypothesis.
3. **All-softmax plus the identical tropical encode/decode, branch norms, and
   output fusion.** This isolates the reduction from the extra nonlinear path.
4. **Within-head channel split versus whole-head split**, including
   many-narrow-head and multi-query/shared-key variants. This tests whether the
   proposed granularity, not merely hybridization, matters.
5. **Finite-`beta` `B_beta`**, learned/frozen and annealed variants. Ordinary
   low-temperature `softmax(s)V` is not the same control because it cannot
   choose a different predecessor per value channel.
6. **Branch-usage controls:** tropical-branch ablation, soft-branch ablation,
   branch dropout, and matched random/frozen allocation. These detect bypass.
7. **Same real cost, not nominal operations.** After measuring tropical kernel
   latency/workspace, let the all-softmax baseline spend that same complete
   training and served budget on the best ordinary use: extra FFN width, depth,
   recurrent steps, or heads. Solvers are ceilings, not the strongest learned
   same-cost baseline.

All controls need identical seeds, examples, tokens, tuning allowance, stopping
rules, and causal packing. A result erased by any stronger same-cost learned
control closes the main lane under the stated contract.

## Would a cheap CPU test change the model-level decision?

**No, not directly.** The current model is underspecified, so a run now would
not test a unique hypothesis. After the revisions below, a cheap CPU run has
one legitimate role: reject the mechanism before GPU rental.

A revised CPU witness should begin with exact operator tests, gradient-coverage
diagnostics, and one tiny interleaved causal learner. It should be stopped if
the tropical branch is dead/bypassed, if the gain depends on privileged graph
serialization, or if its point estimate is below 20% against the strongest
matched CPU control. A positive result may authorize a larger test, but it is
not model-level evidence: tiny synthetic tasks cannot establish protected
language/retrieval non-inferiority, simultaneous confidence across meaningful
domains and seeds, or production-shaped GPU cost.

Under the hard gate, admission ultimately requires at least the preregistered
multi-seed causal mixed suite, simultaneous one-sided bounds, a protected
language/retrieval suite, and real served p50/p95 memory/latency or a frozen
capability-per-complete-cost score. The decisive uncertainty is not whether
max-plus can win a toy DP task—prior work already suggests it can—but whether
the mixed branch adds broad causal-model capability without losing ordinary
language behavior or cost efficiency.

## Required revision before a CPU run

1. Freeze the exact tropical value encode/decode equations, epsilon/clipping,
   branch normalization, residual fusion, initialization, and numerical type.
2. Decide whether "product semiring" is literal. If not, narrow the name and
   claims to shared-score heterogeneous aggregation.
3. State that the Bellman theorem certifies a local primitive only, unless a
   persistent tropical state path preserving raw potentials is added.
4. Freeze a decoder-causal task representation and add serialization/permutation
   leakage controls.
5. Add the seven fatal controls above and a complete measured-cost reallocation
   rule.
6. Preregister branch-use and gradient-health kill thresholds before looking at
   accuracy.
7. Treat the CPU stage as reject-only. Passing it earns a model-scale test; it
   does not satisfy the >=20% model-level gate.

REVISE

## Re-audit after applied revision — 2026-08-02

### Decision

The blockers identified above are resolved sufficiently to write and audit a
preregistration. This does **not** admit implementation, a CPU run, a GPU run,
or a capability claim.

The current screen now defines one unambiguous candidate operator:

\[
O^{\rm soft}_{ic}=\sum_{j\in J_i}\operatorname{softmax}_j(S_{ij})
V^{\rm soft}_{jc},\qquad
O^{\rm max}_{ic}=\max_{j\in J_i}(S_{ij}+V^{\rm max}_{jc}),
\]

with a nonempty causal source set, negative-infinity masking, disjoint columns
of one ordinary real-valued value projection, and no hidden log/positive-value
encoding. Identical non-affine group normalization is applied to the primary
candidate and its matched all-softmax control. The screen also correctly
renames the deployed design heterogeneous/direct-sum attention and limits the
Cartesian-product semiring discussion to motivation.

The proposed three-stage ladder now isolates the intended progression:

1. exact equations and causal masking;
2. learning one relation used by both reductions on the same instances; and
3. a decoder-causal, permuted weighted-automaton task without supplied DP state,
   privileged masks, or fixed topological serialization.

The XOR composition, branch lesions, score-unsharing lesion, serialization
probes, gradient-health kills, and published/architectural controls address the
previous shortcut and bypass objections. The CPU stage is explicitly
reject-only, and all measured tropical overhead is eventually returned to the
ordinary baseline as useful compute. These are the right boundaries.

### Items the preregistration must freeze

These are protocol details, not reasons for another concept revision:

- exact data generators, causal token format, train/OOD supports, and proof
  that each XOR input is independently balanced and unavailable from surface
  statistics;
- model dimensions, branch ratios, initialization, optimizer, schedules,
  token budgets, seeds, stopping rules, and allowed tuning budget for every
  control;
- numerical branch-health thresholds, including what constitutes a dead
  channel, insufficient winner support, branch bypass, and a failed lesion;
- stage-specific reject/pass rules, with no rescue hyperparameter introduced
  after observing accuracy;
- sample sizes and a power calculation for detecting the 20% effect; the exact
  simultaneous-confidence procedure must include both domains and all fatal
  candidate-versus-control comparisons so the strongest control is not chosen
  post hoc without multiplicity correction;
- the CPU-time matching procedure and variance tolerance, while making clear
  that CPU timing cannot establish target-GPU cost; and
- an immutable artifact hash plus a rule that any operator, dataset, control,
  or threshold change creates a new preregistration.

If a preregistration can specify those items without weakening a fatal control,
it earns a separate audit. Only an accepted preregistration may authorize the
small reject-only CPU ladder.

ADMIT PREREGISTRATION

## Audit of CPU preregistration v1 — 2026-08-02

Reviewed artifact: `shared-score-heterogeneous-attention-t84-cpu-preregistration-v1.md`  
Scope: protocol only; no implementation or experiment was run.  
Decision: **revise the preregistration before hashing or implementation**.

### Bottom line

The generator is substantially better than the earlier task sketch. On every
instance the two latent bits are exchangeable and independently balanced
conditional on the shared scores, the XOR prevents either bit alone from
predicting the final label, the query is causally last, and random record order
removes a fixed positional solution. The candidate and most controls are also
specified closely enough to implement.

The preregistration is nevertheless not immutable or statistically valid yet.
There are four blocking classes:

1. the mean of seed-level error ratios is unstable near zero control error and
   is not the natural "20% lower expected error" estimand;
2. lesion effects used to claim causal branch use are point-thresholded and
   excluded from multiplicity despite being part of the positive claim;
3. several controls do not receive the same group normalization or a complete
   target-workload timing match, and the official tropical control is not
   pinned to a source commit and exact causal equations; and
4. artifact authorization is circular: `r_time` requires executing timing,
   but the source audit that authorizes CPU execution requires a manifest that
   already contains `r_time`. Editing the preregistration to mark it accepted
   would also change the hash being accepted.

These are repairable protocol defects, so the verdict is not reject.

### 1. Mathematical and data validity

The target equations and balancing argument are correct. Because `v` and `u`
are independent across their two channels, `y_s` and `y_m` are conditionally
independent fair bits given `q,k`; exact quadrant rejection balancing does not
create a per-example label feature. There is no batch-coupled operation in the
model, so batch balancing itself is not a usable prediction channel.

The following generator details remain executable degrees of freedom:

- "per-field" surface features do not say whether fields are measured before
  or after the orthogonal mixing;
- the logistic regression omits feature standardization, regularization,
  solver, tolerance, maximum iterations, and seed;
- the query-only and one-record-only MLPs omit which record is selected, input
  normalization, optimizer, learning rate, batch sequence, seeds, and whether
  separate models are trained for the three labels;
- the QR sign convention and resulting 14-by-14 matrix are not frozen as an
  artifact, so different NumPy/LAPACK builds can produce a different mixer;
- training RNG streams are not explicitly separated into data, initialization,
  and model randomness. Reusing one process RNG can make architectures see
  different batches after consuming different random numbers; and
- causal attention does not state whether the diagonal is legal. This matters
  for record tokens and the first token, even though only the final query is
  read out.

The leakage probes are also too weak to exclude a nonlinear pooled shortcut.
A linear classifier over summary statistics and a one-record MLP can both fail
while an attention-free permutation-invariant network succeeds. Such success
would not be data leakage—it would be a stronger same-cost learned control.
The score-conditioned cross-token max/maxout control required by the paper
screen is absent from the nine controls.

Required edits:

1. Freeze the mixer itself as float64 hexadecimal values plus SHA-256, not only
   the QR seed. Define surface probes on both named raw fields and mixed token
   coordinates.
2. Give complete equations/configurations and independent seeds for each
   leakage model. For the one-record probe, select a uniformly random record
   using a frozen RNG and train separate declared targets, or state explicitly
   that it predicts only XOR.
3. Use counter-based or independently seeded RNG streams so every architecture
   and paired seed receives byte-identical training batches regardless of
   model RNG consumption.
4. State self-inclusive causal masking `J_i={0,...,i}` for every token, or give
   another complete rule that guarantees a legal source for the first token.
5. Add the promised same-cost score-conditioned pooled max/maxout or DeepSets
   control. If it is intentionally deferred, narrow v1's claim and amend the
   prior screen; passing v1 then cannot mean the fatal control set was cleared.

### 2. Stage-0 ambiguities

The operator tests are appropriate, but the following must be exact before a
source audit:

- run the 1,024 cases in both float64 reference mode and float32 candidate mode;
- define max-shifted `logsumexp` and allow the stated tolerance on both sides of
  `0 <= B_beta-B_inf`, since floating arithmetic can produce a tiny negative;
- define the gradient scalar. For example, sum all max output channels, then
  require `dL/du_jc` to be the channel winner indicator and `dL/ds_j` to equal
  the number of channels won by source `j`;
- define how the "soft hard-limit" counterexample is computed (explicit
  argmax-copy reference, not an unspecified extreme temperature); and
- define the canonical deterministic payload for the bitwise replay. Timestamps,
  paths, JSON key ordering, and environment metadata cannot be part of a
  bitwise-identical tensor-result requirement.

### 3. Architecture and control fairness

The parameter-matched backbone still has unresolved semantics: whether ordinary
RMSNorm has learned affine weights, whether group normalization is applied to
every token or only the final query, the exact activation/order of a repeated
block, and the per-model active parameter counts. These affect both fairness
and the source hash.

More importantly, controls 3--5 are not normalization-matched:

- `soft_narrow` has no counterpart to the candidate's eight four-channel
  normalized output groups;
- `hybrid_independent` has eight four-channel heads but does not state that each
  head output is normalized; and
- `hybrid_whole_head` has dimension-eight heads but does not state how their
  channels are divided into four-channel normalization groups.

This can falsely attribute a normalization benefit to the reducer or can make
the independent-score and whole-head fatal controls artificially weak.

The official tropical control is not yet reproducible. "v2.0.0 equations" is
not a commit hash, and the current upstream README says v2.0.0 can select either
the `tropical-gemm` backend or a PyTorch fallback. The paper's displayed
valuation and the preregistration's `log1p(ReLU(x))` wording are not identical,
so "source faithful" must name which source is authoritative. Its causal mask,
projection dimensions, normalization flags, symmetric score equation, numeric
stabilization, initialization, and CPU backend must be frozen. A two-learning-
rate common grid may also be an unfair straw training recipe if the official
module requires its published initialization or scale. Source: [official
Tropical Attention repository](https://github.com/Baran-phys/Tropical-Attention).

Required edits:

1. State that learned backbone RMSNorm is affine or non-affine and apply the
   candidate's non-affine four-channel output normalization at every query
   position, not only the readout token.
2. Normalize eight predetermined groups of four output channels for
   `soft_narrow`; normalize every dimension-four head for
   `hybrid_independent`; split every dimension-eight head into two fixed
   four-channel groups for `hybrid_whole_head`. Keep unnormalized variants only
   as secondary conventional controls.
3. Freeze a table for candidate and every control: Q/K/V score dimension,
   scale, number of score maps, reducer, group assignment, recurrence, exact
   active parameter count formula, and intended source of every output channel.
4. Pin `tropical_official` to an immutable Git commit and file hashes, force one
   named CPU backend, copy the complete equations into the preregistration,
   specify causal masking, and predeclare either the upstream training recipe
   or two control-appropriate hyperparameter configurations under the same
   total tuning budget as the candidate.
5. Define `soft_time_matched` exactly: each of the two parameter-tied blocks is
   applied `r_time` consecutive times before moving to the next block, or state
   the intended alternative explicitly.

### 4. Timing is matched to the wrong workload

`r_time` is selected only at sequence length 9, while three of four primary OOD
slices use lengths 17 or 33 and half the macro weight is at length 33. Softmax
GEMM and broadcasted add/max kernels need not have the same scaling curve. A
control matched at length 9 can be under- or over-budget on the actual primary
workload. Integer whole-block repetition is also too coarse to represent the
best ordinary use of a partial runtime premium.

Required edits:

1. Time fixed batches at lengths 9, 17, and 33. Weight them `1/4, 1/4, 1/2`
   for the four-slice macro and require the control to be no slower than 105%
   of candidate p50 **and p95** both on that weighted workload and separately
   at length 33.
2. Pin input tensors, memory layout, inference mode, affinity/thread environment,
   backend flags, warmup, clock, RSS method, and round-robin order before the
   timing-only authorization.
3. Add a frozen small menu that can spend fractional slack on ordinary FFN
   width as well as integer tied depth. Select among timing-eligible variants
   using only the frozen development set and a tuning budget charged equally to
   candidate/control. Do not call `r=1` the best use of cost merely because no
   entire repeated block fits.
4. Report and bound training time separately. Equal optimizer steps do not
   imply equal complete training cost when exact-max backward and tied depth
   differ.

### 5. The 20% statistic is not robust near zero error

The current statistic

\[
R_s=1-e_{s,c}/e_{s,m}
\]

targets the mean of per-seed ratios. It is unbounded below, highly skewed when
a control error is small, and gives disproportionate behavior to seeds with a
near-zero denominator. Setting `R=0` only when both errors are zero does not fix
the discontinuity. A Student-t bound with 16 observations has no credible
finite-sample coverage for that mixture near saturation.

The natural claim "candidate expected error is at least 20% lower" is instead

\[
E[e_c] \leq 0.8E[e_m].
\]

Use the bounded paired seed statistic

\[
G_{smd}=0.8e_{smd}-e_{s,c,d}.
\]

For each slice and macro, require a simultaneous lower bound for `E[G]` to be
nonnegative. Report `1-sum_s e_c/sum_s e_m` as the corresponding effect size.
This removes division by a random seed-level denominator while preserving the
20% null. If the intended estimand truly is mean seed-level relative reduction,
the file must say so and use a preregistered ratio-confidence method robust to
zero denominators; the present t interval is not enough.

Even the stable test is scientifically trivial on a saturated task. Reducing
error from 0.10% to 0.08% is a 20% relative reduction but not the sought
substantial result. Add a frozen saturation/nontriviality rule: the minimum
control macro error must be at least 5.0%, and the candidate must have at least
a one-percentage-point macro absolute-error reduction against every control.
Give the absolute reduction its own simultaneous lower bound. If the control
floor fails, report `TASK SATURATED`; do not call it candidate failure or
success, and do not redesign the task after opening candidate metrics.

Non-inferiority statistics also need explicit definitions. State that the two
soft-bit statements are per-seed candidate-minus-control differences on the
four-slice OOD macro, and the two ID statements are analogous differences on
ID final error; require simultaneous upper bounds <=0.01.

### 6. Multiplicity omits the causal claims

Conditions 6 and 7 are not merely deterministic health observations. They are
the empirical evidence behind the positive phrase "causally using both
branches." Lesion errors vary across training seeds and evaluation examples.
Point estimates of 20% and 10% cannot support that claim while the capability
effects use simultaneous bounds.

Define lesions exactly—zero the named group after its group normalization and
before `W_O` in both layers—and form paired seed statistics for:

- soft lesion increase in soft-bit and final XOR macro error;
- max lesion increase in max-bit and final XOR macro error;
- max-route score-shuffle increase in max-bit macro error; and
- a symmetric soft-route score-shuffle increase in soft-bit macro error,
  which is currently missing even though the claim is that both reducers use
  the shared relation.

Require simultaneous lower bounds for those effects. To avoid another
near-zero relative-effect loophole, require both the stated relative increase
and at least one percentage point absolute increase. Add all lesion statements
and the new absolute-capability statements to the family and recompute the
Bonferroni divisor and critical value; it is no longer 49.

Health rules may remain deterministic artifact kills, but must become
executable definitions:

- define "active max channel" from the variance and gradient criteria;
- name the tensor whose accumulated gradient is measured, whether gradients
  are absolute sums or Euclidean norms, checkpoint, loss, batch grouping, and
  reset schedule;
- define normalized entropy as `H(position)/log(n)` and how ties are counted;
- specify the layers, heads, slices, and aggregation for isolated `W_O` norm
  ratios; and
- apply a branch contribution threshold after separating bias-free linear
  output contributions exactly. Do not infer causal use from norm alone.

### 7. The quoted power is not the pass probability

The stated 86.6% is for one macro contrast under assumed true relative gain
35% and seed-level standard deviation 12 points. V1 must pass nine macro
contrasts, 36 slice contrasts, four non-inferiority contrasts, lesions, and
health kills simultaneously. Even if the nine macro events were independent
and each had 86.6% power, their conjunction would be only about
`0.866^9 = 27%` before the other gates. Correlation may improve or worsen this;
the current file does not model it. Near-zero control error also makes the
assumed 12-point ratio standard deviation implausible.

Required edits:

1. Relabel 86.6% as single-contrast conditional power, not design power.
2. Recompute power for the stable `G=0.8e_m-e_c` statistic over a table of
   control macro errors at 5%, 10%, 20%, and 40%, plausible paired seed
   standard deviations, and the revised family critical value.
3. Give a fixed simulation or analytic sensitivity analysis for the nine-macro
   conjunction at correlations 0, 0.5, and 0.9. State plainly if 16 seeds make
   v1 a high-false-negative screen.
4. Freeze whether an underpowered/inconclusive confidence result closes only
   this preregistered attempt or the scientific lane. It must not be described
   as evidence that the operator is ineffective unless the point estimate is
   also below the frozen practical threshold.

### 8. Total CPU workload is not yet a "cheap" test

Before leakage probes or timing, v1 schedules ten architectures, two tuning
learning rates, and sixteen evaluation seeds:

- 20 tuning training jobs;
- 160 evaluation training jobs;
- 180 total model-training jobs;
- 270,000 optimizer steps; and
- 34,560,000 training examples at batch size 128.

With two block applications per ordinary model this is 69.12 million
block-example applications when `r_time=1`, rising to 89.856 million when the
time-matched control uses `r_time=4`, before backward-cost differences. The
protocol also requests at least thousands of batch forward timings, 100,000
logistic examples, leakage MLP training, Stage 0, and artifact replays.

That may still be reasonable for width 32 on one CPU thread, but the document
does not estimate wall time or set a resource ceiling. Required edit: derive a
static worst-case forward/backward operation count after v2 controls are final,
run only the separately authorized timing calibration, then freeze a maximum
CPU core-hour and RSS budget. Exceeding it must stop v1 without silently
reducing controls, seeds, or steps. The timing calibration itself is not a
learning result and may not be used to alter the candidate.

### 9. Artifact sequencing is circular

The present sequence cannot be followed:

1. the file is hashed;
2. audits accept that hash;
3. the file is edited to say `ACCEPTED`, changing the accepted hash;
4. the pre-execution manifest is required to contain `r_time`; but
5. selecting `r_time` requires CPU execution before the source audit returns
   `RUN CPU`.

Required replacement sequence:

1. Write v2 with status `FROZEN CANDIDATE`, hash it, and never edit that file.
2. Record both audit verdicts in a separate append-only acceptance manifest
   referencing the exact v2 hash. If either requires a semantic edit, create
   v3; do not mutate v2.
3. Implement without execution. Hash every source file, the frozen mixer,
   official upstream commit/files, dependency lock/environment, and static
   architecture manifest.
4. Obtain a source audit that may authorize **TIMING CALIBRATION ONLY**.
5. Execute only the frozen timing matrix. Append raw timing artifacts and the
   mechanically selected time-matched control/hash to the manifest.
6. Obtain a second source/artifact audit returning `RUN STAGE 0`.
7. Run Stage 0. Only its complete pass authorizes generator-integrity probes;
   only their complete pass authorizes training. Each transition is an
   append-only receipt referencing all prior hashes.
8. Decode accuracy only after all 16 seeds and lesion/health artifacts are
   hashed. Any source, environment, threshold, or selected-control change
   starts a new preregistration version.

### Required disposition

Do not hash v1 as accepted, implement it, or run its timing calibration. Produce
v2 with the exact edits above and submit the complete frozen text for another
independent audit. The research direction remains admissible; the current
protocol does not.

REVISE PREREGISTRATION

## Re-audit of exact CPU preregistration v2 — 2026-08-02

Reviewed immutable candidate SHA-256:
`a40b7928078d3aa9bf8514b7a6349ab09a76477ebdf071426e5d4e92f4b049d5`.  
Scope: document, source-provenance, and arithmetic audit only; no model was
implemented, trained, timed, or evaluated.  
Decision: **v2 requires a narrowly scoped v3**.

### Verified correct

The following checks independently reproduce the frozen declarations:

- preregistration file SHA-256:
  `a40b7928078d3aa9bf8514b7a6349ab09a76477ebdf071426e5d4e92f4b049d5`;
- mixer text SHA-256:
  `f84aa219820c84314da2eede2e0e8b45bc83c881a1b01d2b1d6973d1cd8537c8`;
- parsing all 196 hexadecimal values and serializing row-major little-endian
  binary64 produces 1,568 bytes with SHA-256
  `2c1ac147b8f896e889695097fa732e2811ec8aeabe45fddc211482f08a45c973`;
- wrapped Tropical source SHA-256:
  `af11263442321708748c06cb41c1493bf8da9f15e1e6099aaa293008ed03ef51`;
- removing the first six provenance lines gives
  `5d06665382adc632d63028f7eab3b1c62e84ca183c6d0fb594813da8a19a7068`;
  and the file downloaded directly at upstream commit
  `e3c12f3e7c401245b9b5577d7181922b0150efc6` has the same hash.

The main mathematical/statistical repairs from v1 are correct:

- `G=0.8*e_control-e_candidate` has the right sign and tests
  `E[e_candidate] <= 0.8 E[e_control]` without a random denominator;
- `A=e_control-e_candidate` has the right sign for the one-point absolute gate;
- the saturation, non-inferiority, lesion, shuffle, and double-dissociation
  contrasts all use the correct bound direction;
- the listed family arithmetic is `60+12+12+3+3+12+2=104`;
- `0.05/104=0.0004807692307692308`; and
- the independently evaluated Student-t quantile is exactly
  `t(15,1-0.05/104)=4.092099746816105961...`, matching the file.

The parameter arithmetic is also correct. The standard model has 17,190
parameters; the pinned Tropical control's `64+1,216+6,976=8,256` parameters per
block make it 17,190; and the DeepSets control has 17,090. Group normalization
is now matched for the narrow, independent-score, and whole-head fatal
controls. RNG streams and self-inclusive record masking are sufficiently
specified to produce paired data without model-RNG contamination.

The workload counts reproduce: 13 architectures, 26 tuning jobs, 208 evaluation
jobs, 351,000 optimizer steps, 44,928,000 training examples, and a maximum
221,184,000 tied sublayer-example applications. The eight-core-hour wording is
actually eight **single-core** hours because affinity pins execution to one
logical CPU; v3 should retain that clearer phrase.

The single-contrast power table also reproduces by direct noncentral-t
integration: 20.58%, 49.48%, 78.42%, 94.27%, and 99.91% before rounding. The
high-false-negative interpretation is honest. `0.7842^12` is about 5.4%, so the
reported zero-correlation twelve-macro conjunction is internally consistent;
the correlated simulations are explicitly illustrative rather than guaranteed
power.

Finally, the v1 authorization circle is closed. V2 is never edited after
hashing; acceptance lives in a separate manifest; a source audit can authorize
fixtures/timing only; timing selection is appended; and a second audit is
required before Stage 0.

### Blocking issue 1 — a pass does not require correct subdecisions

V2 can pass while the auxiliary `y_s` and `y_m` classifiers remain near chance.
For example, candidate and soft controls can all have about 45% soft-bit error,
the candidate can compute XOR directly with low final error, and lesions can
move soft/max auxiliary errors from 45% to 55%. That can satisfy relative and
absolute lesions, non-inferiority, and double dissociation without showing that
the model learned either intended subdecision well.

XOR mathematically depends on the two latent bits, but a neural network trained
on the final label is not logically required to represent or recover those bits.
The auxiliary losses and causal lesions show influence, not competence. This
leaves a shortcut compatible with the positive sentence that every example
"requires both" reducers.

Required v3 edit:

- add two simultaneous candidate competence statements: the one-sided upper
  bound on candidate four-slice OOD-macro error must be at most 0.20 for `y_s`
  and at most 0.20 for `y_m`;
- include both in the primary family, making `K=106`;
- use `alpha=0.05/106=0.0004716981132075472` and
  `tcrit=4.1014945453569363208` at 15 degrees of freedom; and
- update the descriptive power table. At the same standardized distances its
  values become approximately 20.4%, 49.2%, 78.2%, 94.2%, and 99.9%.

If 20% subdecision error is not the intended competence threshold, v3 must
freeze another threshold before execution. Omitting an absolute competence
gate is not admissible.

### Blocking issue 2 — block-two route shuffling has two meanings

The route shuffle is exact for block one but ambiguous for block two. A block-one
final-query shuffle changes the query residual. Consequently block two's query,
and therefore its shared QK score row, differs from the intact forward pass.
V2 does not say whether block two receives:

1. a donor row stored from each donor's intact forward pass; or
2. a donor row computed after all receivers have already undergone the block-one
   perturbation.

Those interventions are not equivalent. They can produce materially different
lesion effects and double-dissociation results. "The other branch stays paired"
also needs to mean paired with the receiver's current perturbed hidden state,
not silently restored from the intact pass.

Required v3 edit: define the perturbation recursively. A complete unambiguous
choice is:

1. compute block-one scores from intact incoming states;
2. feed `S1[pi(b)]` only to the named branch of receiver `b`, while the other
   branch uses `S1[b]`; update receiver hidden states normally;
3. compute block-two scores for every example from those now-perturbed incoming
   hidden states;
4. feed `S2[pi(b)]` only to the named branch and `S2[b]` to the other branch;
5. retain receiver values, masks, labels, and record ordering at both blocks;
   and use the same stored derangement `pi` unless a separately seeded
   block-specific mapping is explicitly frozen.

The fixture must store enough intermediate hashes to prove this sequence. Using
intact donor block-two rows is also testable, but it is a different frozen
intervention and must be stated if chosen.

### Blocking issue 3 — timing text permits different calibrations

"In a fresh process per variant" and "1,000 measured forwards in 20
round-robin blocks" do not define one schedule. It could mean one persistent
process that runs all 20 blocks, one new process per block, or sequential
complete-model runs in a rotated variant order. The text also does not state
whether 200/1,000 counts apply separately to every length or are divided across
lengths, and it freezes the input seed but not the untrained model initialization
seed. These choices can change p50/p95 eligibility near the 5% boundary and
therefore select a different primary control.

Required v3 edit: replace the timing paragraph with one executable schedule.
For example:

- model initialization seed is `843001` for every timing variant;
- for each of 20 rounds and each variant in a frozen round-rotated order,
  launch a fresh affinity-pinned process;
- for **each** length 9, 17, and 33 in a frozen order, perform 10 unrecorded
  warmups followed by 50 recorded forwards, giving exactly 200 warmups and
  1,000 measurements per variant per length;
- pool the 1,000 measurements by `(variant,length)` before p50/p95/MAD and the
  frozen weighted calculation;
- record absolute `VmHWM` without implying that Linux high-water RSS can be
  reset or baseline-subtracted; and
- state that fixture generation, calibration, process-launch overhead if
  charged, training, evaluation, perturbations, and reporting timings either
  are or are not inside the eight single-core-hour ceiling. Freeze one timer
  start/stop definition.

Another fully specified schedule is acceptable, but the choice cannot be left
to implementation after results-sensitive 5% eligibility is frozen.

### Non-blocking clarifications worth carrying into v3

- A failed lower confidence bound for the 5% control-error floor is more
  precisely `CONTROL FLOOR NOT ESTABLISHED` than proof of literal task
  saturation. The conservative no-success consequence is correct.
- Define the `CONTINUE/STOP` decoder output for required-control numerical
  failure as `PROTOCOL INVALID`, not evidence that the candidate lost.
- Bind the precomputed power-sensitivity script or canonical output hashes in
  the manifest. These calculations do not affect acceptance, but this would
  make every frozen number provenance-complete.

### Disposition

Do not accept or implement SHA
`a40b7928078d3aa9bf8514b7a6349ab09a76477ebdf071426e5d4e92f4b049d5`.
Create v3 changing only the subdecision competence family, recursive route
shuffle, timing schedule, resulting `K/tcrit/power` numbers, and the listed
clarifications. The algebra, generator, controls, bounded capability statistic,
resource ceiling, and authorization sequence otherwise survive re-audit.

REVISE V2

## Audit of effective CPU preregistration v3 — 2026-08-02

Effective immutable protocol, in precedence order:

1. v2 SHA-256
   `a40b7928078d3aa9bf8514b7a6349ab09a76477ebdf071426e5d4e92f4b049d5`;
2. v3 overlay SHA-256
   `b10c376e386c916bfb25bc11a0af2ef3f2433439b99d98c5e5515af2047b6bbf`.

Scope: protocol and arithmetic audit only. No implementation, model forward,
training, fixture generation, or timing calibration was performed.

### Verification

Both file hashes reproduce exactly. The v3 overlay states a complete precedence
rule: every v2 clause remains normative unless a named v3 section replaces it,
and v3 wins on conflict. Acceptance binds the two hashes in order without
editing either file. The overlay's replacement references are sufficiently
specific to construct one effective document.

The two competence statements close the direct-XOR loophole. Candidate OOD
macro upper bounds for both `y_s` and `y_m` must be at most 0.20, so a result
cannot pass merely by predicting final XOR while leaving both auxiliary
subdecisions near chance. The revised family arithmetic is correct:

`60 + 12 + 12 + 3 + 3 + 12 + 2 + 2 = 106`.

The declared one-sided alpha and critical value also reproduce:

\[
\alpha'=0.05/106=0.0004716981132075472,
\qquad
t_{15,1-\alpha'}=4.1014945453569363208.
\]

The updated single-contrast sensitivities are consistent with that critical
value. The zero-correlation twelve-macro conjunction is approximately
`0.7818^12=5.2%`; the other correlation figures remain explicitly illustrative.
The interpretation is honest: this is a high-false-negative reject-only screen,
not an 80%-powered proof of the full conjunction. A failed interval with a
point effect at or above 20% is inconclusive and cannot authorize GPU work.

The recursive route-shuffle now has one meaning. Block one uses a full-slice
cached donor score row only in the named branch; block two first recomputes all
score rows from the recursively perturbed hidden states and only then applies
the same full-slice derangement. Receiver values, masks, order, labels, and the
other branch remain receiver-paired. Full-cache hashing before donor lookup
prevents a minibatch-local or intact-block-two alternative.

The timing schedule is deterministic and results-independent:

- fixed input and model seeds;
- 17 closed variant IDs;
- 20 rotated variant rounds in fresh, non-overlapping pinned workers;
- a frozen rotated order over all three lengths;
- exactly 200 warmups and 1,000 timed calls per `(variant,length)`;
- one explicit complete-forward callable boundary; and
- absolute `VmHWM`, a single frozen whole-matrix MAD retry, and mechanically
  selected eligibility under the unchanged 95--105% rules.

The resource ledger now has closed boundaries. It charges summed active
monotonic wall time from first experiment operation through final fsync/hash,
all workers are sequential, every experimental stage is included, and the
only exclusions are pre-execution design/audit work and idle gaps. Crossing
eight single-core hours, 2 GiB per-process `VmHWM`, or 2 GiB of new artifacts
can only produce `RESOURCE-INCONCLUSIVE`; it cannot weaken the protocol.

Probe reproducibility is closed by the frozen 100,000-example quadrant-balanced
fixture, explicit continuing PCG64 stream, batch-boundary cursor rule, and
per-tensor initialization. Initialization is independent of module construction
order; official-Tropical normal and Kaiming initializers are separately frozen.
One canonical-path-sorted AdamW parameter group, explicit decay behavior,
disabled fused/foreach variants, exact zero-grad/backward/clip/step order, and
the probe no-clip exception remove optimizer degrees of freedom. The static
source audit must, as already required, hash the canonical tensor-path table
before any execution authorization.

Failure and replay behavior is also closed. The only repeat is the numeric
whole-timing-matrix MAD retry. Every other infrastructure, semantic, numeric,
process, or required-control failure is `PROTOCOL INVALID`, never a scored
candidate win or loss. The pre-Stage-0 decoder is immutable and can return only
the five declared status words without revealing metrics or rankings.

Finally, the authorization chain is acyclic: two audits accept the ordered hash
pair in a separate manifest; implementation is hashed without execution; a
source audit may authorize fixtures/timing only; timing artifacts and the
selected control are appended; a second audit may authorize Stage 0; and every
later transition requires a hash-bound receipt. No effective-v3 outcome
authorizes GPU or rental work.

### Decision

No remaining result-dependent architecture, data, statistic, perturbation,
timing, replay, resource, or decoder choice was found in the effective ordered
pair. Acceptance is limited to the preregistration text. The independent source
and artifact audits required by the protocol remain mandatory before any
execution.

ACCEPT EFFECTIVE V3

## Audit of effective CPU preregistration v4 — 2026-08-02

Effective immutable protocol, in precedence order:

1. v2 SHA-256
   `a40b7928078d3aa9bf8514b7a6349ab09a76477ebdf071426e5d4e92f4b049d5`;
2. v3 SHA-256
   `b10c376e386c916bfb25bc11a0af2ef3f2433439b99d98c5e5515af2047b6bbf`;
3. v4 SHA-256
   `085abc5e703ee4a064a837f66cf3ffb82e282c1fe78fec12cc53870a0db774b8`.

Scope: ordered-document and arithmetic audit only. No fixture, model,
calibration, forward pass, optimizer step, or decoder was implemented or run.

### Hashes and precedence

All three hashes reproduce exactly. V4 supplies an unambiguous precedence rule:
v2 is normative, then v3, then v4 on conflict; none is edited. Each v4 section
names the effective clause it replaces or completes. No v4 clause changes the
scientific claim, the bounded capability estimand, the 106-statement family,
the critical value, the controls, or the high-false-negative interpretation
accepted in the v3 audit.

V4.1 is scoped to learned-model batches. The deterministic Stage-0
padding/mask identity check remains an operator-only v2 test; it does not place
mixed-length examples in a learned-model batch.

### Homogeneous batching and RNG

Training now draws one `n` before any example fields for a step, conditions all
quadrant collection on that `n`, and removes per-example length draws. This
produces one dense `128 x (n+1) x 14` tensor without padding and preserves iid
uniform length across steps. Because the already frozen per-step PCG64 stream
is shared, every architecture receives byte-identical length and examples.

Development and probe data are exactly stratified into five fixed length
buckets. Their totals and quadrants reproduce the previous contracts:

- development: `5 * 2,000 = 10,000`, with 500 examples per quadrant per
  length; and
- probes: `5 * 20,000 = 100,000`, with 5,000 examples per quadrant per
  length, hence 25,000 of each quadrant overall.

Only fixed-size extracted probe features may be concatenated across buckets;
token models never mix lengths. Main ID/OOD slices were already fixed-length.
No label, support, macro weight, or evaluation count changes.

### Restored time-control operator

The tied control now has one exact recurrence. Within each of two
parameter-untied blocks it performs `r_attention` complete pre-norm attention
residual updates, then `r_ffn` complete pre-norm FFN residual updates. Q/K/V,
scores, reductions, group RMS, and output are recomputed on every repetition;
weights alone are tied. The two blocks never share parameters. This matches
the v2 parameter and timing claims and introduces no hidden state or extra
parameter.

### Parameter paths and initialization

V4 enumerates every trainable tensor for all thirteen architectures and every
probe. The lists agree with the accepted parameter arithmetic:

- ordinary Transformer-like models have input, two eight-tensor blocks, final
  norm, and six readout tensors;
- the official Tropical model replaces only each block's four attention
  tensors with its three max-plus matrices and output matrix;
- DeepSets and all probes have complete separate grammars; and
- non-affine group RMS contributes no parameter.

The architecture-qualified strings also distinguish every selected tied-time
variant. They fully determine V3's per-tensor SHA-256 seed derivation and the
canonical optimizer order. Requiring the initial source audit to compare names
and shapes against this grammar prevents implementation-time additions or
renaming before execution.

### Exact derangement and recursive perturbation

The descending Sattolo loop draws `j` from the exclusive range `[0,i)`, swaps
`pi[i]` and `pi[j]`, and therefore constructs a single-cycle derangement for
10,000 examples. The frozen assertions establish bijection and no fixed point;
the direction `receiver b -> donor pi[b]` is explicit. The same stored map is
then used in both blocks under V3's recursively perturbed full-slice cache
semantics. No minibatch-local, inverse-map, intact-block-two, or independently
redrawn interpretation remains.

### Decoder, ledger, replay, and authorization

Decoder source, exhaustive word-mapping tests, and hash now precede the first
source decision and therefore precede fixture/timing execution. The same
immutable five-word decoder handles timing selection and every later receipt.
This closes the former gap in which timing could occur before decoder review.

Resource time is counted exactly once at top-level controller boundaries;
sequential child-worker lifetimes cannot be double-counted. The ledger includes
all experimental operations through final fsync/hash while excluding only
audit/design work and idle gaps. A single append-only run directory, physical
`st_blocks*512` accounting, absolute `VmHWM`, no deletion/truncation/linking or
outside outputs, and an invalid—not replayable—outcome for an incomplete
controller close the disk/RSS/recovery loopholes. V3's numeric ceilings and
sole whole-matrix MAD retry remain unchanged.

The authorization graph remains acyclic: both auditors first accept the exact
ordered triple in a separate manifest; implementation and decoder are then
hashed without experiment execution; the first source audit may authorize only
fixtures/timing; the selected timing control is appended; and a second audit is
required before Stage 0. No CPU learning stage, GPU, or rental is authorized by
this document audit.

### Statistical non-regression

V4 does not override V3.6. The two `<=0.20` candidate subdecision competence
bounds remain primary. The family remains

`60 + 12 + 12 + 3 + 3 + 12 + 2 + 2 = 106`,

with one-sided `alpha=0.05/106` and
`tcrit=4.1014945453569363208` at 15 degrees of freedom. The stable
`G=0.8e_control-e_candidate`, one-point absolute gains, 5% control floors,
non-inferiority, perturbation multiplicity, double dissociation, and
high-false-negative/no-GPU interpretation all survive unchanged.

### Decision

No result-dependent batching, operator, initialization, derangement, decoder,
resource, replay, precedence, or statistical choice remains in the effective
ordered triple. Acceptance is limited to this preregistration. Both audit
receipts and the separately required source/artifact decisions remain mandatory
before any execution.

ACCEPT EFFECTIVE V4
