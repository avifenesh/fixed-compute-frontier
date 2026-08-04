# T84 shared-score heterogeneous attention — CPU preregistration v1

Date frozen: 2026-08-02  
Status: **DRAFT FOR INDEPENDENT AUDIT; EXECUTION NOT AUTHORIZED**

This preregistration covers only the deterministic operator checks and the
single-row learned-relation falsifier described below. It does not authorize
the weighted-automaton stage, language-model integration, GPU work, or a rental.
Passing v1 is reject-only evidence and earns a separately preregistered causal
weighted-automaton test.

The candidate screen and two independent audits are:

- [revised T84 screen](product-semiring-attention-t84-paper-screen.md)
- [independent audit A](product-semiring-attention-t84-independent-audit-a.md)
- [independent audit B](product-semiring-attention-t84-independent-audit-b.md)

## 1. Frozen claim and possible observation

For each attention head, compute one causal relation score

\[
S_{ij}=Q_iK_j^T/\sqrt{d_h}.
\]

Partition its value channels into equal soft and max groups:

\[
O^{\rm soft}_{ic}=\sum_{j\in J_i}\operatorname{softmax}_j(S_{ij})V^{\rm soft}_{jc},
\qquad
O^{\rm max}_{ic}=\max_{j\in J_i}(S_{ij}+V^{\rm max}_{jc}).
\]

The narrow claim is that reusing a rank-`d_h` relation score for a
channel-dependent max reduction is a materially better allocation of fixed
width than forcing every value channel to share one probability distribution.

The only possible positive observation in v1 is:

> On one same-instance task in which every example requires both a signed
> softmax-weighted comparison and a channel-specific max-plus comparison, the
> candidate has at least 20% lower final-decision error than **every** frozen
> parameter- or CPU-time-matched learned control, with a simultaneous lower
> confidence bound reaching 20%, while preserving the soft subdecision and
> causally using both branches.

Even that observation is an operator-learning result, not a major model-stage
success and not a smarter-model claim.

## 2. Immutable environment and numeric rules

- Device: CPU only. CUDA tensors and GPU kernels are forbidden.
- Framework: the locally installed PyTorch version recorded in the run
  manifest. No dependency may be installed after this file is frozen.
- Dtype: IEEE float32 for models; float64 for reference operator calculations
  and metric aggregation.
- Determinism: `torch.use_deterministic_algorithms(True)`, one intra-op thread,
  one inter-op thread, and explicit Python/NumPy/PyTorch seeds.
- Causal masking: every record precedes one final query token; masked logits are
  negative infinity; every query has at least one legal source.
- Group normalization: for each query, head, and frozen four-channel group,
  divide by `sqrt(mean(x^2) + 1e-6)`. It is non-affine and has no running state.
- Exact-max ties use PyTorch's native first-index subgradient. Generator values
  are continuous, so exact ties have probability zero before float rounding.
- Any NaN, infinity outside a legal mask, nondeterministic replay, or reference
  mismatch is a hard failure, not a tuning opportunity.

The implementation may contain only this candidate, the controls below, data
generation, metrics, and artifact capture. Any semantic change creates `v2` and
requires a new audit.

## 3. Stage 0 — deterministic operator checks

Use reference seed `840000` and 1,024 float64 examples at each legal-source
count `n in {1,4,8,16,32}` with `r=4`, two soft coordinates, and two max
coordinates. Draw `q`, `k`, signed soft values, and real max potentials iid
uniform on `[-2,2]`. Pad to 32 records and apply the causal mask.

Required checks:

1. candidate soft output equals direct float64 softmax-weighted values;
2. candidate max output equals direct `max(score + potential)` per channel;
3. padding/masking does not change either output;
4. the finite-`beta` implementation at `beta=8` obeys
   `0 <= B_beta-B_inf <= log(n)/8 + 1e-12`;
5. away from ties, max-input gradients equal the analytic winner indicator;
6. the counterexample `S=[10,0]`, `V=[0,100]^T` returns soft hard-limit `0`
   and max-plus value `100`; and
7. bitwise repeat of the complete Stage-0 artifact is identical.

Float64 forward tolerance is `1e-10` absolute and relative. Float32 candidate
versus float64 reference tolerance is `2e-5` absolute and relative. Every check
must pass before training code may execute.

## 4. Stage 1 generator — one example needs both reducers

### 4.1 Latent fields

For each example, draw:

- query `q in R^4`;
- record keys `k_j in R^4`;
- signed evidence values `v_j in R^2`; and
- max-plus potentials `u_j in R^2`.

All coordinates are iid uniform on `[-a,a]`. Define

\[
s_j=q^Tk_j/2,
\qquad
A=\sum_j\operatorname{softmax}_j(s)v_j,
\qquad
B_c=\max_j(s_j+u_{jc}).
\]

The independently symmetric subdecision bits and final target are

\[
y_s=\mathbf{1}[A_0>A_1],
\qquad
y_m=\mathbf{1}[B_0>B_1],
\qquad
y=y_s\mathbin{\mathrm{XOR}}y_m.
\]

Conditional on the shared scores, the two comparisons are independent and each
is balanced by exchangeability of its iid value channels. Every generated batch
is additionally rejection-balanced to contain exactly 25% of each
`(y_s,y_m)` pair. With batch size 128, every quadrant contains 32 examples.

### 4.2 Token representation

A record token before mixing has 14 fields:

`[record_bit, query_bit, k(4), v(2), u(2), q_slot(4)]`.

A query token is `[0,1,zeros(8),q(4)]`; a record token has a zero `q_slot`.
A fixed 14-by-14 orthogonal matrix generated by QR decomposition from NumPy
PCG64 seed `840014` mixes all fields before the trainable input projection.
The matrix is shared by every architecture, training seed, and split and is
stored in the run artifact. Record order is freshly permuted for every example.
There are no position embeddings or graph masks.

### 4.3 Frozen supports

- Train: `n` uniform over integers 4 through 8, `a=1`.
- Development validation: same support, disjoint seeds.
- ID integrity test: `n=8`, `a=1`.
- OOD-L16: `n=16`, `a=1`.
- OOD-L32: `n=32`, `a=1`.
- OOD-V2: `n=8`, `a=2`.
- OOD-L32V2: `n=32`, `a=2`.

Each fixed evaluation slice contains 10,000 exactly quadrant-balanced examples.
The four OOD slices, excluding ID, form the primary macro. The ID slice is an
integrity/protected slice. Evaluation data seed is `841000 + slice_index` and
is identical for every architecture and training seed.

### 4.4 Pre-training leakage and balance kills

Before model training, each split must have exactly 25% of every `(y_s,y_m)`
quadrant and 50% of `y`. Train a logistic regression on 100,000 train examples
using only these surface features: sequence length, per-field means, standard
deviations, minima, maxima, L1/L2 norms, first record, last record, and the sum
of record tokens. On each evaluation slice, its accuracy for `y_s`, `y_m`, and
`y` must be at most 55%. A direct query-only or one-record-only two-layer MLP
with hidden width 64, trained for the same 1,500 steps, must be at most 55% on
`y`. Failure invalidates the generator and forbids model training; thresholds
may not be repaired in v1.

## 5. Frozen shared backbone

Except where a control explicitly changes the attention reducer or head count,
all models use:

- trainable input projection `14 -> 32`;
- hidden width 32;
- two pre-norm causal Transformer blocks;
- four heads of dimension 8;
- bias-free Q/K/V/O attention projections;
- RMSNorm before attention and before the FFN, epsilon `1e-6`;
- FFN `32 -> 64 -> 32` with GELU and no bias;
- no dropout, no positional encoding, and no weight tying across the two blocks;
- final RMSNorm and one query-token readout with three independent two-logit
  classifiers for `y_s`, `y_m`, and `y`;
- loss `0.5*CE(y_s) + 0.5*CE(y_m) + CE(y)`; and
- Xavier-uniform linear initialization with gain 1 and zero readout biases.

The candidate splits each eight-channel head into four signed-soft and four
real max-plus channels and applies the frozen four-channel group normalization
to both groups before concatenation and `W_O`.

## 6. Frozen controls

All controls keep width 32, two blocks, FFN width 64, token format, readout,
loss, data, and training budget unless stated.

1. `soft_standard`: four ordinary softmax heads, no post-attention group norm.
2. `soft_grouped`: four softmax heads; split each value head 4+4 and apply the
   candidate's group norm separately to both soft groups.
3. `soft_narrow`: 32 ordinary one-dimensional softmax heads, no group norm.
4. `hybrid_independent`: eight dimension-4 heads; four complete soft heads and
   four complete max heads, giving independent score maps at fixed total QKV
   width.
5. `hybrid_whole_head`: four dimension-8 heads; two complete soft heads and two
   complete max heads.
6. `hard_shared`: four dimension-8 heads; each chooses `argmax_j S_ij` once and
   copies that record's full value vector, then uses the same 4+4 group norm.
7. `finite_beta_8`: the candidate layout, but max channels use
   `logsumexp(8*(S+V))/8` during both training and inference.
8. `tropical_official`: the official Tropical Attention v2.0.0 equations:
   `log1p(ReLU(x))`, shared per-head-dimension max-plus Q/K/V projections,
   symmetric tropical Hilbert score, channelwise max-plus context, `expm1`, and
   linear output. It receives the same residual backbone and readout. Any
   parameter shortfall is spent by choosing the largest FFN hidden width not
   exceeding the candidate's total active parameter count; the remaining
   difference must be below 2% and is reported.
9. `soft_time_matched`: `soft_grouped` with each attention+FFN block applied a
   tied integer number `r_time` of times. Before tuning or training, benchmark
   untrained candidate and `r in {1,2,3,4}` on the frozen CPU timing protocol;
   choose the largest `r` whose p50 is no more than 105% of candidate p50. If no
   repeated version qualifies, use `r_time=1`.

The candidate must beat all nine. The strongest control is never selected after
the result; all contrasts are primary. `tropical_official` is a source-faithful
control, not a claim that its upstream runtime numbers are valid.

## 7. Training and tuning budget

- Optimizer: AdamW, betas `(0.9,0.95)`, epsilon `1e-8`, weight decay `0.01`.
- Candidate learning rates: exactly `{3e-4, 1e-3}` for every architecture.
- Development tuning: seed `842000`, 1,500 fixed steps per learning rate and
  architecture. Choose the rate with lower mean final-decision error across
  10,000 ID and 10,000 OOD-L16 validation examples; exact ties choose `3e-4`.
- Evaluation training seeds: integers `842100` through `842115` inclusive.
- Each evaluation run: 1,500 optimizer steps, batch size 128, fresh exactly
  quadrant-balanced batches, gradient-norm clip 1.0, no scheduler, no early
  stopping, and no checkpoint selection. The final step is evaluated.
- No architecture receives an additional learning rate, warmup, annealing,
  straight-through estimator, initialization, loss, or step budget after any
  accuracy is observed. Such variants require v2 and are not rescue runs.

Training examples, validation examples, optimizer steps, tuning candidates,
failed seeds, wall time, CPU model, PyTorch build, parameter count, and peak RSS
are recorded for every architecture. A crash is a worst-score run, not silently
rerun, unless the full deterministic run reproduces the same infrastructure
failure before reading any metric.

## 8. CPU timing and real-cost control

Set PyTorch intra/inter-op threads to one. At batch 128 and sequence length 9,
run 200 warmups and 1,000 timed no-grad forward passes per model in 20
round-robin blocks. Record p50, p95, mean, standard deviation, and peak RSS.
The untrained timing is used only to freeze `r_time`; trained timing is reported
but cannot establish GPU production cost. Timing variance is acceptable only
when the median absolute deviation divided by p50 is at most 5%; otherwise the
timing calibration is repeated once from a fresh process before metrics are
read. A second failure invalidates time matching and forbids training.

No CPU speed result qualifies as a model success. A later GPU stage, if ever
earned, gives measured overhead to an all-softmax model as extra useful compute.

## 9. Frozen metrics, multiplicity, and power

For seed `s`, model `m`, and primary OOD slice `d`, let `e_smd` be final XOR
classification error. Define the per-seed macro as the unweighted mean over the
four OOD slices. For each of the nine controls, compute candidate relative error
reduction

\[
R_{sdm}=1-e_{s,\mathrm{candidate},d}/e_{s,m,d}.
\]

If a control error is zero, the contrast fails; if both errors are zero, set
relative improvement to zero. The primary estimate is the arithmetic mean of
the 16 seed-level ratios. Also report pooled error, absolute percentage-point
change, and headroom consumed; they cannot replace the frozen ratio.

Use a one-sided paired Student-t lower bound across the 16 training seeds. The
family contains 49 planned statements:

- 45 capability contrasts: nine controls times the macro plus four OOD slices;
- two soft-bit non-inferiority contrasts, candidate versus `soft_standard` and
  `soft_grouped`; and
- two ID final-error non-inferiority contrasts against the same controls.

Bonferroni familywise alpha is `0.05`, so every one-sided bound uses
`alpha'=0.05/49`, 15 degrees of freedom, and critical value
`t=3.7229661737`. No normal or example-level pseudo-replication replaces seeds.

The design has about 86.6% one-sided power for the macro claim if the true
per-seed relative improvement is 35%, its standard deviation is 12 percentage
points, and the relevant critical value is 3.72297. If observed variance is
larger, the test remains frozen and may be inconclusive; seeds are not added.

## 10. Pass, stop, and branch-health rules

T84 v1 passes only if all conditions hold:

1. Stage 0 and every generator integrity check pass.
2. Against **each** of nine controls, the simultaneous lower bound for macro
   relative error reduction is at least 20%.
3. Against every control on every OOD slice, the simultaneous lower bound is
   nonnegative.
4. On soft-bit error, the simultaneous upper bound for candidate minus both
   soft controls is at most 1 absolute percentage point.
5. On ID final error, the corresponding upper bound is at most 1 point.
6. Candidate max-branch and soft-branch lesions each increase their matching
   subdecision error by at least 20% relative **and** final XOR error by at least
   10% relative on the primary macro.
7. Shuffling the max branch's shared score rows across examples increases the
   max-subdecision error by at least 20% relative.
8. Before `W_O`, at least 75% of candidate max channels have output standard
   deviation above `1e-3` and accumulated gradient norm above `1e-8` over the
   first 1,024 ID examples.
9. For every active max channel on every slice, normalized winner-position
   entropy is at least 0.85, and every legal record position wins at least once.
10. Each branch's isolated contribution through `W_O` has mean query-token L2
    norm at least 10% of the unlesioned attention output norm.

Branch-health rules are deterministic integrity kills, not post-hoc statistical
claims and are not used to enlarge the 49-test family. Any failed condition
closes v1. A 10--under-20% point estimate is recorded as arguable but does not
earn another seed or GPU. A single-digit result closes the lane immediately.

There is no early futility look at evaluation metrics. All 16 seeds run unless
Stage 0, generator integrity, determinism, numerical validity, or branch health
hard-fails before accuracy is opened. Results are decoded only after all frozen
artifacts and hashes exist.

## 11. Artifact and immutability contract

Before implementation:

1. record the SHA-256 of this file in a sidecar;
2. obtain two independent preregistration audits;
3. mark this file `ACCEPTED` only if both audits admit the same protocol.

After implementation but before execution, capture a manifest containing the
preregistration hash, source hashes, environment, CPU identity, all parameter
counts, the source revision of the official tropical equations, timing-selected
`r_time`, and a dry structural diff showing that every frozen architecture and
threshold is present. An independent source audit must return `RUN CPU`.

Any edit to an operator, generator, support, metric, threshold, control,
training budget, timing rule, or confidence procedure invalidates the hash and
requires `v2`. Result files are append-only. No GPU or rented instance is
authorized by v1 under any outcome.
