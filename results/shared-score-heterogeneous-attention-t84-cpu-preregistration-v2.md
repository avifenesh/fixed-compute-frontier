# T84 shared-score heterogeneous attention — CPU preregistration v2

Date written: 2026-08-02  
Status: **FROZEN CANDIDATE FOR INDEPENDENT AUDIT; NO EXECUTION AUTHORIZED**

This file replaces v1 prospectively. V1 was never accepted, implemented, or
run. This file covers a deterministic operator check and one tiny learned
same-relation falsifier. It does not authorize a weighted-automaton test, a
language model, GPU work, or a rental. A v2 pass is reject-only evidence that
can earn a separately preregistered causal model stage; it is not a 20% major
model-stage success and not a smarter-model claim.

The exact claim is deliberately narrow:

> On the frozen task below, a feature-group split that reuses one learned QK
> score for ordinary softmax and channelwise max-plus reductions has at least
> 20% lower expected final-decision error and at least one percentage point
> lower absolute error than every named parameter- or CPU-time-matched control,
> under simultaneous seed-level bounds, without saturating the task, losing the
> soft subproblem, or bypassing either branch.

No v2 result may be generalized to every same-cost learned alternative. The
positive sentence applies only to the twelve controls frozen in section 7.

## 1. Frozen runtime and external sources

The only allowed Python is:

`runtime/cpkv-topk-001-venv/bin/python`

The frozen environment is Python 3.14.4, NumPy 2.3.5, PyTorch 2.13.0+cpu, and
mpmath 1.3.0 on x86-64. No package may be installed or upgraded. Models use CPU
float32; operator references and metric aggregation use float64. CUDA tensors,
GPU kernels, `torch.compile`, TF32, and mixed precision are forbidden.

PyTorch intra-op and inter-op thread counts are one. Deterministic algorithms
are enabled. Training and timing are pinned with `taskset` to logical CPU 16 of
the local Intel Core Ultra 9 275HX. `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.
The CPU governor, microcode, kernel, BLAS configuration, and complete package
versions are captured before any authorized execution. A mismatch creates a
new preregistration version.

The official Tropical Attention reference is frozen to upstream commit
`e3c12f3e7c401245b9b5577d7181922b0150efc6`. Its original
`TropicalAttention.py` SHA-256 is
`5d06665382adc632d63028f7eab3b1c62e84ca183c6d0fb594813da8a19a7068`.
The local provenance-wrapped snapshot is
`references/tropical-attention-e3c12f3e-TropicalAttention.py`, SHA-256
`af11263442321708748c06cb41c1493bf8da9f15e1e6099aaa293008ed03ef51`;
removing its first six provenance lines reproduces the original hash.

The fixed orthogonal mixer is stored as hexadecimal binary64 in
`results/shared-score-heterogeneous-attention-t84-mixer-v2.txt`, file SHA-256
`f84aa219820c84314da2eede2e0e8b45bc83c881a1b01d2b1d6973d1cd8537c8`.
Its canonical row-major little-endian float64 bytes have SHA-256
`2c1ac147b8f896e889695097fa732e2811ec8aeabe45fddc211482f08a45c973`.

## 2. Candidate operator and exact mask

For head hidden dimension `d_h`, one score is computed once:

\[
S_{ij}=Q_iK_j^T/\sqrt{d_h}.
\]

Four value coordinates use ordinary signed attention and four use real max-plus
potentials:

\[
O^s_{ic}=\sum_{j\in J_i}\operatorname{softmax}_j(S_{ij})V^s_{jc},
\qquad
O^m_{ic}=\max_{j\in J_i}(S_{ij}+V^m_{jc}).
\]

Tokens are `[record_0,...,record_(n-1),query]`. Record row `i<n` may attend
exactly to `J_i={0,...,i}`. The final query row `i=n` may attend exactly to
`J_n={0,...,n-1}`: its self-edge is illegal. Thus every row has a legal source,
and the target and query attention both range over records only. The same mask
is used in both blocks, every control, every perturbation, and the references.
Illegal scores are IEEE negative infinity before the reducer. Record outputs
matter because block two can read contextualized records; only the final query
is classified.

For every token and head, each predetermined four-channel group is divided by

\[
\sqrt{\operatorname{mean}_{c=1}^4 x_c^2+10^{-6}}.
\]

This group RMS operation is non-affine and stateless. It occurs after reduction
and before concatenation/`W_O`. The candidate applies it independently to its
soft and max groups.

## 3. Stage 0 deterministic identities

Seed `840000` generates 1,024 cases for each legal-source count
`n in {1,4,8,16,32}` with score rank four, two signed-soft coordinates, and two
max coordinates. Draw `q,k,v,u` iid uniform `[-2,2]`, pad to 32 records, and use
the exact mask above. Run every case in float64 reference mode and float32
candidate mode.

The following all must pass:

1. soft output equals a direct max-shifted float64 softmax reference;
2. max output equals direct channelwise `max(S+u)`;
3. changing padded values cannot change either output;
4. stable `B_8=max(x)+log(sum(exp(8*(x-max(x)))))/8` obeys
   `-1e-12 <= B_8-B_inf <= log(n)/8+1e-12`;
5. gradient cases are resampled until every channel's top-two `S+u` gap is at
   least `1e-6`; with scalar loss equal to the sum of all max outputs,
   `dL/du_jc` is the winner indicator and `dL/dS_j` is the number of channels
   won by source `j`;
6. the explicit argmax-copy reference for `S=[10,0]`, `V=[0,100]^T` returns
   zero, while direct max-plus returns 100; and
7. a second fresh process produces a bitwise-identical canonical tensor payload.

The replay payload is an ordered little-endian tensor bundle containing seeds,
`q,k,v,u`, masks, outputs, winner indices, and gradients only. It excludes
timestamps, paths, environment strings, and unordered metadata. Float64 forward
absolute and relative tolerance is `1e-10`; float32 versus float64 is `2e-5`.
Any NaN, illegal infinity, nondeterminism, or mismatch stops v2.

## 4. Frozen learned-relation generator

### 4.1 Latents and labels

For each example draw `q in R^4`, record keys `k_j in R^4`, signed values
`v_j in R^2`, and max potentials `u_j in R^2`, iid uniform `[-a,a]`. Define

\[
s_j=q^Tk_j/2,\quad
A=\sum_j\operatorname{softmax}(s)_jv_j,\quad
B_c=\max_j(s_j+u_{jc}),
\]

\[
y_s=1[A_0>A_1],\qquad y_m=1[B_0>B_1],\qquad y=y_s\operatorname{XOR}y_m.
\]

Conditional on `q,k`, the two bits are independent fair bits by exchangeability
of the independent value-channel pairs. Each batch of 128 nevertheless contains
exactly 32 examples from each `(y_s,y_m)` quadrant. Candidates are drawn in
stream order until each bin is full, concatenated in quadrant order
`(0,0),(0,1),(1,0),(1,1)`, then permuted once by the batch data stream. This
selection does not alter the uniform `n` distribution because each bit is fair
conditional on `q,k,n`.

### 4.2 Tokens and frozen mixer

The raw record is

`[record_bit=1, query_bit=0, k(4), v(2), u(2), q_slot=zeros(4)]`.

The raw query is

`[record_bit=0, query_bit=1, zeros(8), q_slot=q(4)]`.

Every raw row right-multiplies the frozen 14-by-14 matrix from section 1 before
the trainable model sees it. Record order is independently permuted once per
example after latent generation and before mixing. There are no position
embeddings, task labels, graph masks, padding tokens, or privileged target
fields.

The mixer construction, retained only as a reproducibility check, is NumPy
PCG64 seed `840014`, `standard_normal((14,14), dtype=float64)`, `np.linalg.qr`,
then multiply Q column `c` by -1 iff `R[c,c]<0`. The stored hexadecimal matrix,
not a fresh QR call, is authoritative.

### 4.3 Supports and exact fixed datasets

- train and in-support development: `n` uniform over integers 4 through 8,
  `a=1`;
- ID integrity: `n=8,a=1`;
- OOD-L16: `n=16,a=1`;
- OOD-L32: `n=32,a=1`;
- OOD-V2: `n=8,a=2`;
- OOD-L32V2: `n=32,a=2`.

The primary macro is the unweighted mean of the four OOD slices. ID is
protected, not primary. Each fixed slice has 10,000 examples, exactly 2,500 per
quadrant. Slice order and seeds are ID/`841000`, L16/`841001`, L32/`841002`,
V2/`841003`, and L32V2/`841004`. Development contains 10,000 in-support
examples from seed `841100`. No OOD slice participates in tuning.

For fixed datasets, draw examples sequentially, keep the first 2,500 in each
quadrant, concatenate in the quadrant order above, then apply one final stored
dataset permutation. Fixture generation emits raw and mixed little-endian
float32 tensors, float64 latent/target references, and SHA-256 hashes before any
model training.

### 4.4 RNG and pairing contract

NumPy PCG64 is the sole data RNG. For training seed `s` and optimizer step `t`,
instantiate a fresh generator with seed `10_000_000 + 10_000*s + t`; therefore
every architecture sees byte-identical batches independent of model RNG use.
The generator consumes draws in this order per candidate example: `n`, `q`,
`k`, `v`, `u`, record permutation; after quadrant collection it draws the batch
permutation. Evaluation uses the analogous single stream from its slice seed.

PyTorch model initialization uses a separate generator seeded by training seed
`s`; modules are created in the frozen table order in source. Perturbation
permutations use separate seeds in section 10. No stochastic model operation or
dropout is allowed.

## 5. Generator and shortcut kills

All features below are computed separately on raw tokens and on mixed tokens.
For each view concatenate: `n/32`; the 14-coordinate record mean, population
standard deviation, minimum, maximum, sum, L1 norm contribution, and squared-L2
sum; first and last permuted record; and the query row. Standardize using the
100,000-example train feature mean and population standard deviation with
epsilon `1e-6`.

For each view and each label `y_s,y_m,y`, train a bias-affine linear two-logit
classifier in PyTorch with seed `840200 + 10*view + label`, AdamW at `1e-2`, no
weight decay, full deterministic batches of 4,096 for 2,000 steps. A separate
query-only MLP receives the 14 mixed query coordinates. A separate one-record
MLP receives query, one uniformly selected mixed record, and `n/32`; the record
index comes from PCG64 seed `840300+example_index`. Both are `input->64->2`
GELU models, Xavier initialized, AdamW `(0.9,0.95)`, LR `1e-3`, no decay,
batch 128, and 1,500 steps on the `y` target with data streams `840301` and
`840302`.

Here `view=0` is raw, `view=1` is mixed, and label IDs are `y_s=0,y_m=1,y=2`.
Each probe seed initializes both its model and an independent PCG64 index
stream. That stream repeatedly permutes the 100,000 training indices, emits
consecutive batches, and starts a fresh permutation when exhausted; the last
short batch is carried into the front of the next permutation to keep batch
size fixed. Evaluation is one deterministic pass in stored fixture order.

Every probe must be at most 55% accurate on every fixed slice and repeat
bitwise. Failure stops v2 before model training. These are leakage checks, not
proof that all attention-free solutions fail; the fatal conditioned-DeepSets
control in section 7 tests a nonlinear pooled solution.

## 6. Shared learned backbone

The standard backbone is:

- bias-free input projection `14->32`;
- hidden width 32 and two pre-norm causal blocks;
- learned-affine RMSNorm, scale initialized to one, epsilon `1e-6`, before
  attention and FFN;
- four heads of dimension eight unless a control says otherwise;
- bias-free Q/K/V/O; attention scale `1/sqrt(d_h)`;
- bias-free FFN `32->64->32`, exact GELU (`approximate='none'`);
- no dropout, positional encoding, checkpoint selection, or weight tying
  between the two blocks;
- final learned-affine RMSNorm and three independent `32->2` readouts with
  trainable biases initialized to zero; and
- loss `0.5 CE(y_s)+0.5 CE(y_m)+CE(y)`.

Standard linear weights use Xavier-uniform gain one; norm scales are one. The
standard backbone has 17,190 active parameters: input 448, two blocks each
`64 norm + 4,096 attention + 4,096 FFN = 8,256`, final norm 32, and readouts
198. All standard-attention controls and the candidate have this count.

Group RMS is applied at every token. A dimension-eight head splits consecutive
coordinates 0--3 and 4--7; a dimension-four head is one group; the 32 scalar
heads of a grouped narrow control are concatenated in head order and split into
eight consecutive groups of four.

## 7. Twelve frozen controls

The candidate must beat every control below. Unless stated, models retain the
17,190-parameter backbone, common mask, data, tuning, and training budget.

1. `soft_standard`: H=4,d=8, all softmax, no output group RMS.
2. `soft_grouped`: H=4,d=8, all softmax, two normalized groups per head.
3. `soft_narrow`: H=32,d=1, all softmax, no output group RMS.
4. `soft_narrow_grouped`: H=32,d=1, all softmax, eight normalized groups after
   head concatenation.
5. `hybrid_independent`: H=8,d=4; four complete soft heads and four complete
   channelwise max-plus heads, each head normalized as one group. This is the
   fixed-width independent-score and score-conditioned cross-token max control.
6. `hybrid_whole_head`: H=4,d=8; two soft heads and two max-plus heads; every
   head is split into two normalized four-channel groups.
7. `hard_shared`: H=4,d=8; one `argmax_j S_ij` per head copies the winner's
   complete eight-channel value, followed by the two group normalizations.
8. `finite_beta_8`: candidate layout, but max coordinates use stable
   `logsumexp(8*(S+V_c))/8`; both groups are normalized.
9. `all_max_shared`: H=4,d=8; every coordinate uses channelwise
   `max_j(S+V_c)` and each head has two normalized groups. This isolates whether
   heterogeneous reducers matter under the candidate score.
10. `tropical_official`: the source pinned in section 1 with
    `tropical_proj=True,tropical_norm=False,symmetric=True`; it computes
    `z=log1p(ReLU(h))`, splits H=4,d=8, applies three head-shared max-plus 8x8
    projections, scores `-(max_c(q_c-k_c)-min_c(q_c-k_c))`, applies the common
    causal mask, computes channelwise max-plus context, then `expm1` and a
    bias-free 32x32 output. Pure PyTorch broadcast/add/max is forced; no
    `tropical-gemm` import or backend selection is allowed. Its three tropical
    matrices retain upstream `torch.randn` initialization and its output retains
    PyTorch's upstream default linear initialization. FFN width is 109, making
    each block exactly `64 + 1,216 + 6,976 = 8,256` parameters and the whole
    model 17,190. This explicit causal adaptation is ours; it is not attributed
    to upstream.
11. `deepsets_conditioned`: an attention-free query-conditioned control. For
    every record, concatenate its 14 mixed coordinates with the 14 mixed query
    coordinates, apply bias-free `28->74->74` with GELU, then concatenate
    coordinatewise mean and max over records with the 14-coordinate query.
    Apply bias-free `162->32`, learned RMSNorm, bias-free `32->64->32` GELU,
    final learned RMSNorm, and the common readouts. It has 17,090 active
    parameters, 0.58% fewer than candidate, and no positional input.
12. `soft_time_matched`: the mechanically selected tied-repetition variant of
    `soft_grouped` from section 8. It retains 17,190 active parameters.

The broad learned-temperature, recurrent, and sparse-attention families are not
claimed cleared by v2. They are deferred unless this reject-only screen passes.
A semantic/numerical candidate crash fails v2. A required-control crash makes
v2 invalid and incapable of a positive result; it is never scored as control
error one. Only a common infrastructure failure reproduced before any metric is
decoded may replay the entire affected stage once.

## 8. Timing-only calibration and complete-cost control

Timing calibration is a separately authorized non-learning operation. Use
fixed contiguous float32 inputs from seed `843000`, batch 128, and lengths
9,17,33. In a fresh affinity-pinned process per variant, use inference mode,
200 warmups and 1,000 measured forwards in 20 round-robin blocks. Record
`perf_counter_ns` p50/p95/mean/SD, median absolute deviation, and `/proc/self/status`
`VmHWM`. Allocator warmup occurs before RSS baseline capture.

Candidate timing is compared with 16 tied `soft_grouped` variants
`(r_attention,r_ffn) in {1,2,3,4}^2`. In each of the two blocks, apply the same
attention residual with pre-norm `r_attention` consecutive times, then the same
FFN residual with pre-norm `r_ffn` times; parameters are tied within repetitions
but not across the two blocks.

For p50 and p95 separately define the target-workload number as
`0.25*T_len9 + 0.25*T_len17 + 0.50*T_len33`. A variant is eligible only when
both weighted p50/p95 and length-33 p50/p95 are between 95% and 105% of the
candidate. Among eligible variants choose lexicographically maximum
`(r_attention+r_ffn, r_attention, r_ffn)`; timing, never accuracy, makes this
choice. If none is eligible, report `TIME MATCH UNAVAILABLE` and v2 cannot
pass. Each model's MAD/p50 must be at most 5%; otherwise repeat the entire timing
matrix once in fresh processes before any training. A second failure stops v2.

All trained architectures report wall time and peak RSS. After training, one
fixed seed `842100` per architecture is timed by the same matrix for reporting
only; it does not alter selection or prove production GPU cost.

## 9. Training, tuning, and static resource ceiling

Every architecture has exactly two learning-rate candidates `{3e-4,1e-3}`.
For each, train seed `842000` for 1,500 steps and select the lower final XOR
error on the in-support development set only; exact ties choose `3e-4`. OOD and
ID test slices remain unopened. AdamW uses betas `(0.9,0.95)`, epsilon `1e-8`,
weight decay `0.01`, batch 128, global gradient clip one, no scheduler, no
warmup, no early stop, and no checkpoint selection.

Evaluation seeds are integers `842100` through `842115`. Each trains exactly
1,500 steps and the final state is evaluated. There are 13 architectures, 26
tuning jobs, 208 evaluation jobs, 351,000 optimizer steps, and 44,928,000
training examples before staged rejection saves. With the time control at its
maximum repetition, the static ceiling is 221,184,000 tied attention/FFN
sublayer-example applications. Base fixed-slice evaluation is 10.4 million
model-example evaluations, plus candidate perturbations and health probes.

The complete v2 CPU ceiling is eight single-core hours, 2 GiB process peak RSS,
and 2 GiB new artifacts. Exceeding any ceiling stops v2 as resource-inconclusive;
controls, steps, seeds, and thresholds are not reduced. No GPU fallback exists.

## 10. Causal perturbations and health

Perturbations are inference-only with no retraining.

- A soft or max lesion sets the named normalized groups to zero after group RMS
  and before `W_O`, at every token, head, and both blocks.
- A route shuffle changes only the named branch's final-query score row in both
  blocks. For every fixed-`n` slice and training seed, a stored Sattolo-cycle
  derangement from PCG64 seed `850000 + 100*training_seed + slice_index +
  10*branch` maps each example to a different example. Values, masks, labels,
  record score rows, and the other branch's final-query score stay paired.

Slice IDs are L16=1, L32=2, V2=3, L32V2=4; branch IDs are soft=0 and max=1.

All perturbation effects use the four-slice OOD macro and the 16 paired seeds.
Section 11 gives their simultaneous inference. It includes a double
dissociation: soft lesion must damage `y_s` more than `y_m`, and max lesion must
damage `y_m` more than `y_s`.

Health uses the intact candidate. For each seed/layer/head/max-channel, collect
the raw pre-group max output at the final query on the first 1,024 ID examples.
A channel is active only if its population SD exceeds `1e-3` and the Euclidean
norm of gradients of the mean full three-readout loss with respect to that raw
tensor, accumulated over eight ordered batches of 128 with gradients reset per
batch and squared norms summed, exceeds `1e-8`. Every seed must have at least
75% of its 32 max channels active.

For every active seed/layer/head/channel and every fixed slice, count final-query
winners by record position; ties choose PyTorch's first index. Require
`-sum p log(p)/log(n) >= 0.85` and every legal position to win at least once.
For each seed/block/slice and each branch, zero the other normalized branch and
apply bias-free `W_O`; the ratio of mean query-token L2 contribution to mean
intact pre-residual attention-output L2 norm must be at least 0.10. These are
intersection health kills that can only reject; they are not inferential causal
claims.

## 11. Stable estimands and 104-way simultaneous family

Let `e_smd` be final XOR error for training seed `s`, model `m`, and slice `d`.
The macro error is the arithmetic mean of that seed/model's four OOD slice
errors. For every control and each macro-or-slice statement, define the bounded
paired statistic

\[
G_{smd}=0.8e_{smd}-e_{s,candidate,d}.
\]

The claim “at least 20% lower expected error” passes iff the one-sided
simultaneous lower bound for `E[G]` is nonnegative. The point effect reported is
`1-sum_s e_candidate/sum_s e_control`; a zero summed control error cannot pass
the saturation rule and receives no pseudocount. For macro absolute gain use
`A_sm=e_sm-e_s,candidate,m`. For task saturation use `C_sm=e_sm`.

For any seed statistic `x_s`, with `n=16`, lower bound is
`mean(x)-tcrit*sample_sd(x)/4` and upper bound is
`mean(x)+tcrit*sample_sd(x)/4`; if sample SD is exactly zero the bound is the
mean. The model-based paired Student-t family uses one-sided Bonferroni alpha
`0.05/104 = 0.0004807692307692308`, 15 degrees of freedom, and frozen critical
value `tcrit=4.092099746816106`.

The 104 primary statements are:

- 60 relative-capability statements: 12 controls times macro plus four slices;
- 12 macro absolute-gain statements;
- 12 control macro saturation statements;
- three OOD-macro soft-bit non-inferiority statements against
  `soft_standard`, `soft_grouped`, and `soft_narrow_grouped`;
- three ID final-error non-inferiority statements against those controls;
- six causal effects, each with one relative and one absolute statement:
  soft lesion on soft-bit and final error, max lesion on max-bit and final
  error, soft-score shuffle on soft-bit, and max-score shuffle on max-bit; and
- two absolute double-dissociation statements.

Required bounds are:

1. all 60 `G` lower bounds are at least zero;
2. all 12 macro `A` lower bounds are at least `0.01`;
3. all 12 control macro-error lower bounds are at least `0.05`; otherwise the
   outcome is `TASK SATURATED`, never success;
4. candidate-minus-control soft-bit OOD-macro and final-error ID upper bounds
   are at most `0.01` for the three named soft controls;
5. soft/max lesion matching-subdecision error uses lower bounds on
   `e_lesion-1.2e_intact >=0`; lesion final error uses
   `e_lesion-1.1e_intact >=0`; each corresponding absolute increase lower bound
   is at least `0.01`;
6. each route-shuffle matching-subdecision effect uses
   `e_shuffle-1.2e_intact` lower bound at least zero and absolute-increase lower
   bound at least `0.01`; and
7. the lower bound for
   `[(e_soft_lesion,ys-e_intact,ys)-(e_soft_lesion,ym-e_intact,ym)]` and for
   `[(e_max_lesion,ym-e_intact,ym)-(e_max_lesion,ys-e_intact,ys)]` is at least
   `0.01`.

Also report pooled errors, absolute points, relative point effects, headroom
consumed, every seed value, and unadjusted intervals; none can replace the
frozen simultaneous gates.

## 12. Power and false-negative interpretation

The design is intentionally hard to pass. At the revised critical value,
single-contrast power depends on the standardized distance between the true
effect and the 20% boundary. A frozen two-million-draw noncentral-t sensitivity
calculation (PCG64 seed `840104`) gives:

| `(0.8 E[e_m]-E[e_c])/SD(G)` | 0.75 | 1.00 | 1.25 | 1.50 | 2.00 |
|---|---:|---:|---:|---:|---:|
| single-contrast power | 20.6% | 49.5% | 78.4% | 94.3% | 99.9% |

For a true 35% reduction, the distance beyond the 20% boundary is `0.15e_m`.
If `SD(G)=0.12e_m`, the standardized value is 1.25 at control errors 5%, 10%,
20%, and 40% (absolute margin/SD respectively `.0075/.006`, `.015/.012`,
`.030/.024`, `.060/.048`), so per-contrast power is only about 78.4%.

An illustrative one-million-draw equicorrelated noncentral-t simulation with 12
macro contrasts, correlated normal numerators and independent chi-square
denominators gives joint macro-pass probabilities 5.4%, 22.2%, and 36.6% at
numerator correlations 0, 0.5, and 0.9 for standardized value 1.25. These are
sensitivity calculations from PCG64 seeds `840140`, `840145`, and `840149`, not
guaranteed design power; real covariance and the other 92 gates are unknown.
Thus v2 is a high-false-negative screen. A failed
confidence bound closes this frozen attempt. It is evidence against the lane
only when the corresponding point effect is also below 20%; a point effect
below 10% closes the main lane, 10--under-20% is arguable but not success, and a
point at or above 20% with a failed bound is inconclusive and earns no GPU.

## 13. Staged no-rescue execution order

Nothing executes merely because this file exists. After the authorization
sequence in section 14, stages are:

1. timing calibration and fixed-fixture generation only;
2. Stage 0 operator/replay checks;
3. generator balance and shortcut kills;
4. candidate LR tuning, all 16 candidate seeds, all perturbations, and health;
5. complete LR tuning and all 16 seeds for controls, in this order:
   `finite_beta_8`, `all_max_shared`, `hybrid_independent`,
   `soft_narrow_grouped`, `tropical_official`, `deepsets_conditioned`,
   `hybrid_whole_head`, `soft_grouped`, `soft_standard`, `soft_narrow`,
   `hard_shared`, `soft_time_matched`.

After a stage or one complete 16-seed control contrast is hashed, a separate
decoder may reveal only `CONTINUE`, `STOP-REJECT`, `TASK SATURATED`, or
`RESOURCE-INCONCLUSIVE`. No metric is inspected within a 16-seed contrast. The
final `tcrit` and all 104 planned statements apply from the first decoded
contrast. A failed gate stops; there is no rescue LR, seed, threshold, support,
or extra run. Infrastructure exceptions follow section 7 only.

## 14. Immutable authorization sequence

1. Hash this exact v2 byte sequence and never edit it.
2. Obtain two independent preregistration audits of that hash. Record acceptance
   in a separate append-only manifest. A semantic correction creates v3.
3. Implement without executing experiment source. Hash source, tests, static
   architecture/parameter manifests, this preregistration, both audits, mixer,
   tropical snapshot, and environment.
4. Obtain an independent source decision that may authorize only
   `FIXTURES AND TIMING`.
5. Generate/hash fixed fixtures and run only the timing matrix. Mechanically
   append the selected time control and raw timing hashes.
6. Obtain a second independent source/artifact decision `RUN STAGE 0`.
7. Each passed stage appends a receipt binding all prior hashes before the next
   stage. Accuracy is decoded only at the complete-stage boundaries in section
   13.

Any source, environment, operator, generator, support, control, normalization,
training budget, perturbation, metric, threshold, timing rule, or confidence
change creates a new preregistration version. Under no v2 outcome is GPU or
rental work authorized.
