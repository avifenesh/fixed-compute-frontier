# Top-k compiled pushdown acquisition: CPU preregistration

Date: 2026-08-01  
Lineage: `CPKV-TOPK-001`  
Status: **IMPLEMENTATION AUTHORIZED BY TWO INDEPENDENT REVIEWS; TRAINING LOCKED
UNTIL ORACLE AND MANIFEST PASS**

## 1. The one question

When a finite 32-configuration weighted pushdown model learns only from raw
next-token strings, does it acquire a useful LIFO mechanism whose predictive
advantage survives causal compilation to three online configurations?

This is not a production-language, GPU, or novelty test.  It is the minimum
acquisition bridge required before spending on either.

## 2. Frozen decision

`S32` is the training model.  After every prefix it retains exactly the 32
highest-weight complete configurations reachable from its own previous retained
state.  `K1`, `K2`, and `K3` are the same frozen checkpoint run with smaller
causal beams, without retraining.

Advance only if all three claims pass:

1. **acquisition:** `S32` materially beats every matched learned control on
   both ambiguous families and both extrapolation bins;
2. **mechanism:** stack-read and ancestry interventions selectively remove the
   delayed-dependency gain; and
3. **compression:** `K3` retains the causal path mass and at least 90% of the
   `S32` predictive gain.

If `S32` helps but `K3` fails, close `CPKV-TOPK-001`.  If `DIRECT3` or `HARD3`
matches `S32`, the training-wide compiler claim fails even if a three-stack
architecture remains interesting under a new lineage.  No `k>3`, new grammar,
width, threshold, optimizer, or replacement seed may rescue this lineage after
results are opened.

## 3. Exact learned operator

All vectors have width 64.  Given token `x_t`, previous shared hidden state
`h_(t-1)`, and previous weighted stack read `r_(t-1)`:

\[
e_t=E[x_t],
\]

\[
h_t=\tanh(W_e e_t+W_hh_{t-1}+W_rr_{t-1}+b_h),
\]

\[
u_t(c)=W_ah_t+T[q(c),topSymbol(c),:],
\qquad p_t(a\mid c)=softmax(u_t(c))_a.
\]

`W_a h_t` is computed exactly once per token and reused for every configuration;
only the table row and softmax vary with `(q, topSymbol)`.  The dense ledger
therefore charges the 15-by-64 action projection once, not once per beam state.

There are three controller states, three push symbols plus root, and 15 actions
`q' x {push(gamma),pop,noop}`.  Root pops are masked before softmax.  A push
payload is shared by all configurations created at the token:

\[
\tilde v_t=W_vh_t,\qquad
v_t=\tilde v_t/\max(1,\|\tilde v_t\|_2).
\]

Candidate unnormalized probability is parent probability times action
probability.  Byte-identical `(new-q, complete pop-reachable stack)` candidates
are merged by summation; top 32 are selected by descending weight with
lexicographic structural-key tie breaking; only then are weights renormalized.
The read and next-token logits are

\[
r_t=\sum_{c\in S32_t}P_t(c)v(top(c)),
\]

\[
z_t=W_yh_t+U_yr_t+b_y.
\]

The root payload is zero and initial state is `(h_0,r_0)=(0,0)` with one
configuration `(q=0,root,weight=1)`.  The prediction from `z_t` is for
`x_(t+1)`.  Inputs contain only raw tokens plus `BOS/EOS`; action, split, count,
and grammar-branch labels are never exposed.

For vocabulary size `V`, the exact parameter count is

\[
|theta|=193V+17,588.
\]

This comprises `E`, three 64-by-64 recurrence matrices, `b_h`, the 15-by-64
action projection, the `3*4*15=180` transition table, `W_v`, two 64-by-`V`
output matrices, and output bias.  Implementation tests must reproduce this
count.

`S32` is an exact finite operator, not an approximation certificate for an
infinite NPDA.  Training gradients pass through the selected configurations
and their normalized weights; selection/tie decisions themselves have zero
gradient almost everywhere.  FP32 is used for learned tensor computation;
evaluation configuration probabilities, merging, retained-mass accounting,
and reported KL use FP64.  Stable softmax subtracts the maximum.

## 4. Exact data distributions

Each family has its own vocabulary and model.  A range `[l,u]` means a discrete
uniform integer.  Content bits/types are independent fair Bernoulli draws.

- **D1, deterministic typed nesting:** sample `n`; draw opening types
  `w in {open0,open1}^n`; emit `w` followed by the uniquely typed closers in
  reverse order.
- **A1, latent-center palindrome:** sample half-length `n` and parity
  `o~Bernoulli(1/2)`; draw `w in {0,1}^n` and, if `o=1`, a fair center bit;
  emit `w + center_if_any + reverse(w)`.  There is no center marker.
- **A2, ambiguous counting union:** sample branch `b~Bernoulli(1/2)`, tied count
  `n`, and free count `m` independently.  If `b=0`, emit `a^n b^n c^m`; if
  `b=1`, emit `a^m b^n c^n`.  Duplicate intersection strings retain their
  generator multiplicity.
- **X4, analytic sensitivity assay:** no learning.  Four equiprobable,
  future-distinguishable configurations have orthogonal unit payloads before a
  delayed reveal.  Top-3 causal history mass is exactly 0.75; a larger value is
  an implementation failure.

Ranges and sample counts are:

| Split | Range for `n,m` | Sequences/family | Use |
|---|---:|---:|---|
| train | 1-8 | 20,000 per model seed | gradients |
| development | 9-12 | 5,000 fixed | checkpoint selection |
| evaluation E1 | 13-20 | 10,000 fixed | locked decision |
| evaluation E2 | 21-32 | 10,000 fixed | locked extrapolation |

Training data use the model seed.  Fixed base generator seeds are D1=`9101`,
A1=`9201`, and A2=`9301`, with split offsets development=`0`, E1=`1`, E2=`2`
and NumPy `PCG64DXSM`.  The implementation manifest
must record exact token maps plus SHA-256 hashes of generator source and all
development/training split files before the first optimizer step.  E1/E2 are
committed by expected SHA-256 but are not materialized until a receipt hashes
all frozen checkpoints; a separate evaluator then writes them once and verifies
the commitments.

The checkpoint-freeze receipt must bind the current manifest SHA-256 and exactly
180 entries: all 3 families x 6 independently trained arms
`{S32,DIRECT3,HARD3,MLP,RNN32,LINK32}` x 10 seeds.  Every entry contains a
unique project-relative path, SHA-256, selected epoch in 1-20, and development
NLL.  The evaluator recomputes every on-disk hash and rejects missing, extra,
duplicate, outside-project, or malformed entries before materializing E1/E2.
It also verifies every frozen source hash in the manifest.  Each entry points
to a `cpkv-topk-001-run-v1` JSON object binding its key and manifest, the hashed
training/development files, 20 finite development NLLs, 20 completed epochs,
6,260 optimizer steps, 400,000 seen examples, and a hashed selected FP32 NPZ.
The evaluator recomputes the earliest epoch within `1e-8` of the minimum and
loads the NPZ with pickles disabled.  Tensor names/shapes are fixed by arm and
must contain exactly `193V+17,588` finite FP32 coefficients.

Training and decision NLL are token-weighted, exclude `BOS`, and include `EOS`.
Sequence-weighted NLL is reported but cannot change the decision.  Delayed
forced-choice positions are all typed closers for D1, mirrored-suffix bits for
A1, and the correct stop/continue boundary at the `b` and `c` block ends for
A2.  Accuracy compares the correct next token only with its single legal
continue/stop distractor.

Forced-choice ties count as incorrect.  Ordinary greedy decoding breaks an
exact logit tie by lexicographic token string.  The earlier scaffold slice is
every target before the first delayed position, reported both in aggregate and
separately for every total non-special-token length; no length slice may be
pooled away.

## 5. Learned arms and matched controls

Every control keeps the common embedding, hidden recurrence, and output
equations.  The `5,236` stack-module parameters (`W_a,T,W_v`) are replaced or
matched exactly.

1. `S32`: the operator above.
2. `K1/K2/K3`: frozen `S32`, smaller causal beam, no retraining.
3. `DIRECT3`: the identical K3 forward operator trained from initialization.
4. `HARD3`: three named lanes initialized in controller states 0, 1, and 2.
   Each chooses one action by straight-through argmax of its 15-way softmax;
   lane read weights are their normalized cumulative selected-action
   probabilities.  Router and payload parameters are shared.  At evaluation it
   is exactly three deterministic stacks, not a branching beam.
5. `MLP`: replace the stack module by
   `r=W2*tanh(W1*h+b1)+b2`, width 40 (5,224 parameters), plus 12 learned gains
   applied to the first 12 output coordinates, totaling 5,236.
6. `RNN32`: recurrent state `s` of width 32 with
   `s=tanh(Ah+Bs+b_s)`, `r=Cs+b_r` (5,216 parameters), plus 20 learned output
   gains, totaling 5,236.
7. `LINK32`: identical to `S32`, except a retained push links to the most
   recently allocated node before the current token rather than the pushing
   configuration's current top.  Multiple retained pushes allocate in
   structural-key order.  It is independently trained.  LIFO code computes
   the same chronological pointer as a discarded dummy so pointer arithmetic
   and storage differ only in which parent is written.

All arms report executed operations and state; equal parameters alone are not
called compute matching.  `DIRECT3` tests the value of training-wide search,
`HARD3` tests three-expert containment, and `LINK32` tests whether ancestry is
specifically LIFO.

The evaluator also reports peak logical auxiliary bytes per sequence, excluding
Python-object overhead: 256 bytes per width-64 FP32 payload, 9 bytes per
persistent `(parent:uint32, source:uint32, symbol:uint8)` node, 16 bytes per
`(controller:uint32, pointer:uint32, weight:float64)` configuration, and 128
bytes for the RNN32 state.  This diagnostic FP32 ledger is distinct from the
BF16 target-serving ledger in Section 2.

## 6. Training and checkpoint rule

- ten fixed model seeds `1701` through `1710`;
- batch size 64, exactly 20 epochs, no early stopping;
- AdamW: learning rate `3e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, weight
  decay `0.01`, global gradient norm clip `1.0`;
- linear warmup for the first 5% of optimizer steps, then cosine decay to
  `3e-5` at the final step;
- checkpoint after each epoch; choose lowest development token NLL, breaking
  ties within `1e-8` in favor of the earliest epoch; and
- no restart selection, replacement seeds, or arm-specific schedule.

Each family/seed's 20,000-example training file is generated once with
`PCG64DXSM(model_seed)`, hashed, and shared byte-for-byte by all arms.
Initialization uses independent name-addressed streams
`PCG64DXSM(hash64(model_seed, parameter_name))`, where `hash64` is the
little-endian integer encoded by the first eight bytes of
`SHA256(f"{model_seed}:{parameter_name}")`.  Embeddings and the transition
table are zero-mean normal with standard deviation 0.02, dense matrices use
Xavier uniform with gain 1 and bounds
`+-sqrt(6/(fan_in+fan_out))`, biases are zero, and control gains are one.  This
prevents parameter declaration order from changing another arm's common
initialization.

Epoch `e` uses one shared permutation from
`PCG64DXSM(model_seed+100000+e)`.  A batch contains 64 unpadded sequences;
per-sequence target losses are summed and divided by the total number of
non-BOS targets, including EOS.  AdamW decay applies only to two-dimensional
learned matrices (including embeddings), excluding biases, transition tables,
and gains.  HARD3 straight-through argmax uses no sampling noise.  CPU
execution sets deterministic algorithms, one intra-op thread, and FP32 without
mixed precision.

Run on CPU.  If all arms for one family/seed exceed six CPU-hours or 32 GiB
peak RSS, stop before opening evaluation data and return to the paper ledger.
The frozen budget may not be silently reduced.

Before the first run, a length-stratified timing preflight reports CPU process
time, peak RSS, retained configurations, candidates, merges, softmaxes,
payload reads/writes, push/node writes, and dense MACs.  Initial-weight timing
is only a projection: the checkpoint-freeze validator recomputes actual
per-family/seed totals from all six run receipts and enforces the limits again.

## 7. Exact mass and approximation measurements

Teacher-force `S32` and a structural K3 shadow with the same `S32` hidden
trajectory.  Inside every `S32` configuration propagate two unnormalized
weights: total path weight and weight of action histories that survived every
preceding K3 selection.  Merging sums both; `S32` normalization applies the
same denominator.  Define

\[
M_t^{(3)}=
\sum_{\pi_{1:t}\ \text{survived all K3 prefixes}}
P_{S32}(\pi_{1:t}),
\qquad \delta_t=1-M_t^{(3)}.
\]

This counts causal action-history coverage, not full `S32` mass that happens to
merge later into a currently retained configuration.  Configuration identity
includes controller state and the entire pop-reachable
`(symbol,payload-source)` stack.

With payload norm at most one, the renormalized covered-history read must obey

\[
\|r_t-\hat r_t^{covered}\|_2\le2\delta_t.
\]

For K3, require every seed and both A1/A2 bins to have `delta_t<=0.01` on at
least 99% of prefixes, mean `delta_t<=0.01` per bin, and no algebra-bound
violation beyond `1e-6`.

## 8. Recurrent and causal measurements

### 8.1 Recurrent-state teacher-forced rollout

Feed each frozen arm the same ground-truth token prefix while allowing hidden
states, stack reads, and future router logits to evolve independently.  Do not
call this free rollout.  K3 must have mean `KL(p_S32||p_K3)<=0.01` nat/token and
no family/bin above 0.02.

### 8.2 Checkpoint interventions

On frozen `S32` and K3 checkpoints:

- `ZERO-READ`: replace `r_t` by zero after the stack update;
- `CHRONO-LINK`: on every push write the same chronological parent rule used by
  `LINK32`.  This changes no learned parameter, action-logit formula, payload
  value, beam width, or selection rule.  The changed parent relation can alter
  later configuration merging, node allocation, and payload reads; those are
  causal consequences of removing pop-reachable ancestry, not quantities held
  fixed.  Report their executed-state counters beside the clean trace.  The
  intervention therefore identifies dependence on ancestry, not an
  equal-executed-work treatment effect.

For each seed, family, and bin, let `A_C` be delayed accuracy of the control
with lowest NLL in that same fixed cell, `A_clean` the clean stack model's
accuracy, and `A_int` its intervened accuracy.  Define disappearance

\[
D=\frac{A_{clean}-A_{int}}{A_{clean}-A_C}.
\]

The positive-acquisition gate guarantees the denominator.  For each
intervention, model (`S32` and K3), family, and bin, all ten seeds must have
`D>0`, the median must be at least 0.80, and no seed may be below 0.60.
Accuracy before the first delayed-dependency position may change by no more
than one absolute point in any seed.  Otherwise the gain is not identified as
a LIFO-stack effect.

### 8.3 Free continuation outcome

Use each locked evaluation prefix immediately before its first delayed-choice
position as a prompt.  A sampled reference suffix is **not** a unique target:
A1 can retain several center/length hypotheses and A2 contains a free count.

For each prompt, analytically enumerate the conditional generator distribution
over legal EOS position and count/length outcomes.  Generate greedily and draw
10,000 total paired ancestral continuations stratified over prompts with fixed
seed `9401`.  Each locked split contains 10,000 prompts, so every prompt
contributes exactly one ancestral continuation per model.  The same
`PCG64DXSM(9401)` uniform array is reused across arms and seeds.  Categorical
sampling follows frozen token-map order.  Strata are `(family, bin, prefix
length excluding BOS, last token)`.  The horizon is one generated token beyond
the longest legal content: `2u+1` for D1, `2u+2` for A1, and `3u+1` for A2;
failure to emit EOS is both a late and a no-EOS error.  Analytic conditionals
retain generator multiplicity, including duplicate A2 intersection strings.
Report:

- grammar-valid completion rate;
- premature/late EOS rates;
- total variation between model and analytic conditional length/count outcome
  histograms, averaged over fixed `(family, bin, prefix-length, last-token)`
  strata; and
- prompt-conditional teacher-forced suffix NLL as the proper predictive score:
  initialize the same incremental decoder on the prompt, then score every
  reference suffix token from the first delayed-choice token through EOS.  Sum
  those token losses per prompt, then average the complete-suffix NLL over all
  prompts in the cell; also report the scored suffix-token count.

For greedy decoding, compare to the analytic MAP continuation with
lexicographic tie breaking; do not compare to the one sampled reference suffix.
For each scalar outcome and each seed, form improvement over the strongest
control in that seed/family/bin (sign-reversing error metrics).  K3 retention is
the ratio of its improvement to `S32` improvement.  Each family/bin must have
median retention at least 0.90 and no seed below 0.80.  Undefined or
nonpositive denominators fail.  No tokenwise KL is computed across different
generated histories.

## 9. Pass thresholds without small-sample inference

For every control `j` and seed, define token-NLL gain

\[
G_{S,j}=N_j-N_{S32}.
\]

`S32` acquisition passes only if, separately on A1 and A2 and both E1/E2 bins:

- all ten seeds have `G_(S,j)>0` for every control;
- the across-seed median is at least 0.05 nat/token against every control;
- all ten seeds improve delayed forced-choice accuracy against every control;
  and the median improvement is at least five absolute points; and
- no earlier length/scaffold slice regresses by more than 0.01 nat/token.

No confidence interval or p-value is claimed.  All-seed directionality and the
material median threshold are the frozen diagnostic rule.

For each seed let `N_C` be the lowest evaluation NLL among all controls, making
the comparison conservative.  After the positive acquisition gate, K3
compression passes only if every seed has positive gain over every control,
the median

\[
R_3=\frac{N_C-N_{K3}}{N_C-N_{S32}}
\]

is at least 0.90 in every family/bin, and no seed is below 0.80.  Delayed
accuracy uses the same per-seed ratio

\[
R_{acc}=\frac{A_{K3}-A_C}{A_{S32}-A_C};
\]

its median must be at least 0.90 in every family/bin and no seed may be below
0.80.  Metrics are computed per seed over the complete locked prompt set before
ratios or medians; prompts are never pooled across model seeds.  NaNs,
nonpositive denominators, or clipped ratios fail.

D1 must show positive stack gain and K1 retention but cannot compensate for A1
or A2.  Concretely, D1 uses the same all-seed, median `0.05` nat/token,
median five-point delayed-accuracy, and no-early-regression acquisition
thresholds as A1/A2.  Its K1 NLL and delayed-accuracy retention use the same
median `0.90` and per-seed `0.80` floors as K3.  X4 must reproduce its analytic
mass and read-error values before any learned evaluation is opened.  Report
every seed, arm, position type, and length bin; no best-of-N summaries.

## 10. Integrity and next authorization

The implementation may be written only after this paper is independently
authorized.  Before training, freeze a manifest containing source hashes,
generated split hashes, the complete locked evaluator and tests, the passing
length-six oracle receipt, exact parameter/MAC/state tables, package versions,
and every command.  Exhaustive length-at-most-six comparison against a second
enumerator must pass for transition merging, root behavior, action-history
coverage, and the `2delta` bound.

Development data may select checkpoints but cannot alter the model or gates.
Locked evaluation uses exactly 60 deterministic shards, one for each
lexicographically sorted `(family, split, seed)` cell over D1/A1/A2, E1/E2, and
seeds 1701--1710.  Shard `i` is written once as `shard-{i:02d}.json`; it binds
the manifest hash, checkpoint-freeze receipt hash, locked-evaluator source hash,
index, and cell key, and contains exactly one teacher-forced and one continuation
cell.  The merge accepts exactly those 60 filenames, rejects missing, duplicate,
extra, lineage/hash/key/index-mismatched cells, sorts by frozen cell key, then
computes the gates and writes the single final result once.  Rerunning a missing
shard is allowed because its prompts, paired uniforms, arithmetic, and output
path are fully deterministic; overwriting an existing shard is forbidden.  A
code change after reveal requires new split hashes and a new lineage.

A complete pass authorizes only a target-kernel paper audit.  It does not claim
a smarter production model.  Before any later GPU run, sample the local GPU for
60 seconds; use it only if it has no foreign compute process, maximum
utilization at most 5%, temperature below 70 C, and at least 24 GiB free.
Otherwise rent a new isolated H100 without touching existing instances.
