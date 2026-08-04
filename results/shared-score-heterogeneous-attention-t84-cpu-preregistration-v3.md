# T84 shared-score heterogeneous attention — CPU preregistration v3

Date written: 2026-08-02  
Status: **FROZEN V3 OVERLAY FOR INDEPENDENT AUDIT; NO EXECUTION AUTHORIZED**

The effective v3 protocol is the ordered pair:

1. `shared-score-heterogeneous-attention-t84-cpu-preregistration-v2.md`,
   SHA-256 `a40b7928078d3aa9bf8514b7a6349ab09a76477ebdf071426e5d4e92f4b049d5`;
2. this v3 overlay.

V2 was independently rejected before implementation or execution. Every v2
rule remains normative except where this file explicitly replaces it. On a
conflict, v3 controls. Acceptance must bind both hashes in order and must not
edit either file. This overlay closes only the exact ambiguities found by the
two v2 audits; it does not widen the scientific claim or authorize a run.

## V3.1 Frozen 100,000-example probe fixture

This section replaces the unspecified probe-training corpus in v2 section 5.

Generate one fixed in-support corpus from NumPy 2.3.5 PCG64 seed `840190`, with
`n` uniform over integers 4 through 8 and `a=1`. Use the exact latent draw,
record permutation, quadrant collection, quadrant order, final permutation,
token encoding, and stored mixer rules in v2 section 4. Keep the first 25,000
examples in each `(y_s,y_m)` quadrant, concatenate quadrants in the frozen
order, and apply one final permutation drawn from the same continuing stream.
Store raw float64 latents/labels, raw and mixed little-endian float32 tokens,
and SHA-256 hashes before any probe or model training.

For each probe, its declared PCG64 index seed defines an infinite index stream.
Draw a fresh permutation of `0..99,999`, consume it sequentially, and when a
batch crosses its end, append exactly the required prefix of the next freshly
drawn permutation and retain that next permutation's cursor. No index is
dropped or repeated except through a later independent epoch permutation.
Probe model initialization uses the declared seed through the per-tensor rule
in V3.4. Raw/mixed view IDs and label IDs remain those in v2.

## V3.2 Timing fixtures and executable process schedule

This section replaces the first three paragraphs of v2 section 8. The
95--105% eligibility, weighted p50/p95 equations, MAD rule, and lexicographic
time-control selection from v2 remain unchanged.

### Inputs and callable boundary

Generate three valid, exactly quadrant-balanced batches of 128 examples by the
v2 batch algorithm with `a=1`:

- length 9: `n=8`, PCG64 seed `843010`;
- length 17: `n=16`, PCG64 seed `843011`;
- length 33: `n=32`, PCG64 seed `843012`.

Store their mixed contiguous little-endian float32 token tensors, exact masks,
and hashes before calibration. Every timing variant receives the identical
bytes for a length. The timed callable is the complete inference forward from
mixed 14-coordinate tokens and mask through input projection, both blocks,
final norm, and all three query readouts. Loss, fixture loading, model
construction, process launch, and artifact serialization are outside each
individual forward sample, but their active wall time is charged by V3.3.

Every untrained timing model uses training seed `843001` and the per-tensor
initialization rule in V3.4. Variant ID 0 is the candidate. For
`r_attention,r_ffn in {1,2,3,4}`, the soft variant ID is
`1 + 4*(r_attention-1) + (r_ffn-1)`.

### Twenty-round schedule

For round `r=0..19`, visit the 17 variant IDs in order
`[(r+k) mod 17 for k=0..16]`. Launch a new affinity-pinned worker process for
each visited variant; the worker exits after that round. Inside worker `(r,v)`,
visit length IDs in order `[(r+v+k) mod 3 for k=0..2]`, where IDs 0,1,2 mean
lengths 9,17,33. For each length, run 10 unrecorded complete forwards and then
50 individually timed complete forwards under `torch.inference_mode()`.

Thus every `(variant,length)` has exactly 200 warmups and 1,000 recorded
samples. Pool its 1,000 samples before computing p50, p95, mean, SD, MAD, and
the frozen weighted quantities. CPU execution is synchronous; the timer is
`time.perf_counter_ns` immediately around the Python model call. The returned
three logits remain referenced until the call has ended. No output reduction
or `.item()` is added inside the sample.

Record the absolute Linux `VmHWM` for each fresh worker after all three lengths;
do not subtract a baseline or claim the high-water mark resets within a
worker. Report the maximum and all individual values. The same controller and
schedule perform the one allowed full-matrix MAD retry; the retry is charged.

## V3.3 Closed eight-hour and artifact ledger

This section replaces the resource-ceiling paragraph in v2 section 9.

The ceiling is eight hours of summed **active monotonic wall time** for all
authorized v3 controller invocations and their sequential affinity-pinned
workers, plus 2 GiB maximum absolute per-process `VmHWM` and 2 GiB total new
artifacts. Each invocation records `CLOCK_MONOTONIC` immediately before its
first experiment-related operation and after its last artifact fsync/hash, and
appends that duration to a SHA-256 hash-chained ledger. Workers may not overlap.

Charged time includes fixed-fixture generation, process launch, timing and its
frozen MAD retry, Stage 0, probe training/evaluation, LR tuning, all optimizer
steps, fixed-slice evaluation, perturbations, health forward/backward work,
artifact serialization, hashing, and decoder execution. It excludes prior
paper/design calculations, source writing, independent audit time, and idle
time between separately authorized stages. The controller checks the
cumulative ledger before and after every stage. Crossing any ceiling yields
`RESOURCE-INCONCLUSIVE`; no partial artifact can be used as positive evidence.

## V3.4 Per-tensor initialization and exact optimizer step

This section replaces the source-order initialization sentence in v2 section
4.4 and completes v2 sections 5, 6, 7, and 9.

Every trainable tensor has a frozen canonical path containing architecture ID,
block index if any, module name, and tensor name. Sort paths lexicographically.
For training seed `s`, derive its CPU generator seed as

`int.from_bytes(SHA256(UTF8("T84v3|s|canonical_path"))[:8], "little") & (2^63-1)`,

where decimal `s` and the literal canonical path replace the two fields. Each
tensor is initialized independently with that generator, so module construction
and unrelated tensor draws cannot change it.

- Standard, candidate, DeepSets, and probe linear weights use Xavier-uniform
  gain one; all learned RMS scales are ones and all biases are zeros.
- The three official-Tropical projection matrices use independent standard
  normal draws, matching the pinned upstream `torch.randn` rule.
- The official-Tropical output matrix uses Kaiming-uniform with
  `a=sqrt(5)`, matching the pinned bias-free `nn.Linear` reset rule.

Every AdamW optimizer has one parameter group containing all trainable tensors
in canonical-path order. Weight decay applies to norms and biases as well as
weights unless v2 explicitly sets probe decay to zero. Freeze
`foreach=False,fused=False,capturable=False,differentiable=False,amsgrad=False,
maximize=False`; other values are those in v2. A training step is exactly:
`zero_grad(set_to_none=True)`, forward, mean declared loss, backward,
`clip_grad_norm_(ordered_parameters,1.0,norm_type=2,error_if_nonfinite=True,
foreach=False)`, then `optimizer.step()`. Leakage probes have no gradient clip;
their otherwise identical optimizer step omits that call.

## V3.5 Recursive full-slice route shuffle

This section replaces the route-shuffle definition in v2 section 10. The same
stored Sattolo derangement `pi` is used in both blocks. It maps the full 10,000
example slice, never a minibatch, and has no fixed points.

For the named branch and a given slice/seed:

1. Starting from intact block-one inputs, compute and cache every example's
   complete block-one final-query score row `S1[b]`. Record-token rows are not
   perturbed.
2. Receiver `b` uses donor row `S1[pi(b)]` only for the named branch and its own
   row `S1[b]` for the other branch. Both branches use receiver `b`'s own
   values, mask, record order, and labels. Finish group RMS, `W_O`, residual,
   and FFN normally for all examples and cache the resulting hidden states.
3. From those recursively perturbed incoming hidden states, compute and cache
   every example's block-two final-query score row `S2[b]`.
4. Receiver `b` uses `S2[pi(b)]` only for the named branch and `S2[b]` for the
   other branch, again retaining its own values/mask/order/label, then completes
   block two and the readout.

Processing may use deterministic minibatches only to construct full-slice
caches. No donor lookup occurs until that block's complete 10,000-row score
cache is hashed. Hash both score caches, the derangement, intermediate hidden
states, and final logits. This is a recursively perturbed-donor intervention;
intact block-two donor scores are forbidden.

## V3.6 Candidate subdecision competence and 106-way family

This section replaces all occurrences of `104`, `0.05/104`, and
`4.092099746816106` in v2 sections 11--13.

Add two primary statements: across the 16 seeds, the one-sided simultaneous
upper bound on the candidate's four-slice OOD-macro `y_s` error is at most
`0.20`, and the corresponding upper bound on `y_m` error is at most `0.20`.
The family is therefore 106 statements:

`60 capability + 12 absolute + 12 control-floor + 3 soft non-inferiority +
3 ID non-inferiority + 12 causal + 2 double-dissociation + 2 competence`.

Use one-sided Bonferroni alpha
`0.05/106 = 0.0004716981132075472`, 15 degrees of freedom, and
`tcrit = 4.1014945453569363208`. Every staged contrast uses this final critical
value from its first decode.

At standardized distances 0.75, 1.00, 1.25, 1.50, and 2.00, the replacement
single-contrast noncentral-t sensitivities are approximately 20.4%, 49.2%,
78.2%, 94.2%, and 99.9%. The replacement illustrative 12-macro joint estimates
at numerator correlations 0, 0.5, and 0.9 are 5.2%, 21.8%, and 36.1%, from one
million draws with PCG64 seeds `840160`, `840165`, and `840169`. They remain
high-false-negative sensitivity calculations, not claimed design power. The
power calculation source and canonical output hashes must join the source
manifest before Stage 0.

V2's phrase “the other 92 gates” is replaced by “the other 94 gates.”

## V3.7 Failure, replay, and decoder closure

This section replaces the infrastructure replay exception and the four-word
decoder vocabulary in v2 sections 7 and 13.

There is no ad hoc infrastructure replay. A process, OS, storage, affinity,
interpreter, numerical, semantic, or required-control failure makes the frozen
attempt `PROTOCOL INVALID`. The sole repeat is the already frozen complete MAD
timing-matrix retry, which occurs only under its numeric rule and is fully
charged. Recovery requires a new preregistration version; no failure is scored
as evidence for the candidate.

Before Stage 0, independently audit and hash decoder source that maps each
complete receipt to exactly one of:

- `CONTINUE`: every currently decidable gate passes and resources remain;
- `STOP-REJECT`: a valid candidate/contrast gate fails;
- `CONTROL FLOOR NOT ESTABLISHED`: a simultaneous 5% control-error lower bound
  fails, without claiming literal saturation;
- `RESOURCE-INCONCLUSIVE`: a frozen resource ceiling is crossed; or
- `PROTOCOL INVALID`: any integrity, source, environment, process, numerical,
  semantic, required-control, or unexpected-condition failure.

The decoder may output no metric, architecture ranking, or diagnostic while
returning a stage word. Complete artifacts stay sealed until the protocol
terminates or every planned control passes. Its source/hash cannot change after
the first execution authorization.

## V3.8 Effective authorization

No v3 action is authorized until two independent audits accept the exact
ordered `(v2 hash, v3 hash)` pair in a separate append-only manifest. The v2
authorization sequence then applies to the effective pair, with V3.7's decoder
hash added to the pre-Stage-0 source manifest. Any further semantic edit creates
v4. No GPU or rental is authorized under any outcome.
