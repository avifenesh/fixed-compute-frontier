# Heterogeneous family discovery T7 — adjacent-scale preregistration

Frozen before any T7 training measurement: 2026-07-31

## Decision being tested

T6 found a qualitative acquisition gap at 36.6M parameters: compilation learned
256 exact procedures from examples while ordinary AdamW training remained far
from exact after twice the mixed-step budget.  T7 asks whether that gap survives
an untouched 110.8M scale, a calibrated Muon control, three paired seeds, and
complete resource accounting.

The primary result is **not** a small language-NLL delta.  The primary gate is
exact acquisition across four latent program families versus failure by both
calibrated-Muon and AdamW controls after 2x training.  Natural NLL is only a
protected noninferiority condition: exact synthetic capability does not count
if it damages the language model.

## Frozen model and untouched slices

- One tied-embedding, pre-RMSNorm causal LM: vocabulary 49,152, width 640,
  16 blocks, ten 64-wide heads, SwiGLU width 1,728, context 128, and exactly
  110,776,960 deployed parameters.
- Two existing heads remain unrotated for the interpreter; eight language heads
  use RoPE.  Every arm has the same dense matrices, vocabulary, forward graph,
  BF16 arithmetic, checkpoint form, KV state, and inference FLOPs.
- Model seeds are 3,149, 3,499, and 3,853.  Execution order is rotated across
  seeds to reduce thermal/order bias.
- The task-world seed 2,000,033 and evaluation seed 3,000,017 were not used in
  T6 or the optimizer pilot.  They generate new supports, prefixes, random
  decoys, mixed examples, and held-out evaluations.
- The sealed FineWeb-Edu train and document-disjoint validation streams retain
  SHA-256 values `1871a8a790e2b2ae5273f1a46bc9fa2e39cbe95d5cb7537bde5a2b3e35cb464a`
  and `889188f0e4c43cd49b2933ac462e8bd367520f15fe08d324f169f7eca962ce7b`.

## Frozen discovery problem

There are 320 task IDs.  Each structured task depends on an unknown 8-of-16
support and one unknown family: parity, strict majority, exact-half, or count
modulo 3.  There are 64 tasks per family and 64 independent random-label
decoys.  Each task provides 96 charged input/output examples.

The candidate sees only task IDs, bit vectors, and labels.  It enumerates all
12,870 supports and all four frozen family tables.  It accepts a task only when
exactly one program has zero prefix error; otherwise it abstains.  The expected
work ledger separately records integer matrix multiply-adds, Boolean family
comparisons, world-generation time, and discovery time.

An accepted program is written into existing task-token rows and one shared
fixed-width interpreter occupying 28 hidden coordinates, two heads, and at most
128 SwiGLU channels.  The solver is deleted.  There is no added token row,
sidecar, adapter, expert, retrieval store, runtime graph, or test-time compute.

## Frozen optimizer calibration

The optimizer pilot used development seed 2,719, natural text only, identical
initializations/batches, and no T7 capability data.  The finite Muon arm with
the lowest step-250 held-out NLL was selected by a rule frozen before the pilot.
The winner was matrix learning rate 0.005 (NLL 6.52323, versus AdamW 6.64774).

Hidden-layer matrices use five-step Newton--Schulz Muon with
`match_rms_adamw`, momentum 0.95, Nesterov updates, and zero weight decay.
The tied embedding and RMSNorm vectors use AdamW at 3e-4.  Clip norm is 1.0;
the warmup/cosine multiplier is common.  The pilot artifact hash is
`5f4792b25162b1b3abdc6fb7c9948774a225e64b29e3edfbcdac144ad107c8cb`.

The direct binary cross-entropy given to controls is already the maximally
informative target for these examples.  Multi-token prediction cannot reveal
an unavailable extra future label here, so it is not a stronger applicable
objective control for the claimed acquisition gap.

## Arms and budgets

Every twentieth mixed step is algorithmic; the other 19 use paired natural
batches.  All controls consume every one of the 30,720 prefix examples through
direct supervised gradients (480 updates).  The compiler consumes those same
examples through discrete discovery.

- `muon_1x`: calibrated hybrid Muon/AdamW, 480 prefix updates, 1,000 mixed
  steps.
- `compiler_muon_1x`: exact discovery/compilation, no prefix gradient updates,
  the same hybrid optimizer, 1,000 mixed steps.
- `muon_2x`: calibrated hybrid Muon/AdamW, 480 prefix updates, 2,000 mixed
  steps.
- `adamw_2x`: AdamW at 3e-4, 480 prefix updates, 2,000 mixed steps.

Candidate checkpoints are evaluated at 250, 500, and 1,000.  The 2x controls
are adjudicated at 2,000.  Evaluation batches are identical within a paired
seed.  No failed arm may be silently restarted, shortened, or replaced.

## Frozen gates

All gates must pass independently in all three seeds:

1. Every arm starts from an identical initialization hash.  Discovery recovers
   all 256 structured supports and families exactly and accepts zero random
   decoys.
2. At candidate steps 250, 500, and 1,000, every family has at least 99% mean
   and worst-task accuracy; protected copy is at least 99%; independent random
   decoys remain between 40% and 60%.
3. At step 2,000, **both** Muon and AdamW controls remain below 80% mean on
   parity and modulo-3 while protected copy is at least 95%.
4. At each paired candidate checkpoint, natural validation NLL is no more than
   0.5% above `muon_1x`; candidate NLL improves from step 500 to 1,000.
5. Every loss and pre-clip gradient norm is finite, maximum loss is below 100,
   no step is skipped or retried, and no candidate sample is discarded.
6. Parameter count, vocabulary, inference graph, dense inference FLOPs, and
   checkpoint representation are exactly identical across arms.

## Resource ledger

For each arm, record total wall/GPU seconds; sampled power and estimated
joules; current on-demand instance rate and run cost; peak allocated/reserved
HBM; optimizer-state bytes; raw and model-presented natural/algorithm tokens;
unique natural positions; prefix examples; BF16 forward/backward model FLOPs;
FP32 Muon Newton--Schulz FLOPs; AdamW elementwise FLOPs; evaluation FLOPs; and
failed/retried work.  World generation is charged to every arm.  Discovery is
additionally charged to the compiler arm.

## Interpretation boundary

Passing would establish a two-scale, three-seed qualitative acquisition result
against a calibrated matrix-optimizer envelope with an unchanged deployed LM.
It would not establish broad natural-language reasoning improvement: the task
families and interpreter remain synthetic and human-specified.  A passing T7
therefore advances the mechanism to an untouched compositional/natural transfer
test; it does not justify calling a 0–3% NLL difference the breakthrough.

Failure of exactness, strong-control separation, protected language quality,
or accounting closes this scaled formulation without post-result threshold or
seed changes.

## Frozen implementation hashes

- Full runner: `cddd3ad6ae1531ceff2de8bd6a101f0d0d1ba90e45ad3e81a3b90a4ed6f2d793`
- Pilot/model source: `78733f13ae09ae7476b197a0ce31dd11b1fdcc9460d07c5f90929664e52591f6`
- Structural tests: `b7492bcc7d5d428eb7f3eb907c9fb866be75c5b7004305f7117f7d9cadb7bcb6`
- T6 predecessor artifact: `0748e0d39c1315d434f2bff664af45120cae7c8c542e4b69c5202565de326576`
