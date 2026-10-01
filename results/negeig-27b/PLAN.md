# Negative-eigenvalue retrofit on Qwen3.8-27B: plan and budget (pre-registration)

Status: **signed off 2026-10-01** ("signed off, use the nebius box, start the engineering"); hardware revised the same
night to two preemptible 8x B300 boxes (owner choice "B300 spot, H200 fallback"). Data built and verified (`DATA.md`,
manifests beside this file).

Outcome (2026-10-01, `stageA-log.md`): **stopped after Stage A.** S1 past the training length, wide minus ctrl, -0.4
points against the +10 bar, so Stage B did not run. The on-call probe the owner asked for before any kill shows no gain
from the gate and a token cost; the tier-1 probe shows the mechanism is real on the 27B (+16.1 points at 4x, almost all
parity). The plan below is the pre-registration as signed off.

## Question

On the 4B and 9B bases the widened-beta gate added state tracking the control could not reach (tier-1 at 4x: +48
and +43 points over LoRA alone, 3 seeds each; `../negeig-retrofit-preregistration.md`). This step asks whether it
does the same on the model we actually use, Qwen3.8-27B (post-trained, `qwen3_5` family: 48 Gated DeltaNet + 16
attention layers, the same `beta = sigmoid(b)` limit), on state tracking written like real work, and whether that
transfers to agentic use without costing tool use, knowledge, code or Hebrew.

## Arms

`wide` = LoRA r16 on the Gated DeltaNet projections plus the zero-initialized widened-beta gate; `ctrl` = the same LoRA
alone. Same data, steps, seeds and learning rate. Gate G0 on the 27B before anything else (bit-identical at W = 0,
gradient in all 48 GDN layers). The decay gate is not used (no gain at 4B).

## Stages, each with a go/no-go

**Stage A: supervised (S1 plus replay).** Mix per step: S1 sessions (six domains, 32 to 256 events, 25% Hebrew in
four), self-distilled chat replay (the untouched model's own answers to 6,002 human-written prompts, plus Hebrew
versions it translates), self-distilled tool-use replay (its own score-1.0 runs on the agentic pool), and the clean
LM replay (32M tokens: English web, 20% code, 20% Hebrew web). Loss on assistant turns only; thinking off for S1.
Learning rate measured first on a short sweep (1e-4 and 3e-4 per arm, as the 4B lane taught), then 2 seeds per arm.
About 75M tokens per run (the 4B and 9B switches needed 50 to 75M).

- **Go to Stage B only if** `wide` beats `ctrl` by at least 10 points on the S1 eval past the training length
  (sessions of 512 and 1,024 events, accuracy after event 256), mean over seeds, and no general eval regresses
  beyond the gates below.

**Stage B: reinforcement learning (S2 plus the agentic pool).** GRPO on the Hebrew lane's verl stack (its recipe:
LoRA, AdamW 1e-5, n = 8 rollouts, no std normalization, KL 0.01 to the stage-A adapter, clip 0.2/0.28), items: S2
(6,000 episodes, four domains) and the lane's agentic pool (parallel, irrelevance, missing parameter, multi-turn,
chain, long result, unsupported). One seed per arm, from each arm's best stage-A seed.

## What decides

1. **Primary (Stage A):** S1 eval accuracy past the training length, `wide` against `ctrl`, at least +10 points.
2. **Transfer (Stage B):** S2 eval reward and tau-bench (retail, airline) pass^1, `wide` against `ctrl`, paired.
3. **No regression**, each arm against the untouched model: BFCL v4 every category within 3 points (the Hebrew
   lane's gate); MMLU-Pro scored by option text within 1 point (letter scoring reported beside it; the 9B showed
   letter scoring can mislead); HumanEval within 3 points; IFEval within 2 points; Hebrew: Global-MMLU he within 1 point.
4. **Mechanism check:** the gate is used (beta above 1) on task tokens and stays near its general-text rate
   elsewhere; per-position accuracy curves to 1,024 events.

## Engineering before the first run (agent time, not GPU time)

| piece | file | check |
|---|---|---|
| data-parallel LoRA trainer with the gate (one bf16 replica per GPU: 54 GB of weights fits a B300 or H200, so no sharding) | `experiments/negeig_27b/train27.py` | CPU smoke on a tiny qwen3_5: losses fall, KL to the untouched model is exactly 0 at step 0, resume reproduces the same losses; on the box: G0 on the 27B, then a 30-step throughput smoke per arm |
| full S1 evaluator (every session, one record per answer) | `eval27.py` | oracle head 45/45, a wrong end-of-turn token fails every answer |
| pair launcher: wide on GPUs 0-3, ctrl on 4-7, same seed, resumes after preemption | `run_pair.sh` | dry run with a stub torchrun |
| widened-beta gate in SGLang's `qwen3_5` Gated DeltaNet | `sglang/` | logits match the HF patch on 32 prompts; bit-identical at W = 0 |
| self-distillation job (chat, Hebrew translation, agentic runs) | `selfdistill/` | fake-server dry run on CPU; spot-read 50 answers per source on the box |
| box create, bootstrap, acceptance gate, preemption watchdog, teardown | `box/` | `bash -n`; acceptance emits ACCEPT_OK before any run |

Checkpoint evals (every 50 steps and at step 0): S1 teacher-forced exact match on a fixed subset (16 sessions per
domain at 32, 256 and 512 events), split at event 256; clean-replay NLL; KL(untouched || current) on replay text;
gate firing rate on S1 and on text. With the Qwen3.8 think-off template every answer, in history or not, follows the
same empty think block, so one forward per session scores each answer exactly as greedy decoding would (under the
canonical tokenization, so a lower bound on string-level greedy).

## Compute and budget (revised 2026-10-01, prices re-checked)

Boxes: two preemptible 8x B300 (288 GB each) on Nebius uk-south1, quota 32 GPUs. Each box runs one wide/ctrl pair on
the same hardware (4 GPUs a run), so every comparison is paired within a box; the training runs resume from their
newest checkpoint after a preemption.

Prices per 8-GPU box-hour, read 2026-10-01 (Nebius from the provider's billing calculator on the exact spec; the web
page "from $0.99" preemptible figure was a floor, not the price):

| provider | 8x B300 preemptible / spot | 8x B300 on demand | 8x H200 |
|---|---|---|---|
| Nebius | **$34.4** | $62.8 (web page: $76 from 2026-10-01) | $19.6 preemptible, $36 on demand |
| Verda | $34.5 | $69 | not checked |
| RunPod | none listed | $55 to $63 | not checked |
| Vast | no 8x B300 offer (8x B200 $62.5, US only) | | |
| Jarvis | no B300 | | $31.9 on demand |

Nebius B300 preemptible ties Verda spot for the cheapest B300. Fallback if B300 preemptible capacity is absent or
preempts repeatedly: Nebius H200 preemptible ($19.6), then Jarvis H200 on demand ($31.9). B300 has about 2.27x the
dense bf16 compute of an H200 (vendor sheets, assess-method), so B300 preemptible is also the cheaper per token.

Recipe changes from the assessments (assess-method.md, assess-speed.md): size Stage A in steps at a batch (700 steps
at 64 S1 sessions a step, pre-registered extension rungs to 1,200 and 1,800), sweep {1e-4, 1e-3} as the first 300
steps of the real runs (box A at one rate, box B at the other), and the trainer now batches by LPT with no padding
(simulated useful fraction 0.55 before, 0.99 after).

Throughput is still estimated (2.85k to 3.8k tokens/s per B300 GPU, from the measured H200 rate); the 30-step smoke on
the box replaces it, and I report the measured hours before Stage A proper.

| scenario (assess-method) | GPU-hours | box-hours | at $34.4 preemptible | at $62.8 on demand |
|---|---|---|---|---|
| Stage A core: setup, distillation, sweep, 2 seeds x 2 arms x 700 steps, evals | 109 | 13.6 | **$470** | $856 |
| Stage A envelope: core plus the 1,200 and 1,800 rungs and the probes | 220 | 27.5 | $946 | $1,727 |

The core is the run I start. The extension rungs and probes are pre-registered but conditional; they come back to the
owner for a go before they spend beyond the core. Stage B is re-planned after the Stage A result, on the same boxes.

## Risks

- The SGLang patch is the one piece we have not built; if it slips, Stage A still runs (training and S1 eval on HF,
  slower evals), Stage B waits for it.
- The post-trained 27B may resist the switch differently from a base model; the sweep and per-checkpoint S1 evals
  show it early.
- Tool use: the office SFT broke BFCL in the Hebrew lane. Here the supervised data has no tool calls and carries
  self-distilled tool-use replay, and the regression gate is checked after Stage A before any RL.
