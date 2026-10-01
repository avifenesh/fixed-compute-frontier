# Stage A run log: Qwen3.8-27B widened-beta retrofit

Times are UTC (Israel is UTC+3). Receipts on the rig: `/data/ai-ml/models/_runs/negeig-27b/box/{a,b}/`.

## Hardware actually used (2026-10-01)

Planned: two preemptible 8x B300 in uk-south1. What happened between 23:25 and 00:00:

- uk-south1 B300 preemptible: box a started, was preempted about 10 minutes later; box b got no capacity (code 8).
- uk-south1 B300 on demand: no capacity (code 8) for two attempts.
- eu-west2 B300: the billing calculator has no active price for the SKU (PrototypeNotFound).
- All four B300 instances were deleted; the provider lists confirm it (teardown.sh TEARDOWN_CONFIRMED).
- At 00:00 Nebius switched to its 2026-10-01 price list (web page: H200 $5.40, B300 $9.50 per GPU-hour on demand) and
  the calculator stopped pricing H200 and B300 for a while. On-demand at the new rate would put Stage A core near
  $1,200, above what the owner approved; preemptible H200 was $19.6 a box-hour before the switch.

Running: two preemptible 8x H200 (141 GB) in eu-north1, `negeig27-h200p-a` and
`negeig27-h200p-b`, created 00:01 and 00:06. The trainer resumes after a preemption;
`box/preempt_watch.sh --restart` watches both. This is the owner's H200 fallback, run preemptible to stay inside the
approved cost.

## Acceptance (box b unless noted)

| check | result |
|---|---|
| real CUDA allocation and GEMM, all 8 GPUs | ok, sm_90, torch 2.14 cu130 (both boxes) |
| fla kernel check, beta in (0, 2) against the fp32 reference | pass (forward and dq/dk/dv rel err about 0.0055, same as beta in (0, 1)) |
| G0 on the 27B | bit-identical at W = 0 (max diff 0.0); random W: beta > 1 on 45% of entries; 48/48 gates get gradient |
| SGLang 0.5.20 gate parity (box a), eager and CUDA graphs | G0 stock repeats (16,444 numbers, 0 differ); G1 zeros server = stock bit for bit, patch engaged; G2 random W: effect corr 0.995, rel err 0.099, top-1 agree 0.982 (stock 0.989); G3 decode = prefill. All PASS in both modes |
| memory, longest training row (9,765 tokens) | 60.6 GiB with gradient checkpointing; out of memory without it, so checkpointing stays on for H200 |
| throughput, full Stage A mix, about 177k tokens a step | 23.0 s a step on 4 GPUs (7.9 to 8.1k tokens/s a run); 12.4 s on 8 GPUs (1.85x); LPT keeps ranks within 0.1% of tokens and 5% of compute time |
| checkpoint eval | about 70 s on 4 GPUs (110 s with the recall guard) |

## Self-distillation (the untouched model's own outputs, SGLang 0.5.20, think-off, vendor non-thinking sampling)

| set | result |
|---|---|
| Hebrew translations of English prompts | 1,708 valid of 1,854 attempts (dropped: numbers changed 79, not Hebrew 47, other 20) |
| chat replay | 7,404 kept of 7,502 (5,944 English, 1,460 Hebrew); dropped: truncated 46, language mismatch 41, refusal 9, other 2 |
| tool replay (agentic pool, train split, 4 samples an item, at most 2 kept an item) | 3,759 rows from 7,988 episodes; solved rate 0.96 to 1.0 on eight kinds, 0.71 on missing_param |
| files | `data/sd/chat_sft.jsonl` sha256 d4707211..., `data/sd/tool_sft.jsonl` sha256 41972da0... |

## Untouched model baseline (eval27, all 1,200 S1 eval sessions, 65,948 answers)

Overall 69.8%; answers after event 256: 69.7%; after event 512: 70.0%. By domain: codetrace 53.3, custody 64.5, fsys
70.2, orders 74.4, toggles 77.1, ops 79.0. Hebrew 72.0, English 69.4. The untouched 27B does not decay with session
length: its errors are per question, flat to 1,024 events. That differs from the 4B and 9B bases, which failed with
length; the past-length gap may open differently here.

Smoke, 20 steps at a 1e-3 warm-up (both arms): S1 70.1 to 80.6 (wide) and 81.1 (ctrl), past 256 events 69.4 to 78.4 and
78.3; KL to the untouched model 0.003 and 0.008; recall 100% at 16k and 32k; tool token agreement 0.99 and 0.98.

## Method changes from the 4B/9B trainer (decided before the sweep)

- Gradient clipping is per group (LoRA and gate, each at 1.0). At step 1 the gate's gradient norm is 17.3 and the LoRA
  norm 0.28; a joint clip would rescale the wide arm's LoRA step by a factor the ctrl arm never sees.
- Batches are drawn by step, the same on any world size, and assigned to ranks longest-first (no padding).
- Tool rows used for the agreement eval are held out of training.

## LR sweep (launched 00:59, both pairs with --steps 700, seed 0, Stage A recipe = train27 defaults)

Box a: wide and ctrl at 1e-4 (`runs/sweep-lr1e-4/`). Box b: wide and ctrl at 1e-3 (`runs/sweep-lr1e-3/`).

### Layout change at step 150: one arm per box (owner approved 2026-10-01 ~04:25 IDT, "sounds like a good deal")

8 GPUs a run measured 1.85x the 4-GPU step rate (12.4 s against 23.0 s), so the owner approved a 4-box layout for about
$35 more and about 3.5 hours sooner. Box c (`negeig27-h200p-c`, eu-north1) came up at 01:30. Box d hit two limits:
eu-north1 allows 3 public IPv4 addresses per tenant region (vpc.ipv4-address.public.count, limit 3), and eu-west1 had no
H200 capacity (NotEnoughResources). Box d (eu-west1) stays STOPPED and the
preemption watcher retries its start every 3 minutes.

At step 150 the 1e-4 pair was split (`migrate_pair.sh`): wide stays on box a with all 8 GPUs, ctrl moved to box c with
all 8 GPUs, both resuming from the step-150 checkpoint (train27 draws the same global batch on any world size; the CPU
tests show a resume on another world size reproduces the losses). The 1e-3 pair stays 4 + 4 on box b until box d starts.

### Pre-registered decision at step 300 (assess-method.md, written here before step 300)

Average the evals at steps 250 and 300.

1. Safety, each arm at its LR. Pass only if all hold: `kl_to_untouched` at most 0.08; replay NLL at most the untouched
   value plus 0.05 nats (untouched 1.9694, so at most 2.0194); tool token agreement at least 0.95; no training-loss spike
   above 3x the running median.
2. Signal, wide against ctrl at the same LR. Present if any holds: `s1:len256:le256` gap of at least 5 points; wide
   answer NLL at least 10% (relative) below ctrl and the gap widening over the last 100 steps; gate rate on S1 tokens at
   least 1.5x the rate on text. Report `s1:len512:gt256` beside it.
3. Choose: 1e-3 if it passes safety and shows signal; 1e-4 if 1e-3 fails safety; if both pass and neither shows signal,
   extend both pairs to 600 steps and read again with the same rule, then 1e-3 if it passes safety, else 1e-4.
4. The winning box continues as seed 0 to step 700. The other box stops its pair, archives it, and starts seed 1 from
   step 0 at the winning LR.

### Decision at step 300 (applied 03:06): 1e-4

Averages of the step-250 and step-300 evals:

| run | S1 all | in length (256-event sessions) | past 256 (512-event sessions) | answer NLL | KL | replay NLL | tool token agree | gate S1 / text |
|---|---|---|---|---|---|---|---|---|
| wide 1e-4 | 90.7 | 91.3 | 88.9 | 0.0638 | 0.0082 | 1.9461 | 0.988 | 0.084 / 0.053 |
| ctrl 1e-4 | 90.4 | 91.2 | 88.8 | 0.0640 | 0.0153 | 1.9460 | 0.983 | n/a |
| wide 1e-3 | 89.2 | 90.0 | 87.3 | 0.0809 | **0.152** | **2.039** | 0.954 | 0.406 / 0.362 |
| ctrl 1e-3 | 92.0 | 92.6 | 90.2 | 0.0572 | 0.068 | 1.944 | 0.967 | n/a |

Rule: 1e-3 fails safety (wide arm: KL above 0.08, replay NLL above 2.019), so the winner is 1e-4. At 1e-4 both arms pass
safety; signal is present only through the gate-rate criterion (S1 1.58x text); the in-length gap (+0.05 points) and the
answer-NLL gap (0.3%) are not. Box b stopped its 1e-3 pair at step 300 (archived in `runs/sweep-lr1e-3/`) and started
seed 1 at 1e-4 (`runs/sweep-lr1e-4/{wide,ctrl}_s1`, 4 + 4 GPUs, box d is still without capacity). Seed 0 at 1e-4
continues to step 700 on boxes a (wide) and c (ctrl).

Reading so far: at 1e-3 with W tied to the LoRA rate, the gate opens on 36% of ordinary text and the wide arm drifts
from the untouched model (KL 0.15) while ctrl reaches the best S1 score (92%); at 1e-4 the two arms are level through
step 400 (wide 92.0, ctrl 92.1; past 256: 90.4 and 90.6). The untouched 27B already handles length (flat 70% to 1,024
events), so the length gap the 4B and 9B showed has no room to open here unless it appears later in training.

### Seed 0 at step 700 (full S1 eval, 1,200 sessions, 65,948 answers)

| | all | in length (<= 256 events) | past 256 | past 512 | 1,024-event sessions past 256 | Hebrew |
|---|---|---|---|---|---|---|
| untouched | 69.8 | 69.8 | 69.7 | 70.0 | 70.0 | 72.0 |
| wide s0 | 93.2 | 94.3 | 92.3 | 91.9 | 92.1 | 93.9 |
| ctrl s0 | 93.6 | 94.7 | 92.8 | 92.5 | 92.7 | 94.0 |

Primary criterion (wide minus ctrl past the training length): -0.4 points against a +10 bar. Both arms gain about 23 points
and generalize from 256-event training sessions to 1,024-event sessions; the gate adds nothing on S1. Seed 1 (box b) tracks
seed 0 (step 200: wide 88.4, ctrl 87.6).

Owner, 04:2x IDT: approved the pre-registered extension, then paused it ("wait"), and redirected: tear down a and c, run
the targeted probe, and before any kill test the product target first, the on-call agent over long sessions, measuring
token usage per task ("it might be the gain, less thinking needed"). Boxes a and c were deleted at 04:36 and 04:42 UTC after
a final pull (TEARDOWN_CONFIRMED from the provider lists). The 1,200 and 1,800 rungs are not run.

Go to Stage B (PLAN.md): wide beats ctrl by at least 10 points past the training length (sessions of 512 and 1,024
events, answers after event 256), mean over two seeds, with every general gate inside its band.

## Probes after Stage A (2026-10-01)

Two probes run before any kill decision, on three preemptible 8x H200 in eu-north1 (the region's 3 public IPv4
addresses): box b (on-call, untouched then ctrl_s0), box e (`negeig27-h200p-e`,
created 06:10, tier-1) and box f (`negeig27-h200p-f`, created 06:38, no capacity
until 06:48, on-call wide_s0). Seed 1 stopped at step 400 on the owner's call (resumable on box b).

Box e acceptance: real CUDA allocation on 8 GPUs; NCCL NVLS on (2.358 against 3.011 ms); G1 PASS; G0 bit-identical at
W = 0 (max diff 0.0); random W puts beta > 1 on 43.6% of entries; 48/48 gates get gradient.

### On-call probe (the product target: an on-call agent over long sessions)

`probe/run_all.sh`. Each arm is served by SGLang 0.5.20 with the merged LoRA and its gate (DP 8, radix cache, qwen3
tool and reasoning parsers) and runs the on-call eval with thinking off and on: 64 sessions, 16 each at 64, 256, 1,024
and 2,048 events, 7,477 turns. The cap is 16,384 tokens per turn with thinking on and 1,024 off; tokens are counted per
task. S1 free generation with thinking off (192 sessions, 14,686 turns) runs beside it. S1 with thinking on was stopped
at 06:4x for the untouched arm: it shared the GPUs with the on-call thinking job, the critical path, and S1 is not the
target. It stays resumable (`run_all.sh --jobs s1gen:on`).

Untouched, partial at 06:30 (thinking off complete later): thinking off 68.0% of action turns right, flat with length
(64 events 68.6, 256 71.1, 1,024 65.7, 2,048 69.5), 14.8% of sessions exact, about 9.0k tokens per task. Thinking on, first
17 sessions (all 64 events): 100% right, 82.4% exact, 11.1k tokens per task of which 9.0k reasoning. The question for
ctrl and wide: does either close the thinking gap with thinking off, or reach the same accuracy with fewer reasoning
tokens.

Untouched thinking off, final (64 sessions): 65.7% (64 events 68.6, 256 71.1, 1,024 64.0, 2,048 65.8), 12.5% of sessions
exact. Thinking on runs about 280 turns for a 2,048-event session at about 2k reasoning tokens a turn, so each arm's
thinking job takes hours, not the hour planned. At 07:09 box f was split so ctrl does not wait for box b: wide_s0 on GPUs
0 to 3 (DP 4, port 30000) and ctrl_s0 on GPUs 4 to 7 (DP 4, port 30001, PROBE_ROOT probe_ctrl), each with its own SGLang
process pattern and a GPU-scoped nvidia-smi so neither arm's shutdown touches the other. Decode here is latency bound
(4 to 6 requests a GPU at about 60 tokens/s each), so the shared box costs each arm little. Wide resumed from its ledger
(engagement ok, 4 gate lines); ctrl serves with no gate lines (sha256 6017470c...). Box b runs the untouched arm only.

### On-call results (stopped 07:45, boxes b and f deleted 07:47, TEARDOWN_CONFIRMED)

Thinking off ran to completion on all three arms (64 sessions, every length to 2,048 events). Thinking on did not
finish on any arm. A 2,048-event session needs about 280 turns, and every arm's reasoning grows turn by turn. Untouched,
median reasoning tokens a turn: 368 at turns 0 to 9, 603 at 10 to 19, 1,103 at 20 to 29 (17% of turns at the 16,384
cap), 5,479 at 30 to 39 (28%), 13,982 at 40 to 49 (46%); late generations took about 270 s each. Wide and ctrl start
the same climb later: over the last 120 generations of the 1,024-event sessions the mean completion was 11.7k tokens
for untouched, 6.0k for wide and 3.3k for ctrl. The runs were stopped with 32 (untouched), 23 (wide) and 23 (ctrl)
sessions finished. `probe/score_partial.py` scores the closed turns of the rest from gens.jsonl; `report.py --partial`
pairs two arms only on the turns both reached, and `--align` cuts all three to their common prefix (about 1,500 of
7,477 turns, events up to about 512). Reports: `/data/ai-ml/models/_runs/negeig-27b/probe-report-20261001/`.

Thinking off (complete):

| arm | action turns right % | 64 | 256 | 1,024 | 2,048 | exact sessions % | total tokens per task |
|---|---|---|---|---|---|---|---|
| untouched | 65.7 | 68.6 | 71.1 | 64.0 | 65.8 | 12.5 | 14.5k |
| ctrl s0 | 83.1 | 88.5 | 89.1 | 81.9 | 82.8 | 26.6 | 3.9k |
| wide s0 | 84.0 | 89.1 | 89.2 | 83.1 | 83.7 | 25.0 | 8.8k |

Thinking on, all three arms cut to the same turns (the length columns cover only the early turns of each session):

| arm | right % | 64 | 256 | 1,024 | 2,048 | reasoning per turn, median / mean | reasoning per task | turns truncated % |
|---|---|---|---|---|---|---|---|---|
| untouched | 95.0 | 100.0 | 96.8 | 92.6 | 93.8 | 570 / 2,235 | 61.8k | 5.4 |
| ctrl s0 | 95.8 | 95.8 | 95.3 | 96.4 | 95.7 | 182 / 883 | 34.1k | 2.2 |
| wide s0 | 90.0 | 96.9 | 88.5 | 92.4 | 86.8 | 545 / 2,597 | 58.7k | 3.7 |

Paired, 95% bootstrap over sessions:

| comparison | metric | difference | interval |
|---|---|---|---|
| wide - ctrl, thinking off | accuracy | +0.95 points | [-1.45, 3.10] |
| wide - ctrl, thinking off | total tokens per task | +4,915 | [3,150, 6,880] |
| wide - ctrl, thinking on | accuracy | -5.79 points | [-9.38, -2.19] |
| wide - ctrl, thinking on | reasoning per turn | +1,715 | [1,188, 2,216] |
| wide - ctrl, thinking on | reasoning per turn, both right | +932 | [666, 1,200] |
| wide - ctrl, thinking on | reasoning per task | +24.7k | [15.5k, 33.8k] |
| ctrl - untouched, thinking on (common prefix) | accuracy | +0.83 points | [-2.21, 3.81] |
| ctrl - untouched, thinking on (common prefix) | reasoning per task | -27.7k | [-41.7k, -14.5k] |
| wide - ctrl, S1 free generation, thinking off (192 sessions) | accuracy | +0.01 points | [-0.48, 0.50] |

By position, each arm on its own closed turns, thinking on, events 129 to 256: untouched 81.2%, ctrl 92.4%, wide 83.9%;
events 257 to 512: 59.0% (n 87), 96.0% (n 21), 67.9% (n 14).

Reading: the token saving is real, and it comes from the LoRA, not the gate. On the common prefix ctrl matches untouched
accuracy with 45% less reasoning per task, and it holds accuracy deeper into the session where untouched degenerates.
Wide reasons about as much as untouched and is 5.8 points below ctrl. With thinking off, wide and ctrl tie at every
length to 2,048 events, and wide writes longer replies. On the product target the gate adds no accuracy and costs
tokens. One seed per arm.

### Tier-1 mechanism probe (box e)

`probe/tier1/run_t1.sh`: wide and ctrl trained on the tier-1 state-tracking tasks (parity, swaps, code swaps) with LM
replay, evaluated on windows of 1x, 4x and 16x the training length. D = 100 x (wide - ctrl) of accuracy in the 4x window.
Step 1,200: GO if D >= 10 and the gate is used; KILL if D < 10 and ctrl reaches 0.60; otherwise extend. Step 2,400: GO if
D >= 10 with the gate used, else KILL. The 4B reference is wide 0.784 against ctrl 0.300. Step 0: both arms 0.300 at 4x
(W = 0, identical by construction). About 6.6 s a step.

**Rung 1, step 1,200 (`decide.py`, 07:59): GO.** Test set (128 sequences per task of 1,024 steps), paired bootstrap
over sequences:

| window | untouched | ctrl s0 | wide s0 | D, points | interval |
|---|---|---|---|---|---|
| 1x tier-1 | 0.311 | 0.689 | 0.838 | +14.9 | [14.0, 15.8] |
| 4x tier-1 | 0.299 | 0.322 | 0.483 | **+16.1** | [15.4, 16.8] |
| 16x tier-1 | 0.299 | 0.299 | 0.306 | +0.7 | [0.4, 1.0] |
| 4x parity | 0.501 | 0.500 | 0.937 | +43.7 | [42.0, 45.3] |
| 4x swap | 0.196 | 0.199 | 0.222 | +2.4 | [1.4, 3.3] |
| 4x codeswap | 0.199 | 0.267 | 0.289 | +2.2 | [0.9, 3.6] |
| 4x tier 2 | 0.268 | 0.267 | 0.267 | +0.0 | [-0.5, 0.5] |

Gate used: beta above 1 on 41.7% of task tokens in the busiest layer, 16.6% averaged over the 48 layers; on validation,
17.7% of task tokens against 9.3% of text. Step-0 identity holds (0 differing entries), configs differ only in the arm.
Switch step (validation 1x at least 0.95): wide parity 350, codeswap 600, swap none; ctrl parity none, codeswap 800, swap
none. KL to the untouched model 0.195 (wide) and 0.168 (ctrl); held-out NLL 1.960 and 1.955 (untouched 2.138).

The mechanism is there on the 27B: the gate gives it parity past the training length, which the LoRA alone never
reaches (ctrl stays at chance, 0.500, at 4x). It is narrower than on the 4B in the same 1,200 steps and the same token
stream (4B seed 0: wide 0.684, ctrl 0.301): swap did not switch in either arm, and codeswap switched in both, so the
4x gain is almost all parity. Nothing extrapolates to 16x. One seed. The rule stops at GO, so rung 2 was not run;
`ckpt_001200.pt` of both arms is kept on the rig for a continuation.

## Decision (2026-10-01, 11:10 IDT)

The retrofit stops here for Qwen3.8-27B. Under the pre-registered rule Stage B needs wide to beat ctrl by 10 points on
S1 past the training length; it is -0.4. The owner asked for the product target before any kill: on the on-call agent
over long sessions the gate adds no accuracy with thinking off (+0.95 [-1.45, 3.10]), costs 5.8 points with thinking on
[-9.38, -2.19], and costs tokens in both modes. The tier-1 GO shows the mechanism is real on this model; our workloads do
not use it, because the untouched 27B already tracks state over 1,024 events (S1 flat at 70%) and the LoRA alone takes
both S1 and on-call where they need to go.

What carries forward is the LoRA, not the gate: on-call, ctrl lifts thinking-off accuracy from 65.7% to 83.1% at 3.9k
tokens per task (untouched 14.5k), and with thinking on it matches untouched accuracy at 27.7k fewer reasoning tokens
per task [-41.7k, -14.5k]. That is a fine-tuning result on one seed, not a gate result.
