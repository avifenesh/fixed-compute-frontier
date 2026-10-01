# Method assessment: Stage A and Stage B for the Qwen3.8-27B retrofit

Date 2026-10-01. Scope: the training method in `PLAN.md` (signed off 2026-10-01, two preemptible 8x B300 boxes).
No machine was rented or started for this assessment. Every external claim links a page that was opened in this
session (arXiv links were read as abstract pages, blog and vendor pages in full). Repo evidence is cited as
`file:line` (paths are in this worktree unless they start with `research/hebrew-`, the Hebrew lane write-ups in a private repo). Thresholds marked "mine" are
proposals, not measurements.

## Bottom line

1. Keep AdamW (0.9, 0.95), weight decay 0, clip 1.0. The evidence for Muon-family LoRA optimizers is small-scale or
   mixed, so they enter as a probe with their own LR pair, not as the default.
2. Sweep {1e-4, 1e-3}, not {1e-4, 3e-4}. At 4B, 3e-4 behaved like 1e-4 and 1e-3 was the 3x faster rate.
3. Size Stage A in steps at a batch, not in tokens. `PLAN.md` says 75M tokens a run. At `--s1_per_step 64` that is
   about 400 steps, which I judge to be below the point where the 4B and 9B runs switched. That is a judgment, not a
   measurement: the batch evidence counts same-task sequences, and 64 S1 sessions over six domains sit between the
   4B mixed batch that never switched swap and the runs that did. Plan 700 steps with a pre-registered extension to
   1,200 and 1,800 (the 9B continuation, kept as a ceiling), and a conditional batch-128 probe (P5).
4. Lower only the LM replay weight (to 0.15) and keep chat and tool at the trainer default 0.5: a 54% replay share
   of the loss weight, against 60% at the defaults and 50% in the 4B runs. Clean replay delayed the hardest switch at
   4B, but nothing measured supports cutting the chat and tool replay that protects BFCL and IFEval, and the Thinking
   Machines 30% is a data fraction in an SFT mix, not a share of loss weight. The lower weights first proposed (0.10
   chat, 0.25 tool, a 33% share) become probe P1. This is a judgment the owner can flip; P1 prices it.
5. Stage B stays GRPO on the Hebrew lane recipe, behind a cheap reward-variance census and a 50-step pilot. The
   Hebrew lane's RL moved its own reward lane (+2 to +4 points paired against the base, 80 steps, one seed) and moved
   BFCL nothing. A wide-vs-ctrl difference between two RL arms is a smaller effect than that, so Stage B is where the
   plan has the least statistical power.
6. Hardware: the B300 plan already cuts Stage A wall time from 16.5 h (the RTX PRO 6000 plan, `PLAN.md` at commit
   382262c) to 6 h by its own estimate. Better hardware can halve the time in two ways, and only one was counted.
   Per-GPU speed (B300 against H200, about 2.3x on the vendor BF16 sheets) is unmeasured. GPUs per run is the other
   lever: the plan runs 4 of a box's 8 GPUs per run and holds 16 of a 32-GPU quota, so a 4-box layout at 8 GPUs per
   run halves the training step time at about equal GPU-hours. Setup, distillation and evals do not shrink, so Stage A
   wall time falls from 7.3 h to 5.7 h (-23%) for about +22% dollars. A bigger budget therefore buys some speed as
   well as correctness (batch, steps, seeds, probes, Stage B depth). The first smoke decides: see "Budget rule".

## Decision table

| # | Question | Recommendation | Evidence | Confidence |
|---|---|---|---|---|
| 1 | Optimizer, and the gate's W | **AdamW (0.9, 0.95), wd 0, clip 1.0** as the primary. W stays a separate fp32 zero-init group on AdamW, LR tied to the LoRA LR in the sweep, one probe at 3x. LoRA+ (higher LR on B) is a probe with ratio 4 (my choice). Schedule-free: no. A Muon-family LoRA optimizer (PoLoRA is the candidate) only as a probe with its own 2-LR pair; an untuned single run says nothing. | Muon on Adam-pretrained models degrades, scaling with update strength, and LoRA narrows the gap ([2605.10468](https://arxiv.org/abs/2605.10468)). Muon per LoRA factor does not consistently beat Adam, and PoLoRA, a preconditioned LoRA optimizer, reports reaching tuned Adam's held-out loss in 1.2 to 1.7x fewer steps on 1B to 8B code and math instruction tuning ([2607.17620](https://arxiv.org/abs/2607.17620)). sMuon reports "moderate performance improvements" but its results are "model- and eval-dependent" ([2608.14492](https://arxiv.org/abs/2608.14492)). LoRA-specific Muon variants exist ([2606.12921](https://arxiv.org/abs/2606.12921), [2609.02734](https://arxiv.org/abs/2609.02734), [2507.12142](https://arxiv.org/abs/2507.12142)) but the abstracts report no 27B hybrid SFT (TinyShakespeare, LLM and diffusion tasks, 1B to 8B code and math). Muon's ~2x gain is pretraining ([2502.16982](https://arxiv.org/abs/2502.16982)). Muon fails under RLVR ([2605.19282](https://arxiv.org/abs/2605.19282)). LoRA+ claims 1 to 2% and up to ~2x speed-up ([2402.12354](https://arxiv.org/abs/2402.12354)). Schedule-free needs no schedule but no LoRA or plateau-switch evidence ([2405.15682](https://arxiv.org/abs/2405.15682)). Repo config: `experiments/negeig_retrofit/train.py:248`, `experiments/negeig_27b/train27.py:316-317`. The zero-init W is a B-like factor, so LoRA+ logic would favor a higher W rate, but that is speculative here: the untied 2e-4 / 5e-4 recipe at `results/negeig-retrofit-preregistration.md:396-400` belongs to the campaign the owner stopped and archived (`:173-174`), the 4B main runs used one tied 1e-4 (`:25-26`), and in the 400-step sweep the untied rows (LoRA 1e-4 with W 3e-5, 3e-4, 1e-3) did not switch parity (`:189-193`, one seed, noisy). Tying W is the measured choice; P2 stays a probe. | Medium. This is a checked choice, not an inherited one. I cannot rule out a Muon variant winning here. No evidence exists for Muon on a GDN retrofit with a zero-init gate. |
| 2 | LR sweep in about 2 box-hours | **{1e-4, 1e-3}, wide and ctrl, one LR per box, 300 steps at `s1_per_step 64`, constant after 30 warmup steps.** About 1 h per box on B300 (estimate). The sweep is the first 300 steps of the real runs: launch both pairs with `--steps 700` so the winner simply continues (the schedule is constant, `train27.py:363-365`); the recipe lists the `complete.json` and `done` traps. Decide at step 300 with the rule in the recipe: safety bars first, leading indicators second. Do not rank LRs by "did it switch yet". | 4B sweep: 3e-5 no switch by 800, 1e-4 switch at 600, 3e-4 switch at 600, 1e-3 switch by 200, plateau shortened about 3x (`results/negeig-retrofit-preregistration.md:102-105`). A 400-step sweep was "too noisy to rank learning rates" (`:178-196`). LoRA's optimum is about 10x the full fine-tuning LR and nearly rank-independent ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)), but no 27B full fine-tuning LR exists here to anchor that rule, so {1e-4, 1e-3} is simply the two measured 4B points, 10x apart (swap-only sweep at batch 64: 1e-4 switched at step 600, 1e-3 by step 200, `prereg:102-105`). 1e-4 is the LoRA rate the Hebrew stage 1 used on the 27B, with rank 32 (`research/hebrew-stage1-20260915/LANE.md:48`). The 1e-3 general-eval cost was never measured at 4B (`prereg:102-105`), hence the KL bar. | Medium on the bracket (two measured 4B points, no 27B anchor), medium on the decision thresholds (mine). |
| 3 | LoRA targets, rank, LoRA vs full fine-tuning | **LoRA r16 alpha 32 on the GDN projections only** (`in_proj_qkv`, `in_proj_a`, `in_proj_b`, `out_proj`) for the wide-vs-ctrl comparison. **Not** full fine-tuning. Conditional probe P3: add `gate/up/down` (MLP) if ctrl or wide training loss on S1 answers stalls while replay KL stays low. Rank 16 stays. | LoRA learns less and forgets less than full fine-tuning; full fine-tuning learns perturbations of 10 to 100x higher rank ([2405.09673](https://arxiv.org/abs/2405.09673)). LoRA still forgets, growing with parameters tuned and steps, and early stopping does not fix it ([2401.05605](https://arxiv.org/abs/2401.05605)). At matched parameters all-layer or MLP LoRA beats attention-only, and capacity limits show up on large datasets ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)); GDN-only is the analog of attention-only here, so P3 is the honest check. CoT-SFT breaks long-range recall in hybrids (HypeNet, Jet-Nemotron) through W_Q and W_K ([2606.11052](https://arxiv.org/abs/2606.11052)); this run trains on short answers and leaves the attention layers untouched, which I expect to limit that risk (my inference, not measured), and a recall guard watches it. Targets in use: `experiments/negeig_retrofit/train.py:29,46-47`. | Medium. Rank and targets were not varied at 4B or 9B, so this is literature plus the existing 4B/9B setup. |
| 4 | Curriculum, loss, replay ratios | **No short-first phase.** **Answers-only loss** on the answer tokens plus `<|im_end|>` (already the trainer's loss). **`--s1_per_step 64`** (default 32). **Replay weights 0.5 chat / 0.5 tool / 0.15 LM** (trainer default 0.5 each), a 54% replay share of the loss weight (the 4B runs weighted replay 1:1, a 50% share). Keep 8+8+8 replay sequences per step so the anchor stays present. Only the LM weight is lowered, because the measured delay came from clean LM replay; the lower chat and tool weights first proposed (0.10 / 0.25, a 33% share) have no measurement behind them and are probe P1. | The 4B clean-replay runs switched swap in 0 of 3 seeds in 1,200 steps, against 3 of 3 with WikiText (`prereg:88-93`, `:163-168`). In a tiny from-scratch swap-only GDN, batch 16 never switched at any budget and batch 64 and 256 did (`prereg:209-227`). That batch counts same-task sequences, and the 4B main runs had 64 swap sequences a step (`prereg:25-26`), whereas 64 S1 sessions over six domains is about 11 per domain, between the 4B mixed batch 16 that never switched swap (about 3 per task) and the 64-per-task runs that did. So 64 is a floor, not proof, and probe P5 tests 128. The tracking horizon is about 2x the training length (`prereg:154`), so long sessions must be in the mix from step 0: train 32 to 256 events, eval to 1,024 (`DATA.md:31`). Own-policy chat samples worked better as background data than a larger model's samples; at least 30% chat data preserved most of IF-eval, but no mix kept the original score, and SFT on a model's pure self-samples still degraded IF-eval at every practical LR, because finite-batch noise turns it off-policy over time ([On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)). That 30% is a fraction of the SFT data mix, not a share of loss weight, so it does not transfer to a weight: chat at 0.10 would be 6.7% of the loss weight (0.10 of 1.5). The only replay dose measured on this method is the 4B trainer's 1:1 weighting (`experiments/negeig_retrofit/train.py:296-325`), so the main run keeps the trainer default on the chat and tool sets, whose job is protecting IFEval and BFCL (the office SFT broke BFCL in the Hebrew lane, `PLAN.md:98-99`), and lowers only the LM set (FineWeb, code), the foreign part and the one whose clean form delayed the switch at 4B (`prereg:88-93`, `DATA.md:10-11`). The cost is real: the 4B delay occurred at about this share, so the extension ladder may be needed, and P1 measures whether the lower chat and tool weights buy an earlier switch at acceptable KL. Expect some regression anyway; that is why the end gates and the distillation repair exist. Replay of earlier data with LR rewarm and redecay matches full retraining in continual pretraining at 405M parameters ([2403.08763](https://arxiv.org/abs/2403.08763); a pretraining result, cited for the replay principle only). Trainer defaults: `experiments/negeig_27b/train27.py:321-328`. | Medium-low on batch 64 (the evidence counts same-task sequences; probe P5 tests 128) and medium on no curriculum. Low-medium on the replay weights: chat and tool sit on the measured 4B share, while the lower LM weight and the lower chat and tool weights of probe P1 are unmeasured (mine). Whether 32 sessions (about 550 answers) equal the 4B "batch 64" regime is unknown, hence the floor of 64. |
| 5 | Stage B: GRPO vs on-policy distillation vs DPO, Hebrew lane lessons | **GRPO on the Hebrew recipe** (`research/hebrew-rl-20260923/LANE.md:1013-1023`), gated by (a) a reward-variance census on each arm's Stage A adapter and (b) a 50-step pilot on the wide arm. **On-policy distillation only as the repair tool** if Stage A regressed BFCL or IFEval (teacher = untouched model, prompts = tool and chat replay). **DPO: not recommended.** If the pilot reward is flat, fall back to rejection-sampling SFT on S2 (own successful rollouts). Serve the adapter in rollout, no CPU offload on 288 GB cards. | Hebrew lane: RL moved BFCL nothing ("every RL row within about 2 points" of the SFT line, `LANE.md:175`), but it did move its own reward lane: run 4, from the base, 80 steps, paired against the base, office dev90 +3.92 [+2.56, +5.35], v1 held-out +3.01 [+1.67, +4.31], hard held-out +2.12 [+0.72, +3.85] (`LANE.md:193-215`). Every RL run was one seed at about 70 to 80 steps, about 5 min a step with offload weight pushes (`:1119`), smoke step 670 to 698 s. SFT with LoRA was best in distribution in 15 of 18 tool-calling settings, GRPO won transfer by under 1 point on average, SFT then GRPO was rarely best ([2609.17848](https://arxiv.org/abs/2609.17848)). On-policy RL forgets less, biased to KL-minimal solutions ([2509.04259](https://arxiv.org/abs/2509.04259)). LoRA RL matches full fine-tuning even at rank 1 ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)). Distillation restored IF-eval from 79 to 83 after domain SFT (base 85), at a fraction of SFT and RL cost ([On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/); method: [GKD 2306.13649](https://arxiv.org/abs/2306.13649)). Recipe pieces: no std normalization, token-mean clip-higher, dynamic filter on the outcome ([2503.20783](https://arxiv.org/abs/2503.20783), [2503.14476](https://arxiv.org/abs/2503.14476), [2609.13866](https://arxiv.org/abs/2609.13866), origin [2402.03300](https://arxiv.org/abs/2402.03300)). DPO ([2305.18290](https://arxiv.org/abs/2305.18290)) is offline on pairs, with no source here covering long multi-turn tool episodes. | Medium-low. RL does move a reward on this recipe (run 4), so the open questions are whether S2 reward moves and whether wide and ctrl differ under RL, a smaller effect between two RL arms that one seed cannot resolve; the census and pilot measure the first before the money is spent. OPD cannot test transfer: no teacher has the skill. |
| 6 | Checkpoint evaluation | **Every 50 steps** (the trainer already saves and evals then): S1 exact match on the fixed subset, **plus answer NLL** (continuous, from the same logits), replay NLL, `kl_to_untouched`, gate rate on S1 and text, **plus a tool-replay top-1 agreement proxy** (adapter on vs off, same forward, 64 fixed items) **and a recall guard** every 200 steps (16 needle items at 16k and 32k). Abort and continue rules below. Full S1 set, BFCL and the other gates only at ladder rungs and the end. | Exact match is a threshold metric that hides progress before a switch; continuous metrics change smoothly, the general point of [2304.15004](https://arxiv.org/abs/2304.15004) (an analogy: that paper is about scale, not training steps). The switch is abrupt and the plateau is flat in accuracy (`prereg:209-227`, `:178-196`). Forgetting is not avoided by early stopping ([2401.05605](https://arxiv.org/abs/2401.05605)), so early stop cannot be the regression control; gates at the end are. KL on clean replay was 0.036 to 0.046 at 4B and 0.24 to 0.33 with WikiText (`prereg:132-136`), which anchors the bars. Existing evals: `PLAN.md:60-64`, `train27.py:220-287` (exact match, replay NLL, KL, gate rate; answer NLL is not there yet). Recall guard: [2606.11052](https://arxiv.org/abs/2606.11052). | Medium. The tool proxy is not validated against BFCL. KL 0.08 and 0.15 are my bars. |

## Hardware and budget: where faster hardware pays

### What the hardware changes

- The wall time of a run is steps times step time, and step time falls two ways: tokens per second per GPU, and GPUs
  per run. The plan runs 4 of a box's 8 GPUs per run and holds 16 of the 32-GPU quota (`PLAN.md:68,71`), so GPUs per
  run is an open lever. The trainer is data parallel with one bf16 replica per GPU and all-reduces only the LoRA and
  gate gradients, so doubling the GPUs per run halves the training step time at about equal GPU-hours. Limits: per-step
  sequence counts must divide by the world size (`train27.py:348-349`: world 16 fails for 8 replay sequences, world 8
  works); a run then owns a whole box, so wide and ctrl sit on separate boxes (launch them together from one image,
  since the within-box pairing of `PLAN.md:71` is lost); each rank gets 8 S1 sessions instead of 16, so length
  imbalance between ranks grows and the gain is under 2x; setup, distillation and final evals do not shrink; a
  checkpoint cannot resume at another world size (the RNG state is stored per rank, `train27.py:395-404`), so the layout
  is fixed at launch, extension rungs included; `run_pair.sh` hard-codes the pair (at `GPUS_PER_RUN=8` it would put
  ctrl on GPUs 8 to 15), so a one-arm-per-box launcher is needed; and the smoke must time 4 and 8 GPUs on the same box
  before anyone counts on the 2x.
- Measured 27B LoRA training speed on Hopper: 1.6k tok/s per H100 and 1.9k per H200
  (`research/hebrew-stage1-20260915/LANE.md:46-52,86-92`). The two numbers come from different runs (8x H100
  and 4x H200, different setups), so the 1.19x is indicative only. H100 and H200 have the same compute, which suggests
  a compute-bound job (my inference).
- B300 is not measured. Vendor figures opened: 288 GB, 8 TB/s, FP8 dense 5 PFLOPS, NVFP4 dense 15 PFLOPS
  ([RunPod B300 guide](https://www.runpod.io/articles/guides/nvidia-b300), [RunPod rent](https://www.runpod.io/articles/rent/b300)).
  B300 BF16 dense is 2,250 TFLOPS on the [Flopper B300 sheet](https://flopper.io/gpu/nvidia-b300-sxm-288gb)
  (4,500 with 2:4 sparsity), and [Nebius](https://nebius.com/compute/b300) lists 36 PFLOPS BF16 for 8 GPUs, labeled
  sparse, which is the same 4.5 a GPU. H200 is 989 dense (NVIDIA's 1,979 is footnoted as with sparsity,
  [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/)). The ratio is 2.27x, not the 2.5x `PLAN.md:74-76`
  assumes. At equal utilization that gives 3.6k tok/s per GPU from the H100 figure (1.6k) and 4.3k from the H200
  figure (1.9k), against `PLAN.md`'s 4k (16k per 4-GPU run). The owner's test, twice the H200 speed, is **3.8k tok/s
  per GPU**, inside that range, so it is a coin flip until the smoke runs, and Blackwell has to reach the Hopper
  utilization to get there.
- Other checks that can cost speed on B300 and are unverified: fla and Triton kernels for the Gated DeltaNet on the
  B300 architecture (sm_103), and the torch build. If the GDN kernels fall back to the torch path, the speed-up
  vanishes. The G0 and the 30-step smoke cover this.
- Stage B gains, but for a narrower reason than offload. The Hebrew lane ran on 96 GB cards (RTX PRO 6000: 96 GB,
  1.6 TB/s, BF16 dense 500 TFLOPS on a [third-party sheet](https://flopper.io/gpu/nvidia-rtx-pro-6000-blackwell-server-edition/spec-sheet.pdf))
  with FSDP2 and CPU offload, and paid about 5 min a step in weight pushes from a CPU-side LoRA merge (`LANE.md:1118-1120`).
  Its own notes call that fixable ("with 8 x 96 GB there is room to keep them on the cards"), so removing offload is
  not something only B300 does. What B300 does change is decode, which is memory-bound and where it has about 5x the
  bandwidth (8 TB/s against 1.6 TB/s), plus KV room for rollouts. I plan 3 to 5 min a step (unmeasured).

### Sizing correction: steps at a batch, not tokens

`PLAN.md:83` budgets "about 75M tokens a run". One step at `s1_per_step 32` is about 95k tokens (32 sessions at a
2.57k-token mean plus replay), so 75M is about 790 steps of a batch that is below the regime the evidence supports.
At `s1_per_step 64` a step is about 190k tokens and 75M is about 400 steps. The plateau is a step count at a batch
size, and the evidence is thinner than one rule. A tiny from-scratch swap-only GDN never switched at batch 16 and did
at 64 and 256 (`prereg:209-227`). The 4B swap-only LR sweep at batch 64 switched at step 600 (1e-4) and by step 200
(1e-3) (`prereg:102-105`). At 4B, swap switched in 3 of 3 seeds with WikiText replay and 0 of 3 with clean replay
within 1,200 steps (`prereg:88-93`). The 9B main runs had not switched swap at 1,200 steps and did after 600 more
(`prereg:72-79`), and the 9B clean-replay run switched at step 1,200. So 1,800 steps is the 9B's continuation, not a
clean-replay figure, and here it is the pre-registered ceiling. Whether 400 steps is too few at 27B is a judgment: a
post-trained 27B at 1e-3 may switch sooner, and six S1 domains at 64 sessions give about 11 sessions per domain, which
sits between the 4B mixed batch 16 that never switched swap (about 3 per task) and the 64-per-task runs that did.
Budget in steps: **700 at batch 64 (133M tokens), extension rungs at 1,200 (228M) and 1,800 (342M)**. Doubling the
batch doubles tokens per step, so this is the same money per step as before, and the extra spend is the extra
tokens. LoRA is also less tolerant of very large batches ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)),
so do not go above 128 sessions without a probe (P5 is that probe).

### Scenarios (estimates; every time and dollar figure scales with the measured tok/s)

Assumptions, all from `PLAN.md` and the arithmetic above: 16k tok/s per 4-GPU run, so a step at batch 64 is about
12 s; 300 steps = 1.0 h, 400 = 1.3 h, 500 = 1.7 h, 600 = 2.0 h, 700 = 2.3 h. Fixed cost per Stage A, both boxes:
setup and G0 and smoke 1.5 h, self-distillation 0.5 h, final evals 2 h (4 wall h, 8 box-h). Prices per GPU-hour:
$0.99 (Nebius preemptible "from", `PLAN.md:69`, a floor and not a quote), $4.31 (Verda B300 spot, on-demand $8.62,
[verda.com/b300](https://verda.com/b300), read 2026-10-01) and $7.85 (Nebius on-demand, `PLAN.md:68`). RunPod lists
B300 at $6.94 (Community) and $7.89 (Secure), on demand
([RunPod guide](https://www.runpod.io/articles/guides/nvidia-b300)). `PLAN.md` priced its middle case at $4.

| scenario | wall h | box-h | GPU-h | $ at 0.99 / 4.31 / 7.85 |
|---|---|---|---|---|
| PLAN as written (75M tokens a run) | 6.0 | 12 | 96 | 95 / 414 / 754 |
| **A-core**: sweep 300 steps, seed 0 continues to 700, seed 1 fresh to 700 | 7.3 | 13.6 | 109 | 108 / 470 / 856 |
| **A-core, 4-box layout** (8 GPUs per run, one run per box, the whole 32-GPU quota) | 5.7 | 16.6 | 133 | 132 / 573 / 1,043 |
| + extension rung to 1,200 (triggered) | 9.0 | 17 | 135 | 134 / 582 / 1,060 |
| + extension rung to 1,800 (triggered) | 11.0 | 21 | 167 | 165 / 720 / 1,311 |
| + probe box P1 to P4 (third box, 1 h setup + 2 waves of 1 h, runs beside the sweep and the seeds) | no change | +3 | +24 | +24 / +103 / +188 |
| + third seed pair at the winner LR (triggered, 2.3 box-h) | no change | +2.3 | +18 | +18 / +78 / +141 |
| + probe P5 (batch 128, wide-only, 400 steps on all 8 GPUs of a box, triggered) | no change | +1.3 | +11 | +10 / +46 / +83 |
| **Stage A envelope, every trigger fires** (2-box layout, P5 included) | 11.0 | 27.5 | 220 | 218 / 948 / 1,727 |

Rows other than the 4-box row use two boxes with one pair each. The H200 fallback (`PLAN.md:88-89`) scales every
block by 2.5x, which is too coarse, because setup (1.5 h) and distillation (0.5 h) are not compute-bound. Scaling only
training and evals gives 15.3 h for A-core at 2.5x, or 13.2 h at 2.1x (the measured 1.9k tok/s per H200 GPU against
the 4k estimate), against 7.3 h on B300: about 13 to 15 h, not 18 h. That ratio is the per-GPU half of the "halving"
the owner asked about, and it only holds if the B300 smoke confirms it.

The 4-box row is the GPUs-per-run half, and it shows what halving buys and what it does not. Training wall time
halves at the same box-hours (the 300-step sweep 1.0 h to 0.5 h, the 700-step run 2.3 h to 1.2 h, the 1,200 and 1,800
rungs from 1.7 h and 2.0 h to 0.8 h and 1.0 h), but setup, distillation and evals stay, so Stage A falls from 7.3 h
to 5.7 h (-23%) for about +22% dollars (two more boxes idle through their 1.5 h setup, 3 box-h). The layout uses the
whole 32-GPU quota, so the probe box needs a quota raise or waits for boxes to free. Stage B already runs 8 GPUs per
run, so more boxes there buy parallel runs (seeds), not faster runs.

Stage B (8 GPUs per run, one box per run, step time is the unmeasured input; 4 min a step used here, scale
linearly for 3 to 5):

| scope | run-steps | box-h at 4 min | GPU-h | $ at 0.99 / 4.31 / 7.85 |
|---|---|---|---|---|
| B-min: 2 arms x 1 seed x 75 steps (Hebrew lane size) | 150 | 10 | 80 | 79 / 345 / 628 |
| B+: 2 arms x 2 seeds x 200 steps | 800 | 53 | 427 | 423 / 1,840 / 3,352 |

### What the extra budget should buy, in order

1. **Steps and batch in the evidence regime (A-core).** This is the cheapest correctness purchase: about +$13 to +$102
   over the literal plan. It turns "75M tokens" into "700 steps at batch 64".
2. **The probe box (+3 box-h, +$24 to +$188).** It runs during the sweep and the seed runs, so its answers arrive in
   time to set the extension rung and Stage B. P1 the lower chat and tool replay weights (0.10 / 0.25) against the
   main run's 0.5 / 0.5 / 0.15. P2 W LR at 3x. P3 GDN + MLP targets. P4 LoRA+ ratio 4. All wide-only at the sweep's
   best LR, 400 steps, compared with the main wide run by switch step, answer NLL and KL. P5 (batch 128, triggered)
   is in the recipe. A Muon-family probe needs its own LR pair, so it goes last and only if a box is idle.
3. **Halved training time, if the owner wants it: the 4-box layout (+24 GPU-h, so +$24 to +$188 over A-core, 1.6 h
   less wall time).** Buy it only after the smoke measures 8-GPU scaling (Budget rule), because the layout cannot be
   changed mid-run and the launcher does not exist yet.
4. **The reserve ladder to 1,800 steps, pre-authorized.** A null is declared only after 1,800 steps: the 9B needed
   +600 steps beyond 1,200 and its control stayed flat (`prereg:72-79`). A null at 400 or 700 steps says nothing.
5. **Stage B depth.** This is the largest line and the weakest point. One seed and 70 steps cannot separate wide from
   ctrl on transfer. Two seeds per arm is the minimum for a paired claim. Fund it after the Stage A go/no-go, sized
   by the measured step time and the pilot slope. A flat pilot means change the method (rejection-sampling SFT or
   harder S2 items), not add steps. More boxes here buy parallel runs, not faster ones, because each run already
   uses all 8 GPUs of its box.
6. **Not worth buying:** full fine-tuning, a bigger rank, a wider LR grid at 27B (the 4B grid already bracketed it),
   more replay sources, an H200 fallback run as a speed measure (it is the slow arm).

### Budget rule (measure first, then decide)

The 30-step smoke on the box reports tok/s per GPU at real S1 session lengths (4k to 8k tokens), memory peak, and
G0 (bit-identical at W = 0, gradient in all 48 GDN layers). It also times the same batch at 4 GPUs and at 8 GPUs on one box, so the GPUs-per-run lever is measured, not assumed. Then:

| measured tok/s per B300 GPU | action |
|---|---|
| 3.8k or more (2x the H200's 1.9k) | approve the Stage A envelope above and plan Stage B at B+ after the go/no-go |
| 1.9k to 3.8k | run A-core, keep extensions and the probe box, decide them from the result, plan Stage B at B-min plus the pilot |
| below 1.9k | B300 is no faster than H200 in practice: no budget increase, report to the owner with the numbers |

GPUs per run (thresholds mine): if 8 GPUs reach at least 1.7x the 4-GPU step rate on the same box, the 4-box layout is the way to buy the halved time, at about +22% dollars for -23% Stage A wall time (Scenarios). Below 1.5x, stay on two boxes.

Before any rental: price the job across Verda, Nebius, Jarvis, Vast and RunPod, spot and on-demand (repo rule),
and accept a box only after a real CUDA allocation succeeds before the first byte is staged. Tag every pod with the
lane name and destroy it when finished.

## Proposed Stage A recipe

**Config (both arms identical except the gate):**
`--rank 16 --sched constant --warmup 30 --s1_per_step 64 --chat_per_step 8 --tool_per_step 8 --lm_per_step 8
--w_s1 1.0 --w_chat 0.5 --w_tool 0.5 --w_lm 0.15 --eval_every 50`, AdamW (0.9, 0.95), wd 0, clip 1.0, bf16, W in
its own fp32 group with `--w_lr` equal to `--lr`. LoRA on GDN `in_proj_qkv`, `in_proj_a`, `in_proj_b`, `out_proj`.
Answers-only loss. No short-first phase. Launch with `run_pair.sh` (wide on GPUs 0 to 3, ctrl on 4 to 7, `--resume`); that is the 2-box layout, and the 4-box layout needs a one-arm-per-box launcher.

**Sweep.** Box A runs the pair at 1e-4, box B at 1e-3. Launch both with `--steps 700`, decide at step 300 and let
the winner continue: the schedule is constant after warmup, so `--steps` does not change the first 300 steps
(`train27.py:363-365`). Do not launch with `--steps 300` and extend afterwards: `run_pair.sh` skips any run directory
that holds `complete.json` (`run_pair.sh:28`), and `box/resume_wrapper.sh` writes `done` on exit 0 and then refuses
to restart (`resume_wrapper.sh:54,104`). To extend a finished run (the 1,200 and 1,800 rungs), delete
`<run dir>/complete.json`, re-arm with `box/ctl.sh arm` and a higher `--steps` (a new `arm` clears `done`, `release`
does not), and keep the same world size, because the resume RNG state is stored per rank (`train27.py:395-404`).
Checkpoints every 50 steps; 300 steps take about 1 h.

**Decision at step 300** (average the evals at 250 and 300 to damp noise):

1. Safety, for each arm at that LR. Pass only if all hold: `kl_to_untouched` at most 0.08 (about 2x the 4B clean
   level); replay NLL at most the untouched value plus 0.05 nats; tool-replay top-1 agreement at least 0.95; no loss
   spike above 3x the running median.
2. Signal, wide against ctrl at the same LR. Present if any holds: in-length exact match `s1:len256:le256` gap of at
   least 5 points; wide answer NLL at least 10% (relative) below ctrl and widening over the last 100 steps; gate rate
   on S1 tokens at least 1.5x the rate on text (4B: 14 to 17% against 7%, `prereg:132-136`). Also report the gap on
   `s1:len512:gt256`; expect it flat before the switch.
3. Choose. If 1e-3 passes safety and shows signal: 1e-3. If 1e-3 fails safety: 1e-4. If both pass and neither shows
   signal: extend both pairs to 600 steps (free, same runs), read again with the same rule, then 1e-3 if it passes
   safety, else 1e-4. All thresholds here are mine; write them into the run log before step 300.

**Seeds.** The winning box continues as seed 0 to step 700. Its 300-step sweep run at the losing LR is stopped
(`ctl.sh disarm`) and archived, and that box starts seed 1 from step 0 at the winning LR, same pairing (wide and ctrl
on one box, 2-box layout). Preemption resumes from the newest kept checkpoint (see Abort and repair for how many are
kept).

**Probe box.** P1 (replay weights 0.10 chat / 0.25 tool / 0.15 LM against the main run's 0.5 / 0.5 / 0.15), P2 (W LR
at 3x), P3 (GDN plus MLP targets) and P4 (LoRA+ ratio 4), wide-only, 400 steps, at 1e-3 during the sweep and at the
winning LR afterwards. P5 is conditional, see the ladder.

**Ladder and go/no-go.** At step 700, mean over two seeds on the fixed subset: if the wide-minus-ctrl gap on
`s1:len512:gt256` is under 10 points, extend both boxes to 1,200, then 1,800 (procedure in Sweep), and if answer NLL is flat too, run P5 beside the extension: batch 128 with the replay counts doubled to 16 each, wide-only, 400 steps on all 8 GPUs of a box, about 1.3 box-h, compared with the main wide run by switch step and answer NLL. Run the full S1 set and the regression
gates (`PLAN.md:43-47`) on the final rung only. Go needs at least +10 points past the training length and every
gate inside its band. Declare a null only after 1,800 steps. Add a third seed pair when the two-seed gap lies between
5 and 15 points or the seeds disagree on go/no-go. Pick the Stage B start point per arm from both seeds, not from the
better one, unless Stage B funds one seed.

**Abort and repair.** If `kl_to_untouched` reaches 0.15 on two consecutive evals, or tool agreement drops under 0.90,
or the recall guard loses more than 10 points of its step-0 accuracy: stop both arms of that pair at the same step,
restart at half the LR from the last passing checkpoint (step t minus 50, where t is the first breach), and log it.
Three trainer or launcher changes are needed first, all checkable on CPU. (1) `--resume` ignores a new `--lr`:
`opt.load_state_dict` and `sched.load_state_dict` restore each group's `lr` and `initial_lr` and the scheduler's
`base_lrs` (`train27.py:399-400`). A CPU check with torch AdamW and LambdaLR in this review confirmed that the LR
stays at the old value after load and moves only when all three are overwritten, so halve them after loading, for
both the LoRA and the W group. (2) Evals and saves run every 50 steps and only two resumable checkpoints are kept
(`train27.py:300,485-491`), so at the second breach (t plus 50) the kept `ckpt_*.pt` are t and t plus 50, both after
the first breach. Keep four (`[:-4]`), or restart from `trainable_{t-50}.pt` (adapter and gate only, fresh Adam state,
30 warmup steps), which is never deleted. (3) A run directory that holds `complete.json` is skipped by `run_pair.sh`,
so a repaired run that had already finished needs that file removed first. After Stage A, if BFCL or IFEval miss their
gates, repair with on-policy distillation from the
untouched model on tool and chat replay prompts before any RL. Write the final checkpoint from a short linear
cooldown branch (about 100 steps to zero LR, about 20 min a pair) if the constant-LR checkpoint fails a general gate
only by noise-level margins; this cooldown is my proposal and has no source behind it.

**Stage B hook.** Reward-variance census: 8 rollouts on 200 S2 items per arm adapter; if fewer than 30% of groups have
mixed outcomes (mine), fix S2 difficulty before training. Then the 50-step wide-arm pilot on the Hebrew recipe
(LoRA r16 alpha 32, AdamW 1e-5, n = 8, no std normalization, dynamic filter on the outcome, clip 0.2/0.28 token-mean,
low-variance KL 0.01 to the Stage A adapter, truncated IS 2.0, temperature 0.7, adapter served in rollout). A flat
reward slope after 50 steps changes the method; a moving slope releases the paired runs.

## Sources that were opened

Blog and vendor pages, read in full: [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/),
[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/),
[RunPod B300 guide](https://www.runpod.io/articles/guides/nvidia-b300),
[RunPod rent B300](https://www.runpod.io/articles/rent/b300),
[Flopper RTX PRO 6000 sheet](https://flopper.io/gpu/nvidia-rtx-pro-6000-blackwell-server-edition/spec-sheet.pdf),
[NVIDIA RTX PRO 6000 page](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) (lists BF16 at 1 PFLOP with no dense or sparse label in the page text; the Flopper sheet gives 500 TFLOPS dense and 1,000 with 2:4 sparsity, so 1 PFLOP matches the sparse figure),
[Verda B300](https://verda.com/b300) (268 GB VRAM, on-demand $8.62/h and spot $4.31/h per GPU),
[Flopper B300 sheet](https://flopper.io/gpu/nvidia-b300-sxm-288gb) (BF16 2,250 TFLOPS dense, 4,500 with 2:4 sparsity),
[Nebius HGX B300](https://nebius.com/compute/b300) (BF16 36 PFLOPS for 8 GPUs, labeled sparse, so 4.5 a GPU sparse and 2.25 dense),
[NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/) (BF16 1,979 TFLOPS, footnoted as with sparsity, so 989 dense).

arXiv abstract pages only (full papers not read): 2411.12537, 2502.10297, 2412.06464 (theory: negative
eigenvalues, one reflection per token); 2605.10468, 2607.17620, 2608.14492, 2606.12921, 2609.02734, 2507.12142,
2502.16982, 2605.19282 (optimizers); 2402.12354, 2405.15682, 2405.09673, 2401.05605, 2403.08763 (LoRA, schedule,
forgetting); 2509.04259, 2306.13649, 2305.18290, 2503.20783, 2503.14476, 2402.03300, 2609.17848, 2609.13866
(post-training methods); 2304.15004 (metrics); 2606.11052 (hybrid recall). All at `https://arxiv.org/abs/<id>`.

## Open items and what is assumed

- B300 throughput and the 2x claim are unmeasured. B300 BF16 dense (2,250 TFLOPS) and H200 (989) come from a
  third-party sheet plus Nebius and NVIDIA pages that label the sparse figure, and they put the compute ratio at
  2.27x, not the 2.5x in `PLAN.md`. The RTX BF16 number is a third-party sheet. The 8-GPU-per-run scaling is
  unmeasured. Kernel support for the GDN on the B300 architecture is unverified.
- The 4-box layout is unmeasured and needs a launcher that does not exist yet. Per-rank length imbalance at 8
  sessions a rank, and preemptible capacity for four 8-GPU boxes against the 32-GPU quota, are unverified.
- Whether 64 S1 sessions reach the swap-switch regime is unproven; probe P5 is the test.
- Prices: Verda ($8.62 on demand, $4.31 spot per GPU) and RunPod ($6.94, $7.89) are from their pages, read
  2026-10-01 and dated by the vendor, not by a quote. The Nebius figures ($7.85 on demand, preemptible "from $0.99")
  come from `PLAN.md`, read by the orchestrator. Vast and Jarvis were not checked. The five-provider check, spot and
  on-demand, is still due before a rental. Preemptible prices move, and "from $0.99" is a floor.
- The 27B is post-trained. Its switch timing may differ from the 4B and 9B base models. The gate W LR ratio is
  untested at scale. The 1e-3 general-eval cost is unmeasured at any size.
- Whether Qwen3.8 was Adam-pretrained is assumed, not verified; it is the premise of the optimizer mismatch source.
- Replay weights, the KL bars (0.08, 0.15), the signal thresholds, the 30% variance bar, the tool proxy and the
  recall guard are my proposals. The tool proxy is unvalidated against BFCL.
- Stage B step time (3 to 5 min) is a planning figure. Stage B dollars scale linearly with it.
