# Qwen3.8-27B retrofit: research-speed assessment on B300 and H200

Date: 2026-10-01. Scope: how to make Stage A training and evaluation faster on 8x B300 (fallback 8x H200), and whether
"better hardware that halves the time" justifies a bigger budget. Assessment only. No code was changed, no box was
rented, no GPU was used. The only local runs were two small CPU scripts (the padding simulation in section 3 and the
conv-leak test in section 5.1). A reviewer pass on 2026-10-01 corrected prices, the RTX basis, the 9B to 27B scaling
factor, the vocab, SDPA and surogate claims against raw sources; the list is in section 13.

Label key used in every number below:

- **[M]** measured by a run or log that exists on disk (path given).
- **[S]** read from a source I opened (section 11).
- **[D]** arithmetic I did from [M] or [S] inputs. Inputs are named, so it can be redone.
- **[E]** estimate. A judgment with no direct measurement. Treat as a prior, not a receipt.

## 0. The answer

1. **Faster GPUs alone are unlikely to halve the whole Stage A block.** The PLAN's own block table has training and the
   LR sweep at 2.0 h of 6.0 h wall. Setup (1.5 h) does not shrink with GPU speed or count. Self-distillation (0.5 h) and
   evals (2.0 h) shrink with more GPUs and partly with faster ones, but no scaling for them is measured anywhere. If they
   did not shrink at all, infinitely fast training would still leave 4.0 h, a 1.5x cut. If they shrank fully, the floor
   is the 1.5 h of setup, a 4x cut. The truth is between, and only a timed smoke on the box settles it. The training
   slice alone does halve: B300 is 2.3 to 3.1x faster than the RTX PRO 6000 box on the same trainer (section 8.3).
   [D from PLAN.md]
2. **The training slice can be halved, and most of it is free.** `train27.py` as written pads micro-batches and draws
   sessions per rank at random. On real S1 training lengths, padding plus waiting for the slowest rank means it does
   useful work on only 41 to 60% of what it computes (0.41 to 0.60 across the four settings simulated). One session per
   micro-batch plus token-balanced rank assignment reaches 97 to 100%. That is 1.5 to 1.8x on the blended Stage A token
   mix, at zero GPU cost and with identical gradient math. [D, CPU simulation, section 3]
3. **The PLAN's 16k tokens/s per 4-GPU run is optimistic.** My estimate for the B300 is 6.7k to 9.0k tokens/s per run
   with the batching as written (1.8 to 2.4x below the PLAN) and 11.2k to 14.9k with the fix (1.07 to 1.43x below).
   [E for the B300 base rate, D for the rest]
4. **If the owner wants wall time halved across the whole Stage A block**, the only route is the fix plus twice the GPUs
   (8 GPUs per run, 4 boxes, the full 32-GPU quota). That gives about 3.9 to 4.3 h instead of 6.1 to 6.9 h (a 1.6x cut;
   it is a 2x cut only against the trainer as written, 7.6 to 8.8 h) for about 25% more GPU-hours. [E]
5. **A bigger budget does not buy speed here. Price risk is the budget question.** Stage A costs 98 to 110 GPU-hours
   with the fix. That is $97 to $109 at the listed preemptible floor of $0.99 and $931 to $1,045 at the on-demand B300
   price of $9.50, which the Nebius price table lists as effective 2026-10-01 (today). The PLAN's $750 was computed at
   $7.85 and is stale: its own 96 GPU-hours cost $912 at $9.50. H200 on demand rose the same way, from $4.50 to $5.40.
   ("Contact us" on the same page belongs to the GB300/GB200 NVL72 rows, not to the B300.) Per token, on-demand B300 is
   between 0.9x and 1.2x the cost of on-demand H200 at $5.40 (section 8.3), because both prices rose about 20%. The
   preemptible B300 floor is about 9.6x cheaper per token than on-demand B300, but it is a dynamic "from" figure, so
   treat the floor as a hope and the $9.50 column as the ceiling. [D, S]

## 1. What is being sped up

LoRA r16 on the Gated DeltaNet projections (`in_proj_qkv`, `in_proj_a`, `in_proj_b`, `out_proj`, read from
`train27.py` `LORA_TARGETS`) plus the zero-init fp32 widened-beta gate per GDN layer, on Qwen3.8-27B (64 layers: 48 GDN
and 16 full attention, hidden 5120, intermediate 17408, vocab 248,320: this is the stock model, not the
Hebrew-extended one whose padded vocab is 264,576 and whose H100/H200 runs are the baselines in section 2). One bf16 replica
per GPU, 4 GPUs per run, one wide/ctrl pair per box (`run_pair.sh`). Per optimizer step: 32 S1 sessions, 8 chat, 8 tool
and 8 LM chunks of 512 tokens (defaults in `train27.py`). About 75M real tokens per run, 600 to 700 steps, so about
107k tokens per step [D], of which S1 is roughly 80%. S1 mean is about 2.5k tokens: 2,551 over all 30,000 S1 train
sessions with the real Qwen3.8 tokenizer and chat template (reviewer re-count), 2,523 on the builder's 2,400-session
proxy-tokenizer sample, 2,572 from the manifest's real-tokenizer means (`data-s1-manifest.md`, 200 sampled sessions per
domain), so 32 sessions are about 81k to 82k tokens. [D]

Trainable parameters per run [D from the model config]: LoRA about 28M (per GDN layer 246k on `in_proj_qkv`, 165k on
the two small projections, 180k on `out_proj`) plus 11.8M fp32 gate weights, about 40M in total. One step's gradient
all-reduce is about 160 MB in fp32, which is negligible on NVLink and tolerable even on PCIe.

## 2. Baselines

| what | number | label | where |
|---|---|---|---|
| 4B, 1 GPU (RTX PRO 6000), 40M real tokens | 5,905 tok/s, 17.1 GB, 6,802 s | [M] | `/data/ai-ml/models/_runs/negeig-retrofit/box/main_wide_s0/train_log.json` |
| 9B, 1 GPU (RTX PRO 6000), 40M real tokens | 4,372 tok/s, 29.2 GB, 9,187 s (ctrl 4,459) | [M] | `/data/ai-ml/models/_runs/negeig-retrofit/box/runs/main9_wide_s0/train_log.json` |
| gate overhead, wide vs ctrl | about 2% wall (4B 2.0%, 9B 1.9%) | [D from the wide and ctrl rows of each size] | same files |
| 27B on 8x H100, full-vocab loss, r32 on 7 modules, seq 2048 | 12.9k tok/s total (1.6k/GPU), 69 GiB peak. About 26% MFU at 162 GFLOP/token against about 990 dense | tok/s and memory [M], MFU [D] | `research/hebrew-stage1-20260915/LANE.md` |
| 27B on 4x H200, same trainer, MB3 x seq 4096 | 7.6k tok/s total (1.9k/GPU), 112 GiB peak. About 31% MFU by the same arithmetic | tok/s and memory [M], MFU [D] | same |
| 9B to 27B scaling | 3.52x compute per token by layer-body parameters (6.92B vs 24.35B), so about 1.24k tok/s/GPU on the RTX box (4,372 / 3.52), about 1.22k after the 0.98 lever-1 factor | [D], assumes equal MFU at 27B | parameter count from the two model configs |
| PLAN's B300 rate | 4k tok/s/GPU, 16k per 4-GPU run (2.5x the H100's 1.6k) | [E] in PLAN.md | PLAN.md |
| B300 dense BF16 | 2,250 TFLOPS, 8 TB/s, 288 GB (268 to 270 GB usable), NVLink 1.8 TB/s | [S] (Nebius B300 page lists 36 PFLOPS BF16 for 8 GPUs sparse, so 2.25 PF dense per GPU is [D]) | section 11 |
| H200 dense BF16 | about 990 TFLOPS ([D] from NVIDIA's 1,979 with sparsity), 4.8 TB/s, 141 GB, NVLink 900 GB/s | [S] and [D] | section 11 |
| RTX PRO 6000 | 96 GB, 1,792 GB/s, PCIe Gen5, no NVLink. A repo note puts dense BF16 near 252 TFLOPS (scaled from FP32, not an NVIDIA figure). I make no MFU claim against it | [S] and a repo note | section 11 |
| RTX PRO 6000 price | Vast 8-GPU box $11.79/h = $1.474/GPU-h (the 4B/9B retrofit runs, about $122 in total). Nebius lists RTX PRO 6000 at $1.80 on demand and $0.79 preemptible "from" | [M] record, [S] Nebius | `results/negeig-retrofit-preregistration.md` line 110; section 8.3 |

The 4B and 9B rows are 8 concurrent single-GPU processes per box (`box_main.sh`, `box_main_m.sh`: `CUDA_VISIBLE_DEVICES=$gpu
... &` then `wait`) on synthetic fixed-length tasks (`tasks.make`, its own `pad()`), with gradient checkpointing on and no
all-reduce. They are a per-GPU compute baseline, not a DDP number, and the 27B figure derived from them carries the
assumptions in the row above. [D]

The 27B H100/H200 runs used fully dense fixed 4,096-token or 2,048-token windows, so those rates contain no padding
tax. They are not like-for-like with `train27.py` (full-vocab loss on every token there, answer-only chunked head here;
LoRA r32 on 7 modules there, r16 on GDN projections here), so they are a conservative base for this trainer. [D]

B300 per-GPU rate used below: H200 1.9k times 1.5 to 2.0 = 2.85k to 3.8k tok/s on padding-free dense batches. [E]
Peak ratios are 2.27x compute and 1.67x bandwidth [D], and real MFU on a new part is usually below its predecessor's at
first, so I take 1.5 to 2.0x rather than 2.27x. The PLAN's 4k/GPU sits just above the top of that range.

## 3. The free fix: padding and straggler tax in `train27.py` [D, CPU simulation]

What the trainer does today: each rank draws its own sessions, sorts them by length and cuts micro-batches under a
padded-token cap (`microbatches()`, `micro_tokens=32768`), then runs an `attention_mask` through the decoder. The
step ends when the slowest rank finishes (single all-reduce per step). Two losses result:

- **Padding.** A micro-batch costs `n x longest` tokens, not the sum of real lengths.
- **Straggler.** Ranks draw different total lengths, and the all-reduce waits for the largest.

Simulation (3,000 steps, S1 training lengths from the six domain train files, 400 sessions per domain, proxy
tokenizer in the builder's run, then rerun by the reviewer on all 30,000 sessions with the real tokenizer, table below;
cost model: time proportional to processed tokens; `pad_*` mimics `microbatches()` at cap 32768;
`nopad_*` is one session per micro-batch; `*_lpt` is longest-first assignment of the step's sessions to the
least-loaded rank). Useful fraction is real tokens divided by world size times the slowest rank's processed tokens.
Script: scratchpad `pad_sim2.py`, run under `nice -n 10 taskset -c 8-15`, `OMP_NUM_THREADS=2`.

| world, S1 sessions/step | pad + random (today) | pad + LPT | no-pad + random | no-pad + LPT | gain, today to no-pad + LPT |
|---|---|---|---|---|---|
| 4, 32 (the PLAN layout) | 0.547 | 0.590 | 0.814 | 0.991 | 1.81x |
| 8, 32 | 0.413 | 0.438 | 0.682 | 0.971 | 2.35x |
| 8, 64 | 0.517 | 0.579 | 0.753 | 0.988 | 1.91x |
| 8, 128 | 0.599 | 0.654 | 0.814 | 0.996 | 1.66x |

Reviewer replication, all 30,000 S1 train sessions, real Qwen3.8 tokenizer and template (same cost model, same cap
32768, 3,000 steps, seed 2; scratchpad `pad_sim_real.py`, run under `nice -n 10 taskset -c 8-15`). Every cell is within
0.01 of the table above. [M, CPU]

| world, S1 sessions/step | pad + random (today) | pad + LPT | no-pad + random | no-pad + LPT | gain |
|---|---|---|---|---|---|
| 4, 32 | 0.552 | 0.594 | 0.817 | 0.991 | 1.80x |
| 8, 32 | 0.420 | 0.442 | 0.684 | 0.972 | 2.31x |
| 8, 64 | 0.521 | 0.584 | 0.756 | 0.988 | 1.90x |
| 8, 128 | 0.603 | 0.658 | 0.816 | 0.995 | 1.65x |

At world 4, no-pad alone is 1.49x and LPT adds another 1.22x. LPT without no-pad barely helps (0.547 to 0.590),
because padded cost is not additive in real tokens. Both parts are needed.

Blended gain. S1 is about 80% of tokens. LM chunks are fixed 512 tokens, so they pad nothing. Chat and tool samples are
variable and I did not simulate them. With S1 at 0.547 and the rest at an assumed 0.9, blended efficiency is
0.80/0.547 + 0.20/0.9 = 1.68 time units per real unit, so 0.59, and the fix is about 1.65x. I quote 1.5 to 1.8x. [E on
the 0.9]

Limits of this model: (a) time is taken proportional to tokens, but a one-session micro-batch of 2.5k tokens has a
smaller GEMM M than a padded 32k-token micro-batch, which I estimate costs 0 to 10% [E]; (b) attention and GDN scaling
with length are ignored; (c) closed by the real-tokenizer replication above; (d) chat and tool
lengths are not simulated. The gain is large enough that none of these can erase it, but the exact multiple needs the
box smoke in section 10.

Why it is safe. The loss is already normalised by global per-source token counts (`all_reduce(counts)`), so changing
which rank holds which session changes no gradient. Mean S1 training session is about 2.5k tokens and the longest of all
30,000 is 9,621 (real tokenizer; the manifest's 200-per-domain sample showed 9.2k), under `max_len` 12,288, so no session
needs splitting. A probable side benefit, read in `transformers/masking_utils.py` (`_ignore_causal_mask_sdpa`) and
`sdpa_attention.py`: a batch of one with an all-ones mask skips mask creation, so SDPA is called with `is_causal=True` and
no mask, the case PyTorch's flash and cuDNN backends accept. A padded batch keeps a mask, transformers then expands KV with
`repeat_kv`, and SDPA most likely runs the memory-efficient kernel. The transformers "math kernel" comment concerns
`enable_gqa` together with a mask, not the padded case, so it does not show a math fallback here. Which kernel runs today
was not profiled. [S, local source; [E] on the consequence]

Implementation size: the change is in the draw and plan section of the step loop. All ranks draw the step's global
session list from a shared seeded stream, assign by LPT on real token counts, and run batch-1 micro-batches. LM chunks
can keep batching (fixed length). Small, local to `train27.py`; I did not edit it.

## 4. Ranked levers

Speedup is on Stage A training wall time unless stated. Risk is to correctness of the experiment, not just to schedule.

| # | lever | expected speedup | risk | how to verify |
|---|---|---|---|---|
| 1 | One-session micro-batches plus token-balanced (LPT) rank assignment (section 3) | 1.5 to 1.8x blended, 1.81x on S1 at world 4 [D] | low: same gradient, `attention_mask` disappears for batch 1 | (a) CPU smoke on tiny qwen3_5: same global batch, padded vs new path, loss equal within fp32 tolerance; (b) box smoke: log per-rank step time, max minus min under 5% and real tok/s vs the padded path |
| 2 | B300 instead of H200 | 1.5 to 2.0x [E] | low for correctness; sm_103 toolchain is the risk (section 9) | 30-step smoke per arm, measured tok/s replaces the estimate |
| 3 | Gradient checkpointing off, or selective (every second layer) | 1.2 to 1.4x on B300, about 1.1x on H200 [E]. Arithmetic: 6N to 4N FLOP/token is 1.5x on matmuls [D] | memory: about 19 MB/token of activations without checkpointing is my own count [E], so a 9.6k-token session (the longest S1 train session is 9,621) needs about 183 GB plus 54 GB of weights | one-step probe on the longest training session (9,621 tokens), read `max_memory_allocated`; go selective if peak exceeds about 240 GB |
| 4 | 8 GPUs per run across 4 boxes instead of 4 per run across 2 | about 1.95x per run with lever 1 (0.971 vs 0.991 useful fraction [D]); only 1.5x without it (0.413 vs 0.547) | design: a wide/ctrl comparison is no longer paired within one box; capacity for 4 preemptible boxes is unknown | owner decision; check Nebius quota and capacity first |
| 5 | FlashQLA backend for the GDN chunk kernels | 1.0 to 1.15x [E]. README claims 2 to 3x forward and 2x backward vs fla Triton 0.5.0 [S], but the GDN share of a step is unprofiled, so end-to-end is bounded by that share | medium: its numerics at beta in (1,2) are unverified (the retrofit computes beta outside the kernel and leaves `allow_neg_eigval` at its default of False, which fla's verifier accepts; the verifier also requires K = V = 128, which the 27B's linear head dims of 128 meet, so fla would route to it if installed); one arm could silently differ from the other | gradient parity vs fla Triton and an fp32 reference on random inputs with beta in (0,2), all of dq, dk, dv, dbeta, dg; confirm the backend name in both arms' logs; install on both arms or neither |
| 6 | `torch.compile` on pieces (RMSNorm, SwiGLU, gate math) | 1.0 to 1.05x [E] | medium: fla Triton kernels are opaque to Inductor and ragged shapes recompile | A/B over 3 interleaved blocks; skip unless the profile shows elementwise kernels above 10% |
| 7 | CUDA graphs | not applicable | high: ragged shapes, LoRA and optimizer steps | none, do not pursue |
| 8 | Fused or chunked CE over the 248,320 vocab | at most 1.02x [E] | low | already done for the answer-only head (chunks of 2,048 under `checkpoint`); skip Liger, it is not installed |
| 9 | Attention backend (FA4, cuDNN SDPA) for the 16 attention layers | about 1.0x. Attention is about 1% of FLOPs at 2.5k context and about 3% at 9k [D] | FA4 on sm_103 runtime unverified | none needed; just avoid masks (lever 1 does) |
| 10 | Sequence packing with `cu_seqlens` | 0 to 1.05x over lever 1 [E] | high: leakage (section 5.1) | only if ever done: packed-vs-alone logits test, section 5.1 |
| 11 | FSDP2 | at most 1.0x | medium: fp32 gate beside bf16 weights under mixed-precision FSDP2 is unverified | not needed, weights are 54 GB |
| 12 | Freeze more, LoRA on fewer layers | about 0x: the first GDN layer is layer 0, so backward must reach it anyway | changes the method | do not do it |

Levers 1, 2 and 3 compound: 1.5 to 1.8 times 1.5 to 2.0 times 1.0 to 1.4 is 2.3 to 5.0x on training time, B300 without
checkpointing versus the trainer as written on an H200 with it. [D on [E] inputs]

## 5. Lever notes

### 5.1 Sequence packing for GDN (levers 1 and 10)

Packing needs two separate things, and both must be right.

- The chunk kernel needs `cu_seqlens`. transformers 5.17 forwards `cu_seq_lens_q` to it (local source read,
  `modeling_qwen3_5.py`). transformers 5.2.0 to 5.8.1 dropped it (ms-swift issue 9618: "causal-conv / recurrent state
  leaks across samples silently", fixed in 5.9.0+). [S]
- The conv1d needs `seq_idx`. With `causal-conv1d` not installed (true on the negeig venv), transformers uses
  `F.conv1d` with left padding, which ignores boundaries. A 10-line CPU test that copies that formula (pad 3, crop,
  silu, width 4) on random data shows the second document's first three positions differ from the standalone result
  by max-abs 1.803, 1.221 and 0.324, and are exactly equal from position 3 on, with the first document untouched.
  [M, scratchpad `conv_leak.py`; magnitudes are random-data magnitudes, the pattern is the point]
- The hub wrapper filters kwargs to the live signature, so a missing `seq_idx` is dropped without an error. [S, read]
- Real-model size of the leak: surogate PR 216 measured mean 0.83 nats and max 2.70 nats per row on Qwen3.5-0.8B with
  32 packed Qwen3.8-rendered rows (bf16, LoRA r8), but in a setup where both the conv1d boundary leak and the recurrent-state
  carry-over were present. HF 5.17 already forwards `cu_seq_lens_q` to the chunk kernel, so only the conv1d part would
  remain here and those numbers overstate it; the conv part is confined to the first 3 positions of each following
  document (width-4 kernel), as the test above shows. [S, PR body read] For this project a leak would feed one session's tail state into the following session's first
  tokens, which is exactly the signal a state-tracking test is meant to isolate.

Fixes if packing is wanted: install `causal-conv1d` and pass `seq_idx`, or call fla's Triton `causal_conv1d` with
`cu_seqlens`. Neither was tried here.

Recommendation: do not pack. Lever 1 reaches a 0.97 to 0.99 useful fraction with no boundary risk. Packing would only
raise the GEMM M per micro-batch, marginal at 2.5k tokens on a 5120-wide model. Length bucketing alone is what the
trainer does today, and section 3 shows it is insufficient without balancing.

### 5.2 DDP full replica vs FSDP2 (lever 11)

Use DDP. 54 GB of bf16 weights fit a B300 with 200+ GB to spare and an H200 with about 87 GB. The gradient
all-reduce is 160 MB per step. FSDP2 would all-gather 54 GB per forward, and again per backward and recompute, about
0.06 s per gather at 900 GB/s ideal, so about 0.18 s per micro-batch [D]. An 8k-token micro-batch is 1.3 PFLOP at
162 GFLOP/token, which is about 4 s at the 0.31 PFLOP/s per GPU the H200 run achieved (derived from the measured
1.9k tok/s) [D]. The gathers can overlap, so the loss is small but the gain is zero, and the fp32 gate plus bf16
weights under FSDP2 mixed precision is an extra verification burden. FSDP2 on 8 GPUs spanning PCIe-class links would
change this, but both target boxes have NVLink.

### 5.3 Gradient checkpointing (lever 3)

FLOPs per token with LoRA on GDN projections: forward 2N, recompute 2N (checkpointing), activation gradient 2N, weight
gradients negligible. That is 6N, or 162 GFLOP/token for 27B. Without checkpointing it is 4N. The 1.5x FLOP cut becomes
1.2 to 1.4x wall because GDN kernels, norms and elementwise ops do not scale with the GEMM share. [E]

Memory decides it. My count is about 300 KB per layer per token (MLP gate, up and act at 17,408 each, GDN qkv, z and
conv outputs, norms) times 64 layers, about 19 MB/token [E, unmeasured]. On a B300 (268 GB usable) that fits any
session up to about 11k tokens by that count, training sessions top out at 9,621, but `max_len` is 12,288 and replay
sources are variable, so cap or go selective. On an H200 it fits about 4.5k tokens, so keep checkpointing on.

### 5.4 Attention kernels on Blackwell (lever 9)

- FlashAttention-2 is Ampere/Ada/Hopper. FA3 is Hopper only. FA4 (`flash_attn.cute`, CuTeDSL) supports Hopper and
  Blackwell, and PR 2572 (merged 2026-05-24) fixed the architecture assertion that excluded sm_103 in the non-cu13 CuTeDSL build. SGLang's
  own fix for the B300 failure (PR 25576) was the `[cu13]` extra dependency, so on a cu130 stack use the cu13 variant. [S]
- In FA4 `main` `interface.py`, checked by grepping the raw source, a dedicated SM100/SM110 head_dim 256 2-CTA kernel
  exists for both forward and backward (`BlackwellFusedMultiHeadAttentionForward/Backward`). The 16 attention layers
  here have head_dim 256 with 24 query and 4 KV heads, so they are covered on paper. The hd256 forward has no softcap,
  block sparsity or learnable sink. [S] (An earlier web summary said hd256 backward was unsupported on SM100. The raw
  source shows it is supported. The summary was wrong.)
- Runtime on sm_103 and which release tag contains those kernels are unverified.
- PyTorch FlexAttention has a FA4 backend on Hopper and Blackwell (blog, 2026-03-04, 1.2 to 3.2x over its Triton
  path on compute-bound workloads). [S] Not needed for training at 1 to 3% attention FLOPs.

The mask is the one unknown. A padded batch passes a mask, and SDPA then most likely leaves its flash path for the
memory-efficient kernel, which does not materialise the score matrix, so this is a speed question, not an OOM. Only if
SDPA fell back to the math kernel (no source I read shows that, and it was not profiled) would T = 9,000 materialise 24
heads x 9,000 squared x 2 B, about 3.9 GB per layer, as a worst case [D]. Lever 1 removes the mask either way.

### 5.5 Loss on answer tokens only, chunked CE (lever 8)

`seq_loss` already selects labelled positions before the head and runs the head in 2,048-row chunks under
`checkpoint`. The head costs 2 x 5120 x 248,320 = 2.5 GFLOP per selected token [D], and only answer tokens (plus every
token of LM chunks) pay it, so under 2% of step FLOPs [E]. Nothing to gain.

### 5.6 Independent runs per box vs one 8-GPU run (lever 4)

The current `run_pair.sh` already runs two 4-GPU runs per box. With today's batching, world 4 beats world 8 per GPU
by 0.547/0.413 = 1.32x [D]. With lever 1 the difference is 0.991 vs 0.971, 2%, so the layout becomes a scheduling and
design choice: pairs keep wide and ctrl on identical hardware and survive preemption independently.

## 6. Evaluation speed

### 6.1 Budget by block

The PLAN's 2 h eval block (untouched model plus 4 adapters: S1 full, BFCL v4, MMLU-Pro, HumanEval, IFEval,
Global-MMLU he) is not the S1 forward pass. S1 is cheap; generation evals dominate.

### 6.2 S1, teacher-forced exact match, one forward per session

The 6 domains at 200 eval sessions each average 7,665, 7,745, 7,270, 5,570, 8,961 and 10,966 tokens, about 9.6M tokens
per model [D from the manifests]. Forward only is 2N = 54 GFLOP/token, so about 520 PFLOP [D]. At 35% MFU that is about
82 s on 8 B300 (2.25 PFLOP/s peak each) [D, [E] on the 35% MFU]. On a 4-GPU group it is about 2.7 min per model, so
five models take about 7 min on all 8 GPUs or about 14 min on one 4-GPU group. The longest eval session is 34,094
tokens (manifest, `orders` eval), which is fine forward-only under `no_grad` but should be
included in the memory probe in section 10. Levers: data-parallel replicas (54 GB fits, no TP), one session
per micro-batch (no padding), sort by length and assign by LPT, run under `no_grad`, answer-only head. Verification:
oracle head 45/45 and a wrong end-of-turn token failing every answer (already in the PLAN for `eval27.py`).

### 6.3 Generation evals on SGLang

- Use data-parallel replicas (dp = 8, TP 1) on a box, or 4 replicas per arm. Throughput-bound evals scale with
  replicas, latency-bound ones (multi-turn BFCL, tau-bench) scale with concurrency. [E]
- Merge the LoRA into weights for ctrl arms and serve a stock qwen3_5 checkpoint. The wide arm needs the gate, which
  cannot merge: it needs the unbuilt SGLang patch.
- Prefix caching: SGLang `main` has a mamba radix cache (`mamba_radix_cache_strategy`: auto, no_buffer, extra_buffer,
  extra_buffer_lazy; `mamba_track_interval` default 256; int8 mamba checkpoint). [S, read from
  `arg_groups/fields/exec_.py` on main, not from the 0.5.20 tag.] The 0.5.20 release notes (2026-09-18) I read
  have no Gated DeltaNet-specific item. [S] Local memra notes
  say hybrid prefix cache only serves extension shapes (continuation hits, fan-out 0%). That is a memra fact and
  not proof for SGLang, so verify with a two-turn hit-rate probe before counting on it.
- If the patch slips, HF `generate` on BFCL and tau-bench is much slower, 5 to 20x by my estimate [E].
- Cheap win on the critical path: start S1 and replay NLL evals on the training GPUs as soon as a pair finishes, while
  the other pair's generation evals run.

## 7. Recommended configurations

### 7.1 8x B300 (primary)

- DDP, one bf16 replica per GPU, the existing flat gradient all-reduce, 4 GPUs per run, wide on GPUs 0 to 3 and ctrl
  on 4 to 7 (`run_pair.sh` as is).
- Batching: lever 1. One S1, chat or tool session per micro-batch, no mask. LM chunks (512 tokens) batched freely.
  Global seeded draw, LPT assignment on real token counts. Per-source token normalisation unchanged.
- Checkpointing: probe first. Off if the longest (9,621-token) session peaks under about 240 GB, otherwise selective
  (every second layer), otherwise on.
- GDN kernels: fla Triton (0.5.2). FlashQLA only after the gradient-parity gate, on both arms together.
- Attention: `sdpa`, no mask. Do not install flash-attn or FA4 for training.
- No `torch.compile`, no CUDA graphs, no packing, no FSDP2.
- Allocator: none of the 27B scripts (`train27.py`, `run_pair.sh`, `box/*.sh`) set `PYTORCH_CUDA_ALLOC_CONF`, so
  `expandable_segments` is off today. Set it only if the section 10 memory probe shows fragmentation (one-session
  micro-batches of varying length make that plausible). The Hebrew lane set `NCCL_NVLS_ENABLE=0`, because NVLink SHARP
  multicast failed inside RunPod containers (`run_stage1.sh` comment, `LANE.md` TRAP line). That was a container
  problem, not a property of the GPUs, and I do not know whether a Nebius VM needs it. Note the mismatch:
  `box/accept.sh` runs its NCCL check with `NCCL_NVLS_ENABLE=0`, while `run_pair.sh` does not set it, so acceptance does
  not test the configuration training runs in. Run the all-reduce with and without it (section 10, item 3) and set the
  same value in both.
- Scale-out option for the owner: 8 GPUs per run on 4 boxes (32 GPUs) halves the training slice at about the same
  GPU-hours for that slice, see section 8.

### 7.2 8x H200 (fallback)

- Same as above with these differences: checkpointing on (selective at most, only if a probe shows headroom, 141 GB
  leaves about 87 GB for activations), no FlashQLA unless its gate passed on the B300 first, and expect 1.5 to 2.0x the
  B300 wall time [E]. The PLAN's 2.5x (0.4x compute) is a little above the 2.27x peak-FLOPs ratio, so it is
  conservative.
- Preemptible H200 is listed from $0.79 [S]. The PLAN fallback uses on-demand, which is $5.40 from 2026-10-01 (it was
  $4.50): the PLAN's $36 box-hour is $43.20, and its 23 fallback box-hours cost about $994, not about $830. A preemptible
  H200 at a low price could beat an on-demand B300 on cost per token, at the price of preemption risk.

## 8. Time versus budget

### 8.1 Stage A training slice only (4 runs x 75M real tokens)

Base rates: B300 2.85k to 3.8k tok/s/GPU [E], H200 1.9k [M], RTX PRO 6000 1.24k [D]. Efficiency factors from section 3
(0.59 for today's batching, 0.98 for the fix, both on blended tokens). No-checkpoint factor 1.2 to 1.4 [E].
Hours per run = 75M / (4 x rate x efficiency).

| setup | tok/s per 4-GPU run | hours per run | label |
|---|---|---|---|
| PLAN as written (B300) | 16,000 | 1.3 | [E] in PLAN |
| B300, trainer as written | 6.7k to 9.0k | 2.3 to 3.1 | [E] |
| B300, lever 1 | 11.2k to 14.9k | 1.4 to 1.9 | [E] |
| B300, levers 1 and 3 | 13.4k to 20.9k | 1.0 to 1.55 | [E] |
| B300, lever 1, 8 GPUs per run | 22k to 29k | 0.7 to 0.95 | [E] |
| H200, trainer as written | 4.5k | 4.6 | [D on M, assuming fixed-shape base] |
| H200, lever 1 | 7.4k | 2.8 | [D on M] |
| RTX PRO 6000 (first PLAN revision), lever 1 | 4.9k | 4.3 | [D] |

Comparison the owner's sentence points at: B300 over the RTX box at 4 GPUs per run on the same trainer is 2.3 to 3.1x
faster [D], which is the halving of the training slice. It assumes the RTX box keeps its 9B MFU at 27B and that a
4-GPU all-reduce over PCIe costs nothing, both unmeasured. The same trainer fix applies to both.

### 8.2 Whole Stage A block

Fixed blocks from the PLAN: setup 1.5 h, self-distillation 0.5 h, sweep 0.5 h, training 1.5 h, evals 2.0 h, 6.0 h
total. I scale the sweep and training (2.0 h in the PLAN) by the ratio of the PLAN's 16k tok/s to each rate above,
and leave setup fixed. Distillation and evals scale with GPU count (and with GPU speed, for H200). [E]

| scenario (2 pairs, 16 GPUs unless stated) | wall h | GPU-h | at $0.99 (preemptible "from", dynamic) | at $4.00 (illustrative) | at $9.50 (on demand, from 2026-10-01) |
|---|---|---|---|---|---|
| PLAN as written | 6.0 | 96 | $95 | $384 | $912 (the PLAN said $750 at $7.85) |
| B300, trainer as written | 7.6 to 8.8 | 122 to 141 | $121 to $140 | $488 to $564 | $1,159 to $1,340 |
| B300, lever 1 | 6.1 to 6.9 | 98 to 110 | $97 to $109 | $392 to $440 | $931 to $1,045 |
| B300, levers 1 and 3 | 5.5 to 6.4 | 88 to 102 | $87 to $101 | $352 to $408 | $836 to $969 |
| B300, lever 1, 32 GPUs (8 per run, 4 boxes) | 3.9 to 4.3 | 125 to 138 | $124 to $137 | $500 to $552 | $1,188 to $1,311 |
| H200, lever 1 | about 10 | about 160 | | | at $5.40: about $865 (at the superseded $4.50: about $720) |

How the rows are built (reviewer reproduced all of them): the 16-GPU B300 rows are wall = 1.5 + 0.5 + 2.0 x (16k / run
tok/s) + 2.0 h with distillation and evals held at the PLAN's 0.5 h and 2.0 h, and GPU-h = 16 x wall. The 32-GPU row
halves distillation and evals (twice the GPUs) and keeps setup at 1.5 h. The H200 row scales distillation and evals by
the 1.5 to 2.0x slower rate. So the 16-GPU rows sit at the pessimistic end of the range in section 0 point 1.

The table leaves out the in-training checkpoint evals (`--eval_every 50`, so 13 to 15 per run, each 16 S1 sessions per
domain and length plus 64 LM chunks). Forward only is 54 GFLOP/token against 162 for training, which is about 8% of
training compute [D, FLOPs, not wall-clock; they are spread over the ranks]. On the scaled training slice that is about
+0.2 to +0.4 h and +3 to +6 GPU-h per block, and I do not know whether the PLAN's 2.0 h already includes it.

The 32-GPU row costs about 25% more GPU-hours than the 16-GPU lever-1 row because setup and acceptance run on twice
the boxes. It is the only row that brings the whole block near half of the PLAN-as-written wall time versus the
trainer as written (7.6 to 8.8 h down to 3.9 to 4.3 h). Preemption loss is not modelled. With checkpoints every 50
steps a preemption costs at most 50 of about 700 steps plus restart time.

### 8.3 Cost per million real tokens [D]

| box | rate basis | $/M tokens |
|---|---|---|
| B300, lever 1, $9.50 on demand (from 2026-10-01) | 2.79k to 3.72k tok/s/GPU | $0.71 to $0.95 |
| B300, lever 1, $4.00 (illustrative) | same | $0.30 to $0.40 |
| B300, lever 1, $0.99 preemptible floor | same | $0.07 to $0.10 |
| B300, trainer as written, $9.50 | 0.59 of the above | $1.20 to $1.60 |
| H200, lever 1, $5.40 (from 2026-10-01; $4.50 before) | 1.86k tok/s/GPU | $0.81 (was $0.67) |
| RTX PRO 6000 on Vast, lever 1, $1.474/GPU-h ($11.79 per 8-GPU hour, the record) | about 1.22k tok/s/GPU (section 2) | $0.34 |
| RTX PRO 6000 on Nebius, lever 1, $1.80 on demand (or $0.79 preemptible "from") | same | $0.41 (or $0.18) |

Break-even. B300 on-demand beats H200 on-demand on cost per token only if it is more than 1.76x faster ($9.50 over
$5.40), or more than 2.11x faster against the superseded $4.50 [D]. My 1.5 to 2.0x [E] straddles 1.76x, so on demand the
two cost about the same per token (0.9x to 1.2x) and the B300's gain is wall time. Versus the RTX box on Vast (2.3 to
3.1x faster at $1.474/GPU-h [D]), B300 on-demand costs about 2.1 to 2.8x more per token, and B300 at the $0.99
preemptible floor costs about 3.4 to 4.5x less [D]. So the price of a preemptible B300 hour, not its speed, sets the
budget.

Price status to show the owner. The Nebius price table, read as raw HTML on 2026-10-01 (not from a summary), lists on
demand B300 at $7.85 through 2026-09-30 and $9.50 from 2026-10-01, H200 $4.50 then $5.40, H100 $3.85 then $4.50, B200
$7.15 then $8.50, RTX PRO 6000 $1.80 flat. Preemptible "from" prices: B300 $0.99, B200 $0.99, H200 $0.79, H100 $0.79,
RTX PRO 6000 $0.79, all dynamic. "Contact us" appears only on the GB300/GB200 NVL72 rows. The PLAN's $7.85 and $4.50
figures are stale as of today. Confirm the price in effect in the Nebius console price calculator before renting. [S]

### 8.4 What to tell the owner about "bigger budget"

- Speed is not the reason to raise the budget. With lever 1, Stage A is 98 to 110 GPU-hours, inside or near the PLAN's 96.
- The reason to raise it is the price band: $97 to $109 (floor) up to $931 to $1,045 (on-demand, $9.50) for the same
  work, against the PLAN's $750. The budget ceiling should be set against the on-demand column (about $1,050 before any
  preemption or retry margin), since preemptible capacity is not guaranteed.
- Cutting the whole-block wall time from 6.1 to 6.9 h to 3.9 to 4.3 h costs capacity and about 25% more GPU-hours (about
  $260 more at $9.50), needs all 32 GPUs of quota at once and gives up box-paired comparisons. That is a design call for the owner, and worth it only if calendar time matters more than pairing.
- Stage B (GRPO, re-planned after Stage A per the PLAN) is most likely rollout-bound (verl plus SGLang generation), so
  few of the training levers apply [E]. Its speed depends on SGLang on sm_103 and on the unbuilt gate patch. I have no measured number for it.

## 9. sm_103 support status (late 2026)

sm_103 is B300. None of this was run on a B300. No B300 was available and none was rented.

| component | established | not established | label |
|---|---|---|---|
| PyTorch cu130 | Issue 159779 lists sm_103 (B300/GB300) in CUDA 13.0 enablement, nightlies from 2025-08-22. Local venv has torch 2.14.0+cu130 | The local torch arch list has no explicit sm_103 entry, so it relies on sm_100 binary or PTX compatibility. Needs a real CUDA alloc and matmul on the box | [S], [M] local version |
| Triton | Verda blog (2025-12-17): stable PyTorch's bundled Triton lacked SM103 then; workaround was nightly 2.11 cu130 or building Triton from source. Local Triton 3.8.0 `ptxas` lists sm_103, sm_103a, sm_103f | Whether the bundled Triton compiles fla kernels for sm_103 in the torch build that will be installed | [S], [M] local `ptxas` |
| fla 0.5.2 | Source read: `IS_NVIDIA_SM100` tests compute capability major 10, which covers sm_103 | Kernel runs on sm_103 not seen. Needs fwd and bwd parity smoke | [S] read |
| FlashQLA | README lists SM90, SM100, SM103 and SM120. fla's verifier (local `flash_qla.py`) admits it when `IS_NVIDIA_HOPPER or IS_NVIDIA_SM100` (capability major 10 covers sm_103), for bf16 or fp16 with K = V = 128 (the 27B's linear head dims are 128), and refuses `allow_neg_eigval=True` | Correctness at beta in (1,2); kernel run on sm_103 | [S], local source read |
| flash-attn FA2 / FA3 | FA2 not Blackwell; FA3 Hopper only | n/a | [S] |
| flash-attn FA4 (CuTeDSL) | PR 2572 merged 2026-05-24 fixed the arch assertion in the non-cu13 CuTeDSL build; SGLang issue 25564 hit it on B300 with `flash_fwd_sm100.py:162` ("Only SM 10.x and 11.x are supported") and closed via SGLang PR 25576, which added the `[cu13]` extra dependency. hd256 fwd and bwd SM100 kernels are in `main` | Release tag with the fix; runtime on sm_103 | [S] |
| SGLang 0.5.20 | Release notes (2026-09-18): TRT-LLM attention kernels for DeepSeek-V4 cover SM100 and SM103; 0.5.19 was the last release with cu12x wheels | Qwen3.5/3.8 GDN path on sm_103; mamba radix cache details at the 0.5.20 tag (I read `main`) | [S] |
| verl | The repo moved to `verl-project/verl`. Issue 6949: the official image broke on B300/GB300 (sm_103) with "no kernel image", and the wheels are built for arch list "9.0;10.0"; issue 7468 tracks it (read 2026-10-01, current status not recorded, recheck before Stage B) | Whether any current image or wheel runs on sm_103 (plan on a source build with sm_103 or a box smoke). Hebrew RL lane pinned verl main 12ebe0cb on other hardware | [S] |

## 10. First-box checklist (ordered, cheap first)

1. Real CUDA allocation and a bf16 matmul on every GPU, `torch.cuda.get_device_capability()` logged (expect (10, 3)).
   `nvidia-smi` answering proves nothing.
2. A Triton kernel compile and run, then fla `chunk_gated_delta_rule` fwd and bwd against the fp32 reference at
   beta in (0,2). Record which backend (Triton or FlashQLA) fla chose.
3. NCCL all-reduce across the 4 GPUs of each run, with and without `NCCL_NVLS_ENABLE=0`.
4. G0 on the 27B (already in the PLAN).
5. 30-step smoke per arm with a 3-step `torch.profiler` window: record tok/s, per-rank step-time spread, peak memory,
   and the GDN kernel share of a step. That share decides levers 5 and 6, and the memory number decides lever 3.
6. One-step peak-memory probe at the longest training session with checkpointing on, selective and off, and a
   forward-only probe at the longest eval session (34,094 tokens).
7. Only then the LR sweep.

## 11. Sources opened (and what each supports)

- PLAN.md, DATA.md and the three manifests under `results/negeig-27b/`.
- `train27.py`, `run_pair.sh` (`experiments/negeig_27b/`), `patch.py`, `train.py` (`experiments/negeig_retrofit/`).
- Run logs `/data/ai-ml/models/_runs/negeig-retrofit/box/*/train_log.json` (4B and 9B rows, [M]).
- `research/hebrew-stage1-20260915/{LANE.md,train_stage1.py}` (27B H100/H200 rows, [M]; windows are dense fixed length).
- Local source: `transformers/models/qwen3_5/modeling_qwen3_5.py`, `transformers/masking_utils.py`,
  `transformers/integrations/sdpa_attention.py`, fla `ops/gated_delta_rule/backends/flash_qla.py`, `utils/_device.py`,
  `modules/conv/*`, `modules/fused_linear_cross_entropy.py`; local versions: torch 2.14.0+cu130, triton 3.8.0, fla 0.5.2,
  transformers 5.17.0, peft 0.21.0 (causal-conv1d, flash-attn, liger-kernel not installed).
- https://github.com/pytorch/pytorch/issues/159779 (sm_103 in CUDA 13.0 enablement)
- https://verda.com/blog/nvidia-b200-and-b300-gpu-architecture-and-software-stack (Triton SM103 note, 268 GB usable)
- https://github.com/Dao-AILab/flash-attention/pull/2572 (FA4 sm_103 assertion fix, merge date)
- https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/flash_attn/cute/interface.py (hd256 fwd/bwd kernels,
  read by grep on the raw file)
- https://github.com/sgl-project/sglang/issues/25564 and the release notes at https://github.com/sgl-project/sglang/releases (v0.5.20)
- SGLang `main` mamba options in `python/sglang/srt/arg_groups/fields/exec_.py`
- https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/ (FA4 backend in FlexAttention, 2026-03-04)
- https://github.com/QwenLM/FlashQLA (2 to 3x forward, 2x backward vs fla Triton 0.5.0, SM90 to SM120 in the README)
- https://github.com/modelscope/ms-swift/issues/9618 and https://github.com/invergent-ai/surogate/pull/216 (packing leak; the PR body shows the 0.8B probe had both leaks)
- https://nebius.com/prices (raw HTML table, reviewer re-read: $9.50 B300 and $5.40 H200 from 2026-10-01, preemptible "from"
  prices, RTX PRO 6000) and https://nebius.com/compute/b300 (B300 memory, bandwidth, BF16 sparse total)
- https://github.com/verl-project/verl/issues/6949 and issues/7468 (sm_103 "no kernel image", arch list "9.0;10.0")
- `results/negeig-retrofit-preregistration.md` line 110 (Vast 8x RTX PRO 6000 at $11.79/h), `box/accept.sh` and
  `run_pair.sh` (NCCL and allocator env), `box_main.sh` and `box_main_m.sh` (concurrent single-GPU launch)
- Reviewer CPU reruns (scratchpad, not committed): `real_lens.py` and `pad_sim_real.py` (all 30,000 S1 train sessions, real
  tokenizer)
- https://www.nvidia.com/en-us/data-center/h200/ (141 GB, 4.8 TB/s, 1,979 TFLOPS BF16 with sparsity, 900 GB/s NVLink)
- https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/ (RTX PRO 6000 memory, interconnect)

## 12. Numbers that are derived, estimated or unsourced

- The B300 per-GPU training rate (2.85k to 3.8k) is [E]: H200's measured 1.9k times 1.5 to 2.0.
- The 0.9 efficiency for chat and tool samples is [E] and the 80% S1 token share is [D] from the defaults times the mean
  session length (chat and tool lengths were not measured).
- The builder's padding simulation used a proxy tokenizer; the reviewer's rerun on all 30,000 S1 train sessions with the
  real tokenizer agrees within 0.01 per cell (section 3). Both model time as proportional to processed tokens and ignore
  GEMM-size effects and attention scaling.
- RTX PRO 6000 dense BF16 (about 252 TFLOPS) comes from a repo note that scaled it from FP32, not from NVIDIA, and I
  use it for nothing. The RTX price is sourced (Vast $1.474/GPU-h, preregistration line 110; Nebius $1.80 and $0.79 from).
  The RTX 27B rate of about 1.24k tok/s/GPU is [D] from the measured 9B rate over 3.52 and assumes equal MFU at 27B and
  free PCIe all-reduce. H200 dense of about 990 is [D] from 1,979 with sparsity. B300's 2,250 dense is [D] from a sparse system figure and a search summary.
- Activation memory of about 19 MB/token is my own tally, not a measurement.
- The 5 to 20x HF generate slowdown, the 1.0 to 1.15x FlashQLA end-to-end bound and the 1.2 to 1.4x no-checkpoint gain
  are [E]. The GDN share of a step has never been profiled.
- Fixed-block scaling in section 8.2 (setup, distillation and eval times) assumes the PLAN's block times are right.
- The in-training eval overhead (about 8%) is FLOP-derived, not wall-clock.
- Nebius prices are a 2026-10-01 snapshot. Preemptible is a dynamic "from" figure, and which price applies in the
  hours around the 2026-10-01 switch is unconfirmed.

## 13. Reviewer corrections applied (2026-10-01)

- B300 after 2026-10-01 is $9.50, not "Contact us" (that label is on the NVL72 rows); H200 is $5.40. Sections 0, 7.2, 8.2,
  8.3 and 8.4 recomputed; the PLAN's $750 and $830 are stale.
- RTX PRO 6000 price is in the record ($1.474/GPU-h on Vast), not assumed at $1.00; 9B to 27B scaling is 3.52x, not about
  3x, so the RTX 27B rate is about 1.24k, not 1.46k tok/s/GPU, and B300 is 2.3 to 3.1x faster than it, not 2.0 to 2.6x.
- Section 0 point 1 no longer says distillation and evals do not shrink; they do with GPU count, and the in-training eval
  overhead (about 8%) is now stated.
- Vocab is 248,320 (stock model), not 264,576; the longest S1 train session is 9,621 tokens; the padding simulation is
  replicated on the real tokenizer.
- The SDPA "math kernel" comment and the 3.9 GB per layer figure are reframed as an unprofiled worst case; the surogate
  PR 216 numbers include the recurrent-state leak that HF 5.17 avoids; `expandable_segments` is not set by the 27B
  scripts; `box/accept.sh` and training disagree on `NCCL_NVLS_ENABLE`; FlashQLA needs K = V = 128; FA4 needs the cu13
  variant; verl has a known sm_103 failure; the 4B/9B baselines are concurrent single-GPU synthetic runs.
