# Negative-eigenvalue retrofit of a pretrained Gated DeltaNet hybrid: protocol

Status: **4B and 9B done, no kill gate trips; every open question explained; rented box torn down 2026-09-30 07:31 IDT.** Next (owner's call, not started): a real-use fine-tune on Qwen3.8-27B
Date: 2026-09-29

## What decides (owner, 2026-09-29)

The owner asked for the research in the idea card, not for this repository's ledger rules. The
result is judged by the card's kill gates, as revised in the assessment the owner approved
(`experiments/negeig_retrofit/analyze.py`):

1. Kill if `wide` beats `ctrl` by less than **10 accuracy points** on the tier-1 tasks in the
   4x length window (steps 65-256; mean over parity, swap, codeswap and seeds).
2. Kill if general evals drop by more than **1 point**: full MMLU-Pro (12,032 items, zero-shot
   letter scoring) and HumanEval pass@1 (164, greedy), mean over seeds, `wide` vs `ctrl`.
3. Kill if the widened range is never used (beta above 1 on under 1% of tokens in every layer).

Reported alongside, not deciding: the 16x window, per-task differences with paired bootstrap
intervals, tier-2 tasks (expected to fail by construction), locked seeds, held-out WikiText NLL,
and gate usage. The ledger-style sections further down (20% floor, P1-P5 slices, G0 manifest,
audit) were written before this ruling and are kept only as a record; they do not decide.

## Main runs (2026-09-29, rented 8x RTX PRO 6000, Vast 53398345)

Recipe (`box_main_4b.txt`): mixed tier 1 at fixed counts per step (16 parity + 64 swap + 64 codeswap, single-task
micro-batches), LoRA rank 16 plus the gate at learning rate 1e-4 for both arms, constant after 5% warmup, 1,200
steps, 12 WikiText replay chunks per step, validation eval every 100 steps. Final test eval: seed 10,000, 128
sequences per task at 1,024 steps; then full MMLU-Pro and HumanEval. Seeds 0-2 per arm. Box checks before any
staging: a CUDA allocation on all 8 GPUs, G0 on the box (bit-identical at W = 0, gradient in all 24 layers).

Test set, accuracy in the 1x / 4x / 16x windows:

| run | parity | swap | codeswap | MMLU-Pro | HumanEval |
|---|---|---|---|---|---|
| wide s0 | 1.00 / 0.88 / 0.50 | 1.00 / 0.55 / 0.20 | 1.00 / 0.63 / 0.20 | 40.37 | 19.5 |
| wide s1 | 1.00 / 0.98 / 0.58 | 1.00 / 0.79 / 0.24 | 1.00 / 0.72 / 0.20 | 41.00 | 28.0 |
| wide s2 | 1.00 / 0.98 / 0.56 | 1.00 / 0.74 / 0.22 | 1.00 / 0.79 / 0.22 | 39.99 | 32.3 |
| ctrl s0 | 0.73 / 0.50 / 0.50 | 0.39 / 0.20 / 0.20 | 0.51 / 0.20 / 0.20 | 39.30 | 17.1 |
| ctrl s1 | 0.68 / 0.50 / 0.50 | 0.39 / 0.20 / 0.20 | 0.45 / 0.20 / 0.20 | 32.90 | 8.5 |
| ctrl s2 | 0.65 / 0.50 / 0.50 | 0.36 / 0.20 / 0.20 | 0.88 / 0.20 / 0.20 | 35.52 | 18.9 |
| untouched base | | | | 42.23 | 50.0 |

Card verdict (`analyze.py`, `results/negeig-retrofit/main/analysis.json`): **no kill.**
1. Tier-1 mean at 4x: wide 0.784, ctrl 0.300, **+48.3 points [38.4, 54.1]** (bar: 10). At 16x: +2.4 [0.1, 4.0].
   Tier 2 (s5full, z3): both arms at chance in every window.
2. General evals, wide against ctrl: MMLU-Pro +4.5 points, HumanEval +11.8. Neither drops.
3. Gate used: beta above 1 on 29% to 33% of tokens in the most-used layer, per seed.

**Cost against the untouched base, and its cause.** Both arms lost general ability against the base (MMLU-Pro
-1.8 wide, -6.3 ctrl; HumanEval -23 wide, -35 ctrl). The HumanEval completions have correct bodies but WikiText-103
formatting: `a , b = b , a % b`, `space @-@ delimited`, and a space-led `\n def ...` after the function that the
harness's stop strings do not cut, so the extra text breaks the program. Batched and single generation fail the
same items (checked on 16 lost items), so it is not a padding artifact. The replay corpus taught its tokenized
format; the swap-only runs at about 18% replay and fewer steps lost nothing. A clean-replay pair (FineWeb-Edu plus
Python, `prep_replay_clean.py`) is running to test that.

## 9B confirmation (2026-09-30, Qwen3.5-9B-Base 68c46c4b, same recipe as the 4B main runs)

Test set, 1x / 4x / 16x windows:

| run | parity | swap | codeswap | MMLU-Pro | HumanEval |
|---|---|---|---|---|---|
| wide s0 | 1.00 / 0.98 / 0.60 | 0.41 / 0.20 / 0.20 | 1.00 / 0.28 / 0.20 | 48.93 | 51.8 |
| wide s1 | 1.00 / 0.99 / 0.70 | 0.40 / 0.20 / 0.20 | 1.00 / 0.52 / 0.21 | 49.72 | 45.1 |
| wide s2 | 1.00 / 0.80 / 0.50 | 0.39 / 0.20 / 0.20 | 1.00 / 0.62 / 0.20 | 48.74 | 47.0 |
| ctrl s0-s2 | 0.68-0.70 / 0.50 / 0.50 | 0.38-0.40 / 0.20 | 0.43-0.48 / 0.20 | 48.61-50.08 | 29.9-42.7 |
| wide, clean replay s0 | 1.00 / 0.77 / 0.50 | 0.97 / 0.52 / 0.20 | 1.00 / 0.62 / 0.21 | 49.48 | 54.3 |
| untouched base | | | | 54.84 | 60.4 |

Tier-1 mean at 4x: wide about 0.53, ctrl about 0.30 (+23 points, above the 10-point bar). Parity switched in every
wide run and holds better at 16x than at 4B (0.60-0.70). Swap did not switch in any main 9B seed within 1,200
steps; the three seeds end in the pre-switch state (first 8 steps exact, 1x accuracy still rising 0.29 to
0.42-0.45), the same signature 4B wide s0 showed one eval before its switch, and the clean-replay 9B run switched at
step 1,200.

**Continuations (+600 steps from each final state, fresh task stream; `results/negeig-retrofit/explain/`): swap was
delayed, not blocked.** All three 9B wide seeds switched swap: 1x / 4x / 16x 1.00 / 0.61 / 0.20, 1.00 / 0.58 /
0.20, 1.00 / 0.66 / 0.21; codeswap 4x 0.40, 0.90, 0.59; parity 4x 0.97, 0.98, 0.87. The 9B control continued the
same way extrapolates nothing (parity 0.92 / 0.51, swap 0.43 / 0.20, codeswap 0.66 / 0.20). After continuation the
9B tier-1 mean at 4x is about 0.73 against 0.30 (+43 points), in line with the 4B's +48. The larger model needs more
steps for swap on this recipe. HumanEval of the continued wide runs 50.6 / 45.7 / 41.5, MMLU-Pro 50.1 / 48.5 / 46.8.

**The remaining HumanEval gap with clean replay is not formatting.** Clean 9B wide lost 21 of the base's passes and
gained 11 others (net -10 of 164; discordant 21 against 11, p about 0.1); the lost completions are ordinary logic
errors (an off-by-one palindrome, an unsorted pair, `sorted` where a set comparison was needed, `<` for `<=`, one
`pass` stub), no WikiText-style text.

**Clean replay delays the hardest switch at 4B.** Two fresh 4B clean-replay seeds: parity 1.00 / 0.93 and 1.00 /
0.78, codeswap 0.67 and 0.61 at 4x, swap unswitched in both (0.32-0.33 / 0.20); seed 0 stays unswitched on parity
and swap after 1,800 steps. So under clean replay swap switched in 0 of 3 4B seeds within 1,200 steps (3 of 3 with
WikiText), parity in 2 of 3, codeswap in 3 of 3; the 9B clean run switched swap at step 1,200. Replay that holds the
model at its pretrained distribution competes with the hardest switch; the replay dose is a recipe question for the
next model. 9B clean-replay pair: MMLU-Pro wide 49.48, ctrl 52.06 (the letter interference again), HumanEval wide
54.3, ctrl 52.4.

**Why 9B parity holds at 16x and 4B parity does not (`horizon_diag.py --task parity`).** The 9B wide s1 parity head
(layer 30, head 6) sets beta 1.9989 on every 1 and 0.0000 on every 0 with decay 0.9999: eigenvalue -0.9988 on a
flip, +0.9998 on a keep, about 1,330 steps before the state fades to 20%. The 4B wide s1 parity head (layer 28, head
28) flips at beta 1.9613 (eigenvalue -0.961), about 41 steps to 20% by magnitude. The larger model found a far more
exact reflection.

**Learning rate (swap only, batch 64, constant, wide, seed 1).** 3e-5: no switch by step 800. 1e-4: switch at step
600, 4x 0.81-0.86 by 1,200. 3e-4: switch at 600, 4x 0.61 at 800. 1e-3: switch by step 200, 4x 0.80-0.90 from step
200 to 600 (0.76 at 800). A higher rate shortens the plateau about threefold; its general-eval cost was not
measured. The ctrl run at 1e-3 was stopped at step 200 for the 16x check and not requeued.

**Tier 2, both arms (laptop):** s5full wide 0.21 / 0.20, ctrl 0.21 / 0.20; z3 wide 0.56 / 0.34, ctrl 0.58 / 0.34
(1x / 4x). One reflection per token does not reach tier 2, as the theory says.

**Box and cost.** Vast 53398345 (8x RTX PRO 6000 Server 600 W, California), 21:09 to 07:31 IDT at $11.79/h, about
$122; a first box stuck pulling its image for 21 minutes was destroyed (storage-only billing). Teardown confirmed
from the instance listing.

## Explaining the general-eval numbers (2026-09-30, `results/negeig-retrofit/diagnostics/`)

**Letter bias.** The tasks answer with ' A'..' E', the tokens MMLU-Pro's zero-shot letter scoring reads. Every
fine-tuned model predicts A-E more and F-J about half as often as the base (`mmlu_letters.py`). Removing each
model's per-letter prior (calibrated accuracy) moves the numbers: 4B base 47.76; 4B wide 45.36, ctrl 42.79
(wide +2.6); 9B base 56.05; 9B wide 50.13, ctrl 52.44 (wide -2.3, all three seeds).

**Cloze scoring removes the 9B gap.** MMLU-Pro scored by the likelihood of each option's text, no letters
(`mmlu_cloze.py`, 3,008 questions): 9B base 29.95; wide 26.80 / 27.43 / 27.26 (mean 27.16); ctrl 26.43 / 27.89 /
25.90 (mean 26.74); wide with clean replay 28.56. Wide is 0.4 ahead of ctrl. The calibrated letter gap at 9B is
letter-format interference, not lost knowledge; both arms lose about 3 points to the base from the WikiText drift.

**Where the interference comes from.** The swap task feeds letter tokens as inputs (the ball's starting cup). On
MMLU-Pro letter prompts the 9B wide gate opens (beta > 1) 1.18-1.26x more often at option-letter tokens than at
other tokens, 3-5x in a few layers (layer 4: 4.8x, layer 17: 3.9x); the 4B wide gate 1.03-1.09x
(`gate_on_letters.py`). Correlational, and consistent with the scale difference. Future task designs must not reuse
multiple-choice letters as labels or inputs.

**Drift on ordinary text** (`gate_kl.py`, held-out web + code and WikiText). KL from the untouched base per token
on web + code: WikiText-replay runs 0.24-0.33 in both arms (NLL +0.23 to +0.31 nats), clean-replay runs 0.036-0.046
(NLL at or below the base). The WikiText replay causes the drift, in both arms. The gate is not task-specific: it
opens on about 7% of ordinary tokens (busiest layer 19-26%) against 14-17% on task tokens, but its cost there is
small (4B clean replay: wide 0.044 against ctrl 0.036). At 9B wide drifts less than ctrl (0.24-0.27 against
0.27-0.30).

## Follow-ups on the rented box (2026-09-30, `results/negeig-retrofit/followup/`)

Test set, 1x / 4x / 16x windows, one seed each unless shown:

| run | parity | swap | codeswap | MMLU-Pro | HumanEval |
|---|---|---|---|---|---|
| wide, 128-step training (s0) | 0.67 / 0.51 / 0.50 | **1.00 / 0.96 / 0.40** | **1.00 / 0.98 / 0.28** | 41.26 | 32.9 |
| ctrl, 128-step training (s0) | 0.68 / 0.50 / 0.50 | 0.38 / 0.20 / 0.20 | 0.48 / 0.20 / 0.20 | 28.91 | 9.8 |
| wide + decay gate s0 | 1.00 / 0.97 / 0.54 | 0.32 / 0.20 / 0.20 | 1.00 / 0.58 / 0.20 | 38.04 | 28.0 |
| wide + decay gate s1 | 0.61 / 0.50 / 0.50 | 0.35 / 0.20 / 0.20 | 1.00 / 0.67 / 0.20 | 40.66 | 29.3 |
| wide + decay gate s2 | 1.00 / 0.93 / 0.51 | 1.00 / 0.78 / 0.25 | 1.00 / 0.43 / 0.20 | 39.41 | 37.8 |
| ctrl + decay gate s0 | 0.62 / 0.50 / 0.50 | 0.39 / 0.20 / 0.20 | 0.45 / 0.20 / 0.20 | 31.14 | 17.7 |
| wide, clean replay (s0) | 0.72 / 0.50 / 0.50 | 0.40 / 0.20 / 0.20 | 1.00 / 0.74 / 0.20 | 42.72 | 43.9 |
| ctrl, clean replay (s0) | 0.72 / 0.50 / 0.50 | 0.38 / 0.20 / 0.20 | 0.55 / 0.20 / 0.20 | 43.48 | 43.3 |

**Horizon follows training length.** With 128-step training sequences, swap is exact to step 192 (0.90 at 193-256,
0.73 at 257-384, 0.53 at 385-512) and codeswap exact to step 256 (0.95 at 193-256, 0.63 at 257-384); with 64-step
sequences both fall to about 0.4-0.6 by steps 193-256. In both cases tracking holds to about 2x the training length.
Parity did not switch in this seed.

**The decay gate is dropped.** It gives no 16x gain (parity 0.51-0.54, swap at most 0.25, against 0.50-0.58 and
0.20-0.24 without it) and switching got less reliable (seed 1 missed parity and swap, seed 0 missed swap). At 64-step
training nothing pushes alpha to 1, so the extra capacity goes unused.

**Clean replay restores general ability and slowed the switch.** FineWeb-Edu plus Python replay (12 chunks per
step) brings MMLU-Pro to the base (42.7 against 42.2) and HumanEval to 43.9 (WikiText: 19.5), so the WikiText
formatting was the main cause of the general loss; about 6 HumanEval points stay lost in both arms. The same seed
and task stream switched only codeswap (at step 800) and never parity or swap, where the WikiText run switched parity
by step 400 and swap by 1,000. Real text keeps the model near its pretrained distribution and competes with the
switch; one seed, so delayed and blocked are not yet separable.

## Before the main runs: learning rate and learnability (2026-09-29)

The first main campaign ran on a learning rate picked, not measured, and was judged on loss. The owner
stopped it: evaluate at every checkpoint, check what is being learned, measure the learning rate. It was
stopped and archived (`main_aborted_guessed_lr/`). What followed, all on Qwen3.5-4B-Base, validation set
(seed 20,000, 32 sequences of 256 steps), accuracy in the 1x window / 4x window. Chance: parity 0.50,
swap and codeswap 0.20, z3 0.33. Curves: `results/negeig-retrofit/curves/`.

**Sweep, 400 steps, seed 0, mixed 5-task batch of 16, cosine.** LoRA and gate learning rates tied unless
shown.

| arm | lr (LoRA / W) | parity | swap | codeswap | beta > 1 |
|---|---|---|---|---|---|
| ctrl | 3e-5 | 0.49 / 0.50 | 0.21 / 0.21 | 0.21 / 0.19 | 0 |
| ctrl | 1e-4 | 0.53 / 0.50 | 0.23 / 0.21 | 0.25 / 0.19 | 0 |
| ctrl | 3e-4 | 0.54 / 0.49 | 0.26 / 0.20 | 0.25 / 0.21 | 0 |
| ctrl | 1e-3 | 0.59 / 0.49 | 0.26 / 0.21 | 0.25 / 0.20 | 0 |
| wide | 3e-5 | 0.52 / 0.49 | 0.22 / 0.20 | 0.24 / 0.20 | 11% |
| wide | 1e-4 | **1.00 / 0.84** | 0.24 / 0.21 | 0.26 / 0.20 | 15% |
| wide | 3e-4 | 0.56 / 0.49 | 0.24 / 0.20 | 0.27 / 0.19 | 25% |
| wide | 1e-3 | **1.00 / 0.73** | 0.27 / 0.21 | 0.23 / 0.20 | 35% |
| wide | 1e-4 / 3e-5 | 0.52 / 0.49 | 0.22 / 0.20 | 0.22 / 0.20 | 7% |
| wide | 1e-4 / 3e-4 | 0.53 / 0.50 | 0.23 / 0.20 | 0.25 / 0.20 | 24% |
| wide | 1e-4 / 1e-3 | 0.54 / 0.49 | 0.23 / 0.21 | 0.21 / 0.20 | 33% |

Parity switches on abruptly (steps 300 and 350 in the two runs that got it). Within 400 steps the switch
is too noisy to rank learning rates. No run learned swap or codeswap.

**Long runs, 1,600 steps, seed 1, same batch.**

| arm | lr | parity | swap | codeswap | z3 | beta > 1 |
|---|---|---|---|---|---|---|
| wide | 1e-4 | **1.00 / 0.92** (switch at step 600) | 0.30 / 0.21 | 0.30 / 0.20 | 0.40 / 0.34 | 15% |
| ctrl | 1e-3 | 0.92 / **0.51** | 0.36 / 0.21 | 0.34 / 0.21 | 0.52 / 0.34 | 0 |

The control fits parity inside the training length and does not extrapolate; the widened gate
extrapolates to 4x. That is the card's prediction, on one seed each. Swap and codeswap stay at chance in
the 4x window in both arms; training loss on swap moved from 1.61 (chance) to about 1.36-1.46.

**Why swap is not learned: the plateau (G2c, G2d).** A tiny from-scratch fla GDN (2 layers, width 128)
with beta in (0, 2), on the same token format, trained on swap only. Switch = first eval with 4x accuracy
above 0.6. Before the switch every run sits on the same plateau: the first 4 to 6 swaps right, then chance.
The narrow control shows the same plateau, so the plateau does not use the negative range; only the switch
does.

| batch | length | lr | switch (sequences) |
|---|---|---|---|
| 64 | 64 | 1e-3 | seeds 0-3: 192k, 96k, 70k, none by 397k |
| 64 | 128 | 1e-3 | seeds 1-2: 83k, none by 397k |
| 256 | 64 | 1e-3 | 154k (600 steps) |
| 64 | 64 | 3e-3 | 218k |
| 16 | 64 | 1e-3 | none by 128k |
| 16 | 128 | 1e-3 | none by 400k |
| 16, 5 tasks mixed | 64 | 1e-3 | none by 51k swap sequences (parity on, as in the retrofit) |

So the retrofit runs above (about 5k swap sequences, batch 16 shared by five tasks) were never in a regime
where swap could switch on. Batch size matters, not only the sequence count: batch 16 did not switch at
any budget tried. The published results used far larger budgets (Grazzi et al. and DeltaProduct train
at batch 512 to 1024 for 100 epochs).

**Codeswap and the tier-1 mix from scratch (G2e).** Codeswap switches fast with the widened range:
12.8k sequences on both seeds (4x 0.74 and 0.78 at the switch). The narrow control fits it partly
in the training length (1x 0.53) and stays at chance in 4x after 198k sequences. Codeswap asks what
variable `a` holds, which is a direct read of the delta-rule state (`S q` with `q` fixed); swap asks
where the ball is, which needs the inverse, and is the slow one. Tier-1 mixed at batch 192 (about 64
per task): parity switched by 19k sequences, codeswap by 38k, swap not by 595k (about 198k swap
sequences).

**The retrofit learns swap (4B, `diag/swap_wide_lr1e-4_b64`).** Wide, swap only, batch 64, constant
learning rate 1e-4, seed 1, stopped at step 1,600:

| step | swap 1x | swap 4x | first 8 steps |
|---|---|---|---|
| 0 | 0.20 | 0.20 | 0.44 0.34 0.25 0.25 0.22 0.16 0.16 0.12 |
| 200 | 0.28 | 0.20 | 1.00 1.00 1.00 1.00 0.94 0.78 0.72 0.38 |
| 400 | 0.34 | 0.20 | 1.00 1.00 1.00 1.00 1.00 1.00 0.97 0.91 |
| 600 | 0.92 | 0.34 | all 1.00 |
| 800 | 1.00 | 0.62 | all 1.00 |
| 1200 | 1.00 | 0.86 | all 1.00 |
| 1600 | 1.00 | 0.82 | all 1.00 |

The control on the same recipe (`diag/swap_ctrl_lr1e-4_b64`) never switched: 1x crept from 0.25 to
0.37 as the plateau stretched to about 8 correct swaps, 4x stayed at 0.20-0.21 through step 1,600.
Swap 4x at step 1,600: wide 0.81, ctrl 0.20 (one seed each).

**Codeswap on the 4B (`diag/codeswap_wide_lr1e-4_b64`).** Wide, codeswap only, effective batch 64 (4
micro-batches of 16), constant learning rate 1e-4, 800 steps: 1x / 4x at steps 300, 400, 500, 600, 700, 800
were 0.37 / 0.20, 0.55 / 0.21, 0.94 / 0.28, 1.00 / 0.55, 1.00 / 0.49, 1.00 / 0.57. Training loss 0.002 by
step 700. The switch came at the same point as swap (steps 400 to 600), but 4x settles lower (about 0.55
against swap's 0.81). Each codeswap step costs 12.3 s against 3.3 s for swap (about 450 tokens per
sequence against 70). The control on the same recipe (`diag/codeswap_ctrl_lr1e-4_b64`) fits the
training length slowly (1x 0.52 at step 500, 0.89 at step 800, still rising) and stays at chance past it
(4x 0.20-0.22 at every checkpoint). Codeswap 4x at step 800: wide 0.57, ctrl 0.21.

**Tier-1 at 4x, one seed each: parity 0.92 vs 0.51 (+41), swap 0.81 vs 0.20 (+61), codeswap 0.57 vs 0.21
(+36).** In all three the control fits the training length and does not extrapolate. 

**General evals on the single-task adapters (kill gate 2, before any main run).** Full MMLU-Pro (12,032
items, zero-shot letter scoring) and HumanEval pass@1 (164, greedy), about 12 minutes per model. Paired
differences with 95% bootstrap intervals over items (`compare_general.py`,
`results/negeig-retrofit/general/`). Untouched base: MMLU-Pro 42.23, HumanEval 50.00.

| adapter | MMLU-Pro | HumanEval |
|---|---|---|
| swap ctrl | 43.20 (+0.97 vs base) | 46.34 (-3.66) |
| swap wide | 42.79 (+0.57) | 50.00 (0.00) |
| swap wide - ctrl | -0.41 [-1.01, +0.17] | +3.66 [-1.83, +9.15] |
| codeswap ctrl | 27.60 (-14.63 [-15.48, -13.78]) | 36.59 (-13.41) |
| codeswap wide | 40.72 (-1.51 [-2.34, -0.72]) | 46.34 (-3.66 [-10.37, +3.66]) |
| codeswap wide - ctrl | +13.12 [+12.43, +13.85] | +9.76 [+1.83, +17.68] |

The card's gate (wide against ctrl) passes on both tasks. The control trained on codeswap collapsed: it
keeps pushing to fit the training length without the flip, and that costs 14.6 MMLU-Pro points. Against the
untouched base, codeswap-trained wide still loses 1.5 MMLU-Pro points (interval excludes zero). The likely
cause is replay dilution: a codeswap step is about 29k task tokens against about 1k WikiText replay tokens
(3.5%); the swap runs had about 18% and lost nothing. The mixed pilot therefore gets 12 replay chunks every
step (about 6k tokens against about 35k task tokens).

**16x check (validation seed, 32 sequences of 1,024 steps, `eval16.py`, `results/negeig-retrofit/eval16/`).**
Every wide adapter decays to chance by 16x. Accuracy by position:

| adapter | 1-64 | 65-128 | 129-192 | 193-256 | 257-384 | 385-512 | 513-1024 |
|---|---|---|---|---|---|---|---|
| parity wide (5-task run) | 1.00 | 1.00 | 0.97 | 0.79 | 0.60 | 0.49 | 0.51 |
| parity ctrl (5-task run, lr 1e-3) | 0.93 | 0.51 | 0.48 | 0.51 | 0.49 | 0.51 | 0.51 |
| swap wide | 1.00 | 1.00 (65-96), 0.99 (97-128) | 0.98 (129-160), 0.85 (161-192) | 0.70 | 0.37 (257-320) | 0.22 | 0.20 |
| codeswap wide | 1.00 | 0.94 | 0.59 | 0.36 | 0.21 | 0.20 | 0.20 |

The controls fall to chance right after step 64. The retrofit extends the working range from 1x to about 2
to 3x, not to arbitrary length.

**Why it stops (`horizon_diag.py` on the swap adapter).** One head carries swap: layer 5, head 14, with
beta 1.9986 on every step token (lowest 1.992) and eigenvalue along the key -0.9954. The reflection is
essentially exact. Its decay is alpha = 0.9968 per token, which shrinks the state to 20% after about 350
steps; swap accuracy reaches chance at about step 320. Training sequences of 64 steps give almost no
pressure on alpha (0.9968^64 = 0.81). The other 48 heads with beta above 1.5 decay within a few steps.
Two ways to extend the horizon: train on longer sequences (inside the card's method; the rented box runs a
128-step pair), or give the decay a zero-initialized gate like beta's so alpha can reach 1 (a method
extension, not run).

GPU training here is not bit-deterministic run to run (parameter differences of 1e-5 to 1e-4 after three
steps on a tiny model, in both the old and the new trainer); the task and replay streams are identical
across arms for a seed.

The swap switch came between steps 400 and 600 (26k to 38k sequences), faster than any from-scratch run
(70k to 192k, or never). Beta above 1 on 15% of tokens. Codeswap did not come along (chance), and
parity, untrained here, stayed at chance. Queue (`diag_stage2d.txt`): the control on the same recipe
at learning rates 1e-4 and 1e-3, a codeswap-only wide run (effective batch 64 through 4 micro-batches),
then wide learning-rate probes at 3e-4, 1e-3 and 3e-5.

## Claim

Qwen3.5's Gated DeltaNet layers compute `beta = sigmoid(b)`, so every token's
transition `alpha (I - beta k k^T)` has eigenvalues in `[0, alpha]`. A linear
recurrence with only nonnegative eigenvalues cannot represent parity at
arbitrary length in finite precision (Grazzi et al., arXiv 2411.12537, ICLR
2025). Widening beta to `(0, 2)` admits a reflection per token (eigenvalue
`alpha (1 - beta)` down to `-alpha`), which is enough for parity and for
permutation tasks whose steps are single transpositions.

The claim under test: this capability can be **retrofitted into an already
pretrained hybrid** with a small gated parameter and a short fine-tune, and it
shows up as length generalization that the same fine-tune without the widened
range does not reach. Only from-scratch models ship the widened range today
(Olmo-Hybrid-7B, Solar-Open2-250B, GDN-2); Qwen3.5/3.8, Kimi Linear and K3, and
GLM-5.3-Flash do not. No published retrofit was found (assessment of the idea
card, 2026-09-29: all 105 citers of 2411.12537 and 67 of DeltaProduct
2502.10297 scanned by title).

What this can and cannot establish under this ledger: a win is a **component
result** on synthetic state tracking plus a code-format transfer task. It is not
a smarter model until an end-to-end task moves.

## What one reflection per token can and cannot do

- Tier 1 (reachable): parity (Z2); one ball under five cups where each token is
  one transposition; the same group written as Python tuple swaps.
- Tier 2 (not reachable by one real reflection per token per layer, reported as
  expected-fail): full S5 with an arbitrary permutation per token (needs up to
  four reflections: Grazzi Thm 3, DeltaProduct, Complex KDA Thm 4); running sum
  mod 3 (needs a rotation; Qwen's decay is a positive scalar, so beta-only
  widening gives real eigenvalues only).

## Arms (identical except for the gate)

Base: Qwen3.5-4B-Base, revision `1001bb4d826a52d1f399e183466143f4da7b741b`
(24 GDN + 8 full-attention layers, hidden 2560, 32 value heads, 16 key heads).

- `ctrl`: LoRA rank 16, alpha 32, on every GDN layer's `in_proj_qkv`,
  `in_proj_a`, `in_proj_b`, `out_proj`. The library forward is untouched, so
  beta stays in `(0, 1)`. LoRA on `in_proj_a` lets both arms move the decay
  toward 1, which a long parity chain needs.
- `wide`: the same LoRA plus, per GDN layer, a zero-initialized
  `W in R^{32 x 2560}` and
  `t = tanh(W x)`, `beta = s + t * (2 - s if t >= 0 else s)`, `s = sigmoid(b)`.
  At `W = 0` the model is bit-identical to `ctrl` at step 0 (gate G0), every
  token can reach beta in `(0, 2)` whatever its pretrained `s`, and `W` gets a
  nonzero gradient at zero. Extra parameters: 24 x 32 x 2560 = 1,966,080
  (0.05% of the base); extra serving cost: one 2560 x 32 matvec per GDN token.

The patch rewrites only the line `beta = b.sigmoid()` of the installed
transformers 5.17.0 forward (asserted) and routes through the same fla kernel.

## Data

Each task is a token stream; labels are read from the LM head at chosen
positions and **never enter the stream**, so no arm can copy a previous answer.
Prediction is the argmax over the task's label tokens (' A'..' E').

| task | tier | step token(s) | label |
|---|---|---|---|
| parity | 1 | '0' or '1' | running parity |
| swap | 1 | one of 10 single-token transpositions (' ab'..' de') | cup holding the ball |
| codeswap | 1 | `x,y=y,x\n` (4 to 5 tokens) | current value of variable `a` |
| s5full | 2 | one of 120 single-token words, each an arbitrary permutation | cup holding the ball |
| z3 | 2 | '0', '1' or '2' | running sum mod 3 |

Labels are recomputed from the token stream by an independent reference
(`tasks.reference_check`) before any run.

Replay: WikiText-103 raw train, 512-token chunks, next-token loss on 4 chunks
every second step, identical order across arms for a seed. Held-out guard:
256 chunks of WikiText-103 test.

## Training (frozen before the main runs)

64 task steps per training sequence; batch 16 task sequences with tasks drawn
uniformly; AdamW (0.9, 0.95), no weight decay; LoRA learning rate 2e-4; `W`
learning rate 5e-4; 5% warmup then cosine; gradient clip 1.0; bf16 autocast;
gradient checkpointing. **1,200 steps.** Seeds 0 to 4 for each arm; for a given
seed the task stream and replay chunks are identical across arms. The smoke run
(30 steps, wide arm, seed 99, not a result) measured about 580 tokens/s and a
13.96 GB peak on the RTX 5090 Laptop (24 GB), so one run is about 1.3 to 1.5
hours including eval. Launcher: `experiments/negeig_retrofit/run_main.sh`
(resumable; arms alternate within each seed; CPU capped to cores 8-15 at nice 10).
An untouched-base reference (`ctrl`, zero steps) is evaluated first.

## Evaluation

128 fixed sequences per task at 64, 256 and 1,024 steps (1x, 4x, 16x the
training length), same sequences for every run. Windows are taken on the
1,024-step set: steps 1-64 (1x), 65-256 (4x) and 257-1,024 (16x).

## Primary endpoint and gate

Tier-1 error rate in the **4x window (steps 65-256)**, averaged over the three
tier-1 tasks with equal weight. (Amended before any main run: the G2 positive
control showed a from-scratch widened GDN at 0.99 parity accuracy in the 4x
window but only 0.61 to 0.67 in the 16x window, against 0.50 for the (0, 1)
control, so a 16x primary could return a null even when the mechanism works.
The 16x window is kept as a secondary endpoint.) Relative error reduction of `wide` over `ctrl`:
`1 - err_wide / err_ctrl`.

**Success (ledger floor):** relative error reduction of at least 20%, with the
one-sided 95% lower bound of a hierarchical bootstrap (resample seed pairs, then
sequences within each seed; 10,000 resamples) at or above 20%. A point estimate
of 20% or more with a bound below 20% is not a success. 10-20% is an arguable
finding allowing one bounded replication; below 10% closes the candidate.

## Secondary endpoints (reported, not gated)

- The same relative error reduction in the 16x window.
- Locked seeds per arm and task in the 4x window (accuracy of at least 0.99), used
  by the kill rule, and in the 16x window (reporting only). Fisher exact test on
  locked counts.
- Per-task error reductions in every window.
- Tier-2 accuracy at every window (expected near chance for both arms).
- Gate usage: fraction of beta above 1 and mean |t| per GDN layer during eval.

## Guards (the ledger's protected slices P1 to P5)

Each is a noninferiority check of `wide` against `ctrl` unless stated; any
failure blocks a success claim regardless of the primary endpoint.

- P1: held-out WikiText-103 NLL; the upper 95% bound of the relative increase
  over `ctrl` must not exceed 0.5% (hierarchical bootstrap: seeds within each arm,
  then shared chunks).
- P2: tier-1 accuracy in the 1x window may not drop more than 0.02 (absolute).
- P3: tier-2 accuracy in the 1x window may not drop more than 0.02.
- P4: no single tier-1 task may drop more than 0.02 in the 4x window.
- P5: held-out NLL against the untouched base (zero-step reference run); the
  upper 95% bound of the relative increase must not exceed 2%.

These map this training experiment onto the ledger's G0 manifest schema
(`manifests/negeig-retrofit-4b.json`), whose slices and cells were written for
serving experiments.

## Gates before the main runs

- G0: `wide` at `W = 0` is bit-identical to the untouched model on all five
  tasks and on English text; random `W` produces beta above 1 and gradient on
  every GDN layer's `W`.
- G1: fla's chunked kernel with beta in `(0, 2)` has relative error at most twice
  its `(0, 1)` error plus 1e-3, forward and every input gradient, at T = 256 and
  1,024, decays down to 0.9 and 0.99. **PASSED** (`g1_kernel.json`).
- G2: a tiny from-scratch fla GatedDeltaNet on this exact task format reaches at
  least 0.95 parity accuracy in the 16x window with `allow_neg_eigval` and stays
  at or below 0.60 without it. **FAILED as frozen** (`g2_positive.json`): parity
  16x 0.67 and 0.61 with the widened range, 0.50 without; 4x 0.99 and 0.99 vs
  0.50; swap 4x 0.86 and 0.89 vs 0.19 and 0.20. The format shows the effect at 4x
  and fades by 16x even from scratch. No single factor fixes the fade
  (`g2b_diagnose.json`, parity, widened range, 16x): fp32 weights 0.51 and 0.62;
  DeltaNet without the decay gate 0.74 and 0.53; training length 128 instead of
  64, 0.74 and 0.78 (4x window 1.00 and 1.00). It is general length-generalization
  decay, consistent with arXiv 2609.18966. This moved the primary window to 4x
  before any main run.
- G0 on the 4B base: **PASSED** (`g0_noop_4b.json`); also passed on the 9B.

## Kill rules

(Implemented in `analyze.py`; the independent label check `tasks.reference_check`
runs at the start of every training run.)

- Primary relative error reduction below 10%, or its point estimate at or
  above 10% with no tier-1 task showing any `wide` seed locked (accuracy of at
  least 0.99) in the 4x window. (The lock window moved from 16x to 4x with the
  primary window, in the same pre-run amendment.)
- The general-language guard fails.
- `wide` never uses beta above 1 (fraction below 1% in every layer), which
  would mean the mechanism was not engaged.

## Scale-up rule

Only after a success or an arguable finding at 4B: one confirmation on
Qwen3.5-9B (QLoRA if memory requires), two seeds per arm, same protocol.

## Deviations from the idea card, and why

- Full S5 is not the headline: one reflection per token cannot express it.
- The Recirculation baseline is dropped: it is a different cost axis (serial
  prefill), it has no state-tracking result, and it has not been ported to a
  hybrid.
- The trainable set is widened to the full GDN projection set in both arms, so a
  null result is not a readout-capacity artifact.
- The kill gate is restated as a relative error reduction with a confidence
  bound, per this ledger's floor, instead of "10 points".
- The additive-write ablation suggested by arXiv 2609.18966 is not run: every
  GDN layer's write path carries language, so zeroing it is not a clean
  intervention in a pretrained model. The 4x/16x windows are the test of that
  failure mode.
- MMLU-Pro is replaced by the paired held-out NLL guard for the 4B base model
  run on a laptop GPU; it may be added to the 9B confirmation.

## Audit

An independent audit (2026-09-29, before any main run) returned "admit after
fixes". Fixed before launch: the kill rule is now part of the computed verdict;
`tasks.reference_check` runs inside `train.py`; a G0 manifest exists and passes
`frontier_g0.py`; the NLL guard bootstrap resamples seeds as well as chunks; the
patch docstring states that the accelerate-hook decorator is dropped (a no-op on
one GPU, G0 bit-identical). Noted, not changed: the cached single-token decode
path uses the widened beta but is not exercised; with five seeds per arm the
seed-level bootstrap has coarse support, so a lower bound close to 20% is read
with caution.
