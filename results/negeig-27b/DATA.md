# Training data for the Qwen3.8-27B retrofit (real-use step)

Goal: fine-tune Qwen3.8-27B (post-trained, served think-off for these tasks) with LoRA plus the widened-beta
gate (`wide`) against LoRA alone (`ctrl`), and show that the state tracking the 4B and 9B runs learned transfers to
agent, business and on-call work, without losing tool use, chat or code. Lessons from the 4B/9B lane
(`results/negeig-retrofit-preregistration.md`) that shape every set below:

- **No multiple-choice letters** as answers or as salient inputs. The 9B gate keyed on letter tokens and interfered
  with letter-scored evals. Answers are names, numbers, paths, statuses and versions.
- **Clean replay only** (FineWeb-Edu, code, the model's own chat), never WikiText: its tokenized format leaked into
  code generation. Replay dose is a tuned knob: strong replay delayed the hardest switch at 4B.
- **Training sessions as long as real sessions.** Tracking held to about 2x the training length at 4B and 9B.
- **Dense supervision.** The switch needs many supervised checkpoints: a question every few events, not one at the end.

## S1. State-tracking sessions (supervised, exact ground truth)

A session is a multi-turn chat. The system prompt says what is being tracked. Each user turn delivers a batch of
events (1 to 16) and ends with one question about the current state; the assistant answers with the value only.
Loss on assistant turns only. The model must carry the state; nothing in a user turn restates it.

| domain | events | questions | mechanism mix |
|---|---|---|---|
| `custody` | items handed between people and places ("Dana hands the badge to Omer") | who or where has item X now | swaps (exchanges), moves, distractors |
| `toggles` | feature flags, services, locks, doors switched on/off, acquired/released, toggled | is X on; who holds lock L | toggles (parity), sets, distractors |
| `fsys` | a shell session: `mv`, `cp`, `rm`, `mkdir`, `cd`, renames over a file tree | where is file X; does path P exist; what is in directory D | renames and moves (permutations), creation, deletion |
| `codetrace` | a Python program: assignments, tuple swaps, `x, y = y, x`, flag flips, list element swaps, increments | value of v after line n | swaps, toggles, overwrites |
| `ops` | incidents opened, acknowledged, mitigated, rolled back, resolved; deploys; on-call handoffs; alerts | which incidents are open; version of service S; who is on call | overwrites, toggles, handoffs (swaps) |
| `orders` | orders placed, paid, shipped, returned, address changed, items moved between orders | status, address, items of order N | overwrites, moves between containers |

- Languages: English for all; Hebrew for `custody`, `toggles`, `ops` and `orders` (about 25% of those sessions).
- Lengths: train sessions of 32 to 256 events (bucketed); eval sessions of 32, 256, 512 and 1,024 events.
- Entity pools (people, items, services, files, variables) are split: eval uses names never seen in training.
- Every generator has a reference simulator and an independent checker that recomputes every answer from the
  rendered text; a build fails on any mismatch.
- About 5% of questions ask about something unchanged since the start, and some turns carry no relevant event
  (the answer is unchanged); the model must not assume every event matters.

## S2. Agentic state environments (reinforcement learning, later)

Tool-using episodes whose simulators keep the true state: an ops console (list incidents, acknowledge, roll back,
query deploys), a file-system shell, an order desk. The reward is the correctness of final answers and the final
environment state. Built on the Hebrew lane's `agentic_env.py` pattern (token-in-token-out, `qwen3_coder` parsing),
with a simulator in place of canned results.

## S3. Behavior replay

The Hebrew lane's `agentic_pool.py` items (single, parallel, irrelevance, missing parameter, multi-turn, chain,
long result, unsupported), in both RL and SFT, with targets from the untouched model itself. These protect exactly
the BFCL behaviors the office SFT broke.

## S4. General replay

The clean corpus (`prep_replay_clean.py`: FineWeb-Edu plus Python) as a language-model loss, and chat responses
from the untouched model on diverse instructions. No output of a hosted model is used as a training label.

## Evaluation (held out)

S1 eval split (longer sessions, unseen names, both languages); BFCL v4 (the Hebrew lane's harness); tau-bench;
ORCA-bench public set; MMLU-Pro scored by option text (cloze) and by letter; HumanEval; IFEval.
