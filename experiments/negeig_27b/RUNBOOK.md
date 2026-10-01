# Qwen3.8-27B negative-eigenvalue retrofit: box runbook

From a new Nebius box to a finished Stage A, then to teardown, for the two-box layout (box A and box B, 8 GPUs each),
with the four-box variant marked where it differs. Every step names who acts (RIG or BOX), the exact command, and the
line that proves it worked. Do not go on to the next step without that line.

The pre-registered rules this file applies are in `results/negeig-27b/stageA-log.md`. The general evals are described in
`experiments/negeig_27b/evals/README.md`. The B300 throughput and cost figures below are planning numbers; the live run
is on H200.

## 0. Rules

- RIG acts through ssh helpers (`rig.sh`) and the scripts in `box/`. BOX scripts run on the box under `$E27/box/`.
  Nothing on a box ever reaches back to the rig.
- Quote every command for `bx`/`bxd` with SINGLE quotes, so `$E27 $W $ACC $RUNS $SD $MODEL_DIR $DATA $PYBIN` expand on
  the box, where `boxenv.sh` has already been sourced.
- The live boxes (`negeig27-h200p-a`, `-b`, `-c`, `-d`) and the live `preempt_watch.sh` belong to another session. This
  runbook never touches them. `instances.tsv` holds their rows too, so every destructive command here passes `--ids`.
  `teardown.sh` without `--ids` acts on EVERY row of the table.
- Owner gates (stop and ask, do not decide): on-demand billing (`--ondemand`, or the `b300w` and `h200` targets, which
  are on demand), the 1.5x to 1.7x throughput band (step 6), any "owner" result of the sweep reader (step 8), every
  extension rung past 700 steps (step 9), and any box type other than a preemptible B300 pair. A preemptible B300 pair is
  inside the approved envelope. On-demand B300 puts the core run at about $856 against $470 preemptible (the envelope at $1,727
  against $946).
- If the host refuses `teardown.sh ... --yes` (destructive provider calls can be refused even with chat approval), print
  the exact line and hand it to the owner to run. Do not wrap it in a script to get around the refusal.
- Never serve customer traffic from these boxes. They are development boxes.
- Receipts are the proof. A step that ends without its receipt line is not done, whatever the exit status says.
- Scratch on the rig stays under `$RECV` and is deleted when the lane closes.

## 1. Rig setup (RIG, once per shell)

The login shell is zsh and the helpers need bash.

```bash
# runbook-block: setup
# (the login shell is zsh: run `bash` first, then paste this block)
REPO=$(git rev-parse --show-toplevel)            # run from inside the checkout that holds this file
E27R=$REPO/experiments/negeig_27b
BOX=$E27R/box
. "$BOX/rig.sh"                                  # bx bxd bxs bxw bxpull, resolve_ip, SSH_OPTS
STATE=${NEGEIG27_STATE:-$HOME/.local/state/negeig27}
A=negeig27-r2-a; B=negeig27-r2-b
LANE_SRC=${SELFDISTILL_LANE_DIR:?the Hebrew RL lane dir}   # self-distillation needs agentic_env.py from it
RECV=/data/ai-ml/models/_runs/negeig-27b/pair2   # everything pulled from the boxes lands here
mkdir -p "$RECV"
idof() { awk -F'\t' -v n="$1" '$1 == n { print $2 }' "$STATE/instances.tsv"; }

# arm BOX TAG SEED LR [STEPS]   arm one run under the watchdog on BOX (one training unit per box; arming again replaces it)
#   ARM_ENV='ARMS=wide GPUS_PER_RUN=8'   (4-box layout: one arm on all 8 GPUs)    ARM_FLAGS='--lr_override'  (repair, step 8)
arm() {
  local box=$1 tag=$2 seed=$3 lr=$4 steps=${5:-700} cmd
  cmd=". \"\$ACC/grad_ckpt.env\"; ${ARM_ENV:-} exec bash \"\$E27/run_pair.sh\" $tag $seed --lr $lr --steps $steps \$GRAD_CKPT_FLAGS ${ARM_FLAGS:-}"
  bx "$box" "bash \"\$E27/box/ctl.sh\" arm --cmd $(printf '%q' "$cmd") --ckpt-dir \"\$RUNS/$tag\""
}
```

Ok when `type bx bxd bxs bxw bxpull arm idof` finds all seven functions. The names `negeig27-r2-a` and `-b` must match `negeig27-[a-z0-9-]+` and not exist in `instances.tsv`.

## 2. Create the boxes (RIG)

Dry run first. It is read-only (platform, preset, preemptible allowed, quota, billing calculator).

```bash
bash "$BOX/create_box.sh" --target b300 --count 2 --names "$A $B"
```

Ok when the last line is `DRY RUN: nothing was created`. Then create (preemptible B300 in uk-south1 is the target's
billing):

```bash
bash "$BOX/create_box.sh" --target b300 --count 2 --names "$A $B" --yes
```

Ok when each box prints `<name>  id=<id>  ip=<ip>  ssh-ok`. If it says `ssh NOT ready`, run
`resolve_ip "$A"` and `wait_ssh "$(resolve_ip "$A")" 60` before going on.

Capacity: error code 8 (no capacity) or `PrototypeNotFound` means try the next target. The targets, from
`common.sh` `target_config`:

| target | region | platform | billing |
|---|---|---|---|
| `b300` | uk-south1 | gpu-b300-sxm | preemptible |
| `b300w` | eu-west2 | gpu-b300-sxm | on demand (owner gate) |
| `h200` | eu-north1 | gpu-h200-sxm | on demand (owner gate) |
| `h200w` | eu-west1 | gpu-h200-sxm | preemptible |

`--preemptible` or `--ondemand` overrides the billing of any target. `h200w` preemptible is the live lane's type and is the
fallback that needs no billing decision, but an H200 box needs the owner's call on the throughput classes (step 6).

Record the ids and confirm from the provider's own list:

```bash
IDA=$(idof "$A"); IDB=$(idof "$B")
printf '%s %s\n%s %s\n' "$A" "$IDA" "$B" "$IDB" | tee "$RECV/ids.txt"
bash "$BOX/teardown.sh" --list                    # read-only; shows every lane box, the live ones too
```

Ok when both ids are non-empty and `--list` shows them RUNNING with label `lane=negeig-27b`.

Preemption watcher. One watcher reads every row of the table each pass, so a running one already covers the new boxes.

```bash
pgrep -af '[p]reempt_watch.sh'                    # a hit: reuse it. Check its timeout; the live one runs 30 minutes
```

If there is none, start one with a deadline (it loops forever by itself, and a second watcher would race on
`instance start`):

```bash
nohup timeout 16h nice -n 10 bash "$BOX/preempt_watch.sh" --restart --interval 60 --start-retry-s 180 \
  >>"$STATE/watch.log" 2>&1 </dev/null &
echo $! >"$STATE/watch.pid"; disown
```

Ok when `tail -n 3 "$STATE/events.log"` shows a pass with no `PREEMPTED` for your ids. Stop only a watcher you started, at
step 10: `kill "$(cat "$STATE/watch.pid")"`.

## 3. Prepare every box (RIG drives, BOX works)

Order matters: the real CUDA allocation (`bootstrap.sh --stage base`) comes before any data or weights.

```bash
# runbook-block: prepare
for b in "$A" "$B"; do bash "$BOX/push_data.sh" "$b" --code-only; done
for b in "$A" "$B"; do bxd "$b" boot-base 'bash "$E27/box/bootstrap.sh" --stage base'; done
for b in "$A" "$B"; do bxw "$b" boot-base 1800 20; done
```

Ok when each `bxw` ends `STATE done label=boot-base rc=0` and the log tail holds `BOOTSTRAP_OK stage=base`. The box
also has `$W/.cuda_ok` now (`bx "$A" 'ls -l "$W/.cuda_ok"'`); the model stage refuses without it. If a GPU fails the
allocation test, delete that box (step 10 for its id) and create another. Do not stage data on it.

Data, code and the Hebrew lane (the lane holds `agentic_env.py`, which self-distillation needs). The pushes are
bandwidth-capped (40000 KB/s) and run one after the other:

```bash
for b in "$A" "$B"; do bash "$BOX/push_data.sh" "$b" --hebrew-lane "$LANE_SRC"; done
```

Ok when each ends `done (...)` and prints a `du -sh`. The data push leaves out the rig's `sd/` folder on purpose: it holds
the live lane's replay, and a fresh box must not find a replay this run never generated. The replay reaches a box only
from step 4, so `replay=none` in the first acceptance is true. Then the full bootstrap (venvs, model from the pinned HF revision,
the watchdog unit; it installs `negeig27-train.service` but does not enable it):

```bash
for b in "$A" "$B"; do bxd "$b" boot-all 'bash "$E27/box/bootstrap.sh" --stage all'; done
for b in "$A" "$B"; do bxw "$b" boot-all 10800 30; done
```

Ok when each log ends `BOOTSTRAP_OK stage=all`. The 54 GB download is the long part. `bxw` returning 124 means it is
still running: call the same `bxw` again. 125 means the box did not answer five polls (preempted or network); see
Appendix A.

General-eval setup (needs network, touches no GPU; do it before `accept.sh` so the throughput number is not taken
against a busy CPU):

```bash
for b in "$A" "$B"; do bxd "$b" prep-evals 'bash "$E27/evals/prepare_box.sh" all'; done
for b in "$A" "$B"; do bxw "$b" prep-evals 7200 30; done
for b in "$A" "$B"; do bx "$b" 'bash "$E27/evals/prepare_box.sh" check'; done
```

Ok when `check` exits 0 on both (it lists every gap otherwise).

Acceptance (the real trainer, NCCL NVLS A/B, SGLang plugin checks, the grad_ckpt probe). No replay exists yet, so the
grad_ckpt decision stays undecided until step 4:

```bash
for b in "$A" "$B"; do bxd "$b" accept 'bash "$E27/box/accept.sh" --arm wide'; done
for b in "$A" "$B"; do bxw "$b" accept 21600 60; done
for b in "$A" "$B"; do bxpull "$b" accept "$RECV/$b/accept"; done
```

Ok when each log ends with a line starting `ACCEPT_OK <time> host=... gpus=8 ... nvls=<0|1> replay=none grad_ckpt=undecided
smoke_tok_per_s=... per_gpu=... peak_gib=... speedup_8_vs_4=...`. A line `ACCEPT_FAILED <step>` names the step to fix; rerun
with `accept.sh --arm wide --from <step>` (a step that passed on this boot is skipped). `nvls=` is the value that
`run_pair.sh` reads from `$W/nccl_nvls.env`; do not set it by hand.

Keep the `ACCEPT_OK` lines of both boxes. Step 6 reads `per_gpu` and `speedup_8_vs_4` from them.

## 4. Self-distillation (BOX), then the replay onto both boxes

Box A generates the chat replay, box B the tool replay. Both serve the UNTOUCHED model. The boxes cannot reach each
other, so the rig moves the files.

Serve (`serve` returns when /health answers and one chat request is answered; the server itself keeps running):

```bash
for b in "$A" "$B"; do bxd "$b" sd-serve 'bash "$E27/box/sd_box.sh" serve --port 30000 --dp 8'; done
for b in "$A" "$B"; do bxw "$b" sd-serve 3600 20; done
```

Ok when each log has `SD_SERVE_OK port=30000 dp=8 pid=<pid>`. `SD_SERVE_FAILED probe: ...` leaves a receipt in
`$ACC/sd_serve_probe.json`.

Smoke first, into a separate output directory, so a bad generator wastes minutes and not hours:

```bash
bxd "$A" sd-smoke 'SD_OUT="$W/selfdistill-smoke" SD_FG=1 bash "$E27/box/sd_box.sh" chat --limit 50 --n-translate 10'
bxd "$B" sd-smoke 'SD_OUT="$W/selfdistill-smoke" SD_FG=1 bash "$E27/box/sd_box.sh" tools --item-limit-per-kind 2 --n-samples 2'
bxw "$A" sd-smoke 3600 20; bxw "$B" sd-smoke 3600 20
for b in "$A" "$B"; do bx "$b" 'SD_OUT="$W/selfdistill-smoke" bash "$E27/box/sd_box.sh" finalize'; done
```

Ok when the logs end `SD_CHAT_EXIT 0` and `SD_TOOLS_EXIT 0` and finalize prints `SD_FINALIZE_OK chat_sft=N tool_sft=M
out=...selfdistill-smoke` on each box (each box reports its own half; the other half is 0). Read a few rows:
`bx "$A" 'head -c 2000 "$W/selfdistill-smoke/spot_read.jsonl"'`. Then delete the smoke output:
`for b in "$A" "$B"; do bx "$b" 'rm -rf "$W/selfdistill-smoke"'; done`.

Full run. `SD_FG=1` keeps the generator in the foreground of the `bxd` job, so `bxw` sees its exit status:

```bash
bxd "$A" sd-chat  'SD_FG=1 bash "$E27/box/sd_box.sh" chat'
bxd "$B" sd-tools 'SD_FG=1 bash "$E27/box/sd_box.sh" tools'
bxw "$A" sd-chat  43200 120
bxw "$B" sd-tools 43200 120
```

Ok when the logs end `SD_CHAT_EXIT 0` and `SD_TOOLS_EXIT 0`. Exit 2 means the generator aborted (the server went away or a
request failed every retry): check `bx "$A" 'bash "$E27/box/sd_box.sh" status'`, restart the server with the `serve`
line above if it is gone, and run the SAME `bxd` line again. Every ledger resumes. Progress any time:
`bx "$A" 'bash "$E27/box/sd_box.sh" status'`.

Finalize on each box, then move the files with the rig (it pulls both receipts sets, checks them, and pushes both replay
files to `$W/data/sd/` on every box with a sha256 check):

```bash
for b in "$A" "$B"; do bx "$b" 'bash "$E27/box/sd_box.sh" finalize'; done
bash "$BOX/sd_sync.sh" --chat-box "$A" --tool-box "$B" --dest "$RECV/selfdistill"
```

Ok when `finalize` prints `SD_FINALIZE_OK chat_sft=N tool_sft=M` on each box (N on A, M on B, the other half 0), and
`sd_sync.sh` ends `SD_SYNC_OK chat_rows=N tool_rows=M boxes=2` and wrote `$RECV/selfdistill/sd_sync.json`. For a
third and fourth box (four-box layout) add `--also "$C" --also "$D"` to this same line once those boxes are prepared.

Read the sample before any training. This is a required step, not a courtesy:

```bash
shuf -n 25 "$RECV/selfdistill/chat/spot_read.jsonl" | cut -c1-600
shuf -n 25 "$RECV/selfdistill/tools/spot_read.jsonl" | cut -c1-600
cat "$RECV/selfdistill/chat/filter_report.json" "$RECV/selfdistill/tools/filter_report.json"
```

Ok when the rows look like the untouched model's own think-off answers (no chat-template markers, no truncated tool
calls, no empty answers) and the filter reports show the drop counts you expect. Anything odd stops the run: report it.

Length check (rows over `--max_len 12288` are dropped by the trainer; the check counts them):

```bash
for b in "$A" "$B"; do bx "$b" 'bash "$E27/box/sd_box.sh" lengths'; done
```

Ok when each prints `SD_LENGTHS_OK tool_rows_over_max_len=.. tool_usable=../.. tool_train=.. chat_usable=../..`
(receipt `$ACC/sd_lengths.json`). `SD_LENGTHS_FAIL` prints a `PROBLEM:` line for each cause.

Free the GPUs, then re-run the smoke part of acceptance with the replay present. This is the real grad_ckpt decision:

```bash
for b in "$A" "$B"; do bx "$b" 'bash "$E27/box/sd_box.sh" stop'; done
for b in "$A" "$B"; do bxd "$b" accept2 'REQUIRE_REPLAY=1 bash "$E27/box/accept.sh" --arm wide --from train_smoke'; done
for b in "$A" "$B"; do bxw "$b" accept2 21600 60; done
for b in "$A" "$B"; do bxpull "$b" accept "$RECV/$b/accept"; done
```

Ok when `stop` prints `SD_STOP_OK every GPU freed` and the new `ACCEPT_OK` line says `replay=present` with
`grad_ckpt=on` or `grad_ckpt=off` (not `undecided`). `$ACC/grad_ckpt.env` now holds `GRAD_CKPT_FLAGS`, which `arm` reads.
The H200 row for reference: 60.6 GiB peak with checkpointing, and `--grad_ckpt 0` ran out of memory, which IS the decision (on).

The live lane's finished replay already exists on the rig, in `sd/` under the prepared data set. Using it instead of
regenerating is an owner decision. If the owner takes it, run `bash "$BOX/push_data.sh" "$b" --with-sd` for each box (the
only call that pushes `sd/`), then compare the sha256 of both files on the box against that lane's `sd_sync.json`.

## 5. Parity (BOX, one box of the pair)

The SGLang plugin gates G0 to G3 (`sglang/README.md`). They need all GPUs free and the train unit inactive, so run them
before arming. One box is enough when both boxes are the same type:

```bash
bxd "$A" parity 'bash "$E27/box/parity.sh" --mode both'
bxw "$A" parity 14400 60
```

Ok when the log ends `PARITY_OK` (exit 0; marker `$ACC/parity/PARITY_OK`). Other results:

- `PARITY_FAILED` (1): a gate failed. Read the first differing layer. Do not loosen a threshold.
- `PARITY_INCONCLUSIVE` (2): a noise check failed. For G2, rerun with `--target-t 0.15`.
- `PARITY_BROKEN step=<s>` (3): a setup step broke (plugin, HF reference, server). Fix and rerun.
- `PARITY_WAIVED` (0) is not a pass of the waived gate.

Pull the receipts: `bxpull "$A" accept/parity "$RECV/$A/parity"`.

## 6. Throughput decision (RIG, from the two ACCEPT_OK lines)

`speedup_8_vs_4` is 8 GPUs on one arm against 4 GPUs on one arm:

| speedup_8_vs_4 | layout |
|---|---|
| at least 1.7 | four boxes, one arm each (`ARM_ENV='ARMS=wide GPUS_PER_RUN=8'` and `'ARMS=ctrl ...'`) |
| 1.5 to 1.7 | owner decides |
| under 1.5 | two boxes, a pair each (default) |

`per_gpu` (tokens per second per GPU) on B300, planning figures: at least 3.8k approves the envelope ($946
preemptible), 1.9k to 3.8k runs the core only ($470), under 1.9k goes to the owner with the row. On H200 the owner decides
from the first measured row. Say in the report which row you read.

Four-box layout: repeat steps 2, 3 and the second `accept.sh` for two more boxes (`negeig27-r2-c`, `-d`), include them
in `sd_sync.sh --also`, and arm the sweep as four single-arm runs (step 7).

## 7. Arm the LR sweep (RIG arms, BOX trains)

Box A trains both arms at 1e-4, box B both arms at 1e-3, seed 0, the Stage A recipe (the trainer defaults), 700 steps,
with the grad_ckpt flag accept decided:

```bash
arm "$A" sweep-lr1e-4 0 1e-4
arm "$B" sweep-lr1e-3 0 1e-3
```

Ok when `bx "$A" 'bash "$E27/box/ctl.sh" status'` prints `state: running` (or `no_status_yet` for the first minute) with
the arming line `armed: ...train.cmd, checkpoints ...`. In two or three minutes `ctl.sh logs 30` shows both arms
past `config`. If `run_pair.sh` refuses to start, its message names the cause (a missing replay in Stage A, a missing
`$W/nccl_nvls.env`, a missing `--lr`). Do not bypass it with `ALLOW_NO_REPLAY` or `ALLOW_NO_NVLS` for a real run.

```bash
bx "$A" 'bash "$E27/box/ctl.sh" status'     # exit 0 running|done|held, 1 failed, 2 not armed
bx "$A" 'bash "$E27/box/ctl.sh" logs 40'
```

`ctl.sh status` also lists `arm_failed:` lines. `run_pair.sh` retries an arm itself (8 launches, 2 with no progress), so an
`arm_failed` line means that arm is dead and the other keeps running: read the reason in it and in
`$RUNS/<tag>/<arm>_s0/launch.log`.

Four-box layout, the same sweep as four single-arm runs (`ctl.sh` keeps one unit per box):

```bash
ARM_ENV='ARMS=wide GPUS_PER_RUN=8' arm "$A" sweep-lr1e-4 0 1e-4
ARM_ENV='ARMS=ctrl GPUS_PER_RUN=8' arm "$B" sweep-lr1e-4 0 1e-4
ARM_ENV='ARMS=wide GPUS_PER_RUN=8' arm "$C" sweep-lr1e-3 0 1e-3
ARM_ENV='ARMS=ctrl GPUS_PER_RUN=8' arm "$D" sweep-lr1e-3 0 1e-3
```

## 8. The step-300 decision (RIG)

The evals at steps 250 and 300 decide. Pull log files only (the layout the reader wants is
`<tag>/{wide,ctrl}_s0/log.jsonl`), then read:

```bash
# runbook-block: sweep
S=$RECV/sweep
pull_logs() { bxpull "$1" "runs/$2" "$S/$2" --include='*/' --include='log.jsonl' --include='complete.json' \
  --include='arm_failed' --include='launch.log' --exclude='*'; }
pull_logs "$A" sweep-lr1e-4; pull_logs "$B" sweep-lr1e-3
python3 "$BOX/sweep_read.py" --runs "$S" --tag sweep-lr1e-3 --tag sweep-lr1e-4 --json "$S/sweep_read.json"
```

In the four-box layout pull each tag from both of its boxes into the same `$S/<tag>` (the arms sit in different
directories, so the pulls merge).

Ok when the last line is `SWEEP_READ choice=<tag|extend|owner> steps=250,300 abort=<yes|no> <tag>:safety=..,signal=..`.
Exit 1 means an eval at step 250 or 300 is missing: the runs have not got there yet.

Watch for the abort rule every 100 steps or so with the same reader on one tag and an earlier step:
`python3 "$BOX/sweep_read.py" --runs "$S" --tag sweep-lr1e-3 --steps 100,150`. It reports `abort` per arm.

The pre-registered rule (applied by the reader, in `results/negeig-27b/stageA-log.md`):

- Safety, per arm, average of the evals at 250 and 300: `kl_to_untouched` at most 0.08, replay NLL at most 1.9694 plus 0.05,
  tool token agreement at least 0.95, no training-loss spike above 3x the running median.
- Signal, wide against ctrl: the `s1:len256:le256` gap at least 5 points, or wide NLL at least 10% below ctrl with the gap
  still widening, or gate rate on S1 at least 1.5x the text rate.
- Choice: 1e-3 if safe with signal; 1e-4 if 1e-3 fails safety; both safe and no signal: extend; anything else: owner.

Apply the choice. The winner keeps running to 700 as seed 0 and is not touched. The loser stops and is archived, and its
box starts seed 1 at the winning LR from step 0:

```bash
# winner = sweep-lr1e-3 (box B), loser box A:
bx "$A" 'bash "$E27/box/ctl.sh" hold "sweep loser"'
bxpull "$A" runs/sweep-lr1e-4 "$RECV/archive/sweep-lr1e-4" --include='*/' --include='trainable_*.pt' --include='log.jsonl' --exclude='*'
bx "$A" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'      # every GPU near 0 MiB before the next arm
arm "$A" stageA-s1 1 1e-3
# winner = sweep-lr1e-4 (box A): hold B the same way (archive runs/sweep-lr1e-3 from B), then   arm "$B" stageA-s1 1 1e-4
```

Ok when the new box's `ctl.sh status` is running and its log shows `stageA-s1`.

- `extend`: do nothing now. Both pairs keep running (the runs go to 700). Read again when step 600 exists:
  `sweep_read.py ... --steps 550,600`. If it still says `extend`, take `sweep-lr1e-3` when its tag shows `safety=pass` on the
  `SWEEP_READ` line, else `sweep-lr1e-4`, and apply as above.
- `owner`: print the reader's `choice:` reason line to the owner, let both pairs keep running, and wait.

Abort rule (any arm, the reader prints `ABORT_RULE_HIT` and the first breach step): `kl_to_untouched` at least 0.15 on two
consecutive evals, tool agreement under 0.90, or a recall loss over 10 points. Act within about 100 steps. Stop both arms of that
box at the same step, move aside every checkpoint newer than step T minus 50 (resume finds `ckpt_*.pt` by glob, newest
first, so a renamed file is invisible to it), and re-arm at half the LR with `--lr_override`, which applies the new
`--lr` and `--w_lr` after loading.

The repair has a deadline. The trainer keeps the newest four checkpoints, one every 50 steps, so the checkpoint at T minus 50
is deleted by the save at step T plus 150. If a run is held after that, every checkpoint left is a post-breach one: moving them
all aside leaves none, and `--resume` with no checkpoint silently starts from step 0 (it does not fail). So the box command
below checks every arm directory FIRST and moves nothing unless each one still holds a checkpoint at or below T minus 50; it
prints `ABORT_REPAIR_REFUSED <dir> ...` and exits 3 otherwise, and the `&&` keeps `arm` from running. Read the breach step
when the reader prints it, not at the next polling round: the run must be held by step T plus 100.

```bash
# runbook-block: abort
T=300; TAGX=sweep-lr1e-3                                 # the breach step and the tag on that box
bx "$B" 'bash "$E27/box/ctl.sh" hold "abort rule"'
bx "$B" "T=$T; TAGX=$TAGX; "'
nd=0
for d in "$RUNS/$TAGX"/*_s0; do
  [ -d "$d" ] || continue
  nd=$((nd + 1)); clean=""
  for f in "$d"/ckpt_*.pt; do
    [ -e "$f" ] || continue
    n=${f##*/ckpt_}; n=${n%.pt}
    [ $((10#$n)) -le $((T - 50)) ] && clean=$n
  done
  [ -n "$clean" ] || { echo "ABORT_REPAIR_REFUSED $d holds no checkpoint at or below step $((T - 50)); newest: $(ls "$d" | grep "^ckpt_.*[.]pt$" | tail -n1)"; exit 3; }
done
[ "$nd" -gt 0 ] || { echo "ABORT_REPAIR_REFUSED no *_s0 directory under $RUNS/$TAGX"; exit 3; }
for d in "$RUNS/$TAGX"/*_s0; do
  for f in "$d"/ckpt_*.pt; do [ -e "$f" ] || continue; n=${f##*/ckpt_}; n=${n%.pt}; [ $((10#$n)) -gt $((T - 50)) ] && mv "$f" "$f.aborted"; done
done
ls "$RUNS/$TAGX"/*_s0/
' && ARM_FLAGS='--lr_override' arm "$B" "$TAGX" 0 5e-4
```

Ok when `ls` shows the newer checkpoints as `.pt.aborted` and the new log lines show the run resuming at a step at or
below T minus 50. The step-T records in `log.jsonl` are overwritten by the rewound run; the reader takes the last record
of each step. On `ABORT_REPAIR_REFUSED` nothing was moved and the unit stays held: do not `arm` or `ctl.sh release` by hand
(either resumes from a post-breach checkpoint). Give the owner the refusal line; the choices are a restart of that tag from
step 0 at the half LR (the steps up to T are the cost) or taking the loss of that tag. This procedure is derived from
`train27.py` and has not been run on a box.

## 9. Stage A to 700 and the general evals

Both boxes now train to 700 steps. Poll with `ctl.sh status` and pull the logs every few hours for the reader. A box that
shows `state: done` is finished.

When a box is done (its two arms hold `complete.json`; the final adapters are `trainable_000700.pt`), run on it, one box
at a time per box, with `TAG` and `SEED` that box trained (`sweep-lr1e-3` and 0 for the winner, `stageA-s1` and 1 for the
other):

```bash
# runbook-block: evals
# eval27 BOX NAME ARM TRAINABLE: the full S1 eval (every eval session) of one adapter, or of the untouched model
# (--arm ctrl --trainable none). Needs all 8 GPUs free: run it after that box's training unit is done.
eval27() {
  local cmd
  cmd="cd \"\$E27\" && \"\$PYBIN/torchrun\" --standalone --nproc_per_node 8 eval27.py --model \"\$MODEL_DIR\" --arm $3 --trainable \"$4\" --s1 \"\$DATA/s1\" --out \"\$W/eval27/$2\""
  bxd "$1" "eval27-$2" "$cmd" && bxw "$1" "eval27-$2" 7200 60
}
# evalarm BOX NAME KIND [TRAINABLE]: serve one arm, run the five general evals, stop the server. Resumable: rerun on rc 3.
evalarm() {
  local box=$1 name=$2 kind=$3 tr=${4:-} up down rc
  up="cd \"\$E27/evals\" && bash serve_arm.sh up --arm $name --kind $kind"
  down="cd \"\$E27/evals\" && bash serve_arm.sh down"
  if [ -n "$tr" ]; then up="$up --trainable \"$tr\""; down="$down --purge"; fi
  bxd "$box" "up-$name" "$up" && bxw "$box" "up-$name" 4200 30 || return $?
  bxd "$box" "ev-$name" "cd \"\$E27/evals\" && bash run_general.sh --arm $name" && bxw "$box" "ev-$name" 28800 60
  rc=$?
  bxd "$box" "down-$name" "$down" && bxw "$box" "down-$name" 900 15
  return $rc
}
```

Use them per box (names `base`, `wide_s$SEED`, `ctrl_s$SEED`; the untouched model is the base for THAT box's compare):

```bash
# runbook-block: evalcalls
BOXX=$B; TAG=sweep-lr1e-3; SEED=0                      # the box, its tag, its seed
W_TR="\$RUNS/$TAG/wide_s$SEED/trainable_000700.pt"      # \$RUNS is expanded on the box
C_TR="\$RUNS/$TAG/ctrl_s$SEED/trainable_000700.pt"
eval27 "$BOXX" wide_s$SEED wide "$W_TR"
eval27 "$BOXX" ctrl_s$SEED ctrl "$C_TR"
eval27 "$BOXX" untouched ctrl none
evalarm "$BOXX" base base
evalarm "$BOXX" wide_s$SEED wide "$W_TR"
evalarm "$BOXX" ctrl_s$SEED ctrl "$C_TR"
```

Ok when each `eval27-*` ends with its summary and each `ev-*` ends rc 0 (`run_general.sh` exit 0 all complete, 3 some
incomplete: run the same `evalarm` again, 4 the server went away: `serve_arm.sh up --replace` and rerun, any other code an
eval failed for good, see `logs/<eval>.log` under its arm directory). One server runs at a
time; `serve_arm.sh up` takes up to 45 minutes. Optional noise floor on one box: `evalarm "$BOXX" base_repeat base`.

Compare on the box (needs `$W/evalvenv/bin/python`), then pull the evals:

```bash
bx "$BOXX" 'cd "$E27/evals" && for a in wide_s'$SEED' ctrl_s'$SEED'; do "$W/evalvenv/bin/python" compare_general.py --base "$W/eval/base" --arm "$W/eval/$a" --out-json "$W/eval/$a.vs_base.json" --out-md "$W/eval/$a.vs_base.md"; done'
bxpull "$BOXX" eval "$RECV/$BOXX/eval"; bxpull "$BOXX" eval27 "$RECV/$BOXX/eval27"
```

Add `--base-repeat "$W/eval/base_repeat"` when the noise floor was run. Ok when each report has `"valid": true` (`false` means
INVALID, read `protocol_problems`) and each gate verdict is `PASS`, `FAIL`, `PASS_UNRESOLVED` or `FAIL_UNRESOLVED`.
UNRESOLVED means the sample is too small to decide at that band; it is not a pass. The bands, in points: BFCL v4 3,
MMLU-Pro 1, HumanEval 3, IFEval 2, Global-MMLU he 1.

Stage B go (owner's call, from these numbers): wide beats ctrl by at least 10 points on `s1:len512:gt256`, the mean over
two seeds, and every general gate is in its band. Read the gap per seed with the reader on the pulled logs:
`python3 "$BOX/sweep_read.py" --runs "$S" --tag <tag> --seed <n> --steps 700`, take the `beside: s1:len512:gt256 gap` line of
each seed and average the two. Extension rungs (1,200 and then 1,800 steps) need an owner go each; a null result is
declared only after 1,800.

## 10. Receipts and teardown (RIG)

Pull everything while the boxes are alive (`teardown.sh --pull` copies `runs`, `accept`, `logs` and `run`, but not `eval`
or `eval27`, and never the merged models):

```bash
for b in "$A" "$B"; do bxpull "$b" runs "$RECV/$b/runs" --exclude='ckpt_*.pt'; bxpull "$b" logs "$RECV/$b/logs"; done
```

Ok when `du -sh "$RECV"/*/` shows both boxes and `ls "$RECV/$A/runs"` shows the run directories with `log.jsonl` and
`trainable_*.pt`. `teardown.sh --pull` copies `ckpt_*.pt` again (they are under 2 GB each), so a run that will be extended keeps its
checkpoints either way.

Teardown. Dry run first (it prints the exact commands and touches nothing), always with `--ids`:

```bash
IDA=$(idof "$A"); IDB=$(idof "$B")
bash "$BOX/teardown.sh" --ids "$IDA $IDB" --pull "$RECV/pulled"
```

Ok when it lists exactly your two ids, both with label `lane=negeig-27b`. A STOPPED (preempted) box cannot be pulled:
`nebius compute instance start --id <id>` first, or add `--skip-unreachable`. Then:

```bash
bash "$BOX/teardown.sh" --ids "$IDA $IDB" --pull "$RECV/pulled" --yes
```

Ok when the last line is `TEARDOWN_CONFIRMED (subset)`. `TEARDOWN_INCOMPLETE` (exit 1) lists what the provider still
shows; rerun the same line. If the host refuses the `--yes` call, hand the owner that exact line. Confirm from the provider
afterwards: `bash "$BOX/teardown.sh" --list` shows no row for your ids and no `negeig27-r2-*` disk.

Stop the watcher only if you started it: `kill "$(cat "$STATE/watch.pid")"; rm -f "$STATE/watch.pid"`. Then
delete `$RECV/pulled` copies you do not need and any scratch you made; the receipts that matter move into
`results/negeig-27b/` in a normal PR (self-review comment, revuto, green CI).

## Appendix A. Preemption and a lost box

A preemptible box that is preempted is STOPPED, not deleted. Its disks stay and its public IP changes.

- `bxw` returns 125 (no answer five polls in a row) or `ctl.sh status` stops answering: look at
  `tail "$STATE/events.log"`. The watcher logs `PREEMPTED`, restarts the box and starts the train unit.
- Without a watcher: `nebius compute instance start --id "$IDA"`, then `resolve_ip "$A"` (it asks the provider for the
  current IP on every call), then `bash "$BOX/push_data.sh" "$A" --code-only`, then
  `bx "$A" 'bash "$E27/box/ctl.sh" status'`. A reboot restarts the unit by itself (it is enabled by `arm`); if the state is
  `held` or `failed`, read why and `ctl.sh release`.
- A box that reads NOT_FOUND is gone with its disks. Whatever was pulled is what exists. Create a box and start again from
  step 2; the pulled `$RECV/<box>/runs` is the only copy of its checkpoints.
- The replacement has no replay: the data push skips `sd/`, and the generators' box may be the one that is gone. The rig
  still holds the pulled copy and its receipt, so push that (no pull, no source box), then repeat the second acceptance
  (`$A` stands for the replacement's name):

```bash
bash "$BOX/sd_sync.sh" --push-only --dest "$RECV/selfdistill" --also "$A"
bxd "$A" accept2 'REQUIRE_REPLAY=1 bash "$E27/box/accept.sh" --arm wide --from train_smoke'
```

  Ok when the push ends `SD_PUSH_OK chat_rows=N tool_rows=M boxes=1`. It refuses unless `$RECV/selfdistill/sd_sync.json`
  exists and its hashes equal the files in `$RECV/selfdistill`. Until the replay is there, Stage A refuses to start
  (`run_pair.sh` stops on a missing replay), which is the intended stop. A box that was only stopped keeps its own
  `$W/data/sd/` and a later push does not touch it.
- A detached `bxd` job survives the ssh session but not a reboot: after a reboot run `bxs` on the label. `STATE dead` means
  rerun the same `bxd` line (the generators and the evals are resumable; a training run is restarted by the unit).

## Appendix B. When a line does not come

| you see | meaning | do |
|---|---|---|
| `ACCEPT_FAILED <step>` | that acceptance step failed | fix it, `accept.sh --arm wide --from <step>` |
| `ACCEPT_FAILED train_smoke` with `nvls=1` and a hang (`smoke part a hit SMOKE_TIMEOUT_S`) or an NCCL error in the arm logs | the NVLS A/B ran 8 ranks in one group; training runs two concurrent 4-rank groups, which the A/B did not exercise | `bxd "$b" accept3 'NVLS_PIN=0 bash "$E27/box/accept.sh" --arm wide --only nccl'`, then the same `--from train_smoke` line; report it |
| `BOOTSTRAP` log ends without `BOOTSTRAP_OK` | a stage failed; the `ERROR:` line says which | fix, rerun the same stage |
| `bxw` rc 124 | still running after the deadline | call `bxw` again with a new deadline |
| `bxw` rc 125 | the box did not answer five polls | Appendix A |
| `bxw` rc 126 | the job died with no exit status (reboot, kill -9) | `bxs`, then rerun the `bxd` line |
| `bxd` exit 3 | that label is still running on the box | `bxw` it, or pick another label |
| `bxd` exit 5 | the job did not start | read the ssh error; check `resolve_ip` |
| `SD_SERVE_FAILED` | the 17 x 23 probe did not answer 391 | `$ACC/sd_serve_probe.json`, the server log in `$LOGS` |
| `SD_CHAT_EXIT 2` / `SD_TOOLS_EXIT 2` | the generator aborted | restart `serve`, rerun the same line |
| `SD_LENGTHS_FAIL` | rows do not render or too many are over length | read the `PROBLEM:` lines |
| `PARITY_FAILED` | a parity gate failed | read the first differing layer |
| `evalarm` returns at once with `up-<name>` failed | `serve_arm.sh up` failed and stopped its own server; `evalarm` does not run `down` then | `bxs "$BOXX" up-<name> 40`, fix, rerun the same `evalarm` line; a merged model left behind goes with `serve_arm.sh down --purge` |
| `ctl.sh status` exit 1, `arm_failed:` | an arm gave up | its `launch.log`; `ctl.sh release` after the fix |
| `SWEEP_READ choice=owner` | the rule does not cover the result | report the reason, wait |
| `TEARDOWN_INCOMPLETE` | the provider still lists something | rerun the same line |

## Appendix C. Who does what

| step | RIG | BOX |
|---|---|---|
| create, ids, watcher | `create_box.sh`, `teardown.sh --list`, `preempt_watch.sh` | |
| prepare | `push_data.sh`, `bxd`/`bxw` | `bootstrap.sh`, `prepare_box.sh`, `accept.sh` |
| self-distillation | `sd_sync.sh`, the spot-read | `sd_box.sh serve/chat/tools/finalize/lengths/stop` |
| parity | `bxd`/`bxw` | `parity.sh` |
| sweep and Stage A | `arm`, `bxpull`, `sweep_read.py` | `ctl.sh`, `run_pair.sh`, `train27.py` |
| evals | `evalarm`, `eval27`, `bxpull` | `eval27.py`, `serve_arm.sh`, `run_general.sh`, `compare_general.py` |
| teardown | `teardown.sh --ids ... --yes` (owner runs it if the host refuses) | |

## What only a real box proves

Real tokens per second and peak memory (and whether `--grad_ckpt 0` fits), the NVLS A/B on this fabric, the SGLang
self-distillation flags under DP 8, real tokenizer lengths, the systemd path (the tests run `NO_SYSTEMD=1`), the
`negeig: active` engagement line of the real plugin, the parity numbers, and the abort repair. Hangs are not detected by
the per-arm retry (only exits are); watch `ctl.sh logs` for a stalled step counter.
