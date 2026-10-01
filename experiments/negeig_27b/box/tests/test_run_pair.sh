#!/usr/bin/env bash
# CPU tests of run_pair.sh with the stub trainer: replay and LR/NVLS rules, launch shape, retry, SIGTERM, skip.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RP=$E27_DIR/run_pair.sh

run() { "$RP" "$@" >"$B/out.txt" 2>&1; RC=$?; }

section "T1 pair, Stage A defaults"
mk_box t1
run tagA 0 --lr 1e-4 --steps 700
eq "exit 0" "$RC" 0
eq "wide launched once" "$(launches wide)" 1
eq "ctrl launched once" "$(launches ctrl)" 1
eq "wide on GPUs 0-3" "$(stub_field wide 0 cuda)" "0,1,2,3"
eq "ctrl on GPUs 4-7" "$(stub_field ctrl 0 cuda)" "4,5,6,7"
eq "4 ranks a run" "$(stub_field wide 0 nproc)" 4
eq "distinct ports" "$(stub_field wide 0 port) $(stub_field ctrl 0 port)" "29501 29502"
eq "NVLS value from the file reaches the trainer" "$(stub_field wide 0 nvls)" 0
has "chat replay from the canonical dir" "$(stub_field wide 0 argv)" "--chat_replay $B/data/sd/chat_sft.jsonl"
has "tool replay from the canonical dir" "$(stub_field wide 0 argv)" "--tool_replay $B/data/sd/tool_sft.jsonl"
has "--w_lr follows --lr" "$(stub_field wide 0 argv)" "--lr 1e-4 --steps 700 --w_lr 1e-4"
has "--resume always passed" "$(stub_field ctrl 0 argv)" "--resume"
has "arm name and out dir" "$(stub_field ctrl 0 argv)" "--arm ctrl .*--out $B/runs/tagA/ctrl_s0"
check "wide complete.json" test -f "$B/runs/tagA/wide_s0/complete.json"
check "ctrl complete.json" test -f "$B/runs/tagA/ctrl_s0/complete.json"
eq "same seed both arms" "$(stub_field wide 0 argv | grep -o -- '--seed [0-9]*')" "$(stub_field ctrl 0 argv | grep -o -- '--seed [0-9]*')"

section "T2 missing replay"
mk_box t2
rm "$B/data/sd/tool_sft.jsonl"
run tagA 0 --lr 1e-4
eq "Stage A refuses, exit 2" "$RC" 2
has "names the missing file" "$B/out.txt" "tool:$B/data/sd/tool_sft.jsonl"
eq "no trainer started" "$(wc -l <"$STUB_LOG" | tr -d ' ')" 0
ALLOW_NO_REPLAY=1 run tagA 0 --lr 1e-4
eq "ALLOW_NO_REPLAY=1 runs" "$RC" 0
has "warns that the mix is missing" "$B/out.txt" "WARNING no replay for tool:"
hasnt "no --tool_replay passed" "$(stub_field wide 0 argv)" "--tool_replay"
has "chat replay still passed" "$(stub_field wide 0 argv)" "--chat_replay"
mk_box t2b
: >"$B/data/sd/chat_sft.jsonl"
run tagA 0 --lr 1e-4
eq "an empty replay file counts as missing" "$RC" 2
mk_box t2c
rm "$B/data/sd/chat_sft.jsonl" "$B/data/sd/tool_sft.jsonl"
STAGE=probe run tagP 0 --steps 5
eq "STAGE=probe runs without replay and without --lr" "$RC" 0

section "T3 replay env overrides"
mk_box t3
echo x >"$B/c.jsonl"; echo x >"$B/t.jsonl"
CHAT_REPLAY=$B/c.jsonl TOOL_REPLAY=$B/t.jsonl run tagA 0 --lr 1e-4
has "CHAT_REPLAY wins" "$(stub_field wide 0 argv)" "--chat_replay $B/c.jsonl"
has "TOOL_REPLAY wins" "$(stub_field wide 0 argv)" "--tool_replay $B/t.jsonl"
mk_box t3b
mkdir -p "$B/alt"; echo x >"$B/alt/chat_sft.jsonl"; echo x >"$B/alt/tool_sft.jsonl"
SD=$B/alt run tagA 0 --lr 1e-4
has "SD dir override moves both defaults" "$(stub_field wide 0 argv)" "--chat_replay $B/alt/chat_sft.jsonl .*--tool_replay $B/alt/tool_sft.jsonl"

section "T4 learning rate rules"
mk_box t4
run tagA 0 --steps 10
eq "Stage A without --lr refused" "$RC" 2
has "says why" "$B/out.txt" "explicit --lr"
ALLOW_DEFAULT_LR=1 run tagA 0 --steps 10
eq "ALLOW_DEFAULT_LR=1 runs" "$RC" 0
hasnt "no --w_lr invented without --lr" "$(stub_field wide 0 argv)" "--w_lr"
mk_box t4b
run tagA 0 --lr=1e-3 --steps 10
has "--lr=VALUE form feeds --w_lr" "$(stub_field wide 0 argv)" "--w_lr 1e-3"
mk_box t4c
run tagA 0 --lr 1e-3 --w_lr 5e-4 --steps 10
eq "an explicit --w_lr is kept once" "$(stub_field wide 0 argv | grep -o -- '--w_lr' | wc -l | tr -d ' ')" 1
has "and keeps its value" "$(stub_field wide 0 argv)" "--w_lr 5e-4"

section "T5 NCCL_NVLS_ENABLE file"
mk_box t5
rm "$B/nccl_nvls.env"
run tagA 0 --lr 1e-4
eq "Stage A refuses without the file" "$RC" 2
has "names accept.sh" "$B/out.txt" "accept.sh writes it"
ALLOW_NO_NVLS=1 run tagA 0 --lr 1e-4
eq "ALLOW_NO_NVLS=1 runs" "$RC" 0
eq "and leaves NVLS unset" "$(stub_field wide 0 nvls)" "None"
mk_box t5b
printf '# measured\nNCCL_NVLS_ENABLE=1\n' >"$B/nccl_nvls.env"
NCCL_NVLS_ENABLE=0 run tagA 0 --lr 1e-4
eq "the file wins over the environment" "$(stub_field wide 0 nvls)" 1
has "and says so" "$B/out.txt" "WARNING NCCL_NVLS_ENABLE=0 in the environment, .* says 1"
mk_box t5c
echo "garbage" >"$B/nccl_nvls.env"
run tagA 0 --lr 1e-4
eq "a file without the line is refused" "$RC" 2
mk_box t5d
echo "NCCL_NVLS_ENABLE=1" >"$B/custom.env"
NVLS_FILE=$B/custom.env run tagA 0 --lr 1e-4
eq "NVLS_FILE override is read" "$(stub_field wide 0 nvls)" 1

section "T6 PY and PYBIN"
mk_box t6
PY=$B/venv/bin run tagA 0 --lr 1e-4
eq "PY as a directory is refused" "$RC" 2
has "explains the rename" "$B/out.txt" "PYBIN"
mk_box t6b
PY=$B/venv/bin/python run tagA 0 --lr 1e-4
eq "PY as the python binary is accepted and unused" "$RC" 0
mk_box t6c
mkdir -p "$B/otherbin"; cp "$B/venv/bin/torchrun" "$B/otherbin/torchrun"
PYBIN=$B/otherbin run tagA 0 --lr 1e-4
eq "PYBIN override picks the torchrun" "$RC" 0
mk_box t6d
rm "$B/venv/bin/torchrun"
run tagA 0 --lr 1e-4
eq "missing torchrun refused before any launch" "$RC" 2

section "T7 one arm, 8 GPUs (4-box layout)"
mk_box t7
ARMS=wide run tagB 0 --lr 1e-3 --steps 700
eq "exit 0" "$RC" 0
eq "one launch" "$(wc -l <"$STUB_LOG" | tr -d ' ')" 1
eq "8 ranks" "$(stub_field wide 0 nproc)" 8
eq "GPUs 0-7" "$(stub_field wide 0 cuda)" "0,1,2,3,4,5,6,7"
eq "first port" "$(stub_field wide 0 port)" 29501
mk_box t7b
ARMS=ctrl run tagB 1 --lr 1e-3
eq "ctrl alone on 8 GPUs" "$(stub_field ctrl 0 cuda)" "0,1,2,3,4,5,6,7"
has "seed 1 in the dir name" "$(ls "$B/runs/tagB")" "ctrl_s1"
mk_box t7c
ARMS=wide GPUS_PER_RUN=4 run tagB 0 --lr 1e-3
eq "GPUS_PER_RUN=4 with one arm uses GPUs 0-3" "$(stub_field wide 0 cuda)" "0,1,2,3"
mk_box t7d
ARMS="wide ctrl" GPUS_PER_RUN=8 run tagB 0 --lr 1e-3
eq "two 8-GPU arms on an 8-GPU box refused" "$RC" 2
mk_box t7e
ARMS="wide;ls" run tagB 0 --lr 1e-3
eq "a bad arm name refused" "$RC" 2
mk_box t7f
EXPECT_GPUS=4 run tagB 0 --lr 1e-3
eq "EXPECT_GPUS=4 splits 2 and 2" "$(stub_field wide 0 nproc)$(stub_field ctrl 0 nproc)" 22

section "T8 retry that makes progress"
mk_box t8
plan wide ckfail:50 ckfail:100 ok
run tagA 0 --lr 1e-4 --steps 200
eq "exit 0 after two crashes with progress" "$RC" 0
eq "wide launched three times" "$(launches wide)" 3
eq "ctrl launched once" "$(launches ctrl)" 1
check "wide completed" test -f "$B/runs/tagA/wide_s0/complete.json"
check "no arm_failed marker" test ! -e "$B/runs/tagA/wide_s0/arm_failed"
has "resume step logged" "$B/runs/tagA/wide_s0/launch.log" "ckpt_step=100"
eq "ctrl was not restarted by wide's crashes" "$(grep -c 'launch attempt' "$B/runs/tagA/ctrl_s0/launch.log")" 1
mk_box t8b
plan wide ckfail:50 ckfail:100 ckfail:150 ckfail:200 ckfail:250 ckfail:300 ok
ARM_MAX_ATTEMPTS=4 run tagA 0 --lr 1e-4 --steps 400
eq "ARM_MAX_ATTEMPTS caps launches per invocation" "$(launches wide)" 4
check "arm_failed written at the cap" test -f "$B/runs/tagA/wide_s0/arm_failed"
eq "run_pair exits non-zero" "$((RC != 0))" 1
eq "ctrl still completed" "$(test -f "$B/runs/tagA/ctrl_s0/complete.json" && echo yes)" yes

section "T9 retry without progress"
mk_box t9
plan wide 'fail*'
run tagA 0 --lr 1e-4 --steps 100
eq "non-zero exit" "$((RC != 0))" 1
eq "wide launched ARM_MAX_NOPROGRESS times" "$(launches wide)" 2
check "arm_failed marker" test -f "$B/runs/tagA/wide_s0/arm_failed"
has "marker has the reason" "$B/runs/tagA/wide_s0/arm_failed" "rc 1 after 2 attempts"
check "ctrl completed beside it" test -f "$B/runs/tagA/ctrl_s0/complete.json"
has "launch.log says ARM_FAILED" "$B/runs/tagA/wide_s0/launch.log" "ARM_FAILED"
plan wide ok
run tagA 0 --lr 1e-4 --steps 100
eq "a later invocation recovers the arm" "$RC" 0
check "marker removed" test ! -e "$B/runs/tagA/wide_s0/arm_failed"
eq "ctrl was skipped, not rerun" "$(launches ctrl)" 1
mk_box t9b
plan wide fail ckfail:50 fail ok
ARM_MAX_NOPROGRESS=2 run tagA 0 --lr 1e-4 --steps 100
eq "progress between failures resets the no-progress count" "$RC" 0
eq "four wide launches" "$(launches wide)" 4

section "T10 complete arms are skipped"
mk_box t10
run tagA 0 --lr 1e-4 --steps 50
n=$(wc -l <"$STUB_LOG" | tr -d ' ')
run tagA 0 --lr 1e-4 --steps 50
eq "second run exits 0" "$RC" 0
eq "second run launches nothing" "$(wc -l <"$STUB_LOG" | tr -d ' ')" "$n"
has "says complete" "$B/runs/tagA/wide_s0/launch.log" "complete, skipping"

section "T11 SIGTERM stops every arm and is not retried"
mk_box t11
plan wide 'sleep*'; plan ctrl 'sleep*'
"$RP" tagA 0 --lr 1e-4 >"$B/out.txt" 2>&1 &
pid=$!
for _ in $(seq 1 50); do [ "$(wc -l <"$STUB_LOG" | tr -d ' ')" -ge 2 ] && break; sleep 0.2; done
eq "both arms up" "$(wc -l <"$STUB_LOG" | tr -d ' ')" 2
kill -TERM "$pid"
for _ in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
check_not "run_pair exited" kill -0 "$pid"
wait "$pid"; RC=$?
eq "exit 143" "$RC" 143
eq "no relaunch after TERM: wide" "$(launches wide)" 1
eq "no relaunch after TERM: ctrl" "$(launches ctrl)" 1
check_not "no arm_failed after a stop" test -e "$B/runs/tagA/wide_s0/arm_failed"
has "stop logged" "$B/runs/tagA/wide_s0/launch.log" "stopped on request"
check_not "no stub left running" pgrep -f "^python3 .*stub_torchrun.py .*--out $B/runs"
mk_box t11b
plan wide 'fail*'
ARM_BACKOFF_S=37 "$RP" tagA 0 --lr 1e-4 >"$B/out.txt" 2>&1 &
pid=$!
for _ in $(seq 1 50); do [ "$(launches wide)" -ge 1 ] && sleep 0.5 && break; sleep 0.2; done
kill -TERM "$pid"
for _ in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
check_not "TERM during the backoff ends run_pair within 10 s" kill -0 "$pid"
wait "$pid" 2>/dev/null
check_not "no backoff sleep left behind" pgrep -f '^sleep 37$'

section "T12 checkpoint retention with two arms in one tag dir"
mk_box t12
run tagA 0 --lr 1e-4 --steps 400 --save_every 50
eq "exit 0" "$RC" 0
eq "each arm keeps 4 checkpoints" "$(find "$B/runs/tagA/wide_s0" -name 'ckpt_*.pt' | wc -l | tr -d ' ')$(find "$B/runs/tagA/ctrl_s0" -name 'ckpt_*.pt' | wc -l | tr -d ' ')" 44

section "T13 train_pair.cmd.example: the template a ctl.sh arm --cmd-file launches"
mk_box t13
mkdir -p "$B/code/experiments" "$B/accept"
ln -s "$E27_DIR" "$B/code/experiments/negeig_27b"
CMD_T=$BOX_DIR/train_pair.cmd.example
cmdrun() { bash "$CMD_T" >"$B/out.txt" 2>&1; RC=$?; }
cmdrun
eq "no accept/grad_ckpt.env: exit 1" "$RC" 1
has "says to run accept.sh first" "$B/out.txt" "run accept.sh first"
eq "nothing launched" "$(wc -l <"$STUB_LOG" | tr -d ' ')" 0
printf 'GRAD_CKPT_DECISION=off\nGRAD_CKPT_FLAGS="--grad_ckpt 0"\n' >"$B/accept/grad_ckpt.env"
cmdrun
eq "defaults: exit 0" "$RC" 0
has "stage A lr, steps and the measured grad_ckpt reach the trainer" "$(stub_field wide 0 argv)" "--lr 1e-4 --steps 700 --grad_ckpt 0 .*--w_lr 1e-4|--lr 1e-4 --steps 700 .*--grad_ckpt 0"
has "same flags on ctrl" "$(stub_field ctrl 0 argv)" "--grad_ckpt 0"
has "default tag directory" "$(stub_field wide 0 argv)" "--out $B/runs/stageA/wide_s0"
eq "wide and ctrl launched once each" "$(launches wide)$(launches ctrl)" 11
mk_box t13b
mkdir -p "$B/code/experiments" "$B/accept"
ln -s "$E27_DIR" "$B/code/experiments/negeig_27b"
printf 'GRAD_CKPT_DECISION=on\nGRAD_CKPT_FLAGS="--grad_ckpt 1"\n' >"$B/accept/grad_ckpt.env"
TAG=lrB SEED=1 LR=1e-3 STEPS=20 cmdrun
eq "TAG SEED LR STEPS overrides: exit 0" "$RC" 0
has "lr 1e-3, 20 steps, grad_ckpt 1, seed 1" "$(stub_field wide 0 argv)" "--lr 1e-3 .*--steps 20 .*--grad_ckpt 1|--steps 20 .*--grad_ckpt 1"
has "seed 1 in the arm directory" "$(stub_field wide 0 argv)" "--out $B/runs/lrB/wide_s1"
has "--w_lr follows the swept lr" "$(stub_field wide 0 argv)" "--w_lr 1e-3"
hasnt "the template passes no MODEL, DATA, RUNS, PY or PYBIN of its own" "$(grep -v '^#' "$CMD_T")" "(^|[ ;])(MODEL|DATA|RUNS|PY|PYBIN)="
finish
