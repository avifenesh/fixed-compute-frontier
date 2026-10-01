#!/usr/bin/env bash
# CPU test of run_pair.sh under the REAL resume_wrapper.sh and ctl.sh (NO_SYSTEMD, SKIP_GPU_CHECK), stub trainer.
# Covers the crash accounting contract: the wrapper keys on the checkpoint signature, run_pair.sh retries per arm.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
export NO_SYSTEMD=1 SKIP_GPU_CHECK=1
CTL=$BOX_DIR/ctl.sh; WRAP=$BOX_DIR/resume_wrapper.sh

arm() { # tag seed maxcrashes [run_pair flags...]  (arms the wrapper with a run_pair.sh command, no start)
  local tag=$1 seed=$2 mc=$3; shift 3
  "$CTL" arm --cmd "exec bash $E27_DIR/run_pair.sh $tag $seed $*" --ckpt-dir "$B/runs/$tag" --max-crashes "$mc" --backoff 0 \
    --log "$B/train.log" --no-start >"$B/ctl.out" 2>&1
}
run_wrapper() { timeout 120 bash "$WRAP" >"$B/wrap.out" 2>&1; WRC=$?; }
rundir() { echo "$B/run"; }

section "W1 one arm fails twice without progress, the wrapper restarts and the arm recovers"
mk_box w1
plan wide fail fail ok
arm tagA 0 3 --lr 1e-4 --steps 100
eq "armed" "$?" 0
run_wrapper
eq "wrapper exit 0" "$WRC" 0
check "done marker" test -f "$B/run/done"
check_not "no failed marker" test -e "$B/run/failed"
eq "wide launches: 2 without progress, then the good one" "$(launches wide)" 3
eq "ctrl ran once" "$(launches ctrl)" 1
check "both arms complete" test -f "$B/runs/tagA/wide_s0/complete.json" -a -f "$B/runs/tagA/ctrl_s0/complete.json"
check_not "crash counter cleared on done" test -e "$B/run/crash.count"

section "W2 an arm that can never start: bounded, then failed"
mk_box w2
plan wide 'fail*'
arm tagA 0 3 --lr 1e-4 --steps 100
run_wrapper
eq "wrapper exit 0 (systemd must not loop it)" "$WRC" 0
check "failed marker" test -f "$B/run/failed"
check_not "no done marker" test -e "$B/run/done"
eq "wide launches: ARM_MAX_NOPROGRESS(2) x MAX_CRASHES(3)" "$(launches wide)" 6
eq "ctrl ran once, its complete.json made later invocations skip it" "$(launches ctrl)" 1
has "failed file names the crash count" "$B/run/failed" "3 crashes at the same checkpoint"
"$CTL" status >"$B/status.out" 2>&1; SRC=$?
eq "ctl status exit 1 when failed" "$SRC" 1
has "ctl status shows the failed arm marker" "$B/status.out" "arm_failed"

section "W3 progress with --keep_ckpts 4 keeps resetting the counter (signature rolls with the files)"
mk_box w3
plan wide ckfail:50 ckfail:100 ckfail:150 ckfail:200 ckfail:250 ckfail:300 ok
export ARM_MAX_ATTEMPTS=1
arm tagA 0 2 --lr 1e-4 --steps 400 --save_every 50 --keep_ckpts 4
run_wrapper
unset ARM_MAX_ATTEMPTS
eq "wrapper exit 0" "$WRC" 0
check "done, never failed (rolling checkpoint names count as progress)" test -f "$B/run/done" -a ! -e "$B/run/failed"
eq "wide launched 7 times" "$(launches wide)" 7
eq "wide keeps 4 checkpoints" "$(find "$B/runs/tagA/wide_s0" -name 'ckpt_*.pt' | wc -l | tr -d ' ')" 4

section "W4 SIGTERM to the wrapper stops the whole group, no retry, not a crash"
mk_box w4
plan wide 'sleep*'; plan ctrl 'sleep*'
arm tagA 0 3 --lr 1e-4 --steps 100
bash "$WRAP" >"$B/wrap.out" 2>&1 &
wp=$!
for _ in $(seq 1 75); do [ "$(wc -l <"$STUB_LOG" | tr -d ' ')" -ge 2 ] && break; sleep 0.2; done
eq "both arms up under the wrapper" "$(wc -l <"$STUB_LOG" | tr -d ' ')" 2
kill -TERM "$wp"
for _ in $(seq 1 100); do kill -0 "$wp" 2>/dev/null || break; sleep 0.2; done
check_not "wrapper exited" kill -0 "$wp"
kill -KILL "$wp" 2>/dev/null
wait "$wp" 2>/dev/null; WRC=$?
eq "wrapper exit 0" "$WRC" 0
has "status stopped" "$B/run/status.json" '"state":"stopped"'
check_not "no crash counted" test -e "$B/run/crash.count"
check_not "no stub left running" pgrep -f "^python3 .*stub_torchrun.py .*--out $B/runs"
eq "no relaunch: wide" "$(launches wide)" 1
finish
