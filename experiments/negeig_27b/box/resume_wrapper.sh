#!/usr/bin/env bash
# resume_wrapper.sh - runs ON the box under negeig27-train.service (or by hand). Keeps ONE training command
# running and restarts it from its newest checkpoint.
#
# The command is $RUN_DIR/train.cmd (a bash script, written by ctl.sh arm). It is started with
#   NEGEIG_RESUME=1  RESUME_FLAG=--resume
# so `torchrun ... train27.py ... $RESUME_FLAG` resumes when a checkpoint exists and starts fresh when not,
# and a verl launcher with trainer.resume_mode=auto does the same on its own. Checkpoints are the commands' job.
#
# Crash accounting is keyed on the checkpoint signature, not on a time window: after a non-zero exit, if no new
# checkpoint appeared since the previous crash, the same-checkpoint counter goes up; at MAX_CRASHES the job is marked
# `failed` and the wrapper stops (exit 0, so systemd does not loop a job that cannot make progress). A crash
# after a new checkpoint resets the counter to 1. A preemption stops the whole VM, so it never reaches here as a
# crash; the wrapper simply starts again on boot.
# The signature is the path and size of EVERY ckpt_*.pt under the dir, hashed. A pair of arms under one tag dir
# (run_pair.sh: wide_s0/ and ctrl_s0/) counts as progress when either arm advances, and a rolled window (the trainer
# keeps the newest --keep_ckpts, default 4, so the oldest name drops as a new one lands) still changes the hash.
# run_pair.sh retries an arm by itself before it exits non-zero (ARM_MAX_NOPROGRESS launches without a new checkpoint),
# so one dead arm costs the wrapper ONE crash per run_pair.sh invocation: at most ARM_MAX_NOPROGRESS x MAX_CRASHES
# launches at one checkpoint before `failed`. A SIGTERM (stop, ctl.sh hold) is never a crash and is never retried.

# Control files in $RUN_DIR:  train.cmd  run.env  hold  done  failed  crash.count  crash.sig  status.json
# Test hooks (CPU machine): SKIP_GPU_CHECK=1, NEGEIG_W=<scratch dir>.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=boxenv.sh
. "$HERE/boxenv.sh"
mkdir -p "$RUN_DIR" "$LOGS"
# defaults, overridden by run.env
CKPT_DIR=$RUNS; CKPT_GLOB='ckpt_*.pt'; MAX_CRASHES=3; BACKOFF_S=60; LOG=$LOGS/train.log; GPU_WAIT_S=300
# shellcheck source=/dev/null
[ -f "$RUN_DIR/run.env" ] && . "$RUN_DIR/run.env"

exec 9>"$RUN_DIR/wrapper.lock"
flock -n 9 || { blog "another wrapper holds $RUN_DIR/wrapper.lock; exiting"; exit 0; }

status() { # state [extra json fields without braces]
  local st=$1; shift
  printf '{"state":"%s","updated_utc":"%s","restarts":%s,"crashes_no_progress":%s,"last_exit":%s,"ckpt":"%s"%s}\n' \
    "$st" "$(date -u +%FT%TZ)" "${RESTARTS:-0}" "${CRASHES:-0}" "${LAST_EXIT:-null}" "$(ckpt_newest)" "${1:+,$1}" >"$RUN_DIR/status.json.tmp"
  mv "$RUN_DIR/status.json.tmp" "$RUN_DIR/status.json"
}
ckpt_newest() { # newest checkpoint name, or empty
  [ -d "$CKPT_DIR" ] || return 0
  find "$CKPT_DIR" -maxdepth 2 -name "$CKPT_GLOB" -not -name '.*' -printf '%f\n' 2>/dev/null | sort -V | tail -n1
}
ckpt_sig() { # every checkpoint under CKPT_DIR (relative path and size), hashed
  # Not the newest name: the two arms of a pair keep the same ckpt_NNNNNN.pt names under one dir, so the newest NAME
  # stays frozen while the slower arm keeps making real progress. Any new, replaced or rotated checkpoint changes this.
  local l; l=$([ -d "$CKPT_DIR" ] && find "$CKPT_DIR" -maxdepth 2 -name "$CKPT_GLOB" -not -name '.*' -printf '%P:%s\n' 2>/dev/null | sort)
  if [ -z "$l" ]; then echo none; else printf '%s\n' "$l" | sha1sum | cut -c1-16; fi
}

CRASHES=$(cat "$RUN_DIR/crash.count" 2>/dev/null || echo 0)
PREV_SIG=$(cat "$RUN_DIR/crash.sig" 2>/dev/null || echo "")
RESTARTS=0; LAST_EXIT=null; STOPPING=0; CHILD=""; NAP=""

[ -f "$RUN_DIR/hold" ] && { blog "held: $(cat "$RUN_DIR/hold")"; status held; exit 0; }
[ -f "$RUN_DIR/done" ] && { blog "already done"; status "done"; exit 0; }
[ -f "$RUN_DIR/failed" ] && { blog "marked failed: $(cat "$RUN_DIR/failed"); clear with ctl.sh release"; status failed; exit 0; }
[ -f "$RUN_DIR/train.cmd" ] || { blog "no train.cmd; arm with ctl.sh arm"; exit 0; }

stop_child() {
  [ -n "$CHILD" ] || return 0
  local pg; pg=$(cat "$RUN_DIR/child.pgid" 2>/dev/null || true)
  blog "SIGTERM to the training process group ${pg:-$CHILD}"
  if [ -n "$pg" ]; then kill -TERM -- "-$pg" 2>/dev/null; else kill -TERM "$CHILD" 2>/dev/null; fi
  for _ in $(seq 1 90); do kill -0 "$CHILD" 2>/dev/null || return 0; sleep 1; done
  blog "training still alive after 90 s; SIGKILL"
  if [ -n "$pg" ]; then kill -KILL -- "-$pg" 2>/dev/null; else kill -KILL "$CHILD" 2>/dev/null; fi
}
# A signal must not leave a backoff or GPU-wait sleep behind, and must cut the wait short.
nap() { sleep "$1" & NAP=$!; wait "$NAP" || true; NAP=""; }
on_term() { STOPPING=1; [ -n "$NAP" ] && kill "$NAP" 2>/dev/null; stop_child; }
trap on_term TERM INT

gpus_ready() {
  [ "${SKIP_GPU_CHECK:-0}" = 1 ] && return 0
  local n; n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  [ "$n" -ge "$EXPECT_GPUS" ]
}

while :; do
  [ "$STOPPING" = 1 ] && { status stopped; exit 0; }
  [ -f "$RUN_DIR/hold" ] && { blog "held: $(cat "$RUN_DIR/hold")"; status held; exit 0; }
  waited=0
  until gpus_ready; do
    [ "$waited" -ge "$GPU_WAIT_S" ] && { blog "fewer than $EXPECT_GPUS GPUs visible after ${GPU_WAIT_S}s"; status gpu_wait_timeout; exit 70; }
    nap 10; waited=$((waited + 10))
    [ "$STOPPING" = 1 ] && { status stopped; exit 0; }
  done
  blog "starting training (restart $RESTARTS, ckpt sig $(ckpt_sig))"
  status running
  # Own session so the whole tree (torchrun, ray, sglang) can be signalled as one group. The inner shell records
  # its pid, which setsid makes the group id whether or not setsid forks; `setsid -w` waits for it either way.
  rm -f "$RUN_DIR/child.pgid"
  # shellcheck disable=SC2016  # $$ and $1 must expand in the child shell, not here
  NEGEIG_RESUME=1 RESUME_FLAG=--resume setsid -w bash -c 'echo $$ >"$1"; exec bash "$2"' _ \
    "$RUN_DIR/child.pgid" "$RUN_DIR/train.cmd" >>"$LOG" 2>&1 </dev/null &
  CHILD=$!
  while :; do
    wait "$CHILD"; rc=$?
    # a trapped signal interrupts wait with rc > 128 while the child may still run
    if kill -0 "$CHILD" 2>/dev/null; then [ "$STOPPING" = 1 ] && { stop_child; }; continue; fi
    break
  done
  LAST_EXIT=$rc; CHILD=""
  if [ "$STOPPING" = 1 ]; then blog "stopped on request (exit $rc); not counted as a crash"; status stopped; exit 0; fi
  if [ "$rc" -eq 0 ]; then
    date -u +%FT%TZ >"$RUN_DIR/done"; rm -f "$RUN_DIR/crash.count" "$RUN_DIR/crash.sig"
    blog "training command finished with exit 0"; status "done"; exit 0
  fi
  sig=$(ckpt_sig)
  if [ "$sig" = "$PREV_SIG" ]; then CRASHES=$((CRASHES + 1)); else CRASHES=1; fi
  PREV_SIG=$sig
  echo "$CRASHES" >"$RUN_DIR/crash.count"; echo "$PREV_SIG" >"$RUN_DIR/crash.sig"
  blog "training exited $rc; ckpt sig $sig; crashes at this checkpoint $CRASHES of $MAX_CRASHES"
  if [ "$CRASHES" -ge "$MAX_CRASHES" ]; then
    echo "exit $rc, $CRASHES crashes at the same checkpoint (sig $sig) at $(date -u +%FT%TZ)" >"$RUN_DIR/failed"
    status failed; exit 0
  fi
  RESTARTS=$((RESTARTS + 1))
  status backoff
  nap "$BACKOFF_S"
done
