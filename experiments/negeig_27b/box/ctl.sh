#!/usr/bin/env bash
# ctl.sh - runs ON the box (the rig calls it over ssh). Arms, inspects and stops the resumable training job.
#
#   ctl.sh arm (--cmd-file FILE | --cmd 'shell command') --ckpt-dir DIR [--ckpt-glob 'ckpt_*.pt']
#              [--max-crashes 3] [--backoff 60] [--log FILE] [--no-start]
#   ctl.sh status            state, newest checkpoint, unit state, last log lines (exit 0 running|done|held, 1 failed, 2 not armed)
#   ctl.sh hold [reason]     stop the job cleanly (SIGTERM to the group) and keep it from restarting, also across reboot
#   ctl.sh release           clear hold/failed and start it again (resumes from the newest checkpoint)
#   ctl.sh disarm            stop and forget the command (the unit no longer starts at boot)
#   ctl.sh logs [N]          last N lines of the training log
#
# The command runs as `bash train.cmd` with NEGEIG_RESUME=1 and RESUME_FLAG=--resume in the environment. Write it so it
# resumes: for train27.py end the line with $RESUME_FLAG; verl's run_grpo.sh resumes by itself (resume_mode=auto).
# A --cmd-file is copied as is (source boxenv.sh in it if it needs the venv paths); a --cmd string gets that header.
# Pair layout (run_pair.sh, two arms under one tag dir): arm with --ckpt-dir $RUNS/<tag> (checkpoints sit one level down,
# wide_s0/ckpt_*.pt and ctrl_s0/ckpt_*.pt; progress in EITHER arm resets the crash counter). The usual call, one line:
#   ctl.sh arm --cmd '. "$ACC/grad_ckpt.env"; exec bash "$E27/run_pair.sh" sweep-lr1e-4 0 --lr 1e-4 --steps 700 $GRAD_CKPT_FLAGS' \
#              --ckpt-dir "$RUNS/sweep-lr1e-4"
# ($ACC/grad_ckpt.env is what accept.sh decided; GRAD_CKPT_FLAGS is "--grad_ckpt 0" or "--grad_ckpt 1". RUNBOOK.md has the
# exact commands; train_pair.cmd.example is the same thing as a file.) run_pair.sh retries an arm itself and marks a dead
# one with arm_failed, which `status` lists.
# Set NO_SYSTEMD=1 to manage the control files only (used by the CPU test of the wrapper).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=boxenv.sh
. "$HERE/boxenv.sh"
mkdir -p "$RUN_DIR" "$LOGS"
sc() { [ "${NO_SYSTEMD:-0}" = 1 ] && { echo "(NO_SYSTEMD) systemctl $*"; return 0; }; sudo systemctl "$@"; }
jget() { [ -f "$RUN_DIR/status.json" ] && python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2],""))' "$RUN_DIR/status.json" "$1" || true; }

cmd=${1:-}; [ $# -gt 0 ] && shift
case "$cmd" in
  arm)
    CMD_FILE=""; CMD_STR=""; CKPT_DIR_ARG=""; CKPT_GLOB_ARG='ckpt_*.pt'; MAXC=3; BACK=60; LOG_ARG=$LOGS/train.log; START=1
    while [ $# -gt 0 ]; do
      case "$1" in
        --cmd-file) CMD_FILE=$2; shift 2 ;;
        --cmd) CMD_STR=$2; shift 2 ;;
        --ckpt-dir) CKPT_DIR_ARG=$2; shift 2 ;;
        --ckpt-glob) CKPT_GLOB_ARG=$2; shift 2 ;;
        --max-crashes) MAXC=$2; shift 2 ;;
        --backoff) BACK=$2; shift 2 ;;
        --log) LOG_ARG=$2; shift 2 ;;
        --no-start) START=0; shift ;;
        *) bdie "unknown argument: $1" ;;
      esac
    done
    [ -n "$CKPT_DIR_ARG" ] || bdie "--ckpt-dir is required (the directory the command writes checkpoints to)"
    { [ -n "$CMD_FILE" ] && [ -z "$CMD_STR" ]; } || { [ -z "$CMD_FILE" ] && [ -n "$CMD_STR" ]; } || bdie "give exactly one of --cmd-file or --cmd"
    if [ -n "$CMD_FILE" ]; then
      [ -f "$CMD_FILE" ] || bdie "no such file: $CMD_FILE"
      cp "$CMD_FILE" "$RUN_DIR/train.cmd.new"
    else
      { echo '#!/usr/bin/env bash'; echo 'set -eo pipefail'; echo ". \"$HERE/boxenv.sh\""; echo "$CMD_STR"; } >"$RUN_DIR/train.cmd.new"
    fi
    bash -n "$RUN_DIR/train.cmd.new" || bdie "the command has a syntax error"
    sc stop "$TRAIN_UNIT" 2>/dev/null || true
    rm -f "$RUN_DIR"/hold "$RUN_DIR"/done "$RUN_DIR"/failed "$RUN_DIR"/crash.count "$RUN_DIR"/crash.sig "$RUN_DIR"/status.json
    mv "$RUN_DIR/train.cmd.new" "$RUN_DIR/train.cmd"
    mkdir -p "$CKPT_DIR_ARG"
    { printf 'CKPT_DIR=%q\nCKPT_GLOB=%q\nMAX_CRASHES=%q\nBACKOFF_S=%q\nLOG=%q\n' "$CKPT_DIR_ARG" "$CKPT_GLOB_ARG" "$MAXC" "$BACK" "$LOG_ARG"; } >"$RUN_DIR/run.env"
    sc daemon-reload
    if [ "$START" = 1 ]; then sc enable --now "$TRAIN_UNIT"; else sc enable "$TRAIN_UNIT"; fi
    blog "armed: $RUN_DIR/train.cmd, checkpoints $CKPT_DIR_ARG/$CKPT_GLOB_ARG, max crashes at one checkpoint $MAXC, log $LOG_ARG"
    ;;
  status)
    if [ ! -f "$RUN_DIR/train.cmd" ]; then echo "state: not_armed"; exit 2; fi
    # shellcheck source=/dev/null
    [ -f "$RUN_DIR/run.env" ] && . "$RUN_DIR/run.env"
    st=$(jget state); [ -n "$st" ] || st=no_status_yet
    [ -f "$RUN_DIR/hold" ] && st=held
    [ -f "$RUN_DIR/failed" ] && st=failed
    [ -f "$RUN_DIR/done" ] && st="done"
    echo "state: $st"
    [ -f "$RUN_DIR/status.json" ] && cat "$RUN_DIR/status.json"
    [ -f "$RUN_DIR/hold" ] && echo "hold: $(cat "$RUN_DIR/hold")"
    [ -f "$RUN_DIR/failed" ] && echo "failed: $(cat "$RUN_DIR/failed")"
    [ -f "$RUN_DIR/done" ] && echo "done: $(cat "$RUN_DIR/done")"
    echo "checkpoints: $(find "${CKPT_DIR:-$RUNS}" -maxdepth 2 -name "${CKPT_GLOB:-ckpt_*.pt}" -not -name '.*' -printf '%P ' 2>/dev/null | tr ' ' '\n' | sort -V | tail -n6 | tr '\n' ' ')"
    # run_pair.sh gives up on one arm with an arm_failed marker (the other arm keeps running): list them with the reason
    while IFS= read -r af; do
      [ -n "$af" ] && echo "arm_failed: $af: $(cat "${CKPT_DIR:-$RUNS}/$af")"
    done < <(find "${CKPT_DIR:-$RUNS}" -maxdepth 2 -name arm_failed -printf '%P\n' 2>/dev/null | sort)
    [ "${NO_SYSTEMD:-0}" = 1 ] || echo "unit: $(systemctl is-active "$TRAIN_UNIT" 2>/dev/null || true) enabled=$(systemctl is-enabled "$TRAIN_UNIT" 2>/dev/null || true)"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';' && echo
    echo "--- last log lines"; tail -n "${TAIL:-8}" "${LOG:-$LOGS/train.log}" 2>/dev/null || true
    case "$st" in failed) exit 1 ;; esac
    ;;
  hold)
    echo "${*:-held by hand} at $(date -u +%FT%TZ)" >"$RUN_DIR/hold"
    sc stop "$TRAIN_UNIT" || true
    blog "held ($RUN_DIR/hold); release with: ctl.sh release"
    ;;
  release)
    rm -f "$RUN_DIR"/hold "$RUN_DIR"/failed "$RUN_DIR"/crash.count "$RUN_DIR"/crash.sig
    [ -f "$RUN_DIR/train.cmd" ] || bdie "not armed"
    sc start "$TRAIN_UNIT"
    blog "released"
    ;;
  disarm)
    sc disable --now "$TRAIN_UNIT" 2>/dev/null || true
    [ -f "$RUN_DIR/train.cmd" ] && mv "$RUN_DIR/train.cmd" "$RUN_DIR/train.cmd.disarmed.$(date -u +%Y%m%dT%H%M%SZ)"
    blog "disarmed"
    ;;
  logs)
    # shellcheck source=/dev/null
    [ -f "$RUN_DIR/run.env" ] && . "$RUN_DIR/run.env"
    tail -n "${1:-50}" "${LOG:-$LOGS/train.log}"
    ;;
  ""|-h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' ;;
  *) bdie "unknown command: $cmd (arm|status|hold|release|disarm|logs)" ;;
esac
