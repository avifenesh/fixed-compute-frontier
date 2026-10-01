#!/usr/bin/env bash
# rig.sh - RIG side. Source it in a bash shell, never run it:   . experiments/negeig_27b/box/rig.sh
# Five helpers, so the RUNBOOK is one line per step and a long box command never depends on an open ssh session.
# <box> is a name, id or ip from $STATE_DIR/instances.tsv (resolve_ip in common.sh, which also gives the box's CURRENT ip
# after a preemption restart). Nothing here touches the provider; it only opens ssh to a box the lane already owns.
#
#   bx     <box> COMMAND...             run on the box, output and exit status come back here. boxenv.sh is sourced first,
#                                       so $W $E27 $ACC $RUNS $SD $MODEL_DIR ... are set; COMMAND is one shell line (quote it
#                                       once on the rig; it is passed through printf %q, so any quoting inside survives).
#   bxd    <box> LABEL COMMAND...       start COMMAND detached on the box (setsid, nohup: it survives the ssh session). Log
#                                       $W/logs/LABEL.log, exit status $W/logs/LABEL.exit, script $W/logs/LABEL.cmd. Refused
#                                       while LABEL still runs. The previous log of that label is kept as LABEL.log.prev.
#   bxs    <box> LABEL [N]              the last N (20) log lines, then one line `STATE running|done|dead|none label=L rc=R`.
#                                       Exit 0 running or done rc=0, the command's own status when done, 1 dead, 2 none.
#   bxw    <box> LABEL [TIMEOUT_S] [INTERVAL_S]   poll bxs until LABEL ends (default 7200 s, every 30 s), then print it.
#                                       Returns the command's exit status. Its own failures: 124 deadline, 125 the box did
#                                       not answer 5 polls in a row, 126 the command died with no exit status (box reboot,
#                                       kill -9), 2 no such label. A preempted box makes it return 125, never loop.
#   bxpull <box> SUBDIR LOCAL_DIR [rsync args]   rsync $W/SUBDIR/ from the box into LOCAL_DIR/ (ionice idle, nice 10,
#                                       --bwlimit $BXPULL_BWLIMIT KB/s, default 40000; no --delete)
#
# Test hooks: NEGEIG_W (the box root, default /workspace/negeig) is read on the rig for bxpull and on the box for the rest.
# shellcheck disable=SC2317  # the exit arm is reached when the file is run, not sourced
if [ -z "${BASH_VERSION:-}" ]; then echo "rig.sh: source it from bash (run bash first)" >&2; return 1 2>/dev/null || exit 1; fi
RIG_BOX_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$RIG_BOX_DIR/common.sh"

# _rig_ssh <box> <remote command string>: stdin of this function is the remote command's stdin
_rig_ssh() {
  local ip
  ip=$(resolve_ip "$1") || return 255
  # shellcheck disable=SC2029  # the remote command string is built (and quoted) by the caller on purpose
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "$2"
}

bx() {
  local box=${1:-}
  [ $# -ge 2 ] || { echo "usage: bx <box> COMMAND..." >&2; return 2; }
  shift
  # shellcheck disable=SC2016  # ${NEGEIG_W...} is expanded on the box
  local script=". \"\${NEGEIG_W:-/workspace/negeig}/code/experiments/negeig_27b/box/boxenv.sh\"; $*"
  _rig_ssh "$box" "$(printf '%q ' bash -c "$script")" </dev/null
}

# the two box-side scripts: fixed text, the label arrives as $1 (validated on the rig first)
# shellcheck disable=SC2016
_BXD_REMOTE='
W=${NEGEIG_W:-/workspace/negeig}; D=$W/logs; l=$1
mkdir -p "$D" || exit 4
if [ -f "$D/$l.pid" ] && [ ! -f "$D/$l.exit" ]; then
  pid=$(cat "$D/$l.pid" 2>/dev/null)
  if [ -n "$pid" ] && grep -qa "$D/$l.cmd" "/proc/$pid/cmdline" 2>/dev/null; then
    echo "bxd: $l is still running on this box (pid $pid): bxw it, or wait, or kill it first" >&2; exit 3
  fi
fi
cat >"$D/$l.cmd.new" && mv "$D/$l.cmd.new" "$D/$l.cmd" || exit 4
[ ! -f "$D/$l.log" ] || mv -f "$D/$l.log" "$D/$l.log.prev"
rm -f "$D/$l.exit" "$D/$l.pid"
nohup setsid bash -c '"'"'echo $$ >"$3"; bash "$1"; echo $? >"$2.tmp"; mv "$2.tmp" "$2"'"'"' _ "$D/$l.cmd" "$D/$l.exit" "$D/$l.pid" \
  >"$D/$l.log" 2>&1 </dev/null &
for _ in $(seq 1 100); do [ -s "$D/$l.pid" ] && break; sleep 0.1; done
[ -s "$D/$l.pid" ] || { echo "bxd: $l did not start" >&2; exit 5; }
echo "started $l pid $(cat "$D/$l.pid") log $D/$l.log"
'
# shellcheck disable=SC2016
_BXS_REMOTE='
W=${NEGEIG_W:-/workspace/negeig}; D=$W/logs; l=$1; n=${2:-20}
if [ ! -f "$D/$l.cmd" ]; then echo "STATE none label=$l rc="; exit 2; fi
rc=""
if [ -f "$D/$l.exit" ]; then st=done; rc=$(cat "$D/$l.exit")
else
  pid=$(cat "$D/$l.pid" 2>/dev/null)
  if [ -n "$pid" ] && grep -qa "$D/$l.cmd" "/proc/$pid/cmdline" 2>/dev/null; then st=running; else st=dead; fi
fi
tail -n "$n" "$D/$l.log" 2>/dev/null
# a log whose last line has no newline (a tqdm bar ends in \r) would glue STATE onto it and bxw would miss it
[ -n "$(tail -c1 "$D/$l.log" 2>/dev/null)" ] && echo
echo "STATE $st label=$l rc=$rc"
case $st in running) exit 0 ;; done) exit "${rc:-1}" ;; *) exit 1 ;; esac
'

_bx_label_ok() { [[ ${1:-} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "bad label '${1:-}': letters, digits, . _ - only" >&2; return 1; }; }

bxd() {
  local box=${1:-} label=${2:-}
  [ $# -ge 3 ] || { echo "usage: bxd <box> LABEL COMMAND..." >&2; return 2; }
  _bx_label_ok "$label" || return 2
  shift 2
  # shellcheck disable=SC2016
  printf '%s\n' ". \"\${NEGEIG_W:-/workspace/negeig}/code/experiments/negeig_27b/box/boxenv.sh\"" "$*" \
    | _rig_ssh "$box" "$(printf '%q ' bash -c "$_BXD_REMOTE" _ "$label")"
}

bxs() {
  local box=${1:-} label=${2:-} n=${3:-20}
  [ $# -ge 2 ] || { echo "usage: bxs <box> LABEL [N]" >&2; return 2; }
  _bx_label_ok "$label" || return 2
  _rig_ssh "$box" "$(printf '%q ' bash -c "$_BXS_REMOTE" _ "$label" "$n")" </dev/null
}

bxw() {
  local box=${1:-} label=${2:-} timeout=${3:-7200} interval=${4:-30}
  [ $# -ge 2 ] || { echo "usage: bxw <box> LABEL [TIMEOUT_S] [INTERVAL_S]" >&2; return 2; }
  _bx_label_ok "$label" || return 2
  local deadline=$((SECONDS + timeout)) fails=0 out state rc last
  while :; do
    out=$(bxs "$box" "$label" 1 2>&1) || true
    state=$(printf '%s\n' "$out" | sed -n 's/^STATE \([a-z]*\) label=.*/\1/p' | tail -n 1)
    rc=$(printf '%s\n' "$out" | sed -n 's/^STATE [a-z]* label=[^ ]* rc=\(.*\)$/\1/p' | tail -n 1)
    if [ -z "$state" ]; then
      fails=$((fails + 1))
      echo "[$(date -u +%T)Z] $label: no answer from $box ($fails of 5): $(printf '%s' "$out" | tail -n 1 | cut -c1-120)" >&2
      if [ "$fails" -ge 5 ]; then echo "bxw: $box did not answer 5 polls in a row" >&2; return 125; fi
    else
      fails=0
      case $state in
        none) bxs "$box" "$label" 20; return 2 ;;
        done) bxs "$box" "$label" 20 || true; return "${rc:-1}" ;;
        dead) bxs "$box" "$label" 20 || true; echo "bxw: $label died with no exit status on $box (reboot or kill -9)" >&2; return 126 ;;
      esac
      last=$(printf '%s\n' "$out" | grep -v '^STATE ' | tail -n 1 | cut -c1-140)
      echo "[$(date -u +%T)Z] $label running: $last"
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then echo "bxw: $label still not finished after ${timeout}s" >&2; return 124; fi
    sleep "$interval"
  done
}

bxpull() {
  local box=${1:-} sub=${2:-} dest=${3:-} ip
  [ $# -ge 3 ] && [ -n "$sub" ] && [ -n "$dest" ] || { echo "usage: bxpull <box> SUBDIR LOCAL_DIR [rsync args]" >&2; return 2; }
  shift 3
  ip=$(resolve_ip "$box") || return 1
  mkdir -p "$dest" || return 1
  ionice -c3 nice -n 10 rsync -az --partial --human-readable --bwlimit="${BXPULL_BWLIMIT:-40000}" \
    -e "ssh ${SSH_OPTS[*]}" "$@" "$SSH_USER@$ip:${NEGEIG_W:-/workspace/negeig}/$sub/" "$dest/"
}
