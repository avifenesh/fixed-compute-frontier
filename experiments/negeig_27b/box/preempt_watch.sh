#!/usr/bin/env bash
# preempt_watch.sh - RIG side. Detects preemption of the lane's boxes from the Nebius CLI and, with --restart,
# brings them back and makes sure training resumes from its last checkpoint.
#
#   preempt_watch.sh [--once] [--interval 60] [--restart] [--deep] [--max-restarts 6] [--window-s 7200]
#                    [--start-retry-s 120]
#
# How Nebius preemption shows up: the instance is STOPPED (spec.preemptible.on_preemption STOP), not deleted;
# the managed disks stay and the public IP changes on the next start. So the signal is provider state STOPPED
# (or STOPPING) while our intent for that instance is `run` (intent/<id> is written by create_box.sh and set to
# `teardown` by teardown.sh before it deletes anything, so a deliberate stop is never mistaken for preemption).
#
# Per instance in $STATE_DIR/instances.tsv (the lane's own list, nothing else is ever looked at):
#   RUNNING                    ok; a changed public IP is logged (IP_CHANGED) and recorded
#   CREATING STARTING UPDATING transient, ok
#   STOPPING                   PREEMPTED (being stopped), exit 10; it is started once it reads STOPPED
#   STOPPED + intent run       PREEMPTED, exit 10; with --restart: `instance start --id`, wait RUNNING, wait ssh on
#                              the NEW ip, verify 8 GPUs, then on the box: start negeig27-train.service if it is not
#                              active and the run is armed, not held/done/failed (the unit is enabled, so it normally
#                              starts itself at boot; resume_wrapper.sh restarts the training command with
#                              NEGEIG_RESUME=1, and train27.py --resume loads the newest atomic ckpt)
#   NOT_FOUND DELETING ERROR   LOST, exit 11 (a deleted box took its disks with it: re-run create_box.sh,
#                              push_data.sh, bootstrap.sh, accept.sh; the provider lists show what is left)
#   QUERY_FAILED               CLI error (auth, network); retried, exit 11 after 5 in a row
# Also logged on every preemption: the last operations on the instance (best effort, the JSON shape of
# `compute instance list-operations-by-parent` was empty when written, so the filter is tolerant).
#
# Restart-storm guard: SUCCESSFUL starts are counted per instance; more than --max-restarts (6) in --window-s
# (2 h) stops restarting it (PREEMPT_STORM, exit 11): a box that is preempted that often needs the owner (the
# on-demand fallback is `create_box.sh --target h200`). Failed start attempts (no capacity) are retried every
# --start-retry-s and do not count toward the guard.
#
# --deep also asks each RUNNING box for ctl.sh status (the wrapper's state, step and crash count).
# Exit (with --once): 0 all RUNNING (or restarted and verified), 10 some box preempted and not (yet) back,
# 11 lost/error/storm/CLI down. Without --once it loops until interrupted and every pass prints its line.
# Events are appended to $STATE_DIR/events.log (TSV: time id name event detail).
# Run it detached on the rig at low priority, for example:
#   nice -n 10 ./preempt_watch.sh --restart --interval 60 >>"$HOME/.local/state/negeig27/watch.log" 2>&1 &
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$HERE/common.sh"

ONCE=0; INTERVAL=60; RESTART=0; DEEP=0; MAX_RESTARTS=6; WINDOW_S=7200; START_RETRY_S=120
while [ $# -gt 0 ]; do
  case "$1" in
    --once) ONCE=1; shift ;;
    --interval) INTERVAL=$2; shift 2 ;;
    --restart) RESTART=1; shift ;;
    --deep) DEEP=1; shift ;;
    --max-restarts) MAX_RESTARTS=$2; shift 2 ;;
    --window-s) WINDOW_S=$2; shift 2 ;;
    --start-retry-s) START_RETRY_S=$2; shift 2 ;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done
need nebius; need jq; need ssh
state_init
EVENTS=$STATE_DIR/events.log
RESTARTS=$STATE_DIR/restarts.log     # "epoch id", one per SUCCESSFUL start
BOX_W=${NEGEIG_W:-/workspace/negeig}

event() { # id name event detail
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$3" "${4:-}" >>"$EVENTS"
  log "$2 ($1) $3 ${4:-}"
}

recent_ops() { # id project -> compact JSON of the last 3 operations on this resource, or empty
  nb compute instance list-operations-by-parent --parent-id "$2" --page-size 20 --format json 2>/dev/null \
    | jq -c --arg id "$1" '[(.operations // .items // [])[]? | select((.resource_id // .resourceId // "") == $id)] | .[-3:]' 2>/dev/null || true
}

restarts_in_window() { # id
  local now; now=$(date +%s)
  [ -f "$RESTARTS" ] || { echo 0; return; }
  awk -v id="$1" -v now="$now" -v w="$WINDOW_S" '$2==id && now-$1<=w {n++} END {print n+0}' "$RESTARTS"
}

# verify_box id name ip -> 0 when ssh answers, the box has all GPUs and (if armed) the training unit is active.
# A failure leaves $STATE_DIR/attn.<id>: the box is RUNNING at the provider but not proven back, so later passes
# keep reporting it (exit 10) and, with --restart, repeat this verification until it passes.
verify_box() {
  local id=$1 name=$2 ip=$3 rep
  wait_ssh "$ip" 60 || { event "$id" "$name" START_NO_SSH "no ssh on $ip after 10 min"; echo "no ssh" >"$STATE_DIR/attn.$id"; return 1; }
  rep=$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" bash -s -- "$BOX_W" <<'REMOTE' 2>&1
W=$1
gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "gpus=$gpus"
R=$W/run
if [ -f "$R/train.cmd" ] && [ ! -e "$R/hold" ] && [ ! -e "$R/done" ] && [ ! -e "$R/failed" ]; then
  if systemctl is-active --quiet negeig27-train.service; then
    echo "unit=active"
  else
    sudo systemctl reset-failed negeig27-train.service 2>/dev/null
    sudo systemctl start negeig27-train.service && echo "unit=started" || echo "unit=START_FAILED"
  fi
else
  echo "unit=not_armed_or_held_done_failed"
fi
[ -f "$R/status.json" ] && echo "status=$(tr -d '\n' <"$R/status.json" | cut -c1-300)"
exit 0   # the last test above is false on a box with no status.json yet; that must not read as an ssh failure
REMOTE
  ) || { event "$id" "$name" VERIFY_SSH_CMD_FAILED "$(echo "$rep" | tr '\n' ' ' | cut -c1-300)"; echo "ssh cmd failed" >"$STATE_DIR/attn.$id"; return 1; }
  event "$id" "$name" VERIFIED "ip $ip; $(echo "$rep" | tr '\n' ' ' | cut -c1-400)"
  if ! echo "$rep" | grep -q "gpus=${EXPECT_GPUS:-8}\$"; then
    event "$id" "$name" GPU_COUNT_WRONG "$(echo "$rep" | head -n1)"; echo "gpu count" >"$STATE_DIR/attn.$id"; return 1
  fi
  if echo "$rep" | grep -q 'unit=START_FAILED'; then
    event "$id" "$name" UNIT_START_FAILED "see journalctl -u negeig27-train on the box"; echo "unit start" >"$STATE_DIR/attn.$id"; return 1
  fi
  rm -f "$STATE_DIR/attn.$id"
  return 0
}

# restart_one id name project -> 0 if the box is back RUNNING and verify_box passed
restart_one() {
  local id=$1 name=$2 project=$3 now last ip n st out i
  now=$(date +%s); last=$(cat "$STATE_DIR/lastattempt.$id" 2>/dev/null || echo 0)
  if [ $((now - last)) -lt "$START_RETRY_S" ]; then return 1; fi
  n=$(restarts_in_window "$id")
  if [ "$n" -ge "$MAX_RESTARTS" ]; then
    event "$id" "$name" PREEMPT_STORM "$n successful starts in the last ${WINDOW_S}s: not restarting, owner decision"
    STORM=1; return 1
  fi
  echo "$now" >"$STATE_DIR/lastattempt.$id"
  event "$id" "$name" START_REQUEST "start $((n + 1)) of at most $MAX_RESTARTS in ${WINDOW_S}s"
  if ! out=$(nb compute instance start --id "$id" --timeout 10m 2>&1); then
    event "$id" "$name" START_FAILED "$(echo "$out" | tr '\n' ' ' | cut -c1-300)"
    return 1
  fi
  echo "$now $id" >>"$RESTARTS"
  for ((i = 0; i < 60; i++)); do
    st=$(inst_state "$id"); [ "$st" = RUNNING ] && break; sleep 10
  done
  [ "$st" = RUNNING ] || { event "$id" "$name" START_NOT_RUNNING "state $st after 10 min"; return 1; }
  ip=$(inst_ip "$id")
  [ -n "$ip" ] || { event "$id" "$name" START_NO_IP "RUNNING but no public ip"; echo "no ip" >"$STATE_DIR/attn.$id"; return 1; }
  echo "$ip" >"$STATE_DIR/ip.$id"
  forget_host "$ip"   # Nebius recycles IPs: a key left by an earlier box on this address would fail BatchMode ssh
  verify_box "$id" "$name" "$ip"
}

# check_one id -> sets RC_ONE (0, 10, 11)
check_one() {
  local id=$1 name project intent st ip prev
  STORM=0
  name=$(table_field "$id" 1); project=$(table_field "$id" 3)
  intent=$(cat "$INTENT_DIR/$id" 2>/dev/null || echo unknown)
  RC_ONE=0
  if [ "$intent" = teardown ]; then event "$id" "$name" TEARDOWN_INTENT "state $(inst_state "$id"), not watched"; return; fi
  [ "$intent" = run ] || { event "$id" "$name" NO_INTENT "no intent file for this id; treating as run"; }
  st=$(inst_state "$id")
  case "$st" in
    QUERY_FAILED)
      local qf; qf=$(( $(cat "$STATE_DIR/qf.$id" 2>/dev/null || echo 0) + 1 )); echo "$qf" >"$STATE_DIR/qf.$id"
      event "$id" "$name" QUERY_FAILED "consecutive failures: $qf"
      [ "$qf" -ge 5 ] && RC_ONE=11
      return ;;
  esac
  rm -f "$STATE_DIR/qf.$id"
  case "$st" in
    RUNNING)
      ip=$(inst_ip "$id"); prev=$(cat "$STATE_DIR/ip.$id" 2>/dev/null || true)
      if [ -n "$ip" ] && [ "$ip" != "$prev" ]; then
        [ -n "$prev" ] && event "$id" "$name" IP_CHANGED "$prev -> $ip (push_data.sh takes the name, not the ip)"
        echo "$ip" >"$STATE_DIR/ip.$id"; forget_host "$ip"
      fi
      log "$name RUNNING ip ${ip:-none}"
      if [ -f "$STATE_DIR/attn.$id" ]; then
        event "$id" "$name" ATTENTION "RUNNING but not proven back ($(cat "$STATE_DIR/attn.$id")); rm $STATE_DIR/attn.$id once fixed by hand"
        RC_ONE=10
        if [ "$RESTART" = 1 ] && [ -n "$ip" ] && verify_box "$id" "$name" "$ip"; then RC_ONE=0; fi
      fi
      if [ "$DEEP" = 1 ] && [ -n "$ip" ]; then
        ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "NEGEIG_W=$BOX_W $BOX_W/code/experiments/negeig_27b/box/ctl.sh status" 2>&1 | sed "s/^/  [$name] /" || true
      fi ;;
    CREATING|STARTING|UPDATING) log "$name $st (transient)" ;;
    STOPPING)
      event "$id" "$name" PREEMPTED "state STOPPING while intent is $intent; ops $(recent_ops "$id" "$project")"
      RC_ONE=10 ;;
    STOPPED)
      event "$id" "$name" PREEMPTED "state STOPPED while intent is $intent; ops $(recent_ops "$id" "$project")"
      RC_ONE=10
      if [ "$RESTART" = 1 ]; then
        if restart_one "$id" "$name" "$project"; then RC_ONE=0; fi
        [ "${STORM:-0}" = 1 ] && RC_ONE=11
      fi ;;
    NOT_FOUND|DELETING) event "$id" "$name" LOST "state $st: the instance and its managed disks are gone"; RC_ONE=11 ;;
    ERROR) event "$id" "$name" ERROR "provider state ERROR: $(inst_json "$id" | jq -c '.status // {}' 2>/dev/null | cut -c1-300)"; RC_ONE=11 ;;
    *) event "$id" "$name" UNKNOWN_STATE "$st"; RC_ONE=10 ;;
  esac
}

STORM=0
while :; do
  rc=0; ids=$(table_ids)
  [ -n "$ids" ] || { log "no instances in $TABLE"; [ "$ONCE" = 1 ] && exit 0; }
  for id in $ids; do
    check_one "$id"
    [ "$RC_ONE" -gt "$rc" ] && rc=$RC_ONE
  done
  if [ "$ONCE" = 1 ]; then exit "$rc"; fi
  sleep "$INTERVAL"
done
