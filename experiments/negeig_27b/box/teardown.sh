#!/usr/bin/env bash
# teardown.sh - RIG side. Stops and deletes the lane's instances BY ID and confirms from the provider's own
# instance and disk lists that nothing tagged lane=negeig-27b (or a disk named negeig27-*) is left.
#
#   teardown.sh --list                              read-only: what the PROVIDER shows for this lane, both projects
#   teardown.sh [--ids "id id"] [--pull [DIR]] [--pull-max-size 8G] [--skip-unreachable] [--bwlimit KBPS] [--yes]
#
# Default is a dry run that prints the exact commands. With --yes, per instance and in this order:
#   1. intent/<id> := teardown   (first, so preempt_watch.sh never restarts a box that is being removed)
#   2. --pull: rsync $W/runs, $W/accept, $W/logs, $W/run to DIR/<name>/ BEFORE anything is deleted (the managed
#      disks die with the instance). Default DIR is $STATE_DIR/pulled. Merged full-model copies stay behind BY NAME
#      (merged*/ dirs, HF weight shards model-NNNNN-of-NNNNN.safetensors, model.safetensors, pytorch_model*.bin:
#      54 GB sharded into 5 GB pieces would pass a size cap); LoRA adapters, trainable_*.pt and ckpt_*.pt (the
#      optimizer state, under 2 GB) do cross. --pull-max-size (default 8G) is the backstop for anything else huge. A STOPPED
#      (preempted) box cannot be pulled: start it first (`nebius compute instance start --id <id>`) or pass
#      --skip-unreachable to delete without the pull.
#   3. `instance stop --id` (only if RUNNING), then `instance delete --id` (also deletes its managed disks)
#   4. confirm: the instance reads NOT_FOUND by id, and `instance list` and `disk list` of the project show no
#      lane=negeig-27b instance and no negeig27-* disk. Polled up to 10 minutes. Prints TEARDOWN_CONFIRMED, or
#      TEARDOWN_INCOMPLETE with what remains (exit 1). The destroy call's own return is never the confirmation.
# Identity is read firsthand from the provider before any delete: the id must be in $STATE_DIR/instances.tsv AND
# the provider's record must carry label lane=negeig-27b and a name starting negeig27-. Anything else is refused.
# Confirmed rows move to $STATE_DIR/instances.history.tsv.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$HERE/common.sh"

LIST=0; YES=0; IDS=""; PULL=0; PULL_DIR=""; PULL_MAX=8G; SKIP_UNREACH=0; BW=${BWLIMIT_KBPS:-40000}
while [ $# -gt 0 ]; do
  case "$1" in
    --list) LIST=1; shift ;;
    --yes) YES=1; shift ;;
    --ids) IDS=$2; shift 2 ;;
    --pull) PULL=1; if [ $# -gt 1 ] && [[ $2 != --* ]]; then PULL_DIR=$2; shift 2; else shift; fi ;;
    --pull-max-size) PULL_MAX=$2; shift 2 ;;
    --skip-unreachable) SKIP_UNREACH=1; shift ;;
    --bwlimit) BW=$2; shift 2 ;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done
need nebius; need jq
state_init
BOX_W=${NEGEIG_W:-/workspace/negeig}
[ -n "$PULL_DIR" ] || PULL_DIR=$STATE_DIR/pulled

# every project the lane may have resources in: each configured target project plus whatever the table recorded
lane_projects() {
  { for t in b300 b300w h200 h200w; do ( target_config "$t" 2>/dev/null && echo "$T_PROJECT" ) || true; done
    awk -F'\t' 'NF>=3 && $3 != "" {print $3}' "$TABLE"; } | sort -u
}
[ -n "$(lane_projects)" ] || die "no lane project known: no target configured in $TARGETS_FILE and nothing in $TABLE"

show_provider() { # prints what the provider holds for the lane; returns number of items
  local p n=0 rows
  for p in $(lane_projects); do
    rows=$(provider_lane_instances "$p") || { echo "  $p: instance list FAILED"; n=$((n + 1000)); continue; }
    if [ -n "$rows" ]; then echo "$rows" | sed "s|^|  instance [$p] |"; n=$((n + $(echo "$rows" | wc -l))); else echo "  instance [$p] none"; fi
    rows=$(provider_lane_disks "$p") || { echo "  $p: disk list FAILED"; n=$((n + 1000)); continue; }
    if [ -n "$rows" ]; then echo "$rows" | sed "s|^|  disk     [$p] |"; n=$((n + $(echo "$rows" | wc -l))); else echo "  disk     [$p] none"; fi
  done
  return $((n > 255 ? 255 : n))
}

if [ "$LIST" = 1 ]; then
  log "provider view, lane label $LANE_LABEL (profile $NB_PROFILE)"
  show_provider; n=$?
  log "local table $TABLE:"; if [ -s "$TABLE" ]; then cat "$TABLE"; else echo "  (empty)"; fi
  [ "$n" -eq 0 ] && exit 0
  exit 10
fi

IDS_GIVEN=$IDS
[ -n "$IDS" ] || IDS=$(table_ids | tr '\n' ' ')
[ -n "${IDS// /}" ] || { log "no instances in the table; provider view:"; show_provider; exit 0; }

# ---- firsthand identity check and the plan
PLAN=()
for id in $IDS; do
  [ -n "$(table_field "$id" 1)" ] || die "$id is not in $TABLE: refusing to touch an instance this lane did not create"
  name=$(table_field "$id" 1)
  st=$(inst_state "$id")
  case "$st" in
    QUERY_FAILED) die "cannot read $id from the provider (CLI error): not deleting on an unread identity" ;;
    NOT_FOUND) log "$name ($id) already NOT_FOUND at the provider"; PLAN+=("$id|$name|NOT_FOUND"); continue ;;
  esac
  pname=$(inst_json "$id" | jq -r '.metadata.name // ""')
  inst_labels_ok "$id" || die "$name ($id): provider record has no label lane=$LANE_LABEL, refusing"
  case "$pname" in negeig27-*) ;; *) die "$id: provider name '$pname' does not start with negeig27-, refusing" ;; esac
  PLAN+=("$id|$name|$st")
done

log "plan (dry-run=$([ "$YES" = 1 ] && echo no || echo yes)):"
for e in "${PLAN[@]}"; do
  IFS='|' read -r id name st <<<"$e"
  echo "  $name ($id) state $st"
  [ "$st" = NOT_FOUND ] && continue
  echo "    echo teardown >$INTENT_DIR/$id"
  [ "$PULL" = 1 ] && echo "    rsync  $SSH_USER@<ip>:$BOX_W/{runs,accept,logs,run}/  ->  $PULL_DIR/$name/  (max-size $PULL_MAX, bwlimit $BW KB/s)"
  [ "$st" = RUNNING ] && echo "    nebius --profile $NB_PROFILE compute instance stop   --id $id --timeout 15m"
  echo "    nebius --profile $NB_PROFILE compute instance delete --id $id --timeout 15m"
done
echo "  then confirm from: nebius compute instance list / disk list --parent-id <project> --all (label lane=$LANE_LABEL, disks negeig27-*)"
if [ "$YES" != 1 ]; then log "dry run only. Re-run with --yes to execute."; echo "--- provider view now:"; show_provider; exit 0; fi

# ---- execute
rc=0
PLAN_IDS=""; PLAN_NAMES=""; FULL=0
for e in "${PLAN[@]}"; do IFS='|' read -r id name st <<<"$e"; PLAN_IDS="$PLAN_IDS $id"; PLAN_NAMES="$PLAN_NAMES $name"; done
# FULL: the plan is every instance in the table, so ANY lane-labelled instance or negeig27-* disk left in the
# provider lists counts as residue. A subset (--ids) is confirmed for its own instances and disks only.
[ -z "$IDS_GIVEN" ] && FULL=1
for e in "${PLAN[@]}"; do
  IFS='|' read -r id name st <<<"$e"
  [ "$st" = NOT_FOUND ] && continue
  echo teardown >"$INTENT_DIR/$id"
  if [ "$PULL" = 1 ]; then
    ip=$(inst_ip "$id")
    if [ "$st" = RUNNING ] && [ -n "$ip" ] && ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" true 2>/dev/null; then
      mkdir -p "$PULL_DIR/$name"
      # only the dirs that exist: a box that failed before bootstrap has none, and rsync rc 23 on a missing source
      # must not read as a failed pull (it would block the delete of a box the owner is throwing away)
      have=$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "cd '$BOX_W' 2>/dev/null && ls -d runs accept logs run 2>/dev/null" || true)
      [ -n "$have" ] || log "$name has none of runs accept logs run under $BOX_W: nothing to pull"
      for d in $have; do
        log "pull $name:$BOX_W/$d -> $PULL_DIR/$name/$d"
        ionice -c3 nice -n 10 rsync -az --partial --human-readable --bwlimit="$BW" --max-size="$PULL_MAX" \
          --exclude '.ckpt_*.tmp' --exclude 'merged*/' --exclude 'model-[0-9]*-of-[0-9]*.safetensors' \
          --exclude 'model.safetensors' --exclude 'pytorch_model*.bin' -e "ssh ${SSH_OPTS[*]}" "$SSH_USER@$ip:$BOX_W/$d/" "$PULL_DIR/$name/$d/" \
          || { log "pull of $d failed or partial (rsync rc $?)"; [ "$SKIP_UNREACH" = 1 ] || { log "not deleting $name: pull incomplete (--skip-unreachable to override)"; rc=1; continue 2; }; }
      done
    elif [ "$SKIP_UNREACH" = 1 ]; then
      log "WARNING $name is $st or unreachable: deleting WITHOUT a pull (--skip-unreachable)"
    else
      log "cannot pull $name (state $st, ip ${ip:-none}): start it by hand (nebius compute instance start --id $id) or pass --skip-unreachable. Not deleting; intent stays teardown so preempt_watch.sh will not restart it."
      rc=1; continue
    fi
  fi
  if [ "$st" = RUNNING ]; then
    log "stop $name ($id)"
    nb compute instance stop --id "$id" --timeout 15m || log "stop returned an error (delete follows; the list decides)"
  fi
  log "delete $name ($id)"
  nb compute instance delete --id "$id" --timeout 15m || log "delete returned an error (the provider list decides)"
done

# ---- confirm from the provider lists, never from the calls above
log "confirming from the provider lists (up to 10 min)"
remaining=""
for ((i = 0; i < 60; i++)); do
  remaining=""
  for e in "${PLAN[@]}"; do
    IFS='|' read -r id name st <<<"$e"
    s=$(inst_state "$id")
    [ "$s" = NOT_FOUND ] || remaining="$remaining instance:$name($id,$s)"
  done
  for p in $(lane_projects); do
    x=$(provider_lane_instances "$p") || x="LISTFAILED"
    if [ "$FULL" != 1 ] && [ "$x" != LISTFAILED ]; then x=$(echo "$x" | awk -F'\t' -v ids=" $PLAN_IDS " 'index(ids, " " $1 " ")'); fi
    [ -z "$x" ] || remaining="$remaining lane-instances[$p]:$(echo "$x" | tr '\t\n' ' ')"
    x=$(provider_lane_disks "$p") || x="LISTFAILED"
    if [ "$FULL" != 1 ] && [ "$x" != LISTFAILED ]; then
      x=$(echo "$x" | awk -F'\t' -v names="$PLAN_NAMES" 'BEGIN { n = split(names, a, " ") } { for (i = 1; i <= n; i++) if (index($2, a[i] "-") == 1) { print; break } }')
    fi
    [ -z "$x" ] || remaining="$remaining lane-disks[$p]:$(echo "$x" | tr '\t\n' ' ')"
  done
  [ -z "$remaining" ] && break
  sleep 10
done
if [ -z "$remaining" ] && [ "$rc" -eq 0 ]; then
  for e in "${PLAN[@]}"; do IFS='|' read -r id name st <<<"$e"
    row=$(awk -F'\t' -v id="$id" '$2==id' "$TABLE")
    [ -n "$row" ] && printf '%s\t%s\n' "$row" "$(date -u +%FT%TZ)" >>"$STATE_DIR/instances.history.tsv"
    awk -F'\t' -v id="$id" '$2!=id' "$TABLE" >"$TABLE.new" && mv "$TABLE.new" "$TABLE"
  done
  if [ "$FULL" = 1 ]; then
    log "TEARDOWN_CONFIRMED: no lane=$LANE_LABEL instance and no negeig27-* disk in $(lane_projects | tr '\n' ' ')"
  else
    log "TEARDOWN_CONFIRMED (subset):${PLAN_NAMES} and their disks are gone from the provider lists; other lane resources were not touched"
  fi
  exit 0
fi
log "TEARDOWN_INCOMPLETE:${remaining:- (a pull or delete step was skipped, see above)}"
exit 1
