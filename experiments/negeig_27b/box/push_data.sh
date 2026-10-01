#!/usr/bin/env bash
# push_data.sh - RIG side. Pushes the code and the prepared data from the rig to a box. Direction: rig -> box,
# always (the rig has the tokenized Hebrew/agentic data and the Stage A inputs; the boxes never reach back into it,
# so there is no inbound ssh to the rig and no rig credential on a rented box). Re-run it after every preemption
# restart or code change: rsync only moves what differs.
#
#   push_data.sh <name|id|ip> [--code-only] [--with-sd] [--repo DIR] [--data DIR] [--hebrew-lane DIR] [--dry-run] [--bwlimit KBPS]
#
# Lands on the box (paths from boxenv.sh, W=/workspace/negeig):
#   $W/code/experiments/negeig_retrofit/   <- <repo>/experiments/negeig_retrofit   (no __pycache__, no out/)
#   $W/code/experiments/negeig_27b/        <- <repo>/experiments/negeig_27b         (box/ scripts included;
#                                             out/ is data, not code, and goes to $W/data below)
#   $W/data/                               <- --data DIR, else $NEGEIG_DATA_SRC, else <repo>/experiments/negeig_27b/out,
#                                             else the rig's prepared set /data/ai-ml/models/_runs/negeig-27b/data
#                                             (s1 jsonl domains, s2, s4 replay); a missing source is an ERROR, not a warning
#   /workspace/lane/research/hebrew-rl-20260923/  <- --hebrew-lane DIR (verl launch scripts, prompts), optional
# The data push SKIPS <data>/sd/ (the self-distilled replay) unless --with-sd. The rig's prepared set carries the live
# lane's sd/chat_sft.jsonl and tool_sft.jsonl; pushing them onto a fresh box would make accept.sh and run_pair.sh
# find a replay that this run never generated (ALLOW_NO_REPLAY=0 would not fire, the grad_ckpt decision would be taken on it, and
# Stage A could train on it). The replay reaches $W/data/sd/ only through sd_sync.sh after the box's own self-distillation
# (or --with-sd, a deliberate choice to reuse the rig's copy).
# --code-only pushes the scripts and nothing else (used before `bootstrap.sh --stage base`, which is the real-CUDA gate
# that must pass before any data is staged).
# Then run `bootstrap.sh --stage units` again only if the .service template changed.
#
# Load on the rig: ionice idle class, nice 10, rsync --bwlimit (default 40000 KB/s). An unthrottled upload storm
# once flooded 25 GB of swap here; keep the cap. Nothing is deleted on the box (no --delete): a box may hold
# checkpoints under $W/runs that must survive a push.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$HERE/common.sh"

REPO=$(cd "$HERE/../../.." && pwd); DATA_SRC=${NEGEIG_DATA_SRC:-}; LANE_SRC=""; DRY=0; CODE_ONLY=0; WITH_SD=0; BW=${BWLIMIT_KBPS:-40000}; TARGET_ARG=""
W=${NEGEIG_W:-/workspace/negeig}
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO=$2; shift 2 ;;
    --data) DATA_SRC=$2; shift 2 ;;
    --hebrew-lane) LANE_SRC=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --code-only) CODE_ONLY=1; shift ;;
    --with-sd) WITH_SD=1; shift ;;
    --bwlimit) BW=$2; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) die "unknown argument: $1" ;;
    *) [ -z "$TARGET_ARG" ] || die "one box at a time"; TARGET_ARG=$1; shift ;;
  esac
done
[ -n "$TARGET_ARG" ] || die "usage: push_data.sh <name|id|ip> [...]"
need rsync; need ssh
[ -d "$REPO/experiments/negeig_retrofit" ] && [ -d "$REPO/experiments/negeig_27b" ] || die "$REPO has no experiments/negeig_retrofit and negeig_27b"
if [ "$CODE_ONLY" = 0 ] && [ -z "$DATA_SRC" ]; then
  for d in "$REPO/experiments/negeig_27b/out" /data/ai-ml/models/_runs/negeig-27b/data; do
    [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ] && { DATA_SRC=$d; break; }
  done
fi
case "$BW" in ''|*[!0-9]*) die "--bwlimit is KB/s, a whole number" ;; esac

IP=$(resolve_ip "$TARGET_ARG")
log "target $TARGET_ARG -> $IP, bwlimit ${BW} KB/s, dry-run=$DRY"
RSH="ssh ${SSH_OPTS[*]}"
RSYNC=(ionice -c3 nice -n 10 rsync -az --partial --human-readable --info="stats1,progress2" --bwlimit="$BW" -e "$RSH")
[ "$DRY" = 1 ] && RSYNC+=(--dry-run)

# /workspace is created (and handed to the ssh user) by cloud-init; sudo covers a push that beats it, and $W may sit below it.
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "sudo mkdir -p /workspace && sudo chown $SSH_USER:$SSH_USER /workspace && mkdir -p '$W/code/experiments' '$W/data' '$W/runs' '$W/logs' /workspace/lane" \
  || die "cannot reach $SSH_USER@$IP (box still booting? preempted? see preempt_watch.sh)"

push() { # src dst [extra rsync args]
  local src=$1 dst=$2; shift 2
  log "push $src -> $dst"
  "${RSYNC[@]}" "$@" "$src" "$SSH_USER@$IP:$dst"
}
push "$REPO/experiments/negeig_retrofit/" "$W/code/experiments/negeig_retrofit/" \
  --exclude '__pycache__' --exclude '*.pyc' --exclude 'out/' --exclude '*.pt' --exclude '*.safetensors'
push "$REPO/experiments/negeig_27b/" "$W/code/experiments/negeig_27b/" \
  --exclude '__pycache__' --exclude '*.pyc' --exclude 'out/' --exclude '*.pt' --exclude '*.safetensors'
if [ "$CODE_ONLY" = 1 ]; then
  log "--code-only: data not pushed"
elif [ -n "$DATA_SRC" ] && [ -d "$DATA_SRC" ]; then
  if [ "$WITH_SD" = 1 ]; then
    push "$DATA_SRC/" "$W/data/"
  else
    log "data push skips sd/ (the self-distilled replay comes from sd_sync.sh; --with-sd reuses the rig's copy)"
    push "$DATA_SRC/" "$W/data/" --exclude '/sd/'
  fi
else
  die "no data source (tried --data, NEGEIG_DATA_SRC, $REPO/experiments/negeig_27b/out, /data/ai-ml/models/_runs/negeig-27b/data); code is pushed, data is NOT. Pass --data DIR or --code-only."
fi
if [ -n "$LANE_SRC" ]; then
  [ -d "$LANE_SRC" ] || die "--hebrew-lane $LANE_SRC is not a directory"
  # rsync creates only the last path component; the parents must exist
  [ "$DRY" = 1 ] || ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "mkdir -p /workspace/lane/research"
  push "$LANE_SRC/" "/workspace/lane/research/hebrew-rl-20260923/" \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '*.pt' --exclude '*.safetensors' --exclude '*.parquet.bak'
fi
if [ "$DRY" = 0 ]; then
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "chmod +x '$W/code/experiments/negeig_27b/box/'*.sh 2>/dev/null; du -sh '$W/code' '$W/data' 2>/dev/null"
  log "done ($([ "$CODE_ONLY" = 1 ] && echo code only || echo "code and data from ${DATA_SRC:-none}")). Next on the box: bootstrap.sh (base first), then accept.sh"
fi
