#!/usr/bin/env bash
# sd_sync.sh - RIG side. Moves the self-distilled replay files between the boxes. The boxes cannot reach each other or
# the rig, so the rig is the relay: it PULLS chat_sft.jsonl from the box that ran `sd_box.sh chat` and tool_sft.jsonl
# from the box that ran `sd_box.sh tools`, checks them, and PUSHES both onto every box into $W/data/sd/ (the folder
# run_pair.sh and accept.sh read). Every file lands under a temporary name and is renamed only when complete, so a cut
# transfer never leaves a truncated replay under its final name.
#
#   sd_sync.sh --chat-box <name|id|ip> --tool-box <name|id|ip> [--also <box> ...] [--dest DIR] [--bwlimit KBPS]
#              [--dry-run] [--skip-ledgers]
#   sd_sync.sh --push-only --also <box> [--also <box> ...] [--dest DIR] [--bwlimit KBPS] [--dry-run]
#
# Pull  (box $W/selfdistill -> rig DEST/chat and DEST/tools):  the training file, its receipts (manifest_*.json,
#       filter_report.json, spot_read.jsonl: read the spot sample before training), and unless --skip-ledgers the raw
#       ledgers (translations.jsonl, chat_raw.jsonl, tool_raw.jsonl: they let a filter change be re-run without
#       generating again; the rig keeps them past the teardown).
# Check on the rig: both files non-empty, every line parses as JSON with a messages list; the row counts are printed.
# Push  (rig -> $W/data/sd/ on the chat box, the tool box, and each --also box), then sha256 on the box against the
#       rig's copy. A box that is down is an error (the push is not a best-effort step: a box without the replay cannot
#       start Stage A).
# Ends with `SD_SYNC_OK chat_rows=N tool_rows=M boxes=K` and writes DEST/sd_sync.json (hashes, counts, boxes, time).
# Run it again after the second box finishes, or after a preemption: rsync only moves what differs.
#
# --push-only (a replacement box after a loss: the generators' box is gone, the rig's pulled copy is all that is left):
# no pull and no source box. It pushes DEST/chat/chat_sft.jsonl and DEST/tools/tool_sft.jsonl onto the --also boxes with the
# same check and the same sha256 verification, and it REFUSES unless DEST/sd_sync.json (the receipt of a full run) exists and
# its two hashes equal the files now in DEST. Ends `SD_PUSH_OK chat_rows=N tool_rows=M boxes=K` and adds the boxes to the
# receipt; the receipt's sources stay those of the full run.
#
# Default DEST: $NEGEIG_SD_DEST, else /data/ai-ml/models/_runs/negeig-27b/selfdistill. Load on the rig: ionice idle
# class, nice 10, --bwlimit (default 40000 KB/s), same as push_data.sh.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$HERE/common.sh"

W=${NEGEIG_W:-/workspace/negeig}
DEST=${NEGEIG_SD_DEST:-/data/ai-ml/models/_runs/negeig-27b/selfdistill}
BW=${BWLIMIT_KBPS:-40000}
CHAT_BOX=""; TOOL_BOX=""; ALSO=(); DRY=0; LEDGERS=1; PUSH_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --chat-box) CHAT_BOX=$2; shift 2 ;;
    --tool-box) TOOL_BOX=$2; shift 2 ;;
    --also) ALSO+=("$2"); shift 2 ;;
    --dest) DEST=$2; shift 2 ;;
    --bwlimit) BW=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --skip-ledgers) LEDGERS=0; shift ;;
    --push-only) PUSH_ONLY=1; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if [ "$PUSH_ONLY" = 1 ]; then
  [ -z "$CHAT_BOX$TOOL_BOX" ] || die "--push-only pulls nothing: give it --also boxes only, not --chat-box or --tool-box"
  [ "${#ALSO[@]}" -gt 0 ] || die "usage: sd_sync.sh --push-only --also <box> [--also <box>]..."
else
  [ -n "$CHAT_BOX" ] && [ -n "$TOOL_BOX" ] || die "usage: sd_sync.sh --chat-box <box> --tool-box <box> [--also <box>]... (the same box twice is allowed)"
fi
case "$BW" in ''|*[!0-9]*) die "--bwlimit is KB/s, a whole number" ;; esac
need rsync; need ssh; need python3; need sha256sum

CHAT_IP=""; TOOL_IP=""; TARGETS=()
if [ "$PUSH_ONLY" = 0 ]; then
  CHAT_IP=$(resolve_ip "$CHAT_BOX"); TOOL_IP=$(resolve_ip "$TOOL_BOX")
  TARGETS=("$CHAT_IP"); [ "$TOOL_IP" = "$CHAT_IP" ] || TARGETS+=("$TOOL_IP")
fi
for a in ${ALSO[@]+"${ALSO[@]}"}; do
  ip=$(resolve_ip "$a"); dup=0
  for t in ${TARGETS[@]+"${TARGETS[@]}"}; do [ "$t" != "$ip" ] || dup=1; done
  [ "$dup" = 1 ] || TARGETS+=("$ip")
done
if [ "$PUSH_ONLY" = 1 ]; then
  log "push-only: the rig's copy in $DEST -> ${TARGETS[*]}; bwlimit ${BW} KB/s; dry-run=$DRY"
else
  log "chat from $CHAT_IP, tools from $TOOL_IP, replay to ${TARGETS[*]}; dest $DEST; bwlimit ${BW} KB/s; dry-run=$DRY"
fi

RSH="ssh ${SSH_OPTS[*]}"
RSYNC=(ionice -c3 nice -n 10 rsync -az --human-readable --bwlimit="$BW" -e "$RSH")
# shellcheck disable=SC2029  # the command string is built on the rig on purpose ($W is the box path)
rssh() { local ip=$1; shift; ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "$@"; }

# Every target must answer before anything moves: a push to a dead box fails only at the end otherwise.
for ip in "${TARGETS[@]}"; do
  rssh "$ip" true 2>/dev/null || die "cannot reach $SSH_USER@$ip (ssh failed): nothing was transferred"
done

# The source side: the file must exist and be non-empty before anything moves.
remote_rows() { # ip file -> row count, or fails when the file is missing or empty
  rssh "$1" "test -s '$W/selfdistill/$2' && wc -l < '$W/selfdistill/$2'" 2>/dev/null | tr -d ' \r'
}
if [ "$PUSH_ONLY" = 1 ]; then
  [ -s "$DEST/chat/chat_sft.jsonl" ] && [ -s "$DEST/tools/tool_sft.jsonl" ] \
    || die "--push-only needs $DEST/chat/chat_sft.jsonl and $DEST/tools/tool_sft.jsonl (a full sd_sync.sh run makes them)"
  [ -s "$DEST/sd_sync.json" ] || die "--push-only needs $DEST/sd_sync.json, the receipt of a full run: the rig's copy has not been verified against any box"
else
rows=$(remote_rows "$CHAT_IP" chat_sft.jsonl) || die "$CHAT_IP has no non-empty $W/selfdistill/chat_sft.jsonl: sd_box.sh chat (then finalize) on that box first"
[ "${rows:-0}" -gt 0 ] || die "$CHAT_IP: chat_sft.jsonl has no rows"
log "chat box: chat_sft.jsonl $rows rows"
rows=$(remote_rows "$TOOL_IP" tool_sft.jsonl) || die "$TOOL_IP has no non-empty $W/selfdistill/tool_sft.jsonl: sd_box.sh tools (then finalize) on that box first"
[ "${rows:-0}" -gt 0 ] || die "$TOOL_IP: tool_sft.jsonl has no rows"
log "tool box: tool_sft.jsonl $rows rows"
fi
if [ "$DRY" = 1 ]; then
  if [ "$PUSH_ONLY" = 1 ]; then log "--dry-run: would push the rig's copy in $DEST to ${TARGETS[*]}"
  else log "--dry-run: would pull both files and receipts to $DEST, then push to ${TARGETS[*]}"; fi
  exit 0
fi

mkdir -p "$DEST/chat" "$DEST/tools"
[ "$PUSH_ONLY" = 1 ] || rm -f "$DEST/sd_sync.json"  # a failed run must not leave the previous run's success receipt behind
pull() { # ip remote-name dest-dir required(1|0)
  local ip=$1 name=$2 dir=$3 req=$4
  if ! rssh "$ip" "test -s '$W/selfdistill/$name'" 2>/dev/null; then
    [ "$req" = 0 ] || die "$ip: $W/selfdistill/$name is missing"
    return 0
  fi
  log "pull $ip:$name -> $dir"
  "${RSYNC[@]}" "$SSH_USER@$ip:$W/selfdistill/$name" "$dir/$name" || die "pull of $name from $ip failed"
}
if [ "$PUSH_ONLY" = 0 ]; then
  pull "$CHAT_IP" chat_sft.jsonl "$DEST/chat" 1
  for f in manifest_chat.json filter_report.json spot_read.jsonl; do pull "$CHAT_IP" "$f" "$DEST/chat" 0; done
  pull "$TOOL_IP" tool_sft.jsonl "$DEST/tools" 1
  for f in manifest_tools.json filter_report.json spot_read.jsonl; do pull "$TOOL_IP" "$f" "$DEST/tools" 0; done
  if [ "$LEDGERS" = 1 ]; then
    for f in translations.jsonl chat_raw.jsonl; do pull "$CHAT_IP" "$f" "$DEST/chat" 0; done
    pull "$TOOL_IP" tool_raw.jsonl "$DEST/tools" 0
  fi
fi

# Check on the rig: a replay file that does not parse would kill train27 at its first step.
count_rows() { # file -> row count; exits 1 with a message when a line is not JSON or has no messages
  python3 - "$1" <<'PYEOF'
import json, sys
path, n = sys.argv[1], 0
with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as e:
            sys.exit(f"{path}:{i}: not JSON ({e})")
        if not isinstance(row, dict) or not isinstance(row.get("messages"), list) or not row["messages"]:
            sys.exit(f"{path}:{i}: no messages list")
        n += 1
if n == 0:
    sys.exit(f"{path}: no rows")
print(n)
PYEOF
}
CHAT_ROWS=$(count_rows "$DEST/chat/chat_sft.jsonl") || die "chat_sft.jsonl from ${CHAT_IP:-the rig copy} is not a valid replay file"
TOOL_ROWS=$(count_rows "$DEST/tools/tool_sft.jsonl") || die "tool_sft.jsonl from ${TOOL_IP:-the rig copy} is not a valid replay file"
CHAT_SHA=$(sha256sum "$DEST/chat/chat_sft.jsonl" | cut -d' ' -f1)
TOOL_SHA=$(sha256sum "$DEST/tools/tool_sft.jsonl" | cut -d' ' -f1)
if [ "$PUSH_ONLY" = 1 ]; then
  # The receipt of the full run holds the hashes the original boxes were verified against: the copy now in DEST must be that copy.
  WANT=$(python3 - "$DEST/sd_sync.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(d["chat_sha256"], d["tool_sha256"])
PYEOF
  ) || die "$DEST/sd_sync.json is not a readable receipt"
  [ "$WANT" = "$CHAT_SHA $TOOL_SHA" ] \
    || die "the rig's copy in $DEST differs from its receipt $DEST/sd_sync.json (receipt: $WANT, files: $CHAT_SHA $TOOL_SHA): restore the files or run a full sd_sync.sh"
fi
log "checked on the rig: chat $CHAT_ROWS rows ($CHAT_SHA), tool $TOOL_ROWS rows ($TOOL_SHA)"

# Push to every box. --partial-dir (not --partial): an interrupted copy stays in a hidden folder, never under the final name.
PUSH=(ionice -c3 nice -n 10 rsync -az --human-readable --partial-dir=.rsync-partial --bwlimit="$BW" -e "$RSH")
for ip in "${TARGETS[@]}"; do
  rssh "$ip" "mkdir -p '$W/data/sd'" || die "cannot reach $SSH_USER@$ip"
  log "push chat_sft.jsonl and tool_sft.jsonl -> $ip:$W/data/sd/"
  "${PUSH[@]}" "$DEST/chat/chat_sft.jsonl" "$DEST/tools/tool_sft.jsonl" "$SSH_USER@$ip:$W/data/sd/" || die "push to $ip failed"
  got=$(rssh "$ip" "cd '$W/data/sd' && sha256sum chat_sft.jsonl tool_sft.jsonl" | awk '{print $1}' | tr '\n' ' ')
  [ "$got" = "$CHAT_SHA $TOOL_SHA " ] || die "$ip: sha256 of the pushed files differs from the rig's copy (got: $got)"
  log "$ip: both files verified by sha256"
done

python3 - "$DEST/sd_sync.json" "$PUSH_ONLY" "$CHAT_IP" "$TOOL_IP" "$CHAT_ROWS" "$TOOL_ROWS" "$CHAT_SHA" "$TOOL_SHA" "${TARGETS[@]}" <<'PYEOF'
import json, os, sys, time
out, push_only, chat_ip, tool_ip, cr, tr, cs, ts, *boxes = sys.argv[1:]
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
if push_only == "1":
    doc = json.load(open(out))
    doc["boxes"] = sorted(set(doc["boxes"]) | set(boxes))
    doc["push_only_at_utc"] = now
else:
    doc = {"at_utc": now, "chat_from": chat_ip, "tools_from": tool_ip,
           "chat_rows": int(cr), "tool_rows": int(tr), "chat_sha256": cs, "tool_sha256": ts, "boxes": boxes}
with open(out + ".tmp", "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
os.replace(out + ".tmp", out)
PYEOF
if [ "$PUSH_ONLY" = 1 ]; then echo "SD_PUSH_OK chat_rows=$CHAT_ROWS tool_rows=$TOOL_ROWS boxes=${#TARGETS[@]}"
else echo "SD_SYNC_OK chat_rows=$CHAT_ROWS tool_rows=$TOOL_ROWS boxes=${#TARGETS[@]}"; fi
