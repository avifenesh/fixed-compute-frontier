#!/usr/bin/env bash
# CPU tests of push_data.sh (RIG side, rig -> box). The real rsync runs; a fake ssh is the transport and the "box" is a scratch
# tree (W under $SCRATCH, and /workspace rewritten to $SCRATCH/ws so nothing outside the scratch dir is touched).
# What this guards (reviewer finding D1): the rig's prepared data set holds the LIVE lane's sd/chat_sft.jsonl and
# sd/tool_sft.jsonl. A plain data push put them on a fresh box before its own self-distillation ran, so accept.sh read
# replay=present, run_pair.sh found a replay, and the "no replay, stop" guard never fired. Red arms: a data dir with sd/ at
# the top (must not reach the box), a nested directory also named sd (must arrive: the exclude is anchored), --with-sd
# (must arrive), a box that already holds its own sd/ (must survive a re-push untouched), --code-only (no data at all),
# a missing data source (an error), and the other data (s1, s4) which must always arrive.
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
PUSH=${PUSH_SH:-$BOX_DIR/push_data.sh}
mk_box push
export NEGEIG27_STATE=$B/state
mkdir -p "$B/bin" "$B/state"
mk_fake_ssh "$B/bin"
# The fake ssh of lib.sh runs the remote command locally. Wrap it: /workspace is rewritten into the scratch tree and `sudo`
# (used to create /workspace) is a no-op, so the test never needs root and never writes outside $SCRATCH.
mv "$B/bin/ssh" "$B/bin/ssh.real"
cat >"$B/bin/ssh" <<EOS
#!/usr/bin/env bash
args=()
for a in "\$@"; do args+=("\${a//\/workspace/$B/ws}"); done
exec "$B/bin/ssh.real" "\${args[@]}"
EOS
printf '#!/usr/bin/env bash\nexit 0\n' >"$B/bin/sudo"
chmod +x "$B/bin/ssh" "$B/bin/sudo"
export PATH="$B/bin:$PATH"
export RSYNC_LOG=$B/rsync.log FAKE_SSH_LOG=$B/ssh.log

# the fake repo: both code trees (push_data.sh refuses a repo without them)
REPO=$B/repo; mkdir -p "$REPO/experiments/negeig_retrofit" "$REPO/experiments/negeig_27b/box"
echo "code" >"$REPO/experiments/negeig_retrofit/a.py"; echo "code" >"$REPO/experiments/negeig_27b/train27.py"
# the rig's data set, shaped like /data/ai-ml/models/_runs/negeig-27b/data, with the live lane's replay in sd/
DATA=$B/rigdata; mkdir -p "$DATA/s1/hebrew" "$DATA/s1/sd" "$DATA/s4" "$DATA/sd/chat" "$DATA/sd/tool"
echo "s1 row" >"$DATA/s1/hebrew/train.jsonl"; echo "nested sd dir is data, not the replay" >"$DATA/s1/sd/keep.jsonl"
echo "lm replay" >"$DATA/s4/replay_clean_q38.pt"
echo "LIVE LANE chat" >"$DATA/sd/chat_sft.jsonl"; echo "LIVE LANE tool" >"$DATA/sd/tool_sft.jsonl"; echo raw >"$DATA/sd/chat/raw.jsonl"

IP=10.9.8.7
fresh() { rm -rf "$B/ws" "$B/w"; mkdir -p "$B/ws"; export NEGEIG_W=$B/w; : >"$RSYNC_LOG"; }
push() { "$PUSH" "$IP" --repo "$REPO" "$@" >"$B/push.out" 2>&1; RC=$?; }
W_=$B/w

section "P1 a default data push leaves the live lane's replay behind"
fresh; push --data "$DATA"
eq "push exits 0" "$RC" 0
check "s1 data arrived" test -s "$W_/data/s1/hebrew/train.jsonl"
check "s4 LM replay arrived" test -s "$W_/data/s4/replay_clean_q38.pt"
check_not "sd/chat_sft.jsonl did NOT arrive" test -e "$W_/data/sd/chat_sft.jsonl"
check_not "sd/tool_sft.jsonl did NOT arrive" test -e "$W_/data/sd/tool_sft.jsonl"
check_not "no sd/ raw ledger directory arrived either" test -e "$W_/data/sd/chat"
check "a NESTED directory named sd is data and arrived (the exclude is anchored)" test -s "$W_/data/s1/sd/keep.jsonl"
check "the code trees arrived" test -s "$W_/code/experiments/negeig_27b/train27.py"
has "the log says sd/ is skipped" "$B/push.out" "data push skips sd/"
has "the rsync call carries the anchored exclude" "$RSYNC_LOG" "--exclude /sd/"

section "P2 --with-sd is the deliberate opt-in"
fresh; push --data "$DATA" --with-sd
eq "push exits 0" "$RC" 0
eq "chat replay arrived byte for byte" "$(cat "$W_/data/sd/chat_sft.jsonl" 2>/dev/null)" "LIVE LANE chat"
eq "tool replay arrived byte for byte" "$(cat "$W_/data/sd/tool_sft.jsonl" 2>/dev/null)" "LIVE LANE tool"
hasnt "no skip line" "$B/push.out" "data push skips sd/"
hasnt "the rsync call has no sd exclude" "$RSYNC_LOG" "--exclude /sd/"

section "P3 a box that holds its own replay keeps it through a re-push (preemption restart)"
fresh; push --data "$DATA"
mkdir -p "$W_/data/sd"; echo "THIS BOX'S OWN chat" >"$W_/data/sd/chat_sft.jsonl"; echo "THIS BOX'S OWN tool" >"$W_/data/sd/tool_sft.jsonl"
push --data "$DATA"
eq "re-push exits 0" "$RC" 0
eq "own chat replay untouched" "$(cat "$W_/data/sd/chat_sft.jsonl")" "THIS BOX'S OWN chat"
eq "own tool replay untouched" "$(cat "$W_/data/sd/tool_sft.jsonl")" "THIS BOX'S OWN tool"

section "P4 --code-only moves no data at all"
fresh; push --data "$DATA" --code-only
eq "push exits 0" "$RC" 0
check "code arrived" test -s "$W_/code/experiments/negeig_27b/train27.py"
check_not "no s1 data" test -e "$W_/data/s1/hebrew/train.jsonl"
check_not "no sd" test -e "$W_/data/sd/chat_sft.jsonl"

section "P5 refusals"
fresh; push --data "$B/nowhere"
eq "a data dir that does not exist is an error" "$RC" 1
has "  that says the data was NOT pushed" "$B/push.out" "no data source .*data is NOT"
fresh; "$PUSH" >"$B/push.out" 2>&1; RC=$?
eq "no target is a usage error" "$RC" 1
has "  naming the usage" "$B/push.out" "usage: push_data.sh"
fresh; "$PUSH" "$IP" --repo "$REPO" --bogus >"$B/push.out" 2>&1; RC=$?
eq "an unknown argument is refused" "$RC" 1
fresh; "$PUSH" "$IP" --repo "$REPO" --data "$DATA" --bwlimit 4x >"$B/push.out" 2>&1; RC=$?
eq "a non-numeric --bwlimit is refused" "$RC" 1
has "  with the reason" "$B/push.out" "whole number"
helptext=$("$PUSH" --help 2>&1)
has "--help documents --with-sd" "$helptext" "--with-sd"

finish
