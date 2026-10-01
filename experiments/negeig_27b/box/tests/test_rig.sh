#!/usr/bin/env bash
# CPU tests of rig.sh (bx, bxd, bxs, bxw, bxpull). No provider and no box: a fake `ssh` runs the remote command through a
# local `bash -c` exactly as ssh hands it to a login shell (arguments joined by spaces), and a fake box root is a scratch dir
# (NEGEIG_W) whose code/experiments/negeig_27b is a symlink to this tree, so the real boxenv.sh is what bx sources.
# Red arms: a quoting torture string, a failing command, a label that is refused, a second start while one runs, a restart after
# done, a command that dies without an exit status (kill -9 of the shell), a stale pid file of an unrelated process, a deadline,
# a box that never answers (125), a box that answers late, and a pull that must not delete and must honor the bandwidth cap.
# The test labels are single words; the same names are reused only after the earlier command ended.
# Not covered on purpose: the exit file is written through a temp file and mv so a reader never sees a half-written status. A
# mutant that writes it in place cannot be told apart by a deterministic test (the window is microseconds), so it is documented
# here as an equivalent mutant of rig.sh and not claimed as killed.
# shellcheck disable=SC2016  # the commands handed to bx and bxd are expanded on the box
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RIG=${RIG_SH:-$BOX_DIR/rig.sh}
BW_=$SCRATCH/boxroot
mkdir -p "$BW_/code/experiments" "$SCRATCH/bin" "$SCRATCH/state"
ln -s "$E27_DIR" "$BW_/code/experiments/negeig_27b"
export NEGEIG_W=$BW_ NEGEIG27_STATE=$SCRATCH/state
unset ROOT
mk_fake_ssh "$SCRATCH/bin"
export PATH="$SCRATCH/bin:$PATH"
KIDS=()
cleanup_all() {
  local p; for p in ${KIDS[@]+"${KIDS[@]}"}; do kill -9 "$p" 2>/dev/null; done
  # a pid file is trusted only when its process is one of this test's detached commands (the stale-pid test plants a foreign pid)
  local f p; for f in "$BW_"/logs/*.pid; do
    [ -f "$f" ] || continue
    p=$(cat "$f" 2>/dev/null)
    [ -n "$p" ] && grep -qa "$BW_/logs" "/proc/$p/cmdline" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done
  cleanup_scratch
}
trap cleanup_all EXIT
# shellcheck source=/dev/null
. "$RIG"
H=127.0.0.1
D=$BW_/logs
# state_is OUTPUT STATE LABEL RC MESSAGE
state_is() {
  if printf '%s' "$1" | grep -Eq "STATE $2 label=$3 rc=$4\$"; then ok "$5"; else fail "$5: $1"; fi
}
now() { date +%s.%N; }

section "sourcing"
out=$(sh -c '. "$1"' _ "$RIG" 2>&1); rc=$?
eq "a non-bash shell is refused" "$rc" 1
has "  with the reason" "$out" 'source it from bash'

section "bx"
eq "bx sources boxenv.sh (E27)" "$(bx $H 'echo "$E27"')" "$BW_/code/experiments/negeig_27b"
eq "  ACC" "$(bx $H 'echo "$ACC"')" "$BW_/accept"
bx $H 'exit 7'; eq "the command's exit status comes back" "$?" 7
bx $H 'true'; eq "  zero too" "$?" 0
eq "quotes, spaces and expansions survive" "$(bx $H "echo 'a b'  \"c d\" \$((1+2)) \$(echo x)")" 'a b c d 3 x'
eq "a single quote inside double quotes" "$(bx $H "echo \"it's\"")" "it's"
eq "a multi-line command" "$(bx $H $'echo one\necho two' | tr '\n' ' ')" 'one two '
eq "a here-document" "$(bx $H $'cat <<EOF\nline $((2*3))\nEOF')" 'line 6'
printf 'piped\n' | bx $H 'cat' >/dev/null 2>&1; eq "the box command does not read the rig's stdin" "$(printf 'piped\n' | bx $H 'cat | wc -c')" 0
bx $H >/dev/null 2>&1; eq "bx with nothing to run is a usage error" "$?" 2
bx nosuchbox 'true' >/dev/null 2>&1; eq "an unknown box is an error" "$?" 255

section "bxd, bxs, bxw: the normal run"
t0=$(now)
out=$(bxd $H quick 'sleep 2; echo hello-from-box; echo "E27=$E27"'); rc=$?
t1=$(now)
eq "bxd starts" "$rc" 0
has "  prints the label, pid and log" "$out" "^started quick pid [0-9]+ log $D/quick.log\$"
check "  returns before the command ends (detached)" python3 -c "import sys; sys.exit(0 if $t1 - $t0 < 1.8 else 1)"
out=$(bxs $H quick); rc=$?
eq "bxs while it runs: exit 0" "$rc" 0
state_is "$out" running quick '' "  STATE running"
out=$(bxw $H quick 30 1); rc=$?
eq "bxw returns the command's status" "$rc" 0
has "  the log reached the rig" "$out" 'hello-from-box'
has "  with boxenv.sh sourced in the detached command" "$out" "E27=$BW_/code/experiments/negeig_27b"
state_is "$out" "done" quick 0 "  STATE done rc=0"
eq "  the exit file holds 0" "$(cat "$D/quick.exit")" 0
check "  the script is kept" test -s "$D/quick.cmd"
out=$(bxs $H quick); rc=$?; eq "bxs after done rc 0: exit 0" "$rc" 0

section "a failing command"
bxd $H bad 'echo failing; exit 7' >/dev/null
out=$(bxw $H bad 30 1); rc=$?
eq "bxw returns 7" "$rc" 7
state_is "$out" "done" bad 7 "  STATE done rc=7"
bxs $H bad >/dev/null; eq "bxs returns 7 too" "$?" 7

section "a log that ends without a newline (a tqdm bar ends in \\r)"
bxd $H bar $'printf "Fetching 32 files:   3%%|\\r"; sleep 2; printf "Fetching 32 files: 100%%|"' >/dev/null
out=$(bxs $H bar); rc=$?
eq "bxs while it runs: exit 0" "$rc" 0
state_is "$out" running bar '' "  STATE is on its own line while the bar runs"
out=$(bxw $H bar 30 1 2>&1); rc=$?
eq "bxw returns 0, not 125 (no answer)" "$rc" 0
state_is "$out" "done" bar 0 "  STATE done rc=0"

section "quoting and multi-line scripts through bxd"
bxd $H quoting "echo \"it's \$((1+2))\"; echo 'two words'" >/dev/null; bxw $H quoting 30 1 >/dev/null
eq "single and double quotes" "$(sed -n 1p "$D/quoting.log")" "it's 3"
eq "  second line" "$(sed -n 2p "$D/quoting.log")" "two words"
bxd $H multi $'echo one\necho two\nexit 0' >/dev/null; bxw $H multi 30 1 >/dev/null
eq "a multi-line script runs every line" "$(tr '\n' ' ' <"$D/multi.log")" 'one two '

section "labels"
bxd $H 'bad/label' 'true' >/dev/null 2>&1; eq "a slash is refused" "$?" 2
bxd $H '' 'true' >/dev/null 2>&1; eq "an empty label is refused (usage)" "$?" 2
bxd $H '-x' 'true' >/dev/null 2>&1; eq "a leading dash is refused" "$?" 2
bxd $H 'a b' 'true' >/dev/null 2>&1; eq "a space is refused" "$?" 2
bxd $H 'x;touch pwn' 'true' >/dev/null 2>&1; eq "a shell metacharacter is refused" "$?" 2
check_not "  and nothing ran" test -e "$BW_/logs/x;touch pwn.cmd"
bxd $H only >/dev/null 2>&1; eq "bxd without a command is a usage error" "$?" 2

section "no such label"
out=$(bxs $H nolabel); rc=$?
eq "bxs: exit 2" "$rc" 2
state_is "$out" none nolabel '' "  STATE none"
bxw $H nolabel 5 1 >/dev/null 2>&1; eq "bxw: 2" "$?" 2

section "a second start while one runs, and a restart after done"
bxd $H busy "echo \$\$ >$SCRATCH/busy.child; exec sleep 6" >/dev/null
for _ in $(seq 1 50); do [ -s "$SCRATCH/busy.child" ] && break; sleep 0.1; done
KIDS+=("$(cat "$SCRATCH/busy.child")")
bpid=$(cat "$D/busy.pid")
eq "the detached command leads its own session (it survives the ssh session ending)" "$(ps -o sid= -p "$bpid" | tr -d ' ')" "$bpid"
check_not "  which is not this shell's session" test "$(ps -o sid= -p "$bpid" | tr -d ' ')" = "$(ps -o sid= -p $$ | tr -d ' ')"
check "  the pid file names the command's shell (its cmdline carries the script path)" grep -qa "$D/busy.cmd" "/proc/$bpid/cmdline"
out=$(bxd $H busy 'echo second' 2>&1); rc=$?
eq "refused while it runs" "$rc" 3
has "  says why" "$out" 'still running'
eq "  the running script was not replaced" "$(grep -c second "$D/busy.cmd")" 0
kill "$(cat "$SCRATCH/busy.child")"
bxw $H busy 30 1 >/dev/null 2>&1; rc=$?
eq "  killing the command gives its status (143)" "$rc" 143
bxd $H busy 'echo restarted' >/dev/null; rc=$?
eq "a restart after it ended is accepted" "$rc" 0
bxw $H busy 30 1 >/dev/null
eq "  the new run's log" "$(cat "$D/busy.log")" restarted
check "  the old log is kept as .log.prev" test -f "$D/busy.log.prev"
eq "  the exit file is the new one" "$(cat "$D/busy.exit")" 0

section "a restart after a finished run reports running, not the old status"
bxd $H rst 'exit 7' >/dev/null; bxw $H rst 30 1 >/dev/null 2>&1
eq "the first run ended with 7" "$(cat "$D/rst.exit")" 7
bxd $H rst "echo \$\$ >$SCRATCH/rst.child; exec sleep 6" >/dev/null
for _ in $(seq 1 50); do [ -s "$SCRATCH/rst.child" ] && break; sleep 0.1; done
KIDS+=("$(cat "$SCRATCH/rst.child")")
check_not "the old exit file is gone as soon as the restart is accepted" test -e "$D/rst.exit"
out=$(bxs $H rst); rc=$?
state_is "$out" running rst '' "bxs says running, not done rc=7"
eq "  and exits 0" "$rc" 0
bxd $H rst 'echo third' >/dev/null 2>&1; eq "  a third start is refused while the restarted run lives" "$?" 3
kill "$(cat "$SCRATCH/rst.child")"; bxw $H rst 30 1 >/dev/null 2>&1

section "the rig's stdin is never consumed by bxs and bxw (a real ssh would swallow it)"
bxd $H quick 'echo for-stdin' >/dev/null; bxw $H quick 30 1 >/dev/null
eq "the fake ssh does swallow a stdin it is handed (the test is not vacuous)" \
  "$(printf 'keep\n' | FAKE_SSH_SLURP=1 bash -c '. "$1"; _rig_ssh '"$H"' "cat" | tr -d "\n"' _ "$RIG")" keep
eq "bxs leaves the caller's stdin alone" "$(printf 'keep\n' | { FAKE_SSH_SLURP=1 bxs $H quick >/dev/null 2>&1; cat; })" keep
eq "bxw leaves it alone too" "$(printf 'keep\n' | { FAKE_SSH_SLURP=1 bxw $H quick 30 1 >/dev/null 2>&1; cat; })" keep
eq "bx leaves it alone" "$(printf 'keep\n' | { FAKE_SSH_SLURP=1 bx $H true >/dev/null 2>&1; cat; })" keep
printf 'a b\n' | FAKE_SSH_SLURP=1 bxd $H slurp 'echo ran-with-slurp' >/dev/null; bxw $H slurp 30 1 >/dev/null 2>&1
eq "bxd still delivers its script when ssh reads all of stdin" "$(cat "$D/slurp.log")" ran-with-slurp

section "a command that cannot be started"
mkdir -p "$SCRATCH/failbin"; printf '#!/bin/sh\nexit 1\n' >"$SCRATCH/failbin/setsid"; chmod +x "$SCRATCH/failbin/setsid"
out=$(PATH="$SCRATCH/failbin:$PATH" bxd $H nostart 'true' 2>&1); rc=$?
eq "bxd reports the failure (5) when no pid appears" "$rc" 5
has "  says so" "$out" 'nostart did not start'
hasnt "  and does not print a started line" "$out" '^started'

section "a command that dies with no exit status"
bxd $H dying "echo \$\$ >$SCRATCH/dying.child; exec sleep 60" >/dev/null
for _ in $(seq 1 50); do [ -s "$SCRATCH/dying.child" ] && break; sleep 0.1; done
KIDS+=("$(cat "$SCRATCH/dying.child")")
kill -9 "$(cat "$D/dying.pid")"; sleep 0.3
out=$(bxs $H dying); rc=$?
eq "bxs: dead, exit 1" "$rc" 1
state_is "$out" dead dying '' "  STATE dead"
out=$(bxw $H dying 30 1 2>&1); rc=$?
eq "bxw: 126" "$rc" 126
has "  says so" "$out" 'died with no exit status'
kill -9 "$(cat "$SCRATCH/dying.child")" 2>/dev/null
bxd $H dying 'echo again' >/dev/null; rc=$?
eq "a dead label can be started again" "$rc" 0
bxw $H dying 30 1 >/dev/null; eq "  and ends 0" "$?" 0

section "a stale pid file of an unrelated process (a rebooted box)"
mkdir -p "$D"; echo 'true' >"$D/stale.cmd"; rm -f "$D/stale.exit"
sleep 600 & stalepid=$!; disown "$stalepid"; KIDS+=("$stalepid")   # live, unrelated, and never this shell
echo "$stalepid" >"$D/stale.pid"
out=$(bxs $H stale); rc=$?
state_is "$out" dead stale '' "a live unrelated pid is not mistaken for the command"
bxd $H stale 'echo fresh' >/dev/null; rc=$?
eq "  and bxd may start over it" "$rc" 0
bxw $H stale 30 1 >/dev/null

section "deadline, unreachable box, late box"
bxd $H slow "echo \$\$ >$SCRATCH/slow.child; exec sleep 30" >/dev/null
for _ in $(seq 1 50); do [ -s "$SCRATCH/slow.child" ] && break; sleep 0.1; done
KIDS+=("$(cat "$SCRATCH/slow.child")")
t0=$(now); out=$(bxw $H slow 2 1 2>&1); rc=$?; t1=$(now)
eq "bxw gives up at the deadline: 124" "$rc" 124
has "  says so" "$out" 'still not finished after 2s'
check "  within a few seconds" python3 -c "import sys; sys.exit(0 if $t1 - $t0 < 8 else 1)"
kill -9 "$(cat "$SCRATCH/slow.child")" 2>/dev/null; kill -9 "$(cat "$D/slow.pid")" 2>/dev/null
bxd $H quick 'echo done-already' >/dev/null; bxw $H quick 30 1 >/dev/null
t0=$(now); out=$(FAKE_SSH_DOWN=1 bxw $H quick 100 0 2>&1); rc=$?; t1=$(now)
eq "a box that never answers: 125" "$rc" 125
has "  after 5 polls" "$out" '5 polls in a row'
eq "  five failed polls were reported" "$(printf '%s\n' "$out" | grep -c 'no answer from')" 5
check "  quickly, with a 100 s deadline unused" python3 -c "import sys; sys.exit(0 if $t1 - $t0 < 15 else 1)"
pat_run() {  # pat_run PATTERN LABEL: bxw LABEL with the answer pattern, a box command that waits for the release file
  printf '%s' "$1" >"$SCRATCH/pattern"; rm -f "$SCRATCH/release"
  FAKE_SSH_PATTERN_FILE=$SCRATCH/pattern FAKE_SSH_RELEASE=$SCRATCH/release bxw $H "$2" 60 0.2 2>&1
}
rm -f "$SCRATCH/release"
bxd $H quick 'echo done-again' >/dev/null; bxw $H quick 30 1 >/dev/null
out=$(pat_run ff quick); rc=$?
eq "a box that answers after 2 failures: the command's status" "$rc" 0
eq "  two failed polls were reported" "$(printf '%s\n' "$out" | grep -c 'no answer from')" 2
out=$(pat_run ffff quick); rc=$?
eq "four failures in a row are survived" "$rc" 0
eq "  four failed polls were reported" "$(printf '%s\n' "$out" | grep -c 'no answer from')" 4
rm -f "$SCRATCH/release"
bxd $H gate "while [ ! -e '$SCRATCH/release' ]; do sleep 0.1; done; echo released" >/dev/null
out=$(pat_run ffff.ffff. gate); rc=$?
eq "the failure counter resets on an answer (ffff.ffff. is not five in a row)" "$rc" 0
eq "  eight failed polls were reported" "$(printf '%s\n' "$out" | grep -c 'no answer from')" 8
has "  and a running poll was seen between them" "$out" 'gate running'
rm -f "$SCRATCH/release"
bxd $H gate "while [ ! -e '$SCRATCH/release' ]; do sleep 0.1; done; echo released" >/dev/null
out=$(pat_run fffff gate); rc=$?
eq "five failures in a row: 125" "$rc" 125
touch "$SCRATCH/release"; bxw $H gate 30 0.2 >/dev/null 2>&1
section "bxw prints progress while it waits"
bxd $H prog "echo step-one; sleep 3; echo step-two" >/dev/null
out=$(bxw $H prog 30 1 2>&1)
has "a poll line names the label and the last log line" "$out" 'prog running: step-one'
has "  the final tail has every line" "$out" 'step-two'

section "bxpull"
mkdir -p "$BW_/runs/tag/wide_s0" "$BW_/runs/tag/ctrl_s0"
echo '{"event":"step"}' >"$BW_/runs/tag/wide_s0/log.jsonl"
echo big >"$BW_/runs/tag/wide_s0/ckpt_000050.pt"; echo c >"$BW_/runs/tag/ctrl_s0/log.jsonl"
P=$SCRATCH/pulled
RSYNC_LOG=$SCRATCH/rsync.log bxpull $H runs "$P" >/dev/null 2>&1; rc=$?
eq "a pull works" "$rc" 0
check "  the files arrived" test -s "$P/tag/wide_s0/log.jsonl"
check "  nested too" test -s "$P/tag/ctrl_s0/log.jsonl"
has "  default bandwidth cap 40000" "$SCRATCH/rsync.log" '--bwlimit=40000'
RSYNC_LOG=$SCRATCH/rsync2.log BXPULL_BWLIMIT=123 bxpull $H runs "$P" >/dev/null 2>&1
has "  BXPULL_BWLIMIT is honored" "$SCRATCH/rsync2.log" '--bwlimit=123'
echo local >"$P/only-here.txt"
bxpull $H runs "$P" >/dev/null 2>&1
check "  no --delete: a local-only file stays" test -f "$P/only-here.txt"
rm -f "$BW_/runs/tag/wide_s0/log.jsonl"
bxpull $H runs "$P" >/dev/null 2>&1
check "  and a file removed on the box stays in the pulled copy" test -s "$P/tag/wide_s0/log.jsonl"
echo more >"$BW_/runs/tag/wide_s0/extra.json"
bxpull $H runs "" >/dev/null 2>&1; eq "  an empty destination is a usage error, not a pull into the cwd" "$?" 2
bxpull $H runs "$SCRATCH/pulled2" --exclude='*.pt' >/dev/null 2>&1
check "  extra rsync arguments pass through (exclude)" test ! -e "$SCRATCH/pulled2/tag/wide_s0/ckpt_000050.pt"
check "  and the rest is pulled" test -s "$SCRATCH/pulled2/tag/wide_s0/extra.json"
bxpull $H nosuchdir "$SCRATCH/pulled3" >/dev/null 2>&1; check_not "a missing box directory is an error" test "$?" -eq 0
bxpull $H >/dev/null 2>&1; eq "bxpull with no arguments is a usage error" "$?" 2

finish
