#!/usr/bin/env bash
# CPU tests of RUNBOOK.md. Three layers, all against the real file (RUNBOOK_MD overrides it, for the mutation checks):
#   1. text: the file exists, is ASCII, has no em-dash substitutes or banned phrases, keeps its section skeleton.
#   2. static: every fenced block is bash and passes `bash -n` (shellcheck at error severity when installed); runbook_lint.py
#      proves every script, flag, subcommand, receipt line and env name the runbook names exists in the tree, and that no block
#      tears down without --ids, names a live-lane box, kills by name or sleeps in the foreground. Red arms: the same lint run
#      against a scratch copy of the tree with one script, flag, dispatch, receipt or env name removed, and against runbook
#      copies with one bad block appended.
#   3. stub box: the marked blocks (setup, abort, sweep, evals, evalcalls) are RUN. The rig is this shell, the box is a scratch
#      dir behind a fake ssh (a local `bash -c`), the trainer is stub_torchrun.py, `arm` goes through the real ctl.sh and the
#      real resume_wrapper.sh and run_pair.sh, the evals scripts are fakes that log their arguments. What this proves: the
#      quoting of every box command, which side expands which variable, the checkpoint move-aside arithmetic, what pull_logs
#      keeps, and the argv each eval gets. What it cannot prove is in RUNBOOK.md, "What only a real box proves".
# shellcheck disable=SC2016  # the commands handed to bx and the scenario scripts are expanded where they run
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUNBOOK=${RUNBOOK_MD:-$E27_DIR/RUNBOOK.md}
LINT=${RUNBOOK_LINT:-$TESTS_DIR/runbook_lint.py}
ROOT_REAL=${RUNBOOK_ROOT:-$E27_DIR}
export NO_SYSTEMD=1 SKIP_GPU_CHECK=1
RB=$SCRATCH/rb; mkdir -p "$RB"
lint() { python3 "$LINT" "$@"; }

section "text"
check "RUNBOOK.md exists" test -s "$RUNBOOK"
check_not "ASCII only (no em dash, en dash or curly quote)" grep -qP '[^\x00-\x7F]' "$RUNBOOK"
check_not "no ' -- ' used as a dash" grep -qE ' -- ' "$RUNBOOK"
check_not "no line ends in ' --'" grep -qE ' --$' "$RUNBOOK"
check_not "no banned phrase" grep -qiE "happy to|worth noting|delve|leverage|next session|fresh session|pre-existing|preexisting|tomorrow|stop for the day" "$RUNBOOK"
heads=$(grep '^## ' "$RUNBOOK")
for h in '## 0. Rules' '## 1. Rig setup' '## 2. Create the boxes' '## 3. Prepare every box' '## 4. Self-distillation' '## 5. Parity' \
         '## 6. Throughput decision' '## 7. Arm the LR sweep' '## 8. The step-300 decision' '## 9. Stage A to 700' \
         '## 10. Receipts and teardown' '## Appendix A. Preemption' '## Appendix B. When a line does not come' \
         '## Appendix C. Who does what' '## What only a real box proves'; do
  has "heading: $h" "$heads" "^$(printf '%s' "$h" | sed 's/[.]/[.]/g')"
done
eq "section numbers run 0..10 in order" "$(grep -oE '^## [0-9]+\.' "$RUNBOOK" | tr -d '#. \n' )" "012345678910"

section "blocks"
nblocks=$(lint extract "$RUNBOOK" "$RB/x" 2>"$RB/extract.err"); erc=$?
eq "every fence is a terminated bash fence" "$erc" 0
check "there are fenced blocks (not a vacuous lint)" test "${nblocks:-0}" -ge 20
marked=$(cd "$RB/x" && /bin/ls -1 ./*.block 2>/dev/null | sed 's|^\./||; s|\.block$||' | sort | tr '\n' ' ')
eq "the marked blocks the stub box runs" "$marked" "abort evalcalls evals prepare setup sweep "
bad=0; for f in "$RB"/x/[0-9]*.sh; do bash -n "$f" 2>"$RB/syn.err" || { bad=$((bad + 1)); echo "    $(basename "$f"): $(head -n1 "$RB/syn.err")"; }; done
eq "bash -n passes on every block" "$bad" 0
if command -v shellcheck >/dev/null 2>&1; then
  bad=0; for f in "$RB"/x/[0-9]*.sh; do shellcheck -s bash -S error "$f" >"$RB/sc.out" 2>&1 || { bad=$((bad + 1)); head -n6 "$RB/sc.out"; }; done
  eq "shellcheck -S error passes on every block" "$bad" 0
else
  ok "shellcheck not installed: skipped"
fi
eq "exactly one RECV= line in the setup block (the test redirects it)" "$(grep -c '^RECV=' "$RB/x/setup.block")" 1
has "box names match the create_box.sh pattern" "$(grep -E '^A=' "$RB/x/setup.block")" '^A=negeig27-r2-a; B=negeig27-r2-b'
eq "  and no block names a box of the live lane" "$(grep -l 'h200p' "$RB"/x/[0-9]*.sh 2>/dev/null | wc -l | tr -d ' ')" 0

section "refs and safety on the real runbook"
out=$(lint refs "$RUNBOOK" "$ROOT_REAL" 2>&1); rc=$?
eq "refs: every script, flag, subcommand, receipt and env name exists" "$rc" 0
refs_n() { printf '%s\n' "$out" | sed -n "s/^REFS .*$1=\([0-9]*\).*/\1/p"; }
check "  the counts show it read the runbook (scripts >= 50, flags >= 40, subcommands >= 20, receipts >= 15)" \
  test "$(refs_n scripts)" -ge 50 -a "$(refs_n flags)" -ge 40 -a "$(refs_n subcommands)" -ge 20 -a "$(refs_n receipts)" -ge 15
[ "$rc" -eq 0 ] || printf '%s\n' "$out" | grep PROBLEM | head -n 10
out=$(lint safety "$RUNBOOK" 2>&1); rc=$?
eq "safety: no teardown without --ids, no live box, no kill by name, no long sleep" "$rc" 0
has "  reported clean" "$out" '^SAFETY problems=0$'

section "lint red arms: the tree"
mk_root() { # NAME -> RDIR: a scratch copy of the files the lint reads
  RDIR=$SCRATCH/root_$1; rm -rf "$RDIR"; mkdir -p "$RDIR/box" "$RDIR/evals" "$RDIR/selfdistill"
  cp "$ROOT_REAL"/box/*.sh "$ROOT_REAL"/box/*.py "$ROOT_REAL"/box/*.example "$RDIR/box/" 2>/dev/null
  cp "$ROOT_REAL"/evals/*.sh "$ROOT_REAL"/evals/*.py "$RDIR/evals/" 2>/dev/null
  cp -r "$ROOT_REAL"/selfdistill/. "$RDIR/selfdistill/" 2>/dev/null
  cp "$ROOT_REAL"/*.sh "$ROOT_REAL"/*.py "$RDIR/" 2>/dev/null
  return 0
}
red() { # description ROOTDIR expected-regex   (refs must fail with that PROBLEM line)
  local out rc; out=$(lint refs "$RUNBOOK" "$2" 2>&1); rc=$?
  if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -Eq -- "$3"; then ok "$1"; else fail "$1: rc=$rc, want a line /$3/: $(printf '%s' "$out" | grep PROBLEM | head -n2)"; fi
}
mk_root clean
out=$(lint refs "$RUNBOOK" "$RDIR" 2>&1); eq "the scratch copy of the tree is clean (the arms below start from green)" "$?" 0
mk_root noscript; mv "$RDIR/box/sweep_read.py" "$RDIR/box/sweep_read2.py"
red "a renamed script is reported" "$RDIR" 'PROBLEM no such script: .*sweep_read\.py'
mk_root noflag; sed -i 's/--ckpt-dir/--ckptdir/g' "$RDIR/box/ctl.sh"
red "a renamed flag is reported against its script" "$RDIR" 'PROBLEM ctl\.sh has no --ckpt-dir'
mk_root norecpt; grep -rl 'BOOTSTRAP_OK' "$RDIR" | xargs sed -i 's/BOOTSTRAP_OK/BOOTSTRAP_YES/g'
red "a renamed receipt line is reported" "$RDIR" 'PROBLEM receipt BOOTSTRAP_OK is named'
mk_root noenv; grep -rl 'ALLOW_NO_REPLAY' "$RDIR" | xargs sed -i 's/ALLOW_NO_REPLAY/ALLOW_NO_RPLY/g'
red "a renamed env switch is reported" "$RDIR" 'PROBLEM env ALLOW_NO_REPLAY is named'
mk_root nodyn; sed -i 's/SD_\${name^^}_EXIT/SD_X_EXIT/' "$RDIR/box/sd_box.sh"
red "a receipt a script builds from parts is checked too" "$RDIR" 'PROBLEM receipt SD_CHAT_EXIT: box/sd_box.sh no longer builds it'
mk_root nosub; sed -i -E 's/\bfinalize\b/finalise/g' "$RDIR/box/sd_box.sh"
red "a subcommand the script no longer dispatches is reported" "$RDIR" "PROBLEM sd_box\.sh has no 'finalize' command"
mk_root noevals; rm -f "$RDIR/evals/run_general.sh"
red "a missing evals script is reported" "$RDIR" 'PROBLEM no such script: .*run_general\.sh'

section "lint red arms: the runbook"
mutant() { # NAME 'text appended as one more bash block'  -> $RB/m_NAME.md
  { cat "$RUNBOOK"; printf '\n```bash\n%s\n```\n' "$2"; } >"$RB/m_$1.md"
}
sred() { # NAME description regex   (safety must fail with it)
  local out rc; out=$(lint safety "$RB/m_$1.md" 2>&1); rc=$?
  if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -Eq -- "$3"; then ok "$2"; else fail "$2: rc=$rc: $out"; fi
}
rred() { # NAME description regex   (refs must fail with it)
  local out rc; out=$(lint refs "$RB/m_$1.md" "$ROOT_REAL" 2>&1); rc=$?
  if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -Eq -- "$3"; then ok "$2"; else fail "$2: rc=$rc: $out"; fi
}
mutant td 'bash "$BOX/teardown.sh" --yes';              sred td "teardown.sh with no --ids is refused" 'teardown\.sh without --ids'
mutant td2 'bash "$BOX/teardown.sh" --pull --yes';      sred td2 "teardown.sh --pull with no --ids is refused" 'teardown\.sh without --ids'
mutant live 'bx negeig27-h200p-a "ls"';                 sred live "a box of the live lane is refused" 'live lane'
mutant kill 'pkill -f train27';                         sred kill "kill by name is refused" 'kills by name'
mutant kill2 'killall python3';                         sred kill2 "killall is refused" 'kills by name'
mutant slp 'sleep 600';                                 sred slp "a long foreground sleep is refused" 'long foreground sleep'
mutant okb 'bash "$BOX/teardown.sh" --list'
out=$(lint safety "$RB/m_okb.md" 2>&1); eq "  a --list teardown is fine (the guard is not blanket)" "$?" 0
mutant nof 'bash "$BOX/no_such_script.sh" --x';         rred nof "an unknown script in a new block is reported" 'PROBLEM no such script: .*no_such_script\.sh'
mutant badflag 'bash "$BOX/ctl.sh" status --no-such-flag'; rred badflag "an unknown flag in a new block is reported" 'PROBLEM ctl\.sh has no --no-such-flag'
mutant badrec 'echo FAKE_THING_OK';                     rred badrec "a receipt no script prints is reported" 'PROBLEM receipt FAKE_THING_OK'
{ cat "$RUNBOOK"; printf '\n```\nplain\n```\n'; } >"$RB/m_nolang.md"
out=$(lint extract "$RB/m_nolang.md" "$RB/y" 2>&1); rc=$?
eq "a fence with no language is refused by extract" "$rc" 1
has "  with the reason" "$out" 'every runbook block is ```bash'
{ cat "$RUNBOOK"; printf '\n```bash\necho open\n'; } >"$RB/m_open.md"
out=$(lint extract "$RB/m_open.md" "$RB/y" 2>&1); eq "an unterminated fence is refused by extract" "$?" 1
{ cat "$RUNBOOK"; printf '\n```bash\nif true; then\n```\n'; } >"$RB/m_syn.md"
lint extract "$RB/m_syn.md" "$RB/z" >/dev/null 2>&1
check_not "a block with a syntax error fails bash -n (the arm the real check relies on)" bash -n "$RB/z/$(printf '%02d' "$((nblocks + 1))").sh"

# ---------------------------------------------------------------------------------------------------------------------
# The stub box. mk_rbox NAME: mk_box plus the code tree the box scripts expect ($W/code/experiments/negeig_27b: links to the
# real run_pair.sh, train27.py and box/, a REAL evals dir holding fakes that log their arguments), the decision file accept.sh
# writes, and a fake torchrun for the evals that logs its working directory and argv.
section "stub box"
mk_fake_ssh "$SCRATCH/bin"
export PATH="$SCRATCH/bin:$PATH"
export NEGEIG27_STATE=$SCRATCH/state; mkdir -p "$NEGEIG27_STATE"
mk_rbox() {
  mk_box "$1"
  local C=$B/code/experiments/negeig_27b
  mkdir -p "$C/evals" "$B/accept"
  ln -s "$E27_DIR/box" "$C/box"; ln -s "$E27_DIR/run_pair.sh" "$C/run_pair.sh"; ln -s "$E27_DIR/train27.py" "$C/train27.py"
  cat >"$C/evals/serve_arm.sh" <<'EOS'
#!/usr/bin/env bash
# fake: log the call, fail an `up` when FAKE_UP_RC is set
echo "serve_arm cwd=$PWD $*" >>"$NEGEIG_W/evals.log"
[ "$1" = up ] && [ -n "${FAKE_UP_RC:-}" ] && exit "$FAKE_UP_RC"
exit 0
EOS
  cat >"$C/evals/run_general.sh" <<'EOS'
#!/usr/bin/env bash
echo "run_general cwd=$PWD $*" >>"$NEGEIG_W/evals.log"
exit "${FAKE_RG_RC:-0}"
EOS
  # the evals run torchrun eval27.py: the stub trainer would try to train, so this one only records
  rm -f "$B/venv/bin/torchrun"
  cat >"$B/venv/bin/torchrun" <<'EOS'
#!/usr/bin/env bash
{ printf '%s' "$PWD"; printf '|%s' "$@"; echo; } >>"$NEGEIG_W/torchrun.log"
EOS
  chmod +x "$C/evals/serve_arm.sh" "$C/evals/run_general.sh" "$B/venv/bin/torchrun"
  echo 'GRAD_CKPT_FLAGS="--grad_ckpt 1"' >"$B/accept/grad_ckpt.env"
  : >"$B/evals.log"; : >"$B/torchrun.log"
  C27=$C
}
# the training arms need the stub trainer as torchrun again
use_stub_trainer() {
  rm -f "$B/venv/bin/torchrun"
  printf '#!/usr/bin/env bash\nexec python3 "%s/stub_torchrun.py" "$@"\n' "$TESTS_DIR" >"$B/venv/bin/torchrun"; chmod +x "$B/venv/bin/torchrun"
}

# the setup block with its one absolute-path side effect (mkdir of the receive dir) pointed at the scratch dir
RB_SETUP=$RB/setup.t
sed "s|^RECV=[^ ]*|RECV=$SCRATCH/recv|" "$RB/x/setup.block" >"$RB_SETUP"
eq "only the RECV= line differs from the real setup block" "$(diff "$RB/x/setup.block" "$RB_SETUP" | grep -c '^<')" 1
export RB_SETUP E27_DIR SCRATCH RB
SCN_TIMEOUT=${SCN_TIMEOUT:-300}
scn() { # NAME: the scenario body comes on stdin; it runs in bash in the checkout after the setup block, A and B pointing at the box
  local f=$RB/scn_$1.sh
  { echo 'set -u'; echo 'cd "$E27_DIR"'; echo '. "$RB_SETUP"'; echo 'A=127.0.0.1; B=127.0.0.1'
    echo 'sleep() { command sleep 0.2; }'; cat; } >"$f"
  SOUT=$(timeout "$SCN_TIMEOUT" bash "$f" 2>&1); SRC=$?
  printf '%s\n' "$SOUT" >"$RB/scn_$1.out"
}
wrapper() { timeout 120 bash "$BOX_DIR/resume_wrapper.sh" >"$B/wrap.out" 2>&1; WRC=$?; }

section "setup block"
mk_rbox su
cat >"$NEGEIG27_STATE/instances.tsv" <<EOS
negeig27-r2-a	id-aaa	proj	b300	uk-south1	1	2026-10-01T00:00:00Z
negeig27-r2-b	id-bbb	proj	b300	uk-south1	1	2026-10-01T00:00:00Z
negeig27-r2	id-short	proj	b300	uk-south1	1	2026-10-01T00:00:00Z
EOS
scn setup <<'EOS'
type bx bxd bxs bxw bxpull arm idof >/dev/null || { echo MISSING_FUNCTION; exit 3; }
echo "A_ID=$(idof negeig27-r2-a) B_ID=$(idof negeig27-r2-b) SHORT_ID=$(idof negeig27-r2) NONE_ID=[$(idof negeig27-nope)]"
echo "E27R=$E27R"; echo "BOX=$BOX"; echo "RECV=$RECV"; test -d "$RECV" && echo RECV_MADE
EOS
eq "the setup block sources and defines bx bxd bxs bxw bxpull arm idof" "$SRC" 0
has "  idof returns the id of an exact name (a prefix of another name does not match)" "$SOUT" 'A_ID=id-aaa B_ID=id-bbb SHORT_ID=id-short NONE_ID=\[\]$'
has "  E27R is this tree" "$SOUT" "E27R=$E27_DIR\$"
has "  BOX is its box dir" "$SOUT" "BOX=$E27_DIR/box\$"
has "  RECV was made under the scratch dir" "$SOUT" 'RECV_MADE'

section "arm: the command that reaches the box"
mk_rbox arm1; use_stub_trainer
scn arm1 <<'EOS'
arm "$B" sweep-lr1e-3 0 1e-3
EOS
eq "arm returns 0" "$SRC" 0
has "  ctl.sh armed the unit" "$SOUT" 'armed: .*/run/train\.cmd'
check "  train.cmd exists on the box" test -s "$B/run/train.cmd"
has "  it sources the decision file at start time (not at arm time)" "$B/run/train.cmd" '^\. "\$ACC/grad_ckpt\.env"; +exec bash "\$E27/run_pair\.sh" sweep-lr1e-3 0 --lr 1e-3 --steps 700 \$GRAD_CKPT_FLAGS *$'
has "  run.env keeps the checkpoint dir of the tag" "$B/run/run.env" "CKPT_DIR=$B/runs/sweep-lr1e-3"
wrapper
eq "the real wrapper runs it to done" "$WRC" 0
check "  done marker" test -f "$B/run/done"
eq "  wide launched once" "$(launches wide)" 1
eq "  ctrl launched once" "$(launches ctrl)" 1
eq "  8 GPUs per arm is NOT the layout of a pair: 4 per arm" "$(stub_field wide 0 nproc)" 4
for tok in '--lr 1e-3' '--steps 700' '--grad_ckpt 1' '--resume'; do has "  argv has $tok" "$(stub_field wide 0 argv)" "(^| )$tok( |$)"; done
hasnt "  argv has no stray empty argument" "$(stub_field wide 0 argv)" '  '
"$BOX_DIR/ctl.sh" status >"$B/status.out" 2>&1
has "  ctl.sh status says done" "$B/status.out" '^state: done'

section "arm: ARM_ENV, steps, seed, ARM_FLAGS, the decision file"
mk_rbox arm2; use_stub_trainer
scn arm2 <<'EOS'
ARM_ENV='ARMS=wide GPUS_PER_RUN=8' arm "$B" stageA-s1 1 1e-4 450
EOS
eq "arm with ARM_ENV and an explicit step count" "$SRC" 0
has "  train.cmd carries the env prefix" "$B/run/train.cmd" 'ARMS=wide GPUS_PER_RUN=8 exec bash "\$E27/run_pair\.sh" stageA-s1 1 --lr 1e-4 --steps 450 '
wrapper
eq "  wrapper done" "$WRC" 0
eq "  only the wide arm launched" "$(launches wide)/$(launches ctrl)" "1/0"
eq "  on all 8 GPUs" "$(stub_field wide 0 nproc)" 8
has "  --steps 450 reached the trainer" "$(stub_field wide 0 argv)" '(^| )--steps 450( |$)'
check "  the arm dir carries the seed" test -f "$B/runs/stageA-s1/wide_s1/complete.json"

mk_rbox arm3; use_stub_trainer
scn arm3 <<'EOS'
ARM_FLAGS='--lr_override' arm "$B" sweep-lr1e-3 0 5e-4
EOS
wrapper
eq "ARM_FLAGS=--lr_override: wrapper done" "$WRC" 0
has "  the flag reached the trainer" "$(stub_field wide 0 argv)" '(^| )--lr_override( |$)'
has "  with the new LR" "$(stub_field wide 0 argv)" '(^| )--lr 5e-4( |$)'
has "  and once per arm" "$(stub_field ctrl 0 argv)" '(^| )--lr_override( |$)'

mk_rbox arm4; use_stub_trainer
echo 'GRAD_CKPT_FLAGS="--grad_ckpt 0"' >"$B/accept/grad_ckpt.env"
scn arm4 <<'EOS'
arm "$B" sweep-lr1e-3 0 1e-3
EOS
wrapper
has "a decision file changed after arm time is read at start time (--grad_ckpt 0)" "$(stub_field wide 0 argv)" '(^| )--grad_ckpt 0( |$)'

mk_rbox arm5; use_stub_trainer
rm -f "$B/accept/grad_ckpt.env"
scn arm5 <<'EOS'
arm "$B" sweep-lr1e-3 0 1e-3
EOS
NEGEIG_RESUME=1 RESUME_FLAG=--resume bash "$B/run/train.cmd" >"$B/direct.out" 2>&1; drc=$?
check "no decision file: the train command fails instead of guessing a default" test "$drc" -ne 0
eq "  and nothing was launched" "$(launches wide)/$(launches ctrl)" "0/0"

mk_rbox arm6; use_stub_trainer
scn arm6 <<'EOS'
arm "$B" sweep-lr1e-3 0 1e-3
arm "$B" stageA-s1 1 1e-4
EOS
eq "arming again replaces the command (one unit per box)" "$SRC" 0
has "  train.cmd is the second tag" "$B/run/train.cmd" 'run_pair\.sh" stageA-s1 1 --lr 1e-4'
hasnt "  and the first tag is gone from it" "$B/run/train.cmd" 'sweep-lr1e-3'
has "  run.env follows" "$B/run/run.env" 'CKPT_DIR=.*/runs/stageA-s1'

section "abort: the move-aside"
mk_rbox ab
mkdir -p "$B/runs/sweep-lr1e-3/wide_s0" "$B/runs/sweep-lr1e-3/ctrl_s0" "$B/runs/sweep-lr1e-3/wide_s1"
for a in wide_s0 ctrl_s0 wide_s1; do for s in 000200 000250 000300 000350; do : >"$B/runs/sweep-lr1e-3/$a/ckpt_$s.pt"; done; : >"$B/runs/sweep-lr1e-3/$a/trainable_000350.pt"; done
cp -r "$B/runs/sweep-lr1e-3" "$RB/ab_before"
scn abspy <<'EOS'
arm() { echo "ARM_CALL box=$1 tag=$2 seed=$3 lr=$4 steps=${5:-none} ARM_FLAGS=${ARM_FLAGS:-}"; }
. "$RB/x/abort.block"
EOS
eq "the abort block runs" "$SRC" 0
for a in wide_s0 ctrl_s0; do
  d=$B/runs/sweep-lr1e-3/$a
  eq "  $a: only the checkpoints newer than T-50 are set aside" "$(cd "$d" && /bin/ls -1 | grep ckpt | tr '\n' ' ')" "ckpt_000200.pt ckpt_000250.pt ckpt_000300.pt.aborted ckpt_000350.pt.aborted "
  check "  $a: the trainable adapters are not touched" test -f "$d/trainable_000350.pt"
done
eq "  the other seed's directory is untouched" "$(cd "$B/runs/sweep-lr1e-3/wide_s1" && /bin/ls -1 | tr '\n' ' ')" "$(cd "$RB/ab_before/wide_s1" && /bin/ls -1 | tr '\n' ' ')"
has "  the unit was held with the reason" "$B/run/hold" '^abort rule at '
has "  ls shows the renamed files" "$SOUT" 'ckpt_000350\.pt\.aborted'
has "  re-arm at half the LR on the same tag, seed 0" "$SOUT" 'ARM_CALL box=127\.0\.0\.1 tag=sweep-lr1e-3 seed=0 lr=5e-4 steps=none ARM_FLAGS=--lr_override$'
# the same block with the real arm: the unit's command carries the override and the new LR
mk_rbox ab2; use_stub_trainer
mkdir -p "$B/runs/sweep-lr1e-3/wide_s0" "$B/runs/sweep-lr1e-3/ctrl_s0"
for a in wide_s0 ctrl_s0; do for s in 000250 000300; do : >"$B/runs/sweep-lr1e-3/$a/ckpt_$s.pt"; done; done
scn abreal <<'EOS'
. "$RB/x/abort.block"
EOS
eq "the abort block with the real arm" "$SRC" 0
has "  train.cmd has the half LR and --lr_override" "$B/run/train.cmd" 'run_pair\.sh" sweep-lr1e-3 0 --lr 5e-4 --steps 700 \$GRAD_CKPT_FLAGS --lr_override'
check_not "  re-arming cleared the hold" test -e "$B/run/hold"
eq "  the checkpoint at T was set aside, T-50 kept" "$(/bin/ls "$B/runs/sweep-lr1e-3/wide_s0" | tr '\n' ' ')" "ckpt_000250.pt ckpt_000300.pt.aborted "
wrapper
eq "  the wrapper resumes with the override in the trainer argv" "$(stub_field wide 0 argv | grep -c -- '--lr_override')" 1

section "abort: the guard refuses a repair that would leave no checkpoint (a resume with none starts at step 0)"
# keep_ckpts 4 and save_every 50: the checkpoint at T-50 is gone once the run reaches T+150. Reviewer finding D2.
abort_scn() { # NAME: run the abort block with a recording fake arm (fresh box dir already built by the caller)
  scn "$1" <<'EOS'
arm() { echo "ARM_CALL box=$1 tag=$2 seed=$3 lr=$4"; }
. "$RB/x/abort.block"
EOS
}
listing() { (cd "$1" && /bin/ls -1 | tr '\n' ' '); }
# 1. held late: only post-breach checkpoints remain in both arms
mk_rbox ag1
mkdir -p "$B/runs/sweep-lr1e-3/wide_s0" "$B/runs/sweep-lr1e-3/ctrl_s0"
for a in wide_s0 ctrl_s0; do for s in 000300 000350 000400 000450; do : >"$B/runs/sweep-lr1e-3/$a/ckpt_$s.pt"; done; done
abort_scn ag1
check "held at T+150 (no checkpoint at or below T-50): the block fails" test "$SRC" -ne 0
eq "  with exit 3" "$SRC" 3
has "  and names the refusal" "$SOUT" 'ABORT_REPAIR_REFUSED .*(ctrl|wide)_s0 holds no checkpoint at or below step 250; newest: ckpt_000450\.pt'
hasnt "  arm was not called (no rewind to step 0)" "$SOUT" 'ARM_CALL'
for a in wide_s0 ctrl_s0; do
  eq "  $a: nothing was moved aside" "$(listing "$B/runs/sweep-lr1e-3/$a")" "ckpt_000300.pt ckpt_000350.pt ckpt_000400.pt ckpt_000450.pt "
done
has "  the unit is still held" "$B/run/hold" '^abort rule at '
# 2. all or nothing: one arm still has T-50, the other does not. The healthy arm must not be touched either.
mk_rbox ag2
mkdir -p "$B/runs/sweep-lr1e-3/wide_s0" "$B/runs/sweep-lr1e-3/ctrl_s0"
for s in 000250 000300 000350 000400; do : >"$B/runs/sweep-lr1e-3/wide_s0/ckpt_$s.pt"; done
for s in 000300 000350 000400 000450; do : >"$B/runs/sweep-lr1e-3/ctrl_s0/ckpt_$s.pt"; done
abort_scn ag2
eq "one arm past the deadline: exit 3" "$SRC" 3
eq "  the arm that still had T-50 was not touched (all or nothing)" "$(listing "$B/runs/sweep-lr1e-3/wide_s0")" "ckpt_000250.pt ckpt_000300.pt ckpt_000350.pt ckpt_000400.pt "
eq "  the late arm was not touched" "$(listing "$B/runs/sweep-lr1e-3/ctrl_s0")" "ckpt_000300.pt ckpt_000350.pt ckpt_000400.pt ckpt_000450.pt "
has "  the refusal names the arm that has none" "$SOUT" 'ABORT_REPAIR_REFUSED .*ctrl_s0 holds no checkpoint'
hasnt "  arm was not called" "$SOUT" 'ARM_CALL'
# 3. the last step the repair still works: held at T+100, T-50 present
mk_rbox ag3
mkdir -p "$B/runs/sweep-lr1e-3/wide_s0" "$B/runs/sweep-lr1e-3/ctrl_s0"
for a in wide_s0 ctrl_s0; do for s in 000250 000300 000350 000400; do : >"$B/runs/sweep-lr1e-3/$a/ckpt_$s.pt"; done; done
abort_scn ag3
eq "held at T+100: the repair goes through" "$SRC" 0
for a in wide_s0 ctrl_s0; do
  eq "  $a: T-50 kept, the three newer set aside" "$(cd "$B/runs/sweep-lr1e-3/$a" && /bin/ls -1 | tr '\n' ' ')" "ckpt_000250.pt ckpt_000300.pt.aborted ckpt_000350.pt.aborted ckpt_000400.pt.aborted "
done
has "  arm is called after the move" "$SOUT" 'ARM_CALL box=127\.0\.0\.1 tag=sweep-lr1e-3 seed=0 lr=5e-4$'
# 4. no arm directory at all (wrong tag, a box that never started the run)
mk_rbox ag4
mkdir -p "$B/runs"
abort_scn ag4
eq "no *_s0 directory: exit 3" "$SRC" 3
has "  and says so" "$SOUT" 'ABORT_REPAIR_REFUSED no \*_s0 directory under .*/runs/sweep-lr1e-3'
hasnt "  arm was not called" "$SOUT" 'ARM_CALL'
# 5. an empty arm directory (the glob matches nothing) is a refusal, not a silent pass
mk_rbox ag5
mkdir -p "$B/runs/sweep-lr1e-3/wide_s0"
abort_scn ag5
eq "an arm directory with no checkpoint at all: exit 3" "$SRC" 3
hasnt "  arm was not called" "$SOUT" 'ARM_CALL'
# 6. an already-moved-aside file does not count as a clean checkpoint
mk_rbox ag6
mkdir -p "$B/runs/sweep-lr1e-3/wide_s0"
for s in 000250 000300; do : >"$B/runs/sweep-lr1e-3/wide_s0/ckpt_$s.pt.aborted"; done; : >"$B/runs/sweep-lr1e-3/wide_s0/ckpt_000350.pt"
abort_scn ag6
eq "a .pt.aborted file at or below T-50 is not a checkpoint: exit 3" "$SRC" 3
eq "  and the live post-breach file stays" "$(listing "$B/runs/sweep-lr1e-3/wide_s0")" "ckpt_000250.pt.aborted ckpt_000300.pt.aborted ckpt_000350.pt "

section "sweep: pull_logs and the reader"
mk_rbox sw
cat >"$RB/gen.py" <<'PYEOF'
import json, os, sys
runs, tag, gap = sys.argv[1], sys.argv[2], float(sys.argv[3])
keys = {"kl_to_untouched": 0.02, "replay_nll": 1.975, "tool_token_agree": 0.97, "tool_turn_agree": 0.9, "s1:len256:le256": 0.60,
        "s1:len512:gt256": 0.50, "s1:all": 0.55, "nll:all": 1.0, "gate_rate_s1": 0.2, "gate_rate_text": 0.2, "eval_s": 5.0,
        "recall:16384": 0.95, "recall:32768": 0.90}
for arm in ("wide", "ctrl"):
    d = os.path.join(runs, tag, arm + "_s0"); os.makedirs(d, exist_ok=True)
    lr = 1e-3 if tag.endswith("1e-3") else 1e-4
    with open(os.path.join(d, "log.jsonl"), "w") as f:
        f.write(json.dumps({"event": "config", "world": 8, "n_gates": 48 if arm == "wide" else 0,
                            "args": {"lr": lr, "w_lr": lr, "steps": 700, "arm": arm}}) + "\n")
        for s in range(10, 301, 10):
            f.write(json.dumps({"event": "step", "step": s, "loss": {"s1": 1.0, "chat": 1.2, "tool": 0.8, "lm": 2.0},
                                "gnorm": 1.0, "lr": lr, "lr_w": lr}) + "\n")
        for s in (0, 50, 100, 150, 200, 250, 300):
            r = {"event": "eval", "step": s, **keys}
            if s == 0:
                r.update({"kl_to_untouched": 0.0, "replay_nll": 1.9694, "tool_token_agree": 1.0})
            if arm == "wide" and s >= 250:
                r["s1:len256:le256"] = 0.60 + gap
            if arm == "ctrl":
                r["gate_rate_s1"] = r["gate_rate_text"] = 0.0
            f.write(json.dumps(r) + "\n")
    open(os.path.join(d, "ckpt_000300.pt"), "w").write("x" * 64)
    open(os.path.join(d, "trainable_000300.pt"), "w").write("x" * 64)
    open(os.path.join(d, "stdout.log"), "w").write("noise\n")
    open(os.path.join(d, "launch.log"), "w").write("launch\n")
    open(os.path.join(d, "complete.json"), "w").write("{}\n")
PYEOF
python3 "$RB/gen.py" "$B/runs" sweep-lr1e-3 0.10; python3 "$RB/gen.py" "$B/runs" sweep-lr1e-4 0.0
echo "dead" >"$B/runs/sweep-lr1e-4/ctrl_s0/arm_failed"
export RSYNC_LOG=$RB/rsync.log; : >"$RSYNC_LOG"
scn sweep <<'EOS'
. "$RB/x/sweep.block"
EOS
eq "the sweep block runs to the reader's verdict" "$SRC" 0
has "  the last line is the choice, from the 1e-3 tag that carries the wide gap" "$(printf '%s\n' "$SOUT" | grep '^SWEEP_READ ' | tail -n1)" '^SWEEP_READ choice=sweep-lr1e-3 steps=250,300 abort=no '
for tag in sweep-lr1e-3 sweep-lr1e-4; do for a in wide_s0 ctrl_s0; do
  d=$SCRATCH/recv/sweep/$tag/$a
  check "  $tag/$a: log.jsonl pulled" test -s "$d/log.jsonl"
  check "  $tag/$a: complete.json pulled" test -f "$d/complete.json"
  check "  $tag/$a: launch.log pulled" test -f "$d/launch.log"
  check_not "  $tag/$a: no checkpoint pulled" test -e "$d/ckpt_000300.pt"
  check_not "  $tag/$a: no trainable pulled" test -e "$d/trainable_000300.pt"
  check_not "  $tag/$a: no stdout.log pulled" test -e "$d/stdout.log"
done; done
check "  arm_failed is pulled (the reader and the human see a dead arm)" test -f "$SCRATCH/recv/sweep/sweep-lr1e-4/ctrl_s0/arm_failed"
check "  the reader's json is written" test -s "$SCRATCH/recv/sweep/sweep_read.json"
check "  every pull is capped and runs at idle priority (the rig stays usable)" test "$(grep -c -- '--bwlimit=' "$RSYNC_LOG")" -ge 2
has "  the rig-side rsync names the source dir under runs/" "$RSYNC_LOG" ":$B/runs/sweep-lr1e-4/"
# a pull of a log-less box: the reader exits 1 and says why, so the block does not print a stale choice
rm -rf "$SCRATCH/recv"; rm -f "$B/runs/sweep-lr1e-3/wide_s0/log.jsonl"
scn sweep2 <<'EOS'
. "$RB/x/sweep.block"
EOS
check "a missing log makes the block end non-zero (no stale SWEEP_READ line)" test "$SRC" -ne 0
hasnt "  and no choice line is printed" "$SOUT" '^SWEEP_READ choice='
unset RSYNC_LOG

section "evals: eval27 and evalarm"
mk_rbox ev
scn ev <<'EOS'
. "$RB/x/evals.block"
. "$RB/x/evalcalls.block"
echo "BOXX=$BOXX"
EOS
eq "the evals block and the calls block run" "$SRC" 0
TR=$B/runs/sweep-lr1e-3
eq "  three eval27 runs" "$(wc -l <"$B/torchrun.log" | tr -d ' ')" 3
want_pre="$C27|--standalone|--nproc_per_node|8|eval27.py|--model|$B/model|--arm"
eq "  wide: cwd, argv, the trainable path expanded on the box, --out by name" "$(sed -n 1p "$B/torchrun.log")" "$want_pre|wide|--trainable|$TR/wide_s0/trainable_000700.pt|--s1|$B/data/s1|--out|$B/eval27/wide_s0"
eq "  ctrl" "$(sed -n 2p "$B/torchrun.log")" "$want_pre|ctrl|--trainable|$TR/ctrl_s0/trainable_000700.pt|--s1|$B/data/s1|--out|$B/eval27/ctrl_s0"
eq "  the untouched model: ctrl arm, trainable none" "$(sed -n 3p "$B/torchrun.log")" "$want_pre|ctrl|--trainable|none|--s1|$B/data/s1|--out|$B/eval27/untouched"
EA=$B/evals.log
eq "  nine eval-arm calls: three of (up, run_general, down)" "$(wc -l <"$EA" | tr -d ' ')" 9
ed=$C27/evals
eq "  base: up, no --trainable" "$(sed -n 1p "$EA")" "serve_arm cwd=$ed up --arm base --kind base"
eq "  base: the five evals" "$(sed -n 2p "$EA")" "run_general cwd=$ed --arm base"
eq "  base: down without --purge (nothing was merged)" "$(sed -n 3p "$EA")" "serve_arm cwd=$ed down"
eq "  wide: up with the trainable expanded on the box" "$(sed -n 4p "$EA")" "serve_arm cwd=$ed up --arm wide_s0 --kind wide --trainable $TR/wide_s0/trainable_000700.pt"
eq "  wide: run_general" "$(sed -n 5p "$EA")" "run_general cwd=$ed --arm wide_s0"
eq "  wide: down --purge (the merged model of that arm goes)" "$(sed -n 6p "$EA")" "serve_arm cwd=$ed down --purge"
eq "  ctrl: up" "$(sed -n 7p "$EA")" "serve_arm cwd=$ed up --arm ctrl_s0 --kind ctrl --trainable $TR/ctrl_s0/trainable_000700.pt"
eq "  ctrl: down --purge" "$(sed -n 9p "$EA")" "serve_arm cwd=$ed down --purge"
check "  every label left an exit file of 0 on the box" test "$(cat "$B"/logs/*.exit | sort -u | tr '\n' ' ')" = "0 "

section "evals: failure paths"
mk_rbox ev2
export FAKE_RG_RC=3
scn ev2 <<'EOS'
. "$RB/x/evals.block"
evalarm "$B" base base; echo "EVALARM_RC=$?"
EOS
has "run_general rc 3 (some incomplete) comes back as evalarm's status" "$SOUT" 'EVALARM_RC=3$'
has "  and down still ran, so the next evalarm finds the GPUs free" "$B/evals.log" 'down$'
unset FAKE_RG_RC
mk_rbox ev3
export FAKE_UP_RC=7
scn ev3 <<'EOS'
. "$RB/x/evals.block"
evalarm "$B" wide_s0 wide '$RUNS/x/trainable_000700.pt'; echo "EVALARM_RC=$?"
EOS
has "a failing serve_arm up returns its status" "$SOUT" 'EVALARM_RC=7$'
eq "  and nothing else ran (the evals never meet a server that is not up)" "$(grep -c . "$B/evals.log")" 1
hasnt "  no down on this path (serve_arm.sh stops its own server when up fails; Appendix B says so)" "$B/evals.log" ' down'
has "Appendix B has a row for this case" "$RUNBOOK" 'evalarm. returns at once with .up-<name>. failed'
unset FAKE_UP_RC
mk_rbox ev4
scn ev4 <<'EOS'
. "$RB/x/evals.block"
evalarm "$B" base base; echo "FIRST=$?"
evalarm "$B" base base; echo "SECOND=$?"
EOS
has "evalarm is rerunnable under the same names (labels are reusable once finished)" "$SOUT" 'SECOND=0$'
eq "  six calls in the log" "$(wc -l <"$B/evals.log" | tr -d ' ')" 6

finish
