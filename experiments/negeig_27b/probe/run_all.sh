#!/usr/bin/env bash
# run_all.sh: the generative probe, arm by arm, on ONE GPU box (8x H200). Runs ON the box. Never on the rig.
#
#   run_all.sh [--arms "untouched ctrl_s0 wide_s0"] [--modes "off on"] [--tasks "oncall s1gen"] [--jobs "oncall:on ..."]
#              [--print-cmd]
#
# --jobs replaces the modes x tasks product with an explicit list of task:mode entries (e.g. "oncall:off oncall:on
# s1gen:off"), so a slow job can wait for a later invocation while the rest of an arm runs now.
#
# Per arm:
#   1. serve   evals/serve_arm.sh up (merge-lora with its delta_kept gate, export-gates, SGLang 0.5.20 with DP, health
#              under a deadline, the 17*23 liveness probe, the engagement check from the server log). This script adds
#              --tool-call-parser qwen3_coder --reasoning-parser qwen3 through serve_arm's --extra, the radix cache
#              (RADIX, on: every turn re-sends the whole session) and the context length (CTX). A server already up
#              for this arm with the same launch is reused.
#   2. check   probe_common.py check-server: a tool call parsed in both modes, reasoning split out in thinking mode
#              and absent in think-off, a history message carrying reasoning_content accepted. Fails the arm otherwise.
#   3. jobs    run_oncall.py and run_s1gen.py for every mode (thinking off and on, plus GREEDY_MODES), each session
#              pinned to one DP rank (--dp-ranks DP: its history stays in that rank's radix cache), all at once by
#              default (PARALLEL=1; the sessions are sequential inside, so the longest thinking session is the
#              critical path and the others fill the GPUs around it). Each job runs under `timeout JOB_TIMEOUT_S`; a
#              watch loop polls /health every POLL_S and kills the jobs when the server is gone for HEALTH_FAILS polls
#              in a row or the arm passes ARM_DEADLINE_S.
#   4. down    serve_arm.sh down (with --purge when PURGE=1: drop the 54 GB merged copy; a rerun merges it again,
#              deterministically: stochastic rounding with a fixed seed).
# Then report.py over every run -> $PROBE_ROOT/report.{json,md}.
#
# RESUMABLE at every level: a job whose receipt says complete is skipped, an arm whose jobs are all complete is not
# served again, and an unfinished job resumes mid-session from its gens.jsonl (rerun the same command after a
# preemption). Before the first arm, the plans (no server) check that the longest session plus the token cap plus a
# reply allowance (64 tokens a turn) fits CTX; SGLang 0.5.20 refuses a request that does not fit, it never truncates.
# The first failing arm stops the script (exit 1); exit 0 means every job is complete and the report is written.
#
# Receipts ($PROBE_ROOT/receipts): plan.<task>_<mode>.json, <arm>.trainable.json, <arm>.serve.json (serve_arm's
# arm.json), <arm>.server_check.json, <arm>.<task>_<mode>.json (exit code, complete, times, the job summary),
# report.json. Logs: $PROBE_ROOT/logs. Runs: $PROBE_ROOT/runs/<arm>/<task>_<mode>/{ledger,gens}.jsonl.
#
# Env (default): PROBE_ROOT ($W/probe) PROBE_PY ($PY, the training venv: transformers) PROBE_TOKENIZER ($MODEL_DIR)
#   ONCALL_ITEMS ($DATA/oncall/oncall.eval.jsonl) S1_DIR ($DATA/s1) ONCALL_ARGS ("--per-length 16")
#   S1_ARGS ("--lengths 256,1024 --per-cell 16") WIDE_TRAINABLE / CTRL_TRAINABLE
#   ($W/runs/sweep-lr1e-4/{wide,ctrl}_s0/trainable_000700.pt) WIDE_SHA256 / CTRL_SHA256 (the step-700 files on the
#   rig; set empty to skip the check) CTX (131072) RADIX (on) DP (8) PORT (30000) MAX_TOKENS_ON (16384)
#   MAX_TOKENS_OFF (1024) HISTORY_REASONING (drop) REASONING_EFFORT (unset: the template default) ON_LIMIT (continue)
#   CONC_ONCALL (128) CONC_S1 (192) PARALLEL (1) GREEDY_MODES ("": e.g. "off" adds a greedy think-off run)
#   JOB_TIMEOUT_S (28800) ARM_DEADLINE_S (JOB_TIMEOUT_S + 1800) POLL_S (30) HEALTH_FAILS (6) PURGE (1)
#   SERVE_ARM (../evals/serve_arm.sh) SELFDISTILL_LANE_DIR ($LANE_DIR)
set -u -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
E27=${E27:-$(cd "$HERE/.." && pwd)}
BOXDIR=${BOXDIR:-$E27/box}
SRC_E27=$E27; SRC_BOXDIR=$BOXDIR
# shellcheck source=/dev/null
. "$SRC_BOXDIR/boxenv.sh"
E27=$SRC_E27; BOXDIR=$SRC_BOXDIR

PROBE_ROOT=${PROBE_ROOT:-$W/probe}
PROBE_RUNS=$PROBE_ROOT/runs
REC=$PROBE_ROOT/receipts
LOGD=$PROBE_ROOT/logs
SERVE_ROOT=$PROBE_ROOT/serve
PROBE_PY=${PROBE_PY:-$PY}
PROBE_TOKENIZER=${PROBE_TOKENIZER:-$MODEL_DIR}
ONCALL_ITEMS=${ONCALL_ITEMS:-$DATA/oncall/oncall.eval.jsonl}
S1_DIR=${S1_DIR:-$DATA/s1}
ONCALL_ARGS=${ONCALL_ARGS:---per-length 16}
S1_ARGS=${S1_ARGS:---lengths 256,1024 --per-cell 16}
WIDE_TRAINABLE=${WIDE_TRAINABLE:-$W/runs/sweep-lr1e-4/wide_s0/trainable_000700.pt}
CTRL_TRAINABLE=${CTRL_TRAINABLE:-$W/runs/sweep-lr1e-4/ctrl_s0/trainable_000700.pt}
WIDE_SHA256=${WIDE_SHA256-1122621b0e9c75bcc0a51b28ca580626d118ddf7074efe410dae9a33928a0d2b}
CTRL_SHA256=${CTRL_SHA256-6017470c2c83ad899c9e00498e43d5c5a0bbca27eafcab7d625f1c7b3710cea9}
CTX=${CTX:-131072}; RADIX=${RADIX:-on}; DP=${DP:-8}; PORT=${PORT:-30000}
MAX_TOKENS_ON=${MAX_TOKENS_ON:-16384}; MAX_TOKENS_OFF=${MAX_TOKENS_OFF:-1024}
HISTORY_REASONING=${HISTORY_REASONING:-drop}; REASONING_EFFORT=${REASONING_EFFORT:-}; ON_LIMIT=${ON_LIMIT:-continue}
CONC_ONCALL=${CONC_ONCALL:-128}; CONC_S1=${CONC_S1:-192}; PARALLEL=${PARALLEL:-1}; GREEDY_MODES=${GREEDY_MODES:-}
JOB_TIMEOUT_S=${JOB_TIMEOUT_S:-28800}; ARM_DEADLINE_S=${ARM_DEADLINE_S:-$((JOB_TIMEOUT_S + 1800))}
POLL_S=${POLL_S:-30}; HEALTH_FAILS=${HEALTH_FAILS:-6}; PURGE=${PURGE:-1}
SERVE_ARM=${SERVE_ARM:-$E27/evals/serve_arm.sh}
export SELFDISTILL_LANE_DIR=${SELFDISTILL_LANE_DIR:-$LANE_DIR}
PARSER_FLAGS="--tool-call-parser qwen3_coder --reasoning-parser qwen3"

ARMS="untouched ctrl_s0 wide_s0"; MODES="off on"; TASKS="oncall s1gen"; JOBLIST=""; PRINT=0
die() { echo "run_all: ERROR: $*" >&2; exit 1; }
say() { echo "[$(date -u +%FT%TZ)] run_all: $*" >&2; }
need() { [ $# -ge 2 ] || die "$1 needs a value"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --arms) need "$@"; ARMS=$2; shift 2 ;;
    --modes) need "$@"; MODES=$2; shift 2 ;;
    --tasks) need "$@"; TASKS=$2; shift 2 ;;
    --jobs) need "$@"; JOBLIST=$2; shift 2 ;;
    --print-cmd) PRINT=1; shift ;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//' >&2; exit 2 ;;
    *) echo "run_all: unknown argument: $1" >&2; exit 2 ;;
  esac
done
for v in CTX DP PORT MAX_TOKENS_ON MAX_TOKENS_OFF CONC_ONCALL CONC_S1 JOB_TIMEOUT_S ARM_DEADLINE_S POLL_S HEALTH_FAILS; do
  [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "$v='${!v}' is not a positive integer"
done
case "$RADIX" in on|off) ;; *) die "RADIX must be on or off" ;; esac
case "$ON_LIMIT" in continue|end) ;; *) die "ON_LIMIT must be continue or end" ;; esac
for m in $MODES $GREEDY_MODES; do case "$m" in on|off) ;; *) die "a mode is on or off (got '$m')" ;; esac; done
for t in $TASKS; do case "$t" in oncall|s1gen) ;; *) die "a task is oncall or s1gen (got '$t')" ;; esac; done
for a in $ARMS; do case "$a" in untouched|ctrl_*|wide_*) ;; *) die "arm '$a': untouched, ctrl_<x> or wide_<x>" ;; esac; done

kind_of() { case "$1" in untouched) echo base ;; ctrl_*) echo ctrl ;; wide_*) echo wide ;; esac; }
trainable_of() { case "$1" in ctrl_*) echo "$CTRL_TRAINABLE" ;; wide_*) echo "$WIDE_TRAINABLE" ;; *) echo "" ;; esac; }
sha_of_arm() { case "$1" in ctrl_*) echo "$CTRL_SHA256" ;; wide_*) echo "$WIDE_SHA256" ;; *) echo "" ;; esac; }
label_of() { # mode [greedy]
  if [ "$1" = on ]; then printf think_on; else printf think_off; fi
  [ "${2:-}" = greedy ] && printf _greedy
  return 0
}
JOBS=() # "task mode greedy?" entries
if [ -n "$JOBLIST" ]; then
  TASKS=""
  for e in $JOBLIST; do
    t=${e%%:*}; m=${e#*:}
    case "$t" in oncall|s1gen) ;; *) die "--jobs entry '$e': the task is oncall or s1gen" ;; esac
    case "$m" in on|off) ;; *) die "--jobs entry '$e': the mode is on or off" ;; esac
    JOBS+=("$t $m"); case " $TASKS " in *" $t "*) ;; *) TASKS="$TASKS $t" ;; esac
  done
  [ "${#JOBS[@]}" -gt 0 ] || die "--jobs is empty"
else
  for m in $MODES; do for t in $TASKS; do JOBS+=("$t $m"); done; done
fi
for m in $GREEDY_MODES; do for t in $TASKS; do JOBS+=("$t $m greedy"); done; done

job_cmd() { # arm task mode [greedy] -> prints the command words, one per line
  local arm=$1 task=$2 mode=$3 greedy=${4:-} lab out mt
  lab=$(label_of "$mode" "$greedy"); out=$PROBE_RUNS/$arm/${task}_$lab
  if [ "$mode" = on ]; then mt=$MAX_TOKENS_ON; else mt=$MAX_TOKENS_OFF; fi
  local -a c=("$PROBE_PY" "$HERE/run_$task.py" --thinking "$mode" --arm "$arm" --out-dir "$out"
              --base-url "http://127.0.0.1:$PORT" --tokenizer "$PROBE_TOKENIZER" --max-tokens "$mt"
              --dp-ranks "$DP")
  [ "$greedy" = greedy ] && c+=(--greedy)
  if [ "$mode" = on ]; then
    c+=(--history-reasoning "$HISTORY_REASONING")
    [ -z "$REASONING_EFFORT" ] || c+=(--reasoning-effort "$REASONING_EFFORT")
  fi
  if [ "$task" = oncall ]; then
    c+=(--items "$ONCALL_ITEMS" --concurrency "$CONC_ONCALL" --on-limit "$ON_LIMIT")
    # shellcheck disable=SC2206  # a word list on purpose
    c+=($ONCALL_ARGS)
  else
    c+=(--s1 "$S1_DIR" --concurrency "$CONC_S1")
    # shellcheck disable=SC2206
    c+=($S1_ARGS)
  fi
  printf '%s\n' "${c[@]}"
}

pyj() { "$PROBE_PY" -c 'import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: print(""); sys.exit(0)
for k in sys.argv[2].split("."):
    d=d.get(k) if isinstance(d,dict) else None
print("" if d is None else (json.dumps(d) if isinstance(d,(dict,list)) else d))' "$@"; }
job_complete() { [ "$(pyj "$REC/$1.json" complete)" = True ]; }
health() { [ "$(curl -s -o /dev/null -m 10 -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = 200 ]; }

if [ "$PRINT" = 1 ]; then
  for arm in $ARMS; do
    k=$(kind_of "$arm"); tr=$(trainable_of "$arm")
    echo "# arm $arm ($k)"
    echo "EVAL_ROOT=$SERVE_ROOT $SERVE_ARM up --replace --arm probe_$arm --kind $k ${tr:+--trainable $tr }--radix $RADIX --ctx $CTX --dp $DP --port $PORT --extra \"$PARSER_FLAGS\""
    echo "$PROBE_PY $HERE/probe_common.py check-server --base-url http://127.0.0.1:$PORT --tokenizer $PROBE_TOKENIZER --dp-ranks $DP --out $REC/$arm.server_check.json"
    for j in "${JOBS[@]}"; do
      # shellcheck disable=SC2086
      mapfile -t words < <(job_cmd "$arm" $j)
      echo "timeout $JOB_TIMEOUT_S ${words[*]}"
    done
    echo "EVAL_ROOT=$SERVE_ROOT $SERVE_ARM down$([ "$PURGE" = 1 ] && [ "$k" != base ] && echo ' --purge')"
  done
  echo "$PROBE_PY $HERE/report.py --root $PROBE_RUNS --out-json $PROBE_ROOT/report.json --out-md $PROBE_ROOT/report.md"
  exit 0
fi

mkdir -p "$PROBE_RUNS" "$REC" "$LOGD" "$SERVE_ROOT" || die "cannot create $PROBE_ROOT"
[ -x "$PROBE_PY" ] || command -v "$PROBE_PY" >/dev/null 2>&1 || die "PROBE_PY=$PROBE_PY is not an interpreter"
[ -f "$PROBE_TOKENIZER/tokenizer.json" ] || die "PROBE_TOKENIZER=$PROBE_TOKENIZER has no tokenizer.json"
[ -f "$SELFDISTILL_LANE_DIR/agentic_env.py" ] || die "no lane at $SELFDISTILL_LANE_DIR (push_data.sh --hebrew-lane)"
# shellcheck disable=SC2086  # ONCALL_ITEMS may be a glob (run_oncall.py --items expands it too)
case " $TASKS " in *" oncall "*) ls $ONCALL_ITEMS >/dev/null 2>&1 || die "no on-call items at $ONCALL_ITEMS (data/oncall.py --split eval --out ...)" ;; esac
case " $TASKS " in *" s1gen "*) [ -d "$S1_DIR" ] || die "no S1 dir $S1_DIR" ;; esac

# ---------------------------------------------------------------------------------------------------------------
# plans: selection, turns, the longest gold context plus the cap must fit the server's context
for j in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  set -- $j
  lab=$(label_of "$2" "${3:-}")
  pf=$REC/plan.${1}_$lab.json
  # recomputed on every invocation (about a minute): a plan cached from other ONCALL_ARGS, S1_ARGS or caps would lie
  mapfile -t words < <(job_cmd plan "$1" "$2" "${3:-}")
  pdir=$PROBE_ROOT/plans/${1}_$lab
  rm -f "$pdir/plan.json"
  "${words[@]}" --plan --out-dir "$pdir" >"$LOGD/plan.${1}_$lab.log" 2>&1 || { tail -n 20 "$LOGD/plan.${1}_$lab.log" >&2; die "plan for $1 $lab failed"; }
  cp "$pdir/plan.json" "$pf" || die "plan for $1 $lab wrote no plan.json"
  need_ctx=$(pyj "$pf" context_needed)
  [ -n "$need_ctx" ] || die "$pf has no context_needed"
  [ "$need_ctx" -le "$CTX" ] || die "$1 $lab needs a context of $need_ctx tokens (longest gold session + cap), CTX is $CTX: raise CTX or lower the cap"
  say "plan $1 $lab: $(pyj "$pf" turns_total) turns, longest context $need_ctx <= $CTX"
done

stop_jobs() { # pids of session leaders: signal each whole session (timeout and the runner under it)
  local p; for p in "$@"; do kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; done
  sleep 10
  for p in "$@"; do kill -KILL -- "-$p" 2>/dev/null; done
  return 0
}

# ---------------------------------------------------------------------------------------------------------------
for arm in $ARMS; do
  kind=$(kind_of "$arm"); tr=$(trainable_of "$arm"); sarm=probe_$arm
  todo=()
  for j in "${JOBS[@]}"; do
    # shellcheck disable=SC2086
    set -- $j
    job_complete "$arm.${1}_$(label_of "$2" "${3:-}")" || todo+=("$j")
  done
  if [ "${#todo[@]}" -eq 0 ]; then say "arm $arm: every job complete, not serving it"; continue; fi

  if [ "$kind" != base ]; then
    [ -f "$tr" ] || die "arm $arm: trainable $tr not found"
    want=$(sha_of_arm "$arm")
    got=$(sha256sum "$tr" | cut -d' ' -f1)
    [ -z "$want" ] || [ "$got" = "$want" ] || die "arm $arm: $tr has sha256 $got, expected $want (the step-700 file on the rig)"
    printf '{"arm":"%s","trainable":"%s","sha256":"%s","expected":"%s"}\n' "$arm" "$tr" "$got" "$want" >"$REC/$arm.trainable.json"
  fi

  # 1. serve (reuse a healthy server of this arm with the same launch)
  reuse=0
  if [ "$(pyj "$SERVE_ROOT/serving.json" arm)" = "$sarm" ] && health \
     && [ "$(pyj "$SERVE_ROOT/$sarm/arm.json" engagement.ok)" = True ] \
     && [ "$(pyj "$SERVE_ROOT/$sarm/serve.json" extra_args)" = "$PARSER_FLAGS" ] \
     && [ "$(pyj "$SERVE_ROOT/$sarm/serve.json" radix)" = "$RADIX" ] \
     && [ "$(pyj "$SERVE_ROOT/$sarm/serve.json" context_length)" = "$CTX" ]; then
    reuse=1; say "arm $arm: reusing the running server"
  fi
  if [ "$reuse" = 0 ]; then
    say "arm $arm: serve_arm.sh up ($kind)"
    up=(env "EVAL_ROOT=$SERVE_ROOT" "$SERVE_ARM" up --replace --arm "$sarm" --kind "$kind" --radix "$RADIX" --ctx "$CTX"
        --dp "$DP" --port "$PORT" --extra "$PARSER_FLAGS")
    [ "$kind" = base ] || up+=(--trainable "$tr")
    "${up[@]}" >"$LOGD/$arm.serve.log" 2>&1 || { tail -n 30 "$LOGD/$arm.serve.log" >&2; die "arm $arm: serve_arm.sh up failed (log $LOGD/$arm.serve.log)"; }
  fi
  cp "$SERVE_ROOT/$sarm/arm.json" "$REC/$arm.serve.json" || die "arm $arm: no arm.json after up"

  # 2. both parsers engaged
  if ! "$PROBE_PY" "$HERE/probe_common.py" check-server --base-url "http://127.0.0.1:$PORT" --tokenizer "$PROBE_TOKENIZER" \
       --dp-ranks "$DP" --out "$REC/$arm.server_check.json" >"$LOGD/$arm.server_check.log" 2>&1; then
    cat "$LOGD/$arm.server_check.log" >&2
    env "EVAL_ROOT=$SERVE_ROOT" "$SERVE_ARM" down >/dev/null 2>&1
    die "arm $arm: the server check failed (receipt $REC/$arm.server_check.json): a parser is not engaged"
  fi

  # 3. jobs, all at once (PARALLEL=1) or one after another, under a deadline with a health watch
  run_batch() { # job entries...
    local -a pids=() names=() starts=()
    local e words lab name
    for e in "$@"; do
      # shellcheck disable=SC2086
      set -- $e
      lab=$(label_of "$2" "${3:-}"); name=$arm.${1}_$lab
      mapfile -t words < <(job_cmd "$arm" "$1" "$2" "${3:-}")
      say "start $name"
      # own session, so a stop reaches the runner behind timeout as well (kill -- -PGID)
      setsid timeout "$JOB_TIMEOUT_S" "${words[@]}" >>"$LOGD/$name.log" 2>&1 </dev/null &
      pids+=($!); names+=("$name"); starts+=("$(date -u +%FT%TZ)")
    done
    local t0=$SECONDS fails=0 alive i
    while :; do
      alive=0
      for i in "${!pids[@]}"; do kill -0 "${pids[$i]}" 2>/dev/null && alive=1; done
      [ "$alive" = 1 ] || break
      sleep "$POLL_S"
      if health; then fails=0; else fails=$((fails + 1)); fi
      if [ "$fails" -ge "$HEALTH_FAILS" ]; then
        say "arm $arm: /health failed $fails polls in a row, stopping the jobs"; stop_jobs "${pids[@]}"; break
      fi
      if [ $((SECONDS - t0)) -ge "$ARM_DEADLINE_S" ]; then
        say "arm $arm: past ARM_DEADLINE_S=$ARM_DEADLINE_S, stopping the jobs"; stop_jobs "${pids[@]}"; break
      fi
    done
    local rc out summ
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}"; rc=$?
      name=${names[$i]}; out=$PROBE_RUNS/$arm/${name#"$arm".}
      summ=$out/summary.json
      "$PROBE_PY" - "$REC/$name.json" "$name" "$rc" "${starts[$i]}" "$summ" "$LOGD/$name.log" <<'PYEOF'
import calendar, json, os, sys, time
path, name, rc, start, summ, logf = sys.argv[1:]
s = {}  # a summary older than this start belongs to an earlier invocation
if os.path.exists(summ) and os.path.getmtime(summ) + 1 >= calendar.timegm(time.strptime(start, '%Y-%m-%dT%H:%M:%SZ')):
    s = json.load(open(summ))
doc = {'job': name, 'exit': int(rc), 'complete': bool(s.get('complete')) and int(rc) == 0, 'started': start,
       'ended': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'log': logf, 'summary': s}
json.dump(doc, open(path, 'w'), indent=1)
PYEOF
      say "$name: exit $rc, complete $(pyj "$REC/$name.json" complete)"
    done
  }
  if [ "$PARALLEL" = 1 ]; then run_batch "${todo[@]}"; else for j in "${todo[@]}"; do run_batch "$j"; done; fi

  # 4. down
  down=(env "EVAL_ROOT=$SERVE_ROOT" "$SERVE_ARM" down)
  [ "$PURGE" = 1 ] && [ "$kind" != base ] && down+=(--purge)
  "${down[@]}" >"$LOGD/$arm.down.log" 2>&1 || { cat "$LOGD/$arm.down.log" >&2; die "arm $arm: serve_arm.sh down failed"; }
  for j in "${todo[@]}"; do
    # shellcheck disable=SC2086
    set -- $j
    job_complete "$arm.${1}_$(label_of "$2" "${3:-}")" || die "arm $arm: job ${1} $(label_of "$2" "${3:-}") is not complete (see $LOGD); rerun to resume"
  done
  say "arm $arm: done"
done

# ---------------------------------------------------------------------------------------------------------------
"$PROBE_PY" "$HERE/report.py" --root "$PROBE_RUNS" --out-json "$PROBE_ROOT/report.json" --out-md "$PROBE_ROOT/report.md" \
  >"$LOGD/report.log" 2>&1 || { cat "$LOGD/report.log" >&2; die "report.py failed"; }
cp "$PROBE_ROOT/report.json" "$REC/report.json"
say "report: $PROBE_ROOT/report.md"
exit 0
