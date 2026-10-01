#!/usr/bin/env bash
# Run the five Stage A general evals against the arm that serve_arm.sh is serving. Runs ON the GPU box.
#
#   run_general.sh --arm NAME [options]
#
# Evals, in this order: mmlupro (selfcheck first, then cloze + letter), humaneval (canonical check first), ifeval,
# gmmlu (Global-MMLU he), bfcl (v4, every category, think off). Each one is its own ev_*.py with a journal, so the whole
# script is RESUMABLE: re-run the same command after any stop and finished items are not asked again; an eval whose
# result is already complete for the same subset is skipped. Exit 3 of an eval (some items failed, the journal keeps the
# rest) is retried up to --retries passes; 1 and 2 (bad data, bad arguments, selfcheck) are not retried; 4 (server down)
# stops everything.
#
# Output (all under the arm dir, default $EVAL_ROOT/NAME, the one serve_arm.sh wrote arm.json into):
#   results/<eval>.json   per-item outcomes, the input of compare_general.py
#   receipts/<eval>.status.json, receipts/run_general.json, receipts/mmlupro.selfcheck.json, receipts/humaneval.canonical.json
#   journal/<eval>.jsonl  resume state          logs/<eval>.log  what each ev_*.py printed
# Exit: 0 all five complete; 3 some incomplete after the retries; 1 an eval failed for good; 4 the server went away;
# 2 bad command line.
#
# Options (env var in brackets):
#   --arm NAME            required; must be the arm in $EVAL_ROOT/serving.json (a result is only as good as that label)
#   --out DIR             arm dir [ARM_DIR, $EVAL_ROOT/NAME]
#   --evals a,b,c         subset of mmlupro,humaneval,ifeval,gmmlu,bfcl [all five]
#   --base-url URL        [EVAL_BASE_URL; else the port in serving.json]
#   --retries N           passes per eval on exit 3 [3]            --backoff-s S  wait between passes [20]
#   --stride-<eval> N     evaluate every N-th item of that eval (1 = the full set; the default for all five)
#   --limit N             first N selected items of every eval (smoke runs; the result records it)
#   --extra-<eval> "ARGS" appended to that eval's command line (e.g. --extra-bfcl "--workers 256")
#   --selfcheck N         mmlupro logprob semantics check on N items before the eval [48]; 0 skips
#   --overlap             run bfcl in the background beside the other four (fills the GPUs while the small evals tail)
#   --fresh               discard journals and receipts, start every eval over
#   --allow-unpinned      tests only: skip the pins.json checks of the data files
#   --print-cmd           print the commands and exit
# Env: EVAL_PY [$W/evalvenv/bin/python] EVAL_DATA [$W/evaldata] EVAL_TOKENIZER [the base model dir, for every arm]
#      BFCL_VENV [$W/bfcl-venv] EVAL_ROOT [$W/eval] PASS_TIMEOUT_S [14400] BFCL_WORKERS [16 x dp, from serve.json]
#      NLTK_DATA [<EVAL_DATA>/nltk_data]
set -u -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
E27=${E27:-$(cd "$HERE/.." && pwd)}
BOXDIR=${BOXDIR:-$E27/box}
# shellcheck source=/dev/null
. "$BOXDIR/boxenv.sh"
EVAL_ROOT=${EVAL_ROOT:-$W/eval}
EVAL_PY=${EVAL_PY:-$W/evalvenv/bin/python}
EVAL_DATA=${EVAL_DATA:-$W/evaldata}
EVAL_TOKENIZER=${EVAL_TOKENIZER:-$MODEL_DIR}
BFCL_VENV=${BFCL_VENV:-$W/bfcl-venv}
PASS_TIMEOUT_S=${PASS_TIMEOUT_S:-14400}
ALL_EVALS="mmlupro humaneval ifeval gmmlu bfcl"

die() { echo "run_general: ERROR: $1" >&2; exit "${2:-1}"; }
usage() { sed -n '2,/^set -u/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//' >&2; exit 2; }
say() { echo "[$(date -u +%FT%TZ)] run_general: $*" >&2; }
need() { [ $# -ge 2 ] || die "$1 needs a value" 2; }
isint() { case "${1:-}" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }

ARM=""; OUT=${ARM_DIR:-}; EVALS=""; BASE_URL=${EVAL_BASE_URL:-}; RETRIES=3; BACKOFF_S=20; LIMIT=0; SELFCHECK=48
OVERLAP=0; FRESH=0; UNPINNED=0; PRINT=0
declare -A EV_STRIDE=([mmlupro]=1 [humaneval]=1 [ifeval]=1 [gmmlu]=1 [bfcl]=1)
declare -A EV_EXTRA=([mmlupro]="" [humaneval]="" [ifeval]="" [gmmlu]="" [bfcl]="")
stride_of() { echo "${EV_STRIDE[$1]}"; }
extra_of() { echo "${EV_EXTRA[$1]}"; }
valid_eval() { case "$1" in mmlupro|humaneval|ifeval|gmmlu|bfcl) return 0 ;; *) return 1 ;; esac; }

while [ $# -gt 0 ]; do
  case "$1" in
    --arm) need "$@"; ARM=$2; shift 2 ;;
    --out) need "$@"; OUT=$2; shift 2 ;;
    --evals) need "$@"; [ -n "$2" ] || die "--evals needs a list" 2; EVALS=$2; shift 2 ;;
    --base-url) need "$@"; BASE_URL=$2; shift 2 ;;
    --retries) need "$@"; RETRIES=$2; shift 2 ;;
    --backoff-s) need "$@"; BACKOFF_S=$2; shift 2 ;;
    --limit) need "$@"; LIMIT=$2; shift 2 ;;
    --selfcheck) need "$@"; SELFCHECK=$2; shift 2 ;;
    --stride-*) n=${1#--stride-}; valid_eval "$n" || { echo "run_general: unknown eval in $1" >&2; usage; }
                isint "${2:-}" && [ "$2" -ge 1 ] || die "$1 needs an integer >= 1" 2
                EV_STRIDE[$n]=$2; shift 2 ;;
    --extra-*) n=${1#--extra-}; valid_eval "$n" || { echo "run_general: unknown eval in $1" >&2; usage; }
               need "$@"; EV_EXTRA[$n]=$2; shift 2 ;;
    --overlap) OVERLAP=1; shift ;;
    --fresh) FRESH=1; shift ;;
    --allow-unpinned) UNPINNED=1; shift ;;
    --print-cmd) PRINT=1; shift ;;
    -h|--help) usage ;;
    *) echo "run_general: unknown argument: $1" >&2; usage ;;
  esac
done

[ -n "$ARM" ] || { echo "run_general: --arm is required" >&2; usage; }
case "$ARM" in *[!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-]*|'') die "arm name '$ARM' must match [A-Za-z0-9_.-]+" 2 ;; esac
isint "$RETRIES" && [ "$RETRIES" -ge 1 ] || die "--retries needs an integer >= 1" 2
isint "$BACKOFF_S" || die "--backoff-s needs an integer" 2
isint "$LIMIT" || die "--limit needs an integer" 2
isint "$SELFCHECK" || die "--selfcheck needs an integer" 2
isint "$PASS_TIMEOUT_S" && [ "$PASS_TIMEOUT_S" -ge 1 ] || die "PASS_TIMEOUT_S needs an integer >= 1" 2
if [ -z "$EVALS" ]; then
  LIST=$ALL_EVALS
else
  LIST=""
  for n in $(echo "$EVALS" | tr ',' ' '); do
    valid_eval "$n" || die "--evals: '$n' is not one of ${ALL_EVALS// /, }" 2
    LIST="$LIST $n"
  done
  # keep the canonical order whatever the order given
  ORD=""
  for n in $ALL_EVALS; do case " $LIST " in *" $n "*) ORD="$ORD $n" ;; esac; done
  LIST=${ORD# }
  [ -n "$LIST" ] || die "--evals is empty" 2
fi
OUT=${OUT:-$EVAL_ROOT/$ARM}
SERVING=$EVAL_ROOT/serving.json
RES_NAME() { case "$1" in gmmlu) echo gmmlu_he ;; *) echo "$1" ;; esac; }
SCRIPT_OF() { case "$1" in mmlupro) echo ev_mmlupro.py ;; humaneval) echo ev_humaneval.py ;; ifeval) echo ev_ifeval.py ;; gmmlu) echo ev_gmmlu.py ;; bfcl) echo ev_bfcl.py ;; esac; }

pj() { "$EVAL_PY" -c 'import json,sys
d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    d=d[k]
print(d if not isinstance(d,(dict,list)) else json.dumps(d))' "$@"; }

# ---------------------------------------------------------------------------------------------------------------
# what is being served must be the arm named here, and it must have proved its engagement
if [ "$PRINT" = 0 ]; then
  [ -x "$EVAL_PY" ] || command -v "$EVAL_PY" >/dev/null 2>&1 || die "EVAL_PY=$EVAL_PY is not an interpreter: prepare_box.sh venv"
  [ -f "$SERVING" ] || die "no $SERVING: serve_arm.sh up --arm $ARM first"
  srv_arm=$(pj "$SERVING" arm) || die "cannot read $SERVING"
  [ "$srv_arm" = "$ARM" ] || die "the server is serving arm '$srv_arm', not '$ARM' (serve_arm.sh down, then up --arm $ARM)"
  [ -f "$OUT/arm.json" ] || die "no $OUT/arm.json: the arm has no engagement record (serve_arm.sh up writes it)"
  eng_ok=$(pj "$OUT/arm.json" engagement.ok 2>/dev/null || echo missing)
  [ "$eng_ok" = True ] || die "$OUT/arm.json says engagement.ok=$eng_ok: this server is not proven to be arm '$ARM'"
  [ -n "$BASE_URL" ] || BASE_URL="http://127.0.0.1:$(pj "$SERVING" port)"
  DP=$(pj "$OUT/arm.json" serve.dp_size 2>/dev/null || echo 8)
  CTXLEN=$(pj "$OUT/arm.json" serve.context_length 2>/dev/null || echo 0)
  isint "$CTXLEN" && [ "$CTXLEN" -ge 131072 ] || say "WARNING: context_length is $CTXLEN, not 131072: BFCL multi_turn_long_context items overflow and are scored as failures"
else
  BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
  DP=8
fi
isint "$DP" || DP=8
BFCL_WORKERS=${BFCL_WORKERS:-$((16 * DP))}

case " $LIST " in *" bfcl "*)
  [ "$PRINT" = 1 ] || [ -x "$BFCL_VENV/bin/python" ] || die "BFCL_VENV=$BFCL_VENV has no bin/python: prepare_box.sh bfcl" ;; esac
case " $LIST " in *" mmlupro "*|*" bfcl "*)
  [ "$PRINT" = 1 ] || [ -f "$EVAL_TOKENIZER/config.json" ] || die "EVAL_TOKENIZER=$EVAL_TOKENIZER has no config.json (the base model dir)" ;; esac

[ "$PRINT" = 1 ] || mkdir -p "$OUT/logs" "$OUT/receipts" "$OUT/results" "$OUT/journal" || die "cannot create $OUT"
export EVAL_DATA EVAL_TOKENIZER EVAL_BASE_URL="$BASE_URL" EVAL_SERVED_MODEL=${EVAL_SERVED_MODEL:-qwen3.8-27b}
export NLTK_DATA=${NLTK_DATA:-$EVAL_DATA/nltk_data}
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false

# ---------------------------------------------------------------------------------------------------------------
common_args() { # eval
  local n=$1 a=(--arm "$ARM" --out "$OUT" --base-url "$BASE_URL" --data "$EVAL_DATA")
  a+=(--stride "$(stride_of "$n")")
  [ "$LIMIT" -gt 0 ] && a+=(--limit "$LIMIT")
  [ "$FRESH" = 1 ] && a+=(--fresh)
  [ "$UNPINNED" = 1 ] && a+=(--allow-unpinned)
  printf '%s\n' "${a[@]}"
}

eval_cmd() { # eval -> the command line, one word per line
  local n=$1 extra
  extra=$(extra_of "$n")
  local words=("$EVAL_PY" "$HERE/$(SCRIPT_OF "$n")")
  while IFS= read -r w; do words+=("$w"); done < <(common_args "$n")
  case "$n" in
    mmlupro) words+=(--tokenizer "$EVAL_TOKENIZER") ;;
    bfcl) words+=(--bfcl-venv "$BFCL_VENV" --tokenizer "$EVAL_TOKENIZER" --workers "$BFCL_WORKERS") ;;
  esac
  # shellcheck disable=SC2206
  [ -n "$extra" ] && words+=($extra)
  printf '%s\n' "${words[@]}"
}

is_done() { # eval: complete result for this subset already on disk (and not --fresh)
  [ "$FRESH" = 0 ] || return 1
  local n=$1 res stride
  res=$OUT/results/$(RES_NAME "$n").json
  [ -f "$res" ] || return 1
  stride=$(stride_of "$n")
  "$EVAL_PY" -c 'import json,sys
d=json.load(open(sys.argv[1])); s=d.get("subset") or {}
ok=d.get("complete") is True and int(s.get("stride",1))==int(sys.argv[2]) and int(s.get("limit",0))==int(sys.argv[3]) and int(s.get("offset",0))==0
sys.exit(0 if ok else 1)' "$res" "$stride" "$LIMIT" 2>/dev/null
}

healthy() { [ "$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$BASE_URL/health" 2>/dev/null)" = 200 ]; }

timed() { timeout -k 30 "$PASS_TIMEOUT_S" "$@"; }

run_pre() { # eval -> 0 ok, else the rc to record (the eval is not run)
  local n=$1 log=$OUT/logs/$1.log rc
  case "$n" in
    mmlupro)
      [ "$SELFCHECK" -gt 0 ] || return 0
      local sc=$OUT/receipts/mmlupro.selfcheck.json
      if [ "$FRESH" = 0 ] && [ -f "$sc" ] && [ "$(pj "$sc" ok 2>/dev/null)" = True ]; then return 0; fi
      say "mmlupro: logprob selfcheck on $SELFCHECK items"
      local a=("$EVAL_PY" "$HERE/ev_mmlupro.py" --arm "$ARM" --out "$OUT" --base-url "$BASE_URL" --data "$EVAL_DATA"
               --tokenizer "$EVAL_TOKENIZER" --selfcheck "$SELFCHECK")
      [ "$UNPINNED" = 1 ] && a+=(--allow-unpinned)
      timed "${a[@]}" >>"$log" 2>&1; rc=$?
      [ "$rc" = 0 ] || { say "mmlupro: selfcheck exit $rc (see $log and $sc): the logprob semantics are not what the scorer assumes"; return "$rc"; }
      ;;
    humaneval)
      local cc=$OUT/receipts/humaneval.canonical.json
      if [ "$FRESH" = 0 ] && [ -f "$cc" ] && [ "$(pj "$cc" ok 2>/dev/null)" = True ]; then return 0; fi
      local a=("$EVAL_PY" "$HERE/ev_humaneval.py" --check-canonical --data "$EVAL_DATA")
      [ "$UNPINNED" = 1 ] && a+=(--allow-unpinned)
      timed "${a[@]}" >"$cc" 2>>"$log"; rc=$?
      [ "$rc" = 0 ] || { say "humaneval: the canonical solutions do not all pass in this sandbox (exit $rc, see $cc)"; return "$rc"; }
      ;;
  esac
  return 0
}

run_one() { # eval -> writes $STATE/<eval>.rc as "rc passes seconds"
  local n=$1 log=$OUT/logs/$1.log t0=$SECONDS pass=0 rc=0
  if is_done "$n"; then
    say "$n: complete result already on disk, skipped"
    echo "0 0 0 skipped" >"$STATE/$n.rc"; return 0
  fi
  run_pre "$n"; rc=$?
  if [ "$rc" != 0 ]; then echo "$rc 0 $((SECONDS - t0)) pre" >"$STATE/$n.rc"; return 0; fi
  local cmd=()
  while IFS= read -r w; do cmd+=("$w"); done < <(eval_cmd "$n")
  # --fresh applies to the first pass only: a retry must keep what the first pass finished
  while :; do
    pass=$((pass + 1))
    healthy || { say "$n: $BASE_URL/health is not 200"; echo "4 $pass $((SECONDS - t0)) down" >"$STATE/$n.rc"; return 0; }
    say "$n: pass $pass of $RETRIES"
    timed "${cmd[@]}" >>"$log" 2>&1; rc=$?
    local keep=()
    for w in "${cmd[@]}"; do [ "$w" = --fresh ] || keep+=("$w"); done
    cmd=("${keep[@]}")
    case "$rc" in
      0) break ;;
      3|124) [ "$pass" -ge "$RETRIES" ] && break
             say "$n: exit $rc (incomplete), retrying in ${BACKOFF_S}s"; sleep "$BACKOFF_S" ;;
      *) break ;;
    esac
  done
  say "$n: exit $rc after $pass pass(es), $((SECONDS - t0))s"
  echo "$rc $pass $((SECONDS - t0)) run" >"$STATE/$n.rc"
}

# ---------------------------------------------------------------------------------------------------------------
if [ "$PRINT" = 1 ]; then
  echo "arm=$ARM out=$OUT base_url=$BASE_URL evals=$LIST overlap=$OVERLAP retries=$RETRIES"
  for n in $LIST; do
    [ "$n" = mmlupro ] && [ "$SELFCHECK" -gt 0 ] && echo "pre mmlupro: selfcheck $SELFCHECK items"
    [ "$n" = humaneval ] && echo "pre humaneval: ev_humaneval.py --check-canonical"
    echo "cmd $n: $(eval_cmd "$n" | tr '\n' ' ')"
  done
  exit 0
fi

STATE=$OUT/receipts/.run.$$
mkdir -p "$STATE" || die "cannot create $STATE"
trap 'jobs -p | xargs -r kill 2>/dev/null; rm -rf "$STATE"' EXIT
T0=$SECONDS
START=$(date -u +%FT%TZ)
say "arm $ARM on $BASE_URL: ${LIST// /, } (retries $RETRIES, overlap $OVERLAP) -> $OUT"

BG_PID=""
FG=$LIST
if [ "$OVERLAP" = 1 ]; then
  case " $LIST " in *" bfcl "*)
    run_one bfcl & BG_PID=$!
    FG=""
    for n in $LIST; do [ "$n" = bfcl ] || FG="$FG $n"; done ;; esac
fi
for n in $FG; do
  run_one "$n"
  [ "$(cut -d' ' -f1 "$STATE/$n.rc")" = 4 ] && break
done
[ -n "$BG_PID" ] && wait "$BG_PID"

# ---------------------------------------------------------------------------------------------------------------
worst=0
down=0; err=0; inc=0
REPORT=$STATE/report.tsv
: >"$REPORT"
for n in $LIST; do
  if [ -f "$STATE/$n.rc" ]; then read -r rc passes secs how <"$STATE/$n.rc"; else rc=-1; passes=0; secs=0; how=notrun; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$n" "$rc" "$passes" "$secs" "$how" >>"$REPORT"
  case "$rc" in
    0) ;;
    4) down=1 ;;
    3|124) inc=1 ;;
    *) err=1 ;;
  esac
done
if [ "$down" = 1 ]; then worst=4; elif [ "$err" = 1 ]; then worst=1; elif [ "$inc" = 1 ]; then worst=3; fi

"$EVAL_PY" - "$REPORT" "$OUT/receipts/run_general.json" "$ARM" "$START" "$((SECONDS - T0))" "$worst" "$BASE_URL" "$LIMIT" <<'PYEOF'
import json, os, sys
rep, out, arm, start, secs, worst, url, limit = sys.argv[1:9]
evals = {}
for line in open(rep):
    n, rc, passes, s, how = line.rstrip("\n").split("\t")
    evals[n] = {"exit": int(rc), "passes": int(passes), "seconds": int(s), "how": how}
doc = {"schema": "negeig-general-run/1", "arm": arm, "started": start, "seconds": int(secs), "exit": int(worst),
       "base_url": url, "limit": int(limit), "evals": evals}
tmp = out + ".tmp"
with open(tmp, "w") as f:
    json.dump(doc, f, indent=1, sort_keys=True)
    f.write("\n")
os.replace(tmp, out)
PYEOF

say "summary (eval, exit, passes, seconds):"
while IFS=$'\t' read -r n rc passes secs how; do say "  $n exit=$rc passes=$passes seconds=$secs ($how)"; done <"$REPORT"
case "$worst" in
  0) say "all ${LIST// /, } complete in $((SECONDS - T0))s. Next: compare_general.py --base $EVAL_ROOT/base --arm $OUT --out-json $OUT/vs_base.json --out-md $OUT/vs_base.md" ;;
  3) say "incomplete: re-run the same command to resume; the failed items are in receipts/*.status.json" ;;
  4) say "the server went away: serve_arm.sh status, restart with serve_arm.sh up --replace, then re-run the same command" ;;
  *) say "an eval failed for good (see logs/<eval>.log); the others ran" ;;
esac
exit "$worst"
