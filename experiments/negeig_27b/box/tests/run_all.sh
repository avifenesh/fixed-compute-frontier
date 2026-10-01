#!/usr/bin/env bash
# Every CPU check of the box scripts: bash -n and shellcheck over every script, python compile of the helpers, then each
# test under nice and taskset so the desktop stays responsive. No GPU, no network, no provider: stub torchrun, a fake
# SGLang server, a fake check_parity.py, stub nvidia-smi and ssh. Scratch lives under $TMPDIR and goes away on exit.
#
#   tests/run_all.sh [--quick] [test_name ...]     --quick skips the slow suites (test_parity.sh, test_sd.sh)
#   CPUS=0-3 (taskset list)   NICE=10   TMPDIR (default: a fresh dir under /tmp, removed at the end)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BOX=$(cd "$HERE/.." && pwd)
E27=$(cd "$BOX/.." && pwd)
CPUS=${CPUS:-0-3}
NICE=${NICE:-10}
QUICK=0; ONLY=()
for a in "$@"; do
  case "$a" in
    --quick) QUICK=1 ;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) ONLY+=("$a") ;;
  esac
done
OWN_TMP=0
if [ -z "${TMPDIR:-}" ] || [ ! -d "${TMPDIR:-}" ]; then TMPDIR=$(mktemp -d "/tmp/negeig-box-all.XXXXXX") && OWN_TMP=1; fi
export TMPDIR
# shellcheck disable=SC2329  # runs from the EXIT trap
cleanup() { [ "$OWN_TMP" = 0 ] || rm -rf "$TMPDIR"; }
trap cleanup EXIT
export PYTHONDONTWRITEBYTECODE=1
cap() { nice -n "$NICE" taskset -c "$CPUS" "$@"; }
FAILED=()

echo "== bash -n"
for f in "$BOX"/*.sh "$BOX"/train_pair.cmd.example "$HERE"/*.sh "$E27/run_pair.sh"; do
  bash -n "$f" || { echo "  FAIL bash -n $f"; FAILED+=("bash -n $(basename "$f")"); }
done
echo "== python compile"
for f in "$BOX"/*.py "$HERE"/*.py; do
  python3 - "$f" <<'PYEOF' || { echo "  FAIL compile $f"; FAILED+=("compile $(basename "$f")"); }
import ast, sys
ast.parse(open(sys.argv[1]).read(), sys.argv[1])
PYEOF
done
echo "== shellcheck (warnings and errors; SC2034 is off because boxenv.sh and common.sh define variables other scripts read)"
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck -x -P SCRIPTDIR -S warning -e SC2034 "$BOX"/*.sh "$BOX"/train_pair.cmd.example "$HERE"/*.sh "$E27/run_pair.sh" \
    || FAILED+=("shellcheck")
else
  echo "  FAIL shellcheck is not installed"; FAILED+=("shellcheck missing")
fi

SUITES=(test_run_pair test_wrapper test_smoke_report test_sweep_read test_accept test_rig test_push_data test_runbook)
[ "$QUICK" = 1 ] || SUITES+=(test_sd test_parity)
[ "${#ONLY[@]}" -eq 0 ] || SUITES=("${ONLY[@]}")
for t in "${SUITES[@]}"; do
  echo "== $t"
  start=$SECONDS
  out=$(cap timeout 900 bash "$HERE/$t.sh" 2>&1); rc=$?
  echo "$out" | grep -E '^  FAIL|passed,' || true
  echo "   ($t: rc=$rc, $((SECONDS - start)) s)"
  [ "$rc" -eq 0 ] || { FAILED+=("$t"); echo "$out" | tail -n 25; }
done

echo
if [ "${#FAILED[@]}" -eq 0 ]; then echo "run_all: every check passed"; exit 0; fi
echo "run_all: FAILED: ${FAILED[*]}"; exit 1
