#!/usr/bin/env bash
# run_tests.sh: the CPU tests of the general eval runner, capped so the machine stays responsive.
#
#   run_tests.sh                       every test module
#   run_tests.sh test_unit test_bfcl   the named modules (without .py)
#   run_tests.sh -k name               extra unittest arguments are passed through after the modules (use with a module)
#
# Env: EVAL_TEST_PY (interpreter, default ~/.venvs/negeig/bin/python, else python3), EVAL_TEST_CPUS (taskset list,
# default 8-15; skipped when the machine has no such CPUs), EVAL_TEST_TIMEOUT_S (default 1800), EVAL_TEST_DATA,
# EVAL_TEST_IFEVAL_PY, EVAL_TEST_NLTK_DATA, EVAL_TEST_BFCL (real-data tests skip when these are not set up, see README).
# No GPU is used or needed. The scratch dir is removed on exit. Exit code: unittest's.
set -u -o pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${EVAL_TEST_PY:-$HOME/.venvs/negeig/bin/python}
[ -x "$PY" ] || PY=$(command -v python3) || { echo "run_tests: no python" >&2; exit 2; }
CPUS=${EVAL_TEST_CPUS:-8-15}
TMO=${EVAL_TEST_TIMEOUT_S:-1800}
MODS=()
PASS=()
for a in "$@"; do
  case "$a" in
    test_*) MODS+=("$a") ;;
    *) PASS+=("$a") ;;
  esac
done
[ "${#MODS[@]}" -gt 0 ] || MODS=(test_unit test_evals test_run_general test_bfcl test_serve_arm)

TD=$(mktemp -d "${TMPDIR:-/tmp}/negeig-evtests.XXXXXX") || exit 2
trap 'rm -rf "$TD"' EXIT
CAP=(nice -n 10)
if command -v taskset >/dev/null 2>&1 && taskset -c "$CPUS" true 2>/dev/null; then CAP+=(taskset -c "$CPUS"); fi

cd "$HERE" || exit 2
TMPDIR=$TD "${CAP[@]}" env OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false \
  timeout "$TMO" "$PY" -W error::ResourceWarning -m unittest "${MODS[@]}" "${PASS[@]}"
