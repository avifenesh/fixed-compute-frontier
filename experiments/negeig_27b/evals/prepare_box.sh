#!/usr/bin/env bash
# prepare_box.sh: one-time setup of the general eval runner on the GPU box. Runs ON the box, after bootstrap.sh
# (needs uv, the repo mirror under $CODE and network). Nothing here touches a GPU or a running server.
#
#   prepare_box.sh all                 evals venv, BFCL venv, data, nltk data, then a self check
#   prepare_box.sh venv                $W/evalvenv: numpy, transformers 5.17.0, pyarrow, nltk, langdetect, ... (pinned)
#   prepare_box.sh bfcl                $W/bfcl-venv + $W/gorilla at the pinned commit, patched, editable install
#   prepare_box.sh data                prepare_data.py: pinned datasets and the IFEval harness into $W/evaldata
#   prepare_box.sh nltk                punkt and punkt_tab into $W/evaldata/nltk_data
#   prepare_box.sh check               verify everything above without changing it (reports every gap, exit 1 if any)
#
# Environment overrides: NEGEIG_W (workspace root), EVAL_PYTHON (default 3.12 for uv), PREP_OFFLINE=1 (skip network
# steps that already have their result; fail where one is needed), UV_BIN.
# Exit codes: 0 done, 1 a step failed, 2 usage.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E27="$(cd "$HERE/.." && pwd)"
BOXDIR=${BOXDIR:-$E27/box}
# shellcheck disable=SC1091
. "$BOXDIR/boxenv.sh"

EVALVENV=${EVALVENV:-$W/evalvenv}
BFCLVENV=${BFCLVENV:-$W/bfcl-venv}
GORILLA=${GORILLA:-$W/gorilla}
EVAL_DATA=${EVAL_DATA:-$W/evaldata}
EVAL_PYTHON=${EVAL_PYTHON:-3.12}
UV=${UV_BIN:-uv}
PINS=$HERE/pins.json
EVPY=$EVALVENV/bin/python

usage() { sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "prepare_box: ERROR: $1" >&2; exit "${2:-1}"; }
say() { echo "[$(date -u +%FT%TZ)] prepare_box: $*" >&2; }

# any_py: an interpreter for the stdlib-only helpers (pins.json reads, bfcl_patch.py): the training venv python when
# it exists, else python3.
any_py() {
  if [ -x "$PY" ]; then echo "$PY"; else command -v python3; fi
}

# pin PATH: a value out of pins.json (dotted path, e.g. bfcl.commit).
pin() {
  local py
  py=$(any_py) || die "no python to read $PINS"
  "$py" - "$PINS" "$1" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    d = d[k]
print(d)
PYEOF
}

need_uv() {
  command -v "$UV" >/dev/null 2>&1 && return 0
  [ "${PREP_OFFLINE:-0}" = 1 ] && die "uv is missing and PREP_OFFLINE=1"
  say "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv install failed"
  export PATH=$HOME/.local/bin:$PATH
  command -v "$UV" >/dev/null 2>&1 || die "uv still missing after the install"
}

# ----------------------------------------------------------------------------------------------------------
do_venv() {
  need_uv
  local nltk lang imm absl
  nltk=$(pin ifeval_harness.python_deps.nltk) || die "pins"
  lang=$(pin ifeval_harness.python_deps.langdetect) || die "pins"
  imm=$(pin ifeval_harness.python_deps.immutabledict) || die "pins"
  absl=$(pin ifeval_harness.python_deps.absl-py) || die "pins"
  [ -x "$EVPY" ] || "$UV" venv --python "$EVAL_PYTHON" "$EVALVENV" || die "uv venv $EVALVENV failed"
  say "installing the evals venv into $EVALVENV"
  "$UV" pip install --python "$EVPY" "numpy" "pyarrow" "transformers==5.17.0" "tokenizers" "huggingface_hub" \
    "nltk==$nltk" "langdetect==$lang" "immutabledict==$imm" "absl-py==$absl" || die "uv pip install failed"
  check_venv || die "the evals venv failed its own check"
  say "evals venv ready: $EVPY"
}

check_venv() {
  [ -x "$EVPY" ] || { echo "prepare_box: no evals venv python at $EVPY (run: prepare_box.sh venv)" >&2; return 1; }
  "$EVPY" - "$PINS" <<'PYEOF' || return 1
import importlib.metadata as m, json, sys
pins = json.load(open(sys.argv[1]))
want = dict(pins["ifeval_harness"]["python_deps"])
want["transformers"] = "5.17.0"
bad = []
for name, ver in want.items():
    try:
        got = m.version(name)
    except m.PackageNotFoundError:
        bad.append(f"{name} missing (want {ver})")
        continue
    if got != ver:
        bad.append(f"{name} {got} (want {ver})")
for name in ("numpy", "pyarrow", "tokenizers"):
    try:
        m.version(name)
    except m.PackageNotFoundError:
        bad.append(f"{name} missing")
if bad:
    print("prepare_box: evals venv problems: " + "; ".join(bad), file=sys.stderr)
    sys.exit(1)
PYEOF
}

# ----------------------------------------------------------------------------------------------------------
do_bfcl() {
  need_uv
  local commit repo want_req got_req
  commit=$(pin bfcl.commit) || die "pins"
  repo=$(pin bfcl.repo) || die "pins"
  want_req=$(pin bfcl.requirements_sha256) || die "pins"
  got_req=$(sha256sum "$HERE/bfcl/requirements.txt" | cut -d' ' -f1)
  [ "$got_req" = "$want_req" ] || die "bfcl/requirements.txt sha256 $got_req != pinned $want_req"

  if [ ! -d "$GORILLA/.git" ]; then
    [ "${PREP_OFFLINE:-0}" = 1 ] && die "no $GORILLA and PREP_OFFLINE=1"
    say "fetching $repo at $commit (shallow)"
    mkdir -p "$GORILLA" || die "cannot create $GORILLA"
    git -C "$GORILLA" init -q || die "git init"
    git -C "$GORILLA" remote add origin "$repo" || die "git remote add"
    git -C "$GORILLA" fetch -q --depth 1 origin "$commit" || die "git fetch of $commit failed"
    git -C "$GORILLA" checkout -q --detach FETCH_HEAD || die "git checkout"
  fi
  local head
  head=$(git -C "$GORILLA" rev-parse HEAD) || die "not a git checkout: $GORILLA"
  [ "$head" = "$commit" ] || die "$GORILLA is at $head, pinned $commit (remove the directory to refetch)"

  # the patch is idempotent: a patched tree is a no-op, a tree that is neither clean nor patched exits 2
  "$(any_py)" "$HERE/bfcl/bfcl_patch.py" --gorilla "$GORILLA" >/dev/null || die "bfcl_patch.py failed on $GORILLA"

  [ -x "$BFCLVENV/bin/python" ] || "$UV" venv --python "$(pin bfcl.python)" "$BFCLVENV" || die "uv venv $BFCLVENV failed"
  say "installing the pinned BFCL requirements into $BFCLVENV"
  "$UV" pip install --python "$BFCLVENV/bin/python" -r "$HERE/bfcl/requirements.txt" || die "requirements install failed"
  say "installing the patched BFCL package (editable, no deps)"
  SETUPTOOLS_SCM_PRETEND_VERSION=0+negeig \
    "$UV" pip install --python "$BFCLVENV/bin/python" --no-deps -e "$GORILLA/berkeley-function-call-leaderboard" \
    || die "editable install failed"
  check_bfcl || die "the BFCL venv failed its own check"
  say "BFCL ready: venv $BFCLVENV, tree $GORILLA"
}

check_bfcl() {
  local py
  py=$(any_py) || return 1
  [ -x "$BFCLVENV/bin/python" ] && [ -x "$BFCLVENV/bin/bfcl" ] \
    || { echo "prepare_box: no BFCL venv with bin/bfcl at $BFCLVENV (run: prepare_box.sh bfcl)" >&2; return 1; }
  [ -d "$GORILLA/.git" ] || { echo "prepare_box: no gorilla checkout at $GORILLA" >&2; return 1; }
  "$py" "$HERE/bfcl/bfcl_patch.py" --gorilla "$GORILLA" --check >/dev/null 2>&1 \
    || { echo "prepare_box: $GORILLA is not patched (bfcl_patch.py --check)" >&2; return 1; }
  "$BFCLVENV/bin/python" - "$GORILLA" "$(pin bfcl.registry_key)" <<'PYEOF' || return 1
import pathlib, sys
import bfcl_eval
root = pathlib.Path(bfcl_eval.__file__).resolve().parent
want = pathlib.Path(sys.argv[1]).resolve()
if want not in root.parents:
    print(f"prepare_box: bfcl_eval imports from {root}, not from {want} (not the editable patched tree)", file=sys.stderr)
    sys.exit(1)
mc = (root / "constants" / "model_config.py").read_text()
if f'"{sys.argv[2]}": ModelConfig(' not in mc:
    print(f"prepare_box: registry key {sys.argv[2]} missing from {root}/constants/model_config.py", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# ----------------------------------------------------------------------------------------------------------
do_data() {
  [ -x "$EVPY" ] || die "no evals venv (run: prepare_box.sh venv)"
  mkdir -p "$EVAL_DATA" || die "cannot create $EVAL_DATA"
  say "preparing the pinned datasets into $EVAL_DATA"
  "$EVPY" "$HERE/prepare_data.py" --out "$EVAL_DATA" || die "prepare_data.py failed"
  check_data || die "the data failed its own check"
}

check_data() {
  local f
  for f in mmlupro_test.jsonl humaneval_test.jsonl gmmlu_he_test.jsonl gmmlu_he_dev.jsonl ifeval_input_data.jsonl MANIFEST.json; do
    [ -s "$EVAL_DATA/$f" ] || { echo "prepare_box: missing $EVAL_DATA/$f (run: prepare_box.sh data)" >&2; return 1; }
  done
  [ -d "$EVAL_DATA/ifeval_harness" ] || { echo "prepare_box: missing $EVAL_DATA/ifeval_harness" >&2; return 1; }
  "$EVPY" - "$PINS" "$EVAL_DATA" <<'PYEOF' || return 1
import hashlib, json, pathlib, sys
pins = json.load(open(sys.argv[1]))
d = pathlib.Path(sys.argv[2])
bad = []
for name, want in pins["normalized_sha256"].items():
    got = hashlib.sha256((d / name).read_bytes()).hexdigest()
    if got != want:
        bad.append(f"{name} {got[:12]} (pinned {want[:12]})")
if bad:
    print("prepare_box: normalized data differs from the pins: " + "; ".join(bad), file=sys.stderr)
    sys.exit(1)
PYEOF
}

# ----------------------------------------------------------------------------------------------------------
do_nltk() {
  [ -x "$EVPY" ] || die "no evals venv (run: prepare_box.sh venv)"
  mkdir -p "$EVAL_DATA/nltk_data" || die "cannot create $EVAL_DATA/nltk_data"
  if ! check_nltk 2>/dev/null; then
    [ "${PREP_OFFLINE:-0}" = 1 ] && die "nltk data missing and PREP_OFFLINE=1"
    say "downloading punkt and punkt_tab into $EVAL_DATA/nltk_data"
    # nltk 3.10 refuses a download dir when it or an ancestor is group- or world-writable (a box umask of 002 makes
    # /workspace that way), and on that error its downloader waits on stdin forever. Tighten the chain, never block.
    local d="$EVAL_DATA/nltk_data"
    while [ "$d" != / ] && [ -O "$d" ]; do chmod go-w "$d" 2>/dev/null; d=$(dirname "$d"); done
    "$EVPY" -m nltk.downloader -d "$EVAL_DATA/nltk_data" punkt punkt_tab </dev/null || die "nltk download failed"
  fi
  check_nltk || die "nltk data failed its own check"
}

check_nltk() {
  [ -x "$EVPY" ] || return 1
  NLTK_DATA="$EVAL_DATA/nltk_data" "$EVPY" - 2>/dev/null <<'PYEOF' || { echo "prepare_box: punkt or punkt_tab missing in $EVAL_DATA/nltk_data (run: prepare_box.sh nltk)" >&2; return 1; }
import nltk
nltk.data.find("tokenizers/punkt")
nltk.data.find("tokenizers/punkt_tab")
PYEOF
}

do_check() {
  local rc=0
  check_venv || rc=1
  check_bfcl || rc=1
  check_data || rc=1
  check_nltk || rc=1
  [ "$rc" = 0 ] && say "check: all good"
  return $rc
}

# ----------------------------------------------------------------------------------------------------------
[ $# -ge 1 ] || { usage >&2; exit 2; }
case "$1" in
  -h|--help) usage; exit 2 ;;
  venv) do_venv ;;
  bfcl) do_bfcl ;;
  data) do_data ;;
  nltk) do_nltk ;;
  check) do_check || exit 1 ;;
  all) do_venv; do_bfcl; do_data; do_nltk; do_check || exit 1 ;;
  *) echo "prepare_box: unknown mode '$1'" >&2; usage >&2; exit 2 ;;
esac
exit 0
