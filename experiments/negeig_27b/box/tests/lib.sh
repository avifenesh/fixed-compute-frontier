#!/usr/bin/env bash
# Shared helpers for the box tests (CPU only: stub torchrun, fake provider, no GPU). Source it, call mk_box, then test.
# shellcheck disable=SC2034,SC2015  # E27_DIR is read by the tests that source this file; has/hasnt are A && ok || fail by design (ok cannot fail)
# Every test runs against a scratch tree under $TMPDIR (set by run_all.sh) and deletes it on exit.
set -uo pipefail
TESTS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BOX_DIR=$(cd "$TESTS_DIR/.." && pwd)
E27_DIR=$(cd "$BOX_DIR/.." && pwd)
T_PASS=0; T_FAIL=0
export PYTHONDONTWRITEBYTECODE=1  # no __pycache__ in the tree or in the read-only Hebrew lane
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/negeig-box-test.XXXXXX") || { echo "lib.sh: cannot make a scratch dir under ${TMPDIR:-/tmp}" >&2; exit 2; }
[ -d "$SCRATCH" ] || { echo "lib.sh: no scratch dir" >&2; exit 2; }
cleanup_scratch() { [ -n "${SCRATCH:-}" ] && rm -rf "$SCRATCH"; }
trap cleanup_scratch EXIT

ok()   { T_PASS=$((T_PASS + 1)); echo "  ok   $*"; }
fail() { T_FAIL=$((T_FAIL + 1)); echo "  FAIL $*"; }
check() { # description command...   (passes when the command succeeds)
  local d=$1; shift
  if "$@" >/dev/null 2>&1; then ok "$d"; else fail "$d   [$*]"; fi
}
check_not() { local d=$1; shift; if "$@" >/dev/null 2>&1; then fail "$d   [should fail: $*]"; else ok "$d"; fi; }
eq() { # description actual expected
  if [ "$2" = "$3" ]; then ok "$1"; else fail "$1: got '$2', want '$3'"; fi
}
has() { # description file-or-text pattern   (grep -E on a file if it is one, else on the text)
  if [ -f "$2" ]; then grep -Eq -- "$3" "$2" && ok "$1" || fail "$1: no /$3/ in $2"
  else printf '%s' "$2" | grep -Eq -- "$3" && ok "$1" || fail "$1: no /$3/ in text"; fi
}
hasnt() {
  if [ -f "$2" ]; then grep -Eq -- "$3" "$2" && fail "$1: /$3/ found in $2" || ok "$1"
  else printf '%s' "$2" | grep -Eq -- "$3" && fail "$1: /$3/ found in text" || ok "$1"; fi
}
section() { echo "== $*"; }
finish() { echo "$(basename "$0"): $T_PASS passed, $T_FAIL failed"; [ "$T_FAIL" -eq 0 ]; }

# mk_box NAME: a box tree under $SCRATCH/NAME with the stub trainer as the venv torchrun.
# Sets B (tree), and exports NEGEIG_W, EXPECT_GPUS, STUB_LOG, STUB_PLAN_DIR and the path variables the scripts read.
mk_box() {
  B=$SCRATCH/$1; rm -rf "$B"; mkdir -p "$B/venv/bin" "$B/model" "$B/data/s1" "$B/data/s4" "$B/data/sd" "$B/plan" "$B/runs"
  ln -s "$(command -v python3)" "$B/venv/bin/python"
  cat >"$B/venv/bin/torchrun" <<EOS
#!/usr/bin/env bash
exec python3 "$TESTS_DIR/stub_torchrun.py" "\$@"
EOS
  chmod +x "$B/venv/bin/torchrun"
  : >"$B/data/s4/replay_clean_q38.pt"
  echo '{"messages":[]}' >"$B/data/sd/chat_sft.jsonl"; echo '{"messages":[]}' >"$B/data/sd/tool_sft.jsonl"
  echo "NCCL_NVLS_ENABLE=0" >"$B/nccl_nvls.env"
  export NEGEIG_W=$B EXPECT_GPUS=8 STUB_LOG=$B/stub.log STUB_PLAN_DIR=$B/plan
  : >"$STUB_LOG"
  unset MODEL DATA RUNS SD PY PYBIN NVLS_FILE CHAT_REPLAY TOOL_REPLAY LM_REPLAY ARMS GPUS_PER_RUN STAGE ALLOW_NO_REPLAY \
    ALLOW_DEFAULT_LR ALLOW_NO_NVLS NCCL_NVLS_ENABLE ARM_MAX_ATTEMPTS ARM_MAX_NOPROGRESS
  export ARM_BACKOFF_S=0
}
plan() { local arm=$1; shift; : >"$B/plan/$arm.plan"; local x; for x in "$@"; do echo "$x" >>"$B/plan/$arm.plan"; done; rm -f "$B/plan/$arm.count"; }
launches() { # arm -> number of stub launches of that arm
  python3 - "$STUB_LOG" "$1" <<'PYEOF'
import json, sys
print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l)["arm"] == sys.argv[2]))
PYEOF
}
stub_field() { # arm index field -> that field of the arm's Nth launch (0-based)
  python3 - "$STUB_LOG" "$1" "$2" "$3" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if r["arm"] == sys.argv[2]]
v = rows[int(sys.argv[3])][sys.argv[4]]
print(" ".join(v) if isinstance(v, list) else v)
PYEOF
}

# mk_fake_ssh DIR: a fake `ssh` and a logging `rsync` in DIR (put DIR first on PATH). The fake ssh drops the options, takes the
# host and runs the rest through a local `bash -c`, the way a login shell gets it. Knobs (env): FAKE_SSH_LOG (every call, one line),
# FAKE_SSH_DOWN=1 (refuse), FAKE_SSH_PATTERN_FILE and FAKE_SSH_RELEASE (scripted silence), FAKE_SSH_SLURP=1 (read all of stdin);
# RSYNC_LOG (the rig-side rsync arguments).
mk_fake_ssh() {
local d=$1
mkdir -p "$d"
cat >"$d/ssh" <<'EOS'
#!/usr/bin/env bash
# fake ssh: drop the options, take the host, run the rest the way a login shell would
[ -z "${FAKE_SSH_LOG:-}" ] || printf '%s\n' "$*" >>"$FAKE_SSH_LOG"
if [ "${FAKE_SSH_DOWN:-0}" = 1 ]; then echo "ssh: connect to host 127.0.0.1 port 22: Connection refused" >&2; exit 255; fi
# FAKE_SSH_PATTERN_FILE holds one char per call, consumed from the front: f = no answer, . = answer. When the last char is
# taken the file FAKE_SSH_RELEASE is created (a waiting box command watches it). An empty or missing file means always answer.
if [ -n "${FAKE_SSH_PATTERN_FILE:-}" ] && [ -s "$FAKE_SSH_PATTERN_FILE" ]; then
  pat=$(cat "$FAKE_SSH_PATTERN_FILE"); c=${pat:0:1}; printf '%s' "${pat:1}" >"$FAKE_SSH_PATTERN_FILE"
  [ -n "${pat:1}" ] || { [ -z "${FAKE_SSH_RELEASE:-}" ] || : >"$FAKE_SSH_RELEASE"; }
  if [ "$c" = f ]; then echo "ssh: connect timed out" >&2; exit 255; fi
fi
while [ $# -gt 0 ]; do case "$1" in -i|-o|-p|-l|-F) shift 2 ;; -*) shift ;; *) break ;; esac; done
shift
# FAKE_SSH_SLURP=1: like a real ssh without -n, read ALL of the local stdin and hand it to the remote command as its stdin
if [ "${FAKE_SSH_SLURP:-0}" = 1 ]; then t=$(mktemp); cat >"$t"; exec 3<"$t"; rm -f "$t"; exec bash -c "$*" <&3; fi
exec bash -c "$*"
EOS
cat >"$d/rsync" <<EOS
#!/usr/bin/env bash
# logs the rig-side invocation, then runs the real rsync (which calls the fake ssh)
[ -z "\${RSYNC_LOG:-}" ] || printf '%s\n' "\$*" >>"\$RSYNC_LOG"
exec $(command -v rsync) "\$@"
EOS
chmod +x "$d/ssh" "$d/rsync"
}

# mk_accept_box NAME: mk_box plus everything accept.sh touches, all stubbed so the test never reaches a real GPU, the real
# nvidia-smi or a real SGLang server: stub nvidia-smi and systemctl first on PATH, a python shim (the `-` heredoc scripts
# and torch one-liners are answered, everything else runs python3), fake kernel_check.py and g0_noop.py, a fake verl venv whose
# `-m sglang.launch_server` is fake_sglang_server.py, a fake nvcc. Knobs (env): STUB_GPUS (8), STUB_GPU_USED_MIB (per GPU, 0),
# STUB_GPU_USED_IDX ("3:9000" overrides one GPU), STUB_GPU_TOTAL_MIB (275000), STUB_UNIT_ACTIVE=1, STUB_CUDA_RC, STUB_G1=fail,
# STUB_G0=fail, STUB_VERL_RC, STUB_SGLANG_DELAY/_ANSWER/_CRASH.
REAL_PY3=${REAL_PY3:-$(command -v python3)}   # absolute: the fake venv python3 sits first on PATH and must not exec itself
mk_accept_box() {
  mk_box "$1"
  local R=$B/root
  mkdir -p "$B/bin" "$R/verl/.venv/bin" "$B/cuda/bin" "$B/code/experiments/negeig_retrofit" "$B/accept"
  export ROOT=$R NEGEIG_CUDA_HOME=$B/cuda NO_SYSTEMD=0 SGLANG_POLL_S=1 PORT=$((31000 + RANDOM % 2000))
  export SGLANG_PROC_PATTERN="fake_[s]glang_server[.]py"
  export PATH="$B/bin:$PATH"
  unset STUB_GPUS STUB_GPU_USED_MIB STUB_GPU_USED_IDX STUB_GPU_TOTAL_MIB STUB_UNIT_ACTIVE STUB_CUDA_RC STUB_G1 STUB_G0 \
    STUB_VERL_RC STUB_SGLANG_DELAY STUB_SGLANG_ANSWER STUB_SGLANG_CRASH STUB_OOM_GC0 STUB_NCCL_MS0 STUB_NCCL_MS1 \
    STUB_NCCL_MS0_SEQ STUB_NCCL_MS1_SEQ STUB_NCCL_FAIL0 STUB_NCCL_FAIL1 STUB_NCCL_HANG1 STUB_NCCL_BUSBW NVLS_PIN NVLS_ROUNDS \
    NVLS_MIN_GAIN NCCL_PROBE_TIMEOUT_S MIN_NCCL_GBPS REQUIRE_REPLAY SMOKE_LR SMOKE_TIMEOUT_S MIN_TOK_S GRAD_CKPT_MAX_GIB \
    SMOKE_BASELINE_TOK_S SMOKE_MAX_USED_MIB STUB_RATE_PER_GPU STUB_SCALE_8 STUB_OFF_SLOPE STUB_SGLANG_REQLOG STUB_SGLANG_ARGLOG
  rm -f "$B/venv/bin/python"
  cat >"$B/venv/bin/python" <<'EOS'
#!/usr/bin/env bash
# python shim: the box scripts feed GPU-only code through `python -` and `python -c`; answer those, run the rest for real
if [ "$1" = "-" ]; then
  src=$(cat)
  case "$src" in
    *"alloc 8 GiB"*) echo "cuda_alloc stub on ${EXPECT_GPUS} devices"; exit "${STUB_CUDA_RC:-0}" ;;
  esac
  printf '%s\n' "$src" | python3 - "${@:2}"; exit $?
fi
if [ "$1" = "-c" ] && [[ "$2" == *"import torch"* ]]; then echo "torch stub 2.14 cuda 13.0"; exit 0; fi
exec python3 "$@"
EOS
  cat >"$R/verl/.venv/bin/python3" <<EOS
#!/usr/bin/env bash
if [ "\$1" = "-" ]; then cat >/dev/null; echo "sglang 0.5.20 stub verl env"; exit "\${STUB_VERL_RC:-0}"; fi
if [ "\$1" = "-m" ] && [ "\$2" = "sglang.launch_server" ]; then shift 2; exec "$REAL_PY3" "$TESTS_DIR/fake_sglang_server.py" "\$@"; fi
exec "$REAL_PY3" "\$@"
EOS
  printf '#!/usr/bin/env bash\necho "nvcc: stub release 13.0"; echo "Build stub"\n' >"$B/cuda/bin/nvcc"
  cat >"$B/bin/nvidia-smi" <<'EOS'
#!/usr/bin/env bash
# nvidia-smi stub: the query shapes accept.sh uses
n=${STUB_GPUS:-8}; q=""; one=""
for a in "$@"; do case "$a" in --query-gpu=*) q=${a#--query-gpu=} ;; esac; done
while [ $# -gt 0 ]; do case "$1" in -i) one=$2; shift 2 ;; *) shift ;; esac; done
used() { local i=$1 u=${STUB_GPU_USED_MIB:-0}; case ",${STUB_GPU_USED_IDX:-}," in *",$i:"*) u=$(echo ",${STUB_GPU_USED_IDX}," | sed -n "s/.*,$i:\([0-9]*\),.*/\1/p") ;; esac; echo "$u"; }
idxs=$(seq 0 $((n - 1))); [ -z "$one" ] || idxs=$one
case "$q" in
  index,name,compute_cap,driver_version,memory.total) echo "index, name, compute_capability, driver_version, memory.total [MiB]"
    for i in $idxs; do echo "$i, NVIDIA B300 SXM6 AC, 10.3, 580.65, ${STUB_GPU_TOTAL_MIB:-275000} MiB"; done ;;
  compute_cap) for i in $idxs; do echo "10.3"; done ;;
  name) for i in $idxs; do echo "NVIDIA B300 SXM6 AC"; done ;;
  index) for i in $idxs; do echo "$i"; done ;;
  memory.used) for i in $idxs; do used "$i"; done ;;
  memory.total) for i in $idxs; do echo "${STUB_GPU_TOTAL_MIB:-275000}"; done ;;
  index,memory.used) for i in $idxs; do echo "$i, $(used "$i")"; done ;;
  *) echo "nvidia-smi stub: unhandled query '$q'" >&2; exit 9 ;;
esac
EOS
  cat >"$B/bin/systemctl" <<'EOS'
#!/usr/bin/env bash
[ "$1" = "is-active" ] && { [ "${STUB_UNIT_ACTIVE:-0}" = 1 ]; exit $?; }
exit 0
EOS
  cat >"$B/code/experiments/negeig_retrofit/kernel_check.py" <<'EOS'
import json, sys
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "/dev/null"
import os
ok = os.environ.get("STUB_G1") != "fail"
json.dump({"pass": ok}, open(out, "w"))
print("G1 PASS" if ok else "G1 FAIL")
sys.exit(0 if ok else 1)
EOS
  cat >"$B/code/experiments/negeig_retrofit/g0_noop.py" <<'EOS'
import json, os, sys
ok = os.environ.get("STUB_G0") != "fail"
json.dump({"pass": ok, "bit_identical_at_w0": ok, "n_gdn_layers": 48, "n_gates": 48}, open(sys.argv[2], "w"))
sys.exit(0 if ok else 1)
EOS
  echo stub >"$B/model/.negeig_revision"
  chmod +x "$B/venv/bin/python" "$R/verl/.venv/bin/python3" "$B/cuda/bin/nvcc" "$B/bin/nvidia-smi" "$B/bin/systemctl"
}
