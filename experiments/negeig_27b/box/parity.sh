#!/usr/bin/env bash
# parity.sh - runs ON a box, after accept.sh. The SGLang parity gates G0 to G3 of sglang/README.md in one command:
# stock server, zeros server and random-gate server on GPU 0, the HF reference on another GPU, then check_parity.py compare.
#
#   parity.sh [--mode eager|graphs|both] [--target-t 0.5] [--skip G2,G3] [--hf-gpu N | --hf-after] [--fresh]
#
# What each gate claims (sglang/check_parity.py has the arithmetic; PASS needs all four):
#   G0 the stock server repeats itself bit for bit      G1 the patched server at W = 0 equals stock bit for bit
#   G2 random W: the patched server matches the HF patch within the gap between two stock implementations
#   G3 decode (packed kernel, one token) agrees with prefill as well as stock does
#
# Steps, in this order (each writes its output atomically; a rerun keeps what exists, --fresh clears $O first):
#   1 inputs   negeig_sglang.py static-check and install-plugin (verl venv), a check that SGLang discovers the plugin
#              through its entry point, check_parity.py prepare (prompt token ids) and make-random-gates
#   2 hf       the HF references hf_stock.json and hf_rand_t<T>.json (train venv, bf16 27B, about 54 GB) on GPU --hf-gpu
#              (default 1, in the background while the servers run on GPU 0); a one-GPU box runs them after the servers
#   3 servers  per mode: stock, zero (NEGEIG_GATES=zeros), rand (NEGEIG_GATES=<random gates>); every server is launched
#              with the SAME flags (the three --linear-attn-* triton flags, --disable-radix-cache, the same graph config),
#              because a stock server with different kernels would make G1 compare two kernels. The stock arm runs with
#              SGLANG_PLUGINS and NEGEIG_GATES unset. Two collects on stock (G0), one on zero and one on rand.
#              A patched server whose log has no "negeig: active" line, or a stock log with a "negeig:" line, stops the run.
#   4 compare  check_parity.py compare, per mode
# --mode eager serves with CUDA graphs disabled ({"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}),
# graphs with the defaults; both (default) runs eager first and goes on to graphs only when eager passed: a failure in
# eager is a math problem, a failure only in graphs is a graph problem (README, launch recipe).
#
# Result, the last line printed and the exit status:
#   PARITY_OK ...             exit 0   every gate PASS in every mode run; $O/PARITY_OK holds the line
#   PARITY_WAIVED ...         exit 0   --skip waived a gate: not a pass of that gate, no PARITY_OK file
#   PARITY_FAILED ...         exit 1   a gate FAILED (read the first differing layer before loosening anything)
#   PARITY_INCONCLUSIVE ...   exit 2   a gate is INCONCLUSIVE or SKIPPED: not a pass (see the verdict)
#   PARITY_BROKEN step=...    exit 3   the run itself broke: a tool failed, a server would not start or engage
# Receipts under $O ($ACC/parity): prompts.json, rand_gates_t<T>.safetensors, hf_*.json, <mode>/{stock,stock2,zero,
# rand_t<T>}.json and the server logs, <mode>/verdict_t<T>.json and compare_t<T>.txt, launch_<arm>.txt (the exact
# command and environment), parity.json (summary of this run), logs/.
#
# Env: PARITY_OUT ($ACC/parity)  PARITY_PORT (30002)  PARITY_POLLS (150 health polls of SGLANG_POLL_S, 25 minutes)
#      PARITY_MAX_USED_MIB (2048: GPU memory in use that refuses the run)  PARITY_COMPARE_ARGS (extra compare flags,
#      e.g. "--gap-ratio 3": the thresholds are starting points, README)  CHECK_PARITY and NEGEIG_SGLANG (the two scripts,
#      tests replace them)
# Refused while negeig27-train.service is active or a GPU in use holds memory: the servers need GPU 0 (and the HF job one more).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=boxenv.sh
. "$HERE/boxenv.sh"
# shellcheck source=sglang_lib.sh
. "$HERE/sglang_lib.sh"
E27_SRC=$(cd "$HERE/.." && pwd)
CHECK_PARITY=${CHECK_PARITY:-$E27_SRC/sglang/check_parity.py}
NEGEIG_SGLANG=${NEGEIG_SGLANG:-$E27_SRC/sglang/negeig_sglang.py}
O=${PARITY_OUT:-$ACC/parity}
PPORT=${PARITY_PORT:-30002}
POLLS=${PARITY_POLLS:-150}
MAX_USED=${PARITY_MAX_USED_MIB:-2048}
MODE=both; TARGET_T=0.5; SKIP=""; HF_GPU=1; HF_AFTER=0; FRESH=0; INTERNAL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE=$2; shift 2 ;;
    --target-t) TARGET_T=$2; shift 2 ;;
    --skip) SKIP=$2; shift 2 ;;
    --hf-gpu) HF_GPU=$2; shift 2 ;;
    --hf-after) HF_AFTER=1; shift ;;
    --fresh) FRESH=1; shift ;;
    --internal-hf) INTERNAL=hf; shift ;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) bdie "unknown argument: $1" ;;
  esac
done
case "$MODE" in eager|graphs|both) ;; *) bdie "--mode is eager, graphs or both" ;; esac
[[ $TARGET_T =~ ^[0-9]+(\.[0-9]+)?$ ]] || bdie "--target-t '$TARGET_T' is not a positive number"
[[ $HF_GPU =~ ^[0-9]+$ ]] || bdie "--hf-gpu '$HF_GPU' is not a GPU index"
[ -z "$SKIP" ] || [[ $SKIP =~ ^G[0-3](,G[0-3])*$ ]] || bdie "--skip takes a comma list of G0..G3"
case "$PPORT" in ''|*[!0-9]*) bdie "PARITY_PORT must be a whole number" ;; esac

PROMPTS=$O/prompts.json
GATES=$O/rand_gates_t$TARGET_T.safetensors
HF_STOCK=$O/hf_stock.json
HF_RAND=$O/hf_rand_t$TARGET_T.json
PLUGIN=$O/plugin
LOGD=$O/logs
EAGER_CFG='{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}'

broken() { echo "PARITY_BROKEN step=$1: $2" >&2; exit 3; }
nvidia_count() { nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c '[0-9]'; }
atomic() { # tmp final: move a finished output into place
  [ -s "$1" ] && mv -f "$1" "$2"
}

# ---- the HF reference job: the same script, re-entered, so it can run detached in its own process group
if [ "$INTERNAL" = hf ]; then
  [ -x "$PY" ] || exit 1
  for spec in "hf-stock:stock:$HF_STOCK" "hf-rand:$GATES:$HF_RAND"; do
    IFS=: read -r tag mode out <<<"$spec"
    if [ -s "$out" ]; then echo "$tag: kept from an earlier run ($out)"; continue; fi
    echo "[$(date -u +%FT%TZ)] hf-ref $tag mode=$mode on GPU $HF_GPU"
    CUDA_VISIBLE_DEVICES=$HF_GPU "$PY" "$CHECK_PARITY" hf-ref --model "$MODEL_DIR" --prompts "$PROMPTS" --out "$out.tmp" \
      --tag "$tag" --mode "$mode" --device cuda:0 || { echo "hf-ref $tag failed"; exit 1; }
    atomic "$out.tmp" "$out" || { echo "hf-ref $tag wrote no output"; exit 1; }
  done
  exit 0
fi

# ---- inputs and refusals, before anything is launched
[ -f "$MODEL_DIR/config.json" ] || bdie "no model at $MODEL_DIR (bootstrap.sh --stage model)"
[ -x "$VPY" ] || bdie "no verl venv at $VPY (bootstrap.sh --stage verl)"
[ -x "$PY" ] || bdie "no training venv at $PY (bootstrap.sh)"
[ -f "$CHECK_PARITY" ] || bdie "no $CHECK_PARITY (push_data.sh --code-only)"
[ -f "$NEGEIG_SGLANG" ] || bdie "no $NEGEIG_SGLANG (push_data.sh --code-only)"
command -v jq >/dev/null 2>&1 || bdie "jq is not installed"
if [ "${NO_SYSTEMD:-0}" != 1 ] && command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet "$TRAIN_UNIT" 2>/dev/null; then
  bdie "$TRAIN_UNIT is active: ctl.sh hold (or disarm) first, the servers need GPU 0"
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -n | tail -n 1)
[ "${used:-0}" -lt "$MAX_USED" ] || bdie "a GPU holds ${used} MiB (limit $MAX_USED): something else is running (a self-distillation server? sd_box.sh stop)"
NGPU=$(nvidia_count)
[ "${NGPU:-0}" -ge 1 ] || bdie "nvidia-smi lists no GPU"
if [ "$HF_AFTER" = 0 ] && [ "$NGPU" -lt 2 ]; then
  echo "one GPU on this box: the HF references run after the servers (--hf-after)"; HF_AFTER=1
fi
[ "$HF_AFTER" = 1 ] || [ "$HF_GPU" -lt "$NGPU" ] || bdie "--hf-gpu $HF_GPU but the box has $NGPU GPU(s)"
[ "$HF_AFTER" = 1 ] && HF_GPU=0
[ "$(curl -s -o /dev/null -m 3 -w '%{http_code}' "http://127.0.0.1:$PPORT/health" 2>/dev/null)" != 200 ] \
  || bdie "something already answers /health on port $PPORT (set PARITY_PORT)"
CH=$(find_cuda_home) || bdie "no nvcc under /usr/local/cuda*: flashinfer cannot JIT"

[ "$FRESH" = 0 ] || rm -rf "$O"
mkdir -p "$O" "$LOGD"
rm -f "$O/PARITY_OK" "$O/parity.json"   # a failed re-run must not leave an old success marker behind

HF_PID=""
# shellcheck disable=SC2329  # runs from the EXIT trap
cleanup() {
  [ -z "$HF_PID" ] || kill -TERM -- "-$HF_PID" 2>/dev/null
  [ -z "$SGLANG_PID" ] || stop_sglang >/dev/null 2>&1
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

step_run() { # step logname command...: run with the output in $LOGD/<logname>.log, the tail shown on failure
  local step=$1 name=$2; shift 2
  "$@" >"$LOGD/$name.log" 2>&1 || { tail -n 25 "$LOGD/$name.log" >&2; broken "$step" "'${*:1:4}' failed, log $LOGD/$name.log"; }
}

# ================================================================================================ 1 inputs
blog "== inputs"
step_run static_check static_check "$VPY" "$NEGEIG_SGLANG" static-check
step_run install_plugin install_plugin "$VPY" "$NEGEIG_SGLANG" install-plugin --dir "$PLUGIN"
[ -s "$PLUGIN/negeig_sglang.py" ] || broken install_plugin "no $PLUGIN/negeig_sglang.py after install-plugin"
# SGLang finds the plugin through the sglang.srt.plugins entry point of the dist-info install-plugin writes; a server
# that cannot find it would start stock and the zeros arm would then be a copy of stock.
if ! PYTHONPATH="$PLUGIN:${PYTHONPATH:-}" "$VPY" -c '
import importlib.metadata as m
eps = [e for e in m.entry_points(group="sglang.srt.plugins") if e.name == "negeig"]
print("sglang.srt.plugins entry points named negeig:", [e.value for e in eps])
raise SystemExit(0 if len(eps) == 1 else 1)' >"$LOGD/plugin_discovery.log" 2>&1; then
  cat "$LOGD/plugin_discovery.log" >&2; broken plugin_discovery "the negeig entry point is not discoverable from $PLUGIN"
fi
if [ ! -s "$PROMPTS" ]; then
  step_run prepare prepare "$PY" "$CHECK_PARITY" prepare --out "$PROMPTS.tmp" --tokenizer "$MODEL_DIR"
  atomic "$PROMPTS.tmp" "$PROMPTS" || broken prepare "no prompts written"
fi
if [ ! -s "$GATES" ]; then
  step_run make_random_gates make_random_gates "$PY" "$CHECK_PARITY" make-random-gates --config "$MODEL_DIR/config.json" \
    --out "$GATES.tmp" --target-t "$TARGET_T"
  # the real tool writes a safetensors file: a name that does not end in .safetensors is renamed on the way
  atomic "$GATES.tmp" "$GATES" || broken make_random_gates "no gates written"
fi
echo "inputs: prompts $PROMPTS, random gates $GATES (target-t $TARGET_T), plugin $PLUGIN"

# ================================================================================================ 2 hf (background)
start_hf() {
  [ ! -s "$HF_STOCK" ] || [ ! -s "$HF_RAND" ] || { echo "hf references: kept from an earlier run"; return 0; }
  env PARITY_OUT="$O" CHECK_PARITY="$CHECK_PARITY" setsid bash "$0" --target-t "$TARGET_T" \
    --hf-gpu "$HF_GPU" --internal-hf >"$LOGD/hf_ref.log" 2>&1 </dev/null &
  HF_PID=$!
  echo "hf references started in the background on GPU $HF_GPU (pid $HF_PID, log $LOGD/hf_ref.log)"
}
join_hf() {
  [ -n "$HF_PID" ] || return 0
  blog "waiting for the HF references (log $LOGD/hf_ref.log)"
  local rc
  wait "$HF_PID"; rc=$?
  HF_PID=""
  [ "$rc" -eq 0 ] || { tail -n 25 "$LOGD/hf_ref.log" >&2; broken hf "the HF reference job failed (rc $rc), log $LOGD/hf_ref.log"; }
  [ -s "$HF_STOCK" ] && [ -s "$HF_RAND" ] || broken hf "the HF reference job left no output"
}
[ "$HF_AFTER" = 1 ] || start_hf

# ================================================================================================ 3 servers
SERVER_FLAGS=(--tp-size 1 --host 127.0.0.1 --port "$PPORT" --mem-fraction-static 0.85 --context-length 8192
  --mamba-full-memory-ratio 3 --trust-remote-code
  --linear-attn-backend triton --linear-attn-prefill-backend triton --linear-attn-decode-backend triton --disable-radix-cache)

serve() { # arm(stock|zero|rand) mode log
  local arm=$1 mode=$2 slog=$3
  local -a flags=(--model-path "$MODEL_DIR" "${SERVER_FLAGS[@]}") envv=("PATH=${VPY%/*}:$CH/bin:$PATH" "CUDA_HOME=$CH" "CUDA_VISIBLE_DEVICES=0")
  [ "$mode" != eager ] || flags+=(--cuda-graph-config "$EAGER_CFG")
  local -a pre=()
  case "$arm" in
    stock) pre=(-u SGLANG_PLUGINS -u NEGEIG_GATES) ;;
    zero) envv+=("PYTHONPATH=$PLUGIN:${PYTHONPATH:-}" SGLANG_PLUGINS=negeig NEGEIG_GATES=zeros) ;;
    rand) envv+=("PYTHONPATH=$PLUGIN:${PYTHONPATH:-}" SGLANG_PLUGINS=negeig "NEGEIG_GATES=$GATES") ;;
  esac
  {
    echo "arm=$arm mode=$mode"
    echo "env: ${pre[*]:-} ${envv[*]:1}"
    echo "cmd: $VPY -m sglang.launch_server ${flags[*]}"
  } >"$O/launch_${arm}_$mode.txt"
  blog "serve $arm ($mode) on port $PPORT, log $slog"
  : >"$slog"
  env ${pre[@]+"${pre[@]}"} "${envv[@]}" setsid "$VPY" -m sglang.launch_server "${flags[@]}" >"$slog" 2>&1 </dev/null &
  SGLANG_PID=$!
  wait_sglang_healthy "$PPORT" "$SGLANG_PID" "$slog" "$POLLS" || { stop_sglang >/dev/null 2>&1; broken "serve_$arm" "the $arm server did not come up ($mode), log $slog"; }
  if [ "$arm" = stock ]; then
    if grep -q 'negeig:' "$slog"; then stop_sglang >/dev/null 2>&1; broken serve_stock "the stock server log has a negeig: line, so the stock arm was not stock: $slog"; fi
  else
    if ! grep -q 'negeig: active' "$slog"; then stop_sglang >/dev/null 2>&1; broken "serve_$arm" "no 'negeig: active' line in $slog: the plugin did not engage (SGLANG_PLUGINS, PYTHONPATH), this arm would be stock"; fi
    grep 'negeig: active' "$slog" | head -n 1
  fi
}

halt() { # stop the server and check the GPU is free before the next one takes it
  stop_sglang || broken stop "sglang would not stop"
  local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  [ "${u:-99999}" -lt "$MAX_USED" ] || broken stop "GPU 0 still holds ${u} MiB after stopping sglang"
}

collect() { # tag out log
  local tag=$1 out=$2 slog=$3
  rm -f "$out.tmp"
  if ! "$PY" "$CHECK_PARITY" collect --url "http://127.0.0.1:$PPORT" --prompts "$PROMPTS" --out "$out.tmp" --tag "$tag" \
      --server-log "$slog" >"$LOGD/collect_$tag.log" 2>&1; then
    tail -n 25 "$LOGD/collect_$tag.log" >&2; stop_sglang >/dev/null 2>&1; broken "collect_$tag" "check_parity collect --tag $tag failed, log $LOGD/collect_$tag.log"
  fi
  atomic "$out.tmp" "$out" || { stop_sglang >/dev/null 2>&1; broken "collect_$tag" "collect --tag $tag wrote no output"; }
  echo "collected $tag -> $out"
}

run_mode() { # mode -> sets MODE_RC (the compare exit status) and MODE_SUMMARY
  local mode=$1 md=$O/$1
  mkdir -p "$md"
  blog "== servers and collects: $mode"
  if [ ! -s "$md/stock.json" ] || [ ! -s "$md/stock2.json" ]; then
    rm -f "$md/stock.json" "$md/stock2.json"
    serve stock "$mode" "$md/stock.log"
    collect stock "$md/stock.json" "$md/stock.log"
    collect stock-repeat "$md/stock2.json" "$md/stock.log"
    halt
  else echo "stock ($mode): kept from an earlier run"; fi
  if [ ! -s "$md/zero.json" ]; then
    serve zero "$mode" "$md/zero.log"
    collect zero "$md/zero.json" "$md/zero.log"
    halt
  else echo "zero ($mode): kept from an earlier run"; fi
  if [ ! -s "$md/rand_t$TARGET_T.json" ]; then
    serve rand "$mode" "$md/rand_t$TARGET_T.log"
    collect rand "$md/rand_t$TARGET_T.json" "$md/rand_t$TARGET_T.log"
    halt
  else echo "rand ($mode, target-t $TARGET_T): kept from an earlier run"; fi
}

compare_mode() { # mode
  local mode=$1 md=$O/$1 v=$O/$1/verdict_t$TARGET_T.json
  join_hf
  blog "== compare: $mode"
  local -a args=(--stock "$md/stock.json" --stock-repeat "$md/stock2.json" --zero "$md/zero.json" --rand "$md/rand_t$TARGET_T.json"
                 --hf-stock "$HF_STOCK" --hf-rand "$HF_RAND" --out "$v")
  [ -z "$SKIP" ] || args+=(--skip "$SKIP")
  # shellcheck disable=SC2206  # PARITY_COMPARE_ARGS is a word list on purpose
  [ -z "${PARITY_COMPARE_ARGS:-}" ] || args+=(${PARITY_COMPARE_ARGS})
  rm -f "$v"
  "$PY" "$CHECK_PARITY" compare "${args[@]}" 2>&1 | tee "$md/compare_t$TARGET_T.txt" | grep -E '^G[0-3]:'
  MODE_RC=${PIPESTATUS[0]}
  [ -s "$v" ] || broken compare "check_parity compare wrote no verdict (rc $MODE_RC), see $md/compare_t$TARGET_T.txt"
  MODE_SUMMARY=$(jq -r '.summary | to_entries | map("\(.key)=\(.value)") | join(" ")' "$v")
  echo "PARITY_MODE $mode exit=$MODE_RC $MODE_SUMMARY"
}

MODES=("$MODE"); [ "$MODE" != both ] || MODES=(eager graphs)
RESULTS=(); FINAL_RC=0
for m in "${MODES[@]}"; do
  run_mode "$m"
  if [ "$HF_AFTER" = 1 ] && [ -z "$HF_PID" ] && { [ ! -s "$HF_STOCK" ] || [ ! -s "$HF_RAND" ]; }; then
    blog "== hf references (after the servers, GPU 0)"; start_hf
  fi
  compare_mode "$m"
  RESULTS+=("$m:$MODE_RC:$MODE_SUMMARY")
  if [ "$MODE_RC" -ne 0 ]; then
    FINAL_RC=$MODE_RC
    [ "${#MODES[@]}" -eq 1 ] || [ "$m" = "${MODES[${#MODES[@]} - 1]}" ] || echo "mode $m did not pass: the later mode is not run (a failure in eager is a math problem, not a graph problem)"
    break
  fi
done

# ================================================================================================ result
modes_run=$(printf '%s,' "${RESULTS[@]%%:*}"); modes_run=${modes_run%,}
last_summary=${RESULTS[${#RESULTS[@]} - 1]#*:*:}
jq -n --arg out "$O" --arg t "$TARGET_T" --arg modes "$modes_run" --arg skip "$SKIP" --argjson rc "$FINAL_RC" \
  --arg host "$(hostname)" --arg at "$(date -u +%FT%TZ)" --arg res "$(printf '%s\n' "${RESULTS[@]}")" \
  '{at_utc:$at, host:$host, modes_run:($modes|split(",")), target_t:($t|tonumber), skip:$skip, exit:$rc, results:($res|split("\n"))}' >"$O/parity.json"
case "$FINAL_RC" in
  0)
    if [ -n "$SKIP" ]; then
      echo "PARITY_WAIVED $(date -u +%FT%TZ) modes=$modes_run target_t=$TARGET_T waived=$SKIP $last_summary"
    else
      line="PARITY_OK $(date -u +%FT%TZ) modes=$modes_run target_t=$TARGET_T $last_summary receipts=$O"
      echo "$line" | tee "$O/PARITY_OK"
    fi
    exit 0 ;;
  1) echo "PARITY_FAILED modes=$modes_run $last_summary (verdict $O/${modes_run##*,}/verdict_t$TARGET_T.json)"; exit 1 ;;
  *) echo "PARITY_INCONCLUSIVE modes=$modes_run $last_summary (verdict $O/${modes_run##*,}/verdict_t$TARGET_T.json)"; exit 2 ;;
esac
