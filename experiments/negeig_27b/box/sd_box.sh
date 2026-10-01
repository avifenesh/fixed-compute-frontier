#!/usr/bin/env bash
# sd_box.sh - runs ON a box. Self-distillation of the UNTOUCHED Qwen3.8-27B: serve it, generate the chat and tool replay,
# check the files. Every label is the untouched model's own output (think-off), never another model's text.
#
#   sd_box.sh serve    [--port 30000] [--dp 8]   SGLang 0.5.20 from the verl venv: DP over every GPU, think-off, the
#                                                qwen3_coder tool-call parser and the qwen3 reasoning parser. Detached;
#                                                returns when /health answers and one chat request is answered.
#   sd_box.sh chat     [chat_gen.py args]        translations + answers          -> $SD_OUT/chat_sft.jsonl   (box A)
#   sd_box.sh tools    [tool_gen.py args]        agentic pool, N samples/item    -> $SD_OUT/tool_sft.jsonl   (box B)
#   sd_box.sh status                             server, generators, ledger row counts, the two files
#   sd_box.sh finalize [--refilter]              check this box's half (a generator already filters when it ends);
#                                                --refilter reruns run.py filter over the ledgers (no server needed)
#   sd_box.sh lengths  [sd_lengths.py args]      the length check on $SD (both files): rows over --max_len 12288, which
#                                                train27 drops, and rows that do not render. Receipt $ACC/sd_lengths.json
#   sd_box.sh stop     [--force]                 stop the server (refused while a generator runs unless --force)
#
# Two boxes split the work: box A runs `serve; chat`, box B runs `serve; tools`, each ends with `finalize`; then the RIG
# runs sd_sync.sh, which moves chat_sft.jsonl and tool_sft.jsonl onto BOTH boxes ($SD = $W/data/sd, where run_pair.sh and
# accept.sh read them); then `lengths` on each box; then `stop`. The boxes cannot reach each other or the rig.
#
# chat and tools start detached (setsid, log in $LOGS/sd_<name>.log, exit status in $SD_OUT/<name>.exit: 0 done, 2
# aborted because the server went away or a request failed every retry, anything else a crash). Rerun the same command
# after a preemption or an abort: every ledger is resumable. SD_FG=1 runs in the foreground and returns the exit status.
# Anything after the command is passed to the generator last, so it wins over the defaults here (smoke runs:
# `chat --limit 50 --n-translate 10`, `tools --item-limit-per-kind 2 --n-samples 2`).
#
# Env: SD_OUT ($W/selfdistill) SD_PORT (30000) SD_DP (EXPECT_GPUS) SD_CONCURRENCY (256) SD_CONTEXT (32768)
#      SD_POLLS (240 health polls) SD_MAX_USED_MIB (2048: GPU memory in use that refuses `serve`) SD_FG
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=boxenv.sh
. "$HERE/boxenv.sh"
# shellcheck source=sglang_lib.sh
. "$HERE/sglang_lib.sh"
E27_SRC=$(cd "$HERE/.." && pwd)
SDGEN=$E27_SRC/selfdistill
SD_OUT=${SD_OUT:-$W/selfdistill}
SD_PORT=${SD_PORT:-30000}
SD_DP=${SD_DP:-$EXPECT_GPUS}
SD_CONCURRENCY=${SD_CONCURRENCY:-256}
SD_CONTEXT=${SD_CONTEXT:-32768}
SD_POLLS=${SD_POLLS:-240}
SD_MAX_USED_MIB=${SD_MAX_USED_MIB:-2048}
SD_PIDFILE=$RUN_DIR/sd_sglang.pid
SD_LOG=$LOGS/sd_sglang.log
PROMPTS=$DATA/s4/chat_prompts.jsonl
mkdir -p "$RUN_DIR" "$LOGS" "$SD_OUT" "$ACC"

alive() { local p; p=$(cat "$1" 2>/dev/null) && [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
healthy() { [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$SD_PORT/health" 2>/dev/null)" = 200 ]; }
nrows() { if [ -s "$1" ]; then wc -l <"$1" | tr -d ' '; else echo 0; fi; }
max_gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -n | tail -n 1; }

cmd_serve() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --port) SD_PORT=$2; shift 2 ;;
      --dp) SD_DP=$2; shift 2 ;;
      *) bdie "serve: unknown argument $1" ;;
    esac
  done
  if alive "$SD_PIDFILE" && healthy; then blog "already serving on port $SD_PORT (pid $(cat "$SD_PIDFILE"))"; return 0; fi
  if alive "$SD_PIDFILE"; then bdie "a server (pid $(cat "$SD_PIDFILE")) is up but not healthy on port $SD_PORT: sd_box.sh stop --force, then serve again"; fi
  [ -f "$MODEL_DIR/config.json" ] || bdie "no model at $MODEL_DIR (bootstrap.sh --stage model)"
  [ -x "$VPY" ] || bdie "no verl venv at $VPY (bootstrap.sh --stage verl)"
  if [ "${NO_SYSTEMD:-0}" != 1 ] && command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet "$TRAIN_UNIT" 2>/dev/null; then
    bdie "$TRAIN_UNIT is active: ctl.sh hold (or disarm) first, the server needs every GPU"
  fi
  local used; used=$(max_gpu_used)
  [ "${used:-0}" -lt "$SD_MAX_USED_MIB" ] || bdie "a GPU holds ${used} MiB (limit $SD_MAX_USED_MIB): something else is running, the server needs every GPU"
  local ch; ch=$(find_cuda_home) || bdie "no nvcc under /usr/local/cuda*: flashinfer cannot JIT"
  blog "serving the untouched $MODEL_ID on port $SD_PORT, dp $SD_DP, context $SD_CONTEXT; log $SD_LOG"
  : >"$SD_LOG"
  # the verl venv bin carries ninja, which the flashinfer JIT runs by name
  PATH="${VPY%/*}:$ch/bin:$PATH" CUDA_HOME=$ch setsid "$VPY" -m sglang.launch_server \
    --model-path "$MODEL_DIR" --tp-size 1 --dp-size "$SD_DP" --host 127.0.0.1 --port "$SD_PORT" \
    --mem-fraction-static 0.85 --context-length "$SD_CONTEXT" --mamba-full-memory-ratio 3 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --trust-remote-code >"$SD_LOG" 2>&1 </dev/null &
  SGLANG_PID=$!
  echo "$SGLANG_PID" >"$SD_PIDFILE"
  disown "$SGLANG_PID" 2>/dev/null || true
  # the first launch on this architecture JIT-compiles kernels and captures graphs on every replica: 40 minutes by default
  if ! wait_sglang_healthy "$SD_PORT" "$SGLANG_PID" "$SD_LOG" "$SD_POLLS"; then
    stop_sglang >/dev/null 2>&1; rm -f "$SD_PIDFILE"; return 1
  fi
  # One question with a checkable answer, think-off: proves the chat route and the template switch before hours of work.
  local resp content
  resp=$(curl -s -m 300 "http://127.0.0.1:$SD_PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model": "default", "max_tokens": 64, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [{"role": "user", "content": "What is 17 times 23? Answer with the number only."}]}')
  echo "$resp" >"$ACC/sd_serve_probe.json"
  content=$(echo "$resp" | jq -r '.choices[0].message.content // empty')
  blog "probe answer: $content"
  if ! echo "$content" | grep -q 391; then
    echo "SD_SERVE_FAILED probe: expected 391, see $ACC/sd_serve_probe.json" >&2
    stop_sglang >/dev/null 2>&1; rm -f "$SD_PIDFILE"; return 1
  fi
  echo "SD_SERVE_OK port=$SD_PORT dp=$SD_DP pid=$SGLANG_PID"
}

# run_gen NAME SCRIPT ARGS...: the generator against the local server. Detached unless SD_FG=1.
run_gen() {
  local name=$1 script=$2; shift 2
  local pidf=$RUN_DIR/sd_$name.pid exitf=$SD_OUT/$name.exit glog=$LOGS/sd_$name.log
  alive "$pidf" && bdie "$name is already running (pid $(cat "$pidf")), log $glog"
  healthy || bdie "no healthy server on port $SD_PORT: sd_box.sh serve first"
  [ -x "$PY" ] || bdie "no training venv at $PY (bootstrap.sh)"
  [ -f "$SDGEN/$script" ] || bdie "no $SDGEN/$script (push_data.sh --code-only)"
  [ -f "$LANE_DIR/agentic_env.py" ] || bdie "no lane at $LANE_DIR: push_data.sh <box> --hebrew-lane <research/hebrew-rl-20260923>"
  rm -f "$exitf"
  local gcfg=$MODEL_DIR/generation_config.json
  # Defaults point at the rig's paths, so the paths are always explicit. The tools run writes tool_sft.jsonl, the chat
  # run chat_sft.jsonl; each filters its own ledgers when it ends.
  local -a cmd=(env "SELFDISTILL_LANE_DIR=$LANE_DIR" "SELFDISTILL_BASE_URL=http://127.0.0.1:$SD_PORT"
                "$PY" "$script" --base-url "http://127.0.0.1:$SD_PORT" --out-dir "$SD_OUT" --generation-config "$gcfg"
                --concurrency "$SD_CONCURRENCY" "$@")
  if [ "${SD_FG:-0}" = 1 ]; then
    blog "$name: foreground, ${cmd[*]:4}"
    (cd "$SDGEN" && "${cmd[@]}") 2>&1 | tee -a "$glog"
    local rc=${PIPESTATUS[0]}
    echo "$rc" >"$exitf"
    echo "SD_${name^^}_EXIT $rc"
    return "$rc"
  fi
  blog "$name: detached, log $glog, exit status in $exitf"
  # shellcheck disable=SC2016  # the single-quoted script is bash -c's own, it expands $1 $2 itself
  setsid bash -c 'cd "$1" && exitf=$2 && shift 2 && "$@"; rc=$?; echo "$rc" >"$exitf.tmp"; mv "$exitf.tmp" "$exitf"' \
    _ "$SDGEN" "$exitf" "${cmd[@]}" >>"$glog" 2>&1 </dev/null &
  echo "$!" >"$pidf"
  disown "$!" 2>/dev/null || true
  echo "SD_${name^^}_STARTED pid=$(cat "$pidf")"
}

cmd_chat() {
  [ -s "$PROMPTS" ] || bdie "no chat prompts at $PROMPTS (push_data.sh pushes data/s4)"
  run_gen chat chat_gen.py --prompts "$PROMPTS" "$@"
}
cmd_tools() { run_gen tools tool_gen.py "$@"; }

gen_state() { # name -> running | done | aborted | crashed(N) | not started
  local name=$1 e=$SD_OUT/$1.exit
  if alive "$RUN_DIR/sd_$name.pid"; then echo running
  elif [ -f "$e" ]; then
    case "$(cat "$e")" in 0) echo "done" ;; 2) echo "aborted (rerun to resume)" ;; *) echo "crashed (exit $(cat "$e"), see $LOGS/sd_$name.log)" ;; esac
  else echo "not started"; fi
}

cmd_status() {
  local srv=down
  if alive "$SD_PIDFILE"; then if healthy; then srv="healthy (pid $(cat "$SD_PIDFILE"))"; else srv="up, not healthy (pid $(cat "$SD_PIDFILE"))"; fi; fi
  echo "server  : $srv, port $SD_PORT"
  echo "chat    : $(gen_state chat)"
  echo "tools   : $(gen_state tools)"
  local f
  for f in translations.jsonl chat_raw.jsonl tool_raw.jsonl chat_sft.jsonl tool_sft.jsonl; do
    printf 'rows    : %-18s %s\n' "$f" "$(nrows "$SD_OUT/$f")"
  done
  for f in chat tools; do [ ! -f "$LOGS/sd_$f.log" ] || { echo "--- tail of $LOGS/sd_$f.log"; tail -n 3 "$LOGS/sd_$f.log"; }; done
  for f in chat_sft.jsonl tool_sft.jsonl; do printf 'replay  : %-18s %s rows in %s\n' "$f" "$(nrows "$SD/$f")" "$SD"; done
}

cmd_finalize() {
  local refilter=0
  [ "${1:-}" != --refilter ] || refilter=1
  if [ "$refilter" = 1 ]; then
    # run.py filter drops the "meta" block (model, sampling) from the rows it rewrites; the generators write it. Use it
    # to re-read the filters after changing them, not as the normal finish.
    (cd "$SDGEN" && SELFDISTILL_LANE_DIR=$LANE_DIR "$PY" run.py filter --out-dir "$SD_OUT" >"$LOGS/sd_refilter.log") || bdie "run.py filter failed, see $LOGS/sd_refilter.log"
    blog "NOTE: the refilter rewrote the rows without meta.model and meta.sampling; manifest_*.json under $SD_OUT still name them"
  fi
  local half=0 n
  if [ -s "$SD_OUT/chat_raw.jsonl" ]; then
    half=1; n=$(nrows "$SD_OUT/chat_sft.jsonl")
    [ "$n" -gt 0 ] || bdie "chat_raw.jsonl has rows but chat_sft.jsonl is empty or missing: sd_box.sh chat to finish, or finalize --refilter"
  fi
  if [ -s "$SD_OUT/tool_raw.jsonl" ]; then
    half=1; n=$(nrows "$SD_OUT/tool_sft.jsonl")
    [ "$n" -gt 0 ] || bdie "tool_raw.jsonl has rows but tool_sft.jsonl is empty or missing: sd_box.sh tools to finish, or finalize --refilter"
  fi
  [ "$half" = 1 ] || bdie "no ledgers in $SD_OUT: nothing was generated on this box"
  local g
  for g in chat tools; do
    case "$(gen_state $g)" in running*) bdie "$g is still running" ;; aborted*|crashed*) bdie "$g did not finish: $(gen_state $g)" ;; esac
  done
  echo "SD_FINALIZE_OK chat_sft=$(nrows "$SD_OUT/chat_sft.jsonl") tool_sft=$(nrows "$SD_OUT/tool_sft.jsonl") out=$SD_OUT"
}

cmd_lengths() {
  [ -x "$PY" ] || bdie "no training venv at $PY"
  [ -s "$SD/chat_sft.jsonl" ] && [ -s "$SD/tool_sft.jsonl" ] \
    || bdie "$SD needs chat_sft.jsonl and tool_sft.jsonl: run sd_sync.sh on the rig first (this box has chat $(nrows "$SD/chat_sft.jsonl"), tool $(nrows "$SD/tool_sft.jsonl") rows)"
  "$PY" "$HERE/sd_lengths.py" --dir "$SD" --model-dir "$MODEL_DIR" --train27 "$E27_SRC/train27.py" --out "$ACC/sd_lengths.json" "$@"
}

cmd_stop() {
  local force=0 g
  [ "${1:-}" != --force ] || force=1
  for g in chat tools; do
    if alive "$RUN_DIR/sd_$g.pid"; then
      [ "$force" = 1 ] || bdie "$g is running (pid $(cat "$RUN_DIR/sd_$g.pid")): wait for it, or stop --force (it aborts and resumes later)"
      kill -TERM -- "-$(cat "$RUN_DIR/sd_$g.pid")" 2>/dev/null
    fi
  done
  SGLANG_PID=$(cat "$SD_PIDFILE" 2>/dev/null || true)
  stop_sglang || bdie "sglang would not stop"
  rm -f "$SD_PIDFILE"
  local used; used=$(max_gpu_used)
  [ "${used:-0}" -lt "$SD_MAX_USED_MIB" ] || bdie "a GPU still holds ${used} MiB after stopping the server"
  echo "SD_STOP_OK every GPU freed"
}

case "${1:-}" in
  serve) shift; cmd_serve "$@" ;;
  chat) shift; cmd_chat "$@" ;;
  tools) shift; cmd_tools "$@" ;;
  status) cmd_status ;;
  finalize) shift; cmd_finalize "$@" ;;
  lengths) shift; cmd_lengths "$@" ;;
  stop) shift; cmd_stop "$@" ;;
  -h|--help|"") sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; [ -n "${1:-}" ] || exit 2 ;;
  *) bdie "unknown command '$1' (serve chat tools status finalize lengths stop)" ;;
esac
