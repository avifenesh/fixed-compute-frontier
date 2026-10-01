#!/usr/bin/env bash
# SGLang helpers shared by the box scripts (accept.sh, parity.sh, sd_box.sh). Source it; never run it. Runs ON a box.
#
#   stop_sglang            stops the server this shell started (SGLANG_PID, a process group from setsid) and any other
#                          SGLang server or worker on the box; returns 1 if one would not die. SGLANG_PROC_PATTERN
#                          (tests only) replaces the process match so a test never signals a real SGLang server.
#   wait_sglang_healthy PORT PID LOG [TRIES]
#                          polls /health every SGLANG_POLL_S (10) seconds; 0 when it answers 200, 1 when PID died or TRIES
#                          (default 150, 25 minutes: the first launch on a new architecture JIT-compiles kernels and
#                          captures graphs) ran out; the log tail is printed on failure
SGLANG_PID=${SGLANG_PID:-}

stop_sglang() {
  # Match only the server itself: its launch_server argv and the `sglang::scheduler` style process titles the workers
  # set. A bare [s]glang also matches the runner's own `tee $ACC/sglang.log` (the step log is named after the step),
  # so the old check never saw the server gone and failed the step with "sglang would not stop".
  local pat=${SGLANG_PROC_PATTERN:-'[s]glang\.launch_server|^[s]glang::'}
  [ -n "$SGLANG_PID" ] && kill -TERM -- "-$SGLANG_PID" 2>/dev/null
  for sig in TERM KILL; do
    pkill "-$sig" -f "$pat" 2>/dev/null
    for _ in $(seq 1 30); do pgrep -f "$pat" >/dev/null || { SGLANG_PID=""; return 0; }; sleep 2; done
  done
  return 1
}

wait_sglang_healthy() {
  local port=$1 pid=$2 slog=$3 tries=${4:-150} poll=${SGLANG_POLL_S:-10} i code=""
  for ((i = 1; i <= tries; i++)); do
    sleep "$poll"
    kill -0 "$pid" 2>/dev/null || { echo "sglang exited early"; tail -n 40 "$slog"; return 1; }
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null || true)
    if [ "$code" = 200 ]; then echo "sglang healthy after ~$((i * poll)) s"; return 0; fi
  done
  echo "sglang not healthy after $((tries * poll)) s"; tail -n 40 "$slog"
  return 1
}
