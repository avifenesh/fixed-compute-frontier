#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/runtime/cpkv-topk-001-venv/bin/python"
TRAINER="$PROJECT_ROOT/experiments/topk_compiled_pushdown_train.py"
MANIFEST="$PROJECT_ROOT/manifests/topk-pushdown-cpkv001.json"
RUN_ROOT="$PROJECT_ROOT/runtime/cpkv-topk-001"
LOG_ROOT="$PROJECT_ROOT/runtime/cpkv-topk-001-logs"
STATUS_PATH="$PROJECT_ROOT/runtime/cpkv-topk-001-matrix.status"
WORKERS="${1:-64}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
exec 9>"$PROJECT_ROOT/runtime/cpkv-topk-001-matrix.lock"
flock -n 9 || {
  echo "another matrix launcher holds the lock" >&2
  exit 1
}

write_status() {
  local state="$1"
  local completed
  completed="$(find "$RUN_ROOT" -type f -name run.json | wc -l)"
  printf 'state=%s\ncompleted=%s\ntotal=180\nutc=%s\n' \
    "$state" "$completed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATUS_PATH.tmp"
  mv "$STATUS_PATH.tmp" "$STATUS_PATH"
}

run_job() {
  local family="$1"
  local arm="$2"
  local seed="$3"
  local output="$RUN_ROOT/$family/$arm/$seed/run.json"
  local log="$LOG_ROOT/${family}-${arm}-${seed}.log"
  if [[ -f "$output" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$output")"
  "$PYTHON" "$TRAINER" \
    --manifest "$MANIFEST" \
    --family "$family" \
    --arm "$arm" \
    --seed "$seed" >"$log.tmp" 2>&1
  mv "$log.tmp" "$log"
}

export PROJECT_ROOT PYTHON TRAINER MANIFEST RUN_ROOT LOG_ROOT
export -f run_job

write_status running
trap 'write_status failed' ERR INT TERM

{
  for arm in S32 LINK32 DIRECT3 HARD3 RNN32 MLP; do
    for family in A2 D1 A1; do
      for seed in $(seq 1701 1710); do
        printf '%s\t%s\t%s\n' "$family" "$arm" "$seed"
      done
    done
  done
} | xargs -P "$WORKERS" -n 3 bash -c 'run_job "$1" "$2" "$3"' _

completed="$(find "$RUN_ROOT" -type f -name run.json | wc -l)"
if [[ "$completed" -ne 180 ]]; then
  echo "matrix ended with $completed/180 run receipts" >&2
  exit 1
fi
trap - ERR INT TERM
write_status complete
