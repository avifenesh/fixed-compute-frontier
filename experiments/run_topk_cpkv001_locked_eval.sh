#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/runtime/cpkv-topk-001-venv/bin/python"
MATERIALIZER="$PROJECT_ROOT/experiments/topk_compiled_pushdown_evaluator.py"
EVALUATOR="$PROJECT_ROOT/experiments/topk_compiled_pushdown_locked_eval.py"
MANIFEST="$PROJECT_ROOT/manifests/topk-pushdown-cpkv001.json"
RECEIPT="$PROJECT_ROOT/results/topk-pushdown-checkpoints-frozen.json"
SHARD_ROOT="$PROJECT_ROOT/results/topk-pushdown-shards"
FINAL="$PROJECT_ROOT/results/topk-pushdown-locked-evaluation.json"
STATUS_PATH="$PROJECT_ROOT/runtime/cpkv-topk-001-evaluation.status"
WORKERS="${1:-60}"

mkdir -p "$SHARD_ROOT"
exec 9>"$PROJECT_ROOT/runtime/cpkv-topk-001-evaluation.lock"
flock -n 9 || {
  echo "another evaluation launcher holds the lock" >&2
  exit 1
}

write_status() {
  local state="$1"
  local completed
  completed="$(find "$SHARD_ROOT" -maxdepth 1 -type f -name 'shard-*.json' | wc -l)"
  printf 'state=%s\ncompleted=%s\ntotal=60\nutc=%s\n' \
    "$state" "$completed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATUS_PATH.tmp"
  mv "$STATUS_PATH.tmp" "$STATUS_PATH"
}

run_shard() {
  local index="$1"
  local padded
  padded="$(printf '%02d' "$index")"
  local output="$SHARD_ROOT/shard-$padded.json"
  local log="$PROJECT_ROOT/runtime/cpkv-topk-001-shard-$padded.log"
  if [[ -f "$output" ]]; then
    return 0
  fi
  "$PYTHON" "$EVALUATOR" \
    --manifest "$MANIFEST" \
    --checkpoint-freeze-receipt "$RECEIPT" \
    --shard-index "$index" \
    --output "$output" >"$log.tmp" 2>&1
  mv "$log.tmp" "$log"
}

export PROJECT_ROOT PYTHON EVALUATOR MANIFEST RECEIPT SHARD_ROOT
export -f run_shard

write_status materializing
trap 'write_status failed' ERR INT TERM

# Materialize sealed E1/E2 exactly once before parallel readers start.
"$PYTHON" "$MATERIALIZER" \
  --manifest "$MANIFEST" \
  --checkpoint-freeze-receipt "$RECEIPT" \
  >"$PROJECT_ROOT/runtime/cpkv-topk-001-materialization.json"

write_status evaluating
seq 0 59 | xargs -P "$WORKERS" -n 1 bash -c 'run_shard "$1"' _

completed="$(find "$SHARD_ROOT" -maxdepth 1 -type f -name 'shard-*.json' | wc -l)"
if [[ "$completed" -ne 60 ]]; then
  echo "evaluation ended with $completed/60 shards" >&2
  exit 1
fi
if [[ -e "$FINAL" ]]; then
  echo "refusing to overwrite existing final evaluation: $FINAL" >&2
  exit 1
fi
"$PYTHON" "$EVALUATOR" \
  --manifest "$MANIFEST" \
  --checkpoint-freeze-receipt "$RECEIPT" \
  --merge-shards "$SHARD_ROOT" \
  --output "$FINAL" \
  >"$PROJECT_ROOT/runtime/cpkv-topk-001-merge.json"

trap - ERR INT TERM
write_status complete
