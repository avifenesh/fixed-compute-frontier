#!/usr/bin/env bash
# Diagnostic runs with free-form trainer flags, resumable. Each line: name arm lr w_lr steps seed [extra flags...].
# Validation eval every --eval_every (default 200 here), no test-set eval. One GPU job at a time: waits for
# any running train.py first.
set -uo pipefail
PY=${PY:-$HOME/.venvs/negeig/bin/python}
M=/data/ai-ml/hf-models/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b/
R=/data/ai-ml/models/_runs/negeig-retrofit
LG=$R/diag
mkdir -p "$LG"
cd "$(dirname "$0")"
mapfile -t LINES < "${1:?list file}"
for line in "${LINES[@]}"; do
  read -r name arm lr wlr steps seed extra <<< "$line"
  [ -z "${name:-}" ] && continue
  case "$name" in \#*) continue;; esac
  out="$LG/$name"
  [ -f "$out/result_sweep.json" ] && { echo "skip $out"; continue; }
  while pgrep -f "python train.py" >/dev/null; do sleep 30; done
  rm -rf "$out"
  echo "start $name $arm lr=$lr w_lr=$wlr steps=$steps seed=$seed $extra $(date -Is)"
  # shellcheck disable=SC2086
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 nice -n 10 taskset -c 8-15 \
    "$PY" train.py --model "$M" --arm "$arm" --seed "$seed" --steps "$steps" --lr "$lr" --w_lr "$wlr" \
    --eval_every 200 --no_final_eval --out "$out" --replay "$R/replay_wikitext103.pt" $extra > "$out.log" 2>&1 \
    || echo "FAILED $out"
  echo "done $name $(date -Is)"
done
