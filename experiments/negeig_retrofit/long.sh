#!/usr/bin/env bash
# Stage-2b long runs, resumable. Each line: arm lr w_lr steps seed. Validation eval every 100 steps,
# no test-set eval. Waits for any running sweep to finish first (one GPU job at a time).
set -uo pipefail
PY=${PY:-$HOME/.venvs/negeig/bin/python}
M=/data/ai-ml/hf-models/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b/
R=/data/ai-ml/models/_runs/negeig-retrofit
LG=$R/long
mkdir -p "$LG"
cd "$(dirname "$0")"
while pgrep -f "bash experiments/negeig_retrofit/sweep.sh" >/dev/null; do sleep 60; done
while read -r arm lr wlr steps seed; do
  [ -z "$arm" ] && continue
  out="$LG/${arm}_lr${lr}_w${wlr}_n${steps}_s${seed}"
  [ -f "$out/result_sweep.json" ] && { echo "skip $out"; continue; }
  rm -rf "$out"
  echo "start $arm lr=$lr w_lr=$wlr steps=$steps seed=$seed $(date -Is)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 nice -n 10 taskset -c 8-15 \
    "$PY" train.py --model "$M" --arm "$arm" --seed "$seed" --steps "$steps" --lr "$lr" --w_lr "$wlr" \
    --eval_every 100 --no_final_eval --out "$out" --replay "$R/replay_wikitext103.pt" > "$out.log" 2>&1 \
    || echo "FAILED $out"
  echo "done $arm lr=$lr w_lr=$wlr steps=$steps seed=$seed $(date -Is)"
done < "${1:-long_stage2b.txt}"
