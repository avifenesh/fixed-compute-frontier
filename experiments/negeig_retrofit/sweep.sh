#!/usr/bin/env bash
# Learning-rate sweep, resumable. Each line: arm lr w_lr. 400 steps, validation eval every 50,
# no test-set eval (the test set stays untouched until the main runs).
set -uo pipefail
PY=${PY:-$HOME/.venvs/negeig/bin/python}
M=/data/ai-ml/hf-models/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b/
R=/data/ai-ml/models/_runs/negeig-retrofit
SW=$R/sweep
mkdir -p "$SW"
cd "$(dirname "$0")"
while read -r arm lr wlr; do
  [ -z "$arm" ] && continue
  out="$SW/${arm}_lr${lr}_w${wlr}"
  [ -f "$out/result_sweep.json" ] && { echo "skip $out"; continue; }
  rm -rf "$out"
  echo "start $arm lr=$lr w_lr=$wlr $(date -Is)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 nice -n 10 taskset -c 8-15 \
    "$PY" train.py --model "$M" --arm "$arm" --seed 0 --steps 400 --lr "$lr" --w_lr "$wlr" \
    --eval_every 50 --no_final_eval --out "$out" --replay "$R/replay_wikitext103.pt" > "$out.log" 2>&1 \
    || echo "FAILED $out"
  echo "done $arm lr=$lr w_lr=$wlr $(date -Is)"
done < "${1:-sweep_stage1.txt}"
