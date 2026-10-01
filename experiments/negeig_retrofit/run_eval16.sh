#!/usr/bin/env bash
# 16x validation check on the trained single-task and multitask adapters (one GPU job at a time).
set -uo pipefail
PY=${PY:-$HOME/.venvs/negeig/bin/python}
M=/data/ai-ml/hf-models/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b/
R=/data/ai-ml/models/_runs/negeig-retrofit
cd "$(dirname "$0")"
while read -r d ck; do
  [ -f "$R/$d/eval16.json" ] && { echo "skip $d"; continue; }
  echo "eval16 $d $(date -Is)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 nice -n 10 taskset -c 8-15 \
    "$PY" eval16.py --model "$M" --run_dir "$R/$d" --ckpt "$ck" > "$R/$d/eval16.log" 2>&1 < /dev/null || echo "FAILED $d"
  tail -1 "$R/$d/eval16.log" | cut -c1-400
done <<LIST
diag/swap_wide_lr1e-4_b64 ckpt_step1600.pt
diag/swap_ctrl_lr1e-4_b64 trainable.pt
diag/codeswap_wide_lr1e-4_b64 trainable.pt
diag/codeswap_ctrl_lr1e-4_b64 trainable.pt
long/wide_lr1e-4_w1e-4_n1600_s1 trainable.pt
long/ctrl_lr1e-3_w1e-3_n1600_s1 trainable.pt
LIST
echo "eval16 done $(date -Is)"
