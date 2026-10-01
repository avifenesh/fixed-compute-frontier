#!/usr/bin/env bash
# After the training campaign exits: full MMLU-Pro + HumanEval on every run, then the final analysis.
set -uo pipefail
PY=${PY:-$HOME/.venvs/negeig/bin/python}
M=/data/ai-ml/hf-models/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b/
R=/data/ai-ml/models/_runs/negeig-retrofit
cd "$(dirname "$0")"
while pgrep -f "bash experiments/negeig_retrofit/run_main.sh" >/dev/null; do sleep 120; done
echo "general evals start $(date -Is)"
for d in "$R"/main/ctrl_s100 "$R"/main/wide_s0 "$R"/main/ctrl_s0 "$R"/main/wide_s1 "$R"/main/ctrl_s1 \
         "$R"/main/wide_s2 "$R"/main/ctrl_s2 "$R"/main/wide_s3 "$R"/main/ctrl_s3 "$R"/main/wide_s4 "$R"/main/ctrl_s4; do
  [ -f "$d/result.json" ] || { echo "missing $d"; continue; }
  [ -f "$d/general.json" ] && { echo "skip $d"; continue; }
  echo "general $d $(date -Is)"
  HF_HOME=/data/ai-ml/hf-models PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 nice -n 10 taskset -c 8-15 \
    "$PY" general_eval.py --model "$M" --run_dir "$d" > "$d/general.log" 2>&1 || echo "FAILED $d"
done
"$PY" analyze.py --root "$R/main" --out "$R/analysis.json"
echo "general evals done $(date -Is)"
