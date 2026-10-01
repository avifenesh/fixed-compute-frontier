#!/usr/bin/env bash
# General evals (full MMLU-Pro, HumanEval pass@1) on the diagnostic adapters and the untouched base, to test the
# card's second kill gate before any main run. Waits for trainer PID $1 (or none), then runs, then resumes the queue ($2).
set -uo pipefail
PY=${PY:-$HOME/.venvs/negeig/bin/python}
M=/data/ai-ml/hf-models/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b/
R=/data/ai-ml/models/_runs/negeig-retrofit
cd "$(dirname "$0")"
while [ "$1" != none ] && kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "general evals start $(date -Is)"
while read -r d ck; do
  [ -f "$d/general.json" ] && { echo "skip $d"; continue; }
  echo "general $d $ck $(date -Is)"
  HF_HOME=/data/ai-ml/hf-models PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 nice -n 10 taskset -c 8-15 \
    "$PY" general_eval.py --model "$M" --run_dir "$d" --ckpt "$ck" > "$d/general.log" 2>&1 < /dev/null || echo "FAILED $d"  # stdin must not eat the list
  tail -1 "$d/general.log" | cut -c1-300
done <<LIST
$R/main_aborted_guessed_lr/ctrl_s100 trainable.pt
$R/diag/swap_wide_lr1e-4_b64 ckpt_step1600.pt
$R/diag/swap_ctrl_lr1e-4_b64 trainable.pt
$R/diag/codeswap_wide_lr1e-4_b64 trainable.pt
$R/diag/codeswap_ctrl_lr1e-4_b64 trainable.pt
LIST
echo "general evals done $(date -Is)"
[ -n "${2:-}" ] && exec ./long3.sh "$2"
