#!/usr/bin/env bash
# Parallel main runs on a rented multi-GPU box, one run per GPU. Each run trains with a validation eval at
# every checkpoint, then the final test eval (1x/4x/16x, seed 10,000, 128 sequences per task and length,
# held-out WikiText NLL), then full MMLU-Pro and HumanEval. Resumable per run.
# Each list line: gpu name arm lr w_lr steps seed [extra train.py flags...]
set -uo pipefail
W=/workspace/negeig
PY=$W/venv/bin/python
M=$W/model
R=$W/runs
export HF_HOME=$W/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
cd "$(dirname "$0")"
run_one() {
  local gpu=$1 name=$2 arm=$3 lr=$4 wlr=$5 steps=$6 seed=$7; shift 7
  local out=$R/$name
  if [ ! -f "$out/result.json" ]; then
    rm -rf "$out"
    echo "start $name gpu $gpu $(date -Is)"
    # shellcheck disable=SC2068
    CUDA_VISIBLE_DEVICES=$gpu $PY train.py --model $M --arm $arm --seed $seed --steps $steps --lr $lr --w_lr $wlr \
      --eval_every 100 --out $out --replay $W/replay_wikitext103.pt $@ > $out.log 2>&1 < /dev/null \
      || { echo "FAILED train $name $(date -Is)"; return; }
  fi
  if [ ! -f "$out/general.json" ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY general_eval.py --model $M --run_dir $out > $out/general.log 2>&1 < /dev/null \
      || { echo "FAILED general $name $(date -Is)"; return; }
  fi
  echo "done $name $(date -Is)"
}
mapfile -t LINES < "${1:?list file}"
for line in "${LINES[@]}"; do
  case "$line" in ''|\#*) continue;; esac
  # shellcheck disable=SC2086
  run_one $line &
done
wait
echo "all done $(date -Is)"
