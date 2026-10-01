#!/usr/bin/env bash
# Main runs for the negeig retrofit, resumable. Arms alternate within each seed so a partial
# campaign stays balanced. CPU is capped (cores 8-15, nice) to keep the desktop responsive.
set -euo pipefail
PY=${PY:-$HOME/.venvs/negeig/bin/python}
M=$(ls -d /data/ai-ml/hf-models/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b/)
R=/data/ai-ml/models/_runs/negeig-retrofit
MAIN=$R/main
STEPS=1200
mkdir -p "$MAIN"
cd "$(dirname "$0")"
run() {  # arm seed steps
  local out="$MAIN/$1_s$2"
  [ -f "$out/result.json" ] && { echo "skip $out"; return; }
  echo "start $1 s$2 $(date -Is)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 nice -n 10 taskset -c 8-15 \
    "$PY" train.py --model "$M" --arm "$1" --seed "$2" --steps "$3" --out "$out" \
    --replay "$R/replay_wikitext103.pt" > "$out.log" 2>&1
  echo "done $1 s$2 $(date -Is)"
}
mkdir -p "$MAIN"
run ctrl 100 0   # untouched base reference: LoRA B is zero at init and no step is taken
for s in 0 1 2 3 4; do
  run wide "$s" "$STEPS"
  run ctrl "$s" "$STEPS"
done
"$PY" analyze.py --root "$MAIN" --out "$R/analysis.json"
