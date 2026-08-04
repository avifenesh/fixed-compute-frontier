#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

base_python="${T10_PYTHON:-python}"
python_bin="$base_python"
run_mode="${1:-pilot}"
if [[ "$run_mode" != "quick" && "$run_mode" != "pilot" ]]; then
  echo "usage: bash experiments/run_raw_prose_equality_plane_t10_gpu.sh [quick|pilot]" >&2
  exit 2
fi

mkdir -p results
log_file="results/raw-prose-equality-plane-t10-gpu-run.log"
exec > >(tee -a "$log_file") 2>&1

echo "stage=environment"
date --iso-8601=seconds
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"$base_python" - <<'PY'
import json
import platform

try:
    import torch
except ImportError as error:
    raise SystemExit(f"PyTorch is required in the GPU image: {error}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
name = torch.cuda.get_device_name(0)
if "H100" not in name and "H200" not in name:
    raise SystemExit(f"T10 is admitted only on H100/H200 for this run, found {name}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit(f"BF16 is not supported by {name}")
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": name,
    "bf16": True,
}, sort_keys=True))
PY

if ! "$python_bin" -c \
  'import datasets, transformers, pytest; assert datasets.__version__ == "4.0.0"; assert transformers.__version__ == "4.57.6"' \
  >/dev/null 2>&1
then
  echo "stage=install_cpu_dependencies"
  "$base_python" -m pip install --disable-pip-version-check \
    --break-system-packages 'datasets==4.0.0' 'transformers==4.57.6' pytest
fi

"$python_bin" -c \
  'import torch, datasets, transformers, tokenizers, pytest; assert torch.cuda.is_available(); print("datasets=" + datasets.__version__, "transformers=" + transformers.__version__, "tokenizers=" + tokenizers.__version__)'

echo "stage=prepare_and_verify_data"
"$python_bin" experiments/prepare_raw_prose_equality_plane_t10_data.py

echo "stage=cpu_contract_tests"
CUDA_VISIBLE_DEVICES='' "$python_bin" -m pytest -q \
  tests/test_prepare_raw_prose_equality_plane_t10_data.py \
  tests/test_consensus_permutation_raw_prose.py \
  tests/test_raw_prose_equality_plane_t10.py \
  tests/test_raw_prose_equality_plane_t10_train.py

echo "stage=bf16_representation_gate"
"$python_bin" experiments/raw_prose_equality_plane_t10.py --device cuda
"$python_bin" -c \
  'import json, pathlib; p=pathlib.Path("results/raw-prose-equality-plane-t10-quick.json"); a=json.loads(p.read_text()); assert a["all_gates_pass"], a["gates"]; print("representation_all_gates_pass=true")'

if [[ "$run_mode" == "quick" ]]; then
  echo "stage=complete mode=quick"
  exit 0
fi

echo "stage=matched_training_pilot"
"$python_bin" experiments/raw_prose_equality_plane_t10_train.py --device cuda
"$python_bin" - <<'PY'
import json
from pathlib import Path

path = Path("results/raw-prose-equality-plane-t10-training-pilot.json")
artifact = json.loads(path.read_text())
print(json.dumps({
    "pilot_all_gates_pass": artifact["all_gates_pass"],
    "failed_gates": sorted(name for name, passed in artifact["gates"].items() if not passed),
    "best_2x_control": artifact["best_2x_control"],
}, indent=2, sort_keys=True))
PY
echo "stage=complete mode=pilot"
