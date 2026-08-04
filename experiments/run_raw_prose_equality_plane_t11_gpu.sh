#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

base_python="${T11_PYTHON:-python}"
run_mode="${1:-pilot}"
if [[ "$run_mode" != "quick" && "$run_mode" != "pilot" ]]; then
  echo "usage: bash experiments/run_raw_prose_equality_plane_t11_gpu.sh [quick|pilot]" >&2
  exit 2
fi

mkdir -p results
log_file="results/raw-prose-equality-plane-t11-gpu-run.log"
exec > >(tee -a "$log_file") 2>&1

echo "stage=environment"
date --iso-8601=seconds
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"$base_python" - <<'PY'
import json
import platform
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
name = torch.cuda.get_device_name(0)
if "H100" not in name and "H200" not in name:
    raise SystemExit(f"T11 is admitted only on H100/H200, found {name}")
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

if ! "$base_python" -c \
  'import datasets, transformers, pytest; assert datasets.__version__ == "4.0.0"; assert transformers.__version__ == "4.57.6"' \
  >/dev/null 2>&1
then
  echo "stage=install_cpu_dependencies"
  "$base_python" -m pip install --disable-pip-version-check \
    --break-system-packages 'datasets==4.0.0' 'transformers==4.57.6' pytest
fi

"$base_python" -c \
  'import torch, datasets, transformers, tokenizers, pytest; assert torch.cuda.is_available(); print("datasets=" + datasets.__version__, "transformers=" + transformers.__version__, "tokenizers=" + tokenizers.__version__)'

echo "stage=prepare_and_verify_data"
"$base_python" experiments/prepare_raw_prose_equality_plane_t10_data.py

echo "stage=cpu_contract_tests"
CUDA_VISIBLE_DEVICES='' "$base_python" -m pytest -q \
  tests/test_prepare_raw_prose_equality_plane_t10_data.py \
  tests/test_consensus_permutation_raw_prose.py \
  tests/test_raw_prose_equality_plane_t10.py \
  tests/test_raw_prose_equality_plane_t10_train.py \
  tests/test_raw_prose_equality_plane_t11.py \
  tests/test_raw_prose_equality_plane_t11_train.py

echo "stage=bf16_representation_gate"
"$base_python" experiments/raw_prose_equality_plane_t11.py --device cuda
"$base_python" - <<'PY'
import json
from pathlib import Path

artifact = json.loads(Path("results/raw-prose-equality-plane-t11-quick.json").read_text())
assert artifact["all_gates_pass"], artifact["gates"]
assert artifact["writer"]["frozen_parameter_entries"] < 7_824_855 / 4
assert artifact["writer"]["protected_hidden_coordinates_after_block_2"] == 9
print("representation_all_gates_pass=true")
PY

if [[ "$run_mode" == "quick" ]]; then
  echo "stage=complete mode=quick"
  exit 0
fi

echo "stage=matched_training_pilot"
"$base_python" experiments/raw_prose_equality_plane_t11_train.py --device cuda
"$base_python" - <<'PY'
import json
from pathlib import Path

artifact = json.loads(Path("results/raw-prose-equality-plane-t11-training-pilot.json").read_text())
print(json.dumps({
    "pilot_all_gates_pass": artifact["all_gates_pass"],
    "failed_gates": sorted(name for name, passed in artifact["gates"].items() if not passed),
    "best_2x_control": artifact["best_2x_control"],
}, indent=2, sort_keys=True))
PY
echo "stage=complete mode=pilot"
