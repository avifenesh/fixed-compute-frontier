#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${T18_PYTHON:-python}"
mkdir -p results
exec > >(tee -a results/budget-neutral-soft-record-t18-stage0-gpu-run.log) 2>&1

echo "stage=environment"
date --iso-8601=seconds
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"$python_bin" - <<'PY'
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
name = torch.cuda.get_device_name(0)
if "H100" not in name and "H200" not in name:
    raise SystemExit(f"T18 requires H100/H200, found {name}")
print(json.dumps({"gpu": name, "torch": torch.__version__, "cuda": torch.version.cuda}))
PY

echo "stage=contracts"
CUDA_VISIBLE_DEVICES='' "$python_bin" -m pytest -q \
  tests/test_hotpot_real_prose_t12_data.py \
  tests/test_budget_neutral_soft_record_t18.py

echo "stage=operator_and_ledger"
"$python_bin" experiments/budget_neutral_soft_record_t18.py --device cuda
echo "stage=complete"
