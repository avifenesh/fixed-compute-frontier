#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${T18_FLASH2_PYTHON:-python}"
mkdir -p results
exec > >(tee -a results/budget-neutral-soft-record-t18-flash2-gpu-run.log) 2>&1

echo "stage=environment"
date --iso-8601=seconds
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

echo "stage=contracts"
CUDA_VISIBLE_DEVICES='' "$python_bin" -m pytest -q \
  tests/test_budget_neutral_soft_record_t18_flash2.py

echo "stage=flash2_runtime_gate"
"$python_bin" experiments/budget_neutral_soft_record_t18_flash2_runtime.py --device cuda
echo "stage=complete"
