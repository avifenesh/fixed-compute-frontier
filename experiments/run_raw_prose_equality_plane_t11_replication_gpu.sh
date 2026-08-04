#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${T11_PYTHON:-python}"

mkdir -p results
log_file="results/raw-prose-equality-plane-t11-replication-gpu-run.log"
exec > >(tee -a "$log_file") 2>&1

echo "stage=replication_environment"
date --iso-8601=seconds
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"$python_bin" - <<'PY'
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
name = torch.cuda.get_device_name(0)
if "H100" not in name and "H200" not in name:
    raise SystemExit(f"replication requires H100/H200, found {name}")
print(json.dumps({"gpu": name, "torch": torch.__version__, "cuda": torch.version.cuda}))
PY

echo "stage=replication_contract_tests"
CUDA_VISIBLE_DEVICES='' "$python_bin" -m pytest -q \
  tests/test_raw_prose_equality_plane_t11.py \
  tests/test_raw_prose_equality_plane_t11_train.py \
  tests/test_raw_prose_equality_plane_t11_replicate.py

"$python_bin" - <<'PY'
import json
from pathlib import Path

path = Path("results/raw-prose-equality-plane-t11-quick.json")
artifact = json.loads(path.read_text())
assert artifact["all_gates_pass"], artifact["gates"]
print("representation_precondition_passed=true")
PY

echo "stage=three_unseen_seed_replication"
"$python_bin" experiments/raw_prose_equality_plane_t11_replicate.py --device cuda
echo "stage=complete mode=replication"
