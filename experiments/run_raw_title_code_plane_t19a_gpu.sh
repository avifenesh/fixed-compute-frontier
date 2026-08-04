#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m pytest tests/test_raw_title_code_plane_t19a.py -q
python experiments/raw_title_code_plane_t19a.py --device cuda \
  --output results/raw-title-code-plane-t19a.json
