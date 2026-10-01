#!/usr/bin/env bash
# Rented-box setup for the negeig main runs. Runs ON the box, from /workspace/negeig/code.
# Pins the laptop's package versions; checks a real CUDA allocation on every GPU before staging anything.
# Usage: box_setup.sh <cu130|cu128>
set -euo pipefail
CU=${1:-cu130}
W=/workspace/negeig
mkdir -p $W/hf $W/runs
export HF_HOME=$W/hf
# triton builds its launcher with gcc against the system Python headers (slim CUDA base images have neither)
command -v gcc >/dev/null && [ -f /usr/include/python3.12/Python.h ] || \
  { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential python3.12-dev curl ca-certificates >/dev/null; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH=$HOME/.local/bin:$PATH
[ -d $W/venv ] || uv venv --python 3.12 $W/venv
PY=$W/venv/bin/python
uv pip install --python $PY "torch==2.14.0" --index-url https://download.pytorch.org/whl/$CU
uv pip install --python $PY transformers==5.17.0 peft==0.21.0 flash-linear-attention==0.5.2 fla-core==0.5.2 \
  accelerate==1.15.0 datasets==5.0.1 safetensors==0.8.0 tokenizers==0.23.2 huggingface-hub==1.33.0 numpy==2.5.3 einops==0.8.2
# real CUDA allocation on every GPU, before anything else is staged
$PY - <<'EOF'
import torch
n = torch.cuda.device_count()
assert n > 0, "no CUDA devices"
for i in range(n):
    x = torch.ones(1 << 20, device=f"cuda:{i}"); torch.cuda.synchronize(i)
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory // 2**30, "GiB ok")
print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", n)
EOF
$W/venv/bin/hf download Qwen/Qwen3.5-4B-Base --revision 1001bb4d826a52d1f399e183466143f4da7b741b --local-dir $W/model
$PY - <<'EOF'
from datasets import load_dataset
load_dataset("TIGER-Lab/MMLU-Pro", split="test"); load_dataset("openai/openai_humaneval", split="test")
print("datasets cached")
EOF
echo "setup done"
