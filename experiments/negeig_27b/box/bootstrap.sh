#!/usr/bin/env bash
# bootstrap.sh - runs ON a box (as the SSH_USER that create_box.sh made). Builds both venvs, downloads the model, installs the watchdog.
#
#   bootstrap.sh [--stage base|train|verl|model|sync|evaldata|units|all]     (default all, in this order:
#                                                                             base train verl model units)
#
# Data direction (the decision, written down): RIG -> BOX. The box cannot reach the rig (it sits behind NAT
# and home-network inbound is closed), so the rig pushes code and data with push_data.sh (rsync over ssh,
# bandwidth-capped). `--stage sync` is the optional opposite leg: with RIG_SRC=user@host:/repo set and the rig
# reachable from the box (a reverse tunnel, `ssh -R 2222:localhost:22 box`, then RIG_SRC=user@localhost:/repo
# with RIG_SSH_PORT=2222), the box pulls the same trees instead. Model weights never travel through the rig:
# the box downloads Qwen/Qwen3.8-27B at the pinned revision straight from Hugging Face.
#
# Order that matters (CLAUDE.md "Renting a box"): `base` ends with a REAL CUDA allocation on every GPU, and
# `model` refuses to start until that receipt exists, so no weights or data are staged on a dead box.
#
# Pins are the same as experiments/negeig_retrofit/box_setup.sh (training venv) and
# research/hebrew-rl-20260923/setup_box.sh (verl venv); change them there first.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=boxenv.sh
. "$HERE/boxenv.sh"
CU=${CU:-cu130}
STAGE=all
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE=$2; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) bdie "unknown argument: $1" ;;
  esac
done
mkdir -p "$W" "$RUNS" "$LOGS" "$RUN_DIR" "$ACC" "$HF_HOME" "$DATA" "$CODE"

stage_base() {
  blog "== base: apt packages, uv, training venv with torch, CUDA allocation on every GPU"
  if ! { command -v gcc >/dev/null && [ -f /usr/include/python3.12/Python.h ] && command -v rsync >/dev/null \
         && command -v jq >/dev/null && command -v git >/dev/null && command -v curl >/dev/null; }; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential python3.12-dev python3.12-venv \
      curl ca-certificates git rsync jq numactl >/dev/null
  fi
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  [ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
  uv pip install --python "$PY" "torch==2.14.0" --index-url "https://download.pytorch.org/whl/$CU"
  nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv
  # A real allocation and a real kernel on every GPU. nvidia-smi answering proves nothing (a 5090 answered every
  # NVML query while cudaGetDeviceCount returned 0). Also prints the arch list so a missing sm_xx is visible early.
  EXPECT_GPUS=$EXPECT_GPUS "$PY" - <<'PYEOF'
import os, sys, torch
n = torch.cuda.device_count()
want = int(os.environ["EXPECT_GPUS"])
assert n > 0, "no CUDA devices"
assert n == want, f"{n} CUDA devices, expected {want}"
caps = set()
for i in range(n):
    x = torch.ones(1 << 28, device=f"cuda:{i}", dtype=torch.bfloat16)          # 512 MiB allocation
    y = (x.view(1 << 14, 1 << 14)[:4096, :4096] @ x.view(1 << 14, 1 << 14)[:4096, :4096]).sum()  # a real bf16 GEMM
    torch.cuda.synchronize(i)
    assert torch.isfinite(y.float()).item(), f"GEMM on cuda:{i} is not finite"
    p = torch.cuda.get_device_properties(i)
    caps.add((p.major, p.minor))
    print(i, p.name, f"sm_{p.major}{p.minor}", p.total_memory // 2**30, "GiB alloc+gemm ok")
print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", n, "arch_list", torch.cuda.get_arch_list())
assert len(caps) == 1, f"mixed compute capabilities {caps}"
major, minor = next(iter(caps))
have = torch.cuda.get_arch_list()
if not any(a in have for a in (f"sm_{major}{minor}", f"sm_{major}0", f"compute_{major}{minor}", f"compute_{major}0")):
    print(f"WARNING: sm_{major}{minor} is not in this torch build's arch list; accept.sh will show whether kernels run", file=sys.stderr)
PYEOF
  date -u +%FT%TZ >"$W/.cuda_ok"
  blog "base ok (receipt $W/.cuda_ok)"
}

stage_train() {
  blog "== train: pinned packages in the training venv"
  [ -x "$PY" ] || bdie "no training venv; run --stage base first"
  uv pip install --python "$PY" transformers==5.17.0 peft==0.21.0 flash-linear-attention==0.5.2 fla-core==0.5.2 \
    accelerate==1.15.0 datasets==5.0.1 safetensors==0.8.0 tokenizers==0.23.2 huggingface-hub==1.33.0 \
    numpy==2.5.3 einops==0.8.2
  "$PY" - <<'PYEOF'
import importlib.metadata as md
want = {"torch": "2.14.0", "transformers": "5.17.0", "peft": "0.21.0", "flash-linear-attention": "0.5.2",
        "accelerate": "1.15.0", "datasets": "5.0.1"}
for k, v in want.items():
    got = md.version(k).split("+")[0]
    assert got == v, f"{k} {got} != {v}"
    print(k, got)
import fla  # noqa: F401
PYEOF
  blog "train ok"
}

stage_verl() {
  blog "== verl $VERL_REF with SGLang 0.5.20 (uv sync --frozen)"
  command -v uv >/dev/null || bdie "uv missing; run --stage base first"
  local ch; ch=$(find_cuda_home) || bdie "no CUDA toolkit with nvcc under /usr/local (flashinfer JIT needs it)"
  export CUDA_HOME=$ch PATH="$ch/bin:$PATH"
  if [ ! -d "$VERL/.git" ]; then
    git init -q "$VERL"
    git -C "$VERL" remote add origin https://github.com/volcengine/verl.git
  fi
  git -C "$VERL" fetch -q --depth 1 origin "$VERL_REF"
  git -C "$VERL" checkout -q --detach "$VERL_REF"
  [ "$(git -C "$VERL" rev-parse HEAD)" = "$VERL_REF" ] || bdie "verl HEAD is not the pin"
  (cd "$VERL" && uv sync --frozen --all-packages --extra sglang --extra fsdp)
  PATH="${VPY%/*}:$PATH" "$VPY" - <<'PYEOF'
import importlib.metadata as md
import torch
for name in ("torch", "sglang", "transformers", "peft", "flash-attn", "verl", "ray"):
    try:
        print(f"{name} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name} MISSING")
assert md.version("sglang") == "0.5.20", md.version("sglang")
n = torch.cuda.device_count()
assert n > 0, "no CUDA device"
for i in range(n):
    torch.ones(1, device=f"cuda:{i}").sum().item()
major, minor = torch.cuda.get_device_capability(0)
print(f"cuda {torch.version.cuda} devices {n} {torch.cuda.get_device_name(0)} sm_{major}{minor}")
PYEOF
  blog "verl ok (flash-attn on this arch is proved by accept.sh step verl_env)"
}

stage_model() {
  blog "== model: $MODEL_ID @ $MODEL_REV -> $MODEL_DIR"
  [ -f "$W/.cuda_ok" ] || bdie "no CUDA receipt: run --stage base first (nothing is staged on a box that has not allocated on every GPU)"
  [ -x "$VENV/bin/hf" ] || bdie "no hf CLI in $VENV; run --stage train first"
  local free_gib; free_gib=$(df -BG --output=avail "$W" | tail -n1 | tr -dc '0-9')
  [ "$free_gib" -ge 200 ] || bdie "only ${free_gib} GiB free under $W; the model, checkpoints and merged copies need more"
  # An HF token, if the repo ever needs one, is read from HF_TOKEN in the environment by the hf CLI; it is never
  # put on a command line (visible in ps) and never pushed by these scripts.
  "$VENV/bin/hf" download "$MODEL_ID" --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
  # Verify against the repo's own index: every shard exists and the byte total matches metadata.total_size.
  MODEL_DIR=$MODEL_DIR "$PY" - <<'PYEOF'
import json, os, pathlib
d = pathlib.Path(os.environ["MODEL_DIR"])
idx = json.loads((d / "model.safetensors.index.json").read_text())
shards = sorted(set(idx["weight_map"].values()))
missing = [s for s in shards if not (d / s).is_file()]
assert not missing, f"missing shards {missing}"
got = sum((d / s).stat().st_size for s in shards)
want = idx["metadata"]["total_size"]
# total_size counts tensor bytes; files add a small safetensors header per shard
assert want <= got <= want * 1.001 + 1_000_000, f"shard bytes {got} vs index total_size {want}"
for f in ("config.json", "tokenizer_config.json"):
    assert (d / f).is_file(), f"missing {f}"
cfg = json.loads((d / "config.json").read_text())
print("shards", len(shards), "bytes", got, "GB", round(got / 1e9, 1), "model_type", cfg.get("model_type"))
PYEOF
  echo "$MODEL_ID $MODEL_REV $(date -u +%FT%TZ)" >"$MODEL_DIR/.negeig_revision"
  blog "model ok"
}

stage_sync() {
  blog "== sync (pull leg; the default direction is push_data.sh from the rig)"
  [ -n "${RIG_SRC:-}" ] || { blog "RIG_SRC not set: nothing to pull. Run push_data.sh on the rig (rig -> box)."; return 0; }
  local ssh_cmd="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new ${RIG_SSH_PORT:+-p $RIG_SSH_PORT} ${RIG_SSH_KEY:+-i $RIG_SSH_KEY}"
  ionice -c3 nice -n 10 rsync -az --partial --bwlimit="${BWLIMIT_KBPS:-40000}" -e "$ssh_cmd" \
    "$RIG_SRC/experiments/negeig_retrofit/" "$RETRO/"
  ionice -c3 nice -n 10 rsync -az --partial --bwlimit="${BWLIMIT_KBPS:-40000}" -e "$ssh_cmd" \
    --exclude out/ --exclude __pycache__/ "$RIG_SRC/experiments/negeig_27b/" "$E27/"
  ionice -c3 nice -n 10 rsync -az --partial --bwlimit="${BWLIMIT_KBPS:-40000}" -e "$ssh_cmd" \
    "$RIG_SRC/experiments/negeig_27b/out/" "$DATA/"
  blog "sync ok"
}

stage_evaldata() {
  blog "== evaldata (optional): eval datasets the regression gates use"
  HF_HOME=$HF_HOME "$PY" - <<'PYEOF'
from datasets import load_dataset
load_dataset("TIGER-Lab/MMLU-Pro", split="test")
load_dataset("openai/openai_humaneval", split="test")
load_dataset("google/IFEval", split="train")
print("eval datasets cached")
PYEOF
}

stage_units() {
  blog "== units: install the resume watchdog (not armed until ctl.sh arm)"
  chmod +x "$BOXDIR"/*.sh 2>/dev/null || true
  [ -x "$BOXDIR/resume_wrapper.sh" ] || bdie "$BOXDIR/resume_wrapper.sh missing; run push_data.sh first"
  sed -e "s|@W@|$W|g" -e "s|@USER@|$(id -un)|g" -e "s|@BOXDIR@|$BOXDIR|g" "$BOXDIR/negeig27-train.service" \
    | sudo tee "/etc/systemd/system/$TRAIN_UNIT" >/dev/null
  sudo systemctl daemon-reload
  # Not enabled here: ctl.sh arm enables it together with the train command, so a reboot without a
  # command never starts a stray job.
  blog "units ok ($TRAIN_UNIT installed; arm with: $BOXDIR/ctl.sh arm --cmd-file FILE --ckpt-dir DIR)"
}

case "$STAGE" in
  base) stage_base ;;
  train) stage_train ;;
  verl) stage_verl ;;
  model) stage_model ;;
  sync) stage_sync ;;
  evaldata) stage_evaldata ;;
  units) stage_units ;;
  all) stage_base; stage_train; stage_verl; stage_model; stage_units ;;
  *) bdie "unknown stage $STAGE" ;;
esac
blog "BOOTSTRAP_OK stage=$STAGE"
