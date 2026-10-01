#!/usr/bin/env bash
# Box-side shared config (sourced by bootstrap.sh, accept.sh, ctl.sh, resume_wrapper.sh). Runs ON the box.
# Every path derives from NEGEIG_W so the watchdog can be exercised on a CPU-only machine against a scratch dir.
ROOT=${ROOT:-/workspace}
W=${NEGEIG_W:-$ROOT/negeig}
CODE=$W/code                       # repo-relative mirror: $CODE/experiments/negeig_retrofit, $CODE/experiments/negeig_27b
RETRO=$CODE/experiments/negeig_retrofit
E27=$CODE/experiments/negeig_27b
BOXDIR=$E27/box
VENV=$W/venv                       # training venv: torch 2.14 cu130, transformers 5.17.0, peft, fla 0.5.2
PYBIN=$VENV/bin                    # the venv bin DIRECTORY (torchrun, python); run_pair.sh and accept.sh call "$PYBIN/torchrun"
PY=$PYBIN/python                   # the python BINARY: box scripts call "$PY", never "$PY/torchrun"
MODEL_ID=Qwen/Qwen3.8-27B
MODEL_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
MODEL_DIR=$W/model
DATA=$W/data                       # rsync target of push_data.sh (s1/, s4/, ...)
SD=$DATA/sd                        # self-distillation replay: chat_sft.jsonl, tool_sft.jsonl (finalize writes them on one
                                   # box, sd_sync.sh copies them to the other). run_pair.sh reads CHAT_REPLAY/TOOL_REPLAY here.
LANE_DIR=$ROOT/lane/research/hebrew-rl-20260923   # push_data.sh --hebrew-lane target; selfdistill reads its pools from here
RUNS=$W/runs                       # training outputs and checkpoints
LOGS=$W/logs
RUN_DIR=$W/run                     # watchdog control files (train.cmd, run.env, hold, done, failed, status.json)
ACC=$W/accept                      # acceptance receipts
NVLS_FILE=$W/nccl_nvls.env         # NCCL_NVLS_ENABLE=0|1, written by accept.sh (measured A/B), read by run_pair.sh and accept.sh
# verl + SGLang 0.5.20 venv, same pin and recipe as research/hebrew-rl-20260923/setup_box.sh
VERL_REF=${VERL_REF:-12ebe0cb4d300c58449fb6c675379e8700015c51}
VERL=$ROOT/verl
VPY=$VERL/.venv/bin/python3
export HF_HOME=$W/hf
export PATH=$HOME/.local/bin:$PATH
EXPECT_GPUS=${EXPECT_GPUS:-8}
TRAIN_UNIT=negeig27-train.service

blog() { echo "[$(date -u +%FT%TZ)] $*"; }
bdie() { echo "ERROR: $*" >&2; exit 1; }

# CUDA toolkit dir for JIT builds (flashinfer, triton helpers). The ubuntu24.04-cuda13.0 image ships 13.0.
# NEGEIG_CUDA_HOME (tests only) is tried first.
find_cuda_home() {
  local c
  for c in ${NEGEIG_CUDA_HOME:-} /usr/local/cuda-13.2 /usr/local/cuda-13.0 /usr/local/cuda; do
    [ -x "$c/bin/nvcc" ] && { echo "$c"; return 0; }
  done
  return 1
}
