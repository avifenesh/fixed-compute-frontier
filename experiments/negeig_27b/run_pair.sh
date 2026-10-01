#!/usr/bin/env bash
# Run the Qwen3.8-27B arms on one box. Default: a wide/ctrl pair, wide on GPUs 0-3 and ctrl on GPUs 4-7, same seed, same
# data, same flags. One arm on all 8 GPUs (the 4-box layout) is the same script with ARMS=wide or ARMS=ctrl.
# Idempotent and preemption-safe: an arm with complete.json is skipped, any other arm resumes from its newest checkpoint
# (train27.py --resume). Re-running this script after a preemption and reboot continues every arm.
#
#   run_pair.sh TAG SEED [train27.py flags...]
#
# Paths come from box/boxenv.sh (the box layout); an env var of the same name set by the caller wins:
#   MODEL (default $MODEL_DIR), DATA (default $W/data), RUNS (default $W/runs), PYBIN (venv bin dir, default $VENV/bin),
#   SD (default $DATA/sd), LM_REPLAY (default $DATA/s4/replay_clean_q38.pt), NVLS_FILE (default $W/nccl_nvls.env).
#   PY is the python BINARY in boxenv.sh and is not read here; PY set to a directory is refused (old meaning).
# Launch shape:
#   ARMS           arms to run, space separated (default "wide ctrl"); arm i runs on GPUs i*G .. i*G+G-1, port 29501+i
#   GPUS_PER_RUN   ranks per arm (default EXPECT_GPUS divided by the number of arms: 4 for two arms, 8 for one)
# The mix (the same for the LR sweep, the smoke and Stage A, so the sweep measures the run that continues):
#   STAGE          A (default) trains on S1 + self-distilled chat and tool replay + LM replay. Any other value uses the
#                  replay files that exist (probes, CPU tests).
#   CHAT_REPLAY, TOOL_REPLAY   default $SD/chat_sft.jsonl and $SD/tool_sft.jsonl (what sd_filters.py finalize writes)
#   ALLOW_NO_REPLAY=1          lets STAGE=A run with a replay file missing: a run without the mix, never a Stage A run
#   STAGE=A also needs --lr (the swept value; ALLOW_DEFAULT_LR=1 takes the trainer default) and sets --w_lr to the same
#   value unless given (the gate group uses the LoRA rate in the recipe). The other flags are the trainer defaults,
#   which ARE the Stage A recipe (rank 16, constant after 30 warmup steps, 64/8/8/8 sessions a step, loss weights
#   1.0/0.5/0.5/0.15, eval and save every 50, keep 4 checkpoints); pass a flag only to change it.
# NCCL: STAGE=A reads NCCL_NVLS_ENABLE from $NVLS_FILE, the value accept.sh measured on this box, and refuses to start
#   without it unless ALLOW_NO_NVLS=1. The file wins over an NCCL_NVLS_ENABLE already in the environment.
# Retry, per arm (the trainer resumes from its newest checkpoint, so a retry loses at most --save_every steps):
#   ARM_MAX_ATTEMPTS (8) launches per arm per invocation; ARM_MAX_NOPROGRESS (2) consecutive failed launches with no new
#   checkpoint step; ARM_BACKOFF_S (30) between launches. A permanently failed arm writes $RUNS/TAG/ARM_sSEED/arm_failed
#   and returns non-zero while the other arm keeps running. A SIGTERM (resume_wrapper.sh stop, ctl.sh hold) stops every
#   arm and is never retried. resume_wrapper.sh counts run_pair.sh exits by the checkpoint signature, so one failed arm
#   costs it one crash per invocation: at most ARM_MAX_NOPROGRESS x MAX_CRASHES launches at one checkpoint before `failed`.
# Examples:
#   run_pair.sh stageA-lr1e-3 0 --lr 1e-3 --steps 700                      (the pair, 4 GPUs a run)
#   ARMS=wide run_pair.sh stageA-lr1e-3 0 --lr 1e-3 --steps 700            (one arm, 8 GPUs, 4-box layout)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
die() { echo "run_pair: $*" >&2; exit 2; }
[ $# -ge 2 ] || die "usage: run_pair.sh TAG SEED [train27.py flags...]"
TAG=$1; SEED=$2; shift 2
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || die "TAG '$TAG' must be letters, digits, . _ -"
[[ $SEED =~ ^[0-9]+$ ]] || die "SEED '$SEED' must be an integer"

# boxenv.sh assigns these unconditionally, so keep what the caller set before it runs.
_MODEL=${MODEL:-}; _DATA=${DATA:-}; _RUNS=${RUNS:-}; _SD=${SD:-}; _PYBIN=${PYBIN:-}; _NVLS=${NVLS_FILE:-}; _PY=${PY:-}
# shellcheck source=box/boxenv.sh
. "$HERE/box/boxenv.sh"
[ -n "$_PY" ] && [ -d "$_PY" ] && die "PY=$_PY is a directory: PY is the python binary now, the venv bin directory is PYBIN"
MODEL=${_MODEL:-$MODEL_DIR}; DATA=${_DATA:-$DATA}; RUNS=${_RUNS:-$RUNS}; PYBIN=${_PYBIN:-$PYBIN}
SD=${_SD:-$DATA/sd}; NVLS_FILE=${_NVLS:-$NVLS_FILE}
LM_REPLAY=${LM_REPLAY:-$DATA/s4/replay_clean_q38.pt}
STAGE=${STAGE:-A}
CHAT_REPLAY=${CHAT_REPLAY:-$SD/chat_sft.jsonl}
TOOL_REPLAY=${TOOL_REPLAY:-$SD/tool_sft.jsonl}
ARMS=${ARMS:-"wide ctrl"}
ARM_MAX_ATTEMPTS=${ARM_MAX_ATTEMPTS:-8}
ARM_MAX_NOPROGRESS=${ARM_MAX_NOPROGRESS:-2}
ARM_BACKOFF_S=${ARM_BACKOFF_S:-30}
PORT_BASE=${PORT_BASE:-29501}

read -r -a ARM_LIST <<<"$ARMS"
[ "${#ARM_LIST[@]}" -ge 1 ] || die "ARMS is empty"
for a in "${ARM_LIST[@]}"; do [[ $a =~ ^[a-z0-9]+$ ]] || die "bad arm name '$a' in ARMS"; done
G=${GPUS_PER_RUN:-$((EXPECT_GPUS / ${#ARM_LIST[@]}))}
[[ $G =~ ^[0-9]+$ ]] && [ "$G" -ge 1 ] || die "GPUS_PER_RUN='$G' is not a positive integer"
[ $((G * ${#ARM_LIST[@]})) -le "$EXPECT_GPUS" ] || die "${#ARM_LIST[@]} arm(s) x $G GPUs is more than EXPECT_GPUS=$EXPECT_GPUS"

# ---- flags the caller passed: --lr and --w_lr decide the gate rate; both --flag v and --flag=v are read
EXTRA=("$@")
flag_value() { # name -> FLAG_FOUND (0|1) and FLAG_VAL, read from EXTRA; both --name v and --name=v
  local name=$1 i; FLAG_FOUND=0; FLAG_VAL=""
  for ((i = 0; i < ${#EXTRA[@]}; i++)); do
    case "${EXTRA[i]}" in
      "--$name") FLAG_FOUND=1; FLAG_VAL=${EXTRA[i + 1]:-}; return 0 ;;
      "--$name="*) FLAG_FOUND=1; FLAG_VAL=${EXTRA[i]#--"$name"=}; return 0 ;;
    esac
  done
}
flag_value lr; LR_FOUND=$FLAG_FOUND; LR=$FLAG_VAL
flag_value w_lr; WLR_FOUND=$FLAG_FOUND

# ---- inputs, checked before any GPU is touched
[ -x "$PYBIN/torchrun" ] || die "no torchrun at $PYBIN/torchrun (PYBIN is the training venv bin directory)"
[ -d "$MODEL" ] || die "no model directory $MODEL"
[ -d "$DATA/s1" ] || die "no S1 data at $DATA/s1 (push_data.sh)"
[ -f "$LM_REPLAY" ] || die "no LM replay at $LM_REPLAY (push_data.sh)"
COMMON=(--model "$MODEL" --seed "$SEED" --s1 "$DATA/s1" --lm_replay "$LM_REPLAY" --resume)
missing=()
for kind in chat tool; do
  var=${kind^^}_REPLAY; path=${!var}
  if [ -s "$path" ]; then COMMON+=("--${kind}_replay" "$path"); else missing+=("$kind:$path"); fi
done
REPLAY_NOTE="chat=${CHAT_REPLAY} tool=${TOOL_REPLAY}"
if [ "${#missing[@]}" -gt 0 ]; then
  if [ "$STAGE" = A ] && [ "${ALLOW_NO_REPLAY:-0}" != 1 ]; then
    die "STAGE=A needs the self-distilled replay, missing or empty: ${missing[*]}. Run selfdistill (RUNBOOK.md), or ALLOW_NO_REPLAY=1 for a run that is not the Stage A mix"
  fi
  echo "run_pair: WARNING no replay for ${missing[*]}: this run does not train on the Stage A mix" >&2
  REPLAY_NOTE="MISSING ${missing[*]}"
fi
if [ "$STAGE" = A ]; then
  if [ "$LR_FOUND" = 0 ] && [ "${ALLOW_DEFAULT_LR:-0}" != 1 ]; then
    die "STAGE=A needs an explicit --lr (the swept value; ALLOW_DEFAULT_LR=1 takes the trainer default)"
  fi
  if [ "$LR_FOUND" = 1 ] && [ "$WLR_FOUND" = 0 ]; then EXTRA+=(--w_lr "$LR"); fi
fi

# ---- NCCL_NVLS_ENABLE: one file, written by accept.sh
NVLS_NOTE="not set"
if [ -f "$NVLS_FILE" ]; then
  nvls=$(sed -n 's/^NCCL_NVLS_ENABLE=\([01]\)$/\1/p' "$NVLS_FILE" | tail -n1)
  [ -n "$nvls" ] || die "$NVLS_FILE has no NCCL_NVLS_ENABLE=0|1 line"
  if [ -n "${NCCL_NVLS_ENABLE:-}" ] && [ "$NCCL_NVLS_ENABLE" != "$nvls" ]; then
    echo "run_pair: WARNING NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE in the environment, $NVLS_FILE says $nvls; using the file" >&2
  fi
  export NCCL_NVLS_ENABLE=$nvls; NVLS_NOTE=$nvls
elif [ "$STAGE" = A ] && [ "${ALLOW_NO_NVLS:-0}" != 1 ]; then
  die "no $NVLS_FILE: accept.sh writes it (the measured NCCL_NVLS_ENABLE for this box). ALLOW_NO_NVLS=1 skips it"
fi

newest_step() { # run dir -> step of its newest ckpt_NNNNNN.pt, -1 when none
  local f n; f=$(find "$1" -maxdepth 1 -name 'ckpt_[0-9]*.pt' -printf '%f\n' 2>/dev/null | sort | tail -n1)
  [ -n "$f" ] || { echo -1; return 0; }
  n=${f#ckpt_}; echo $((10#${n%.pt}))
}

run_arm() { # arm index   (runs in its own subshell)
  local arm=$1 idx=$2
  local out="$RUNS/$TAG/${arm}_s${SEED}"
  local port=$((PORT_BASE + idx)) g0=$((idx * G)) devs pid="" nap="" stop=0 attempt=0 noprog=0 rc before after
  devs=$(seq -s, "$g0" $((g0 + G - 1)))
  mkdir -p "$out"
  say() { echo "$(date -u +%FT%TZ) $arm s$SEED: $*" | tee -a "$out/launch.log"; }
  if [ -f "$out/complete.json" ]; then say "complete, skipping"; return 0; fi
  rm -f "$out/arm_failed"
  trap 'stop=1; [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null; [ -n "$nap" ] && kill "$nap" 2>/dev/null' TERM INT
  while :; do
    attempt=$((attempt + 1)); before=$(newest_step "$out")
    say "launch attempt $attempt on GPUs $devs port $port stage=$STAGE ranks=$G nvls=$NVLS_NOTE replay[$REPLAY_NOTE] ckpt_step=$before"
    CUDA_VISIBLE_DEVICES=$devs "$PYBIN/torchrun" --nproc_per_node "$G" --master_port "$port" \
      "$HERE/train27.py" --arm "$arm" --out "$out" "${COMMON[@]}" "${EXTRA[@]}" >>"$out/stdout.log" 2>&1 &
    pid=$!
    while :; do wait "$pid"; rc=$?; kill -0 "$pid" 2>/dev/null || break; done   # a trapped signal ends wait early
    pid=""
    if [ "$stop" = 1 ]; then say "stopped on request (rc $rc), not retried"; return 143; fi
    if [ "$rc" -eq 0 ]; then say "finished"; return 0; fi
    after=$(newest_step "$out")
    if [ "$after" -gt "$before" ]; then noprog=0; else noprog=$((noprog + 1)); fi
    say "attempt $attempt exited $rc, checkpoint step $before -> $after, no-progress launches $noprog of $ARM_MAX_NOPROGRESS"
    if [ "$noprog" -ge "$ARM_MAX_NOPROGRESS" ] || [ "$attempt" -ge "$ARM_MAX_ATTEMPTS" ]; then
      echo "rc $rc after $attempt attempts, newest checkpoint step $after, $(date -u +%FT%TZ)" >"$out/arm_failed"
      say "ARM_FAILED, giving up for this invocation (see $out/stdout.log)"; return "$rc"
    fi
    sleep "$ARM_BACKOFF_S" & nap=$!; wait "$nap" 2>/dev/null; nap=""
    if [ "$stop" = 1 ]; then say "stopped during backoff"; return 143; fi
  done
}

PIDS=()
# shellcheck disable=SC2329  # called through the trap below
stop_all() { local p; for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done; }
trap stop_all TERM INT
for i in "${!ARM_LIST[@]}"; do
  run_arm "${ARM_LIST[i]}" "$i" &
  PIDS+=($!)
done
rc=0
for p in "${PIDS[@]}"; do
  while :; do wait "$p"; r=$?; kill -0 "$p" 2>/dev/null || break; done
  [ "$r" -eq 0 ] || rc=$r
done
exit "$rc"
