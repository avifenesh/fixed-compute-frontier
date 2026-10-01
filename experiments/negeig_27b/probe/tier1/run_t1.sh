#!/usr/bin/env bash
# Tier-1 mechanism probe on one 8-GPU box (BOX side): wide on GPUs 0-3 and ctrl on GPUs 4-7, same seed, data and flags
# (train_t1.py), then the pre-registered decision (decide.py) at each rung, extending to the next rung only when the rule
# says EXTEND. Idempotent and preemption-safe: an arm whose complete.json has reached the rung is skipped, any other arm
# resumes from its newest checkpoint (--resume), a rung already decided is not re-run, and PROBE_DONE ends it.
#
#   run_t1.sh TAG SEED [train_t1.py flags...]          results in $RUNS/TAG/{wide,ctrl}_sSEED/, decisions in $RUNS/TAG/
#
# Under the watchdog (survives a preemption reboot; the wrapper restarts this script, which picks up where it was):
#   ctl.sh arm --cmd 'exec bash "$E27/probe/tier1/run_t1.sh" t1 0' --ckpt-dir "$RUNS/t1"
#
# Env (an env var set by the caller wins; paths default from box/boxenv.sh):
#   REPLAY        wikitext (default): $DATA/t1/replay_wikitext103.pt, the 4B main recipe's replay (box_main_4b.txt), so the
#                 4B numbers (tier-1 4x wide 0.784, ctrl 0.300) are the direct anchor. The file is not in the Stage A
#                 data set; push it from the rig first (the Qwen3.5 and Qwen3.8 vocabularies are identical, so the 4B
#                 lane's token file is valid as is):
#                   mkdir -p /data/ai-ml/models/_runs/negeig-27b/probe-t1-data/t1
#                   ln /data/ai-ml/models/_runs/negeig-retrofit/replay_wikitext103.pt /data/ai-ml/models/_runs/negeig-27b/probe-t1-data/t1/
#                   bash experiments/negeig_27b/box/push_data.sh <box> --data /data/ai-ml/models/_runs/negeig-27b/probe-t1-data
#                 clean: $DATA/s4/replay_clean_q38.pt (the Stage A corpus; at 4B it delayed the switch). Or a file path.
#   TEXT_EVAL     test chunks for the KL / replay-NLL readout (default $DATA/s4/replay_clean_q38.pt when it exists)
#   RUNGS         "1200 2400" (the rule in decide.py is written for these two); AUTO_EXTEND=1 runs the next rung on EXTEND
#   GRAD_CKPT     auto (default): one mem probe with --grad_ckpt 0 on GPU 0 before the first launch; 0 when its peak is at
#                 most MEM_LIMIT_GIB (115), else 1. The choice is saved in $RUNS/TAG/grad_ckpt.env and kept for resumes.
#                 0 or 1 forces it.
#   GPUS_PER_RUN  ranks per arm (default EXPECT_GPUS / 2); arm i uses GPUs i*G .. i*G+G-1 and port PORT_BASE+i
#   ARM_MAX_ATTEMPTS (8), ARM_MAX_NOPROGRESS (2), ARM_BACKOFF_S (30): the per-arm retry rule of run_pair.sh. A failed arm
#                 writes arm_failed and this script exits non-zero, so resume_wrapper.sh counts a crash and retries later.
# NCCL: NCCL_NVLS_ENABLE comes from $NVLS_FILE (accept.sh) when it exists.
# Box sharing: this script takes all 2 x GPUS_PER_RUN GPUs. Do not run it beside probe/run_all.sh, evals/serve_arm.sh or a
# run_pair.sh on the same box (each of those takes all 8 GPUs too); nothing here checks for them. Under the watchdog there is
# one train.cmd slot per box, which serializes jobs armed through ctl.sh, but not a job started by hand.
# Results to the rig during the run (the box is preemptible; checkpoints stay on the box):
#   . experiments/negeig_27b/box/rig.sh; bxpull <box> runs/t1 /data/ai-ml/models/_runs/negeig-27b/box/<box>/runs/t1 \
#     --exclude 'ckpt_*.pt' --exclude 'trainable_*.pt' --exclude 'mem_probe/'
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
die() { echo "run_t1: $*" >&2; exit 2; }
[ $# -ge 2 ] || die "usage: run_t1.sh TAG SEED [train_t1.py flags...]"
TAG=$1; SEED=$2; shift 2
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || die "TAG '$TAG' must be letters, digits, . _ -"
[[ $SEED =~ ^[0-9]+$ ]] || die "SEED '$SEED' must be an integer"

_MODEL=${MODEL:-}; _DATA=${DATA:-}; _RUNS=${RUNS:-}; _PYBIN=${PYBIN:-}; _NVLS=${NVLS_FILE:-}
# shellcheck source=../../box/boxenv.sh
. "$HERE/../../box/boxenv.sh"
MODEL=${_MODEL:-$MODEL_DIR}; DATA=${_DATA:-$DATA}; RUNS=${_RUNS:-$RUNS}; PYBIN=${_PYBIN:-$PYBIN}; NVLS_FILE=${_NVLS:-$NVLS_FILE}
REPLAY=${REPLAY:-wikitext}
case "$REPLAY" in
  wikitext) LM_REPLAY=$DATA/t1/replay_wikitext103.pt ;;
  clean) LM_REPLAY=$DATA/s4/replay_clean_q38.pt ;;
  *) LM_REPLAY=$REPLAY ;;
esac
TEXT_EVAL=${TEXT_EVAL:-$([ -f "$DATA/s4/replay_clean_q38.pt" ] && echo "$DATA/s4/replay_clean_q38.pt")}
read -r -a RUNG_LIST <<<"${RUNGS:-1200 2400}"
AUTO_EXTEND=${AUTO_EXTEND:-1}
GRAD_CKPT=${GRAD_CKPT:-auto}
[[ $GRAD_CKPT =~ ^(auto|0|1)$ ]] || die "GRAD_CKPT must be auto, 0 or 1"   # refused before anything is written
MEM_LIMIT_GIB=${MEM_LIMIT_GIB:-115}
ARM_MAX_ATTEMPTS=${ARM_MAX_ATTEMPTS:-8}
ARM_MAX_NOPROGRESS=${ARM_MAX_NOPROGRESS:-2}
ARM_BACKOFF_S=${ARM_BACKOFF_S:-30}
PORT_BASE=${PORT_BASE:-29511}
ARMS=(wide ctrl)
G=${GPUS_PER_RUN:-$((EXPECT_GPUS / 2))}
[[ $G =~ ^[0-9]+$ ]] && [ "$G" -ge 1 ] || die "GPUS_PER_RUN='$G' is not a positive integer"
[ $((2 * G)) -le "$EXPECT_GPUS" ] || die "2 arms x $G GPUs is more than EXPECT_GPUS=$EXPECT_GPUS"
[ "${#RUNG_LIST[@]}" -ge 1 ] || die "RUNGS is empty"
for r in "${RUNG_LIST[@]}"; do [[ $r =~ ^[0-9]+$ ]] || die "bad rung '$r' in RUNGS"; done

for a in "$@"; do
  case "$a" in --steps|--steps=*|--arm|--arm=*|--out|--out=*|--seed|--seed=*|--resume|--grad_ckpt|--grad_ckpt=*)
    die "$a is set by run_t1.sh (RUNGS, the arm loop, TAG, SEED, GRAD_CKPT); do not pass it" ;;
  --eval_only|--eval_only=*|--mem_probe|--mem_probe=*|--max_steps_debug|--max_steps_debug=*)
    # each of these ends a launch before --steps, so no rung is ever reached and every arm burns its retries
    die "$a stops train_t1.py before the rung; run it by hand, not through run_t1.sh" ;; esac
done
[ -x "$PYBIN/torchrun" ] || die "no torchrun at $PYBIN/torchrun"
[ -d "$MODEL" ] || die "no model directory $MODEL"
[ -f "$LM_REPLAY" ] || die "no replay file $LM_REPLAY (REPLAY=$REPLAY; see the header for the push from the rig)"
OUT=$RUNS/$TAG
mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) run_t1 $TAG s$SEED: $*" | tee -a "$OUT/run_t1.log"; }
COMMON=(--model "$MODEL" --seed "$SEED" --lm_replay "$LM_REPLAY" --resume)
[ -n "$TEXT_EVAL" ] && COMMON+=(--text_eval "$TEXT_EVAL")
EXTRA=("$@")

if [ -f "$NVLS_FILE" ]; then
  nvls=$(sed -n 's/^NCCL_NVLS_ENABLE=\([01]\)$/\1/p' "$NVLS_FILE" | tail -n1)
  [ -n "$nvls" ] || die "$NVLS_FILE has no NCCL_NVLS_ENABLE=0|1 line"
  export NCCL_NVLS_ENABLE=$nvls
fi
if [ -f "$OUT/PROBE_DONE" ]; then say "already decided: $(cat "$OUT/PROBE_DONE")"; exit 0; fi

# ---- gradient checkpointing: decided once per TAG, the same for both arms and every resume
if [ -f "$OUT/grad_ckpt.env" ]; then
  GRAD_CKPT=$(sed -n 's/^GRAD_CKPT=\([01]\)$/\1/p' "$OUT/grad_ckpt.env" | tail -n1)
  [ -n "$GRAD_CKPT" ] || die "$OUT/grad_ckpt.env has no GRAD_CKPT=0|1 line; delete it to probe again"
elif [ "$GRAD_CKPT" = auto ]; then
  say "mem probe: --grad_ckpt 0 on the largest micro-batch, GPU 0 (limit ${MEM_LIMIT_GIB} GiB)"
  rm -rf "$OUT/mem_probe"; mkdir -p "$OUT/mem_probe"
  CUDA_VISIBLE_DEVICES=0 "$PYBIN/torchrun" --nproc_per_node 1 --master_port $((PORT_BASE + 9)) "$HERE/train_t1.py" \
    --arm wide --out "$OUT/mem_probe" "${COMMON[@]}" "${EXTRA[@]}" --grad_ckpt 0 --mem_probe 2 \
    >"$OUT/mem_probe/stdout.log" 2>&1
  rc=$?
  peak=$(sed -n 's/.*"event": "mem_probe".*"peak_gib": \([0-9.]*\).*/\1/p' "$OUT/mem_probe/log.jsonl" 2>/dev/null | tail -n1)
  if [ "$rc" -eq 0 ] && [ -n "$peak" ] && awk -v p="$peak" -v l="$MEM_LIMIT_GIB" 'BEGIN { exit !(p <= l) }'; then
    GRAD_CKPT=0
  else
    GRAD_CKPT=1
  fi
  say "mem probe rc $rc peak ${peak:-none} GiB -> grad_ckpt $GRAD_CKPT"
  echo "GRAD_CKPT=$GRAD_CKPT" >"$OUT/.grad_ckpt.env.tmp" && mv -f "$OUT/.grad_ckpt.env.tmp" "$OUT/grad_ckpt.env"
else
  echo "GRAD_CKPT=$GRAD_CKPT" >"$OUT/.grad_ckpt.env.tmp" && mv -f "$OUT/.grad_ckpt.env.tmp" "$OUT/grad_ckpt.env"
fi

newest_step() { # run dir -> step of its newest ckpt_NNNNNN.pt, -1 when none
  local f n; f=$(find "$1" -maxdepth 1 -name 'ckpt_[0-9]*.pt' -printf '%f\n' 2>/dev/null | sort | tail -n1)
  [ -n "$f" ] || { echo -1; return 0; }
  n=${f#ckpt_}; echo $((10#${n%.pt}))
}
reached() { # run dir, step -> 0 when complete.json says at least that step and the test record exists
  local s
  s=$(sed -n 's/.*"steps": \([0-9]*\).*/\1/p' "$1/complete.json" 2>/dev/null)
  [ -n "$s" ] && [ "$s" -ge "$2" ] && [ -f "$1/$(printf 'test_step%06d.json' "$2")" ]
}

run_arm() { # arm index rung   (runs in its own subshell)
  local arm=$1 idx=$2 rung=$3
  local out="$OUT/${arm}_s${SEED}" port=$((PORT_BASE + idx)) g0=$((idx * G)) devs pid="" nap="" stop=0
  local attempt=0 noprog=0 rc before after
  devs=$(seq -s, "$g0" $((g0 + G - 1)))
  mkdir -p "$out"
  if reached "$out" "$rung"; then say "$arm at step $rung, skipping"; return 0; fi
  rm -f "$out/arm_failed"
  trap 'stop=1; [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null; [ -n "$nap" ] && kill "$nap" 2>/dev/null' TERM INT
  while :; do
    attempt=$((attempt + 1)); before=$(newest_step "$out")
    say "$arm launch $attempt to step $rung on GPUs $devs port $port grad_ckpt $GRAD_CKPT replay $LM_REPLAY ckpt_step $before"
    CUDA_VISIBLE_DEVICES=$devs "$PYBIN/torchrun" --nproc_per_node "$G" --master_port "$port" "$HERE/train_t1.py" \
      --arm "$arm" --out "$out" --steps "$rung" --grad_ckpt "$GRAD_CKPT" "${COMMON[@]}" "${EXTRA[@]}" \
      >>"$out/stdout.log" 2>&1 &
    pid=$!
    while :; do wait "$pid"; rc=$?; kill -0 "$pid" 2>/dev/null || break; done
    pid=""
    if [ "$stop" = 1 ]; then say "$arm stopped on request (rc $rc)"; return 143; fi
    if [ "$rc" -eq 0 ] && reached "$out" "$rung"; then say "$arm reached step $rung"; return 0; fi
    after=$(newest_step "$out")
    if [ "$after" -gt "$before" ]; then noprog=0; else noprog=$((noprog + 1)); fi
    say "$arm attempt $attempt exited $rc, checkpoint step $before -> $after, no-progress launches $noprog of $ARM_MAX_NOPROGRESS"
    if [ "$noprog" -ge "$ARM_MAX_NOPROGRESS" ] || [ "$attempt" -ge "$ARM_MAX_ATTEMPTS" ]; then
      echo "rc $rc after $attempt attempts at rung $rung, newest checkpoint step $after, $(date -u +%FT%TZ)" >"$out/arm_failed"
      say "$arm ARM_FAILED (see $out/stdout.log)"; return 1
    fi
    sleep "$ARM_BACKOFF_S" & nap=$!; wait "$nap" 2>/dev/null; nap=""
    if [ "$stop" = 1 ]; then say "$arm stopped during backoff"; return 143; fi
  done
}

PIDS=()
# shellcheck disable=SC2329  # called through the trap below
stop_all() { local p; for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done; }
trap stop_all TERM INT

n=${#RUNG_LIST[@]}
for ((k = 0; k < n; k++)); do
  rung=${RUNG_LIST[k]}; final=$([ $((k + 1)) -eq "$n" ] && echo 1 || echo 0)
  dec=$OUT/$(printf 'decision_step%06d.json' "$rung")
  if [ ! -f "$dec" ]; then
    PIDS=()
    for i in "${!ARMS[@]}"; do run_arm "${ARMS[i]}" "$i" "$rung" & PIDS+=($!); done
    rc=0
    for p in "${PIDS[@]}"; do
      while :; do wait "$p"; r=$?; kill -0 "$p" 2>/dev/null || break; done
      [ "$r" -eq 0 ] || rc=$r
    done
    [ "$rc" -eq 0 ] || { say "rung $rung not finished (rc $rc)"; exit "$rc"; }
    "$PYBIN/python" "$HERE/decide.py" --root "$OUT" --seed "$SEED" --step "$rung" --final "$final" \
      | tee -a "$OUT/run_t1.log" || { say "decide.py failed at rung $rung"; exit 1; }
  fi
  verdict=$("$PYBIN/python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["verdict"])' "$dec") \
    || { say "unreadable $dec"; exit 1; }
  say "rung $rung verdict $verdict"
  if [ "$verdict" = EXTEND ] && [ "$final" = 0 ]; then
    [ "$AUTO_EXTEND" = 1 ] && continue
    echo "EXTEND at step $rung, AUTO_EXTEND=0: the owner decides the next rung" >"$OUT/PROBE_DONE"; exit 0
  fi
  echo "$verdict at step $rung ($dec)" >"$OUT/PROBE_DONE"
  exit 0
done
echo "no verdict after rungs ${RUNG_LIST[*]}" >"$OUT/PROBE_DONE"
exit 0
