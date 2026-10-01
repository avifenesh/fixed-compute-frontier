#!/usr/bin/env bash
# accept.sh - runs ON a box after bootstrap.sh. Prints ACCEPT_OK, or ACCEPT_FAILED <step>, and exits 0/1.
#
#   accept.sh [--arm wide] [--from STEP | --only STEP] [--gpus 8]
#
# Steps, in this order; the first failure stops the run (the failing step and its log are printed):
#   1 cuda_alloc    a real allocation and bf16 GEMM on EVERY GPU (count must equal --gpus)
#   2 gpu_report    nvidia-smi name, compute capability, driver, memory; one capability across the box; nvcc
#   3 nccl          all-reduce over every GPU (the fabric the DDP trainer uses) at 64 and 512 MiB: result checked, bus
#                   bandwidth printed and held to MIN_NCCL_GBPS (default 60, an NVLink-vs-PCIe check; 0 disables).
#                   Run with NCCL_NVLS_ENABLE=0 and =1, interleaved, NVLS_ROUNDS (3) each, every run under
#                   NCCL_PROBE_TIMEOUT_S (600): 1 wins only when its median summed time is at least NVLS_MIN_GAIN (0.03)
#                   below the median of 0 and every run with 1 passed; otherwise 0 (a hang, a wrong result or a slow
#                   fabric with 1 is a 0). NVLS_PIN=0|1 skips the A/B. The chosen value is written to ONE file,
#                   $NVLS_FILE (boxenv.sh: $W/nccl_nvls.env), which run_pair.sh (training) reads and this script reads.
#   4 kernel_check  experiments/negeig_retrofit/kernel_check.py: fla chunk_gated_delta_rule with beta in (0, 2)
#                   against (0, 1), frozen pass rule, prints "G1 PASS"
#   5 g0            g0_noop.py on Qwen3.8-27B: widened arm at W = 0 is bit-identical to the untouched model,
#                   W reaches beta > 1 and gets gradient
#   6 verl_env      the verl venv: sglang 0.5.20, flash-attn varlen on this arch (if it fails, run GRPO with ATTN=sdpa)
#   7 sglang        SGLang 0.5.20 serves the UNTOUCHED model and answers one chat request correctly
#   8 train_smoke   the real trainer through run_pair.sh with the Stage A flags, three parts (H = --gpus):
#                   a) two arms (wide, ctrl) of H/2 ranks each: 30 steps, evals and one save at 30, recall, mem_probe 1,
#                      profile_step 10
#                   b) ONE arm (wide) on H ranks, 20 steps, the same global batch: H-rank against H/2-rank tok/s
#                   c) the grad_ckpt probe: mem_probe 2 with --grad_ckpt 0 on the longest row of every pool, decided
#                      against GRAD_CKPT_MAX_GIB (default the smaller of 223 and 0.85 x GPU memory; a CUDA OOM in the
#                      probe is the answer "on")
#                   Receipts in $ACC: smoke_report.json (tok/s per GPU, peak memory, per-rank compute spread, profile
#                   shares, mem_probe, H against H/2), smoke.env, grad_ckpt.env (GRAD_CKPT_DECISION, GRAD_CKPT_FLAGS),
#                   smoke/<tag>_<arm>/ (log.jsonl, profile). Refused while the training unit is active or a GPU holds
#                   more than SMOKE_MAX_USED_MIB (2048): the smoke would collide with a real run.
#                   The smoke uses the self-distilled replay when $SD has both files (replay=present). Before
#                   self-distillation it runs without (replay=none): the tool pool, the longest rows, is not probed and
#                   the grad_ckpt decision is "undecided". Re-run `accept.sh --from train_smoke` after self-distillation.
#                   REQUIRE_REPLAY=1 makes a missing replay a failure. Knobs: SMOKE_LR (1e-4), SMOKE_TIMEOUT_S (5400 a
#                   part), MIN_TOK_S (floor on the H/2-rank steady rate of each arm), SMOKE_BASELINE_TOK_S (15200: the
#                   PLAN's 3,800 per GPU x 4, a planning figure and not a measurement).
#
# --from STEP skips earlier steps that passed on this boot and runs from there. --only STEP runs that one step, prints
# ONLY_OK, and leaves ACCEPT_OK and accept.json as they are (quick re-measures: `--only nccl`; tests). A full or --from run
# removes the old ACCEPT_OK and accept.json when it starts, so a failed re-run leaves no stale success marker.
# Receipts: $ACC/*.log, $ACC/steps.tsv, $ACC/accept.json, and $ACC/ACCEPT_OK on success.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=boxenv.sh
. "$HERE/boxenv.sh"
# shellcheck source=sglang_lib.sh
. "$HERE/sglang_lib.sh"
E27_SRC=$(cd "$HERE/.." && pwd)    # run_pair.sh and train27.py live next to the box directory
ARM=wide; FROM=""; ONLY=""; GPUS=$EXPECT_GPUS
while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM=$2; shift 2 ;;
    --from) FROM=$2; shift 2 ;;
    --only) ONLY=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) bdie "unknown argument: $1" ;;
  esac
done
[ -z "$FROM" ] || [ -z "$ONLY" ] || bdie "--from and --only are exclusive"
EXPECT_GPUS=$GPUS
mkdir -p "$ACC" "$RUNS"
BOOT=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo noboot)
STEPS=(cuda_alloc gpu_report nccl kernel_check g0 verl_env sglang train_smoke)
for sel in "$FROM" "$ONLY"; do
  [ -z "$sel" ] || [[ " ${STEPS[*]} " == *" $sel "* ]] || bdie "unknown step '$sel' (steps: ${STEPS[*]})"
done
[ -z "$ONLY" ] || STEPS=("$ONLY")
PORT=${PORT:-30001}

[ -x "$PY" ] || { echo "ACCEPT_FAILED setup: no training venv at $PY (run bootstrap.sh)"; exit 1; }
[ -x "$PYBIN/torchrun" ] || { echo "ACCEPT_FAILED setup: no torchrun at $PYBIN/torchrun (run bootstrap.sh)"; exit 1; }

step_log() { echo "$ACC/$1.log"; }

# The NCCL_NVLS_ENABLE value run_pair.sh will export for training, from the one file; empty before accept.sh measured it.
nvls_value() { sed -n 's/^NCCL_NVLS_ENABLE=\([01]\)$/\1/p' "$NVLS_FILE" 2>/dev/null | tail -n1; }
# KEY value of an env-style receipt file
env_value() { sed -n "s/^$2=//p" "$1" 2>/dev/null | tail -n1 | tr -d '"'; }

# ------------------------------------------------------------------------------------------------ steps
s_cuda_alloc() {
  EXPECT_GPUS=$EXPECT_GPUS "$PY" - <<'PYEOF'
import os, torch
n = torch.cuda.device_count()
want = int(os.environ["EXPECT_GPUS"])
assert n == want, f"{n} CUDA devices, expected {want}"
for i in range(n):
    x = torch.randn(4096, 4096, device=f"cuda:{i}", dtype=torch.bfloat16)
    y = (x @ x).float().sum()
    big = torch.empty(8 * 2**30, dtype=torch.uint8, device=f"cuda:{i}")  # 8 GiB real allocation
    torch.cuda.synchronize(i)
    assert torch.isfinite(y).item()
    print(f"cuda:{i} {torch.cuda.get_device_name(i)} alloc 8 GiB + bf16 GEMM ok")
    del big
print("cuda_alloc ok on", n, "devices")
PYEOF
}

s_gpu_report() {
  nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv || return 1
  local caps names cnt
  caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u)
  names=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)
  cnt=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  echo "compute capability: $caps"; echo "gpu model: $names"; echo "gpus: $cnt"
  [ "$cnt" -eq "$EXPECT_GPUS" ] || { echo "expected $EXPECT_GPUS GPUs"; return 1; }
  [ "$(echo "$caps" | wc -l)" -eq 1 ] || { echo "mixed compute capabilities"; return 1; }
  [ "$(echo "$names" | wc -l)" -eq 1 ] || { echo "mixed GPU models"; return 1; }
  local ch; ch=$(find_cuda_home) && "$ch/bin/nvcc" --version | tail -n2 || echo "no nvcc under /usr/local (verl stage would have failed)"
  "$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "arch_list", torch.cuda.get_arch_list())'
  echo "CAPS=$caps" >"$ACC/caps.env"; echo "MODELNAME=$names" >>"$ACC/caps.env"
}

# nccl_one NVLS OUT: the probe under NCCL_NVLS_ENABLE=NVLS, a timeout and the bandwidth floor; 0 when it passed.
# The floor only separates NVLink from a PCIe fallback (PCIe Gen5 all-reduce tops out near 40 GB/s bus bandwidth, NVLink
# boxes are several hundred); it is a setup sanity check, not a performance claim. MIN_NCCL_GBPS=0 turns it off.
nccl_one() {
  local v=$1 o=$2 rc
  # shellcheck disable=SC2086  # the sizes are a word list
  MIN_NCCL_GBPS=${MIN_NCCL_GBPS:-60} NCCL_NVLS_ENABLE=$v timeout --kill-after=30 "${NCCL_PROBE_TIMEOUT_S:-600}" \
    "$PYBIN/torchrun" --standalone --nproc_per_node "$EXPECT_GPUS" "$ACC/nccl_probe.py" ${NCCL_PROBE_MIB:-64 512} >"$o" 2>&1
  rc=$?
  echo "--- NCCL_NVLS_ENABLE=$v rc=$rc ($o)"; tail -n 12 "$o"
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    pkill -KILL -f "$ACC/nccl_probe[.]py" 2>/dev/null   # this box's probe only (the path is under $ACC)
    echo "probe timed out or was killed (rc $rc): treated as a hang"
  fi
  return "$rc"
}
nccl_ms() { # probe output -> the summed ms of its NCCL_RESULT lines
  awk '/^NCCL_RESULT/ { for (i = 1; i <= NF; i++) if ($i ~ /^ms=/) { sub("ms=", "", $i); s += $i } } END { printf "%.3f", s }' "$1"
}
median() { printf '%s\n' "$@" | sort -g | awk '{ v[NR] = $1 } END { print (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2 }'; }

s_nccl() {
  cat >"$ACC/nccl_probe.py" <<'PYEOF'
import os, sys, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
# correctness first: a sum of ones over w ranks must be exactly w on every element
chk = torch.ones(1 << 20, device="cuda", dtype=torch.float32)
dist.all_reduce(chk)
assert bool((chk == w).all().item()), f"all_reduce wrong: {chk[:4].tolist()} expected {w}"
sizes = [int(a) for a in sys.argv[1:]] or [512]   # MiB of bf16; the floor applies to the last (largest) size
floor = float(os.environ.get("MIN_NCCL_GBPS", "0"))
busbw = 0.0
for mib in sizes:
    x = torch.ones(mib * 2**20 // 2, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t = time.time()
    iters = 10
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.time() - t) / iters
    busbw = 2 * (w - 1) / w * x.numel() * 2 / dt / 1e9
    if r == 0:
        print(f"NCCL_RESULT mib={mib} ms={dt*1e3:.3f} busbw_gbs={busbw:.1f} world={w} nvls={os.environ.get('NCCL_NVLS_ENABLE', 'unset')}")
    del x
dist.barrier()
dist.destroy_process_group()
if r == 0 and busbw < floor:
    print(f"NCCL_TOO_SLOW: {busbw:.0f} GB/s is below MIN_NCCL_GBPS {floor:.0f}: the GPUs are not on the NVLink fabric")
    sys.exit(1)
PYEOF
  local rounds=${NVLS_ROUNDS:-3} pin=${NVLS_PIN:-} gain=${NVLS_MIN_GAIN:-0.03} r out chosen why m0 m1=""
  local -a t0s=() t1s=()
  local pct; pct=$(awk -v g="$gain" 'BEGIN { printf "%.0f", g * 100 }')
  if [ -n "$pin" ]; then
    [[ $pin =~ ^[01]$ ]] || { echo "NVLS_PIN must be 0 or 1, got '$pin'"; return 1; }
    nccl_one "$pin" "$ACC/nccl_nvls$pin.r1.out" || { echo "NCCL probe failed with NCCL_NVLS_ENABLE=$pin (pinned)"; return 1; }
    m0=$(nccl_ms "$ACC/nccl_nvls$pin.r1.out"); [ "$pin" = 1 ] && m1=$m0
    chosen=$pin; why="pinned by NVLS_PIN, median ${m0} ms"
  else
    local nvls1_ok=1
    for ((r = 1; r <= rounds; r++)); do   # interleaved 0, 1, 0, 1, ... so clock drift hits both alike
      out="$ACC/nccl_nvls0.r$r.out"
      nccl_one 0 "$out" || { echo "NCCL probe failed with NCCL_NVLS_ENABLE=0 (round $r): the box fails the baseline"; return 1; }
      t0s+=("$(nccl_ms "$out")")
      out="$ACC/nccl_nvls1.r$r.out"
      if nccl_one 1 "$out"; then t1s+=("$(nccl_ms "$out")"); else nvls1_ok=0; echo "NVLS=1 failed in round $r: NVLS=0 it is"; break; fi
    done
    m0=$(median "${t0s[@]}")
    if [ "$nvls1_ok" = 1 ]; then
      m1=$(median "${t1s[@]}")
      if awk -v a="$m0" -v b="$m1" -v g="$gain" 'BEGIN { exit !(b <= a * (1 - g)) }'; then
        chosen=1; why="NVLS=1 median ${m1} ms against ${m0} ms for NVLS=0: at least ${pct}% faster"
      else
        chosen=0; why="NVLS=1 median ${m1} ms against ${m0} ms for NVLS=0: not at least ${pct}% faster, the safe value wins ties"
      fi
    else
      chosen=0; why="NVLS=1 failed or hung, NVLS=0 median ${m0} ms"
    fi
  fi
  # one file, written atomically: run_pair.sh and accept.sh read NCCL_NVLS_ENABLE from here and from nowhere else
  { echo "# measured by accept.sh nccl on $(hostname) at $(date -u +%FT%TZ): $why"
    echo "NCCL_NVLS_ENABLE=$chosen"; } >"$NVLS_FILE.tmp" && mv -f "$NVLS_FILE.tmp" "$NVLS_FILE" || return 1
  jq -n --arg host "$(hostname)" --arg at "$(date -u +%FT%TZ)" --arg why "$why" --argjson chosen "$chosen" \
    --arg m0 "$m0" --arg m1 "$m1" --argjson rounds "$rounds" --arg pin "$pin" \
    '{host:$host, at:$at, chosen:$chosen, reason:$why, median_sum_ms_nvls0:$m0, median_sum_ms_nvls1:$m1, rounds:$rounds, pinned:$pin}' \
    >"$ACC/nccl_nvls.json"
  echo "NVLS_DECISION NCCL_NVLS_ENABLE=$chosen ($why) written to $NVLS_FILE"
}

s_kernel_check() {
  cd "$RETRO" || return 1
  CUDA_VISIBLE_DEVICES=0 "$PY" kernel_check.py --out "$ACC/g1_kernel.json" | tee "$ACC/kernel_check.out"
  grep -q '^G1 PASS$' "$ACC/kernel_check.out"
}

s_g0() {
  cd "$RETRO" || return 1
  [ -f "$MODEL_DIR/.negeig_revision" ] || { echo "model not staged ($MODEL_DIR/.negeig_revision missing)"; return 1; }
  CUDA_VISIBLE_DEVICES=0 "$PY" g0_noop.py "$MODEL_DIR" "$ACC/g0_27b_$ARM.json" "$ARM" || return 1
  "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("g0 pass:", r["pass"], "bit_identical_at_w0:", r["bit_identical_at_w0"], "n_gdn_layers:", r["n_gdn_layers"], "n_gates:", r["n_gates"]); sys.exit(0 if r["pass"] else 1)' "$ACC/g0_27b_$ARM.json"
}

s_verl_env() {
  local ch; ch=$(find_cuda_home) || { echo "no nvcc"; return 1; }
  PATH="${VPY%/*}:$ch/bin:$PATH" CUDA_HOME=$ch CUDA_VISIBLE_DEVICES=0 "$VPY" - <<'PYEOF' || return 1
import importlib.metadata as md
import torch
assert md.version("sglang") == "0.5.20", md.version("sglang")
print("sglang", md.version("sglang"), "torch", torch.__version__, "transformers", md.version("transformers"))
try:
    from flash_attn import flash_attn_varlen_func
    q = torch.randn(64, 8, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
    out = flash_attn_varlen_func(q, q, q, cu, cu, 32, 32, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(q[:32].transpose(0, 1), q[:32].transpose(0, 1),
                                                           q[:32].transpose(0, 1), is_causal=True).transpose(0, 1)
    err = (out[:32].float() - ref.float()).abs().max().item()
    assert err < 2e-2, err
    print(f"flash-attn varlen ok on this arch, max err vs sdpa {err:.2e}")
except Exception as e:  # not fatal for acceptance: GRPO then runs with ATTN=sdpa
    print("FLASH_ATTN_FAILED (run_grpo.sh ATTN=sdpa is the fallback):", repr(e)[:300])
PYEOF
}

s_sglang() {
  # The runner pipes every step through tee, so a step runs in its own subshell and the runner's EXIT trap never sees
  # SGLANG_PID: the step cleans up after itself on every exit path (failure, SIGINT, SIGTERM), or a failed probe would
  # leave a server holding GPU 0 for the next step.
  trap '[ -z "$SGLANG_PID" ] || stop_sglang >/dev/null 2>&1' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  local ch; ch=$(find_cuda_home) || { echo "no nvcc"; return 1; }
  local slog="$ACC/sglang_server.log"
  # the verl venv bin carries ninja, which the flashinfer JIT runs by name
  PATH="${VPY%/*}:$ch/bin:$PATH" CUDA_HOME=$ch CUDA_VISIBLE_DEVICES=0 setsid "$VPY" -m sglang.launch_server \
    --model-path "$MODEL_DIR" --tp-size 1 --dp-size 1 --host 127.0.0.1 --port "$PORT" --mem-fraction-static 0.85 \
    --context-length 8192 --mamba-full-memory-ratio 3 --trust-remote-code >"$slog" 2>&1 </dev/null &
  SGLANG_PID=$!
  # the first launch on a new architecture JIT-compiles kernels and captures graphs: allow 25 minutes
  wait_sglang_healthy "$PORT" "$SGLANG_PID" "$slog" 150 || return 1
  # One liveness question with a checkable answer. Think-off, temperature 0: this is a correctness probe of the
  # serving kernels on this arch, not a perf or quality measurement.
  local resp; resp=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model": "default", "max_tokens": 64, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [{"role": "user", "content": "What is 17 times 23? Answer with the number only."}]}')
  echo "$resp" >"$ACC/sglang_chat_response.json"
  local content; content=$(echo "$resp" | jq -r '.choices[0].message.content // empty')
  echo "chat answer: $content"
  echo "$content" | grep -q '391' || { echo "expected 391 in the answer"; return 1; }
  stop_sglang || { echo "sglang would not stop"; return 1; }
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  [ "${used:-99999}" -lt 2048 ] || { echo "GPU 0 still holds ${used} MiB after stopping sglang"; return 1; }
  echo "sglang stopped, GPU 0 freed"
}

# ---- train_smoke ----------------------------------------------------------------------------------------------
# pair_part LABEL TAG "ARMS" RANKS_PER_ARM FLAGS...: one run_pair.sh invocation with the Stage A recipe. One launch per arm
# (a smoke that crashes fails the acceptance, it is not retried) and a wall-clock limit. 0 when every arm exited 0.
pair_part() {
  local label=$1 tag=$2 arms=$3 gpr=$4 rc; shift 4
  echo "--- smoke part $label: tag $tag, arms [$arms] x $gpr ranks, flags: $*"
  EXPECT_GPUS=$EXPECT_GPUS STAGE=A ALLOW_NO_REPLAY=$SMOKE_ALLOW_NO_REPLAY ARMS=$arms GPUS_PER_RUN=$gpr \
    ARM_MAX_ATTEMPTS=1 ARM_MAX_NOPROGRESS=1 \
    timeout --kill-after=60 "${SMOKE_TIMEOUT_S:-5400}" bash "$E27_SRC/run_pair.sh" "$tag" 0 --lr "${SMOKE_LR:-1e-4}" "$@"
  rc=$?
  [ "$rc" -ne 124 ] || echo "smoke part $label hit SMOKE_TIMEOUT_S=${SMOKE_TIMEOUT_S:-5400}"
  return "$rc"
}
show_failed_arms() { # run dirs...
  local d
  for d in "$@"; do
    [ ! -f "$d/arm_failed" ] || echo "arm_failed: $d: $(cat "$d/arm_failed")"
    if [ -f "$d/stdout.log" ]; then echo "--- tail of $d/stdout.log"; tail -n 25 "$d/stdout.log"; fi
  done
}
keep_receipt() { # run dir -> $ACC/smoke/<tag>_<arm>/ : the files a reader needs, without the checkpoints
  local d=$1 name f; name="$(basename "$(dirname "$d")")_$(basename "$d")"
  mkdir -p "$ACC/smoke/$name"
  for f in log.jsonl complete.json profile_rank0.json profile_rank0.txt launch.log arm_failed; do
    if [ -f "$d/$f" ]; then cp "$d/$f" "$ACC/smoke/$name/"; fi
  done
  if [ -f "$d/stdout.log" ]; then tail -n 300 "$d/stdout.log" >"$ACC/smoke/$name/stdout.tail.log"; fi
}

s_train_smoke() {
  local H=$EXPECT_GPUS half used maxgib mib replay base probe_oom=0 rc
  [ "$H" -ge 2 ] && [ $((H % 2)) -eq 0 ] || { echo "--gpus $H must be even and at least 2 (two arms of H/2 ranks)"; return 1; }
  half=$((H / 2))
  [ -n "$RUNS" ] && [ -f "$E27_SRC/run_pair.sh" ] || { echo "no run_pair.sh at $E27_SRC"; return 1; }
  # a smoke on a box that is training would collide with the real run
  if [ "${NO_SYSTEMD:-0}" != 1 ] && command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet "$TRAIN_UNIT" 2>/dev/null; then
    echo "$TRAIN_UNIT is active: ctl.sh hold (or disarm) first, the smoke needs every GPU"; return 1
  fi
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' \
    | awk -F, -v max="${SMOKE_MAX_USED_MIB:-2048}" '$2 > max { printf "GPU %s holds %s MiB; ", $1, $2 }')
  [ -z "$used" ] || { echo "GPUs are not free: $used(a server or a training run is on them)"; return 1; }
  [ -n "$(nvls_value)" ] || { echo "no NCCL_NVLS_ENABLE in $NVLS_FILE: run the nccl step first (accept.sh --only nccl)"; return 1; }
  replay=none
  if [ -s "${CHAT_REPLAY:-$SD/chat_sft.jsonl}" ] && [ -s "${TOOL_REPLAY:-$SD/tool_sft.jsonl}" ]; then replay=present; fi
  if [ "$replay" = none ] && [ "${REQUIRE_REPLAY:-0}" = 1 ]; then
    echo "REQUIRE_REPLAY=1 and no self-distilled replay in ${SD} (chat_sft.jsonl, tool_sft.jsonl): run self-distillation first"; return 1
  fi
  SMOKE_ALLOW_NO_REPLAY=0; [ "$replay" = present ] || SMOKE_ALLOW_NO_REPLAY=1
  echo "replay=$replay, NCCL_NVLS_ENABLE=$(nvls_value), $H GPUs ($half ranks an arm), lr ${SMOKE_LR:-1e-4}"
  [ "$replay" = present ] || echo "WARNING: no replay: the smoke trains without the Stage A mix, the tool pool is not probed and grad_ckpt stays undecided"

  rm -rf "$ACC/smoke" "$RUNS/accept-smoke" "$RUNS/accept-smoke8" "$RUNS/accept-probe"
  # (a) two arms of H/2 ranks: the Stage A flags and a short run that exercises evals, the checkpoint writer, recall,
  # the memory probe (grad_ckpt on) and the profiler
  pair_part a accept-smoke "wide ctrl" "$half" --steps 30 --eval_every 30 --save_every 30 --recall_every 30 --mem_probe 1 --profile_step 10 \
    || { echo "smoke part a failed"; show_failed_arms "$RUNS/accept-smoke/wide_s0" "$RUNS/accept-smoke/ctrl_s0"; return 1; }
  # (b) ONE arm on all H ranks, the same global batch (the trainer's batch does not depend on the world size): timing only
  pair_part b accept-smoke8 "wide" "$H" --steps 20 --eval_every 0 --save_every 0 --recall_every 0 \
    || { echo "smoke part b failed"; show_failed_arms "$RUNS/accept-smoke8/wide_s0"; return 1; }
  # (c) the grad_ckpt decision probe: fwd+bwd with checkpointing OFF on the longest row of every pool, then exit
  pair_part c accept-probe "wide" "$half" --steps 1 --eval_every 0 --save_every 0 --recall_every 0 --mem_probe 2 --grad_ckpt 0
  rc=$?
  if [ "$rc" -ne 0 ]; then
    # checkpointing off did not fit: that IS the decision (on). Any other failure is a failure.
    if grep -qiE 'out of memory|OutOfMemoryError' "$RUNS/accept-probe/wide_s0/stdout.log" 2>/dev/null; then
      probe_oom=1; echo "the grad_ckpt=0 probe ran out of CUDA memory: decision on"
    else
      echo "smoke part c failed (rc $rc), not an out-of-memory"; show_failed_arms "$RUNS/accept-probe/wide_s0"; return 1
    fi
  fi
  local d; for d in "$RUNS"/accept-smoke/*_s0 "$RUNS"/accept-smoke8/*_s0 "$RUNS"/accept-probe/*_s0; do keep_receipt "$d"; done

  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  maxgib=${GRAD_CKPT_MAX_GIB:-$(awk -v m="${mib:-0}" 'BEGIN { c = 0.85 * m / 1024; print (m > 0 && c < 223) ? sprintf("%.1f", c) : 223 }')}
  base=${SMOKE_BASELINE_TOK_S:-15200}
  local -a extra=()
  [ -z "${MIN_TOK_S:-}" ] || extra+=(--min-tok-s "$MIN_TOK_S")
  [ "$probe_oom" = 0 ] || extra+=(--probe-oom)
  "$PY" "$HERE/smoke_report.py" --four "wide=$RUNS/accept-smoke/wide_s0" --four "ctrl=$RUNS/accept-smoke/ctrl_s0" \
    --eight "$RUNS/accept-smoke8/wide_s0" --probe "$RUNS/accept-probe/wide_s0" --out "$ACC/smoke_report.json" \
    --env-out "$ACC/smoke.env.tmp" --max-gib "$maxgib" --baseline-tok-s "$base" --stage "replay=$replay" "${extra[@]}"
  rc=$?
  { cat "$ACC/smoke.env.tmp" 2>/dev/null; echo "SMOKE_REPLAY=$replay"; } >"$ACC/smoke.env"; rm -f "$ACC/smoke.env.tmp"
  # the decision file the runbook sources for Stage A: GRAD_CKPT_FLAGS is what the launch line adds
  local dec flags
  dec=$(jq -r '.grad_ckpt.decision // "undecided"' "$ACC/smoke_report.json" 2>/dev/null || echo undecided)
  case "$dec" in off) flags="--grad_ckpt 0" ;; *) flags="--grad_ckpt 1" ;; esac
  { echo "# written by accept.sh train_smoke on $(hostname) at $(date -u +%FT%TZ), replay=$replay: $(jq -r '.grad_ckpt.reason // "no probe"' "$ACC/smoke_report.json" 2>/dev/null)"
    echo "GRAD_CKPT_DECISION=$dec"
    echo "GRAD_CKPT_FLAGS=\"$flags\""
    echo "GRAD_CKPT_PROBE_PEAK_GIB=$(jq -r '.grad_ckpt.probe_peak_gib // "none"' "$ACC/smoke_report.json" 2>/dev/null)"
    echo "GRAD_CKPT_MAX_GIB=$maxgib"
    echo "REPLAY=$replay"; } >"$ACC/grad_ckpt.env.tmp" && mv -f "$ACC/grad_ckpt.env.tmp" "$ACC/grad_ckpt.env"
  echo "GRAD_CKPT_DECISION=$dec GRAD_CKPT_FLAGS=\"$flags\" (receipt $ACC/grad_ckpt.env)"
  [ "$rc" -eq 0 ] || { echo "smoke_report found problems (listed above, receipt $ACC/smoke_report.json)"; return 1; }
}

# ------------------------------------------------------------------------------------------------ runner
prev_pass() { # step -> 0 if steps.tsv records PASS for it on this boot
  [ -f "$ACC/steps.tsv" ] && awk -F'\t' -v s="$1" -v b="$BOOT" '$1==s && $2=="PASS" && $4==b {f=1} END {exit !f}' "$ACC/steps.tsv"
}
# The sglang step stops the server it started (its own EXIT trap): an --only nccl run must not kill a self-distillation
# server that is up, so the runner itself stops nothing.

[ -n "$ONLY" ] || rm -f "$ACC/ACCEPT_OK" "$ACC/accept.json"
skipping=0; [ -n "$FROM" ] && skipping=1
started=$(date +%s)
for s in "${STEPS[@]}"; do
  if [ "$skipping" = 1 ]; then
    if [ "$s" = "$FROM" ]; then skipping=0; elif prev_pass "$s"; then echo "== $s: PASS on this boot (skipped)"; continue; else skipping=0; fi
  fi
  blog "== $s"
  log=$(step_log "$s"); t0=$(date +%s)
  "s_$s" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  dt=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 0 ]; then
    printf '%s\tPASS\t%s\t%s\n' "$s" "$dt" "$BOOT" >>"$ACC/steps.tsv"; blog "$s PASS (${dt}s)"
  else
    printf '%s\tFAIL\t%s\t%s\n' "$s" "$dt" "$BOOT" >>"$ACC/steps.tsv"
    echo "ACCEPT_FAILED $s (rc $rc, log $log)"; exit 1
  fi
done

if [ -n "$ONLY" ]; then echo "ONLY_OK $ONLY"; exit 0; fi

host=$(hostname); caps=$(env_value "$ACC/caps.env" CAPS)
toks=$(env_value "$ACC/smoke.env" SMOKE_TOK_PER_S); tpg=$(env_value "$ACC/smoke.env" SMOKE_TOK_PER_S_PER_GPU)
peak=$(env_value "$ACC/smoke.env" SMOKE_PEAK_MEM_GIB); sp8=$(env_value "$ACC/smoke.env" SMOKE8_SPEEDUP)
replay=$(env_value "$ACC/smoke.env" SMOKE_REPLAY); gck=$(env_value "$ACC/grad_ckpt.env" GRAD_CKPT_DECISION)
nvls=$(nvls_value)
line="ACCEPT_OK $(date -u +%FT%TZ) host=$host gpus=$EXPECT_GPUS cc=${caps:-?} arm=$ARM nvls=${nvls:-?} replay=${replay:-?} grad_ckpt=${gck:-?} smoke_tok_per_s=${toks:-?} per_gpu=${tpg:-?} peak_gib=${peak:-?} speedup_8_vs_4=${sp8:-?} elapsed_s=$(( $(date +%s) - started ))"
echo "$line" | tee "$ACC/ACCEPT_OK"
jq -n --arg host "$host" --argjson gpus "$EXPECT_GPUS" --arg cc "${caps:-}" --arg arm "$ARM" --arg nvls "${nvls:-}" \
  --arg replay "${replay:-}" --arg gck "${gck:-}" --arg toks "${toks:-}" --arg tpg "${tpg:-}" --arg peak "${peak:-}" \
  --arg sp8 "${sp8:-}" --arg boot "$BOOT" \
  '{result:"ACCEPT_OK", host:$host, gpus:$gpus, compute_cap:$cc, arm:$arm, nccl_nvls_enable:$nvls, replay:$replay,
    grad_ckpt_decision:$gck, smoke_tok_per_s:$toks, smoke_tok_per_s_per_gpu:$tpg, smoke_peak_mem_gib:$peak,
    speedup_8_vs_4:$sp8, boot_id:$boot}' >"$ACC/accept.json"
