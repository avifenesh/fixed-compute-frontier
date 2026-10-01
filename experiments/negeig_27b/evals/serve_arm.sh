#!/usr/bin/env bash
# Serve one arm of the Stage A general eval through SGLang with data parallelism, and prove what is being served.
# Runs ON the GPU box (8x B300, or the H200 fallback). Never on the rig.
#
#   serve_arm.sh up   --arm NAME --kind base|ctrl|wide [--trainable trainable_NNNNNN.pt] [options]
#   serve_arm.sh down [--purge]
#   serve_arm.sh status
#
# up    base : the untouched model, plugin not loaded.
#       ctrl : merge-lora of the LoRA-only adapter into a copy of the base, plugin not loaded. The trainable must
#              hold no gate tensors (export-gates is run and must refuse it: "no negeig_w tensors").
#       wide : merge-lora of the LoRA part, export-gates for the fp32 W, plugin loaded (SGLANG_PLUGINS=negeig,
#              NEGEIG_GATES=<exported file>). Every rank must log "negeig: active" with the exported file's sha256.
#       Steps: validate, merge (receipt must keep delta_kept >= 0.95), export gates, install the plugin, launch with
#       DP, poll /health under a deadline (early exit when the server process dies), 17*23 liveness probe, engagement
#       check from the server log, then write <arm dir>/{serve,engagement,arm}.json. Exit 0 only when all of it holds;
#       on any failure the server is stopped and the reason is on stderr.
# down  stops the server recorded in <eval root>/serving.json (and any other SGLang server on the box), checks the GPUs
#       are free; --purge also deletes the merged model of that arm (about 54 GB).
#
# Options (env var in brackets): --dp N [DP, 8] --tp N [TP, 1] --port P [PORT, 30000] --ctx N [CTX, 131072: the same for
# every arm, the BFCL multi_turn_long_context items need it] --radix on|off [RADIX, off: logprob requests need the radix
# cache off] --ready-s S [READY_S, 2700] --mem-fraction F [MEMFRAC, 0.85] --replace (stop a running server first)
# --print-cmd (validate, print the commands, change nothing) --extra "ARGS" (appended to the launch_server command line;
# it is recorded in serve.json, and every arm of one comparison must use the same).
# Other env: EVAL_ROOT [$W/eval], MERGED_ROOT [$W/merged], PLUGIN_DIR [$W/negeig-plugin], ATTN_BACKEND [unset: SGLang
# default], MAMBA_RATIO [3], SERVE_VPY, SERVE_MODEL_DIR, NEGEIG_CLI [<e27>/sglang/negeig_sglang.py], NVIDIA_SMI.
# The launch flags that matter: all three --linear-attn-* backends are triton on EVERY arm including base (the patch
# raises on a gated layer otherwise, and a comparison needs one kernel set), no speculative decoding, no replayssm,
# no tf32 flag, bf16 explicit.
set -u -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
E27=${E27:-$(cd "$HERE/.." && pwd)}
BOXDIR=${BOXDIR:-$E27/box}
SRC_E27=$E27; SRC_BOXDIR=$BOXDIR
# shellcheck source=/dev/null
. "$SRC_BOXDIR/boxenv.sh"
# boxenv.sh points E27 and BOXDIR at $NEGEIG_W/code, which is this tree on the box and a scratch dir under test
E27=$SRC_E27; BOXDIR=$SRC_BOXDIR
# shellcheck source=/dev/null
. "$BOXDIR/sglang_lib.sh"
VPY=${SERVE_VPY:-$VPY}
MODEL_DIR=${SERVE_MODEL_DIR:-$MODEL_DIR}
NEGEIG_CLI=${NEGEIG_CLI:-$E27/sglang/negeig_sglang.py}
ARM_KIND_PY=$HERE/arm_kind.py
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}
EVAL_ROOT=${EVAL_ROOT:-$W/eval}
MERGED_ROOT=${MERGED_ROOT:-$W/merged}
PLUGIN_DIR=${PLUGIN_DIR:-$W/negeig-plugin}
SERVED_NAME=qwen3.8-27b
GDN_LAYERS=48                # linear_attention layers of Qwen3.8-27B (64 layers, 16 are full attention)
MERGE_SCALE=2.0              # train27.py uses lora_alpha = 2 * rank
MERGE_SEED=20261001
MIN_DELTA_KEPT=0.95

die() { echo "serve_arm: ERROR: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -u/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//' >&2; exit 2; }
step() { blog "serve_arm: $*"; }
pyjson() { # file key   (prints a top-level key of a json file)
  "$VPY" -c 'import json,sys; v=json.load(open(sys.argv[1]))[sys.argv[2]]; print(v if not isinstance(v,(dict,list)) else json.dumps(v))' "$1" "$2"
}
need() { [ $# -ge 2 ] || { echo "serve_arm: $1 needs a value" >&2; usage; }; }
sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

MODE=${1:-}
[ -n "$MODE" ] || usage
shift || true
ARM=""; KIND=""; TRAINABLE=""; PURGE=0; REPLACE=0; PRINT_CMD=0
DP=${DP:-8}; TP=${TP:-1}; PORT=${PORT:-30000}; CTX=${CTX:-131072}; RADIX=${RADIX:-off}
READY_S=${READY_S:-2700}; MEMFRAC=${MEMFRAC:-0.85}; MAMBA_RATIO=${MAMBA_RATIO:-3}; ATTN_BACKEND=${ATTN_BACKEND:-}
EXTRA=${EXTRA:-}
while [ $# -gt 0 ]; do
  case "$1" in
    --arm) need "$@"; ARM=$2; shift 2 ;;
    --kind) need "$@"; KIND=$2; shift 2 ;;
    --trainable) need "$@"; TRAINABLE=$2; shift 2 ;;
    --dp) need "$@"; DP=$2; shift 2 ;;
    --tp) need "$@"; TP=$2; shift 2 ;;
    --port) need "$@"; PORT=$2; shift 2 ;;
    --ctx) need "$@"; CTX=$2; shift 2 ;;
    --radix) need "$@"; RADIX=$2; shift 2 ;;
    --ready-s) need "$@"; READY_S=$2; shift 2 ;;
    --mem-fraction) need "$@"; MEMFRAC=$2; shift 2 ;;
    --extra) need "$@"; EXTRA=$2; shift 2 ;;
    --purge) PURGE=1; shift ;;
    --replace) REPLACE=1; shift ;;
    --print-cmd) PRINT_CMD=1; shift ;;
    -h|--help) usage ;;
    *) echo "serve_arm: unknown argument: $1" >&2; usage ;;
  esac
done

SERVING=$EVAL_ROOT/serving.json

gpu_used_mib_max() { # the largest memory.used over all GPUs; empty when there is no nvidia-smi
  command -v "$NVIDIA_SMI" >/dev/null 2>&1 || return 0
  "$NVIDIA_SMI" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -n | tail -n1
}

# ---------------------------------------------------------------------------------------------------------------
if [ "$MODE" = status ]; then
  [ -f "$SERVING" ] || { echo "no server recorded in $SERVING"; exit 1; }
  cat "$SERVING"
  port=$(pyjson "$SERVING" port)
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null || true)
  echo "health on :$port = ${code:-none}"
  [ "$code" = 200 ]
  exit $?
fi

if [ "$MODE" = down ]; then
  if [ -f "$SERVING" ]; then
    SGLANG_PID=$(pyjson "$SERVING" pid 2>/dev/null || true)
    ARM=$(pyjson "$SERVING" arm 2>/dev/null || true)
  fi
  step "stopping SGLang${ARM:+ (arm $ARM)}"
  stop_sglang || die "an SGLang process would not stop"
  used=$(gpu_used_mib_max)
  if [ -n "$used" ] && [ "$used" -ge 2048 ]; then
    die "a GPU still holds ${used} MiB after stopping SGLang (something else is using it)"
  fi
  rm -f "$SERVING"
  if [ "$PURGE" = 1 ] && [ -n "$ARM" ] && [ -d "$MERGED_ROOT/$ARM" ]; then
    step "removing the merged model $MERGED_ROOT/$ARM"
    rm -rf "${MERGED_ROOT:?}/$ARM"
  fi
  step "stopped, GPUs free${used:+ (max used ${used} MiB)}"
  exit 0
fi

[ "$MODE" = up ] || usage

# ---------------------------------------------------------------------------------------------------------------
# validate
[ -n "$ARM" ] || die "--arm NAME is required"
[[ "$ARM" =~ ^[ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-]+$ ]] || die "arm name '$ARM' must be [A-Za-z0-9_.-]+ (it is a directory name)"
case "$KIND" in base|ctrl|wide) ;; *) die "--kind must be base, ctrl or wide (got '$KIND')" ;; esac
case "$RADIX" in on|off) ;; *) die "--radix must be on or off" ;; esac
for v in DP TP PORT CTX READY_S; do [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "$v='${!v}' is not a positive integer"; done
[[ "$MEMFRAC" =~ ^(0?\.[0-9]+|1(\.0+)?)$ ]] || die "--mem-fraction '$MEMFRAC' is not a fraction in (0, 1]"
[ "$GDN_LAYERS" -ge "$TP" ] && [ $((GDN_LAYERS % TP)) -eq 0 ] || die "--tp $TP does not divide $GDN_LAYERS GDN layers"
if [ "$KIND" = base ]; then
  [ -z "$TRAINABLE" ] || die "kind base takes no --trainable"
else
  [ -n "$TRAINABLE" ] || die "kind $KIND needs --trainable FILE"
  [ -f "$TRAINABLE" ] || die "trainable $TRAINABLE not found"
fi
[ -d "$MODEL_DIR" ] || die "model dir $MODEL_DIR not found"
if [ -z "${ALLOW_NO_REV:-}" ]; then
  [ -f "$MODEL_DIR/.negeig_revision" ] || die "$MODEL_DIR/.negeig_revision missing: the model was not staged by bootstrap.sh"
  grep -q "$MODEL_REV" "$MODEL_DIR/.negeig_revision" || die "$MODEL_DIR is not revision $MODEL_REV"
fi
[ -f "$ARM_KIND_PY" ] || die "$ARM_KIND_PY missing"
if [ "$KIND" != base ]; then [ -f "$NEGEIG_CLI" ] || die "$NEGEIG_CLI missing"; fi
[ -x "$VPY" ] || die "SGLang python $VPY not found or not executable (bootstrap.sh --stage verl)"

ARM_DIR=${ARM_DIR:-$EVAL_ROOT/$ARM}
MERGED=$MERGED_ROOT/$ARM
SLOG=$ARM_DIR/server.log
GATES=$ARM_DIR/gates.safetensors
RANKS=$((DP * TP))
LAYERS_PER_RANK=$((GDN_LAYERS / TP))
SERVE_MODEL=$MODEL_DIR
[ "$KIND" = base ] || SERVE_MODEL=$MERGED

CMD=("$VPY" -m sglang.launch_server --model-path "$SERVE_MODEL" --served-model-name "$SERVED_NAME"
     --tp-size "$TP" --dp-size "$DP" --host 127.0.0.1 --port "$PORT" --dtype bfloat16
     --mem-fraction-static "$MEMFRAC" --context-length "$CTX" --mamba-full-memory-ratio "$MAMBA_RATIO"
     --linear-attn-backend triton --linear-attn-prefill-backend triton --linear-attn-decode-backend triton
     --trust-remote-code)
[ "$RADIX" = off ] && CMD+=(--disable-radix-cache)
[ -z "$ATTN_BACKEND" ] || CMD+=(--attention-backend "$ATTN_BACKEND")
# shellcheck disable=SC2206  # EXTRA is a word list on purpose
[ -z "$EXTRA" ] || CMD+=($EXTRA)

if [ "$PRINT_CMD" = 1 ]; then
  echo "arm=$ARM kind=$KIND ranks=$RANKS arm_dir=$ARM_DIR"
  [ "$KIND" = base ] || echo "merge: $VPY $NEGEIG_CLI merge-lora --base $MODEL_DIR --trainable $TRAINABLE --out $MERGED --scale $MERGE_SCALE --rounding stochastic --seed $MERGE_SEED --min-delta-kept $MIN_DELTA_KEPT"
  [ "$KIND" != wide ] || echo "gates: $VPY $NEGEIG_CLI export-gates --trainable $TRAINABLE --out $GATES --config $MODEL_DIR/config.json"
  if [ "$KIND" = wide ]; then echo "env: PYTHONPATH=$PLUGIN_DIR SGLANG_PLUGINS=negeig NEGEIG_GATES=$GATES"; else echo "env: plugin not loaded (SGLANG_PLUGINS unset)"; fi
  echo "launch: ${CMD[*]}"
  exit 0
fi

CH=$(find_cuda_home) || die "no CUDA toolkit with nvcc (the JIT kernels need one); set NEGEIG_CUDA_HOME"

# one server at a time: a second arm on the same GPUs would OOM or, worse, answer from the wrong model
if pgrep -f '[s]glang\.launch_server|^[s]glang::' >/dev/null 2>&1; then
  [ "$REPLACE" = 1 ] || die "an SGLang server is already running; serve_arm.sh down first (or --replace)"
  step "--replace: stopping the running server"
  stop_sglang || die "the running SGLang server would not stop"
fi
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then die "port $PORT is already in use"; fi
mkdir -p "$ARM_DIR" "$EVAL_ROOT" "$MERGED_ROOT"
rm -f "$ARM_DIR/arm.json" "$ARM_DIR/serve.json" "$ARM_DIR/engagement.json" "$SERVING"
: >"$SLOG"
fail_stop() { echo "serve_arm: ERROR: $*" >&2; stop_sglang >/dev/null 2>&1; rm -f "$SERVING"; exit 1; }

# ---------------------------------------------------------------------------------------------------------------
# the trainable decides the kind, and the declared kind has to agree
GATES_SHA=""
if [ "$KIND" != base ]; then
  step "classify $TRAINABLE"
  kind_json=$("$VPY" "$ARM_KIND_PY" kind --trainable "$TRAINABLE") || die "the trainable is neither a wide nor a ctrl arm: $kind_json"
  detected=$("$VPY" -c 'import json,sys; print(json.loads(sys.argv[1])["kind"])' "$kind_json")
  [ "$detected" = "$KIND" ] || die "declared kind $KIND but the trainable is $detected: $kind_json"
  echo "$kind_json" >"$ARM_DIR/trainable.kind.json"

  # merge the LoRA into a copy of the base. Reuse a finished merge of this very file; anything else is rebuilt.
  sha=$(sha256_of "$TRAINABLE")
  if [ -d "$MERGED" ]; then
    if [ -f "$MERGED/MERGE-RECEIPT.json" ] && [ "$(pyjson "$MERGED/MERGE-RECEIPT.json" trainable_sha256 2>/dev/null)" = "$sha" ] \
       && "$VPY" "$ARM_KIND_PY" receipt --receipt "$MERGED/MERGE-RECEIPT.json" --min-delta-kept "$MIN_DELTA_KEPT" >/dev/null; then
      step "reusing the finished merge $MERGED (same trainable sha256 ${sha:0:12})"
    else
      step "$MERGED exists without a valid receipt for this trainable: removing it"
      rm -rf "${MERGED:?}"
    fi
  fi
  if [ ! -d "$MERGED" ]; then
    step "merge-lora $TRAINABLE into $MERGED"
    "$VPY" "$NEGEIG_CLI" merge-lora --base "$MODEL_DIR" --trainable "$TRAINABLE" --out "$MERGED" --scale "$MERGE_SCALE" \
      --rounding stochastic --seed "$MERGE_SEED" --min-delta-kept "$MIN_DELTA_KEPT" >"$ARM_DIR/merge.log" 2>&1 \
      || { tail -n 20 "$ARM_DIR/merge.log" >&2; die "merge-lora failed (it removes its output when delta_kept is too low)"; }
  fi
  "$VPY" "$ARM_KIND_PY" receipt --receipt "$MERGED/MERGE-RECEIPT.json" --min-delta-kept "$MIN_DELTA_KEPT" >"$ARM_DIR/merge.check.json" \
    || { cat "$ARM_DIR/merge.check.json" >&2; die "merge receipt check failed"; }
  cp "$MERGED/MERGE-RECEIPT.json" "$ARM_DIR/MERGE-RECEIPT.json"

  if [ "$KIND" = wide ]; then
    step "export-gates"
    "$VPY" "$NEGEIG_CLI" export-gates --trainable "$TRAINABLE" --out "$GATES" --config "$MODEL_DIR/config.json" \
      >"$ARM_DIR/gates.export.json" 2>"$ARM_DIR/gates.export.err" \
      || { cat "$ARM_DIR/gates.export.err" >&2; die "export-gates failed"; }
    GATES_SHA=$(sha256_of "$GATES")
    step "gates sha256 ${GATES_SHA:0:16}"
    step "install-plugin and static-check"
    "$VPY" "$NEGEIG_CLI" install-plugin --dir "$PLUGIN_DIR" >"$ARM_DIR/plugin.install.json" || die "install-plugin failed"
    "$VPY" "$NEGEIG_CLI" static-check >"$ARM_DIR/static_check.json" 2>"$ARM_DIR/static_check.err" \
      || { cat "$ARM_DIR/static_check.err" >&2; die "static-check failed: this SGLang is not the one the plugin was written for"; }
  else
    # a ctrl arm has no gate; the check that the file really carries none is that export-gates refuses it
    step "ctrl: export-gates must refuse the file"
    if "$VPY" "$NEGEIG_CLI" export-gates --trainable "$TRAINABLE" --out "$ARM_DIR/ctrl_gates_must_not_exist.safetensors" \
         >/dev/null 2>"$ARM_DIR/ctrl.export.err"; then
      rm -f "$ARM_DIR/ctrl_gates_must_not_exist.safetensors"
      die "ctrl: export-gates accepted $TRAINABLE, it carries gate tensors; this is not a ctrl arm"
    fi
    grep -q "no negeig_w tensors" "$ARM_DIR/ctrl.export.err" || { cat "$ARM_DIR/ctrl.export.err" >&2; die "ctrl: export-gates failed for another reason than 'no negeig_w tensors'"; }
  fi
fi

# ---------------------------------------------------------------------------------------------------------------
# launch. A non-gated arm gets no plugin variable at all: the server must be the stock program.
step "launching SGLang for arm $ARM ($KIND): dp $DP x tp $TP, context $CTX, radix $RADIX"
echo "launch: ${CMD[*]}" >"$ARM_DIR/launch.cmd"
# env takes its options (-u) before the NAME=VALUE words; after them -u would be read as the command to run
if [ "$KIND" = wide ]; then
  PRE=(env "PATH=${VPY%/*}:$CH/bin:$PATH" "CUDA_HOME=$CH" "SGLANG_ENABLE_JIT_DEEPGEMM=0"
       "PYTHONPATH=$PLUGIN_DIR${PYTHONPATH:+:$PYTHONPATH}" "SGLANG_PLUGINS=negeig" "NEGEIG_GATES=$GATES")
else
  PRE=(env -u SGLANG_PLUGINS -u NEGEIG_GATES "PATH=${VPY%/*}:$CH/bin:$PATH" "CUDA_HOME=$CH" "SGLANG_ENABLE_JIT_DEEPGEMM=0")
fi
setsid "${PRE[@]}" "${CMD[@]}" >"$SLOG" 2>&1 </dev/null &
SGLANG_PID=$!
UTC=$(date -u +%FT%TZ)
poll=${SGLANG_POLL_S:-10}
tries=$(((READY_S + poll - 1) / poll))
wait_sglang_healthy "$PORT" "$SGLANG_PID" "$SLOG" "$tries" || fail_stop "SGLang did not become healthy within ${READY_S} s (log $SLOG)"

# the liveness probe: one question with a checkable answer, think-off, temperature 0 (same as box/accept.sh)
step "liveness probe"
resp=$(curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$SERVED_NAME"'", "max_tokens": 64, "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "What is 17 times 23? Answer with the number only."}]}') || true
echo "$resp" >"$ARM_DIR/probe.json"
answer=$("$VPY" -c 'import json,sys
try: print(json.loads(sys.argv[1])["choices"][0]["message"]["content"] or "")
except Exception: print("")' "$resp")
echo "probe answer: $answer"
echo "$answer" | grep -q '391' || fail_stop "the probe did not answer 391 (got: ${answer:0:80}); see $ARM_DIR/probe.json"

# ---------------------------------------------------------------------------------------------------------------
# what was served, recorded in the shape compare_general.py compares between arms
step "write serve.json and check engagement"
GPU_NAME=unknown; GPU_COUNT=0
if command -v "$NVIDIA_SMI" >/dev/null 2>&1; then
  GPU_NAME=$("$NVIDIA_SMI" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | sed 's/^ *//;s/ *$//')
  GPU_COUNT=$("$NVIDIA_SMI" --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
fi
SGLANG_VERSION=$("$VPY" -c 'import sglang; print(sglang.__version__)' 2>/dev/null || echo unknown)
PLUGIN_SHA=""
[ ! -f "$NEGEIG_CLI" ] || PLUGIN_SHA=$(sha256_of "$NEGEIG_CLI")
SERVE_CTX=$CTX SERVE_RADIX=$RADIX SERVE_TP=$TP SERVE_DP=$DP SERVE_PORT=$PORT SERVE_MEM=$MEMFRAC \
SERVE_ATTN=${ATTN_BACKEND:-default} SERVE_VER=$SGLANG_VERSION SERVE_REV=$MODEL_REV SERVE_GPU=$GPU_NAME \
SERVE_NGPU=$GPU_COUNT SERVE_MODEL=$SERVE_MODEL SERVE_UTC=$UTC SERVE_CMD="${CMD[*]}" SERVE_PLUGIN_SHA=$PLUGIN_SHA \
SERVE_GATES_SHA=$GATES_SHA SERVE_KIND=$KIND SERVE_EXTRA=$EXTRA \
"$VPY" - "$ARM_DIR/serve.json" <<'PYEOF'
import json, os, sys
e = os.environ
doc = {
    "context_length": int(e["SERVE_CTX"]), "radix": e["SERVE_RADIX"], "tp_size": int(e["SERVE_TP"]),
    "attn_backend": e["SERVE_ATTN"], "linear_attn_backend": "triton", "linear_prefill_backend": "triton",
    "linear_decode_backend": "triton", "sglang_version": e["SERVE_VER"], "model_rev": e["SERVE_REV"],
    "gpu_name": e["SERVE_GPU"], "dtype": "bfloat16",
    "dp_size": int(e["SERVE_DP"]), "port": int(e["SERVE_PORT"]), "mem_fraction": float(e["SERVE_MEM"]),
    "gpu_count": int(e["SERVE_NGPU"]), "model_dir": e["SERVE_MODEL"], "kind": e["SERVE_KIND"],
    "plugin_sha256": e["SERVE_PLUGIN_SHA"] or None, "gates_sha256": e["SERVE_GATES_SHA"] or None,
    "extra_args": e["SERVE_EXTRA"], "utc": e["SERVE_UTC"], "cmdline": e["SERVE_CMD"],
}
json.dump(doc, open(sys.argv[1], "w"), indent=1)
open(sys.argv[1], "a").write("\n")
PYEOF

ENG_ARGS=(--log "$SLOG" --kind "$KIND" --ranks "$RANKS" --layers "$LAYERS_PER_RANK")
[ -z "$GATES_SHA" ] || ENG_ARGS+=(--gates-sha256 "$GATES_SHA")
"$VPY" "$ARM_KIND_PY" engagement "${ENG_ARGS[@]}" >"$ARM_DIR/engagement.json"
ENG_RC=$?
ASM=(--arm "$ARM" --kind "$KIND" --serve "$ARM_DIR/serve.json" --engagement "$ARM_DIR/engagement.json" --out "$ARM_DIR/arm.json")
[ "$KIND" = base ] || ASM+=(--merge-receipt "$ARM_DIR/MERGE-RECEIPT.json")
"$VPY" "$ARM_KIND_PY" assemble "${ASM[@]}" || fail_stop "could not write arm.json"
if [ "$ENG_RC" -ne 0 ]; then
  cat "$ARM_DIR/engagement.json" >&2
  fail_stop "engagement check failed for $ARM ($KIND): the served program is not the arm (arm.json records it)"
fi

"$VPY" -c 'import json,sys; json.dump({"arm":sys.argv[1],"kind":sys.argv[2],"pid":int(sys.argv[3]),"port":int(sys.argv[4]),"dir":sys.argv[5],"utc":sys.argv[6]}, open(sys.argv[7],"w"), indent=1)' \
  "$ARM" "$KIND" "$SGLANG_PID" "$PORT" "$ARM_DIR" "$UTC" "$SERVING"
step "arm $ARM ($KIND) is up on http://127.0.0.1:$PORT, engagement ok ($(pyjson "$ARM_DIR/engagement.json" active_lines) active lines, $(pyjson "$ARM_DIR/engagement.json" negeig_lines) negeig lines), arm dir $ARM_DIR"
exit 0
