#!/usr/bin/env bash
# CPU tests of the self-distillation tooling: sd_box.sh (serve, chat, tools, status, finalize, lengths, stop), sd_lengths.py
# and the rig-side relay sd_sync.sh. Nothing reaches a GPU, a real SGLang server or a real box:
#   - `serve` launches fake_sglang_server.py through the fake verl venv of mk_accept_box
#   - the generators (chat_gen.py, tool_gen.py: the real ones) talk to selfdistill/sd_fake_server.py
#   - sd_lengths.py renders with train27's own encode_chat and a fake `transformers` (tests/fake_transformers)
#   - sd_sync.sh runs against ssh and rsync stubs that treat a directory per IP as the box
# The generator tests need the Hebrew lane (agentic_env.py and friends); they are skipped, loudly, when it is absent
# (override the location with NEGEIG_TEST_LANE).
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
SDBOX=$BOX_DIR/sd_box.sh
SDSYNC=$BOX_DIR/sd_sync.sh
LENGTHS=$BOX_DIR/sd_lengths.py
SELFDISTILL=$E27_DIR/selfdistill
LANE_SRC=${NEGEIG_TEST_LANE:-${SELFDISTILL_LANE_DIR:-}}
export PYTHONDONTWRITEBYTECODE=1
GEN_PID=""; SLEEPERS=()
cleanup_all() {
  [ -z "$GEN_PID" ] || kill "$GEN_PID" 2>/dev/null
  local p; for p in ${SLEEPERS[@]+"${SLEEPERS[@]}"}; do kill -TERM -- "-$p" 2>/dev/null; kill "$p" 2>/dev/null; done
  [ -z "${SGLANG_PROC_PATTERN:-}" ] || pkill -f "$SGLANG_PROC_PATTERN" 2>/dev/null
  cleanup_scratch
}
trap cleanup_all EXIT

sdb() { "$SDBOX" "$@" >"$B/sd.out" 2>&1; RC=$?; }
nrows() { if [ -s "$1" ]; then wc -l <"$1" | tr -d ' '; else echo 0; fi; }
jqr() { jq -r "$1" "$2"; }
free_port() { echo $((32000 + RANDOM % 2000)); }
live_procs() { pgrep -f "$SGLANG_PROC_PATTERN" | wc -l | tr -d ' '; }

# mk_sd_box NAME: a stubbed accept box plus a model dir, the Hebrew lane (a symlink, read only), the chat prompts, the
# generation config. SD_PORT is a fresh port; SD_POLLS is small so a failing serve ends in seconds.
mk_sd_box() {
  mk_accept_box "$1"
  echo '{"model_type":"qwen3_5"}' >"$B/model/config.json"
  echo '{"temperature":0.7,"top_p":0.8,"top_k":20}' >"$B/model/generation_config.json"
  mkdir -p "$B/root/lane/research"
  [ ! -d "$LANE_SRC" ] || ln -s "$LANE_SRC" "$B/root/lane/research/hebrew-rl-20260923"
  ( cd "$SELFDISTILL" && python3 -c "
import json, run
rows = run.dry_run_prompts()
open('$B/data/s4/chat_prompts.jsonl', 'w').write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
" )
  export SD_PORT; SD_PORT=$(free_port)
  export SD_POLLS=6 SD_OUT=$B/selfdistill SD_CONCURRENCY=8
  unset SD_FG SD_DP SD_CONTEXT SD_MAX_USED_MIB
}

mkrows() { # kind n words [start] -> JSONL on stdout; kind chat|tool
  python3 - "$@" <<'PYEOF'
import json, sys
kind, n, words = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
body = " ".join(["w"] * words)
for i in range(start, start + n):
    if kind == "chat":
        row = {"id": f"c{i}", "messages": [{"role": "user", "content": f"q{i} " + body}, {"role": "assistant", "content": f"a{i} " + body}]}
    else:
        row = {"id": f"t{i}", "messages": [
            {"role": "system", "content": "sys"}, {"role": "user", "content": f"q{i} " + body},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "f", "arguments": {"a": i}}]},
            {"role": "tool", "content": "ok"}, {"role": "assistant", "content": f"done {i}"}],
            "tools": [{"type": "function", "function": {"name": "f"}}]}
    print(json.dumps(row))
PYEOF
}

# ---------------------------------------------------------------------------------------------------------- serve / stop
section "S1 serve: flags, the probe, idempotence, status, stop"
mk_sd_box s1
export STUB_SGLANG_ARGLOG=$B/arglog.jsonl STUB_SGLANG_REQLOG=$B/reqlog.txt
sdb serve --dp 4
eq "serve exit 0" "$RC" 0
has "SD_SERVE_OK names port, dp and pid" "$B/sd.out" "^SD_SERVE_OK port=$SD_PORT dp=4 pid=[0-9]+$"
ARGV=$(jqr '.argv | join(" ")' "$B/arglog.jsonl")
has "model path is the untouched model dir" "$ARGV" "--model-path $B/model "
has "tp 1, dp 4" "$ARGV" "--tp-size 1 --dp-size 4"
has "tool-call parser qwen3_coder" "$ARGV" "--tool-call-parser qwen3_coder"
has "reasoning parser qwen3" "$ARGV" "--reasoning-parser qwen3"
has "bound to loopback on the port" "$ARGV" "--host 127.0.0.1 --port $SD_PORT"
has "context 32768, mamba ratio 3, mem 0.85" "$ARGV" "--mem-fraction-static 0.85 --context-length 32768 --mamba-full-memory-ratio 3"
hasnt "no speculative or lora flags" "$ARGV" "speculative|lora|--quantization"
eq "CUDA_HOME is the toolkit dir" "$(jqr .cuda_home "$B/arglog.jsonl")" "$B/cuda"
eq "the verl venv bin leads PATH (ninja for the flashinfer JIT)" "$(jqr '.path[0]' "$B/arglog.jsonl")" "$B/root/verl/.venv/bin"
has "the probe asks with thinking off" "$B/reqlog.txt" '"enable_thinking": *false'
check "probe receipt written" test -s "$B/accept/sd_serve_probe.json"
check "pidfile written" test -s "$B/run/sd_sglang.pid"
sdb status
has "status: server healthy" "$B/sd.out" "^server  : healthy"
sdb serve --dp 4
eq "second serve exit 0" "$RC" 0
has "second serve says already serving" "$B/sd.out" "already serving"
eq "no second launch" "$(wc -l <"$B/arglog.jsonl" | tr -d ' ')" 1
STUB_GPU_USED_MIB=5000 sdb stop
eq "stop with a GPU still holding memory exits 1" "$RC" 1
has "stop names the memory" "$B/sd.out" "still holds 5000 MiB"
eq "the server is gone anyway" "$(live_procs)" 0
sdb stop
eq "stop exit 0" "$RC" 0
has "SD_STOP_OK" "$B/sd.out" "^SD_STOP_OK every GPU freed$"
check_not "pidfile removed" test -e "$B/run/sd_sglang.pid"
sdb status
has "status: server down" "$B/sd.out" "^server  : down"
has "status: nothing started" "$B/sd.out" "^chat    : not started"

section "S2 serve refusals and failures"
mk_sd_box s2
rm "$B/model/config.json"
sdb serve
eq "no model: exit 1" "$RC" 1
has "no model: says so" "$B/sd.out" "no model at"
echo '{}' >"$B/model/config.json"
ROOT=$B/noroot sdb serve
eq "no verl venv: exit 1" "$RC" 1
has "no verl venv: names it" "$B/sd.out" "no verl venv"
STUB_UNIT_ACTIVE=1 sdb serve
eq "training unit active: exit 1" "$RC" 1
has "training unit active: says hold" "$B/sd.out" "negeig27-train.service is active: ctl.sh hold"
STUB_GPU_USED_IDX="3:9000" sdb serve
eq "a busy GPU: exit 1" "$RC" 1
has "a busy GPU: names the memory" "$B/sd.out" "a GPU holds 9000 MiB"
eq "nothing launched on a refusal" "$(live_procs)" 0
sdb serve --bogus
eq "unknown serve argument: exit 1" "$RC" 1
sdb bogus
eq "unknown command: exit 1" "$RC" 1
sdb
eq "no command: exit 2" "$RC" 2
has "no command prints the usage header" "$B/sd.out" "sd_box.sh serve"
sdb --help
eq "--help exit 0" "$RC" 0
if ls /usr/local/cuda*/bin/nvcc >/dev/null 2>&1; then echo "  skip no-cuda case (this host has a toolkit under /usr/local/cuda*)"
else
  NEGEIG_CUDA_HOME=$B/nocuda sdb serve
  eq "no nvcc: exit 1" "$RC" 1
  has "no nvcc: says flashinfer" "$B/sd.out" "no nvcc"
fi
sleep 300 & SLEEPERS+=($!)
echo $! >"$B/run/sd_sglang.pid"
sdb serve
eq "a live but unhealthy server: exit 1" "$RC" 1
has "says stop --force first" "$B/sd.out" "up but not healthy.*stop --force"
kill "$!" 2>/dev/null; rm -f "$B/run/sd_sglang.pid"

section "S3 serve ends cleanly when the server dies, answers wrong, or never comes up"
mk_sd_box s3
STUB_SGLANG_CRASH=1 sdb serve
eq "crash at launch: exit 1" "$RC" 1
has "crash: sglang exited early" "$B/sd.out" "sglang exited early"
check_not "crash: pidfile removed" test -e "$B/run/sd_sglang.pid"
STUB_SGLANG_ANSWER=392 sdb serve
eq "wrong probe answer: exit 1" "$RC" 1
has "wrong answer: SD_SERVE_FAILED" "$B/sd.out" "SD_SERVE_FAILED probe: expected 391"
eq "wrong answer: the server was stopped" "$(live_procs)" 0
check_not "wrong answer: pidfile removed" test -e "$B/run/sd_sglang.pid"
SD_POLLS=2 STUB_SGLANG_DELAY=60 sdb serve
eq "never healthy: exit 1" "$RC" 1
has "never healthy: says so" "$B/sd.out" "sglang not healthy after 2 s"
eq "never healthy: the server was stopped" "$(live_procs)" 0

section "S4 stop while a generator runs"
mk_sd_box s4
sdb serve
eq "serve ok" "$RC" 0
setsid sleep 300 & SLEEPERS+=($!)
echo $! >"$B/run/sd_chat.pid"
sdb status
has "status: chat running" "$B/sd.out" "^chat    : running"
sdb stop
eq "stop is refused while chat runs" "$RC" 1
has "refusal names stop --force" "$B/sd.out" "chat is running.*stop --force"
eq "the server is still up after a refusal" "$(live_procs)" 1
sdb stop --force
eq "stop --force exit 0" "$RC" 0
check_not "the generator group was signalled" kill -0 "$(cat "$B/run/sd_chat.pid")"
eq "no server left" "$(live_procs)" 0

# ---------------------------------------------------------------------------------------------------------- generators
if [ ! -f "$LANE_SRC/agentic_env.py" ]; then
  echo "  SKIP generator, finalize and chain tests: no lane at $LANE_SRC (set NEGEIG_TEST_LANE)"
  GEN_OK=0
else
  GEN_OK=1
fi
if [ "$GEN_OK" = 1 ]; then
  section "G1 chat and tools against the fake generator server"
  mk_sd_box g1
  GENPORT=$(free_port); [ "$GENPORT" != "$SD_PORT" ] || GENPORT=$((GENPORT + 1))
  ( cd "$SELFDISTILL" && SELFDISTILL_LANE_DIR=$LANE_SRC exec nice -n 10 taskset -c 0-3 python3 sd_fake_server.py --port "$GENPORT" --with-pool ) >"$B/gen_server.log" 2>&1 &
  GEN_PID=$!
  for _ in $(seq 1 40); do [ "$(curl -s -o /dev/null -m 2 -w '%{http_code}' "http://127.0.0.1:$GENPORT/health")" = 200 ] && break; sleep 0.5; done
  eq "fake generator server up" "$(curl -s -o /dev/null -m 2 -w '%{http_code}' "http://127.0.0.1:$GENPORT/health")" 200
  export SD_PORT=$GENPORT

  SD_OUT=$B/sdA sdb status
  has "status before any run: no chat ledger" "$B/sd.out" "rows    : chat_raw.jsonl +0"
  SD_PORT=$(free_port) sdb chat --limit 5
  eq "chat with no server: exit 1" "$RC" 1
  has "chat with no server: says serve first" "$B/sd.out" "no healthy server on port.*serve first"
  ROOT=$B/noroot SD_OUT=$B/sdA sdb chat --limit 5
  eq "chat with no lane: exit 1" "$RC" 1
  has "chat with no lane: names push_data --hebrew-lane" "$B/sd.out" "no lane at .*--hebrew-lane"
  mv "$B/data/s4/chat_prompts.jsonl" "$B/prompts.hold"
  SD_OUT=$B/sdA sdb chat --limit 5
  eq "chat with no prompts: exit 1" "$RC" 1
  has "chat with no prompts: names the file" "$B/sd.out" "no chat prompts at"
  mv "$B/prompts.hold" "$B/data/s4/chat_prompts.jsonl"

  SD_OUT=$B/sdA SD_FG=1 sdb chat --n-translate 8 --limit 20
  eq "chat (foreground) exit 0" "$RC" 0
  has "chat prints its exit line" "$B/sd.out" "^SD_CHAT_EXIT 0$"
  eq "chat.exit holds 0" "$(cat "$B/sdA/chat.exit")" 0
  check "chat_sft.jsonl written" test -s "$B/sdA/chat_sft.jsonl"
  check "chat_raw.jsonl ledger written" test -s "$B/sdA/chat_raw.jsonl"
  check "translations ledger written" test -s "$B/sdA/translations.jsonl"
  check "spot_read written" test -s "$B/sdA/spot_read.jsonl"
  check "filter_report written" test -s "$B/sdA/filter_report.json"
  check_not "a chat run writes no tool_sft.jsonl" test -e "$B/sdA/tool_sft.jsonl"
  ARGVS=$(jqr '.argv | join(" ")' "$B/sdA/manifest_chat.json")
  has "manifest argv: local base url" "$ARGVS" "--base-url http://127.0.0.1:$GENPORT"
  has "manifest argv: this box's out dir" "$ARGVS" "--out-dir $B/sdA"
  has "manifest argv: the checkpoint's generation config" "$ARGVS" "--generation-config $B/model/generation_config.json"
  has "manifest argv: the box prompts file" "$ARGVS" "--prompts $B/data/s4/chat_prompts.jsonl"
  has "manifest argv: concurrency from SD_CONCURRENCY" "$ARGVS" "--concurrency 8"
  eq "manifest: think off" "$(jqr .think "$B/sdA/manifest_chat.json")" off
  eq "manifest: sampling temperature comes from the checkpoint's generation config" "$(jqr .sampling.temperature "$B/sdA/manifest_chat.json")" 0.7
  eq "manifest: sampling top_k comes from the checkpoint's generation config" "$(jqr .sampling.top_k "$B/sdA/manifest_chat.json")" 20
  eq "every chat_sft row names the model in meta" "$(jq -r '.meta.model' "$B/sdA/chat_sft.jsonl" | sort -u)" "fake-qwen3.8-27b"
  eq "every chat_sft row carries messages" "$(jq -c 'select((.messages | type) == "array" and (.messages | length) > 0)' "$B/sdA/chat_sft.jsonl" | wc -l | tr -d ' ')" "$(nrows "$B/sdA/chat_sft.jsonl")"

  SD_OUT=$B/sdB SD_FG=1 sdb tools --item-limit-per-kind 2 --n-samples 2
  eq "tools (foreground) exit 0" "$RC" 0
  has "tools prints its exit line" "$B/sd.out" "^SD_TOOLS_EXIT 0$"
  check "tool_sft.jsonl written" test -s "$B/sdB/tool_sft.jsonl"
  check "tool_raw.jsonl ledger written" test -s "$B/sdB/tool_raw.jsonl"
  check "manifest_tools.json written" test -s "$B/sdB/manifest_tools.json"
  check_not "a tools run writes no chat_sft.jsonl" test -e "$B/sdB/chat_sft.jsonl"
  eq "every tool_sft row carries tools" "$(jq -c 'select((.tools | type) == "array" and (.tools | length) > 0)' "$B/sdB/tool_sft.jsonl" | wc -l | tr -d ' ')" "$(nrows "$B/sdB/tool_sft.jsonl")"
  has "tools argv: the lane dir is the box's" "$(jqr '.lane | keys | join(",")' "$B/sdB/manifest_tools.json")" "agentic_env.py"

  G1=$B
  section "G2 detached run, resume, and the exit-status contract"
  SD_OUT=$B/sdC sdb chat --n-translate 4 --limit 12
  eq "detached chat starts: exit 0" "$RC" 0
  has "SD_CHAT_STARTED prints the pid" "$B/sd.out" "^SD_CHAT_STARTED pid=[0-9]+$"
  for _ in $(seq 1 60); do [ -f "$B/sdC/chat.exit" ] && break; sleep 0.5; done
  eq "chat.exit appears and holds 0" "$(cat "$B/sdC/chat.exit" 2>/dev/null)" 0
  SD_OUT=$B/sdC sdb status
  has "status: chat done" "$B/sd.out" "^chat    : done"
  has "status: ledger rows are counted" "$B/sd.out" "rows    : chat_raw.jsonl +[1-9]"
  has "status: chat log tail" "$B/sd.out" "tail of $B/logs/sd_chat.log"
  ROWS_BEFORE=$(nrows "$B/sdC/chat_raw.jsonl")
  SD_OUT=$B/sdC SD_FG=1 sdb chat --n-translate 4 --limit 12
  eq "rerun (resume) exit 0" "$RC" 0
  eq "a resumed run adds no rows" "$(nrows "$B/sdC/chat_raw.jsonl")" "$ROWS_BEFORE"
  sleep 300 & SLEEPERS+=($!)
  echo $! >"$B/run/sd_tools.pid"
  SD_OUT=$B/sdC sdb tools --item-limit-per-kind 1
  eq "a second tools start is refused while the first runs" "$RC" 1
  has "refusal names the pid" "$B/sd.out" "tools is already running \(pid $!\)"
  kill "$!" 2>/dev/null; rm -f "$B/run/sd_tools.pid"
  # a generator that cannot work: the answering server has no /v1/models, so the run ends non-zero and says so
  mk_sd_box g2
  sdb serve
  eq "serve (fake sglang) ok" "$RC" 0
  SD_FG=1 sdb chat --limit 3
  NZ=$RC
  check "a generator that fails exits non-zero" test "$NZ" -ne 0
  eq "chat.exit records that status" "$(cat "$B/selfdistill/chat.exit")" "$NZ"
  sdb status
  has "status shows the failed generator" "$B/sd.out" "^chat    : (aborted|crashed)"
  sdb stop
  eq "stop ok" "$RC" 0

  section "F1 finalize"
  mk_sd_box f1
  sdb finalize
  eq "finalize with no ledgers: exit 1" "$RC" 1
  has "no ledgers: says nothing was generated" "$B/sd.out" "no ledgers in $B/selfdistill"
  cp -r "$G1/sdA" "$B/sdA"; cp -r "$G1/sdB" "$B/sdB"
  SD_OUT=$B/sdA sdb finalize
  eq "chat half: exit 0" "$RC" 0
  has "chat half: SD_FINALIZE_OK counts both files" "$B/sd.out" "^SD_FINALIZE_OK chat_sft=$(nrows "$B/sdA/chat_sft.jsonl") tool_sft=0 out=$B/sdA$"
  SD_OUT=$B/sdB sdb finalize
  eq "tool half: exit 0" "$RC" 0
  has "tool half: SD_FINALIZE_OK" "$B/sd.out" "^SD_FINALIZE_OK chat_sft=0 tool_sft=$(nrows "$B/sdB/tool_sft.jsonl") out=$B/sdB$"
  cp -r "$B/sdA" "$B/sdD"; rm "$B/sdD/chat_sft.jsonl"
  SD_OUT=$B/sdD sdb finalize
  eq "ledger without its sft file: exit 1" "$RC" 1
  has "names the missing file and --refilter" "$B/sd.out" "chat_raw.jsonl has rows but chat_sft.jsonl is empty or missing.*--refilter"
  SD_OUT=$B/sdD sdb finalize --refilter
  eq "--refilter rebuilds it: exit 0" "$RC" 0
  has "--refilter prints the meta NOTE" "$B/sd.out" "NOTE: the refilter rewrote the rows without meta.model and meta.sampling"
  check "refilter log written" test -s "$B/logs/sd_refilter.log"
  eq "the refiltered file has the same row count as the generator's" "$(nrows "$B/sdD/chat_sft.jsonl")" "$(nrows "$B/sdA/chat_sft.jsonl")"
  cp -r "$B/sdA" "$B/sdE"; echo 2 >"$B/sdE/chat.exit"
  SD_OUT=$B/sdE sdb finalize
  eq "an aborted generator: exit 1" "$RC" 1
  has "an aborted generator: not finished" "$B/sd.out" "chat did not finish: aborted"
  echo 7 >"$B/sdE/chat.exit"
  SD_OUT=$B/sdE sdb finalize
  has "a crashed generator: names the status" "$B/sd.out" "chat did not finish: crashed \(exit 7"
  SD_OUT=$B/sdE sdb status
  has "status shows the abort" "$B/sd.out" "^chat    : crashed \(exit 7"
  echo 2 >"$B/sdE/chat.exit"
  SD_OUT=$B/sdE sdb status
  has "status shows 'rerun to resume'" "$B/sd.out" "^chat    : aborted \(rerun to resume\)"
fi

# ---------------------------------------------------------------------------------------------------------- sd_lengths
section "L1 lengths: a clean pair of files"
mk_sd_box l1
mkrows chat 10 10 >"$B/data/sd/chat_sft.jsonl"
mkrows tool 20 10 >"$B/data/sd/tool_sft.jsonl"
PYTHONPATH=$TESTS_DIR/fake_transformers sdb lengths
eq "lengths exit 0" "$RC" 0
has "chat line" "$B/sd.out" "^SD_LENGTHS chat: rows=10 usable=10 over_max_len=0 no_labels=0 render_errors=0 "
has "tool line" "$B/sd.out" "^SD_LENGTHS tool: rows=20 usable=20 over_max_len=0 no_labels=0 render_errors=0 "
has "final line" "$B/sd.out" "^SD_LENGTHS_OK tool_rows_over_max_len=0 tool_usable=20/20 tool_train=16 chat_usable=10/10$"
hasnt "no warning" "$B/sd.out" "WARNING"
R=$B/accept/sd_lengths.json
eq "receipt ok" "$(jqr .ok "$R")" true
eq "receipt max_len is the trainer's 12288" "$(jqr .max_len "$R")" 12288
eq "tool rows held out for the agreement eval: min(64, usable//5)" "$(jqr .tool_heldout_for_eval "$R")" 4
eq "tool rows left to train on" "$(jqr .tool_train_rows "$R")" 16
check "no scratch temp file left" test ! -e "$B/accept/sd_lengths.tmp"

section "L2 lengths: rows over --max_len are counted and warned about"
mkrows tool 4 400 100 >>"$B/data/sd/tool_sft.jsonl"
PYTHONPATH=$TESTS_DIR/fake_transformers sdb lengths --max_len 300
eq "over-length rows are a warning, not a failure: exit 0" "$RC" 0
has "tool line counts the 4 rows train27 would drop" "$B/sd.out" "^SD_LENGTHS tool: rows=24 usable=20 over_max_len=4 .*\(max_len 300\)"
has "warning names the loss" "$B/sd.out" "^WARNING: tool replay loses 16.7% of its rows"
hasnt "the chat file loses nothing, no chat warning" "$B/sd.out" "WARNING: chat"
has "final line carries the number the owner asked for" "$B/sd.out" "^SD_LENGTHS_OK tool_rows_over_max_len=4 tool_usable=20/24 tool_train=16"
eq "receipt tool_rows_over_max_len" "$(jqr .tool_rows_over_max_len "$R")" 4
eq "receipt warn flag" "$(jqr .pools.tool.warn "$R")" true
eq "receipt over_len_min is the shortest dropped row" "$(jqr '.pools.tool.over_len_min > 300' "$R")" true
PYTHONPATH=$TESTS_DIR/fake_transformers sdb lengths --max_len 300 --warn-frac 0.5
hasnt "--warn-frac 0.5 silences it" "$B/sd.out" "WARNING"

section "L3 lengths: a row that does not render fails the check (train27 would die at start)"
mk_sd_box l3
mkrows chat 5 10 >"$B/data/sd/chat_sft.jsonl"
{ mkrows tool 6 10; echo '{"id":"bad","messages":[{"content":"no role"}],"tools":[]}'; } >"$B/data/sd/tool_sft.jsonl"
PYTHONPATH=$TESTS_DIR/fake_transformers sdb lengths
eq "render error: exit 1" "$RC" 1
has "PROBLEM names the file and the count" "$B/sd.out" "PROBLEM: tool_sft.jsonl: 1 row\(s\) do not render, train27 would die at start"
has "SD_LENGTHS_FAIL" "$B/sd.out" "^SD_LENGTHS_FAIL "
eq "receipt ok false" "$(jqr .ok "$B/accept/sd_lengths.json")" false
eq "receipt names the bad row" "$(jqr '.pools.tool.first_render_errors[0].id' "$B/accept/sd_lengths.json")" bad

section "L4 lengths: unlabelled rows, junk, empties, nothing usable, no tokenizer"
mk_sd_box l4
mkrows chat 5 10 >"$B/data/sd/chat_sft.jsonl"
{ mkrows tool 6 10; echo '{"id":"nolabel","messages":[{"role":"user","content":"hi"}]}'; } >"$B/data/sd/tool_sft.jsonl"
PYTHONPATH=$TESTS_DIR/fake_transformers sdb lengths
eq "a row with no assistant turn is dropped, not fatal: exit 0" "$RC" 0
has "counted as no_labels" "$B/sd.out" "^SD_LENGTHS tool: rows=7 usable=6 over_max_len=0 no_labels=1 "
echo 'not json at all' >>"$B/data/sd/chat_sft.jsonl"
PYTHONPATH=$TESTS_DIR/fake_transformers sdb lengths
eq "a non-JSON line: exit 1" "$RC" 1
has "names the line" "$B/sd.out" "PROBLEM: .*chat_sft.jsonl:6: not JSON"
mkrows chat 5 10 >"$B/data/sd/chat_sft.jsonl"
PYTHONPATH=$TESTS_DIR/fake_transformers sdb lengths --max_len 5
eq "nothing usable: exit 1" "$RC" 1
has "nothing usable is a problem" "$B/sd.out" "PROBLEM: chat_sft.jsonl: nothing usable \(5 over 5, 0 unlabelled\)"
: >"$B/data/sd/tool_sft.jsonl"
PYTHONPATH=$TESTS_DIR/fake_transformers sdb lengths
eq "an empty file: refused by sd_box before python runs, exit 1" "$RC" 1
has "says to run sd_sync.sh on the rig" "$B/sd.out" "run sd_sync.sh on the rig first \(this box has chat 5, tool 0 rows\)"
PYTHONPATH=$TESTS_DIR/fake_transformers python3 "$LENGTHS" --dir "$B/data/sd" --model-dir "$B/model" >"$B/py.out" 2>&1
eq "sd_lengths.py on an empty file: exit 1" "$?" 1
has "sd_lengths.py names the missing or empty file" "$B/py.out" "PROBLEM: tool_sft.jsonl: missing or empty"
mkrows tool 3 10 >"$B/data/sd/tool_sft.jsonl"
python3 "$LENGTHS" --dir "$B/data/sd" --model-dir "$B/no-such-model" >"$B/py.out" 2>&1
eq "no tokenizer: exit 2" "$?" 2
has "no tokenizer: says where it looked" "$B/py.out" "ERROR: no tokenizer at $B/no-such-model"
rm -f "$B/data/sd/chat_sft.jsonl"
PYTHONPATH=$TESTS_DIR/fake_transformers sdb lengths
eq "a missing chat file: exit 1" "$RC" 1
has "the refusal names both files" "$B/sd.out" "needs chat_sft.jsonl and tool_sft.jsonl"

# ---------------------------------------------------------------------------------------------------------- sd_sync
# The rig side. Boxes are directories under $FAKE_BOXES/<ip>/rbox (W=/rbox); the ssh and rsync stubs rewrite /rbox.
mk_rig() { # NAME -> the stub ssh and rsync on PATH, two boxes, a scratch DEST and state dir
  mk_box "$1"
  FAKE_BOXES=$B/boxes; FAKE_RIG_DIR=$B/rig; mkdir -p "$B/bin" "$FAKE_BOXES" "$FAKE_RIG_DIR"
  export FAKE_BOXES FAKE_RIG_DIR FAKE_RIG_LOG=$B/rig/calls.log NEGEIG27_STATE=$B/state NEGEIG_SD_DEST=$B/dest
  : >"$FAKE_RIG_LOG"
  export PATH="$B/bin:$PATH"
  unset STUB_SSH_DOWN BWLIMIT_KBPS
  local ip; for ip in 10.0.0.1 10.0.0.2 10.0.0.3; do mkdir -p "$FAKE_BOXES/$ip/rbox/selfdistill"; done
  cat >"$B/bin/ssh" <<'EOS'
#!/usr/bin/env bash
# ssh stub: the remote is a directory per IP under $FAKE_BOXES; the remote W is /rbox
while [ $# -gt 0 ]; do
  case "$1" in -i|-o|-p|-l|-F) shift 2 ;; -*) shift ;; *) break ;; esac
done
ip=${1#*@}; shift
echo "ssh $ip $*" >>"$FAKE_RIG_LOG"
if [ "$ip" = "${STUB_SSH_DOWN:-}" ] || [ ! -d "$FAKE_BOXES/$ip" ]; then echo "ssh: connect to host $ip port 22: Connection timed out" >&2; exit 255; fi
cmd="$*"
exec bash -c "${cmd//\/rbox/$FAKE_BOXES/$ip/rbox}"
EOS
  cat >"$B/bin/rsync" <<'EOS'
#!/usr/bin/env bash
# rsync stub: user@ip:/rbox/... paths map into $FAKE_BOXES/<ip>/rbox/...; each file lands under a temp name and is renamed.
# $FAKE_RIG_DIR/fail_dst: a call whose destination contains that text writes a partial file into the partial dir and exits 30.
# $FAKE_RIG_DIR/corrupt_dst: the same match copies the file with a byte appended and reports success.
echo "rsync $*" >>"$FAKE_RIG_LOG"
args=(); pd=""
while [ $# -gt 0 ]; do
  case "$1" in -e) shift 2 ;; --partial-dir=*) pd=${1#*=}; shift ;; -*) shift ;; *) args+=("$1"); shift ;; esac
done
n=${#args[@]}; dst=${args[$((n - 1))]}
tolocal() { local p=$1 ip
  if [[ $p == *@*:* ]]; then ip=${p#*@}; ip=${ip%%:*}
    if [ "$ip" = "${STUB_SSH_DOWN:-}" ] || [ ! -d "$FAKE_BOXES/$ip" ]; then echo "ssh: connect to host $ip: timed out" >&2; exit 255; fi
    echo "$FAKE_BOXES/$ip${p#*:}"
  else echo "$p"; fi; }
ldst=$(tolocal "$dst")
for ((i = 0; i < n - 1; i++)); do
  src=$(tolocal "${args[$i]}")
  if [ -d "$ldst" ] || [[ $dst == */ ]]; then out=$ldst/$(basename "$src"); mkdir -p "$ldst"; else out=$ldst; mkdir -p "$(dirname "$out")"; fi
  [ -f "$src" ] || { echo "rsync: link_stat \"$src\" failed: No such file or directory" >&2; exit 23; }
  if [ -s "$FAKE_RIG_DIR/fail_dst" ] && [[ $dst == *"$(cat "$FAKE_RIG_DIR/fail_dst")"* ]]; then
    mkdir -p "$(dirname "$out")/$pd"; head -c 7 "$src" >"$(dirname "$out")/$pd/$(basename "$out")"
    echo "rsync error: timeout in data send/receive (code 30)" >&2; exit 30
  fi
  tmp=$(dirname "$out")/.$(basename "$out").$$
  cp "$src" "$tmp"
  if [ -s "$FAKE_RIG_DIR/corrupt_dst" ] && [[ $dst == *"$(cat "$FAKE_RIG_DIR/corrupt_dst")"* ]]; then echo x >>"$tmp"; fi
  mv "$tmp" "$out"
done
EOS
  chmod +x "$B/bin/ssh" "$B/bin/rsync"
}
sync_run() { NEGEIG_W=/rbox "$SDSYNC" "$@" >"$B/sync.out" 2>&1; RC=$?; }
box_sd() { echo "$FAKE_BOXES/$1/rbox/data/sd/$2"; }
seed_halves() { # chat rows on .1, tool rows on .2, with their receipts and ledgers
  local a=$FAKE_BOXES/10.0.0.1/rbox/selfdistill b=$FAKE_BOXES/10.0.0.2/rbox/selfdistill
  mkrows chat "${1:-5}" 8 >"$a/chat_sft.jsonl"; mkrows tool "${2:-4}" 8 >"$b/tool_sft.jsonl"
  echo '{"kind":"chat"}' >"$a/manifest_chat.json"; echo '{"from":"chat box"}' >"$a/filter_report.json"; echo '{"s":1}' >"$a/spot_read.jsonl"
  echo '{"t":1}' >"$a/translations.jsonl"; echo '{"r":1}' >"$a/chat_raw.jsonl"
  echo '{"kind":"tools"}' >"$b/manifest_tools.json"; echo '{"from":"tool box"}' >"$b/filter_report.json"; echo '{"s":2}' >"$b/spot_read.jsonl"
  echo '{"r":2}' >"$b/tool_raw.jsonl"
}
rsync_calls() { grep -c '^rsync ' "$FAKE_RIG_LOG" || true; }

section "R1 sd_sync: usage and argument errors"
mk_rig r1
sync_run
eq "no arguments: exit 1" "$RC" 1
has "usage names both boxes" "$B/sync.out" "usage: sd_sync.sh --chat-box"
sync_run --chat-box 10.0.0.1
eq "one box only: exit 1" "$RC" 1
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2 --bwlimit fast
eq "a non-numeric --bwlimit: exit 1" "$RC" 1
has "bwlimit message" "$B/sync.out" "--bwlimit is KB/s"
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2 --bogus
eq "unknown argument: exit 1" "$RC" 1
sync_run --chat-box no-such-box --tool-box 10.0.0.2
eq "an unknown box name: exit 1" "$RC" 1
has "says the box is not in the table" "$B/sync.out" "no instance 'no-such-box'"
sync_run --help
eq "--help exit 0" "$RC" 0
has "--help prints the header" "$B/sync.out" "sd_sync.sh --chat-box"

section "R2 sd_sync: dry run moves nothing"
mk_rig r2
seed_halves
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2 --dry-run
eq "dry-run exit 0" "$RC" 0
has "dry-run says so and prints the row counts" "$B/sync.out" "chat box: chat_sft.jsonl 5 rows"
has "dry-run names the tool count" "$B/sync.out" "tool box: tool_sft.jsonl 4 rows"
eq "no rsync call" "$(rsync_calls)" 0
check_not "DEST not created" test -e "$B/dest"
check_not "box .1 has no replay" test -e "$(box_sd 10.0.0.1 chat_sft.jsonl)"

section "R3 sd_sync: pull from each half, check, push to both boxes, verify by hash"
mk_rig r3
seed_halves
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
eq "sync exit 0" "$RC" 0
has "SD_SYNC_OK with counts and box count" "$B/sync.out" "^SD_SYNC_OK chat_rows=5 tool_rows=4 boxes=2$"
for ip in 10.0.0.1 10.0.0.2; do
  eq "box $ip: chat_sft.jsonl is the chat box's file" "$(sha256sum <"$(box_sd $ip chat_sft.jsonl)")" "$(sha256sum <"$FAKE_BOXES/10.0.0.1/rbox/selfdistill/chat_sft.jsonl")"
  eq "box $ip: tool_sft.jsonl is the tool box's file" "$(sha256sum <"$(box_sd $ip tool_sft.jsonl)")" "$(sha256sum <"$FAKE_BOXES/10.0.0.2/rbox/selfdistill/tool_sft.jsonl")"
done
eq "the two filter reports did not clobber each other (chat)" "$(jqr .from "$B/dest/chat/filter_report.json")" "chat box"
eq "the two filter reports did not clobber each other (tools)" "$(jqr .from "$B/dest/tools/filter_report.json")" "tool box"
for f in chat/manifest_chat.json chat/spot_read.jsonl chat/translations.jsonl chat/chat_raw.jsonl tools/manifest_tools.json tools/spot_read.jsonl tools/tool_raw.jsonl; do
  check "receipt or ledger pulled: $f" test -s "$B/dest/$f"
done
check_not "the replay files are the only things pushed to a box (no ledgers on the other box)" test -e "$FAKE_BOXES/10.0.0.2/rbox/selfdistill/chat_raw.jsonl"
eq "sd_sync.json counts" "$(jqr '[.chat_rows, .tool_rows, (.boxes | length)] | join(",")' "$B/dest/sd_sync.json")" "5,4,2"
eq "sd_sync.json hash of the chat file" "$(jqr .chat_sha256 "$B/dest/sd_sync.json")" "$(sha256sum <"$B/dest/chat/chat_sft.jsonl" | cut -d' ' -f1)"
eq "sd_sync.json sources" "$(jqr '[.chat_from, .tools_from] | join(",")' "$B/dest/sd_sync.json")" "10.0.0.1,10.0.0.2"
PUSHES=$(grep '^rsync .*--partial-dir=.rsync-partial' "$FAKE_RIG_LOG")
eq "two pushes, each with --partial-dir (never --partial)" "$(echo "$PUSHES" | grep -c .)" 2
has "pushes are throttled (default 40000 KB/s)" "$PUSHES" "--bwlimit=40000"
hasnt "no rsync call deletes anything" "$(cat "$FAKE_RIG_LOG")" "--delete"
eq "every rsync call runs inside the ssh transport" "$(grep -c '^rsync .*-e ssh ' "$FAKE_RIG_LOG")" "$(rsync_calls)"
eq "no leftover temp file on a box" "$(find "$FAKE_BOXES" -name '.*chat_sft*' -o -name '.*tool_sft*' | wc -l | tr -d ' ')" 0
: >"$FAKE_RIG_LOG"
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2 --bwlimit 1234 --skip-ledgers --dest "$B/dest2"
eq "rerun with --bwlimit --skip-ledgers --dest: exit 0" "$RC" 0
has "the bandwidth limit reaches rsync" "$FAKE_RIG_LOG" "--bwlimit=1234"
check_not "--skip-ledgers pulls no ledger" test -e "$B/dest2/chat/chat_raw.jsonl"
check "the receipts are still pulled" test -s "$B/dest2/chat/manifest_chat.json"

section "R4 sd_sync: one box for both halves, and --also"
mk_rig r4
a=$FAKE_BOXES/10.0.0.1/rbox/selfdistill
mkrows chat 3 8 >"$a/chat_sft.jsonl"; mkrows tool 2 8 >"$a/tool_sft.jsonl"
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.1
eq "the same box twice: exit 0" "$RC" 0
has "one target" "$B/sync.out" "^SD_SYNC_OK chat_rows=3 tool_rows=2 boxes=1$"
eq "one push only" "$(grep -c -- '--partial-dir' "$FAKE_RIG_LOG")" 1
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.1 --also 10.0.0.3 --also 10.0.0.3 --also 10.0.0.1
eq "--also adds a box once: exit 0" "$RC" 0
has "two targets" "$B/sync.out" "boxes=2$"
eq "the third box got the chat file" "$(sha256sum <"$(box_sd 10.0.0.3 chat_sft.jsonl)")" "$(sha256sum <"$a/chat_sft.jsonl")"

section "R5 sd_sync: refusals before anything moves"
mk_rig r5
seed_halves
rm "$FAKE_BOXES/10.0.0.2/rbox/selfdistill/tool_sft.jsonl"
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
eq "a missing tool file: exit 1" "$RC" 1
has "names the file and the command that makes it" "$B/sync.out" "no non-empty /rbox/selfdistill/tool_sft.jsonl: sd_box.sh tools"
eq "nothing was pulled" "$(rsync_calls)" 0
: >"$FAKE_BOXES/10.0.0.2/rbox/selfdistill/tool_sft.jsonl"
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
eq "an empty tool file: exit 1" "$RC" 1
rm "$FAKE_BOXES/10.0.0.1/rbox/selfdistill/chat_sft.jsonl"
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
eq "a missing chat file: exit 1" "$RC" 1
has "names the chat command" "$B/sync.out" "sd_box.sh chat"
seed_halves
STUB_SSH_DOWN=10.0.0.3 sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2 --also 10.0.0.3
eq "a target that is down: exit 1" "$RC" 1
has "says which box cannot be reached" "$B/sync.out" "cannot reach .*@10.0.0.3 .*nothing was transferred"
eq "refused before any transfer" "$(rsync_calls)" 0
check_not "no replay landed on a reachable box either" test -e "$(box_sd 10.0.0.1 chat_sft.jsonl)"

section "R6 sd_sync: a bad file is refused on the rig, before it reaches a box"
mk_rig r6
seed_halves
echo 'this line is not json' >>"$FAKE_BOXES/10.0.0.1/rbox/selfdistill/chat_sft.jsonl"
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
eq "a non-JSON line: exit 1" "$RC" 1
has "names the file and the line" "$B/sync.out" "chat_sft.jsonl:6: not JSON"
has "says it is not a valid replay file" "$B/sync.out" "chat_sft.jsonl from 10.0.0.1 is not a valid replay file"
check_not "nothing pushed to box .1" test -e "$(box_sd 10.0.0.1 chat_sft.jsonl)"
check_not "nothing pushed to box .2" test -e "$(box_sd 10.0.0.2 tool_sft.jsonl)"
check_not "no receipt of success" test -e "$B/dest/sd_sync.json"
seed_halves
echo '{"id":"x","note":"no messages"}' >>"$FAKE_BOXES/10.0.0.2/rbox/selfdistill/tool_sft.jsonl"
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
eq "a row without messages: exit 1" "$RC" 1
has "says no messages list" "$B/sync.out" "tool_sft.jsonl:5: no messages list"

section "R7 sd_sync: a cut transfer never leaves a truncated file under its final name, and a rerun heals it"
mk_rig r7
seed_halves
echo "10.0.0.2:/rbox/data" >"$FAKE_RIG_DIR/fail_dst"
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
eq "a cut push: exit 1" "$RC" 1
has "names the box" "$B/sync.out" "push to 10.0.0.2 failed"
check_not "no final-name file on the cut box" test -e "$(box_sd 10.0.0.2 chat_sft.jsonl)"
check "the partial sits in the partial dir" test -s "$FAKE_BOXES/10.0.0.2/rbox/data/sd/.rsync-partial/chat_sft.jsonl"
check_not "no receipt after a failed run" test -e "$B/dest/sd_sync.json"
rm "$FAKE_RIG_DIR/fail_dst"
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
eq "the rerun: exit 0" "$RC" 0
has "SD_SYNC_OK" "$B/sync.out" "^SD_SYNC_OK chat_rows=5 tool_rows=4 boxes=2$"
eq "the cut box now holds the whole file" "$(sha256sum <"$(box_sd 10.0.0.2 chat_sft.jsonl)")" "$(sha256sum <"$FAKE_BOXES/10.0.0.1/rbox/selfdistill/chat_sft.jsonl")"
echo "10.0.0.1:/rbox/selfdistill/chat_sft" >"$FAKE_RIG_DIR/fail_dst"
rm "$FAKE_RIG_DIR/fail_dst"
echo "10.0.0.1:/rbox/data" >"$FAKE_RIG_DIR/corrupt_dst"
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
eq "a push that arrives altered: exit 1" "$RC" 1
has "the hash check names the box" "$B/sync.out" "10.0.0.1: sha256 of the pushed files differs from the rig's copy"
check_not "a failed run removes the previous run's receipt (no stale SD_SYNC_OK evidence)" test -e "$B/dest/sd_sync.json"
hasnt "no SD_SYNC_OK on a failed run" "$B/sync.out" "SD_SYNC_OK"

section "R8 sd_sync --push-only: a replacement box gets the rig's verified copy when the generators' box is gone"
mk_rig r8
seed_halves
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
eq "the full run first: exit 0" "$RC" 0
CHAT_SHA_R8=$(sha256sum <"$B/dest/chat/chat_sft.jsonl" | cut -d' ' -f1)
rm -rf "$FAKE_BOXES/10.0.0.1/rbox" "$FAKE_BOXES/10.0.0.2/rbox"     # both original boxes are lost with their disks
mkdir -p "$FAKE_BOXES/10.0.0.3/rbox"                                 # the replacement, empty
: >"$FAKE_RIG_LOG"
sync_run --push-only --also 10.0.0.3 --dry-run
eq "dry run: exit 0" "$RC" 0
has "dry run says what it would push" "$B/sync.out" "would push the rig's copy in .* to 10.0.0.3"
eq "dry run moves nothing" "$(rsync_calls)" 0
check_not "dry run left nothing on the replacement" test -e "$(box_sd 10.0.0.3 chat_sft.jsonl)"
sync_run --push-only --also 10.0.0.3
eq "push-only to the replacement: exit 0" "$RC" 0
has "ends SD_PUSH_OK with the counts" "$B/sync.out" "^SD_PUSH_OK chat_rows=5 tool_rows=4 boxes=1$"
hasnt "it is not announced as a full sync" "$B/sync.out" "SD_SYNC_OK"
eq "chat replay arrived byte for byte" "$(sha256sum <"$(box_sd 10.0.0.3 chat_sft.jsonl)" | cut -d' ' -f1)" "$CHAT_SHA_R8"
eq "tool replay arrived byte for byte" "$(sha256sum <"$(box_sd 10.0.0.3 tool_sft.jsonl)" | cut -d' ' -f1)" "$(sha256sum <"$B/dest/tools/tool_sft.jsonl" | cut -d' ' -f1)"
eq "nothing was pulled: the only rsync is the push" "$(rsync_calls)" 1
has "the push carries --partial-dir and the default throttle" "$(grep '^rsync ' "$FAKE_RIG_LOG")" "--partial-dir=.rsync-partial .*--bwlimit=40000"
eq "the receipt now lists the replacement beside the original boxes" "$(jqr '.boxes | join(",")' "$B/dest/sd_sync.json")" "10.0.0.1,10.0.0.2,10.0.0.3"
eq "the receipt keeps the full run's sources" "$(jqr '[.chat_from, .tools_from] | join(",")' "$B/dest/sd_sync.json")" "10.0.0.1,10.0.0.2"
eq "the receipt records the push-only time" "$(jqr '.push_only_at_utc | length' "$B/dest/sd_sync.json")" 20
sync_run --push-only --also 10.0.0.3
eq "pushing again is idempotent: exit 0" "$RC" 0
eq "and the receipt does not grow a duplicate box" "$(jqr '.boxes | length' "$B/dest/sd_sync.json")" 3

section "R8b sd_sync --push-only refuses what it cannot vouch for, before anything moves"
mk_rig r8b
seed_halves
mkdir -p "$FAKE_BOXES/10.0.0.3/rbox"
sync_run --push-only --also 10.0.0.3
eq "no rig copy at all: exit 1" "$RC" 1
has "says a full run makes the files" "$B/sync.out" "needs .*chat_sft.jsonl and .*tool_sft.jsonl .*a full sd_sync.sh run makes them"
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
eq "the full run: exit 0" "$RC" 0
rm "$B/dest/sd_sync.json"
: >"$FAKE_RIG_LOG"
sync_run --push-only --also 10.0.0.3
eq "files but no receipt: exit 1" "$RC" 1
has "says the receipt of a full run is needed" "$B/sync.out" "needs .*sd_sync.json, the receipt of a full run"
eq "refused before any transfer" "$(rsync_calls)" 0
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
echo '{"messages":[{"role":"user","content":"added after the receipt"}]}' >>"$B/dest/chat/chat_sft.jsonl"
sync_run --push-only --also 10.0.0.3
eq "a rig copy that changed since the receipt: exit 1" "$RC" 1
has "names the receipt mismatch" "$B/sync.out" "differs from its receipt"
check_not "nothing reached the replacement" test -e "$(box_sd 10.0.0.3 chat_sft.jsonl)"
sync_run --push-only
eq "no --also: exit 1" "$RC" 1
has "usage names --push-only --also" "$B/sync.out" "usage: sd_sync.sh --push-only --also"
sync_run --push-only --chat-box 10.0.0.1 --also 10.0.0.3
eq "--push-only with a source box: exit 1" "$RC" 1
has "says push-only pulls nothing" "$B/sync.out" "--push-only pulls nothing"
sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
STUB_SSH_DOWN=10.0.0.3 sync_run --push-only --also 10.0.0.3
eq "a replacement that is down: exit 1" "$RC" 1
has "says which box cannot be reached" "$B/sync.out" "cannot reach .*@10.0.0.3 .*nothing was transferred"
echo "10.0.0.3:/rbox/data" >"$FAKE_RIG_DIR/corrupt_dst"
sync_run --push-only --also 10.0.0.3
eq "a push that arrives altered: exit 1" "$RC" 1
has "the hash check names the box" "$B/sync.out" "10.0.0.3: sha256 of the pushed files differs from the rig's copy"
hasnt "no SD_PUSH_OK on a failed push" "$B/sync.out" "SD_PUSH_OK"
eq "the receipt does not list the box that failed" "$(jqr '.boxes | map(select(. == "10.0.0.3")) | length' "$B/dest/sd_sync.json")" 0
rm "$FAKE_RIG_DIR/corrupt_dst"

# ---------------------------------------------------------------------------------------------------------- the whole chain
if [ "$GEN_OK" = 1 ]; then
  section "C1 chain: generators on two boxes -> rig relay -> both boxes -> length check -> trainer files"
  mk_rig c1
  cp "$B/../g1/sdA/chat_sft.jsonl" "$B/../g1/sdA/manifest_chat.json" "$B/../g1/sdA/filter_report.json" "$B/../g1/sdA/spot_read.jsonl" "$FAKE_BOXES/10.0.0.1/rbox/selfdistill/" 2>/dev/null
  cp "$B/../g1/sdB/tool_sft.jsonl" "$B/../g1/sdB/manifest_tools.json" "$FAKE_BOXES/10.0.0.2/rbox/selfdistill/" 2>/dev/null
  if [ -s "$FAKE_BOXES/10.0.0.1/rbox/selfdistill/chat_sft.jsonl" ] && [ -s "$FAKE_BOXES/10.0.0.2/rbox/selfdistill/tool_sft.jsonl" ]; then
    sync_run --chat-box 10.0.0.1 --tool-box 10.0.0.2
    eq "the relay accepts the generators' own output: exit 0" "$RC" 0
    CR=$(nrows "$FAKE_BOXES/10.0.0.1/rbox/selfdistill/chat_sft.jsonl"); TR=$(nrows "$FAKE_BOXES/10.0.0.2/rbox/selfdistill/tool_sft.jsonl")
    has "counts match the generators'" "$B/sync.out" "^SD_SYNC_OK chat_rows=$CR tool_rows=$TR boxes=2$"
    mk_sd_box c1b
    cp "$(box_sd 10.0.0.1 chat_sft.jsonl)" "$(box_sd 10.0.0.1 tool_sft.jsonl)" "$B/data/sd/"
    PYTHONPATH=$TESTS_DIR/fake_transformers sdb lengths
    eq "the generators' rows render with train27's own encode_chat: exit 0" "$RC" 0
    has "every chat row is usable" "$B/sd.out" "^SD_LENGTHS chat: rows=$CR usable=$CR over_max_len=0 no_labels=0 render_errors=0"
    has "every tool row is usable" "$B/sd.out" "^SD_LENGTHS tool: rows=$TR usable=$TR over_max_len=0 no_labels=0 render_errors=0"
  else
    fail "C1 chain: the G1 outputs are missing"
  fi
fi

finish
