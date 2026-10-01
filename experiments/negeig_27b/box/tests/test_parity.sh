#!/usr/bin/env bash
# CPU tests of parity.sh: every box command is stubbed (fake SGLang server, fake check_parity.py, fake negeig_sglang.py,
# stub nvidia-smi and python shim from lib.sh). They prove the WIRING of the G0..G3 run, not any number:
#   the three arms are launched with identical flags and different plugin environments, the stock arm is stock even when the
#   caller's shell has the plugin set, eager runs before graphs and a failure stops the run, the HF job runs on another GPU
#   in the background, a rerun keeps what exists, --target-t reruns only the rand pieces, and the exit codes and the last
#   line (PARITY_OK / WAIVED / FAILED / INCONCLUSIVE / BROKEN) say what the run proved.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
PARITY=${PARITY_SCRIPT:-$BOX_DIR/parity.sh}
FAKE_PARITY=$TESTS_DIR/fake_check_parity.py
FAKE_NEGEIG=$TESTS_DIR/fake_negeig_sglang.py
HF_PROC="fake_[c]heck_parity[.]py hf-ref"

cleanup_all() {
  [ -z "${SGLANG_PROC_PATTERN:-}" ] || pkill -f "$SGLANG_PROC_PATTERN" 2>/dev/null
  pkill -f "$HF_PROC" 2>/dev/null
  cleanup_scratch
}
trap cleanup_all EXIT

par() { "$PARITY" "$@" >"$B/par.out" 2>&1; RC=$?; }
jqr() { jq -r "$1" "$2"; }
live_servers() { pgrep -f "$SGLANG_PROC_PATTERN" | wc -l | tr -d ' '; }
live_hf() { pgrep -f "$HF_PROC" | wc -l | tr -d ' '; }
last_line() { grep -v '^$' "$B/par.out" | tail -n 1; }
arm_seq() { # the launches in order: stock:eager zero:eager rand:eager stock:graphs ...
  [ -s "$STUB_SGLANG_ARGLOG" ] || { echo ""; return; }
  jq -r '(if .plugins=="" then "stock" elif .gates=="zeros" then "zero" else "rand" end) + (if (.argv|index("--cuda-graph-config")) then ":eager" else ":graphs" end)' "$STUB_SGLANG_ARGLOG" | tr '\n' ' ' | sed 's/ $//'
}
call_count() { jq -r --arg c "$1" 'select(.ev=="call" and .cmd==$c) | .cmd' "$STUB_PARITY_LOG" | wc -l | tr -d ' '; }
call_tags() { jq -r --arg c "$1" 'select(.ev=="call" and .cmd==$c) | .argv | .[index("--tag")+1]' "$STUB_PARITY_LOG" | tr '\n' ' ' | sed 's/ $//'; }
wiring() { jq -r '.wiring_errors | length' "$1"; }

mk_p() {
  mk_accept_box "$1"
  echo '{"model_type":"qwen3_5"}' >"$B/model/config.json"
  export CHECK_PARITY=$FAKE_PARITY NEGEIG_SGLANG=$FAKE_NEGEIG STUB_PARITY_LOG=$B/calls.log STUB_SGLANG_ARGLOG=$B/args.log
  export PARITY_PORT=$PORT PARITY_POLLS=20
  unset PARITY_OUT PARITY_COMPARE_ARGS PARITY_MAX_USED_MIB SGLANG_PLUGINS NEGEIG_GATES STUB_PREPARE_RC STUB_GATES_RC \
    STUB_COLLECT_FAIL STUB_HF_FAIL STUB_HF_SLEEP STUB_COMPARE_EXIT STUB_COMPARE_EXIT_eager STUB_COMPARE_EXIT_graphs \
    STUB_COMPARE_NOWRITE STUB_STATIC_RC STUB_PLUGIN_NO_ENTRYPOINT STUB_INSTALL_RC STUB_SGLANG_NO_ENGAGE STUB_SGLANG_ALWAYS_BANNER
  : >"$STUB_PARITY_LOG"; : >"$STUB_SGLANG_ARGLOG"
  P=$B/accept/parity
}

section "P1 both modes, 8 GPUs: eager then graphs, three arms each, HF on GPU 1 in the background"
mk_p p1
STUB_HF_SLEEP=4 par
eq "exit 0" "$RC" 0
eq "six launches, eager first, stock then zero then rand" "$(arm_seq)" "stock:eager zero:eager rand:eager stock:graphs zero:graphs rand:graphs"
has "last line PARITY_OK" "$(last_line)" "^PARITY_OK [0-9TZ:-]+ modes=eager,graphs target_t=0.5 G0=PASS G1=PASS G2=PASS G3=PASS"
eq "the same line is in the marker file" "$(cat "$P/PARITY_OK")" "$(last_line)"
eq "no server left behind" "$(live_servers)" 0
eq "no HF job left behind" "$(live_hf)" 0
eq "prepare once" "$(call_count prepare)" 1
eq "random gates once" "$(call_count make-random-gates)" 1
eq "collects: stock twice, zero, rand, per mode" "$(call_tags collect)" "stock stock-repeat zero rand stock stock-repeat zero rand"
eq "two HF references, once" "$(call_count hf-ref)" 2
eq "compare once per mode" "$(call_count compare)" 2
eq "eager verdict: wiring errors" "$(wiring "$P/eager/verdict_t0.5.json")" 0
eq "graphs verdict: wiring errors" "$(wiring "$P/graphs/verdict_t0.5.json")" 0
eq "HF ran on GPU 1" "$(jq -r 'select(.ev=="hf-start") | .cuda_visible' "$STUB_PARITY_LOG" | sort -u | tr '\n' ' ')" "1 "
eq "every server on GPU 0" "$(jq -r '.cuda_visible' "$STUB_SGLANG_ARGLOG" | sort -u | tr '\n' ' ')" "0 "
first_hf=$(jq -r 'select(.ev=="hf-start") | .t' "$STUB_PARITY_LOG" | head -n 1)
first_collect=$(jq -r 'select(.ev=="call" and .cmd=="collect") | .t' "$STUB_PARITY_LOG" | head -n 1)
first_hf_end=$(jq -r 'select(.ev=="hf-end") | .t' "$STUB_PARITY_LOG" | head -n 1)
check "HF job started before the first collect" python3 -c "import sys; sys.exit(0 if float('$first_hf') < float('$first_collect') else 1)"
check "the first HF reference was still running when the first collect ran (background, not serial)" python3 -c "import sys; sys.exit(0 if float('$first_hf_end') > float('$first_collect') else 1)"
check "compare ran after both HF references finished" python3 -c "
import json, sys
ev = [json.loads(l) for l in open('$STUB_PARITY_LOG') if l.strip()]
ends = [e['t'] for e in ev if e['ev'] == 'hf-end']
cmp_ = [e['t'] for e in ev if e['ev'] == 'compare']
sys.exit(0 if len(ends) == 2 and cmp_ and min(cmp_) > max(ends) else 1)"
for f in launch_stock_eager.txt launch_zero_eager.txt launch_rand_eager.txt launch_stock_graphs.txt prompts.json hf_stock.json hf_rand_t0.5.json \
  rand_gates_t0.5.safetensors parity.json eager/compare_t0.5.txt graphs/compare_t0.5.txt plugin/negeig_sglang.py; do
  check "receipt $f" test -s "$P/$f"
done
has "launch receipt of the stock arm unsets the plugin env" "$P/launch_stock_eager.txt" "^env: -u SGLANG_PLUGINS -u NEGEIG_GATES "
has "launch receipt of the zero arm names the plugin and zeros" "$P/launch_zero_eager.txt" "SGLANG_PLUGINS=negeig NEGEIG_GATES=zeros"
has "launch receipt of the rand arm names the gates file" "$P/launch_rand_graphs.txt" "NEGEIG_GATES=$P/rand_gates_t0.5.safetensors"
has "eager launch receipt carries the graph config" "$P/launch_stock_eager.txt" "cuda-graph-config"
hasnt "graphs launch receipt does not" "$P/launch_stock_graphs.txt" "cuda-graph-config"
eq "parity.json exit" "$(jqr .exit "$P/parity.json")" 0
eq "parity.json modes" "$(jqr '.modes_run | join(",")' "$P/parity.json")" "eager,graphs"
has "the banner of the plugin engagement is shown" "$B/par.out" "negeig: active v1: 48 gated GDN layers"

section "P2 a rerun keeps everything that exists and launches nothing"
: >"$STUB_SGLANG_ARGLOG"; : >"$STUB_PARITY_LOG"
par
eq "exit 0" "$RC" 0
eq "no server launched" "$(arm_seq)" ""
eq "no collect, no hf-ref, no prepare" "$(call_count collect)$(call_count hf-ref)$(call_count prepare)$(call_count make-random-gates)" 0000
eq "compare ran again for both modes" "$(call_count compare)" 2
has "says what it kept" "$B/par.out" "stock \(eager\): kept from an earlier run"

section "P3 --target-t 0.15 reruns only the random-gate pieces, keyed by target-t"
: >"$STUB_SGLANG_ARGLOG"; : >"$STUB_PARITY_LOG"
par --target-t 0.15
eq "exit 0" "$RC" 0
eq "only rand arms launched, eager then graphs" "$(arm_seq)" "rand:eager rand:graphs"
eq "new gates made, prompts not" "$(call_count make-random-gates)$(call_count prepare)" 10
eq "one new HF reference (hf-rand), hf-stock kept" "$(jq -r 'select(.ev=="call" and .cmd=="hf-ref") | .argv | .[index("--tag")+1]' "$STUB_PARITY_LOG" | tr '\n' ' ')" "hf-rand "
check "gates file for 0.15 exists" test -s "$P/rand_gates_t0.15.safetensors"
check "the 0.5 gates are still there" test -s "$P/rand_gates_t0.5.safetensors"
check "verdict_t0.15 in both modes" test -s "$P/graphs/verdict_t0.15.json"
has "PARITY_OK names target_t 0.15" "$(last_line)" "target_t=0.15"
eq "rand arm used the 0.15 gates" "$(jq -r 'select(.plugins=="negeig") | select(.gates!="zeros") | .gates' "$STUB_SGLANG_ARGLOG" | sort -u)" "$P/rand_gates_t0.15.safetensors"
eq "wiring errors" "$(wiring "$P/eager/verdict_t0.15.json")$(wiring "$P/graphs/verdict_t0.15.json")" 00

section "P4 --fresh clears the receipts and redoes everything"
: >"$STUB_SGLANG_ARGLOG"; : >"$STUB_PARITY_LOG"
par --fresh --mode eager
eq "exit 0" "$RC" 0
eq "three launches" "$(arm_seq)" "stock:eager zero:eager rand:eager"
eq "prepare ran again" "$(call_count prepare)" 1
check "graphs receipts from the earlier run are gone" test ! -e "$P/graphs"
has "modes=eager only" "$(last_line)" "modes=eager "

section "P5 --mode graphs alone"
mk_p p5
par --mode graphs
eq "exit 0" "$RC" 0
eq "three graphs launches" "$(arm_seq)" "stock:graphs zero:graphs rand:graphs"
hasnt "no cuda-graph-config on any launch" "$STUB_SGLANG_ARGLOG" "cuda-graph-config"

section "P6 one GPU: the HF references run after the servers, on GPU 0"
mk_p p6
STUB_GPUS=1 par --mode eager
eq "exit 0" "$RC" 0
has "says the HF runs after the servers" "$B/par.out" "one GPU on this box: the HF references run after the servers"
eq "HF on GPU 0" "$(jq -r 'select(.ev=="hf-start") | .cuda_visible' "$STUB_PARITY_LOG" | sort -u | tr '\n' ' ')" "0 "
eq "no HF reference while a server was up" "$(jq -r 'select(.ev=="hf-start") | .server_up' "$STUB_PARITY_LOG" | sort -u | tr '\n' ' ')" "false "
eq "three launches" "$(arm_seq)" "stock:eager zero:eager rand:eager"

section "P7 the caller's shell has the plugin set: the stock arm is still stock"
mk_p p7
SGLANG_PLUGINS=negeig NEGEIG_GATES=zeros par --mode eager
eq "exit 0" "$RC" 0
eq "stock arm: no plugin env" "$(jq -r 'select(.plugins=="") | .gates + "|" + .plugins' "$STUB_SGLANG_ARGLOG" | tr '\n' ' ')" "| "
has "no negeig line from the stock server log" "$(grep -c 'negeig:' "$P/eager/stock.log" || true)" "^0$"
eq "wiring errors" "$(wiring "$P/eager/verdict_t0.5.json")" 0

section "P8 compare exit 1 in eager: graphs is not run, no marker, PARITY_FAILED"
mk_p p8
STUB_COMPARE_EXIT_eager=1 par
eq "exit 1" "$RC" 1
eq "only eager launches" "$(arm_seq)" "stock:eager zero:eager rand:eager"
has "PARITY_FAILED line" "$(last_line)" "^PARITY_FAILED modes=eager .*G2=FAIL"
check_not "no PARITY_OK marker" test -e "$P/PARITY_OK"
has "tells why graphs was not run" "$B/par.out" "mode eager did not pass: the later mode is not run"
eq "parity.json exit 1" "$(jqr .exit "$P/parity.json")" 1
eq "no server left behind" "$(live_servers)" 0

section "P9 eager passes, graphs fails: PARITY_FAILED naming both modes, no marker"
mk_p p9
STUB_COMPARE_EXIT_graphs=1 par
eq "exit 1" "$RC" 1
eq "six launches" "$(arm_seq)" "stock:eager zero:eager rand:eager stock:graphs zero:graphs rand:graphs"
has "PARITY_FAILED modes=eager,graphs" "$(last_line)" "^PARITY_FAILED modes=eager,graphs"
check_not "no PARITY_OK marker" test -e "$P/PARITY_OK"
check "eager verdict file kept" test -s "$P/eager/verdict_t0.5.json"

section "P10 compare exit 2: PARITY_INCONCLUSIVE, exit 2, no marker"
mk_p p10
STUB_COMPARE_EXIT=2 par --mode eager
eq "exit 2" "$RC" 2
has "PARITY_INCONCLUSIVE" "$(last_line)" "^PARITY_INCONCLUSIVE modes=eager .*G2=INCONCLUSIVE"
check_not "no marker" test -e "$P/PARITY_OK"

section "P11 --skip: PARITY_WAIVED, no PARITY_OK marker, the skip reaches compare"
mk_p p11
par --mode eager --skip G2,G3
eq "exit 0" "$RC" 0
has "PARITY_WAIVED line names the waived gates" "$(last_line)" "^PARITY_WAIVED .* waived=G2,G3 "
check_not "no PARITY_OK marker for a waived run" test -e "$P/PARITY_OK"
eq "compare got --skip G2,G3" "$(jq -r 'select(.ev=="call" and .cmd=="compare") | .argv | .[index("--skip")+1]' "$STUB_PARITY_LOG")" "G2,G3"
eq "parity.json records the skip" "$(jqr .skip "$P/parity.json")" "G2,G3"

section "P12 PARITY_COMPARE_ARGS reach compare"
mk_p p12
PARITY_COMPARE_ARGS="--gap-ratio 3 --min-n 50" par --mode eager
eq "exit 0" "$RC" 0
eq "compare argv has the thresholds" "$(jq -r 'select(.ev=="call" and .cmd=="compare") | .argv | join(" ")' "$STUB_PARITY_LOG" | grep -c -- '--gap-ratio 3 --min-n 50')" 1

section "P13 a stale PARITY_OK from an earlier run is removed when the new run fails"
mk_p p13
par --mode eager
eq "first run passes" "$RC" 0
check "marker present" test -s "$P/PARITY_OK"
STUB_COMPARE_EXIT=1 par --mode eager
eq "second run fails" "$RC" 1
check_not "the marker is gone" test -e "$P/PARITY_OK"
eq "parity.json was rewritten by the failing run" "$(jqr .exit "$P/parity.json")" 1

section "P14 the inputs fail loudly before any server starts"
mk_p p14a
STUB_STATIC_RC=1 par
eq "static-check fails: exit 3" "$RC" 3
has "PARITY_BROKEN names the step" "$(last_line)" "^PARITY_BROKEN step=static_check"
eq "no server launched" "$(arm_seq)" ""
mk_p p14b
STUB_INSTALL_RC=1 par
eq "install-plugin fails: exit 3" "$RC" 3
has "step install_plugin" "$(last_line)" "^PARITY_BROKEN step=install_plugin"
mk_p p14c
STUB_PLUGIN_NO_ENTRYPOINT=1 par
eq "plugin not discoverable by entry point: exit 3" "$RC" 3
has "step plugin_discovery" "$(last_line)" "^PARITY_BROKEN step=plugin_discovery"
eq "no server launched" "$(arm_seq)" ""
mk_p p14d
STUB_PREPARE_RC=1 par
eq "prepare fails: exit 3" "$RC" 3
has "step prepare" "$(last_line)" "^PARITY_BROKEN step=prepare"
check_not "a failed prepare leaves no prompts file under its final name" test -s "$P/prompts.json"
mk_p p14e
STUB_GATES_RC=1 par
eq "make-random-gates fails: exit 3" "$RC" 3
has "step make_random_gates" "$(last_line)" "^PARITY_BROKEN step=make_random_gates"
check_not "no gates file under the final name" test -e "$P/rand_gates_t0.5.safetensors"

section "P15 the plugin did not engage: the run stops at that server"
mk_p p15
STUB_SGLANG_NO_ENGAGE=1 par --mode eager
eq "exit 3" "$RC" 3
has "step serve_zero" "$(last_line)" "^PARITY_BROKEN step=serve_zero: no 'negeig: active' line"
eq "stock launched, then zero, then stopped" "$(arm_seq)" "stock:eager zero:eager"
eq "no server left behind" "$(live_servers)" 0
eq "no compare" "$(call_count compare)" 0

section "P15b the stock arm shows the plugin engaged: the run stops at the stock server"
mk_p p15b
STUB_SGLANG_ALWAYS_BANNER=1 par --mode eager
eq "exit 3" "$RC" 3
has "step serve_stock" "$(last_line)" "^PARITY_BROKEN step=serve_stock: the stock server log has a negeig: line"
eq "only stock launched" "$(arm_seq)" "stock:eager"
eq "no server left behind" "$(live_servers)" 0
eq "no compare" "$(call_count compare)" 0
eq "no collect" "$(call_count collect)" 0

section "P16 the server crashes at start"
mk_p p16
STUB_SGLANG_CRASH=1 par --mode eager
eq "exit 3" "$RC" 3
has "step serve_stock" "$(last_line)" "^PARITY_BROKEN step=serve_stock"
has "log tail shown" "$B/par.out" "planned crash"

section "P17 a collect fails: the half-written output is not kept, the server is stopped"
mk_p p17
STUB_COLLECT_FAIL=zero par --mode eager
eq "exit 3" "$RC" 3
has "step collect_zero" "$(last_line)" "^PARITY_BROKEN step=collect_zero"
check_not "no zero.json" test -e "$P/eager/zero.json"
check_not "no zero.json.tmp" test -e "$P/eager/zero.json.tmp"
check "stock outputs kept" test -s "$P/eager/stock2.json"
eq "no server left behind" "$(live_servers)" 0
: >"$STUB_SGLANG_ARGLOG"
STUB_COLLECT_FAIL="" par --mode eager
eq "the rerun finishes from where it stopped: exit 0" "$RC" 0
eq "the rerun launched zero and rand only" "$(arm_seq)" "zero:eager rand:eager"

section "P18 an HF reference fails: PARITY_BROKEN step=hf, nothing is compared, servers are gone"
mk_p p18
STUB_HF_FAIL=hf-rand par --mode eager
eq "exit 3" "$RC" 3
has "step hf" "$(last_line)" "^PARITY_BROKEN step=hf"
eq "compare never ran" "$(call_count compare)" 0
check_not "no hf_rand.json under its final name" test -e "$P/hf_rand_t0.5.json"
check "hf_stock.json kept" test -s "$P/hf_stock.json"
eq "no HF job left behind" "$(live_hf)" 0
section "P18b --hf-gpu 0 puts the HF job on the server's GPU: the stub refuses it and the run reports it"
mk_p p18b
STUB_HF_SLEEP=3 par --mode eager --hf-gpu 0
eq "exit 3 (the shared GPU is an error, not a silent slowdown)" "$RC" 3
has "step hf" "$(last_line)" "^PARITY_BROKEN step=hf"

section "P19 refusals before launch"
mk_p p19a
STUB_UNIT_ACTIVE=1 par
eq "unit active: exit 1" "$RC" 1
has "says which unit" "$B/par.out" "negeig27-train.service is active"
eq "nothing launched" "$(arm_seq)" ""
mk_p p19b
STUB_GPU_USED_IDX="2:9000" par
eq "a GPU in use: exit 1" "$RC" 1
has "says how much" "$B/par.out" "a GPU holds 9000 MiB"
mk_p p19c
rm -f "$B/model/config.json"
par
eq "no model: exit 1" "$RC" 1
has "says so" "$B/par.out" "no model at"
mk_p p19d
rm -f "$B/root/verl/.venv/bin/python3"
par
eq "no verl venv: exit 1" "$RC" 1
has "says so" "$B/par.out" "no verl venv"
mk_p p19e
CHECK_PARITY=$B/nope.py par
eq "no check_parity.py: exit 1" "$RC" 1
mk_p p19f
if compgen -G "/usr/local/cuda*/bin/nvcc" >/dev/null; then
  ok "no-nvcc case skipped: this host has a CUDA toolkit under /usr/local, find_cuda_home would find it"
else
  rm -f "$B/cuda/bin/nvcc"
  par
  eq "no nvcc: exit 1" "$RC" 1
  has "says so" "$B/par.out" "no nvcc"
fi
mk_p p19g
( env -u STUB_SGLANG_ARGLOG python3 "$TESTS_DIR/fake_sglang_server.py" --port "$PARITY_PORT" >/dev/null 2>&1 & echo $! >"$B/squatter.pid" )
sleep 1
par
eq "something already answers on the port: exit 1" "$RC" 1
has "says so" "$B/par.out" "already answers /health on port"
kill "$(cat "$B/squatter.pid")" 2>/dev/null
eq "nothing launched by the run" "$(arm_seq)" ""
mk_p p19h
STUB_GPUS=2 par --hf-gpu 5
eq "--hf-gpu beyond the GPU count: exit 1" "$RC" 1
has "says so" "$B/par.out" "but the box has 2 GPU"

section "P20 bad arguments"
mk_p p20
par --mode sideways;     eq "bad --mode: exit 1" "$RC" 1
par --target-t abc;      eq "bad --target-t: exit 1" "$RC" 1
par --skip G7;           eq "bad --skip: exit 1" "$RC" 1
par --hf-gpu x;          eq "bad --hf-gpu: exit 1" "$RC" 1
par --bogus;             eq "unknown argument: exit 1" "$RC" 1
par --help;              eq "--help: exit 0" "$RC" 0
has "--help prints the gate list" "$B/par.out" "G0 the stock server repeats itself"
eq "no launches from bad arguments" "$(arm_seq)" ""

section "P21 SIGTERM mid-run: no server and no HF job survive"
mk_p p21
STUB_HF_SLEEP=60 setsid "$PARITY" --mode eager >"$B/par.out" 2>&1 &
PP=$!
for _ in $(seq 1 40); do [ -s "$STUB_SGLANG_ARGLOG" ] && break; sleep 0.5; done
check "the first server was launched" test -s "$STUB_SGLANG_ARGLOG"
sleep 1
eq "an HF job is running before the TERM" "$(live_hf)" 1
kill -TERM "$PP"
wait "$PP"; RC=$?
eq "exit 143" "$RC" 143
sleep 1
eq "no server left behind" "$(live_servers)" 0
eq "no HF job left behind" "$(live_hf)" 0
check_not "no PARITY_OK marker" test -e "$P/PARITY_OK"

section "P22 a failing PARITY_BROKEN never writes PARITY_OK, and compare writing no verdict is reported"
mk_p p22
STUB_COMPARE_NOWRITE=1 par --mode eager
eq "exit 3" "$RC" 3
has "step compare" "$(last_line)" "^PARITY_BROKEN step=compare: check_parity compare wrote no verdict"

finish
