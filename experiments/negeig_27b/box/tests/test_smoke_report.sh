#!/usr/bin/env bash
# CPU tests of smoke_report.py: synthetic train27 output dirs (log.jsonl, profile, ckpt, complete.json) in, one receipt out.
# The red arms: a missing profile, mem_probe, checkpoint or complete.json, a different global batch between the 4-rank and
# the 8-rank run, a rate below the floor, a probe with checkpointing on, an unprobed pool (undecided), and a probe that ran out of memory.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
REPORT=$BOX_DIR/smoke_report.py
D=$SCRATCH/sr; mkdir -p "$D"

# gen DIR [options]: a train27.py output dir. Steady rate = RATE tok/s over the last window; step events at 1, 11, ..., last.
gen() {
  python3 - "$@" <<'PYEOF'
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--world", type=int, default=4); ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--rate", type=float, default=16000.0); ap.add_argument("--batch", type=int, default=160000)
ap.add_argument("--skew", type=int, default=0, help="added to the global batch at every step")
ap.add_argument("--peak", type=float, default=170.0)
ap.add_argument("--no-profile", action="store_true"); ap.add_argument("--no-ckpt", action="store_true")
ap.add_argument("--no-complete", action="store_true"); ap.add_argument("--torn", action="store_true")
ap.add_argument("--probe", default="", help="pool:tokens:peak:grad_ckpt,...")
ap.add_argument("--only-events", type=int, default=0, help="write only the first N step events")
a = ap.parse_args()
os.makedirs(a.dir, exist_ok=True)
ev = [{"event": "config", "world": a.world, "steps": a.steps}]
for spec in [s for s in a.probe.split(",") if s]:
    pool, tok, peak, gc = spec.split(":")
    ev.append({"event": "mem_probe", "pool": pool, "tokens": int(tok), "grad_ckpt": int(gc), "s": 1.5, "peak_gib": float(peak)})
idx = sorted(set(list(range(0, a.steps, 10)) + [a.steps - 1]))
t_extra = 0.0
steps = []
for i in idx:
    n = i + 1
    batch = a.batch + (n % 7) * 100 + a.skew
    tokens = n * batch
    t_train = tokens / a.rate
    if n == a.steps:
        t_extra = 6.0    # one eval and one save at the last step: tok_per_s counts them, train_tok_per_s does not
    elapsed = t_train + 12.0 + t_extra
    base, rem = divmod(batch, a.world)
    lpt = [base + (1 if r < rem else 0) for r in range(a.world)]
    steps.append({"event": "step", "step": n, "loss": {"s1": 1.0}, "gnorm": 1.0, "lr": 1e-4, "lr_w": 1e-4,
                  "tok_per_s": tokens / elapsed, "train_tok_per_s": tokens / t_train, "elapsed_s": elapsed,
                  "rank_compute_s": [4.0 + 0.1 * (r % 3) for r in range(a.world)], "lpt_load_tokens": lpt,
                  "peak_mem_gib": a.peak, "w_norm": 0.0})
if a.only_events:
    steps = steps[:a.only_events]
ev += steps
with open(os.path.join(a.dir, "log.jsonl"), "w") as f:
    for e in ev:
        f.write(json.dumps(e) + "\n")
    if a.torn:
        f.write('{"event": "step", "step": 99, "loss"')
if not a.no_profile:
    json.dump({"device_share": {"gdn": 0.31, "attention": 0.12, "gemm": 0.40}}, open(os.path.join(a.dir, "profile_rank0.json"), "w"))
if not a.no_ckpt:
    open(os.path.join(a.dir, "ckpt_%06d.pt" % a.steps), "w").write("x")
if not a.no_complete:
    json.dump({"steps": a.steps}, open(os.path.join(a.dir, "complete.json"), "w"))
PYEOF
}
probe_ok="s1:4096:120:0,chat:6000:150:0,tool:12288:190:0"
rep() { python3 "$REPORT" "$@" >"$D/out.txt" 2>&1; RC=$?; }
jqr() { jq -r "$1" "$D/rep.json"; }
envv() { sed -n "s/^$1=//p" "$D/env.txt"; }
fresh() { rm -rf "$D"; mkdir -p "$D"; }

section "M1 healthy runs: rates, scaling, class, decision off, env file"
fresh
gen "$D/w4" --rate 16000 --probe "s1:4096:90:1,chat:6000:92:1,tool:12288:95:1"
gen "$D/c4" --rate 15000 --probe "s1:4096:90:1"
gen "$D/e8" --world 8 --steps 20 --rate 28800 --no-profile --no-ckpt
gen "$D/p" --world 4 --steps 1 --probe "$probe_ok"
rep --four wide="$D/w4" --four ctrl="$D/c4" --eight "$D/e8" --probe "$D/p" --out "$D/rep.json" --env-out "$D/env.txt" --stage replay=present
eq "exit 0" "$RC" 0
eq "no problems" "$(jqr '.problems | length')" 0
eq "wide steady rate over the last window" "$(jqr '.four_rank.wide.steady_tok_per_s')" 16000.0
eq "ctrl steady rate" "$(jqr '.four_rank.ctrl.steady_tok_per_s')" 15000.0
eq "per GPU = rate / world" "$(jqr '.four_rank.wide.steady_tok_per_s_per_gpu')" 4000.0
eq "the window is steps 21 to 30" "$(jqr '.four_rank.wide.window_steps | join(",")')" 21,30
eq "8-rank window is steps 11 to 20" "$(jqr '.eight_rank.window_steps | join(",")')" 11,20
eq "speedup 8 vs 4 of the first arm" "$(jqr '.scaling.speedup_8_vs_4')" 1.8
eq "same global batch" "$(jqr '.scaling.same_global_batch')" true
has "layout hint at 1.8x favors the 4-box layout" "$(jqr '.scaling.layout_hint')" "4-box"
eq "profile shares reported" "$(jqr '.four_rank.wide.profile_device_share.gdn')" 0.31
eq "profile other = the rest" "$(jqr '.four_rank.wide.profile_device_share.other')" 0.17
eq "the rank compute spread is over 1" "$(jqr '.four_rank.wide.rank_compute_spread_max_over_mean > 1')" true
eq "throughput class at 4000 per GPU" "$(jqr '.throughput.plan_class_b300_thresholds')" "approve the envelope"
eq "decision off at a 190 GiB peak" "$(jqr '.grad_ckpt.decision')" off
eq "probe peak is the largest pool" "$(jqr '.grad_ckpt.probe_peak_gib')" 190.0
eq "the grad_ckpt-on peaks are kept" "$(jqr '.grad_ckpt.grad_ckpt_on_peaks_gib.wide.tool')" 95.0
eq "SMOKE_TOK_PER_S is the slowest arm" "$(envv SMOKE_TOK_PER_S)" 15000.0
eq "SMOKE_TOK_PER_S_PER_GPU" "$(envv SMOKE_TOK_PER_S_PER_GPU)" 3750.0
eq "SMOKE_PEAK_MEM_GIB" "$(envv SMOKE_PEAK_MEM_GIB)" 170.0
eq "SMOKE8_SPEEDUP" "$(envv SMOKE8_SPEEDUP)" 1.8
eq "SMOKE8_TOK_PER_S" "$(envv SMOKE8_TOK_PER_S)" 28800.0
eq "GRAD_CKPT_DECISION" "$(envv GRAD_CKPT_DECISION)" off
has "the summary prints the decision" "$D/out.txt" "GRAD_CKPT_DECISION=off"
hasnt "no SMOKE PROBLEM lines" "$D/out.txt" "SMOKE PROBLEM"

section "M2 throughput classes and the speedup hint"
fresh
gen "$D/w4" --rate 8000 --probe "s1:4096:90:1"
gen "$D/e8" --world 8 --steps 20 --rate 12800 --no-profile --no-ckpt
rep --four wide="$D/w4" --eight "$D/e8" --out "$D/rep.json" --env-out "$D/env.txt"
eq "exit 0" "$RC" 0
eq "2000 per GPU is core only" "$(jqr '.throughput.plan_class_b300_thresholds')" "core only"
eq "1.6x is between the lines" "$(jqr '.scaling.speedup_8_vs_4')" 1.6
has "the owner decides in the gap" "$(jqr '.scaling.layout_hint')" "owner decides"
fresh
gen "$D/w4" --rate 4000 --probe "s1:4096:90:1"
gen "$D/e8" --world 8 --steps 20 --rate 5200 --no-profile --no-ckpt
rep --four wide="$D/w4" --eight "$D/e8" --out "$D/rep.json"
eq "1000 per GPU reports to the owner" "$(jqr '.throughput.plan_class_b300_thresholds')" "report to the owner"
has "1.3x favors two boxes" "$(jqr '.scaling.layout_hint')" "two boxes"

section "M3 red arms: a missing receipt is a problem, exit 1"
fresh
gen "$D/w4" --no-profile --probe "s1:4096:90:1"
rep --four wide="$D/w4" --out "$D/rep.json"
eq "no profile: exit 1" "$RC" 1
has "says no profile_rank0.json" "$D/out.txt" "no profile_rank0.json"
fresh
gen "$D/w4"
rep --four wide="$D/w4" --out "$D/rep.json"
eq "no mem_probe event: exit 1" "$RC" 1
has "says no mem_probe event" "$D/out.txt" "no mem_probe event"
fresh
gen "$D/w4" --no-ckpt --probe "s1:4096:90:1"
rep --four wide="$D/w4" --out "$D/rep.json"
eq "no checkpoint: exit 1" "$RC" 1
has "says no ckpt" "$D/out.txt" "no ckpt_"
fresh
gen "$D/w4" --no-complete --probe "s1:4096:90:1"
rep --four wide="$D/w4" --out "$D/rep.json"
eq "no complete.json: exit 1" "$RC" 1
has "says the run did not finish" "$D/out.txt" "did not finish"
fresh
rep --four wide="$D/none" --out "$D/rep.json"
eq "no log at all: exit 1" "$RC" 1
has "says no log.jsonl" "$D/out.txt" "no log.jsonl"
fresh
gen "$D/w4" --only-events 1 --probe "s1:4096:90:1"
rep --four wide="$D/w4" --out "$D/rep.json"
eq "one step event: exit 1" "$RC" 1
has "says it needs two step events" "$D/out.txt" "need at least 2"
fresh
gen "$D/w4" --torn --probe "s1:4096:90:1"
rep --four wide="$D/w4" --out "$D/rep.json"
eq "a torn last log line is skipped, the run reports" "$RC" 0
eq "...with the rate intact" "$(jqr '.four_rank.wide.steady_tok_per_s')" 16000.0

section "M4 red arm: the global batch must not depend on the world size"
fresh
gen "$D/w4" --probe "s1:4096:90:1"
gen "$D/e8" --world 8 --steps 20 --rate 28800 --no-profile --no-ckpt --skew 5000
rep --four wide="$D/w4" --eight "$D/e8" --out "$D/rep.json"
eq "a different batch: exit 1" "$RC" 1
has "says the global batch differs" "$D/out.txt" "global batch differs"
eq "same_global_batch is false in the receipt" "$(jqr '.scaling.same_global_batch')" false
fresh
gen "$D/w4" --probe "s1:4096:90:1"
gen "$D/e8" --world 8 --steps 20 --rate 28800 --no-profile --no-ckpt --no-complete
rep --four wide="$D/w4" --eight "$D/e8" --out "$D/rep.json"
eq "an unfinished 8-rank run: exit 1" "$RC" 1

section "M5 red arm: the rate floor"
fresh
gen "$D/w4" --rate 16000 --probe "s1:4096:90:1"
gen "$D/c4" --rate 9000 --probe "s1:4096:90:1"
rep --four wide="$D/w4" --four ctrl="$D/c4" --min-tok-s 12000 --out "$D/rep.json"
eq "an arm below --min-tok-s: exit 1" "$RC" 1
has "names the arm and the floor" "$D/out.txt" "4-rank ctrl: 9000.0 tok/s is below the floor 12000"
rep --four wide="$D/w4" --four ctrl="$D/c4" --min-tok-s 8000 --out "$D/rep.json"
eq "both above the floor: exit 0" "$RC" 0

section "M6 grad_ckpt decision: on over the cap, undecided without the longest pool, probe must run with checkpointing off"
fresh
gen "$D/w4" --probe "s1:4096:90:1"
gen "$D/p" --world 4 --steps 1 --probe "s1:4096:120:0,chat:6000:150:0,tool:12288:231:0"
rep --four wide="$D/w4" --probe "$D/p" --out "$D/rep.json" --env-out "$D/env.txt"
eq "a 231 GiB probe peak: exit 0" "$RC" 0
eq "decision on" "$(jqr '.grad_ckpt.decision')" on
has "the reason names the cap" "$(jqr '.grad_ckpt.reason')" "exceeds 223"
eq "env says on" "$(envv GRAD_CKPT_DECISION)" on
rep --four wide="$D/w4" --probe "$D/p" --max-gib 240 --out "$D/rep.json"
eq "the same probe is off under a 240 GiB cap" "$(jqr '.grad_ckpt.decision')" off
gen "$D/p2" --world 4 --steps 1 --probe "s1:4096:120:0"
rep --four wide="$D/w4" --probe "$D/p2" --out "$D/rep.json" --env-out "$D/env.txt"
eq "only S1 probed: exit 0, it is not a failure" "$RC" 0
eq "decision undecided" "$(jqr '.grad_ckpt.decision')" undecided
has "the reason says to rerun after self-distillation" "$(jqr '.grad_ckpt.reason')" "accept.sh --from train_smoke"
eq "env says undecided" "$(envv GRAD_CKPT_DECISION)" undecided
gen "$D/p3" --world 4 --steps 1 --probe "s1:4096:120:1,chat:6000:150:0,tool:12288:190:0"
rep --four wide="$D/w4" --probe "$D/p3" --out "$D/rep.json"
eq "a probe pool run with checkpointing on: exit 1" "$RC" 1
has "says on, not off" "$D/out.txt" "probed with grad_ckpt on, not off"
mkdir -p "$D/p4"; echo '{"event":"config","world":4}' >"$D/p4/log.jsonl"
rep --four wide="$D/w4" --probe "$D/p4" --out "$D/rep.json"
eq "an empty probe without --probe-oom: exit 1" "$RC" 1
has "says no mem_probe event in the probe run" "$D/out.txt" "no mem_probe event in the probe run"

section "M7 --probe-oom: running out of memory with checkpointing off is the answer"
fresh
gen "$D/w4" --probe "s1:4096:90:1"
mkdir -p "$D/p5"; echo '{"event":"config","world":4}' >"$D/p5/log.jsonl"
rep --four wide="$D/w4" --probe "$D/p5" --probe-oom --out "$D/rep.json" --env-out "$D/env.txt"
eq "exit 0" "$RC" 0
eq "decision on" "$(jqr '.grad_ckpt.decision')" on
has "the reason says out of CUDA memory" "$(jqr '.grad_ckpt.reason')" "ran out of CUDA memory"
eq "env says on" "$(envv GRAD_CKPT_DECISION)" on
gen "$D/p6" --world 4 --steps 1 --probe "s1:4096:120:0,chat:6000:150:0"
rep --four wide="$D/w4" --probe "$D/p6" --probe-oom --out "$D/rep.json"
eq "pools that fit first are listed" "$(jqr '.grad_ckpt.reason | test("chat 150.0 GiB")')" true
eq "still on" "$(jqr '.grad_ckpt.decision')" on
rep --four wide="$D/w4" --probe "$D/none" --probe-oom --out "$D/rep.json"
eq "no probe log at all under --probe-oom: exit 0, decision on" "$RC$(jqr '.grad_ckpt.decision')" "0on"

section "M8 arguments"
rep --four wide="$D/w4"
eq "--out is required" "$RC" 2
python3 "$REPORT" --help >"$D/h.txt" 2>&1; eq "--help exits 0" "$?" 0
has "--help documents --probe-oom" "$D/h.txt" "probe-oom"

finish
