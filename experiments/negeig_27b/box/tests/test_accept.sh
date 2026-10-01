#!/usr/bin/env bash
# CPU tests of accept.sh: every box command is stubbed (nvidia-smi, python shim, torchrun, verl venv, fake SGLang server).
# Covers the NCCL_NVLS_ENABLE A/B and its one file, the three-part train smoke (launch shapes, receipts, grad_ckpt decision),
# the refusals (busy GPU, active unit, no NVLS file, no replay), --only/--from, and the ACCEPT_OK line.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ACCEPT=$BOX_DIR/accept.sh

acc() { "$ACCEPT" "$@" >"$B/acc.out" 2>&1; RC=$?; }
nvls_launches() { # -> the NCCL_NVLS_ENABLE values of the nccl probe launches, in order, space separated
  python3 - "$STUB_LOG" <<'PYEOF'
import json, sys
print(" ".join(json.loads(l)["nvls"] for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("script") == "nccl_probe.py"))
PYEOF
}
nvls_file() { sed -n 's/^NCCL_NVLS_ENABLE=//p' "$B/nccl_nvls.env"; }
jqr() { jq -r "$1" "$2"; }
nlaunch() { # number of trainer launches (not nccl probes)
  python3 - "$STUB_LOG" <<'PYEOF'
import json, sys
print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("script") != "nccl_probe.py"))
PYEOF
}
arg_of() { # arm index flag -> the value after the flag in that launch's argv
  stub_field "$1" "$2" argv | tr ' ' '\n' | awk -v f="$3" '$0 == f { getline; print; exit }'
}

section "N1 nccl A/B: a tie goes to NVLS=0, rewritten into the one file"
mk_accept_box n1
echo "NCCL_NVLS_ENABLE=1" >"$B/nccl_nvls.env"
acc --only nccl
eq "exit 0" "$RC" 0
has "ONLY_OK printed" "$B/acc.out" "^ONLY_OK nccl$"
eq "interleaved 0,1 x 3 rounds" "$(nvls_launches)" "0 1 0 1 0 1"
eq "tie: NVLS=0" "$(nvls_file)" 0
eq "receipt json chosen 0" "$(jqr .chosen "$B/accept/nccl_nvls.json")" 0
has "decision line names the file" "$B/acc.out" "NVLS_DECISION NCCL_NVLS_ENABLE=0 .* written to $B/nccl_nvls.env"
check_not "--only never writes ACCEPT_OK" test -e "$B/accept/ACCEPT_OK"
eq "probe sizes 64 and 512 MiB" "$(stub_field - 0 argv)" "64 512"
has "the file carries a measured-by comment" "$B/nccl_nvls.env" "^# measured by accept.sh nccl"

section "N2 nccl A/B: NVLS=1 clearly faster wins, and run_pair.sh trains with it"
mk_accept_box n2
STUB_NCCL_MS0=10 STUB_NCCL_MS1=8 acc --only nccl
eq "NVLS=1 chosen" "$(nvls_file)" 1
eq "receipt reason names the gain" "$(jqr .reason "$B/accept/nccl_nvls.json" | grep -c 'at least 3% faster')" 1
: >"$STUB_LOG"
STAGE=A "$E27_DIR/run_pair.sh" tagN2 0 --lr 1e-4 --steps 5 >"$B/rp.out" 2>&1
eq "run_pair.sh exit 0" "$?" 0
eq "run_pair.sh exports the file's NVLS=1 to the trainer" "$(stub_field wide 0 nvls)" 1

section "N3 nccl A/B: a 2% gain is below the 3% bar"
mk_accept_box n3
STUB_NCCL_MS0=10 STUB_NCCL_MS1=9.8 acc --only nccl
eq "NVLS=0 (bar not met)" "$(nvls_file)" 0
STUB_NCCL_MS0=10 STUB_NCCL_MS1=9.8 NVLS_MIN_GAIN=0.01 acc --only nccl
eq "NVLS_MIN_GAIN=0.01 lets the 2% gain win" "$(nvls_file)" 1

section "N4 nccl A/B: medians, not single runs"
mk_accept_box n4
STUB_NCCL_MS0=10 STUB_NCCL_MS1_SEQ=5,50,5 acc --only nccl
eq "one slow NVLS=1 outlier does not flip a median win" "$(nvls_file)" 1
mk_accept_box n4b
STUB_NCCL_MS0=10 STUB_NCCL_MS1_SEQ=20,5,20 acc --only nccl
eq "one fast NVLS=1 outlier does not flip a median loss" "$(nvls_file)" 0

section "N5 nccl A/B: NVLS=1 failing or hanging is a 0, a failing baseline fails the step"
mk_accept_box n5
STUB_NCCL_FAIL1=1 acc --only nccl
eq "exit 0 (the box is fine, NVLS is not)" "$RC" 0
eq "NVLS=0" "$(nvls_file)" 0
eq "stopped at the first failing NVLS=1 run" "$(nvls_launches)" "0 1"
has "reason says failed" "$(jqr .reason "$B/accept/nccl_nvls.json")" "failed or hung"
mk_accept_box n5b
echo "NCCL_NVLS_ENABLE=1" >"$B/nccl_nvls.env"
STUB_NCCL_HANG1=1 NCCL_PROBE_TIMEOUT_S=3 acc --only nccl
eq "a hang under NVLS=1: exit 0" "$RC" 0
eq "a hang under NVLS=1: NVLS=0 written over the old 1" "$(nvls_file)" 0
has "the timeout is reported" "$B/acc.out" "treated as a hang"
check_not "no probe left running" pgrep -f "$B/accept/nccl_probe[.]py"
mk_accept_box n5c
echo "NCCL_NVLS_ENABLE=1" >"$B/nccl_nvls.env"
STUB_NCCL_FAIL0=1 acc --only nccl
eq "baseline NVLS=0 failing: exit 1" "$RC" 1
has "ACCEPT_FAILED nccl" "$B/acc.out" "^ACCEPT_FAILED nccl "
eq "the file is not rewritten on a failed step" "$(nvls_file)" 1

section "N6 nccl: pin and bandwidth floor"
mk_accept_box n6
NVLS_PIN=1 acc --only nccl
eq "pin 1: one probe" "$(nvls_launches)" "1"
eq "pin 1 written" "$(nvls_file)" 1
mk_accept_box n6b
NVLS_PIN=2 acc --only nccl
eq "bad pin: exit 1" "$RC" 1
mk_accept_box n6c
STUB_NCCL_BUSBW=30 acc --only nccl
eq "30 GB/s is below the 60 GB/s floor: fails" "$RC" 1
has "says NCCL_TOO_SLOW" "$B/acc.out" "NCCL_TOO_SLOW"
STUB_NCCL_BUSBW=30 MIN_NCCL_GBPS=0 acc --only nccl
eq "MIN_NCCL_GBPS=0 turns the floor off" "$RC" 0

section "S1 train smoke: three parts, Stage A flags, receipts, grad_ckpt off"
mk_accept_box s1
acc --only train_smoke
eq "exit 0" "$RC" 0
has "ONLY_OK" "$B/acc.out" "^ONLY_OK train_smoke$"
eq "wide launched three times (a, b, c)" "$(launches wide)" 3
eq "ctrl launched once (a)" "$(launches ctrl)" 1
eq "part a wide: 4 ranks on GPUs 0-3" "$(stub_field wide 0 nproc) $(stub_field wide 0 cuda)" "4 0,1,2,3"
eq "part a ctrl: 4 ranks on GPUs 4-7" "$(stub_field ctrl 0 nproc) $(stub_field ctrl 0 cuda)" "4 4,5,6,7"
for f in "--steps 30" "--eval_every 30" "--save_every 30" "--recall_every 30" "--mem_probe 1" "--profile_step 10" "--lr 1e-4" "--w_lr 1e-4"; do
  has "part a has $f" "$(stub_field wide 0 argv)" "$f( |\$)"
done
has "part a out dir is the smoke tag" "$(stub_field wide 0 argv)" "--out $B/runs/accept-smoke/wide_s0"
has "part a carries the replay mix" "$(stub_field wide 0 argv)" "--chat_replay $B/data/sd/chat_sft.jsonl .*--tool_replay $B/data/sd/tool_sft.jsonl"
hasnt "part a does not touch grad_ckpt (the default is the Stage A value)" "$(stub_field wide 0 argv)" "--grad_ckpt"
eq "part b: one arm on 8 ranks, all GPUs" "$(stub_field wide 1 nproc) $(stub_field wide 1 cuda)" "8 0,1,2,3,4,5,6,7"
eq "part b steps" "$(arg_of wide 1 --steps)" 20
eq "part b has no evals, saves or recall" "$(arg_of wide 1 --eval_every)/$(arg_of wide 1 --save_every)/$(arg_of wide 1 --recall_every)" "0/0/0"
has "part b out dir" "$(stub_field wide 1 argv)" "--out $B/runs/accept-smoke8/wide_s0"
eq "part c: 4 ranks (H/2) so NCCL buffers are in the peak" "$(stub_field wide 2 nproc)" 4
eq "part c: mem_probe 2 and grad_ckpt 0" "$(arg_of wide 2 --mem_probe)/$(arg_of wide 2 --grad_ckpt)" "2/0"
has "part c out dir" "$(stub_field wide 2 argv)" "--out $B/runs/accept-probe/wide_s0"
eq "every trainer launch got the NVLS value from the file" "$(for i in 0 1 2; do stub_field wide $i nvls; done | sort -u)" 0
R=$B/accept/smoke_report.json
eq "report has no problems" "$(jqr '.problems | length' "$R")" 0
eq "decision off (probe peak within the cap)" "$(jqr .grad_ckpt.decision "$R")" off
eq "probe covered the tool pool" "$(jqr '.grad_ckpt.pools | has("tool")' "$R")" true
has "grad_ckpt.env flags" "$B/accept/grad_ckpt.env" '^GRAD_CKPT_FLAGS="--grad_ckpt 0"$'
has "grad_ckpt.env decision" "$B/accept/grad_ckpt.env" "^GRAD_CKPT_DECISION=off$"
has "grad_ckpt.env replay" "$B/accept/grad_ckpt.env" "^REPLAY=present$"
has "smoke.env carries tok/s per GPU" "$B/accept/smoke.env" "^SMOKE_TOK_PER_S_PER_GPU=4000"
has "smoke.env carries the 8 vs 4 speedup" "$B/accept/smoke.env" "^SMOKE8_SPEEDUP=1\.8"
has "smoke.env carries replay" "$B/accept/smoke.env" "^SMOKE_REPLAY=present$"
check "profile kept in receipts" test -f "$B/accept/smoke/accept-smoke_wide_s0/profile_rank0.json"
check "log.jsonl kept in receipts" test -f "$B/accept/smoke/accept-smoke_ctrl_s0/log.jsonl"
check "probe log kept in receipts" test -f "$B/accept/smoke/accept-probe_wide_s0/log.jsonl"
check_not "checkpoints are not copied into the receipts" bash -c "ls $B/accept/smoke/*/ckpt_* 2>/dev/null | grep -q ."
acc --only train_smoke
eq "a second run starts clean (complete.json would make run_pair skip)" "$(launches wide)" 6

section "S2 grad_ckpt decision: on when the probe peak is over the cap"
mk_accept_box s2
STUB_OFF_SLOPE=20 acc --only train_smoke
eq "exit 0" "$RC" 0
eq "decision on" "$(jqr .grad_ckpt.decision "$B/accept/smoke_report.json")" on
has "flags keep checkpointing on" "$B/accept/grad_ckpt.env" '^GRAD_CKPT_FLAGS="--grad_ckpt 1"$'
mk_accept_box s2b
STUB_OFF_SLOPE=12 GRAD_CKPT_MAX_GIB=150 acc --only train_smoke
eq "GRAD_CKPT_MAX_GIB moves the line" "$(jqr .grad_ckpt.decision "$B/accept/smoke_report.json")" on
mk_accept_box s2c
STUB_GPU_TOTAL_MIB=200000 STUB_OFF_SLOPE=12 acc --only train_smoke
eq "a smaller card lowers the default cap (0.85 x memory)" "$(jqr .grad_ckpt.max_gib "$B/accept/smoke_report.json")" 166.0
eq "...and the 207 GiB probe is then over it" "$(jqr .grad_ckpt.decision "$B/accept/smoke_report.json")" on

section "S3 grad_ckpt=0 probe out of memory is the answer: on"
mk_accept_box s3
STUB_OOM_GC0=tool acc --only train_smoke
eq "exit 0" "$RC" 0
has "reports the OOM" "$B/acc.out" "ran out of CUDA memory: decision on"
eq "decision on" "$(jqr .grad_ckpt.decision "$B/accept/smoke_report.json")" on
has "reason names the OOM" "$(jqr .grad_ckpt.reason "$B/accept/smoke_report.json")" "out of CUDA memory"
has "flags --grad_ckpt 1" "$B/accept/grad_ckpt.env" '^GRAD_CKPT_FLAGS="--grad_ckpt 1"$'
mk_accept_box s3b
plan wide ok ok fail
acc --only train_smoke
eq "a probe crash that is not an OOM fails the step" "$RC" 1
has "says not an out-of-memory" "$B/acc.out" "smoke part c failed .* not an out-of-memory"

section "S4 no replay yet: runs without the mix, grad_ckpt undecided, REQUIRE_REPLAY refuses"
mk_accept_box s4
rm "$B/data/sd/chat_sft.jsonl" "$B/data/sd/tool_sft.jsonl"
acc --only train_smoke
eq "exit 0" "$RC" 0
has "warns" "$B/acc.out" "WARNING: no replay"
hasnt "no chat or tool replay arguments reach the trainer" "$(stub_field wide 0 argv)" "--(chat|tool)_replay"
eq "decision undecided (the tool pool was not probed)" "$(jqr .grad_ckpt.decision "$B/accept/smoke_report.json")" undecided
has "flags stay on the safe value" "$B/accept/grad_ckpt.env" '^GRAD_CKPT_FLAGS="--grad_ckpt 1"$'
has "replay none" "$B/accept/grad_ckpt.env" "^REPLAY=none$"
mk_accept_box s4b
rm "$B/data/sd/tool_sft.jsonl"
acc --only train_smoke
has "one missing file means none" "$B/accept/grad_ckpt.env" "^REPLAY=none$"
mk_accept_box s4c
rm "$B/data/sd/chat_sft.jsonl" "$B/data/sd/tool_sft.jsonl"
REQUIRE_REPLAY=1 acc --only train_smoke
eq "REQUIRE_REPLAY=1 refuses: exit 1" "$RC" 1
has "says why" "$B/acc.out" "REQUIRE_REPLAY=1 and no self-distilled replay"
eq "nothing was launched" "$(nlaunch)" 0

section "S5 refusals"
mk_accept_box s5
STUB_GPU_USED_IDX=3:9000 acc --only train_smoke
eq "a busy GPU refuses" "$RC" 1
has "names the GPU and its use" "$B/acc.out" "GPU 3 holds 9000 MiB"
eq "nothing launched" "$(nlaunch)" 0
STUB_GPU_USED_MIB=1000 acc --only train_smoke
eq "1000 MiB used (driver context) is fine" "$RC" 0
mk_accept_box s5b
STUB_UNIT_ACTIVE=1 acc --only train_smoke
eq "an active training unit refuses" "$RC" 1
has "says hold or disarm" "$B/acc.out" "negeig27-train.service is active: ctl.sh hold"
eq "nothing launched" "$(nlaunch)" 0
mk_accept_box s5c
rm "$B/nccl_nvls.env"
acc --only train_smoke
eq "no NVLS file refuses" "$RC" 1
has "says run the nccl step" "$B/acc.out" "no NCCL_NVLS_ENABLE in $B/nccl_nvls.env: run the nccl step first"
mk_accept_box s5d
acc --only train_smoke --gpus 3
eq "an odd --gpus refuses" "$RC" 1
has "says even" "$B/acc.out" "must be even"
mk_accept_box s5e
plan ctrl fail
acc --only train_smoke
eq "part a arm failing fails the step" "$RC" 1
has "names part a" "$B/acc.out" "smoke part a failed"
eq "parts b and c did not run" "$(launches wide)" 1
check "the failed arm's stdout tail is shown" grep -q "stub: planned failure" "$B/acc.out"

section "S6 four-GPU box: H/2 = 2 ranks an arm, 4 ranks for the timing run"
mk_accept_box s6
STUB_GPUS=4 acc --only train_smoke --gpus 4
eq "exit 0" "$RC" 0
eq "arms of 2 ranks" "$(stub_field wide 0 nproc)/$(stub_field ctrl 0 nproc)" "2/2"
eq "timing run on 4 ranks" "$(stub_field wide 1 nproc)" 4

section "R1 full run: ACCEPT_OK, steps.tsv, accept.json"
mk_accept_box r1
STUB_NCCL_MS1=8 acc
eq "exit 0" "$RC" 0
has "ACCEPT_OK line" "$B/acc.out" "^ACCEPT_OK .* gpus=8 cc=10\.3 arm=wide nvls=1 replay=present grad_ckpt=off smoke_tok_per_s=[0-9.]+ per_gpu=[0-9.]+ peak_gib=[0-9.]+ speedup_8_vs_4=[0-9.]+ "
check "ACCEPT_OK file" test -s "$B/accept/ACCEPT_OK"
eq "accept.json says ACCEPT_OK" "$(jqr .result "$B/accept/accept.json")" ACCEPT_OK
eq "accept.json carries nvls and the decision" "$(jqr '[.nccl_nvls_enable, .grad_ckpt_decision, .replay] | join(",")' "$B/accept/accept.json")" "1,off,present"
eq "eight PASS rows" "$(grep -c "PASS" "$B/accept/steps.tsv")" 8
eq "steps ran in order" "$(cut -f1 "$B/accept/steps.tsv" | tr '\n' ' ')" "cuda_alloc gpu_report nccl kernel_check g0 verl_env sglang train_smoke "
check "g0 receipt" test -s "$B/accept/g0_27b_wide.json"
check "sglang chat receipt" test -s "$B/accept/sglang_chat_response.json"
check_not "the fake SGLang server was stopped" pgrep -f "fake_[s]glang_server[.]py"
has "the NVLS value in ACCEPT_OK is the file's" "$B/accept/ACCEPT_OK" "nvls=$(nvls_file) "

section "R2 a failing step stops the run and removes the old success marker; --from resumes"
STUB_G1=fail acc
eq "exit 1" "$RC" 1
has "ACCEPT_FAILED kernel_check" "$B/acc.out" "^ACCEPT_FAILED kernel_check "
check_not "old ACCEPT_OK removed" test -e "$B/accept/ACCEPT_OK"
check_not "old accept.json removed" test -e "$B/accept/accept.json"
hasnt "later steps did not run" "$B/acc.out" "== g0"
acc --from kernel_check
eq "--from after the fix: exit 0" "$RC" 0
has "earlier steps skipped as PASS on this boot" "$B/acc.out" "== cuda_alloc: PASS on this boot \(skipped\)"
has "nccl skipped" "$B/acc.out" "== nccl: PASS on this boot \(skipped\)"
hasnt "kernel_check ran" "$B/acc.out" "kernel_check: PASS on this boot"
has "ACCEPT_OK back" "$B/acc.out" "^ACCEPT_OK "
before=$(cat "$B/accept/ACCEPT_OK")
acc --only nccl
eq "--only keeps the marker" "$(cat "$B/accept/ACCEPT_OK")" "$before"
acc --from sglang
eq "--from sglang skips the six before it" "$(grep -cF 'PASS on this boot (skipped)' "$B/acc.out")" 6
has "...and runs sglang and the smoke" "$B/acc.out" "== sglang"
has "...the smoke too" "$B/acc.out" "== train_smoke"

section "R3 the other steps fail loudly"
mk_accept_box r3
STUB_CUDA_RC=1 acc
has "cuda_alloc failure" "$B/acc.out" "^ACCEPT_FAILED cuda_alloc "
mk_accept_box r3b
STUB_GPUS=4 acc --only gpu_report
eq "gpu_report with 4 of 8 GPUs fails" "$RC" 1
has "says expected 8" "$B/acc.out" "expected 8 GPUs"
mk_accept_box r3c
STUB_G0=fail acc --only g0
eq "g0 failure" "$RC" 1
mk_accept_box r3d
STUB_VERL_RC=1 acc --only verl_env
eq "verl_env failure" "$RC" 1
mk_accept_box r3e
STUB_SGLANG_ANSWER=12 acc --only sglang
eq "a wrong chat answer fails sglang" "$RC" 1
has "says expected 391" "$B/acc.out" "expected 391"
check_not "the server is stopped even when the step failed" pgrep -f "fake_[s]glang_server[.]py"
mk_accept_box r3f
STUB_SGLANG_CRASH=1 acc --only sglang
eq "a crashing server fails sglang" "$RC" 1
has "says exited early" "$B/acc.out" "sglang exited early"
mk_accept_box r3g
STUB_SGLANG_DELAY=3 STUB_SGLANG_REQLOG=$B/req.log acc --only sglang
eq "a slow start is waited out" "$RC" 0
has "the probe is think-off and greedy" "$B/req.log" '"enable_thinking": false.*|"temperature": 0'
has "GPU 0 freed message" "$B/acc.out" "sglang stopped, GPU 0 freed"

section "R4 arguments"
mk_accept_box r4
acc --only nosuchstep
eq "unknown step: exit 1" "$RC" 1
has "lists the steps" "$B/acc.out" "unknown step 'nosuchstep'"
acc --from nccl --only nccl
eq "--from and --only are exclusive" "$RC" 1
acc --help
eq "--help exits 0" "$RC" 0
has "--help prints the step list" "$B/acc.out" "train_smoke"
acc --bogus
eq "unknown argument: exit 1" "$RC" 1

finish
