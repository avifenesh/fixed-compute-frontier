#!/usr/bin/env bash
# CPU tests of sweep_read.py: synthetic train27 log.jsonl pairs in, the pre-registered sweep reading out.
# The red arms: every safety bar failing on its own (KL, replay NLL, tool agreement, loss spike) and passing just inside it,
# every signal arm failing and passing, the widening requirement, each row of the choice table (high, low, extend, owner),
# the abort rule (two consecutive KL breaches, tool agreement, recall loss), a step logged twice (last wins), a torn last line,
# a missing log / eval step / key (exit 1), a bad command line (exit 2), and --json.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
READ=${SWEEP_READ:-$BOX_DIR/sweep_read.py}
D=$SCRATCH/sw; mkdir -p "$D"
GEN=$D/gen.py
cat >"$GEN" <<'PYEOF'
"""gen.py RUNS TAG LR [--evals 0,50,..,300] [--set arm:key=val ...] [--setat arm:step:key=val ...] [--dup arm:step:key=val ...]
[--spike arm:step:pool:val] [--torn] [--no-arm ctrl] [--drop-key arm:key]: one sweep tag, wide_s0 and ctrl_s0."""
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument("runs"); ap.add_argument("tag"); ap.add_argument("lr", type=float)
ap.add_argument("--evals", default=",".join(str(s) for s in range(0, 301, 50)))
ap.add_argument("--set", action="append", default=[]); ap.add_argument("--setat", action="append", default=[])
ap.add_argument("--dup", action="append", default=[]); ap.add_argument("--spike", action="append", default=[])
ap.add_argument("--torn", action="store_true"); ap.add_argument("--no-arm", default="")
ap.add_argument("--drop-key", action="append", default=[]); ap.add_argument("--last-step", type=int, default=300)
a = ap.parse_args()
steps = [int(x) for x in a.evals.split(",")]


def val(s):
    return float(s)


def base(arm, step):
    r = {"event": "eval", "step": step, "kl_to_untouched": 0.0 if step == 0 else 0.02,
         "replay_nll": 1.9694 if step == 0 else 1.975, "tool_token_agree": 1.0 if step == 0 else 0.97,
         "tool_turn_agree": 0.9, "s1:len256:le256": 0.60, "s1:len512:gt256": 0.50, "s1:all": 0.55,
         "nll:all": 1.0, "gate_rate_s1": 0.2, "gate_rate_text": 0.2, "eval_s": 5.0,
         "recall:16384": 0.95, "recall:32768": 0.90}
    if arm == "ctrl":
        r["gate_rate_s1"] = r["gate_rate_text"] = 0.0
    return r


recs = {"wide": [], "ctrl": []}
for arm in recs:
    over = {}
    for spec in a.set:
        who, kv = spec.split(":", 1)
        if who == arm:
            k, v = kv.split("=", 1); over.setdefault("*", {})[k] = val(v)
    for spec in a.setat:
        who, st, kv = spec.split(":", 2)
        if who == arm:
            k, v = kv.split("=", 1); over.setdefault(int(st), {})[k] = val(v)
    dups = {}
    for spec in a.dup:
        who, st, kv = spec.split(":", 2)
        if who == arm:
            k, v = kv.split("=", 1); dups.setdefault(int(st), {})[k] = val(v)
    recs[arm].append({"event": "config", "world": 8, "n_gates": 48 if arm == "wide" else 0,
                      "args": {"lr": a.lr, "w_lr": a.lr, "steps": 700, "arm": arm}})
    for s in range(10, a.last_step + 1, 10):
        loss = {"s1": 1.0, "chat": 1.2, "tool": 0.8, "lm": 2.0}
        for spec in a.spike:
            who, st, pool, v = spec.split(":")
            if who == arm and int(st) == s:
                loss[pool] = val(v)
        recs[arm].append({"event": "step", "step": s, "loss": loss, "gnorm": 1.0, "lr": a.lr, "lr_w": a.lr})
    for s in steps:
        if s in dups:  # an earlier record of the same step (a rewound run): the later one must win
            d = base(arm, s); d.update(dups[s]); recs[arm].append(d)
        r = base(arm, s)
        r.update(over.get("*", {}) if s > 0 else {})
        r.update(over.get(s, {}))
        for spec in a.drop_key:
            who, k = spec.split(":")
            if who == arm:
                r.pop(k, None)
        recs[arm].append(r)
for arm, rs in recs.items():
    if a.no_arm == arm:
        continue
    d = os.path.join(a.runs, a.tag, f"{arm}_s0"); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "log.jsonl"), "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")
        if a.torn:
            f.write('{"event": "eval", "step": 350, "kl_to_untou')
PYEOF
gen() { python3 "$GEN" "$@"; }
# run_read RUNSDIR args...: sets OUT (stdout+stderr) and RC
run_read() { OUT=$(python3 "$READ" "$@" 2>&1); RC=$?; }
fresh() { R=$D/$1; rm -rf "$R"; mkdir -p "$R"; }

section "a safe pair with no signal, one tag"
fresh a; gen "$R" t1 1e-4
run_read --runs "$R" --tag t1
eq "exit 0" "$RC" 0
has "safety wide PASS" "$OUT" 'safety wide: PASS'
has "safety ctrl PASS" "$OUT" 'safety ctrl: PASS'
has "signal absent" "$OUT" 'signal: absent'
has "one tag makes no choice" "$OUT" 'choice: n/a'
has "no abort" "$OUT" 'no abort rule hit'
has "last line is SWEEP_READ" "$(printf '%s\n' "$OUT" | tail -n1)" '^SWEEP_READ choice=n/a steps=250,300 abort=no t1:safety=pass,signal=no$'
has "the lr comes from the config event" "$OUT" 'lr 0\.0001'

section "safety bars: each fails alone and passes just inside"
for spec in "kl_to_untouched:0.081:0.079:kl_to_untouched" "replay_nll:2.021:2.019:replay_nll" "tool_token_agree:0.949:0.951:tool_token_agree"; do
  IFS=: read -r key bad good label <<<"$spec"
  fresh "s_$key"; gen "$R" t1 1e-4 --set "wide:$key=$bad"
  run_read --runs "$R" --tag t1
  has "$key $bad fails wide" "$OUT" 'safety wide: FAIL'
  has "$key named in the failing line" "$(printf '%s\n' "$OUT" | grep 'safety wide')" "$label [0-9.]+ \\(FAIL"
  has "ctrl untouched stays PASS ($key)" "$OUT" 'safety ctrl: PASS'
  gen "$R" t1 1e-4 --set "ctrl:$key=$bad"
  run_read --runs "$R" --tag t1
  has "$key $bad fails ctrl too" "$OUT" 'safety ctrl: FAIL'
  gen "$R" t1 1e-4 --set "wide:$key=$good" --set "ctrl:$key=$good"
  run_read --runs "$R" --tag t1
  has "$key $good passes both" "$OUT" 'safety wide: PASS'
  has "$key $good passes ctrl" "$OUT" 'safety ctrl: PASS'
done
fresh s_avg; gen "$R" t1 1e-4 --setat wide:250:kl_to_untouched=0.14 --setat wide:300:kl_to_untouched=0.03
run_read --runs "$R" --tag t1
has "the safety value is the mean over 250 and 300 (0.085 fails)" "$OUT" 'safety wide: FAIL.*kl_to_untouched 0\.0850'
fresh s_avg2; gen "$R" t1 1e-4 --setat wide:250:kl_to_untouched=0.13 --setat wide:300:kl_to_untouched=0.03
run_read --runs "$R" --tag t1
has "mean 0.080 passes" "$OUT" 'safety wide: PASS'
fresh s_early; gen "$R" t1 1e-4 --setat wide:50:kl_to_untouched=0.5 --setat wide:100:kl_to_untouched=0.5
run_read --runs "$R" --tag t1
has "evals outside the averaged steps do not count against safety" "$OUT" 'safety wide: PASS'

section "loss spike: more than 3x the running median of earlier values"
fresh sp1; gen "$R" t1 1e-4 --spike wide:200:tool:3.5
run_read --runs "$R" --tag t1
has "3.5 against a median of 0.8 fails" "$OUT" 'safety wide: FAIL.*loss_spike_ratio'
has "the spike is located" "$OUT" 'largest loss ratio at tool at step 200'
fresh sp2; gen "$R" t1 1e-4 --spike wide:200:tool:2.3
run_read --runs "$R" --tag t1
has "2.3 against 0.8 (2.9x) passes" "$OUT" 'safety wide: PASS'
fresh sp3; gen "$R" t1 1e-4 --spike wide:310:tool:9 --last-step 320
run_read --runs "$R" --tag t1
has "a spike after the last read step is not counted" "$OUT" 'safety wide: PASS'
fresh sp4; gen "$R" t1 1e-4 --spike wide:20:tool:9
run_read --runs "$R" --tag t1
has "a spike with fewer than 3 earlier values is not judged" "$OUT" 'safety wide: PASS'
fresh sp5; gen "$R" t1 1e-4 --spike ctrl:100:lm:7
run_read --runs "$R" --tag t1
has "the ctrl arm is judged by the same bar" "$OUT" 'safety ctrl: FAIL'

section "signal: each arm fails and passes alone"
fresh g1; gen "$R" t1 1e-4 --set wide:s1:len256:le256=0.66
run_read --runs "$R" --tag t1
has "6 points on s1:len256:le256 is signal" "$OUT" 'signal: PRESENT'
has "the gap is printed in points" "$OUT" 's1:len256:le256 gap wide-ctrl \+6\.0 points \(ok'
fresh g2; gen "$R" t1 1e-4 --set wide:s1:len256:le256=0.64
run_read --runs "$R" --tag t1
has "4 points is not signal" "$OUT" 'signal: absent'
fresh g2b; gen "$R" t1 1e-4 --setat wide:250:s1:len256:le256=0.60 --setat wide:300:s1:len256:le256=0.72
run_read --runs "$R" --tag t1
has "the gap is a mean over 250 and 300 (0, 12 points -> 6)" "$OUT" 'signal: PRESENT'
fresh g3; gen "$R" t1 1e-4 --set wide:gate_rate_s1=0.31
run_read --runs "$R" --tag t1
has "gate rate 0.31 vs 0.20 (1.55x) is signal" "$OUT" 'signal: PRESENT'
has "ratio printed" "$OUT" 'ratio 1\.55 \(ok'
fresh g4; gen "$R" t1 1e-4 --set wide:gate_rate_s1=0.28
run_read --runs "$R" --tag t1
has "gate rate 1.4x is not signal" "$OUT" 'signal: absent'
fresh g4b; gen "$R" t1 1e-4 --set wide:gate_rate_s1=0.2 --set wide:gate_rate_text=0.0
run_read --runs "$R" --tag t1
has "a text gate rate of 0 with S1 gating on is infinite ratio, signal" "$OUT" 'signal: PRESENT'
fresh g4c; gen "$R" t1 1e-4 --set wide:gate_rate_s1=0.0 --set wide:gate_rate_text=0.0
run_read --runs "$R" --tag t1
has "both gate rates 0 is not signal (and does not crash)" "$OUT" 'signal: absent'
# answer NLL: wide 15% below ctrl, and the gap widening over the last 100 steps
fresh n1; gen "$R" t1 1e-4 --set wide:nll:all=0.85 --setat wide:150:nll:all=0.95 --setat wide:200:nll:all=0.95
run_read --runs "$R" --tag t1
has "15% below ctrl and widening is signal" "$OUT" 'signal: PRESENT'
has "widening True is printed" "$OUT" 'widening True \(ok'
fresh n2; gen "$R" t1 1e-4 --set wide:nll:all=0.85
run_read --runs "$R" --tag t1
has "15% below ctrl with a flat gap is not signal" "$OUT" 'signal: absent'
has "widening False is printed" "$OUT" 'widening False \(no'
fresh n3; gen "$R" t1 1e-4 --set wide:nll:all=0.93 --setat wide:150:nll:all=0.99 --setat wide:200:nll:all=0.99
run_read --runs "$R" --tag t1
has "7% below ctrl, widening, is not signal (needs 10%)" "$OUT" 'signal: absent'
fresh n4; gen "$R" t1 1e-4 --set wide:nll:all=0.85 --evals 0,50,250,300
run_read --runs "$R" --tag t1
has "no evals 100 steps earlier: widening is not demonstrated, so no signal" "$OUT" 'widening None'
has "...and the run does not crash" "$(printf '%s\n' "$OUT" | tail -n1)" '^SWEEP_READ'
fresh n5; gen "$R" t1 1e-4 --set wide:s1:len512:gt256=0.62
run_read --runs "$R" --tag t1
has "len512:gt256 is printed beside the signal" "$OUT" 'beside: s1:len512:gt256 gap wide-ctrl \+12\.0 points'
has "and is not itself a signal" "$OUT" 'signal: absent'

section "choice table, two tags (given high first: the sort is by lr)"
safe_sig=(--set wide:s1:len256:le256=0.70)
fresh c1; gen "$R" hi 1e-3 "${safe_sig[@]}"; gen "$R" lo 1e-4
run_read --runs "$R" --tag hi --tag lo
has "high passes safety and shows signal: high" "$OUT" 'choice: hi  '
has "last line carries both tags" "$(printf '%s\n' "$OUT" | tail -n1)" 'SWEEP_READ choice=hi steps=250,300 abort=no hi:safety=pass,signal=yes lo:safety=pass,signal=no'
fresh c2; gen "$R" hi 1e-3 --set wide:kl_to_untouched=0.12 "${safe_sig[@]}"; gen "$R" lo 1e-4
run_read --runs "$R" --tag lo --tag hi
has "high fails safety, low passes: low" "$OUT" 'choice: lo  '
has "reason names the failing tag" "$OUT" 'hi fails safety, lo passes'
fresh c3; gen "$R" hi 1e-3 --set wide:kl_to_untouched=0.12; gen "$R" lo 1e-4 --set wide:tool_token_agree=0.9
run_read --runs "$R" --tag hi --tag lo
has "both fail safety: the owner decides" "$OUT" 'choice: owner  .*both fail safety'
fresh c4; gen "$R" hi 1e-3; gen "$R" lo 1e-4
run_read --runs "$R" --tag hi --tag lo
has "both pass, neither shows signal: extend" "$OUT" 'choice: extend  .*extend both pairs to 600 steps.*--steps 550,600'
fresh c5; gen "$R" hi 1e-3; gen "$R" lo 1e-4 "${safe_sig[@]}"
run_read --runs "$R" --tag hi --tag lo
has "high safe without signal while low shows signal: owner" "$OUT" 'choice: owner  .*does not cover'
fresh c6; gen "$R" hi 1e-3; gen "$R" lo 1e-4 --set wide:kl_to_untouched=0.12
run_read --runs "$R" --tag hi --tag lo
has "high safe without signal while low fails safety: owner" "$OUT" 'choice: owner  .*safety fail and signal absent'
fresh c7; gen "$R" hi 1e-3 --set wide:kl_to_untouched=0.12; gen "$R" lo 1e-4 --set ctrl:replay_nll=2.1
run_read --runs "$R" --tag hi --tag lo
has "a ctrl-arm failure fails the whole tag" "$OUT" 'choice: owner'
fresh c8; gen "$R" hi 1e-3 --set wide:replay_nll=2.1 "${safe_sig[@]}"; gen "$R" lo 1e-4 "${safe_sig[@]}"
run_read --runs "$R" --tag hi --tag lo
has "high unsafe, low safe with signal: low" "$OUT" 'choice: lo  '
fresh c9; gen "$R" hi 1e-3 "${safe_sig[@]}"; gen "$R" lo 1e-4 "${safe_sig[@]}"
run_read --runs "$R" --tag lo --tag hi
has "both safe with signal: high (the rule's first line)" "$OUT" 'choice: hi  '
run_read --runs "$R" --tag hi --tag hi
eq "the same tag twice is a bad command line" "$RC" 2

section "the extension read (550,600)"
fresh e1; gen "$R" hi 1e-3 --evals 0,50,100,150,200,250,300,350,400,450,500,550,600 --last-step 600 --set wide:s1:len256:le256=0.7
gen "$R" lo 1e-4 --evals 0,50,100,150,200,250,300,350,400,450,500,550,600 --last-step 600
run_read --runs "$R" --tag hi --tag lo --steps 550,600
eq "exit 0" "$RC" 0
has "reads the later evals" "$OUT" 'evals averaged at \[550, 600\]'
has "high wins on signal" "$OUT" 'choice: hi  '
run_read --runs "$R" --tag hi --tag lo --steps 250,300 --seed 0
eq "the same logs read at 250,300 also work" "$RC" 0

section "a step logged twice: the last record wins"
fresh d1; gen "$R" t1 1e-4 --dup wide:250:kl_to_untouched=0.5
run_read --runs "$R" --tag t1
has "bad first, good last: PASS" "$OUT" 'safety wide: PASS'
fresh d2; gen "$R" t1 1e-4 --setat wide:250:kl_to_untouched=0.5 --setat wide:300:kl_to_untouched=0.5 --dup wide:250:kl_to_untouched=0.01 --dup wide:300:kl_to_untouched=0.01
run_read --runs "$R" --tag t1
has "good first, bad last: FAIL" "$OUT" 'safety wide: FAIL'
fresh d3; gen "$R" t1 1e-4 --torn
run_read --runs "$R" --tag t1
eq "a torn last line is ignored (exit 0)" "$RC" 0
has "...and the read is unchanged" "$OUT" 'safety wide: PASS'

section "the abort rule"
fresh ab1; gen "$R" t1 1e-4 --setat wide:200:kl_to_untouched=0.16 --setat wide:250:kl_to_untouched=0.17
run_read --runs "$R" --tag t1
has "KL >= 0.15 on two consecutive evals: hit" "$OUT" 'ABORT_RULE_HIT t1/wide'
has "the evals are named" "$OUT" 'abort wide: kl_to_untouched >= 0\.15 on evals 200 and 250'
has "last line says abort=yes" "$(printf '%s\n' "$OUT" | tail -n1)" 'abort=yes'
fresh ab2; gen "$R" t1 1e-4 --setat wide:200:kl_to_untouched=0.16
run_read --runs "$R" --tag t1
hasnt "one breach alone is not the rule" "$OUT" 'ABORT_RULE_HIT'
fresh ab3; gen "$R" t1 1e-4 --setat wide:150:kl_to_untouched=0.16 --setat wide:250:kl_to_untouched=0.16
run_read --runs "$R" --tag t1
hasnt "two breaches that are not consecutive are not the rule" "$OUT" 'ABORT_RULE_HIT'
fresh ab4; gen "$R" t1 1e-4 --setat wide:100:tool_token_agree=0.89
run_read --runs "$R" --tag t1
has "tool agreement under 0.90 at any eval: hit" "$OUT" 'abort wide: tool_token_agree 0\.890 < 0\.9 at step 100'
fresh ab5; gen "$R" t1 1e-4 --setat wide:100:tool_token_agree=0.91
run_read --runs "$R" --tag t1
hasnt "0.91 is not the abort (it is a safety miss only if averaged under 0.95)" "$OUT" 'ABORT_RULE_HIT'
fresh ab6; gen "$R" t1 1e-4 --setat wide:200:recall:16384=0.83
run_read --runs "$R" --tag t1
has "recall 12 points under step 0: hit" "$OUT" 'abort wide: recall:16384 0\.830 is 12 points under step 0 \(0\.950\) at step 200'
fresh ab7; gen "$R" t1 1e-4 --setat wide:200:recall:16384=0.86
run_read --runs "$R" --tag t1
hasnt "recall 9 points under is not the rule" "$OUT" 'ABORT_RULE_HIT'
fresh ab8; gen "$R" t1 1e-4 --setat wide:200:recall:32768=0.79
run_read --runs "$R" --tag t1
has "the rule covers every recall length (32768)" "$OUT" 'abort wide: recall:32768'
fresh ab9; gen "$R" t1 1e-4 --setat ctrl:100:tool_token_agree=0.80
run_read --runs "$R" --tag t1
has "the abort rule is read for ctrl as well" "$OUT" 'ABORT_RULE_HIT t1/ctrl'
fresh ab10; gen "$R" t1 1e-4 --evals 0,50,100,150,200,250,300,350,400 --setat wide:350:kl_to_untouched=0.5 --setat wide:400:kl_to_untouched=0.5
run_read --runs "$R" --tag t1
hasnt "evals after the last read step are not part of the abort check" "$OUT" 'ABORT_RULE_HIT'
fresh ab11; gen "$R" hi 1e-3 --setat wide:200:kl_to_untouched=0.16 --setat wide:250:kl_to_untouched=0.17; gen "$R" lo 1e-4
run_read --runs "$R" --tag hi --tag lo
has "in a sweep the abort names the tag and arm" "$OUT" 'ABORT_RULE_HIT hi/wide'
has "...and does not blame the other tag" "$OUT" 'abort wide: none'

section "step-0 replay NLL differs from the bar: a NOTE"
fresh nt; gen "$R" t1 1e-4 --setat wide:0:replay_nll=2.2
run_read --runs "$R" --tag t1
has "NOTE printed" "$OUT" 'NOTE step-0 replay_nll 2\.2000 differs'
fresh nt2; gen "$R" t1 1e-4
run_read --runs "$R" --tag t1 --untouched-replay-nll 2.0
has "--untouched-replay-nll moves the bar (limit 2.05, value 1.975 passes)" "$OUT" 'safety wide: PASS'
run_read --runs "$R" --tag t1 --untouched-replay-nll 1.90
has "a lower untouched value turns the same run into a replay_nll failure" "$OUT" 'safety wide: FAIL'

section "missing inputs: exit 1 naming what is missing"
fresh m1; gen "$R" t1 1e-4 --no-arm ctrl
run_read --runs "$R" --tag t1
eq "no ctrl log: exit 1" "$RC" 1
has "names the missing log" "$OUT" 'no log: .*ctrl_s0/log.jsonl'
fresh m2; gen "$R" t1 1e-4 --evals 0,50,100,150,200,300
run_read --runs "$R" --tag t1
eq "no eval at 250: exit 1" "$RC" 1
has "names the step" "$OUT" 'no eval at step 250'
fresh m3; gen "$R" t1 1e-4 --drop-key wide:tool_token_agree
run_read --runs "$R" --tag t1
eq "an eval without tool_token_agree: exit 1" "$RC" 1
has "names the key" "$OUT" 'has no tool_token_agree'
fresh m4; gen "$R" t1 1e-4 --drop-key wide:gate_rate_text
run_read --runs "$R" --tag t1
eq "an eval without gate_rate_text: exit 1" "$RC" 1
fresh m5; gen "$R" t1 1e-4
run_read --runs "$R" --tag nope
eq "an unknown tag: exit 1" "$RC" 1
run_read --runs "$R" --tag t1 --seed 5
eq "a seed with no run dir: exit 1" "$RC" 1
fresh m6; gen "$R" t1 1e-4; gen "$R" t2 1e-3 --no-arm wide
run_read --runs "$R" --tag t1 --tag t2
eq "one of two tags incomplete: exit 1, no partial choice" "$RC" 1
hasnt "no choice line" "$OUT" '^choice:'

section "bad command line: exit 2"
run_read --runs "$D"
eq "no tag" "$RC" 2
run_read --runs "$D" --tag a --tag b --tag c
eq "three tags" "$RC" 2
run_read --runs "$D" --tag a --steps abc
eq "--steps not integers" "$RC" 2
run_read --runs "$D" --tag a --steps ""
eq "--steps empty" "$RC" 2

section "--json"
fresh j1; gen "$R" hi 1e-3 "${safe_sig[@]}"; gen "$R" lo 1e-4
run_read --runs "$R" --tag hi --tag lo --json "$D/out.json"
eq "exit 0" "$RC" 0
check "the JSON parses" python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$D/out.json"
eq "choice in the JSON" "$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['choice'])" "$D/out.json")" hi
eq "both tags in the JSON" "$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['tags']))" "$D/out.json")" 2
eq "steps in the JSON" "$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['steps'])" "$D/out.json")" "[250, 300]"
check "-h prints the rule" bash -c "python3 '$READ' -h | grep -q untouched-replay-nll"

finish
