#!/usr/bin/env python3
"""Pre-registered decision rule of the 27B tier-1 mechanism probe (written 2026-10-01, before any probe run).

Question: does the widened-beta gate give the Qwen3.8-27B the tier-1 state tracking past the training length that LoRA
alone does not reach, as it did on the 4B and 9B bases? A NO kills the retrofit idea for this model; a YES says the
mechanism is there and the S1 tie (stageA-log.md) means S1 did not need it.

Input: test_stepNNNNNN.json of wide_s<seed> and ctrl_s<seed> under --root (train_t1.py). Test set = train.py's final test
(seed 10,000, 128 sequences per task of 1,024 steps); windows 1x = steps 1-64, 4x = 65-256, 16x = 257-1024.
Primary statistic: D = 100 x (wide - ctrl) of the tier-1 mean accuracy (parity, swap, codeswap) in the 4x window, the
4B lane's kill bar. Interval: paired bootstrap over test sequences (both arms score the same sequences), 10,000 draws.
Gate used = beta > 1 on at least 1% of task tokens in the wide arm's busiest layer (the 4B lane's gate-use check).

  Rung 1, step 1,200 (the 4B budget, same token stream as the 4B seed-0 runs):
    GO      D >= 10 and the gate is used.
    KILL    D < 10 and ctrl's tier-1 4x accuracy >= 0.60: the 27B extrapolates tier-1 without the gate.
    EXTEND  otherwise: both arms continue to step 2,400 (same optimizer, constant LR; the 9B needed 1.5x the 4B budget
            to switch swap).
  Rung 2, step 2,400 (final):
    GO      D >= 10 and the gate is used.
    KILL    D < 10.
  REVIEW, no verdict, when D >= 10 with the gate unused, or when the two arms' configs differ in anything but the arm.
Reported, not deciding: the untouched model (the step-0 test), 1x and 16x and per-task numbers with intervals, tier 2,
the switch step per task in the val evals (first eval with 1x >= 0.95), gate use on task tokens and on text, KL to the
untouched model, the step-0 identity of the two arms (G0 on the test set), and the 4B/9B anchors below.

  decide.py --root RUNS/TAG --seed 0 --step 1200 [--final 0|1]     writes decision_stepNNNNNN.json, prints the verdict
Exit 0 with a verdict, 2 when an input is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TIER1 = ("parity", "swap", "codeswap")
TIER2 = ("s5full", "z3")
WIN = {"w1": 0, "w4": 1, "w16": 2}
BAR_POINTS = 10.0
CTRL_SOLVES = 0.60
GATE_USED = 0.01
SWITCH_W1 = 0.95
CONFIG_IGNORE = {"arm", "out", "resume", "mem_probe", "max_steps_debug", "dist_timeout_min", "eval_only"}
# 4B/9B results on the same test set (results/negeig-retrofit/, analyze.py), tier-1 mean accuracy in the 4x window
ANCHORS = {
    "4b_wikitext_1200_3seeds": {"wide": 0.784, "ctrl": 0.300, "diff_points": 48.3, "ci95": [38.4, 54.1]},
    "4b_wikitext_1200_seed0": {"wide": 0.684, "ctrl": 0.301,
                               "note": "the token stream this probe's seed 0 trains on; switches at val steps 400 "
                                       "(parity) and 900 (swap, codeswap)"},
    "4b_clean_replay_1200": {"wide_s0_s1_s2": [0.478, 0.602, 0.530], "ctrl_s0": 0.300,
                             "note": "clean replay delayed the switch: swap never switched in 1,200 steps"},
    "9b_wikitext_1200": {"wide": 0.53, "ctrl": 0.30, "note": "swap switched only after +600 steps: wide 0.66-0.82"},
}


def load_test(root, arm, seed, step):
    p = Path(root) / f"{arm}_s{seed}" / f"test_step{step:06d}.json"
    return json.loads(p.read_text()) if p.exists() else None


def matrix(rec, task, w):
    L = rec["test_len"]
    return np.array([row[WIN[w]] for row in rec["tasks"][f"{task}@{L}/win"]], dtype=float)


def mean_acc(rec, tasks_, w, idx=None):
    vals = []
    for t in tasks_:
        m = matrix(rec, t, w)
        vals.append((m if idx is None else m[idx]).mean())
    return float(np.mean(vals))


def paired_ci(W, C, tasks_, w, B, seed=20261001):
    n = len(matrix(W, tasks_[0], w))
    rng = np.random.default_rng(seed)
    mw = np.stack([matrix(W, t, w) for t in tasks_])  # tasks x sequences
    mc = np.stack([matrix(C, t, w) for t in tasks_])
    d = (mw - mc).mean(0)  # per-sequence tier mean difference (same sequence index = same test sequence)
    boot = [d[rng.integers(0, n, n)].mean() for _ in range(B)]
    return [100 * float(np.percentile(boot, 2.5)), 100 * float(np.percentile(boot, 97.5))]


def configs(run_dir):
    out = []
    f = Path(run_dir) / "log.jsonl"
    if f.exists():
        for line in f.open():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("event") == "config" and "eval_only" not in r:
                out.append(r["args"])
    return out


def val_curve(run_dir):
    rows = {}
    f = Path(run_dir) / "log.jsonl"
    if f.exists():
        for line in f.open():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("event") == "eval" and r.get("step", -1) >= 0:
                rows[r["step"]] = r
    return [rows[k] for k in sorted(rows)]


def switch_steps(curve):
    out = {}
    for t in TIER1:
        out[t] = next((r["step"] for r in curve if r[t]["w1"] is not None and r[t]["w1"] >= SWITCH_W1), None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="RUNS/TAG with wide_s<seed>/ and ctrl_s<seed>/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--final", type=int, default=0, help="1 at the last rung: no EXTEND")
    ap.add_argument("--B", type=int, default=10_000)
    a = ap.parse_args()
    root = Path(a.root)
    W, C = load_test(root, "wide", a.seed, a.step), load_test(root, "ctrl", a.seed, a.step)
    W0, C0 = load_test(root, "wide", a.seed, 0), load_test(root, "ctrl", a.seed, 0)
    missing = [n for n, r in (("wide", W), ("ctrl", C), ("wide step 0", W0), ("ctrl step 0", C0)) if r is None]
    if missing:
        print(json.dumps({"error": f"missing test records: {missing}", "root": str(root), "step": a.step}))
        sys.exit(2)

    rep = {"root": str(root), "seed": a.seed, "step": a.step, "final": bool(a.final), "accuracy": {}}
    for w in WIN:
        for name, ts in (("tier1", TIER1), ("tier2", TIER2)) + tuple((t, (t,)) for t in TIER1 + TIER2):
            aw, ac = mean_acc(W, ts, w), mean_acc(C, ts, w)
            rep["accuracy"][f"{name}/{w}"] = {
                "wide": aw, "ctrl": ac, "untouched": mean_acc(C0, ts, w), "diff_points": 100 * (aw - ac),
                "diff_points_ci95": paired_ci(W, C, ts, w, a.B if name == "tier1" else 2000)}
    D = rep["accuracy"]["tier1/w4"]["diff_points"]
    ctrl4 = rep["accuracy"]["tier1/w4"]["ctrl"]
    gate_task = W["negeig_stats"]["beta_gt1_frac_per_layer"]
    gate_max = max(gate_task) if gate_task else 0.0
    rep["gate"] = {"wide_test_beta_gt1_max_layer": gate_max,
                   "wide_test_beta_gt1_mean": float(np.mean(gate_task)) if gate_task else None,
                   "ctrl_has_gate": bool(C["negeig_stats"]["beta_gt1_frac_per_layer"])}
    # G0 on the test set: at step 0 both arms are the untouched model, so their answers must agree
    n_diff = sum(int((np.array(W0["tasks"][k]) != np.array(C0["tasks"][k])).sum())
                 for k in W0["tasks"] if k.endswith("/seq"))
    rep["step0_identity"] = {"seq_accuracy_entries_differing": n_diff, "identical": n_diff == 0}
    cw, cc = configs(root / f"wide_s{a.seed}"), configs(root / f"ctrl_s{a.seed}")
    diffs = {}
    if cw and cc:
        for k in sorted((set(cw[-1]) | set(cc[-1])) - CONFIG_IGNORE):
            if cw[-1].get(k) != cc[-1].get(k):
                diffs[k] = [cw[-1].get(k), cc[-1].get(k)]
    rep["config_differences"] = diffs if (cw and cc) else "config records missing"
    vw, vc = val_curve(root / f"wide_s{a.seed}"), val_curve(root / f"ctrl_s{a.seed}")
    rep["switch_step_val_w1_ge_0.95"] = {"wide": switch_steps(vw), "ctrl": switch_steps(vc)}
    last = {arm: (c[-1] if c else None) for arm, c in (("wide", vw), ("ctrl", vc))}
    rep["last_val"] = {arm: ({k: r.get(k) for k in ("step", "tier1", "kl_to_untouched", "replay_nll", "gate_rate_text",
                                                     "gate_rate_tasks", "gate_rate_tasks_max_layer")} if r else None)
                       for arm, r in last.items()}
    rep["heldout_nll"] = {"wide": float(np.mean(W["nll"])), "ctrl": float(np.mean(C["nll"])),
                          "untouched": float(np.mean(C0["nll"]))}
    rep["anchors"] = ANCHORS

    reasons = []
    if not (cw and cc) or diffs:
        verdict = "REVIEW"
        reasons.append(f"arm configs differ or are missing: {diffs or 'missing'}")
    elif D >= BAR_POINTS and gate_max >= GATE_USED:
        verdict = "GO"
        reasons.append(f"tier-1 4x: wide beats ctrl by {D:.1f} points (bar {BAR_POINTS:.0f}), gate used "
                       f"(busiest layer {100 * gate_max:.1f}% beta > 1)")
    elif D >= BAR_POINTS:
        verdict = "REVIEW"
        reasons.append(f"D = {D:.1f} points but the gate is unused (busiest layer {100 * gate_max:.2f}% < 1%)")
    elif ctrl4 >= CTRL_SOLVES:
        verdict = "KILL"
        reasons.append(f"D = {D:.1f} < {BAR_POINTS:.0f} and ctrl reaches {ctrl4:.3f} at 4x on its own: the 27B "
                       f"extrapolates tier-1 without the gate")
    elif not a.final:
        verdict = "EXTEND"
        reasons.append(f"D = {D:.1f} < {BAR_POINTS:.0f} at step {a.step}, ctrl at {ctrl4:.3f}: continue both arms to the "
                       f"next rung")
    else:
        verdict = "KILL"
        reasons.append(f"D = {D:.1f} < {BAR_POINTS:.0f} at the final rung (step {a.step})")
    if not rep["step0_identity"]["identical"]:
        reasons.append(f"note: the step-0 test answers differ between arms in {n_diff} entries (G0 expects 0)")
    rep["verdict"] = verdict
    rep["reasons"] = reasons
    out = root / f"decision_step{a.step:06d}.json"
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(rep, indent=1))
    tmp.replace(out)
    acc = rep["accuracy"]
    print(json.dumps({"verdict": verdict, "reasons": reasons,
                      "tier1": {w: {k: round(acc[f'tier1/{w}'][k], 3) for k in ("wide", "ctrl", "untouched")}
                                for w in WIN},
                      "per_task_w4": {t: {k: round(acc[f'{t}/w4'][k], 3) for k in ("wide", "ctrl")} for t in TIER1},
                      "D_points": round(D, 1), "D_ci95": [round(x, 1) for x in acc["tier1/w4"]["diff_points_ci95"]],
                      "switch": rep["switch_step_val_w1_ge_0.95"], "out": str(out)}))


if __name__ == "__main__":
    main()
