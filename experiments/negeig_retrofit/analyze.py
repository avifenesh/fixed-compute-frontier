#!/usr/bin/env python3
"""Analysis for the negeig retrofit, judged by the idea card's kill gates (as revised in the assessment).

Kill if any of:
  1. wide beats ctrl by less than 10 accuracy points on the tier-1 tasks at 4x length
     (steps 65-256 of the 1,024-step eval set; mean over parity, swap, codeswap and seeds);
  2. general evals drop more than 1 point (MMLU-Pro, HumanEval pass@1; mean over seeds), when present;
  3. the widened range is never used (beta > 1 on under 1% of tokens in every layer).
Reported, not gated: per-task and 16x differences with paired bootstrap intervals, tier-2 (full S5,
mod-3 counting: one reflection per token cannot express them), locked seeds, held-out NLL.
"""
import argparse
import glob
import json
import os

import numpy as np

TIER1 = ("parity", "swap", "codeswap")
TIER2 = ("s5full", "z3")
WIN = {"w1": 0, "w4": 1, "w16": 2}
CHANCE = {"parity": 0.5, "swap": 0.2, "codeswap": 0.2, "s5full": 0.2, "z3": 1 / 3}


def load(root, arm):
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, f"{arm}_s*", "result.json"))):
        seed = int(os.path.basename(os.path.dirname(f)).split("_s")[1])
        runs[seed] = json.load(open(f))
        g = os.path.join(os.path.dirname(f), "general.json")
        if os.path.exists(g):
            runs[seed]["general"] = json.load(open(g))
    return runs


def win(run, task, w):
    return np.array([row[WIN[w]] for row in run["tasks"][f"{task}@1024/win"]], dtype=float)


def acc(runs, seeds, idx, w, tasks):
    return float(np.mean([np.mean([win(runs[s], t, w)[idx[k]].mean() for k, s in enumerate(seeds)]) for t in tasks]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=10_000)
    a = ap.parse_args()
    W, C = load(a.root, "wide"), load(a.root, "ctrl")
    seeds = sorted(set(W) & set(C))
    n = len(W[seeds[0]]["tasks"]["parity@1024/win"])
    ident = [np.arange(n)] * len(seeds)
    rng = np.random.default_rng(20260929)
    rep = {"seeds": seeds, "accuracy": {}}
    for w in ("w1", "w4", "w16"):
        for name, tasks in (("tier1", TIER1), ("tier2", TIER2)) + tuple((t, (t,)) for t in TIER1 + TIER2):
            aw, ac = acc(W, seeds, ident, w, tasks), acc(C, seeds, ident, w, tasks)
            boot = []
            for _ in range(a.B if name == "tier1" else 0):
                ss = list(rng.choice(seeds, size=len(seeds), replace=True))
                idx = [rng.integers(0, n, n) for _ in ss]
                boot.append(acc(W, ss, idx, w, tasks) - acc(C, ss, idx, w, tasks))
            rep["accuracy"][f"{name}/{w}"] = {
                "wide": aw, "ctrl": ac, "diff_points": 100 * (aw - ac),
                **({"diff_points_ci95": [100 * float(np.percentile(boot, 2.5)), 100 * float(np.percentile(boot, 97.5))]}
                   if boot else {}),
                **({"chance": CHANCE[name]} if name in CHANCE else {}),
            }
    rep["per_seed_tier1_w4"] = {s: {"wide": acc(W, [s], [np.arange(n)], "w4", TIER1),
                                    "ctrl": acc(C, [s], [np.arange(n)], "w4", TIER1)} for s in seeds}
    rep["locked"] = {w: {t: {"wide": int(sum(win(W[s], t, w).mean() >= 0.99 for s in seeds)),
                             "ctrl": int(sum(win(C[s], t, w).mean() >= 0.99 for s in seeds))} for t in TIER1 + TIER2}
                     for w in ("w4", "w16")}
    nw = np.mean([W[s]["nll"] for s in seeds])
    nc = np.mean([C[s]["nll"] for s in seeds])
    rep["heldout_nll"] = {"wide": float(nw), "ctrl": float(nc), "rel_diff": float(nw / nc - 1)}
    if 100 in C:
        rep["heldout_nll"]["untouched_base"] = float(np.mean(C[100]["nll"]))
    gt1 = [max(W[s]["negeig_stats"]["beta_gt1_frac_per_layer"]) for s in seeds]
    rep["gate_use"] = {"max_layer_beta_gt1_frac_per_seed": gt1}

    kills = []
    d4 = rep["accuracy"]["tier1/w4"]["diff_points"]
    if d4 < 10:
        kills.append(f"tier-1 4x difference {d4:.1f} points < 10")
    general = {}
    for key in ("mmlu_pro_acc", "humaneval_pass1"):
        vals_w = [W[s]["general"][key] for s in seeds if "general" in W[s] and key in W[s]["general"]]
        vals_c = [C[s]["general"][key] for s in seeds if "general" in C[s] and key in C[s]["general"]]
        if vals_w and vals_c:
            dw = 100 * (np.mean(vals_w) - np.mean(vals_c))
            general[key] = {"wide": float(np.mean(vals_w)), "ctrl": float(np.mean(vals_c)), "diff_points": float(dw),
                            "n_seeds": [len(vals_w), len(vals_c)]}
            if 100 in C and "general" in C[100] and key in C[100]["general"]:
                general[key]["untouched_base"] = C[100]["general"][key]
            if dw < -1:
                kills.append(f"{key} drops {-dw:.2f} points > 1")
    rep["general_evals"] = general or "not run yet"
    if max(gt1) < 0.01:
        kills.append("widened range never used")
    rep["verdict"] = {"kill": bool(kills), "reasons": kills,
                      "general_evals_complete": bool(general) and len(general) == 2}
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("verdict", "general_evals", "heldout_nll", "gate_use")}, indent=1))
    print(json.dumps({k: v for k, v in rep["accuracy"].items() if k.startswith(("tier1", "tier2"))}, indent=1))


if __name__ == "__main__":
    main()
