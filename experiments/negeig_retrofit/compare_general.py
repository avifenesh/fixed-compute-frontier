#!/usr/bin/env python3
"""Paired comparison of general evals (kill gate 2): MMLU-Pro and HumanEval, per item.

Usage: compare_general.py A_run_dir B_run_dir [--base BASE_run_dir]
Reports each run's accuracy, the paired difference B - A in points with a 95% paired-bootstrap interval,
and the items that flipped in each direction. The card's gate: kill if wide drops more than 1 point vs ctrl.
"""

import argparse
import json
import os
import random


def load(d):
    g = json.load(open(os.path.join(d, "general.json")))
    return g["mmlu_items"], g["humaneval_items"]


def paired(a, b, n_boot=4000, seed=0):
    keys = sorted(set(a) & set(b), key=str)
    xa = [float(a[k]) for k in keys]
    xb = [float(b[k]) for k in keys]
    d = [y - x for x, y in zip(xa, xb)]
    n = len(d)
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {
        "n": n, "a": 100 * sum(xa) / n, "b": 100 * sum(xb) / n, "diff": 100 * sum(d) / n,
        "lo": 100 * boots[int(0.025 * n_boot)], "hi": 100 * boots[int(0.975 * n_boot)],
        "a_only": sum(1 for x, y in zip(xa, xb) if x > y), "b_only": sum(1 for x, y in zip(xa, xb) if y > x),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--base")
    ap.add_argument("--out")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    res = {"a": args.a, "b": args.b}
    pairs = [("A", args.a, A), ("B", args.b, B)]
    for name, idx in (("mmlu_pro", 0), ("humaneval", 1)):
        r = paired(A[idx], B[idx])
        res[name] = r
        print(f"{name:10s} n={r['n']:5d}  A {r['a']:6.2f}  B {r['b']:6.2f}  B-A {r['diff']:+6.2f} "
              f"[{r['lo']:+.2f}, {r['hi']:+.2f}]  flips A-only {r['a_only']} B-only {r['b_only']}")
        if args.base:
            Z = load(args.base)
            for tag, _, X in pairs:
                rb = paired(Z[idx], X[idx])
                res[f"{name}_{tag}_vs_base"] = rb
                print(f"{'':10s} {tag} vs base: {rb['diff']:+6.2f} [{rb['lo']:+.2f}, {rb['hi']:+.2f}]")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
