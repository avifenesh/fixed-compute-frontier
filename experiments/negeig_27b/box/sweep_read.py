#!/usr/bin/env python3
"""Read the LR sweep the way results/negeig-27b/stageA-log.md pre-registered it, from train27.py's log.jsonl files.
Reads only; decides nothing by itself that the owner has not already written down. Runs anywhere (stdlib only), usually on
the rig after both boxes' sweep dirs were pulled next to each other:

  sweep_read.py --runs DIR --tag sweep-lr1e-4 --tag sweep-lr1e-3 [--seed 0] [--steps 250,300] [--json OUT]

DIR holds <tag>/{wide,ctrl}_s<seed>/log.jsonl (the layout under $RUNS on a box). One --tag reads one pair (no choice is
made); two --tags read the sweep (the higher --lr in the config event is "high"). The thresholds are the pre-registered ones:

  safety, each arm, mean of the evals at --steps (default 250,300):
      kl_to_untouched <= 0.08   replay_nll <= untouched + 0.05 (untouched 1.9694, --untouched-replay-nll)
      tool_token_agree >= 0.95  no training loss above 3x the running median of that loss's earlier values (<= last step)
  signal, wide against ctrl at the same LR, any one of:
      s1:len256:le256 gap >= 5 points (0.05)
      wide nll:all at least 10% (relative) below ctrl AND that gap wider than 100 steps earlier (evals at --steps minus 100)
      wide gate_rate_s1 >= 1.5 x wide gate_rate_text
      (s1:len512:gt256 is printed beside the signal, not part of it)
  choice (two tags): high if it passes safety and shows signal; low if high fails safety and low passes; both pass and
      neither shows signal: extend both pairs to 600 steps and read again with --steps 550,600. Every other combination
      (both fail safety, high safe without signal while low shows signal or fails safety) is printed as "owner" with the
      reason, because the rule does not cover it.
  abort (every eval up to the last --steps value, not averaged): kl_to_untouched >= 0.15 on two consecutive evals, tool
      agreement < 0.90, or a recall:<len> value more than 10 points under the step-0 eval's. Reported per arm with the
      first breach step.

A step that was logged twice (a run rewound after an abort) counts once, the last record wins. A torn last line is ignored.
Exit codes: 0 read (the last line is SWEEP_READ ...), 1 an input is missing (log, arm, eval step, key), 2 bad command line.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

KL_MAX = 0.08
NLL_MARGIN = 0.05
AGREE_MIN = 0.95
SPIKE = 3.0
GAP_MIN = 0.05
NLL_REL = 0.10
GATE_RATIO = 1.5
ABORT_KL = 0.15
ABORT_AGREE = 0.90
ABORT_RECALL = 0.10
SPIKE_MIN_HISTORY = 3
ARMS = ("wide", "ctrl")


class Missing(Exception):
    pass


def read_log(path: Path) -> dict:
    if not path.is_file():
        raise Missing(f"no log: {path}")
    evals, steps, config = {}, {}, None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line from a kill
        ev = r.get("event")
        if ev == "eval":
            evals[r["step"]] = r
        elif ev == "step":
            steps[r["step"]] = r
        elif ev == "config":
            config = r
    return {"evals": evals, "steps": steps, "config": config, "path": str(path)}


def mean_at(log: dict, key: str, steps: list[int], label: str) -> float:
    vals = []
    for s in steps:
        rec = log["evals"].get(s)
        if rec is None:
            raise Missing(f"{label}: no eval at step {s} in {log['path']}")
        v = rec.get(key)
        if v is None:
            raise Missing(f"{label}: eval at step {s} has no {key}")
        vals.append(float(v))
    return sum(vals) / len(vals)


def loss_spike(log: dict, last_step: int) -> tuple[float, str]:
    """Largest ratio of a logged loss to the median of the same loss's earlier logged values, and where."""
    worst, where = 0.0, ""
    by_key: dict[str, list[tuple[int, float]]] = {}
    for s in sorted(log["steps"]):
        if s > last_step:
            continue
        for k, v in (log["steps"][s].get("loss") or {}).items():
            if v is not None:
                by_key.setdefault(k, []).append((s, float(v)))
    for k, series in by_key.items():
        for i in range(SPIKE_MIN_HISTORY, len(series)):
            med = statistics.median(v for _, v in series[:i])
            if med > 0 and series[i][1] / med > worst:
                worst, where = series[i][1] / med, f"{k} at step {series[i][0]}"
    return worst, where


def safety(log: dict, steps: list[int], label: str, untouched_nll: float) -> dict:
    kl = mean_at(log, "kl_to_untouched", steps, label)
    nll = mean_at(log, "replay_nll", steps, label)
    agree = mean_at(log, "tool_token_agree", steps, label)
    spike, where = loss_spike(log, max(steps))
    checks = {
        "kl_to_untouched": {"value": kl, "limit": KL_MAX, "ok": kl <= KL_MAX},
        "replay_nll": {"value": nll, "limit": untouched_nll + NLL_MARGIN, "ok": nll <= untouched_nll + NLL_MARGIN},
        "tool_token_agree": {"value": agree, "limit": AGREE_MIN, "ok": agree >= AGREE_MIN},
        "loss_spike_ratio": {"value": spike, "limit": SPIKE, "ok": spike <= SPIKE, "where": where},
    }
    return {"checks": checks, "pass": all(c["ok"] for c in checks.values())}


def signal(wide: dict, ctrl: dict, steps: list[int], tag: str) -> dict:
    out = {}
    g = mean_at(wide, "s1:len256:le256", steps, f"{tag}/wide") - mean_at(ctrl, "s1:len256:le256", steps, f"{tag}/ctrl")
    out["len256_le256_gap"] = {"value": g, "limit": GAP_MIN, "ok": g >= GAP_MIN}
    wn, cn = mean_at(wide, "nll:all", steps, f"{tag}/wide"), mean_at(ctrl, "nll:all", steps, f"{tag}/ctrl")
    rel = (cn - wn) / cn if cn else 0.0
    earlier = [s - 100 for s in steps]
    widening = None
    try:
        gap_now = cn - wn
        gap_before = mean_at(ctrl, "nll:all", earlier, f"{tag}/ctrl") - mean_at(wide, "nll:all", earlier, f"{tag}/wide")
        widening = gap_now > gap_before
    except Missing:
        widening = None  # no evals 100 steps earlier: not demonstrated
    out["answer_nll"] = {"wide": wn, "ctrl": cn, "relative_below_ctrl": rel, "limit": NLL_REL, "widening": widening,
                         "ok": rel >= NLL_REL and widening is True}
    gs, gt = mean_at(wide, "gate_rate_s1", steps, f"{tag}/wide"), mean_at(wide, "gate_rate_text", steps, f"{tag}/wide")
    ratio = (gs / gt) if gt > 0 else (float("inf") if gs > 0 else 0.0)
    out["gate_rate"] = {"s1": gs, "text": gt, "ratio": ratio, "limit": GATE_RATIO, "ok": ratio >= GATE_RATIO}
    out["beside_len512_gt256_gap"] = (mean_at(wide, "s1:len512:gt256", steps, f"{tag}/wide")
                                      - mean_at(ctrl, "s1:len512:gt256", steps, f"{tag}/ctrl"))
    out["present"] = any(out[k]["ok"] for k in ("len256_le256_gap", "answer_nll", "gate_rate"))
    return out


def abort_check(log: dict, last_step: int) -> list[str]:
    reasons = []
    evals = [(s, log["evals"][s]) for s in sorted(log["evals"]) if s <= last_step]
    prev_hot = None
    for s, r in evals:
        kl = r.get("kl_to_untouched")
        hot = kl is not None and kl >= ABORT_KL
        if hot and prev_hot is not None:
            reasons.append(f"kl_to_untouched >= {ABORT_KL} on evals {prev_hot} and {s}")
            break
        prev_hot = s if hot else None
    for s, r in evals:
        a = r.get("tool_token_agree")
        if a is not None and a < ABORT_AGREE:
            reasons.append(f"tool_token_agree {a:.3f} < {ABORT_AGREE} at step {s}")
            break
    base = log["evals"].get(0)
    if base is not None:
        for key in sorted(k for k in base if k.startswith("recall:")):
            b0 = base.get(key)
            if b0 is None:
                continue
            for s, r in evals:
                v = r.get(key)
                if v is not None and b0 - v > ABORT_RECALL:
                    reasons.append(f"{key} {v:.3f} is {100 * (b0 - v):.0f} points under step 0 ({b0:.3f}) at step {s}")
                    break
    return reasons


def read_tag(runs: Path, tag: str, seed: int, steps: list[int], untouched: float) -> dict:
    logs = {a: read_log(runs / tag / f"{a}_s{seed}" / "log.jsonl") for a in ARMS}
    cfg = logs["wide"]["config"] or {}
    lr = (cfg.get("args") or {}).get("lr")
    if lr is None:
        raise Missing(f"{tag}: no config event with args.lr in {logs['wide']['path']}")
    res = {"tag": tag, "lr": float(lr), "seed": seed,
           "safety": {a: safety(logs[a], steps, f"{tag}/{a}", untouched) for a in ARMS},
           "signal": signal(logs["wide"], logs["ctrl"], steps, tag),
           "abort": {a: abort_check(logs[a], max(steps)) for a in ARMS},
           "step0_replay_nll": {a: (logs[a]["evals"].get(0) or {}).get("replay_nll") for a in ARMS}}
    res["safety_pass"] = all(res["safety"][a]["pass"] for a in ARMS)
    return res


def choose(tags: list[dict]) -> tuple[str, str]:
    if len(tags) != 2:
        return "n/a", "one tag read: the choice needs both"
    low, high = sorted(tags, key=lambda t: t["lr"])
    hs, hp, ls, lp = high["safety_pass"], high["signal"]["present"], low["safety_pass"], low["signal"]["present"]
    if hs and hp:
        return high["tag"], f"{high['tag']} passes safety and shows signal"
    if not hs and not ls:
        return "owner", f"{high['tag']} and {low['tag']} both fail safety: the rule picks the lower LR only when it is safe"
    if not hs:
        return low["tag"], f"{high['tag']} fails safety, {low['tag']} passes"
    if not hp and ls and not lp:
        return "extend", (f"both pass safety, neither shows signal: extend both pairs to 600 steps and read again with "
                          f"--steps 550,600")
    return "owner", (f"{high['tag']} passes safety without signal while {low['tag']} has safety "
                     f"{'pass' if ls else 'fail'} and signal {'present' if lp else 'absent'}: the rule does not cover this")


def fmt_checks(d: dict, order: tuple[str, ...]) -> str:
    parts = []
    for k in order:
        c = d[k]
        v = c.get("value")
        parts.append(f"{k} {v:.4f} ({'ok' if c['ok'] else 'FAIL'}, limit {c['limit']:.4f})")
    return "; ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True, help="dir holding <tag>/<arm>_s<seed>/log.jsonl")
    ap.add_argument("--tag", action="append", required=True, help="one or two sweep tags")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", default="250,300", help="eval steps to average (default 250,300; 550,600 for the extension)")
    ap.add_argument("--untouched-replay-nll", type=float, default=1.9694)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    try:
        steps = sorted({int(x) for x in a.steps.split(",") if x.strip()})
    except ValueError:
        print(f"sweep_read: --steps wants integers, got {a.steps!r}", file=sys.stderr)
        return 2
    if not steps or len(a.tag) > 2 or len(set(a.tag)) != len(a.tag):
        print("sweep_read: give one or two distinct --tag values and at least one step", file=sys.stderr)
        return 2
    try:
        tags = [read_tag(Path(a.runs), t, a.seed, steps, a.untouched_replay_nll) for t in a.tag]
    except Missing as e:
        print(f"sweep_read: {e}", file=sys.stderr)
        return 1
    choice, why = choose(tags)
    for t in tags:
        print(f"== {t['tag']}  lr {t['lr']:g}  seed {t['seed']}  evals averaged at {steps}")
        for arm in ARMS:
            s = t["safety"][arm]
            print(f"  safety {arm}: {'PASS' if s['pass'] else 'FAIL'}  "
                  + fmt_checks(s["checks"], ("kl_to_untouched", "replay_nll", "tool_token_agree", "loss_spike_ratio")))
            w = s["checks"]["loss_spike_ratio"]["where"]
            if w:
                print(f"    largest loss ratio at {w}")
            n0 = t["step0_replay_nll"][arm]
            if n0 is not None and abs(n0 - a.untouched_replay_nll) > 0.01:
                print(f"    NOTE step-0 replay_nll {n0:.4f} differs from the untouched {a.untouched_replay_nll} by more than "
                      f"0.01: check that the replay set is the one the bar was measured on")
        sg = t["signal"]
        g, n, r = sg["len256_le256_gap"], sg["answer_nll"], sg["gate_rate"]
        print(f"  signal: {'PRESENT' if sg['present'] else 'absent'}")
        print(f"    s1:len256:le256 gap wide-ctrl {100 * g['value']:+.1f} points ({'ok' if g['ok'] else 'no'}, need {100 * GAP_MIN:.0f})")
        print(f"    answer nll wide {n['wide']:.4f} ctrl {n['ctrl']:.4f}: {100 * n['relative_below_ctrl']:.1f}% below, "
              f"widening {n['widening']} ({'ok' if n['ok'] else 'no'}, need {100 * NLL_REL:.0f}% and widening)")
        print(f"    gate rate s1 {r['s1']:.4f} text {r['text']:.4f}: ratio {r['ratio']:.2f} ({'ok' if r['ok'] else 'no'}, need {GATE_RATIO})")
        print(f"    beside: s1:len512:gt256 gap wide-ctrl {100 * sg['beside_len512_gt256_gap']:+.1f} points")
        for arm in ARMS:
            ab = t["abort"][arm]
            print(f"  abort {arm}: {'; '.join(ab) if ab else 'none'}")
    any_abort = [f"{t['tag']}/{arm}" for t in tags for arm in ARMS if t["abort"][arm]]
    print(f"{'ABORT_RULE_HIT ' + ','.join(any_abort) if any_abort else 'no abort rule hit'}")
    print(f"choice: {choice}  ({why})")
    if a.json:
        Path(a.json).write_text(json.dumps({"steps": steps, "tags": tags, "choice": choice, "why": why}, indent=1) + "\n")
    print(f"SWEEP_READ choice={choice} steps={','.join(map(str, steps))} abort={'yes' if any_abort else 'no'} "
          + " ".join(f"{t['tag']}:safety={'pass' if t['safety_pass'] else 'fail'},signal={'yes' if t['signal']['present'] else 'no'}" for t in tags))
    return 0


if __name__ == "__main__":
    sys.exit(main())
