#!/usr/bin/env python3
"""Turn the acceptance smoke runs into one receipt (stdlib only, CPU only). accept.sh calls it; the tests feed it the
stub trainer's output.

  smoke_report.py --four wide=DIR --four ctrl=DIR --eight DIR --probe DIR --out smoke_report.json --env-out smoke.env

Inputs are train27.py output dirs (log.jsonl, profile_rank0.json, ckpt_*.pt, complete.json):
  --four NAME=DIR   a 4-rank arm of the two-arm smoke (30 steps, evals and save at 30, mem_probe 1, profile_step 10)
  --eight DIR       the 8-rank timing run of one arm (20 steps, no evals, no save)
  --probe DIR       the grad_ckpt probe: mem_probe 2 with --grad_ckpt 0, every pool's longest row
What it reports
  tok/s       steady rate from the LAST two step events of each run. The trainer's tok_per_s counts evals and saves and the
              profiler export, so the steady rate uses train_tok_per_s (step time only): tokens = tok_per_s * elapsed_s,
              t_train = tokens / train_tok_per_s, rate = delta tokens / delta t_train. The last window is the one after
              the profiler window (steps 11 to 21 hold the profiled steps).
  spread      per-step max/mean of rank_compute_s (step 1 excluded: warmup) and of lpt_load_tokens
  memory      peak_mem_gib of the run, and the mem_probe peak per pool
  profile     device_share of gdn, attention and gemm from profile_rank0.json (rank 0, three steps)
  scaling     8-rank rate over the 4-rank rate of the same arm, and whether the global batch was the same (the summed
              lpt_load_tokens at steps 1 and 11 must match exactly: the batch does not depend on the world size)
  grad_ckpt   decision from the probe: off when the largest probed peak is at most --max-gib, on otherwise, undecided
              when a pool in --need-pools was not probed (the decision needs the longest tool row, not only S1).
              --probe-oom (accept.sh saw a CUDA out-of-memory in the probe) forces on and is no problem.
Exit 0 when the report has no problems, 1 otherwise (the problems are printed and listed in the JSON).
"""
import argparse
import json
import sys
from pathlib import Path

PLAN_APPROVE_TOK_S_PER_GPU = 3800.0   # PLAN table, 8x B300: at or above approves the envelope
PLAN_CORE_TOK_S_PER_GPU = 1900.0      # between this and the approve line: core only; below: report to the owner


def read_events(d):
    """Events of the newest invocation (after the last config event). A torn last line is skipped."""
    p = Path(d) / "log.jsonl"
    if not p.is_file():
        return None
    evs = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evs.append(json.loads(line))
        except ValueError:
            continue
    last_cfg = max((i for i, e in enumerate(evs) if e.get("event") == "config"), default=0)
    return evs[last_cfg:]


def cumulative(ev):
    """(tokens, t_train) at a step event."""
    tokens = ev["tok_per_s"] * ev["elapsed_s"]
    t_train = tokens / ev["train_tok_per_s"] if ev.get("train_tok_per_s") else None
    return tokens, t_train


def max_over_mean(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    return round(max(xs) / m, 4) if m > 0 else None


def run_summary(name, d, need_complete, problems):
    """One train27.py output dir -> dict; appends to problems."""
    s = {"name": name, "dir": str(d)}
    evs = read_events(d)
    if evs is None:
        problems.append(f"{name}: no log.jsonl in {d}")
        return s
    cfg = next((e for e in evs if e.get("event") == "config"), {})
    steps = [e for e in evs if e.get("event") == "step"]
    s["world"] = cfg.get("world")
    s["step_events"] = [e["step"] for e in steps]
    s["complete"] = (Path(d) / "complete.json").is_file()
    if need_complete and not s["complete"]:
        problems.append(f"{name}: no complete.json (the run did not finish)")
    if len(steps) < 2:
        problems.append(f"{name}: {len(steps)} step event(s), need at least 2 for a steady rate")
        return s
    a, b = steps[-2], steps[-1]
    (tok_a, t_a), (tok_b, t_b) = cumulative(a), cumulative(b)
    if t_a is None or t_b is None or t_b <= t_a or b["step"] - a["step"] < 5:
        problems.append(f"{name}: last two step events ({a['step']}, {b['step']}) give no usable window")
    else:
        rate = (tok_b - tok_a) / (t_b - t_a)
        world = s["world"] or len(b.get("rank_compute_s", [])) or 1
        s["window_steps"] = [a["step"], b["step"]]
        s["tokens_per_step"] = round((tok_b - tok_a) / (b["step"] - a["step"]), 1)
        s["steady_tok_per_s"] = round(rate, 1)
        s["steady_tok_per_s_per_gpu"] = round(rate / world, 1)
        s["previous_window_tok_per_s"] = None
        if len(steps) >= 3 and t_a > 0:
            (tok_0, t_0) = cumulative(steps[-3])
            if t_0 is not None and t_a > t_0:
                s["previous_window_tok_per_s"] = round((tok_a - tok_0) / (t_a - t_0), 1)
        s["cumulative_tok_per_s_incl_evals_saves"] = b["tok_per_s"]
    # step 1 pays the warmup: spread from the later events
    later = [e for e in steps if e["step"] > 1]
    s["rank_compute_spread_max_over_mean"] = max(filter(None, (max_over_mean(e.get("rank_compute_s", [])) for e in later)),
                                                 default=None)
    s["rank_compute_spread_last"] = max_over_mean(steps[-1].get("rank_compute_s", []))
    s["lpt_load_spread_last"] = max_over_mean(steps[-1].get("lpt_load_tokens", []))
    s["peak_mem_gib"] = max((e.get("peak_mem_gib") or 0 for e in steps), default=None)
    s["global_batch_tokens"] = {str(e["step"]): sum(e.get("lpt_load_tokens", [])) for e in steps}
    probes = [e for e in evs if e.get("event") == "mem_probe"]
    s["mem_probe"] = {e["pool"]: {"tokens": e["tokens"], "peak_gib": e["peak_gib"], "grad_ckpt": e.get("grad_ckpt"),
                                  "s": e.get("s")} for e in probes}
    prof = Path(d) / "profile_rank0.json"
    if prof.is_file():
        share = json.loads(prof.read_text()).get("device_share", {})
        s["profile_device_share"] = {k: round(v, 4) for k, v in share.items()}
        s["profile_device_share"]["other"] = round(max(0.0, 1.0 - sum(share.values())), 4)
    s["evals"] = [e.get("step") for e in evs if e.get("event") == "eval"]
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--four", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--eight", metavar="DIR")
    ap.add_argument("--probe", metavar="DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-out")
    ap.add_argument("--max-gib", type=float, default=223.0, help="grad_ckpt off if the largest probed peak is at most this")
    ap.add_argument("--need-pools", default="s1,chat,tool", help="pools the probe must have seen for a grad_ckpt decision")
    ap.add_argument("--min-tok-s", type=float, default=None, help="floor on the 4-rank steady rate of each arm")
    ap.add_argument("--baseline-tok-s", type=float, default=None, help="reference rate of one 4-GPU run, for the ratio")
    ap.add_argument("--stage", default="", help="note stored in the report (for example replay=present)")
    ap.add_argument("--probe-oom", action="store_true",
                    help="the grad_ckpt=0 probe died with a CUDA out-of-memory: that is the answer (decision on), not a problem")
    a = ap.parse_args()

    problems = []
    rep = {"schema": 1, "note": a.stage, "four_rank": {}, "eight_rank": None, "scaling": None, "grad_ckpt": None}

    for spec in a.four:
        name, _, d = spec.partition("=")
        r = run_summary(f"4-rank {name}", d, True, problems)
        if not (Path(d) / "profile_rank0.json").is_file():
            problems.append(f"4-rank {name}: no profile_rank0.json (--profile_step did not run)")
        if not r.get("mem_probe"):
            problems.append(f"4-rank {name}: no mem_probe event (--mem_probe 1 did not run)")
        if not list(Path(d).glob("ckpt_[0-9]*.pt")):
            problems.append(f"4-rank {name}: no ckpt_*.pt (the checkpoint writer did not run)")
        if a.min_tok_s is not None and r.get("steady_tok_per_s") is not None and r["steady_tok_per_s"] < a.min_tok_s:
            problems.append(f"4-rank {name}: {r['steady_tok_per_s']} tok/s is below the floor {a.min_tok_s}")
        rep["four_rank"][name] = r

    if a.eight:
        rep["eight_rank"] = run_summary("8-rank", a.eight, True, problems)

    # 8 ranks against 4 ranks of the same arm (the 8-rank run is the first arm named in --four unless it has the same name)
    four = rep["four_rank"]
    e8 = rep["eight_rank"]
    if e8 and four and e8.get("steady_tok_per_s"):
        ref_name = next(iter(four))
        ref = four[ref_name]
        if ref.get("steady_tok_per_s"):
            speed = e8["steady_tok_per_s"] / ref["steady_tok_per_s"]
            tps = [ref.get("tokens_per_step"), e8.get("tokens_per_step")]
            mismatch = abs(tps[0] - tps[1]) / max(tps) > 0.10 if all(tps) else None
            shared = sorted(set(ref.get("global_batch_tokens", {})) & set(e8.get("global_batch_tokens", {})), key=int)
            same = all(ref["global_batch_tokens"][k] == e8["global_batch_tokens"][k] for k in shared) if shared else None
            if same is False:
                problems.append("scaling: the global batch differs between the 4-rank and the 8-rank run (summed "
                                "lpt_load_tokens at the same step): the batch must not depend on the world size")
            hint = ("4-box layout favored (8 GPUs a run)" if speed >= 1.7 else
                    "two boxes (4 GPUs a run) favored" if speed < 1.5 else "between 1.5x and 1.7x: owner decides")
            rep["scaling"] = {"reference_arm": ref_name, "speedup_8_vs_4": round(speed, 3), "efficiency": round(speed / 2, 3),
                              "tokens_per_step_4rank_8rank": tps, "window_mismatch": mismatch,
                              "same_global_batch": same, "same_batch_steps_compared": shared, "layout_hint": hint}

    # throughput class on the 4-rank per-GPU mean (PLAN thresholds for 8x B300; informational on other hardware)
    per_gpu = [r["steady_tok_per_s_per_gpu"] for r in four.values() if r.get("steady_tok_per_s_per_gpu")]
    if per_gpu:
        mean = sum(per_gpu) / len(per_gpu)
        cls = ("approve the envelope" if mean >= PLAN_APPROVE_TOK_S_PER_GPU else
               "core only" if mean >= PLAN_CORE_TOK_S_PER_GPU else "report to the owner")
        rep["throughput"] = {"per_gpu_4rank_mean": round(mean, 1), "plan_class_b300_thresholds": cls,
                             "plan_thresholds": [PLAN_CORE_TOK_S_PER_GPU, PLAN_APPROVE_TOK_S_PER_GPU]}
        if a.baseline_tok_s:
            rep["throughput"]["ratio_to_baseline"] = round(
                sum(r["steady_tok_per_s"] for r in four.values() if r.get("steady_tok_per_s")) / len(per_gpu) / a.baseline_tok_s, 3)
        # train-only hours for 700 steps at the window's batch size (evals and saves add to this)
        hrs = {}
        for n, r in list(four.items()) + ([("8-rank", e8)] if e8 else []):
            if r and r.get("steady_tok_per_s") and r.get("tokens_per_step"):
                hrs[n] = round(700 * r["tokens_per_step"] / r["steady_tok_per_s"] / 3600, 2)
        rep["projection_train_only_hours_700_steps"] = hrs

    # grad_ckpt decision from the probe
    if a.probe:
        probe_ev = [e for e in (read_events(a.probe) or []) if e.get("event") == "mem_probe"]
        pools = {e["pool"]: e for e in probe_ev}
        need = [p for p in a.need_pools.split(",") if p]
        missing = [p for p in need if p not in pools]
        wrong = [p for p, e in pools.items() if e.get("grad_ckpt") != 0]
        if not pools and not a.probe_oom:
            problems.append("grad_ckpt probe: no mem_probe event in the probe run")
        if wrong:
            problems.append(f"grad_ckpt probe: pools {wrong} were probed with grad_ckpt on, not off")
        peak = max((e["peak_gib"] for e in pools.values()), default=None)
        if a.probe_oom:
            # checkpointing off did not fit on this card for a row the trainer keeps: on is definitive for every replay state
            fit = ", ".join("%s %s GiB" % (p, e["peak_gib"]) for p, e in pools.items())
            dec, why = "on", "the grad_ckpt=0 probe ran out of CUDA memory" + (" (pools that fit first: %s)" % fit if fit else "")
        elif not pools:
            dec, why = "undecided", "no probe events"
        elif missing:
            dec, why = "undecided", f"pool(s) {missing} not probed (no replay yet): run accept.sh --from train_smoke after self-distillation"
        elif peak <= a.max_gib:
            dec, why = "off", f"largest probed peak {peak} GiB is within {a.max_gib} GiB"
        else:
            dec, why = "on", f"largest probed peak {peak} GiB exceeds {a.max_gib} GiB"
        rep["grad_ckpt"] = {"decision": dec, "reason": why, "max_gib": a.max_gib, "probe_peak_gib": peak,
                            "pools": {p: {"tokens": e["tokens"], "peak_gib": e["peak_gib"]} for p, e in pools.items()},
                            "grad_ckpt_on_peaks_gib": {n: {p: v["peak_gib"] for p, v in r.get("mem_probe", {}).items()}
                                                       for n, r in four.items()}}
    rep["problems"] = problems

    Path(a.out).write_text(json.dumps(rep, indent=2) + "\n")
    if a.env_out:
        lines = []
        if four:
            vals = [r["steady_tok_per_s"] for r in four.values() if r.get("steady_tok_per_s")]
            if vals:
                lines.append(f"SMOKE_TOK_PER_S={round(min(vals), 1)}")
                lines.append(f"SMOKE_TOK_PER_S_PER_GPU={round(min(r['steady_tok_per_s_per_gpu'] for r in four.values() if r.get('steady_tok_per_s_per_gpu')), 1)}")
            pk = [r["peak_mem_gib"] for r in four.values() if r.get("peak_mem_gib") is not None]
            if pk:
                lines.append(f"SMOKE_PEAK_MEM_GIB={max(pk)}")
        if rep["scaling"]:
            lines.append(f"SMOKE8_TOK_PER_S={e8['steady_tok_per_s']}")
            lines.append(f"SMOKE8_SPEEDUP={rep['scaling']['speedup_8_vs_4']}")
        if rep["grad_ckpt"]:
            lines.append(f"GRAD_CKPT_DECISION={rep['grad_ckpt']['decision']}")
            lines.append(f"GRAD_CKPT_PROBE_PEAK_GIB={rep['grad_ckpt']['probe_peak_gib']}")
        Path(a.env_out).write_text("\n".join(lines) + "\n")

    # the human summary
    print("SMOKE REPORT" + (f" ({a.stage})" if a.stage else ""))
    for n, r in four.items():
        print(f"  4-rank {n}: {r.get('steady_tok_per_s')} tok/s steady ({r.get('steady_tok_per_s_per_gpu')} per GPU), "
              f"window steps {r.get('window_steps')}, {r.get('tokens_per_step')} tok/step, peak {r.get('peak_mem_gib')} GiB, "
              f"compute spread max/mean {r.get('rank_compute_spread_max_over_mean')}, lpt spread {r.get('lpt_load_spread_last')}")
        if r.get("mem_probe"):
            print("    mem_probe (grad_ckpt on): " + ", ".join(f"{p} {v['tokens']} tok {v['peak_gib']} GiB" for p, v in r["mem_probe"].items()))
        if r.get("profile_device_share"):
            print("    profile share: " + ", ".join(f"{k} {v:.0%}" for k, v in r["profile_device_share"].items()))
    if e8:
        print(f"  8-rank: {e8.get('steady_tok_per_s')} tok/s steady ({e8.get('steady_tok_per_s_per_gpu')} per GPU), "
              f"window steps {e8.get('window_steps')}, peak {e8.get('peak_mem_gib')} GiB, spread {e8.get('rank_compute_spread_max_over_mean')}")
    if rep["scaling"]:
        sc = rep["scaling"]
        print(f"  8 ranks vs 4 ranks ({sc['reference_arm']}): {sc['speedup_8_vs_4']}x, efficiency {sc['efficiency']:.0%}, "
              f"same global batch {sc['same_global_batch']}, window tokens/step {sc['tokens_per_step_4rank_8rank']}, "
              f"window mismatch {sc['window_mismatch']}: {sc['layout_hint']}")
    if rep.get("throughput"):
        t = rep["throughput"]
        print(f"  throughput per GPU (4-rank mean) {t['per_gpu_4rank_mean']}: {t['plan_class_b300_thresholds']} (B300 thresholds)")
        if rep.get("projection_train_only_hours_700_steps"):
            print(f"  700 steps, train time only (hours): {rep['projection_train_only_hours_700_steps']}")
    if rep["grad_ckpt"]:
        g = rep["grad_ckpt"]
        print(f"  GRAD_CKPT_DECISION={g['decision']}: {g['reason']}")
    for p in problems:
        print(f"SMOKE PROBLEM: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
