#!/usr/bin/env python3
"""Paired comparison of one arm against the untouched base on the Stage A general-eval gates (PLAN.md "What decides"
item 3). Reads two arm directories written by run_general.sh (results/<eval>.json with per-item outcomes), pairs the
items by id, and writes one JSON plus one markdown table.

    python compare_general.py --base RUN/base --arm RUN/wide_s1 --out-json cmp.json --out-md cmp.md
    python compare_general.py --base RUN/base --arm RUN/wide_s1 --base-repeat RUN/base_repeat   # noise floor
    python compare_general.py --base RUN/base --power-table                                      # power only

The gates (21) and their bands, in points (arm minus base, negative is a regression):

    bfcl/<each of the 17 categories>  3      mmlupro_cloze  1      humaneval  3      ifeval_strict  2
    gmmlu_he (full he-test)           1

Everything else is report-only: mmlupro_letter, ifeval_loose, the instruction-level IFEval rows, gmmlu_he_pop2000,
bfcl/all. IFEval strict means prompt-level strict accuracy (the owner confirms the reading, PLAN.md only says IFEval).

Statistics: the items are paired, so the delta is (arm-only correct minus base-only correct) over n. The interval is a
percentile bootstrap over the four paired cells (both right, base only, arm only, both wrong) drawn multinomially,
10,000 draws, 95% by default. A gate is

    PASS              the whole interval is inside the band (lo >= -t)
    FAIL              the whole interval is outside it (hi < -t)
    PASS_UNRESOLVED   the interval straddles -t and the point estimate is inside the band
    FAIL_UNRESOLVED   the interval straddles -t and the point estimate is outside the band

UNRESOLVED means the subset is too small to decide at that band, not that the arm is fine. `power` per gate says how
small: the regression this n would have caught 80% of the time, and the smallest true delta it could have certified.
The instruction-level IFEval rows are resampled by prompt (instructions inside one prompt are not independent).
Exit codes: 0 written; 1 with the reason on stderr when the inputs cannot be compared (pins or config differ, a
results dir is missing); 2 for a bad command line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalcommon import BFCL_FULL, read_json, utc, write_json  # noqa: E402

SCHEMA = "negeig-general-compare/1"
EVALS = ("bfcl", "mmlupro", "humaneval", "ifeval", "gmmlu")

BANDS: Dict[str, float] = {f"bfcl/{c}": 3.0 for c in BFCL_FULL}
BANDS.update({"mmlupro_cloze": 1.0, "humaneval": 3.0, "ifeval_strict": 2.0, "gmmlu_he": 1.0})
FULL_N = {f"bfcl/{c}": n for c, n in BFCL_FULL.items()}
FULL_N.update({"mmlupro_cloze": 12032, "mmlupro_letter": 12032, "humaneval": 164, "ifeval_strict": 541,
               "ifeval_loose": 541, "gmmlu_he": 14042, "gmmlu_he_pop2000": 2000})
REPORT_ONLY = ("mmlupro_letter", "ifeval_loose", "ifeval_strict_inst", "ifeval_loose_inst", "gmmlu_he_pop2000",
               "bfcl/all")
GATE_ORDER = [f"bfcl/{c}" for c in BFCL_FULL] + ["mmlupro_cloze", "humaneval", "ifeval_strict", "gmmlu_he"]
Z = 1.959964
Z80 = 2.8  # 1.96 + 0.84: the alpha 5% two-sided, power 80% shift in standard errors

# What the server was launched with must be the same for base and arm, or the delta is not the adapter's alone.
# (dp_size and port are launch plumbing, not numerics, and may differ.) serve_arm.sh writes these into serve.json.
SERVE_KEYS = ("context_length", "radix", "tp_size", "attn_backend", "linear_attn_backend", "linear_prefill_backend",
              "linear_decode_backend", "sglang_version", "model_rev", "gpu_name", "dtype", "extra_args")

RANK = {"PASS": 0, "PASS_UNRESOLVED": 1, "FAIL_UNRESOLVED": 2, "FAIL": 3}


# ----------------------------------------------------------------------------------------------------------
# the math
def paired_cells(base: Dict[str, int], arm: Dict[str, int]) -> Tuple[List[str], int, int, int, int]:
    """(common ids sorted, both right, base only right, arm only right, both wrong)."""
    ids = sorted(set(base) & set(arm))
    a = b = c = d = 0
    for i in ids:
        x, y = int(base[i]), int(arm[i])
        if x and y:
            a += 1
        elif x and not y:
            b += 1
        elif y and not x:
            c += 1
        else:
            d += 1
    return ids, a, b, c, d


def _seed(name: str, seed: int) -> int:
    return (int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big") ^ seed) & 0x7FFFFFFF


def bootstrap_interval(a: int, b: int, c: int, d: int, name: str = "", draws: int = 10000, level: float = 0.95,
                       seed: int = 20261001) -> Tuple[float, float]:
    """Percentile interval of the paired delta in points, resampling the four cells multinomially."""
    import numpy as np

    n = a + b + c + d
    if n == 0:
        return float("nan"), float("nan")
    if b + c == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(_seed(name, seed))
    p = np.array([a, b, c, d], dtype=float) / n
    s = rng.multinomial(n, p, size=draws)
    delta = 100.0 * (s[:, 2] - s[:, 1]) / n
    lo, hi = np.percentile(delta, [50.0 * (1.0 - level), 50.0 * (1.0 + level)])
    return float(lo), float(hi)


def cluster_interval(base: Dict[str, int], arm: Dict[str, int], name: str = "", draws: int = 10000,
                     level: float = 0.95, seed: int = 20261001) -> Tuple[float, float]:
    """Delta interval resampling clusters (the prompt of an IFEval instruction id 'key:idx') with replacement."""
    import numpy as np

    ids = sorted(set(base) & set(arm))
    if not ids:
        return float("nan"), float("nan")
    order: Dict[str, int] = {}
    diff: List[int] = []
    size: List[int] = []
    for i in ids:
        k = i.rsplit(":", 1)[0]
        j = order.setdefault(k, len(order))
        if j == len(diff):
            diff.append(0)
            size.append(0)
        diff[j] += int(arm[i]) - int(base[i])
        size[j] += 1
    diff_a, size_a = np.array(diff, dtype=float), np.array(size, dtype=float)
    if not diff_a.any():
        return 0.0, 0.0
    rng = np.random.default_rng(_seed(name, seed))
    idx = rng.integers(0, len(diff_a), size=(draws, len(diff_a)))
    delta = 100.0 * diff_a[idx].sum(axis=1) / size_a[idx].sum(axis=1)
    lo, hi = np.percentile(delta, [50.0 * (1.0 - level), 50.0 * (1.0 + level)])
    return float(lo), float(hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p of b vs c discordant items (information only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return float(min(1.0, 2.0 * tail))


def classify(delta: float, lo: float, hi: float, band: float) -> Tuple[str, str]:
    """(verdict from the interval, literal verdict from the point estimate)."""
    point = "within" if delta >= -band else "outside"
    if lo >= -band:
        return "PASS", point
    if hi < -band:
        return "FAIL", point
    return ("PASS_UNRESOLVED" if delta >= -band else "FAIL_UNRESOLVED"), point


def power_stats(n: int, q: float, band: float, level: float = 0.95) -> dict:
    """What n paired items with discordant rate q can tell about a band (all in points unless named)."""
    zc = {0.90: 1.644854, 0.95: 1.959964, 0.99: 2.575829}.get(round(level, 2), Z)
    se = 100.0 * math.sqrt(max(q, 0.0) / n) if n else float("nan")
    frac = band / 100.0
    need = q * (zc / frac) ** 2 if frac > 0 else float("nan")
    detect = band + (zc + 0.84) * se
    certify = -band + (zc + 0.84) * se
    return {"se_points": se, "ci_half_width_points": zc * se, "n_needed_to_resolve_at_delta0": need,
            "fail_detect_80pct_points": detect, "pass_certify_80pct_if_true_delta_ge": certify,
            "cannot_certify_at_zero": bool(certify > 0)}


def compare_metric(name: str, bm: dict, am: dict, band: Optional[float], draws: int, level: float,
                   seed: int) -> dict:
    bi = {k: int(v) for k, v in bm["items"].items()}
    ai = {k: int(v) for k, v in am["items"].items()}
    ids, a, b, c, d = paired_cells(bi, ai)
    n = len(ids)
    row: Dict[str, Any] = {"metric": name, "level": am.get("level", "item"), "n": n, "n_base": len(bi),
                           "n_arm": len(ai), "both": a, "base_only": b, "arm_only": c, "neither": d}
    n_full = am.get("n_full") or bm.get("n_full") or FULL_N.get(name)
    row["n_full"] = n_full
    row["coverage"] = (n / n_full) if n_full else None
    if n == 0:
        row.update({"verdict": "MISSING", "note": "no common items"})
        return row
    row["base"] = 100.0 * (a + b) / n
    row["arm"] = 100.0 * (a + c) / n
    delta = 100.0 * (c - b) / n
    row["delta"] = delta
    if row["level"] == "instruction":
        lo, hi = cluster_interval(bi, ai, name, draws, level, seed)
    else:
        lo, hi = bootstrap_interval(a, b, c, d, name, draws, level, seed)
    row["ci_lo"], row["ci_hi"] = lo, hi
    row["discordant_rate"] = (b + c) / n
    row["mcnemar_p"] = mcnemar_exact(b, c)
    row["band"] = band
    if band is not None:
        row["verdict"], row["point_verdict"] = classify(delta, lo, hi, band)
        row["power"] = power_stats(n, (b + c) / n, band, level)
        row["one_item_points"] = 100.0 / n
        row["coarse"] = bool(100.0 / n > band)
    else:
        row["verdict"] = "REPORT"
    return row


# ----------------------------------------------------------------------------------------------------------
# loading
def load_arm(d: Path) -> dict:
    """{eval: result doc}, {metric: (metric doc, eval)}, arm.json."""
    if not d.is_dir():
        raise SystemExit(f"{d}: not a directory")
    results: Dict[str, dict] = {}
    for p in sorted((d / "results").glob("*.json")) if (d / "results").is_dir() else []:
        doc = read_json(p)
        if doc.get("schema", "").split("/")[0] != "negeig-general-eval":
            continue
        results[doc["eval"]] = doc
    metrics: Dict[str, Tuple[dict, str]] = {}
    for ev, doc in results.items():
        for k, m in doc.get("metrics", {}).items():
            metrics[k] = (m, ev)
    info = read_json(d / "arm.json") if (d / "arm.json").is_file() else {}
    return {"dir": str(d), "results": results, "metrics": metrics, "arm_json": info}


def check_compatible(base: dict, arm: dict, allow: bool) -> List[str]:
    """Pins and config must match per eval: a different protocol is a different measurement."""
    problems: List[str] = []
    for ev, ad in arm["results"].items():
        bd = base["results"].get(ev)
        if bd is None:
            continue
        if ad.get("pins") != bd.get("pins"):
            problems.append(f"{ev}: pins differ between base and arm")
        if ad.get("config") != bd.get("config"):
            keys = sorted(k for k in set(ad.get("config", {})) | set(bd.get("config", {}))
                          if ad.get("config", {}).get(k) != bd.get("config", {}).get(k))
            problems.append(f"{ev}: config differs on {keys}")
    bs, as_ = (base["arm_json"].get("serve") or {}), (arm["arm_json"].get("serve") or {})
    if bs and as_:
        for k in SERVE_KEYS:
            if bs.get(k) != as_.get(k):
                problems.append(f"serve: {k} differs (base {bs.get(k)!r}, arm {as_.get(k)!r})")
    if problems and not allow:
        raise SystemExit("base and arm were not measured under one protocol:\n  " + "\n  ".join(problems)
                         + "\n(--allow-config-diff to compare anyway and record the problem)")
    return problems


def arm_validity(info: dict, kind_expected: Optional[str] = None) -> dict:
    eng = info.get("engagement") or {}
    out = {"kind": info.get("kind"), "engagement_ok": eng.get("ok"), "engagement": eng,
           "delta_kept": (info.get("merge_receipt") or {}).get("delta_kept"),
           "trainable_sha256": (info.get("merge_receipt") or {}).get("trainable_sha256")}
    return out


def run_health(side: dict) -> dict:
    """Per eval facts about how the run ended, from the result docs: subset or full, and the BFCL items that were not a
    clean model answer (a request over the context window, or a transport error accepted after its retries)."""
    out: Dict[str, dict] = {}
    for ev, doc in side["results"].items():
        sub = doc.get("subset") or {}
        d = doc.get("diagnostics") or {}
        h: dict = {"subset": bool(int(sub.get("stride") or 1) > 1 or int(sub.get("limit") or 0) > 0
                                  or int(sub.get("offset") or 0) > 0)}
        if ev == "bfcl":
            h["n_context_overflow"] = int(d.get("n_context_overflow") or 0)
            h["n_accepted_transport_errors"] = int(d.get("n_accepted_transport_errors") or 0)
        out[ev] = h
    return out


def health_notes(name: str, health: dict) -> List[str]:
    notes: List[str] = []
    for ev, h in sorted(health.items()):
        if h.get("subset"):
            notes.append(f"{name}: {ev} ran a subset, not the full set")
        if h.get("n_accepted_transport_errors"):
            notes.append(f"{name}: {h['n_accepted_transport_errors']} BFCL items were scored from a transport error "
                         "accepted after its retries (BFCL's checker scores an error as correct on the irrelevance "
                         "categories): re-run the arm until none is left")
    return notes


# ----------------------------------------------------------------------------------------------------------
def overall(rows: List[dict], select=lambda r: True) -> dict:
    gated = [r for r in rows if r.get("band") is not None and select(r)]
    worst_ci = "PASS"
    point_ok = True
    missing = []
    for r in gated:
        v = r["verdict"]
        if v == "MISSING":
            missing.append(r["metric"])
            continue
        if RANK[v] > RANK[worst_ci]:
            worst_ci = v
        if r.get("point_verdict") == "outside":
            point_ok = False
    if missing:
        ci = "INCOMPLETE"
    else:
        ci = worst_ci
    return {"n_gates": len(gated), "ci": ci, "point": ("INCOMPLETE" if missing else ("PASS" if point_ok else "FAIL")),
            "missing": missing,
            "counts": {k: sum(1 for r in gated if r["verdict"] == k)
                       for k in ("PASS", "PASS_UNRESOLVED", "FAIL_UNRESOLVED", "FAIL", "MISSING")}}


def compare(base_dir: os.PathLike, arm_dir: os.PathLike, draws: int = 10000, level: float = 0.95,
            seed: int = 20261001, allow_config_diff: bool = False) -> dict:
    if Path(base_dir).resolve() == Path(arm_dir).resolve():
        raise SystemExit(f"--base and --arm are the same directory ({base_dir}): a run compared with itself proves nothing")
    base, arm = load_arm(Path(base_dir)), load_arm(Path(arm_dir))
    if not base["metrics"]:
        raise SystemExit(f"{base_dir}: no results/*.json")
    if not arm["metrics"]:
        raise SystemExit(f"{arm_dir}: no results/*.json")
    problems = check_compatible(base, arm, allow_config_diff)
    rows: List[dict] = []
    names = list(GATE_ORDER) + [n for n in REPORT_ONLY]
    extra = sorted((set(base["metrics"]) | set(arm["metrics"])) - set(names))
    for name in names + extra:
        bm, am = base["metrics"].get(name), arm["metrics"].get(name)
        band = BANDS.get(name)
        if bm is None or am is None:
            if band is not None:
                rows.append({"metric": name, "band": band, "verdict": "MISSING",
                             "note": "absent in " + ("base" if bm is None else "arm")
                                     + (" and arm" if bm is None and am is None else "")})
            continue
        rows.append(compare_metric(name, bm[0], am[0], band, draws, level, seed))
    av = arm_validity(arm["arm_json"])
    bv = arm_validity(base["arm_json"])
    valid = True
    reasons: List[str] = []

    def invalid(why: str) -> None:
        nonlocal valid
        valid = False
        reasons.append(why)

    if av["engagement_ok"] is False:
        invalid("arm engagement check failed: the arm was not the program it claims to be")
    if bv["engagement_ok"] is False:
        invalid("base engagement check failed")
    # a verdict that is not bound to a served program is not a verdict: a missing record is as invalid as a failed one
    if av["engagement_ok"] is None:
        invalid("arm has no engagement record (arm.json): the verdict is not bound to a served program")
    if bv["engagement_ok"] is None:
        invalid("base has no engagement record (arm.json): it is not proven to be the stock model with no plugin loaded")
    if not arm["arm_json"].get("serve") or not base["arm_json"].get("serve"):
        invalid("arm.json has no serve record for base or arm: the launch settings were not compared")
    if bv["kind"] != "base":
        invalid(f"the base directory's arm.json says kind {bv['kind']!r}, not 'base': it is not the untouched model")
    if av["kind"] not in ("wide", "ctrl", "base"):
        invalid(f"the arm directory's arm.json says kind {av['kind']!r}, not wide or ctrl")
    for who, v in (("arm", av), ("base", bv)):
        ek = (v["engagement"] or {}).get("kind")
        if ek is not None and ek != v["kind"]:
            invalid(f"{who}: the engagement check ran as kind {ek!r} but arm.json says {v['kind']!r}")
    if any(r.get("verdict") == "MISSING" for r in rows if r.get("band") is not None):
        reasons.append("at least one gate has no paired items")
    health = {"base": run_health(base), "arm": run_health(arm)}
    reasons += health_notes("base", health["base"]) + health_notes("arm", health["arm"])
    doc = {
        "schema": SCHEMA, "utc": utc(), "ci_level": level, "draws": draws, "seed": seed,
        "base_dir": base["dir"], "arm_dir": arm["dir"],
        "arm": next((r["arm"] for r in arm["results"].values()), None),
        "base": next((r["arm"] for r in base["results"].values()), None),
        "valid": valid, "validity_notes": reasons, "protocol_problems": problems,
        "arm_info": av, "base_info": bv, "health": health,
        "seconds": {"base": {e: d.get("seconds") for e, d in base["results"].items()},
                    "arm": {e: d.get("seconds") for e, d in arm["results"].items()}},
        "rows": rows,
        "overall": {
            "all": overall(rows),
            "without_coarse": overall(rows, lambda r: not r.get("coarse")),
        },
        "note": ("21 gates at 95% each: a FAIL a hair past a band is weak evidence, a FAIL far past it is not. "
                 "UNRESOLVED gates need more items (see power), not a coin flip. Decisions stay with the owner."),
    }
    if not valid:
        doc["overall"]["all"]["ci"] = "INVALID"
        doc["overall"]["without_coarse"]["ci"] = "INVALID"
    return doc


def noise_floor(base_dir: os.PathLike, repeat_dir: os.PathLike, draws: int, level: float, seed: int) -> dict:
    """Base against a second run of the base: the nondeterminism floor the arm deltas have to be read against."""
    doc = compare(base_dir, repeat_dir, draws, level, seed, allow_config_diff=True)
    rows = [r for r in doc["rows"] if r.get("n")]
    return {"repeat_dir": str(repeat_dir),
            "rows": [{"metric": r["metric"], "n": r["n"], "delta": r["delta"], "ci_lo": r["ci_lo"],
                      "ci_hi": r["ci_hi"], "discordant_rate": r["discordant_rate"]} for r in rows],
            "max_abs_delta_gated": max([abs(r["delta"]) for r in rows if r.get("band") is not None] or [0.0]),
            "max_discordant_rate": max([r["discordant_rate"] for r in rows] or [0.0])}


# ----------------------------------------------------------------------------------------------------------
# rendering
def _f(x: Optional[float], spec: str = ".1f") -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def render_md(doc: dict) -> str:
    L: List[str] = []
    ov = doc["overall"]["all"]
    L.append(f"# General evals: {doc.get('arm')} against {doc.get('base')}\n")
    L.append(f"{doc['utc']}, {int(doc['ci_level'] * 100)}% paired bootstrap interval, {doc['draws']} draws. "
             f"Delta is arm minus base in points; a negative delta is a regression.\n")
    L.append(f"**Overall (interval reading): {ov['ci']}.** Literal point reading: {ov['point']}. "
             f"Gates: {ov['counts']}.")
    wc = doc["overall"]["without_coarse"]
    L.append(f"Without the coarse categories (one item moves more than the band): {wc['ci']} / {wc['point']} "
             f"over {wc['n_gates']} gates.\n")
    if not doc["valid"] or doc["validity_notes"]:
        L.append("Validity notes:")
        for n in doc["validity_notes"]:
            L.append(f"- {n}")
        L.append("")
    if doc["protocol_problems"]:
        L.append("Protocol problems (compared anyway): " + "; ".join(doc["protocol_problems"]) + "\n")
    L.append("| gate | n (cover) | base | arm | delta [CI] | band | verdict | point | q | resolves at n | "
             "catches regression of | flag |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in doc["rows"]:
        if r.get("band") is None:
            continue
        if r["verdict"] == "MISSING":
            L.append(f"| {r['metric']} | | | | | {r['band']:g} | MISSING | | | | | {r.get('note', '')} |")
            continue
        p = r["power"]
        flags = []
        if r.get("coarse"):
            flags.append("coarse")
        if p["cannot_certify_at_zero"]:
            flags.append("cannot certify a PASS even at true zero")
        cov = f" ({r['coverage'] * 100:.0f}%)" if r.get("coverage") is not None else ""
        L.append(f"| {r['metric']} | {r['n']}{cov} | {r['base']:.1f} | {r['arm']:.1f} | "
                 f"{r['delta']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['band']:g} | {r['verdict']} | "
                 f"{r['point_verdict']} | {r['discordant_rate']:.3f} | {_f(p['n_needed_to_resolve_at_delta0'], '.0f')} | "
                 f"{p['fail_detect_80pct_points']:.1f} pts | {'; '.join(flags)} |")
    rep = [r for r in doc["rows"] if r.get("band") is None and r.get("n")]
    if rep:
        L.append("\nReport only (no band):\n")
        L.append("| metric | n (cover) | base | arm | delta [CI] | q |")
        L.append("|---|---|---|---|---|---|")
        for r in rep:
            cov = f" ({r['coverage'] * 100:.0f}%)" if r.get("coverage") is not None else ""
            L.append(f"| {r['metric']} | {r['n']}{cov} | {r['base']:.1f} | {r['arm']:.1f} | "
                     f"{r['delta']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['discordant_rate']:.3f} |")
    ai = doc.get("arm_info", {})
    L.append(f"\nArm: kind {ai.get('kind')}, engagement ok {ai.get('engagement_ok')}, merged delta kept "
             f"{ai.get('delta_kept')}.")
    hb, ha = (doc.get("health") or {}).get("base", {}).get("bfcl"), (doc.get("health") or {}).get("arm", {}).get("bfcl")
    if hb and ha:
        L.append(f"BFCL context overflows (scored wrong, deterministic): base {hb['n_context_overflow']}, arm "
                 f"{ha['n_context_overflow']}. Accepted transport errors: base {hb['n_accepted_transport_errors']}, arm "
                 f"{ha['n_accepted_transport_errors']}.")
    if "noise_floor" in doc:
        nf = doc["noise_floor"]
        L.append(f"\nNoise floor (base against {nf['repeat_dir']}): largest gated |delta| "
                 f"{nf['max_abs_delta_gated']:.2f} points, largest discordant rate {nf['max_discordant_rate']:.3f}.")
    L.append(f"\n{doc['note']}")
    return "\n".join(L) + "\n"


def power_table(base_dir: os.PathLike, qs: List[float], level: float) -> str:
    base = load_arm(Path(base_dir))
    L = ["| gate | n | band | " + " | ".join(f"q={q:g}: half-width / catches" for q in qs) + " |",
         "|---|---|---|" + "---|" * len(qs)]
    for name in GATE_ORDER:
        if name not in base["metrics"]:
            continue
        n = base["metrics"][name][0]["n"]
        band = BANDS[name]
        cells = []
        for q in qs:
            p = power_stats(n, q, band, level)
            cells.append(f"{p['ci_half_width_points']:.1f} / {p['fail_detect_80pct_points']:.1f}")
        L.append(f"| {name} | {n} | {band:g} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="the untouched model's arm directory")
    ap.add_argument("--arm", help="the arm directory to judge")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--ci-level", type=float, default=0.95)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--base-repeat", help="a second run of the base, for the noise floor")
    ap.add_argument("--allow-config-diff", action="store_true")
    ap.add_argument("--power-table", action="store_true",
                    help="print the power of the base's sample sizes at several discordant rates and exit")
    a = ap.parse_args(argv)
    if not 0.5 < a.ci_level < 1.0:
        raise SystemExit("--ci-level must be in (0.5, 1)")
    if a.power_table:
        sys.stdout.write(power_table(a.base, [0.02, 0.04, 0.06, 0.10, 0.15], a.ci_level))
        return 0
    if not a.arm:
        raise SystemExit("--arm is required (or use --power-table)")
    doc = compare(a.base, a.arm, a.draws, a.ci_level, a.seed, a.allow_config_diff)
    if a.base_repeat:
        doc["noise_floor"] = noise_floor(a.base, a.base_repeat, a.draws, a.ci_level, a.seed)
    md = render_md(doc)
    if a.out_json:
        write_json(a.out_json, doc)
    if a.out_md:
        Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_md).write_text(md, encoding="utf-8")
    if not a.out_json and not a.out_md:
        sys.stdout.write(md)
    else:
        ov = doc["overall"]["all"]
        print(f"overall {ov['ci']} (point {ov['point']}) gates {ov['counts']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
