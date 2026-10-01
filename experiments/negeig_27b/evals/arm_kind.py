#!/usr/bin/env python3
"""What an arm is, and the proof that the served program is that arm. Called by serve_arm.sh; the output feeds
compare_general.py through <arm dir>/arm.json.

    arm_kind.py kind        --trainable trainable_000600.pt                 # wide | ctrl (needs torch)
    arm_kind.py receipt     --receipt MERGE-RECEIPT.json [--min-delta-kept 0.95]
    arm_kind.py engagement  --log server.log --kind wide|ctrl|base --ranks 8 [--layers 48] [--gates-sha256 H]
    arm_kind.py assemble    --arm NAME --kind K --serve serve.json [--merge-receipt R] [--engagement E.json] --out arm.json

`engagement` reads the SGLang server log (stderr). The regexes are the ones sglang/check_parity.py uses:

    negeig: active v<N>: <k> gated GDN layers on this rank, gates from <how>, max|W|=<x>
    negeig: installed v<N> on sglang <ver> (pid <pid>)

The plugin's _say() writes every line twice: a bare print to stderr (the line starts with `negeig:`) and logger.info,
which SGLang's root handler renders as `[2026-10-01 00:00:03 DP0 TP1] negeig: ...`. A real log therefore holds both
forms, and `installed` lines from every process that loads the plugin. The check counts the engaged ranks as the larger
of the two forms (a log without the logger form, or without the print form, still counts), so duplicates are not read
as extra ranks and a missing form is not read as a missing rank.

wide: exactly one engaged rank per rank (ranks = dp * tp), every active line reports the same layer count (--layers, 48
for the 27B at tp 1), the gates come from the exported file whose sha256 we computed (--gates-sha256), and max|W| is a
finite number above zero (an all-zero W is the stock model, not the trained gate). Any other `negeig:` line (a warning
or an error) fails the arm, and so does an `installed` line whose version differs from the active lines.
base and ctrl: no `negeig:` line at all, because the plugin is not loaded, and so cannot have changed anything.
Exit 0 ok, 1 not ok (the JSON says why), 2 usage.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import List, Optional

ACTIVE_RE = re.compile(r"negeig: active v(\d+): (\d+) gated GDN layers on this rank, gates from (.*?), max\|W\|=(\S+)")
INSTALLED_RE = re.compile(r"negeig: installed v(\d+) on sglang (\S+) \(pid (\d+)\)")
ANY_NEGEIG_RE = re.compile(r"negeig:")
LOGGER_PREFIX_RE = re.compile(r"\]\s*$")  # `[2026-10-01 00:00:03 DP0 TP1] negeig: ...`: the line went through logging
GATE_KEY_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.linear_attn\.negeig_w(?:\.weight)?$")
KINDS = ("base", "ctrl", "wide")


def classify_keys(keys: List[str]) -> dict:
    gates = [k for k in keys if GATE_KEY_RE.search(k)]
    lora = [k for k in keys if "lora_" in k]
    if gates and not lora:
        kind = "gate_only"
    elif gates:
        kind = "wide"
    elif lora:
        kind = "ctrl"
    else:
        kind = "unknown"
    return {"kind": kind, "n_gate_tensors": len(gates), "n_lora_tensors": len(lora), "n_tensors": len(keys)}


def cmd_kind(a: argparse.Namespace) -> int:
    import torch  # lazy: engagement/receipt/assemble run anywhere

    obj = torch.load(a.trainable, map_location="cpu", weights_only=True)
    state = obj["trainable"] if isinstance(obj, dict) and isinstance(obj.get("trainable"), dict) else obj
    if not isinstance(state, dict):
        print("not a trainable dict", file=sys.stderr)
        return 2
    res = classify_keys(list(state))
    print(json.dumps(res))
    return 0 if res["kind"] in ("wide", "ctrl") else 1


def check_receipt(doc: dict, min_kept: float) -> dict:
    need = ("delta_kept", "delta_kept_worst_tensor", "trainable_sha256", "scale", "rounding")
    missing = [k for k in need if k not in doc]
    problems: List[str] = []
    if missing:
        problems.append(f"merge receipt lacks {missing}")
    else:
        dk, worst = doc["delta_kept"], doc["delta_kept_worst_tensor"]
        if not (isinstance(dk, (int, float)) and math.isfinite(dk)) or dk < min_kept:
            problems.append(f"delta_kept {dk} is below {min_kept}: the merged weights did not take the adapter")
        if not (isinstance(worst, (int, float)) and math.isfinite(worst)) or worst < min_kept / 2:
            problems.append(f"worst tensor delta_kept {worst} is below {min_kept / 2}")
    return {"ok": not problems, "problems": problems,
            **{k: doc.get(k) for k in ("delta_kept", "delta_kept_worst_tensor", "trainable_sha256", "scale", "rounding",
                                       "seed", "targets")}}


def cmd_receipt(a: argparse.Namespace) -> int:
    res = check_receipt(json.loads(Path(a.receipt).read_text()), a.min_delta_kept)
    print(json.dumps(res))
    return 0 if res["ok"] else 1


def _is_logger_form(seg: str) -> bool:
    return bool(LOGGER_PREFIX_RE.search(seg[:seg.index("negeig:")]))


def check_engagement(text: str, kind: str, ranks: int, layers: int = 48, gates_sha256: Optional[str] = None) -> dict:
    # a progress bar writes \r, so a negeig line can share a physical line with it: split on both
    segs = [seg for seg in re.split(r"[\r\n]", text) if ANY_NEGEIG_RE.search(seg)]
    n_any = len(segs)
    active: list = []          # every active line, both forms (the per line checks)
    n_print = n_logger = 0     # active lines per form (the rank count)
    installed: list = []
    other: List[str] = []
    for seg in segs:
        m = ACTIVE_RE.search(seg)
        if m:
            active.append(m.groups())
            if _is_logger_form(seg):
                n_logger += 1
            else:
                n_print += 1
            continue
        m = INSTALLED_RE.search(seg)
        if m:
            installed.append(m.groups())
            continue
        other.append(seg.strip())
    engaged = max(n_print, n_logger)
    problems: List[str] = []
    if kind in ("base", "ctrl"):
        if n_any:
            problems.append(f"{kind}: the log has {n_any} 'negeig:' line(s) but the plugin must not be loaded")
    elif kind == "wide":
        if engaged != ranks:
            problems.append(f"wide: {engaged} 'negeig: active' lines, expected {ranks} (one per rank; "
                            f"{n_print} bare, {n_logger} through logging)")
        if len({a[1] for a in active}) > 1 or any(int(a[1]) != layers for a in active):
            problems.append(f"wide: gated layer counts per rank are {sorted({a[1] for a in active})}, expected {layers}")
        if other:
            problems.append(f"wide: {len(other)} other 'negeig:' line(s) in the log (a warning or error): {other[0][:160]}")
        if active and installed:
            stale = sorted({i[0] for i in installed} - {a[0] for a in active})
            if stale:
                problems.append(f"wide: 'installed' line(s) for patch version(s) {stale}, the active lines are "
                                f"{sorted({a[0] for a in active})}")
        if gates_sha256 is not None:
            bad = [a[2] for a in active if f"sha256={gates_sha256}" not in a[2]]
            if bad:
                problems.append(f"wide: {len(bad)} rank(s) loaded gates that are not the exported file "
                                f"({gates_sha256[:12]}): {bad[0][:120]}")
        for a in active:
            try:
                w = float(a[3])
            except ValueError:
                problems.append(f"wide: max|W| {a[3]!r} is not a number")
                break
            if not math.isfinite(w) or w <= 0.0:
                problems.append(f"wide: max|W| = {a[3]}: the gate is zero or not finite, this is the stock model")
                break
    else:
        raise SystemExit(f"kind {kind!r} not one of {KINDS}")
    return {"ok": not problems, "problems": problems, "kind": kind, "ranks_expected": ranks, "negeig_lines": n_any,
            "active_lines": engaged, "active_bare": n_print, "active_logged": n_logger, "installed_lines": len(installed),
            "other_lines": len(other), "max_w": sorted({a[3] for a in active}),
            "layers_per_rank": sorted({int(a[1]) for a in active}), "patch_versions": sorted({int(a[0]) for a in active})}


def cmd_engagement(a: argparse.Namespace) -> int:
    text = Path(a.log).read_text(errors="replace")
    res = check_engagement(text, a.kind, a.ranks, a.layers, a.gates_sha256)
    res["log"] = Path(a.log).name
    print(json.dumps(res))
    return 0 if res["ok"] else 1


def cmd_assemble(a: argparse.Namespace) -> int:
    doc = {"arm": a.arm, "kind": a.kind, "serve": json.loads(Path(a.serve).read_text())}
    if a.merge_receipt:
        doc["merge_receipt"] = json.loads(Path(a.merge_receipt).read_text())
    if a.engagement:
        doc["engagement"] = json.loads(Path(a.engagement).read_text())
    if a.kind != "base" and not a.merge_receipt:
        print("a non-base arm needs --merge-receipt", file=sys.stderr)
        return 2
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("kind")
    p.add_argument("--trainable", required=True)
    p.set_defaults(fn=cmd_kind)
    p = sub.add_parser("receipt")
    p.add_argument("--receipt", required=True)
    p.add_argument("--min-delta-kept", type=float, default=0.95)
    p.set_defaults(fn=cmd_receipt)
    p = sub.add_parser("engagement")
    p.add_argument("--log", required=True)
    p.add_argument("--kind", required=True, choices=KINDS)
    p.add_argument("--ranks", type=int, required=True)
    p.add_argument("--layers", type=int, default=48)
    p.add_argument("--gates-sha256")
    p.set_defaults(fn=cmd_engagement)
    p = sub.add_parser("assemble")
    p.add_argument("--arm", required=True)
    p.add_argument("--kind", required=True, choices=KINDS)
    p.add_argument("--serve", required=True)
    p.add_argument("--merge-receipt")
    p.add_argument("--engagement")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_assemble)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
