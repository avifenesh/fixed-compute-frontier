#!/usr/bin/env python3
"""Length check of the self-distilled replay files, run ON a box before the first training step.

train27.py renders every replay row with its own encode_chat (the model's template, thinking off) and keeps a row only
when 0 < tokens <= --max_len (default 12288) and it has at least one labelled token. Rows over the limit are dropped
silently by the Pool, so a tool replay whose transcripts run long would train on fewer rows than the file holds, with
no error. This script measures the same thing with the same function and reports it per file:

    rows, usable, over_max_len, no_labels, render_errors, token length mean / p50 / p95 / max (usable rows)

Exit 0: both files read, every row rendered. A WARNING line (and "warn": true) when a file loses more than --warn-frac
of its rows; the owner reads it, nothing stops the run. Exit 1: a file is missing or empty, nothing usable is left in a
file, or a row does not render. Exit 1 on a render error is deliberate: train27 has no guard there and would die at
start. Exit 2: the tokenizer cannot be loaded.

train27 also holds min(--eval_tool_rows 64, usable // 5) tool rows out of training for the agreement eval; the receipt
reports that as tool_heldout_for_eval and the rest as tool_train_rows.

    sd_lengths.py [--dir $W/data/sd] [--model-dir $W/model] [--train27 PATH] [--max_len 12288] [--out FILE]
                  [--warn-frac 0.10] [--eval-tool-rows 64]

The tokenizer comes from the checkpoint dir ($MODEL_DIR), the same files the trainer loads. main(argv, tok=) takes a
tokenizer object so the CPU tests need no transformers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E27 = HERE.parent
sys.path.insert(0, str(E27 / "selfdistill"))

FILES = (("chat", "chat_sft.jsonl", None), ("tool", "tool_sft.jsonl", "tools"))


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError as e:
                    raise ValueError(f"{path}:{n}: not JSON ({e})") from e
    return rows


def measure(name, rows, encode_chat, tok, tools_key, max_len, warn_frac):
    """One pool's numbers, by the trainer's own keep rule."""
    lengths, over, no_labels, errors = [], 0, 0, []
    over_lengths = []
    for i, r in enumerate(rows):
        try:
            ids, mask = encode_chat(tok, r["messages"], r.get(tools_key) if tools_key else None)
        except Exception as e:  # noqa: BLE001 - any renderer failure is a row the trainer would crash on
            errors.append({"row": i, "id": r.get("id", r.get("item_id")), "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        n, labels = len(ids), sum(mask[1:])
        if not 0 < n <= max_len:
            over += 1
            over_lengths.append(n)
        elif labels <= 0:
            no_labels += 1
        else:
            lengths.append(n)
    lengths.sort()
    over_lengths.sort()
    total = len(rows)
    lost = over + no_labels
    out = {"rows": total, "usable": len(lengths), "over_max_len": over, "no_labels": no_labels,
           "render_errors": len(errors), "first_render_errors": errors[:5],
           "lost_frac": round(lost / total, 4) if total else 0.0,
           "len_mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
           "len_p50": pct(lengths, 0.5), "len_p95": pct(lengths, 0.95), "len_max": lengths[-1] if lengths else 0,
           "over_len_min": over_lengths[0] if over_lengths else 0,
           "over_len_max": over_lengths[-1] if over_lengths else 0}
    out["warn"] = bool(total and lost / total > warn_frac)
    out["name"] = name
    return out


def main(argv=None, tok=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=None, help="folder holding chat_sft.jsonl and tool_sft.jsonl (default $W/data/sd)")
    ap.add_argument("--model-dir", default=None, help="checkpoint dir with the tokenizer files (default $W/model)")
    ap.add_argument("--train27", default=str(E27 / "train27.py"), help="the trainer whose encode_chat is measured")
    ap.add_argument("--max_len", type=int, default=12288, help="train27 --max_len; rows above it are dropped")
    ap.add_argument("--warn-frac", type=float, default=0.10)
    ap.add_argument("--eval-tool-rows", type=int, default=64,
                    help="train27 --eval_tool_rows: tool rows held out of training for the agreement eval "
                         "(at most a fifth of the usable rows)")
    ap.add_argument("--out", default=None, help="write the receipt JSON here")
    args = ap.parse_args(argv)
    w = os.environ.get("NEGEIG_W", "/workspace/negeig")
    sd = Path(args.dir or Path(w) / "data" / "sd")
    model_dir = args.model_dir or str(Path(w) / "model")

    from sd_common import load_encode_chat, load_tokenizer
    encode_chat = load_encode_chat(args.train27)
    if tok is None:
        tok = load_tokenizer(model_dir)
        if tok is None:
            print(f"ERROR: no tokenizer at {model_dir}", file=sys.stderr)
            return 2

    receipt, problems = {"dir": str(sd), "max_len": args.max_len, "pools": {}}, []
    for name, fname, tools_key in FILES:
        path = sd / fname
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"{fname}: missing or empty in {sd}")
            continue
        try:
            rows = read_rows(path)
        except ValueError as e:
            problems.append(str(e))
            continue
        m = measure(name, rows, encode_chat, tok, tools_key, args.max_len, args.warn_frac)
        receipt["pools"][name] = m
        if not m["rows"]:
            problems.append(f"{fname}: no rows")
        elif m["render_errors"]:
            problems.append(f"{fname}: {m['render_errors']} row(s) do not render, train27 would die at start; "
                            f"first: {m['first_render_errors'][0]['error']}")
        elif not m["usable"]:
            problems.append(f"{fname}: nothing usable ({m['over_max_len']} over {args.max_len}, {m['no_labels']} unlabelled)")
        print(f"SD_LENGTHS {name}: rows={m['rows']} usable={m['usable']} over_max_len={m['over_max_len']} "
              f"no_labels={m['no_labels']} render_errors={m['render_errors']} len_p50={m['len_p50']} "
              f"len_p95={m['len_p95']} len_max={m['len_max']} (max_len {args.max_len})")
        if m["warn"]:
            print(f"WARNING: {name} replay loses {m['lost_frac']:.1%} of its rows to the {args.max_len}-token cap or "
                  f"missing labels (warn above {args.warn_frac:.0%}); train27 drops them without a message")
    receipt["problems"] = problems
    receipt["ok"] = not problems
    tool = receipt["pools"].get("tool", {})
    receipt["tool_rows_over_max_len"] = tool.get("over_max_len")
    if tool:
        receipt["tool_heldout_for_eval"] = min(args.eval_tool_rows, tool["usable"] // 5)
        receipt["tool_train_rows"] = tool["usable"] - receipt["tool_heldout_for_eval"]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(args.out).with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt, indent=2) + "\n")
        tmp.replace(args.out)
    for p in problems:
        print(f"PROBLEM: {p}", file=sys.stderr)
    print(f"SD_LENGTHS_{'OK' if not problems else 'FAIL'} tool_rows_over_max_len={tool.get('over_max_len')} "
          f"tool_usable={tool.get('usable')}/{tool.get('rows')} tool_train={receipt.get('tool_train_rows')} "
          f"chat_usable={receipt['pools'].get('chat', {}).get('usable')}/{receipt['pools'].get('chat', {}).get('rows')}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
