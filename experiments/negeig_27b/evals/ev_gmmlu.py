#!/usr/bin/env python3
"""Global-MMLU Hebrew (CohereLabs/Global-MMLU, he) through the served model: the Hebrew-lane protocol.

Chat, system message "Answer with only the letter", user message = the subject's first five dev rows as solved shots
plus the question, max_tokens 1, temperature 0, thinking off. The prediction is the first character of the reply,
uppercased, exactly as the Hebrew lane's knowledge.py scored it (no retry on a wrong letter). Prompts are built
byte-identically to that lane's block() (checked against the original on all 14,042 rows in test_evals.py).

Gate: the full he-test (14,042 rows), band 1 point. Reported beside it: gmmlu_he_pop2000, the lane's pinned 2,000-row
population (sorted unique sample_id, Random(20260905).shuffle, first 2000), which is a subset of the same rows.

    python ev_gmmlu.py --arm base --out RUN/base --data evals/data [--stride 4]
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalcommon import (THINK_OFF, EvalRun, add_common_args, chat_text, eval_pins, finish, load_normalized,  # noqa: E402
                        load_pins, metric, resolve_out, canon_sha, EXIT_USAGE)

LETTERS = ["A", "B", "C", "D"]
SYSTEM = "Answer with only the letter"
SHOTS = 5


def block(row: dict, with_answer: bool) -> str:
    q = f"{row['question'].strip()}\n"
    for letter, opt in zip(LETTERS, row["options"]):
        q += f"{letter}. {str(opt).strip()}\n"
    q += "Answer:"
    if with_answer:
        q += f" {row['answer'].strip()}\n\n"
    return q


def build_shots(dev_rows: List[dict]) -> Dict[str, str]:
    """Per subject: its first SHOTS dev rows in file order, each with its answer."""
    by: Dict[str, List[dict]] = {}
    for r in dev_rows:
        by.setdefault(r["subject"], []).append(r)
    return {s: "".join(block(x, True) for x in rows[:SHOTS]) for s, rows in by.items()}


def prompt_for(row: dict, shots: Dict[str, str]) -> str:
    return shots.get(row["subject"], "") + block(row, False)


def pop_ids(rows: List[dict], seed: int, n: int) -> List[str]:
    ids = sorted({r["id"] for r in rows})
    random.Random(seed).shuffle(ids)
    return ids[:n]


def predict(text: str) -> str:
    return text.strip()[:1].upper()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap, workers=128)
    a = ap.parse_args(argv)
    resolve_out(a)
    pins = load_pins()
    g = pins["datasets"]["gmmlu_he"]
    test, test_sha = load_normalized(a, "gmmlu_he_test", pins)
    dev, dev_sha = load_normalized(a, "gmmlu_he_dev", pins)
    if not a.allow_unpinned and (len(test) != g["rows_test"] or len(dev) != g["rows_dev"]):
        print(f"gmmlu he: {len(test)} test / {len(dev)} dev rows, pinned {g['rows_test']} / {g['rows_dev']}", file=sys.stderr)
        return EXIT_USAGE
    shots = build_shots(dev)
    rows = {r["id"]: r for r in test}
    pop = set(pop_ids(test, g["pop2000_seed"], g["pop2000_n"]))
    config = {"eval": "gmmlu_he", "v": 1, "protocol": "chat", "system": SYSTEM, "shots": SHOTS, "max_tokens": 1,
              "temperature": 0, "chat_template_kwargs": THINK_OFF, "pop2000_seed": g["pop2000_seed"],
              "pop2000_n": g["pop2000_n"]}
    run = EvalRun("gmmlu_he", a, config,
                  eval_pins(a, pins, "gmmlu_he", {"gmmlu_he_test.jsonl": test_sha, "gmmlu_he_dev.jsonl": dev_sha}),
                  [r["id"] for r in test])

    def work(item_id: str) -> dict:
        row = rows[item_id]
        prompt = prompt_for(row, shots)
        res = run.server.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                              max_tokens=1, temperature=0, chat_template_kwargs=THINK_OFF)
        text, _ = chat_text(res)
        pred = predict(text)
        return {"pred": pred, "gold": row["answer"], "ok": int(pred == row["answer"]), "subject": row["subject"],
                "psha": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]}

    status = run.execute(work)

    def build():
        recs = run.records()
        items = {r["id"]: r["ok"] for r in recs}
        sub = {r["id"]: r["ok"] for r in recs if r["id"] in pop}
        by_subject: Dict[str, List[int]] = {}
        for r in recs:
            by_subject.setdefault(r["subject"], []).append(r["ok"])
        diag = {"unparsed": sum(1 for r in recs if r["pred"] not in LETTERS),
                "pred_counts": {k: sum(1 for r in recs if r["pred"] == k) for k in LETTERS},
                "prompts_sha256": canon_sha(sorted((r["id"], r["psha"]) for r in recs)),
                "per_subject": {s: [sum(v), len(v)] for s, v in sorted(by_subject.items())}}
        return ({"gmmlu_he": metric(items, n_full=len(test)),
                 "gmmlu_he_pop2000": metric(sub, n_full=len(pop))}, diag)

    return finish(run, status, build)


if __name__ == "__main__":
    sys.exit(main())
