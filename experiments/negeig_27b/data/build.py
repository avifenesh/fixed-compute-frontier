#!/usr/bin/env python3
"""Build S1 state-tracking sessions for one or more domains, verify every answer, write JSONL.

  build.py --domains custody --split train --n 2000 --out ../out/s1 [--tokenizer DIR]

Each line of <out>/<domain>.<split>.jsonl is a Session (common.Session) plus "messages" (system/user/assistant,
think-off at train time). Train lengths are drawn from TRAIN_BUCKETS, eval lengths from EVAL_BUCKETS; the
Hebrew share applies to domains that have Hebrew templates. Seeds are disjoint between splits and domains.
With --tokenizer, token counts are measured through the model's chat template (enable_thinking=False).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_session, to_messages, verify  # noqa: E402

TRAIN_BUCKETS = [32, 64, 128, 192, 256]
EVAL_BUCKETS = [32, 256, 512, 1024]
HE_SHARE = 0.25
DOMAINS = ["custody", "toggles", "fsys", "codetrace", "ops", "orders"]
SPLIT_BASE = {"train": 0, "eval": 50_000_000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--n", type=int, required=True, help="sessions per domain (eval: per domain and length)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--max_batch", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    for di, name in enumerate(a.domains.split(",")):
        mod = importlib.import_module(name)
        langs = ["en", "he"] if "he" in mod.Sim.SYSTEM else ["en"]
        rng = random.Random(SPLIT_BASE[a.split] + 1_000_003 * di)
        jobs = []
        if a.split == "train":
            for i in range(a.n):
                jobs.append((rng.choice(TRAIN_BUCKETS), "he" if len(langs) > 1 and rng.random() < HE_SHARE else "en"))
        else:
            for L in EVAL_BUCKETS:
                for i in range(a.n):
                    jobs.append((L, "he" if len(langs) > 1 and i % 4 == 3 else "en"))
        path = os.path.join(a.out, f"{name}.{a.split}.jsonl")
        toks, turns = [], []
        with open(path, "w") as f:
            for j, (L, lang) in enumerate(jobs):
                seed = SPLIT_BASE[a.split] + 1_000_003 * di + j
                s = make_session(mod.Sim, name, seed, L, lang, a.split, a.max_batch)
                verify(s, mod.replay)
                msgs = to_messages(s)
                row = json.loads(s.to_json())
                row["messages"] = msgs
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                turns.append(len(s.turns))
                if tok is not None and j % max(1, len(jobs) // 200) == 0:
                    toks.append(len(tok(tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False), add_special_tokens=False).input_ids))
        msg = f"{name} {a.split}: {len(jobs)} sessions verified -> {path}; turns mean {statistics.mean(turns):.1f}"
        if toks:
            msg += f"; tokens (sampled {len(toks)}) mean {statistics.mean(toks):.0f} max {max(toks)}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
