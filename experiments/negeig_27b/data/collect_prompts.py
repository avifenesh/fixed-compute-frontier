#!/usr/bin/env python3
"""S4 chat-replay prompts: human-written only, open licences, screened against the evals.

The responses are generated later by the untouched Qwen3.8-27B itself (self-distillation keeps its own chat
behaviour); no hosted-model output is used as a prompt or a label (the Hebrew lane's terms reading, section 5 of its
KNOWLEDGE-20260928.md). Sources:
  databricks/databricks-dolly-15k   CC-BY-SA-3.0, written by Databricks employees (instruction plus optional context)
  OpenAssistant/oasst2              Apache-2.0, volunteer-written; first user turns of message trees, English and Hebrew
Screens: exact and near duplicates (normalised text), length 20 to 3,000 characters, and any 8-word overlap with
IFEval prompts (an eval we report).

  collect_prompts.py --out DIR [--n_en 6000]
"""
import argparse
import json
import os
import random
import re

from datasets import load_dataset


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def grams(t, n=8):
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_en", type=int, default=6000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ife = set()
    for r in load_dataset("google/IFEval", split="train"):
        ife |= grams(r["prompt"])
    rows = []
    for r in load_dataset("databricks/databricks-dolly-15k", split="train"):
        p = r["instruction"] + (("\n\n" + r["context"]) if r["context"].strip() else "")
        rows.append({"prompt": p, "lang": "en", "source": "dolly-15k", "license": "CC-BY-SA-3.0",
                     "category": r["category"]})
    oa = load_dataset("OpenAssistant/oasst2", split="train")
    for r in oa:
        if r["parent_id"] is None and r["role"] == "prompter" and r["lang"] in ("en", "he"):
            rows.append({"prompt": r["text"], "lang": r["lang"], "source": "oasst2", "license": "Apache-2.0",
                         "category": "open"})
    seen, kept, dropped = set(), [], {"dup": 0, "length": 0, "ifeval": 0}
    for r in rows:
        k = norm(r["prompt"])
        if k in seen:
            dropped["dup"] += 1
            continue
        seen.add(k)
        if not 20 <= len(r["prompt"]) <= 3000:
            dropped["length"] += 1
            continue
        if grams(r["prompt"]) & ife:
            dropped["ifeval"] += 1
            continue
        kept.append(r)
    rng = random.Random(0)
    en = [r for r in kept if r["lang"] == "en"]
    he = [r for r in kept if r["lang"] == "he"]
    rng.shuffle(en)
    out = en[: a.n_en] + he
    with open(os.path.join(a.out, "chat_prompts.jsonl"), "w") as f:
        for i, r in enumerate(out):
            f.write(json.dumps({"id": f"chat-{i:05d}", **r}, ensure_ascii=False) + "\n")
    print(json.dumps({"candidates": len(rows), "dropped": dropped, "kept_en": len(en), "kept_he": len(he),
                      "written": len(out), "by_source": {s: sum(r["source"] == s for r in out) for s in ("dolly-15k", "oasst2")}}))


if __name__ == "__main__":
    main()
