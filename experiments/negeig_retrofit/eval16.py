#!/usr/bin/env python3
"""16x check: accuracy in the 1x, 4x and 16x windows (steps 1-64, 65-256, 257-1024) on the validation
seed (20,000; the test set stays untouched), 1,024-step sequences, for trained adapters.

Usage: eval16.py --model M --run_dir D [--ckpt trainable.pt] [--n_seq 32]
Writes <run_dir>/eval16.json.
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

import general_eval
import patch
import tasks
import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="trainable.pt")
    ap.add_argument("--n_seq", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    V = tasks.Vocab(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0 = time.time()
    model, arm = general_eval.build(a.model, a.run_dir, ckpt=a.ckpt)
    patch.enable_stats(model, True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        res = train.eval_tasks(model, V, [1024], a.n_seq, a.batch, pad_id, seed=20_000)
    stats = patch.collect_stats(model)
    out = {"arm": arm, "ckpt": a.ckpt, "n_seq": a.n_seq, "seed": 20_000, "seconds": time.time() - t0}
    for t in tasks.ALL_TASKS:
        w = torch.tensor(res[f"{t}@1024/win"], dtype=torch.float)
        out[t] = {"w1": w[:, 0].mean().item(), "w4": w[:, 1].mean().item(), "w16": w[:, 2].mean().item(),
                  "per_step": res[f"{t}@1024"]}
    if stats["beta_gt1_frac_per_layer"]:
        out["beta_gt1_frac_mean"] = sum(stats["beta_gt1_frac_per_layer"]) / len(stats["beta_gt1_frac_per_layer"])
    json.dump(out, open(os.path.join(a.run_dir, "eval16.json"), "w"))
    print(json.dumps({"arm": arm, **{t: {k: round(out[t][k], 3) for k in ("w1", "w4", "w16")}
                                      for t in tasks.ALL_TASKS}, "seconds": round(out["seconds"])}))


if __name__ == "__main__":
    main()
