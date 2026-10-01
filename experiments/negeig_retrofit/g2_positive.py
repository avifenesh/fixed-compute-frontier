#!/usr/bin/env python3
"""G2 positive control: tiny from-scratch fla GatedDeltaNet, beta in (0,1) vs (0,2), on our task format.

Frozen pass rule: with allow_neg_eigval, parity accuracy in the 16x window (steps 257..1024)
reaches >= 0.95 on at least one of two seeds, and without it stays <= 0.60 on both.
Swap is reported, not gated.
"""

import json
import random
import sys

import torch
import torch.nn.functional as F
from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
from transformers import AutoTokenizer

import tasks

path, outp = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(path)
V = tasks.Vocab(tok)
rng0 = random.Random(0)
used = set()
for t in ("parity", "swap"):
    for _ in range(50):
        ex = tasks.make(t, 64, rng0, V)
        used |= set(ex.ids) | set(ex.labels)
remap = {tid: i for i, tid in enumerate(sorted(used))}


def batch(task, n, L, rng):
    exs = [tasks.make(task, L, rng, V) for _ in range(n)]
    ids = torch.tensor([[remap[x] for x in e.ids] for e in exs]).cuda()
    pos = torch.tensor(exs[0].label_pos).cuda()
    lab = torch.tensor([[remap[x] for x in e.labels] for e in exs]).cuda()
    return ids, pos, lab


def run(task, neg, seed, steps=3000):
    torch.manual_seed(seed)
    cfg = GatedDeltaNetConfig(hidden_size=128, num_hidden_layers=2, num_heads=2, head_dim=64, expand_v=1.0,
                              vocab_size=len(remap), allow_neg_eigval=neg, fuse_cross_entropy=False,
                              hidden_ratio=4)
    m = GatedDeltaNetForCausalLM(cfg).cuda().to(torch.bfloat16)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    rng = random.Random(seed)
    for step in range(steps):
        ids, pos, lab = batch(task, 64, 64, rng)
        h = m.model(input_ids=ids).last_hidden_state[:, pos]
        loss = F.cross_entropy(m.lm_head(h).float().flatten(0, 1), lab.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
    m.eval()
    with torch.no_grad():
        ids, pos, lab = batch(task, 128, 1024, random.Random(99))
        h = m.model(input_ids=ids).last_hidden_state[:, pos]
        acc = (m.lm_head(h).argmax(-1) == lab).float().mean(0)
    return {"w1": acc[:64].mean().item(), "w4": acc[64:256].mean().item(), "w16": acc[256:].mean().item(),
            "final_loss": loss.item()}


res = {}
for task in ("parity", "swap"):
    for neg in (False, True):
        for seed in (0, 1):
            r = run(task, neg, seed)
            res[f"{task}/neg={neg}/seed={seed}"] = r
            print(task, neg, seed, r, flush=True)
par_pos = max(res[f"parity/neg=True/seed={s}"]["w16"] for s in (0, 1))
par_neg = max(res[f"parity/neg=False/seed={s}"]["w16"] for s in (0, 1))
res["pass"] = par_pos >= 0.95 and par_neg <= 0.60
json.dump(res, open(outp, "w"), indent=1)
print("G2 PASS" if res["pass"] else "G2 FAIL", par_pos, par_neg)
