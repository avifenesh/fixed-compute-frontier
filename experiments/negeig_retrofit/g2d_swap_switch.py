#!/usr/bin/env python3
"""G2d: what shortens the swap plateau? Tiny from-scratch GDN, beta in (0,2), swap only.

G2c showed a long plateau (first 4-6 swaps right, then chance) that ended at 192k sequences with batch 64 and
had not ended at 128k with batch 16, one seed each. This measures the switch step across seeds, training
length, batch size and learning rate, to pick a retrofit recipe. Switch = first eval with 4x accuracy > 0.6.
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
for _ in range(200):
    ex = tasks.make("swap", 64, rng0, V)
    used |= set(ex.ids) | set(ex.labels)
remap = {tid: i for i, tid in enumerate(sorted(used))}


def tensors(exs):
    ids = torch.tensor([[remap[x] for x in e.ids] for e in exs]).cuda()
    pos = torch.tensor(exs[0].label_pos).cuda()
    lab = torch.tensor([[remap[x] for x in e.labels] for e in exs]).cuda()
    return ids, pos, lab


@torch.no_grad()
def evaluate(m):
    m.eval()
    rng = random.Random(99)
    ids, pos, lab = tensors([tasks.make("swap", 256, rng, V) for _ in range(128)])
    acc = (m.lm_head(m.model(input_ids=ids).last_hidden_state[:, pos]).argmax(-1) == lab).float().mean(0)
    m.train()
    return round(acc[:64].mean().item(), 3), round(acc[64:].mean().item(), 3)


def run(bs, L, lr, seed, max_seqs=400_000, every=100):
    torch.manual_seed(seed)
    cfg = GatedDeltaNetConfig(hidden_size=128, num_hidden_layers=2, num_heads=2, head_dim=64, expand_v=1.0,
                              vocab_size=len(remap), allow_neg_eigval=True, fuse_cross_entropy=False,
                              hidden_ratio=4)
    m = GatedDeltaNetForCausalLM(cfg).cuda().to(torch.bfloat16)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    curve, step = [], 0
    while step * bs < max_seqs:
        step += 1
        ids, pos, lab = tensors([tasks.make("swap", L, rng, V) for _ in range(bs)])
        h = m.model(input_ids=ids).last_hidden_state[:, pos]
        loss = F.cross_entropy(m.lm_head(h).float().flatten(0, 1), lab.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if step % every == 0:
            w1, w4 = evaluate(m)
            curve.append((step, step * bs, w1, w4))
            if w4 > 0.6:
                return {"switch_step": step, "switch_seqs": step * bs, "switch_tokens": step * bs * L, "curve": curve}
    return {"switch_step": None, "switch_seqs": None, "switch_tokens": None, "curve": curve}


cfgs = [(64, 64, 1e-3, s) for s in (1, 2, 3)] + [(64, 128, 1e-3, s) for s in (1, 2)] + [
    (256, 64, 1e-3, 1), (64, 64, 3e-3, 1), (16, 128, 1e-3, 1)]
res = {}
for bs, L, lr, seed in cfgs:
    k = f"b{bs}_L{L}_lr{lr:g}_s{seed}"
    r = run(bs, L, lr, seed)
    res[k] = r
    print(k, {x: r[x] for x in ("switch_step", "switch_seqs", "switch_tokens")}, "last", r["curve"][-1], flush=True)
    json.dump(res, open(outp, "w"), indent=1)
print("DONE")
