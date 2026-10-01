#!/usr/bin/env python3
"""G2c: how much swap data does a from-scratch widened-beta GDN need before swap switches on?

The retrofit learned parity but not swap in 1600 steps of mixed 5-task training (about 5k swap
sequences). G2 showed a tiny from-scratch GDN with beta in (0,2) learns swap (4x 0.86-0.89) after
3000 steps of batch 64 swap-only (192k sequences). This measures the learning curve, to tell an
under-budget retrofit from a structural failure. Validation every 250 steps on 128 sequences of 256.
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
for t in tasks.ALL_TASKS:
    for _ in range(200):
        ex = tasks.make(t, 64, rng0, V)
        used |= set(ex.ids) | set(ex.labels)
remap = {tid: i for i, tid in enumerate(sorted(used))}


def batch(task_list, n, L, rng):
    exs = [tasks.make(rng.choice(task_list), L, rng, V) for _ in range(n)]
    return exs


def fwd_loss(m, exs):
    Lmax = max(len(e.ids) for e in exs)
    ids = torch.zeros(len(exs), Lmax, dtype=torch.long)
    for i, e in enumerate(exs):
        ids[i, : len(e.ids)] = torch.tensor([remap[x] for x in e.ids])
    h = m.model(input_ids=ids.cuda()).last_hidden_state
    rows, cols, labs = [], [], []
    for i, e in enumerate(exs):
        rows += [i] * len(e.label_pos)
        cols += e.label_pos
        labs += [remap[x] for x in e.labels]
    return F.cross_entropy(m.lm_head(h[rows, cols]).float(), torch.tensor(labs).cuda())


@torch.no_grad()
def evaluate(m, task):
    m.eval()
    rng = random.Random(99)
    exs = [tasks.make(task, 256, rng, V) for _ in range(128)]
    ids = torch.tensor([[remap[x] for x in e.ids] for e in exs]).cuda()
    pos = torch.tensor(exs[0].label_pos).cuda()
    lab = torch.tensor([[remap[x] for x in e.labels] for e in exs]).cuda()
    acc = (m.lm_head(m.model(input_ids=ids).last_hidden_state[:, pos]).argmax(-1) == lab).float().mean(0)
    m.train()
    return {"w1": round(acc[:64].mean().item(), 3), "w4": round(acc[64:].mean().item(), 3),
            "first8": [round(a, 2) for a in acc[:8].tolist()]}


def run(name, task_list, bs, steps, neg, seed, every=250):
    torch.manual_seed(seed)
    cfg = GatedDeltaNetConfig(hidden_size=128, num_hidden_layers=2, num_heads=2, head_dim=64, expand_v=1.0,
                              vocab_size=len(remap), allow_neg_eigval=neg, fuse_cross_entropy=False,
                              hidden_ratio=4)
    m = GatedDeltaNetForCausalLM(cfg).cuda().to(torch.bfloat16)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    rng = random.Random(seed)
    curve, nswap = [], 0
    for step in range(1, steps + 1):
        exs = batch(task_list, bs, 64, rng)
        nswap += sum(e.task == "swap" for e in exs)
        loss = fwd_loss(m, exs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if step % every == 0:
            r = {"step": step, "swap_seqs": nswap, "loss": round(loss.item(), 4), "swap": evaluate(m, "swap")}
            if "parity" in task_list:
                r["parity"] = evaluate(m, "parity")
            curve.append(r)
            print(name, json.dumps(r), flush=True)
    return curve


res = {}
cfgs = [
    ("swaponly_b64_neg", ["swap"], 64, 3000, True),
    ("swaponly_b16_neg", ["swap"], 16, 8000, True),
    ("mixed5_b16_neg", list(tasks.ALL_TASKS), 16, 16000, True),
    ("swaponly_b16_pos", ["swap"], 16, 8000, False),
]
for name, tl, bs, steps, neg in cfgs:
    res[name] = run(name, tl, bs, steps, neg, 0)
    json.dump(res, open(outp, "w"), indent=1)
print("DONE")
