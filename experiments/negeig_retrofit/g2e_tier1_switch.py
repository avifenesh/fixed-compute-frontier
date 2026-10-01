#!/usr/bin/env python3
"""G2e: is codeswap learnable in this format, and does mixing tier-1 tasks block the switch?

Tiny from-scratch fla GDN as in G2d, padded batches (codeswap statements tokenize to different
lengths). Configs: codeswap only at batch 64 with beta in (0,2) and in (0,1); tier-1 mixed
(parity, swap, codeswap) at batch 192, about 64 per task. Switch = first eval with 4x accuracy
above 0.6 on the task. Evaluation: 128 sequences of 256 steps per task.
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
for t in tasks.TIER1:
    for _ in range(300):
        ex = tasks.make(t, 64, rng0, V)
        used |= set(ex.ids) | set(ex.labels)
remap = {tid: i + 1 for i, tid in enumerate(sorted(used))}  # 0 = pad
NV = len(remap) + 1


def forward_logits(m, exs):
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
    return m.lm_head(h[rows, cols]).float(), torch.tensor(labs).cuda()


@torch.no_grad()
def evaluate(m, task):
    m.eval()
    rng = random.Random(99)
    acc = torch.zeros(256)
    for s in range(0, 128, 32):
        exs = [tasks.make(task, 256, rng, V) for _ in range(32)]
        lg, lab = forward_logits(m, exs)
        acc += (lg.argmax(-1) == lab).float().view(32, 256).sum(0).cpu()
    acc /= 128
    m.train()
    return round(acc[:64].mean().item(), 3), round(acc[64:].mean().item(), 3)


def run(task_list, watch, bs, neg, seed, max_seqs, every=100):
    torch.manual_seed(seed)
    cfg = GatedDeltaNetConfig(hidden_size=128, num_hidden_layers=2, num_heads=2, head_dim=64, expand_v=1.0,
                              vocab_size=NV, allow_neg_eigval=neg, fuse_cross_entropy=False, hidden_ratio=4)
    m = GatedDeltaNetForCausalLM(cfg).cuda().to(torch.bfloat16)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    rng = random.Random(seed)
    curve, step, switched = [], 0, {}
    while step * bs < max_seqs:
        step += 1
        exs = [tasks.make(rng.choice(task_list), 64, rng, V) for _ in range(bs)]
        lg, lab = forward_logits(m, exs)
        F.cross_entropy(lg, lab).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if step % every == 0:
            row = {"step": step, "seqs": step * bs}
            for t in watch:
                row[t] = evaluate(m, t)
                if row[t][1] > 0.6 and t not in switched:
                    switched[t] = step * bs
            curve.append(row)
            print(json.dumps(row), flush=True)
            if len(switched) == len(watch):
                break
    return {"switch_seqs": switched, "curve": curve}


cfgs = [
    ("codeswap_b64_neg_s1", ["codeswap"], ["codeswap"], 64, True, 1, 400_000),
    ("codeswap_b64_pos_s1", ["codeswap"], ["codeswap"], 64, False, 1, 200_000),
    ("tier1mix_b192_neg_s1", list(tasks.TIER1), list(tasks.TIER1), 192, True, 1, 600_000),
    ("codeswap_b64_neg_s2", ["codeswap"], ["codeswap"], 64, True, 2, 400_000),
]
res = {}
for name, tl, watch, bs, neg, seed, mx in cfgs:
    print("==", name, flush=True)
    res[name] = run(tl, watch, bs, neg, seed, mx)
    print(name, "switch", res[name]["switch_seqs"], "last", res[name]["curve"][-1], flush=True)
    json.dump(res, open(outp, "w"), indent=1)
print("DONE")
