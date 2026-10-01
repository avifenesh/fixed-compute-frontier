#!/usr/bin/env python3
"""G2 diagnosis: why does the from-scratch widened GDN solve parity at 4x but fade by 16x?

Variants, all with allow_neg_eigval=True, parity, 2 seeds:
  gdn_bf16_64     the G2 setting
  gdn_fp32_64     fp32 parameters (is it weight precision?)
  deltanet_bf16_64  no decay gate (is it alpha < 1?); fla DeltaNet has no fp32 path
  gdn_fp32_128    longer training sequences (is it training length?)
"""

import json
import random
import sys

import torch
import torch.nn.functional as F
from fla.models import DeltaNetConfig, DeltaNetForCausalLM, GatedDeltaNetConfig, GatedDeltaNetForCausalLM
from transformers import AutoTokenizer

import tasks

path, outp = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(path)
V = tasks.Vocab(tok)
rng0 = random.Random(0)
used = set()
for _ in range(50):
    ex = tasks.make("parity", 64, rng0, V)
    used |= set(ex.ids) | set(ex.labels)
remap = {tid: i for i, tid in enumerate(sorted(used))}


def batch(n, L, rng):
    exs = [tasks.make("parity", L, rng, V) for _ in range(n)]
    ids = torch.tensor([[remap[x] for x in e.ids] for e in exs]).cuda()
    pos = torch.tensor(exs[0].label_pos).cuda()
    lab = torch.tensor([[remap[x] for x in e.labels] for e in exs]).cuda()
    return ids, pos, lab


def build(kind, dtype):
    common = dict(hidden_size=128, num_hidden_layers=2, num_heads=2, head_dim=64, vocab_size=len(remap),
                  allow_neg_eigval=True, fuse_cross_entropy=False, hidden_ratio=4)
    if kind == "gdn":
        m = GatedDeltaNetForCausalLM(GatedDeltaNetConfig(expand_v=1.0, **common))
    else:
        m = DeltaNetForCausalLM(DeltaNetConfig(**common))
    return m.cuda().to(dtype)


def run(kind, dtype, L, seed, steps=3000):
    torch.manual_seed(seed)
    m = build(kind, dtype)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    rng = random.Random(seed)
    for _ in range(steps):
        ids, pos, lab = batch(64, L, rng)
        h = m.model(input_ids=ids).last_hidden_state[:, pos]
        loss = F.cross_entropy(m.lm_head(h).float().flatten(0, 1), lab.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
    m.eval()
    with torch.no_grad():
        ids, pos, lab = batch(128, 1024, random.Random(99))
        h = m.model(input_ids=ids).last_hidden_state[:, pos]
        acc = (m.lm_head(h).argmax(-1) == lab).float().mean(0)
    return {"w1": acc[:64].mean().item(), "w4": acc[64:256].mean().item(), "w16": acc[256:].mean().item(),
            "final_loss": loss.item()}


res = {}
VARIANTS = (("gdn_fp32_64", "gdn", torch.float32, 64),
            ("deltanet_bf16_64", "deltanet", torch.bfloat16, 64),  # fla DeltaNet has no fp32 path
            ("gdn_fp32_128", "gdn", torch.float32, 128))
skip = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else set()
for name, kind, dtype, L in VARIANTS:
    if name in skip:
        continue
    for seed in (0, 1):
        r = run(kind, dtype, L, seed)
        res[f"{name}/seed={seed}"] = r
        print(name, seed, r, flush=True)
json.dump(res, open(outp, "w"), indent=1)
