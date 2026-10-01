#!/usr/bin/env python3
"""G0: the widened arm at W = 0 is bit-identical to the untouched model; W reaches beta > 1 and gets gradient."""
import json
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import patch
import tasks

path, outp = sys.argv[1], sys.argv[2]
arm = sys.argv[3] if len(sys.argv) > 3 else "wide"
tok = AutoTokenizer.from_pretrained(path)
V = tasks.Vocab(tok)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0}).eval()
rng = random.Random(5)
seqs = [tasks.make(t, 200, rng, V).ids for t in tasks.ALL_TASKS]
seqs.append(tok.encode("The history of the Roman Empire spans many centuries and " * 20, add_special_tokens=False))
res = {}
with torch.no_grad():
    ref = [model(input_ids=torch.tensor([s]).cuda()).logits.float().cpu() for s in seqs]
    patch.install_for_arm(model, arm)
    got = [model(input_ids=torch.tensor([s]).cuda()).logits.float().cpu() for s in seqs]
res["bit_identical_at_w0"] = all(torch.equal(a, b) for a, b in zip(ref, got))
res["max_abs_diff_at_w0"] = max((a - b).abs().max().item() for a, b in zip(ref, got))
# nonzero W: beta must exceed 1 somewhere, and the gate must receive gradient
print(json.dumps(res), flush=True)
for p_ in model.parameters():
    p_.requires_grad_(False)
layers = patch.gdn_layers(model)
gates = [g for l in layers for g in (getattr(l, "negeig_w", None), getattr(l, "negeig_wa", None)) if g is not None]
for g in gates:
    torch.nn.init.normal_(g.weight, std=0.05)
    g.weight.requires_grad_(True)
patch.enable_stats(model, True)
out = model(input_ids=torch.tensor([seqs[0][:64]]).cuda()).logits
out.float().pow(2).mean().backward()
st = patch.collect_stats(model)
res["arm"] = arm
fb = st["beta_gt1_frac_per_layer"]
res["beta_gt1_frac_mean_random_w"] = sum(fb) / len(fb) if fb else None
res["gate_grad_nonzero"] = sum(int(g.weight.grad is not None and g.weight.grad.abs().sum().item() > 0) for g in gates)
res["n_gates"] = len(gates)
res["n_gdn_layers"] = len(layers)
res["pass"] = (res["bit_identical_at_w0"] and res["gate_grad_nonzero"] == len(gates)
               and (not fb or res["beta_gt1_frac_mean_random_w"] > 0))
json.dump(res, open(outp, "w"), indent=1)
print(json.dumps(res))
