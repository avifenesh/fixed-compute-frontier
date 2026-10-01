#!/usr/bin/env python3
"""Does the widened gate fire on option-letter tokens in MMLU-Pro prompts?

Per GDN layer and head, recompute beta on MMLU-Pro letter-format prompts and compare the fraction of beta > 1 at
option-letter tokens ("A".."J" as option markers and " A".." J") against all other tokens. The swap task feeds
letter tokens as inputs (the ball's starting cup), so a tracking circuit keyed on letters would interfere with
letter-choice scoring.

Usage: gate_on_letters.py --model M --run_dir D [--n 200]; prints and writes <run_dir>/gate_on_letters.json
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import general_eval as G
import train


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="trainable.pt")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model, arm = G.build(a.model, a.run_dir, ckpt=a.ckpt)
    dec, _ = train.core(model)
    layers = [(i, l.linear_attn) for i, l in enumerate(dec.layers) if hasattr(l, "linear_attn")]
    letter_ids = set()
    for c in G.LETTERS:
        for s in (c, " " + c):
            e = tok.encode(s, add_special_tokens=False)
            if len(e) == 1:
                letter_ids.add(e[0])
    caught = {}
    hooks = [m.register_forward_pre_hook(lambda mod, args, kw, i=i: caught.__setitem__(i, kw.get("hidden_states", args[0] if args else None).detach()), with_kwargs=True)
             for i, m in layers]
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    tot = {"letter": [0, 0], "other": [0, 0]}  # [beta>1 count, token-head count]
    per_layer = {i: {"letter": [0, 0], "other": [0, 0]} for i, _ in layers}
    for k in range(0, len(ds), max(1, len(ds) // a.n))[: a.n]:
        r = ds[k]
        opts = "\n".join(f"{G.LETTERS[i]}. {o}" for i, o in enumerate(r["options"]))
        prompt = (f"The following is a multiple choice question about {r['category']}. "
                  f"Answer with the letter of the correct option.\n\nQuestion: {r['question']}\n"
                  f"Options:\n{opts}\nAnswer:")
        ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)[-3000:]]).cuda()
        is_letter = torch.tensor([int(t) in letter_ids for t in ids[0].tolist()], device=ids.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dec(input_ids=ids)
            for i, m in layers:
                h = caught[i]
                b = m.in_proj_b(h)
                beta = m._negeig_beta(b, h).float()[0] if hasattr(m, "_negeig_beta") else b.sigmoid().float()[0]
                gt = beta > 1.0  # [T, heads]
                for key, mask in (("letter", is_letter), ("other", ~is_letter)):
                    c, n = gt[mask].sum().item(), gt[mask].numel()
                    tot[key][0] += c; tot[key][1] += n
                    per_layer[i][key][0] += c; per_layer[i][key][1] += n
    for hk in hooks:
        hk.remove()
    frac = {k: v[0] / max(1, v[1]) for k, v in tot.items()}
    layer_ratio = {i: (d["letter"][0] / max(1, d["letter"][1])) / max(1e-9, d["other"][0] / max(1, d["other"][1]))
                   for i, d in per_layer.items()}
    top = sorted(layer_ratio.items(), key=lambda x: -x[1])[:5]
    out = {"arm": arm, "beta_gt1_frac": frac, "letter_over_other": frac["letter"] / max(1e-9, frac["other"]),
           "top_layers_letter_over_other": top}
    json.dump(out, open(os.path.join(a.run_dir, "gate_on_letters.json"), "w"))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
