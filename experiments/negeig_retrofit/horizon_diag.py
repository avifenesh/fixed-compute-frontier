#!/usr/bin/env python3
"""Why does the widened retrofit stop generalizing after 2-3x the training length?

For a trained adapter, capture every Gated DeltaNet layer's input on task sequences and recompute, per head
and per token, beta (widened), the decay alpha = exp(-exp(A_log) * softplus(a + dt_bias)) and the eigenvalue
along the key, lambda = alpha * (1 - beta). A perfect reflection is lambda = -1; a perfect keep is lambda = +1
with the other directions at alpha = 1. Per-token |lambda| < 1 or alpha < 1 compounds over steps.

Usage: horizon_diag.py --model M --run_dir D --ckpt F --task swap|parity|codeswap [--n_seq 16 --L 256]
Prints the heads that reflect (mean beta > 1.5 on the task's step tokens) with alpha and lambda statistics.
"""

import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import general_eval
import tasks
import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="trainable.pt")
    ap.add_argument("--task", required=True)
    ap.add_argument("--n_seq", type=int, default=16)
    ap.add_argument("--L", type=int, default=256)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    V = tasks.Vocab(tok)
    model, arm = general_eval.build(a.model, a.run_dir, ckpt=a.ckpt)
    dec, _ = train.core(model)
    layers = [(i, l.linear_attn) for i, l in enumerate(dec.layers) if hasattr(l, "linear_attn")]
    caught = {}
    hooks = [m.register_forward_pre_hook(lambda mod, args, kw, i=i: caught.__setitem__(i, (kw.get("hidden_states", args[0] if args else None)).detach()), with_kwargs=True)
             for i, m in layers]
    rng = random.Random(777)
    exs = [tasks.make(a.task, a.L, rng, V) for _ in range(a.n_seq)]
    ids = torch.tensor([e.ids for e in exs]).cuda()
    pos = torch.tensor(exs[0].label_pos).cuda()  # step tokens (the token whose logits carry the answer)
    flip_mask = None
    if a.task == "parity":  # bit 1 must flip, bit 0 must keep
        flip_mask = ids[:, pos] == V.bit[1]
    out = {"arm": arm, "task": a.task, "heads": []}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        dec(input_ids=ids)
        for i, m in layers:
            h = caught[i]
            b = m.in_proj_b(h)
            aa = m.in_proj_a(h)
            beta = m._negeig_beta(b, h).float() if hasattr(m, "_negeig_beta") else b.sigmoid().float()
            g = m._negeig_g(aa, h) if hasattr(m, "_negeig_g") else -m.A_log.float().exp() * F.softplus(aa.float() + m.dt_bias.float())
            alpha = g.exp()
            lam = alpha * (1 - beta)
            B, A, Lm = beta[:, pos], alpha[:, pos], lam[:, pos]  # [n, steps, heads]
            for hd in range(B.shape[-1]):
                bh, ah, lh = B[..., hd], A[..., hd], Lm[..., hd]
                if flip_mask is not None:
                    fb = bh[flip_mask]
                    if fb.mean() < 1.5:
                        continue
                    rec = {"layer": i, "head": hd, "beta_flip": fb.mean().item(), "beta_keep": bh[~flip_mask].mean().item(),
                           "lam_flip_mean": lh[flip_mask].mean().item(), "lam_flip_min_abs": lh[flip_mask].abs().min().item(),
                           "lam_keep_mean": lh[~flip_mask].mean().item(), "alpha_mean": ah.mean().item(),
                           "alpha_min": ah.min().item()}
                else:
                    if bh.mean() < 1.5:
                        continue
                    rec = {"layer": i, "head": hd, "beta_mean": bh.mean().item(), "beta_min": bh.min().item(),
                           "lam_mean": lh.mean().item(), "lam_abs_p10": lh.abs().flatten().quantile(0.1).item(),
                           "alpha_mean": ah.mean().item(), "alpha_min": ah.min().item()}
                out["heads"].append(rec)
    for hk in hooks:
        hk.remove()
    out["heads"].sort(key=lambda r: -r.get("beta_flip", r.get("beta_mean", 0)))
    # horizon implied by per-token magnitude loss: steps until |lambda|^n < 0.2 (or alpha^n for keeps)
    for r in out["heads"]:
        mag = abs(r.get("lam_flip_mean", r.get("lam_mean")))
        r["steps_to_0.2"] = math.log(0.2) / math.log(mag) if 0 < mag < 1 else None
    json.dump(out, open(os.path.join(a.run_dir, f"horizon_{a.task}.json"), "w"), indent=1)
    print(f"{arm} {a.task}: {len(out['heads'])} reflecting heads")
    for r in out["heads"][:12]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
