#!/usr/bin/env python3
"""How much does a trained adapter change the model on ordinary text?

For one adapter, over held-out ordinary text (clean replay test chunks: web + Python; WikiText test chunks), report:
  - KL(base || adapter) per token, the untouched base's next-token distribution against the adapter's;
  - the adapter's own NLL and the base's NLL on the same tokens;
  - for gated arms, how often the widened gate fires on ordinary text (beta > 1, mean |t|) per layer,
    next to how often it fires on task tokens.
The base is the same checkpoint with the gates and LoRA removed (built in the same process).

Usage: gate_kl.py --model M --run_dir D [--ckpt trainable.pt] [--n_chunks 48]; writes <run_dir>/gate_kl.json
"""
import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import general_eval as G
import patch
import tasks


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="trainable.pt")
    ap.add_argument("--n_chunks", type=int, default=48)
    ap.add_argument("--clean", default="/workspace/negeig/replay_clean.pt")
    ap.add_argument("--wikitext", default="/workspace/negeig/replay_wikitext103.pt")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model, arm = G.build(a.model, a.run_dir, ckpt=a.ckpt)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    corpora = {"clean": torch.load(a.clean)["test"][: a.n_chunks].long(),
               "wikitext": torch.load(a.wikitext)["test"][: a.n_chunks].long()}
    out = {"arm": arm}
    for name, X in corpora.items():
        patch.enable_stats(model, True)
        kl_sum = nll_a = nll_b = 0.0
        n = 0
        for i in range(0, X.shape[0], 4):
            ids = X[i : i + 4].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                la = model(input_ids=ids).logits[:, :-1].float()
                lb = base(input_ids=ids).logits[:, :-1].float()
            tgt = ids[:, 1:]
            pa, pb = F.log_softmax(la, -1), F.log_softmax(lb, -1)
            kl_sum += (pb.exp() * (pb - pa)).sum(-1).sum().item()
            nll_a += -pa.gather(-1, tgt[..., None]).sum().item()
            nll_b += -pb.gather(-1, tgt[..., None]).sum().item()
            n += tgt.numel()
        st = patch.collect_stats(model)
        patch.enable_stats(model, False)
        fb = st["beta_gt1_frac_per_layer"]
        out[name] = {"kl_per_token": kl_sum / n, "nll_adapter": nll_a / n, "nll_base": nll_b / n,
                     "beta_gt1_frac_mean": sum(fb) / len(fb) if fb else None,
                     "beta_gt1_frac_max_layer": max(fb) if fb else None, "beta_gt1_per_layer": fb}
    # gate use on task tokens, for reference
    V = tasks.Vocab(tok)
    rng = random.Random(4242)
    patch.enable_stats(model, True)
    for t in ("parity", "swap", "codeswap"):
        exs = [tasks.make(t, 64, rng, V) for _ in range(8)]
        L = max(len(e.ids) for e in exs)
        ids = torch.full((8, L), tok.pad_token_id or 0, dtype=torch.long)
        for j, e in enumerate(exs):
            ids[j, : len(e.ids)] = torch.tensor(e.ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=ids.cuda())
    st = patch.collect_stats(model)
    fb = st["beta_gt1_frac_per_layer"]
    out["tasks"] = {"beta_gt1_frac_mean": sum(fb) / len(fb) if fb else None,
                    "beta_gt1_frac_max_layer": max(fb) if fb else None, "beta_gt1_per_layer": fb}
    json.dump(out, open(os.path.join(a.run_dir, "gate_kl.json"), "w"))
    brief = {k: ({kk: (round(vv, 4) if isinstance(vv, float) else None if vv is None else "...")
                  for kk, vv in v.items() if kk != "beta_gt1_per_layer"} if isinstance(v, dict) else v)
             for k, v in out.items()}
    print(json.dumps(brief))


if __name__ == "__main__":
    main()
