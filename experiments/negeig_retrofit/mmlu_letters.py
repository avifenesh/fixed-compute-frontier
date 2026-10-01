#!/usr/bin/env python3
"""Is the MMLU-Pro drop after task training a letter-bias artifact?

The state-tracking tasks answer with the letter tokens ' A'..' E', the same tokens MMLU-Pro's zero-shot letter
scoring reads. This re-runs general_eval's exact MMLU-Pro prompt and scoring, records each item's option-letter
logits, and reports: raw accuracy (as general_eval), the predicted-letter histogram against the gold histogram,
and accuracy after removing each letter's mean logit over the dataset (a per-model letter prior). If the
fine-tuned models recover to the base after calibration, the drop was letter bias, not lost knowledge.

Usage: mmlu_letters.py --model M --run_dir D [--ckpt trainable.pt]; writes <run_dir>/mmlu_letters.json
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import general_eval as G


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="trainable.pt")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model, arm = G.build(a.model, a.run_dir, ckpt=a.ckpt)
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    letter_ids = [tok.encode(" " + c, add_special_tokens=False)[0] for c in G.LETTERS]
    items = []
    for r in ds:
        opts = "\n".join(f"{G.LETTERS[i]}. {o}" for i, o in enumerate(r["options"]))
        prompt = (f"The following is a multiple choice question about {r['category']}. "
                  f"Answer with the letter of the correct option.\n\nQuestion: {r['question']}\n"
                  f"Options:\n{opts}\nAnswer:")
        items.append((tok.encode(prompt, add_special_tokens=False)[-3000:], len(r["options"]), r["answer_index"]))
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    dec, head = model.get_base_model().model, model.get_base_model().lm_head
    logits_all = [None] * len(items)
    for s in range(0, len(order), 8):
        idx = order[s : s + 8]
        chunk = [items[i] for i in idx]
        L = max(len(c[0]) for c in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, c in enumerate(chunk):
            ids[i, : len(c[0])] = torch.tensor(c[0])
            att[i, : len(c[0])] = 1
        h = dec(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
        last = h[torch.arange(len(chunk)), torch.tensor([len(c[0]) - 1 for c in chunk])]
        lg = head(last).float()[:, letter_ids].cpu()
        for i, k in enumerate(idx):
            logits_all[k] = lg[i]
    X = torch.stack(logits_all)  # [N, 10]
    n_opt = torch.tensor([c[1] for c in items])
    gold = torch.tensor([c[2] for c in items])
    mask = torch.arange(10)[None, :] < n_opt[:, None]
    raw_pred = X.masked_fill(~mask, -1e9).argmax(1)
    # per-letter prior: mean logit of each letter over items where it is a valid option
    prior = torch.stack([X[mask[:, j], j].mean() for j in range(10)])
    cal_pred = (X - prior).masked_fill(~mask, -1e9).argmax(1)
    out = {"arm": arm, "n": len(items), "raw_acc": (raw_pred == gold).float().mean().item(),
           "calibrated_acc": (cal_pred == gold).float().mean().item(),
           "pred_hist": torch.bincount(raw_pred, minlength=10).tolist(),
           "gold_hist": torch.bincount(gold, minlength=10).tolist(), "letter_prior": prior.tolist()}
    json.dump(out, open(os.path.join(a.run_dir, "mmlu_letters.json"), "w"))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k != "letter_prior"}))


if __name__ == "__main__":
    main()
