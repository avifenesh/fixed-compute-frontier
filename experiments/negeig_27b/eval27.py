#!/usr/bin/env python3
"""Full S1 evaluation of one adapter (or the untouched model): teacher-forced exact match on every eval session.

Same scoring as train27.py's checkpoint eval (an answer counts when every answer token and its <|im_end|> is the argmax
given the gold history; with the Qwen3.8 think-off template this equals greedy decoding of that answer under the
canonical tokenization, so it is a lower bound on string-level greedy accuracy). Writes one record per answer
(domain, lang, n_events, after_events, question kind, ok) for position curves, and a summary. Sessions are split over
ranks; the longest eval sessions are about 34k tokens, one forward each.

  torchrun --standalone --nproc_per_node 8 eval27.py --model M --arm wide --trainable RUN/trainable_000600.pt \
      --s1 DATA/s1 --out RUN/eval_s1_600          # --arm ctrl --trainable none scores the untouched model
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from train27 import build_model, core, encode_chat, fwd, load_jsonl, patch, row_messages, spans_of


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=sorted(patch.ARMS), required=True)
    ap.add_argument("--trainable", required=True, help="trainable_*.pt from train27.py, or 'none' for the untouched model")
    ap.add_argument("--s1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="", help="comma list of n_events to keep (default all)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--grad_ckpt", type=int, default=0)
    args = ap.parse_args()

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model, _ = build_model(args, device)
    if args.trainable != "none":
        sd = torch.load(args.trainable, map_location="cpu")
        params = dict(model.named_parameters())
        missing = [k for k in params if params[k].requires_grad and k not in sd]
        assert not missing, f"trainable file lacks {missing[:3]}"
        extra = [k for k in sd if k not in params or not params[k].requires_grad]
        assert not extra, f"trainable file has tensors this --arm/--rank lacks: {extra[:3]}"
        for k, v in sd.items():
            params[k].copy_(v.to(params[k].dtype))
    model.eval()
    dec, head = core(model)

    keep = {int(x) for x in args.lengths.split(",")} if args.lengths else None
    sessions = []
    for p in sorted(glob.glob(os.path.join(args.s1, "*.eval.jsonl"))):
        sessions += [r for r in load_jsonl([p]) if keep is None or r["n_events"] in keep]
    sessions.sort(key=lambda r: (r["domain"], r["lang"], r["n_events"], r["seed"]))
    # longest first within each rank's share balances the tail; the order does not change any score
    mine = sorted(sessions[rank::world], key=lambda r: -r["n_events"])

    recs, fw_tokens = [], []
    patch.enable_stats(model, True)
    for r in mine:
        ids, mask = encode_chat(tok, row_messages(r))
        x = torch.tensor([ids], device=device)
        fw_tokens.append(len(ids))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.dtype == "bf16"):
            h = fwd(dec, x)[0]  # use_cache=False: eval mode would otherwise build a 34k-token KV cache per session
            sp = spans_of(mask)
            assert len(sp) == len(r["turns"]), (len(sp), len(r["turns"]))
            for (a, b), t in zip(sp, r["turns"]):
                lp = torch.log_softmax(head(h[a - 1 : b - 1]).float(), -1)
                y = x[0, a:b]
                recs.append({"domain": r["domain"], "lang": r["lang"], "seed": r["seed"], "n_events": r["n_events"],
                             "after_events": t["meta"]["after_events"], "kind": t["meta"].get("kind"),
                             "ok": bool((lp.argmax(-1) == y).all()), "nll": -lp.gather(-1, y[:, None]).sum().item(),
                             "n_tok": b - a})
    # per layer: sum over this rank's sessions of (beta > 1 fraction x tokens); token-weighted across ranks below,
    # the same weighting as train27.gate_rate_global (patch.py appends one mean per forward per layer)
    st = [sum(f * w for (f, _), w in zip(layer._negeig_stats, fw_tokens))
          for layer in patch.gdn_layers(model) if getattr(layer, "_negeig_stats", None) is not None]
    gathered = [None] * world
    dist.all_gather_object(gathered, {"recs": recs, "gate": st, "tokens": sum(fw_tokens)})
    if rank == 0:
        allr = [x for g in gathered for x in g["recs"]]
        with open(out / "answers.jsonl", "w") as f:
            for x in allr:
                f.write(json.dumps(x) + "\n")

        def acc(sel):
            xs = [x for x in allr if sel(x)]
            if not xs:
                return None
            return {"acc": sum(x["ok"] for x in xs) / len(xs), "n": len(xs),
                    "answer_nll": sum(x["nll"] for x in xs) / sum(x["n_tok"] for x in xs)}

        summ = {"trainable": args.trainable, "arm": args.arm, "n_sessions": len(sessions), "n_answers": len(allr),
                "all": acc(lambda x: True),
                "le256": acc(lambda x: x["after_events"] <= 256), "gt256": acc(lambda x: x["after_events"] > 256),
                "gt512": acc(lambda x: x["after_events"] > 512)}
        for key in ("domain", "lang", "n_events"):
            for v in sorted({x[key] for x in allr}):
                summ[f"{key}={v}"] = acc(lambda x, v=v, key=key: x[key] == v)
                summ[f"{key}={v}:gt256"] = acc(lambda x, v=v, key=key: x[key] == v and x["after_events"] > 256)
        gates = [g["gate"] for g in gathered if g["gate"]]
        ntok = sum(g["tokens"] for g in gathered)
        if gates and ntok:
            summ["gate_rate_per_layer"] = [sum(c) / ntok for c in zip(*gates)]
            summ["gate_rate"] = sum(summ["gate_rate_per_layer"]) / len(summ["gate_rate_per_layer"])
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        print(json.dumps({k: v for k, v in summ.items() if k != "gate_rate_per_layer"}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
