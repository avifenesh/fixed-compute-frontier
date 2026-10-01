#!/usr/bin/env python3
"""Tier-1 mechanism probe for the Qwen3.8-27B retrofit: wide (LoRA + widened-beta gate) against ctrl (LoRA alone) on
the 4B/9B lane's tier-1 tasks, trained with data parallelism and evaluated the way the 4B lane was.

Recipe = experiments/negeig_retrofit/box_main_4b.txt run through train.py, on the 27B:
  data    per step 16 parity + 64 swap + 64 codeswap sequences of 64 task steps (tasks.py), drawn from ONE
          random.Random(seed) in train.py's call order. The Qwen3.8 and Qwen3.5 tokenizers share the vocabulary and every
          task token id, so a seed reproduces train.py's stream token for token, on any world size and after a resume
          (the stream is fast-forwarded on resume, about 7 ms a step). Plus --replay_batch LM replay chunks of
          --replay_len tokens a step: chunk j of step t is train[perm[t * replay_batch + j]] with perm = randperm(seed),
          which is train.py's replay_chunks selection (wrapping only past the end of the file).
  loss    full-vocabulary cross-entropy at every label position (labels never enter the stream), averaged over all label
          positions of the step, plus --w_lm times the mean next-token loss of the replay chunks. That is train.py's
          objective: its per-micro-batch weighting reduces to these global means because every sequence carries
          --train_len labels.
  update  AdamW (0.9, 0.95), no weight decay, LoRA at --lr and the gate at --w_lr (1e-4 tied), linear warmup over --warmup
          steps (60, train.py's 5% of 1,200), then constant, so a later --steps extends a run seamlessly. Gradient
          clipping per group at --clip, as train27 does; --clip_joint gives train.py's joint clip.
  model   train27.build_model: LoRA r16 on the Gated DeltaNet projections, the zero-init fp32 gate for wide, LoRA init
          paired across arms, rank 0's tensors broadcast.
Data parallelism: every rank builds the whole step, cuts it into micro-batches that do not depend on the world size
(single-task groups of max(1, --micro_tokens // length) sequences, train.py's rule, and replay groups of --replay_micro
chunks), assigns whole micro-batches to ranks longest-first (train27.lpt), scales each micro-batch's summed loss by the
step's global count and all-reduces the flattened gradient once. The world size changes only float summation order.

Evaluation is train.py's eval_tasks (argmax over each task's label set at every label position; the sequences of one
length come from random.Random(seed + 7919 * L + 104729 * task_index); per-sequence accuracy in the 1x / 4x / 16x
windows = steps 1-64 / 65-256 / 257-1024), sharded over ranks by whole batches, so every number is the same on any
world size:
  val   every --eval_every steps and at step 0: --val_n sequences of --val_len steps at --val_seed (train.py's eval_val:
        32 x 256, seed 20,000). Per task w1, w4, first8 and the label NLL, tier-1 means, the gate rate (beta > 1) per layer
        on task tokens, and train27.eval_lm on --text_eval (replay NLL, KL to the untouched model, gate rate on text).
        Event "eval" in log.jsonl.
  test  at step 0 (the untouched model: LoRA B = 0 and W = 0, bit-identical by G0) and at --steps (and --test_steps):
        --test_n sequences of --test_len steps at --test_seed (train.py's final test: 128 x 1,024, seed 10,000, the set
        behind the 4B numbers). Written to test_stepNNNNNN.json in train.py's result.json schema ("tasks", "nll",
        "negeig_stats"), so analyze.py's readers work on it. Event "test" in log.jsonl.
Checkpoints, resume and the replica check are train27's (save_ckpt, pick_resume, eval_logged, replicas_differ). A resume
re-runs the val and test evals of the checkpoint step when they are missing. complete.json records the step reached.
--eval_only TRAINABLE|none scores one adapter (or the untouched model) on the val and test sets and exits.

  torchrun --standalone --nproc_per_node 4 train_t1.py --model M --arm wide --seed 0 --out RUN --lm_replay R.pt
  python decide.py --root RUNS/TAG --seed 0 --step 1200          (the pre-registered decision rule)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
E27 = HERE.parent.parent
sys.path.insert(0, str(E27))
import train27 as T27  # noqa: E402  (also puts experiments/negeig_retrofit on sys.path and imports patch)

sys.path.insert(0, str(E27.parent / "negeig_retrofit"))
import patch  # noqa: E402
import tasks  # noqa: E402

WINDOWS = ((0, 64), (64, 256), (256, 1024))  # train.py eval_tasks: steps 1-64, 65-256, 257-1024


def label_sets(V):
    return {"parity": V.letter[:2], "z3": V.letter[:3], "swap": V.letter, "s5full": V.letter, "codeswap": V.letter}


def parse_counts(s):
    out = [(t, int(n)) for t, n in (x.split(":") for x in s.split(","))]
    assert out and all(t in tasks.ALL_TASKS and n > 0 for t, n in out), out
    return out


# ------------------------------------------------------------------ data
class Stream:
    """train.py's task stream: one random.Random(seed), per step `for task, n in counts: n x tasks.make(task, L)`.
    batch(step) returns that step's [(task, examples)]; asking for a later step fast-forwards, an earlier one replays."""

    def __init__(self, V, seed, counts, train_len):
        self.V, self.seed, self.counts, self.L = V, seed, counts, train_len
        self.rng, self.next = random.Random(seed), 0

    def _gen(self):
        return [(t, [tasks.make(t, self.L, self.rng, self.V) for _ in range(n)]) for t, n in self.counts]

    def batch(self, step):
        if step < self.next:
            self.rng, self.next = random.Random(self.seed), 0
        while self.next < step:
            self._gen()
            self.next += 1
        self.next += 1
        return self._gen()


def replay_perm(n, seed):
    """train.py replay_chunks: torch.randperm(n) under a generator seeded with the run seed."""
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))


def step_chunks(train, perm, step, k):
    return [train[int(perm[(step * k + j) % len(perm)])].tolist() for j in range(k)]


def plan_step(groups, chunks, micro_tokens, replay_micro):
    """The step's micro-batches, identical on every world size: ("t1", examples) single-task groups of
    max(1, micro_tokens // length) sequences (train.py's rule; micro_tokens 0 = one group per task) and ("lm", chunks)
    groups of replay_micro chunks. Each comes with its padded-token cost for the rank assignment."""
    mbs = []
    for _, exs in groups:
        per = max(1, micro_tokens // max(len(e.ids) for e in exs)) if micro_tokens else len(exs)
        mbs += [("t1", exs[i : i + per]) for i in range(0, len(exs), per)]
    if chunks:
        mbs += [("lm", chunks[i : i + replay_micro]) for i in range(0, len(chunks), replay_micro)]
    cost = [len(b) * max(len(e.ids) if k == "t1" else len(e) for e in b) for k, b in mbs]
    return mbs, cost


# ------------------------------------------------------------------ loss
def t1_loss(model, exs, device):
    """Summed full-vocabulary cross-entropy at the label positions of a micro-batch and the label count. Head logits in
    checkpointed chunks of 1024 rows, as train.py does."""
    dec, head = T27.core(model)
    h = T27.hidden(dec, [e.ids for e in exs], device)
    rows, cols, labs = [], [], []
    for i, e in enumerate(exs):
        rows += [i] * len(e.label_pos)
        cols += e.label_pos
        labs += e.labels
    hs = h[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]
    y = torch.tensor(labs, device=device)

    def ce(x, t):
        return F.cross_entropy(head(x).float(), t, reduction="sum")

    tot = sum(checkpoint(ce, hs[s : s + 1024], y[s : s + 1024], use_reentrant=False) for s in range(0, len(labs), 1024))
    return tot, len(labs)


# ------------------------------------------------------------------ evaluation
def gate_layers(model, tokens, device, world):
    """Per GDN layer, the token-weighted beta > 1 rate and mean |t| over every forward since enable_stats on every
    rank; ([], []) for an arm without the gate. tokens[i] = padded token count of forward i (patch.py appends one
    per-forward mean per layer). The all-reduce runs on every rank (same arm, same layer count)."""
    sums = []
    for layer in patch.gdn_layers(model):
        st = getattr(layer, "_negeig_stats", None)
        if st is not None:
            assert len(st) == len(tokens), (len(st), len(tokens))
            sums += [sum(f * w for (f, _), w in zip(st, tokens)), sum(m * w for (_, m), w in zip(st, tokens))]
    t = torch.tensor(sums + [float(sum(tokens))], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(t)
    if not sums or t[-1].item() == 0:
        return [], []
    v = (t[:-1] / t[-1]).tolist()
    return v[0::2], v[1::2]


@torch.no_grad()
def eval_tasks(model, V, L, n_seq, batch, device, rank, world, seed, task_list=tasks.ALL_TASKS, nll=False):
    """train.py's eval_tasks for one length, sharded by whole batches (a batch is computed the same way on any world
    size). Returns train.py's keys {task}@{L} (per-step accuracy), {task}@{L}/seq, {task}@{L}/win, plus {task}@{L}/nll
    (mean label cross-entropy over the full vocabulary) when nll, and the gate statistics of these forwards."""
    model.eval()
    dec, head = T27.core(model)
    hw = head.base_layer.weight if hasattr(head, "base_layer") else head.weight
    sets = label_sets(V)
    jobs = []
    for task in task_list:
        rng = random.Random(seed + 7919 * L + 104729 * tasks.ALL_TASKS.index(task))
        exs = [tasks.make(task, L, rng, V) for _ in range(n_seq)]
        jobs += [(task, s, exs[s : s + batch]) for s in range(0, n_seq, batch)]
    owner, _ = T27.lpt([len(c) * max(len(e.ids) for e in c) for _, _, c in jobs], world)
    patch.enable_stats(model, True)
    mine, fw_tokens = {}, []
    for (task, s, chunk), o in zip(jobs, owner):
        if o != rank:
            continue
        cand = torch.tensor(sets[task], device=device)
        wc = hw[cand]
        h = T27.hidden(dec, [e.ids for e in chunk], device)
        fw_tokens.append(h.shape[0] * h.shape[1])
        rows, nl = [], 0.0
        for i, e in enumerate(chunk):
            hs = h[i, torch.tensor(e.label_pos, device=device)]
            pred = (hs.to(wc.dtype) @ wc.T).argmax(-1)
            gold = torch.tensor([sets[task].index(lab) for lab in e.labels], device=device)
            rows.append((pred == gold).cpu().tolist())
            if nll:
                y = torch.tensor(e.labels, device=device)
                nl += sum(F.cross_entropy(head(hs[a : a + 1024]).float(), y[a : a + 1024], reduction="sum").item()
                          for a in range(0, len(e.labels), 1024))
        mine[(task, s)] = (rows, nl)
    gfrac, gmag = gate_layers(model, fw_tokens, device, world)
    patch.enable_stats(model, False)
    got = {}
    for part in T27.all_gather(mine, world):
        got.update(part)
    out = {}
    for task in task_list:
        rows, nl = [], 0.0
        for s in range(0, n_seq, batch):
            r, x = got[(task, s)]
            rows += r
            nl += x
        correct = torch.tensor(rows, dtype=torch.bool)
        out[f"{task}@{L}"] = correct.float().mean(0).tolist()
        out[f"{task}@{L}/seq"] = correct.float().mean(1).tolist()
        out[f"{task}@{L}/win"] = [
            [correct[i, a : min(b, L)].float().mean().item() if a < L else None for a, b in WINDOWS] for i in range(n_seq)]
        if nll:
            out[f"{task}@{L}/nll"] = nl / correct.numel()
    model.train()
    return out, {"beta_gt1_frac_per_layer": gfrac, "abs_t_per_layer": gmag}


def window_means(res, L, task_list=tasks.ALL_TASKS):
    out = {}
    for t in task_list:
        w = res[f"{t}@{L}/win"]
        out[t] = {k: (sum(r[j] for r in w) / len(w) if w[0][j] is not None else None)
                  for j, k in enumerate(("w1", "w4", "w16"))}
    tier1 = [t for t in tasks.TIER1 if t in task_list]
    if tier1:
        out["tier1"] = {k: (sum(out[t][k] for t in tier1) / len(tier1) if out[tier1[0]][k] is not None else None)
                        for k in ("w1", "w4", "w16")}
    return out


@torch.no_grad()
def chunk_nll(model, chunks, device, rank, world):
    """Per-chunk mean next-token NLL (train.py eval_nll), in chunk order, on every rank."""
    model.eval()
    dec, head = T27.core(model)
    mine = {}
    for i in range(rank, chunks.shape[0], world):
        x = chunks[i : i + 1].long().to(device)
        h = T27.fwd(dec, x)[0, :-1]
        y = x[0, 1:]
        tot = sum(F.cross_entropy(head(h[s : s + 1024]).float(), y[s : s + 1024], reduction="sum").item()
                  for s in range(0, h.shape[0], 1024))
        mine[i] = tot / h.shape[0]
    got = {}
    for part in T27.all_gather(mine, world):
        got.update(part)
    model.train()
    return [got[i] for i in range(chunks.shape[0])]


def atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def test_path(out, step):
    return Path(out) / f"test_step{step:06d}.json"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=sorted(patch.ARMS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lm_replay", required=True, help="torch file {'train','test'} of token chunks (train.py --replay)")
    ap.add_argument("--text_eval", default="", help="chunk file whose test split eval_lm reads (default --lm_replay)")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w_lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--clip_joint", action="store_true", help="train.py's joint clip instead of train27's per group")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--task_counts", default="parity:16,swap:64,codeswap:64")
    ap.add_argument("--train_len", type=int, default=64)
    ap.add_argument("--replay_batch", type=int, default=12)
    ap.add_argument("--replay_len", type=int, default=512)
    ap.add_argument("--replay_micro", type=int, default=4)
    ap.add_argument("--w_lm", type=float, default=1.0)
    ap.add_argument("--micro_tokens", type=int, default=4096, help="padded-token cap of a task micro-batch")
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="fp32 only for CPU tests")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_batch", type=int, default=4, help="sequences per eval forward")
    ap.add_argument("--val_n", type=int, default=32)
    ap.add_argument("--val_len", type=int, default=256)
    ap.add_argument("--val_seed", type=int, default=20_000)
    ap.add_argument("--test_n", type=int, default=128)
    ap.add_argument("--test_len", type=int, default=1024)
    ap.add_argument("--test_seed", type=int, default=10_000)
    ap.add_argument("--test_steps", default="0", help="steps with a test eval besides --steps (0 = the untouched model)")
    ap.add_argument("--lm_eval_chunks", type=int, default=32)
    ap.add_argument("--nll_n", type=int, default=256, help="replay test chunks in the test record's 'nll' list")
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--keep_ckpts", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval_only", default="", help="trainable_*.pt or 'none': val and test evals of it, then exit")
    ap.add_argument("--mem_probe", type=int, default=0, help="1: fwd+bwd of the largest micro-batch first; 2: then exit")
    ap.add_argument("--max_steps_debug", type=int, default=0)
    ap.add_argument("--dist_timeout_min", type=int, default=30)
    args = ap.parse_args()

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                            timeout=datetime.timedelta(minutes=args.dist_timeout_min))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    autocast = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.dtype == "bf16")  # noqa: E731

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    V = tasks.Vocab(tok)
    tasks.reference_check(V)  # independent label recompute from the token stream; raises on a mismatch
    counts = parse_counts(args.task_counts)

    model, n_gates = T27.build_model(args, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    T27.sync_trainable(trainable, world)
    lora_p = [p for n, p in model.named_parameters() if p.requires_grad and "negeig_w" not in n]
    w_p = [p for n, p in model.named_parameters() if p.requires_grad and "negeig_w" in n]
    groups = [{"params": lora_p, "lr": args.lr}] + ([{"params": w_p, "lr": args.w_lr}] if w_p else [])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))
    total = args.max_steps_debug or args.steps

    rep = torch.load(args.lm_replay)
    lm_train, lm_test = rep["train"][:, : args.replay_len], rep["test"][:, : args.replay_len]
    del rep
    # n_lm below counts replay_len - 1 targets a chunk; a narrower file would make the slice shorter and the lm loss would
    # be divided by targets that do not exist (a silent down-weighting of the replay term)
    assert lm_train.shape[1] == args.replay_len and lm_test.shape[1] == args.replay_len, (
        f"--replay_len {args.replay_len} but the chunks of {args.lm_replay} hold {lm_train.shape[1]} tokens")
    text = torch.load(args.text_eval)["test"][:, : args.replay_len] if args.text_eval else lm_test
    text_eval = text[: args.lm_eval_chunks]
    nll_chunks = lm_test[replay_perm(lm_test.shape[0], 1234)[: args.nll_n]]  # train.py: replay_chunks(test, 256, seed 1234)
    perm = replay_perm(lm_train.shape[0], args.seed)
    stream = Stream(V, args.seed, counts, args.train_len)
    test_at = {int(x) for x in args.test_steps.split(",") if x != ""} | {args.steps}

    log = open(out / "log.jsonl", "a") if rank == 0 else None

    def emit(rec):
        if log:
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)

    def run_val(step):
        t0 = time.time()
        with autocast():
            res, gate = eval_tasks(model, V, args.val_len, args.val_n, args.eval_batch, device, rank, world,
                                   args.val_seed, nll=True)
            lm = T27.eval_lm(model, text_eval, device, rank, world)
        wm = window_means(res, args.val_len)
        ev = {"event": "eval", "step": step, "tier1": wm["tier1"],
              **{t: {**wm[t], "first8": [round(a, 3) for a in res[f"{t}@{args.val_len}"][:8]],
                     "nll": res[f"{t}@{args.val_len}/nll"]} for t in tasks.ALL_TASKS},
              "gate_rate_tasks": (sum(gate["beta_gt1_frac_per_layer"]) / len(gate["beta_gt1_frac_per_layer"])
                                  if gate["beta_gt1_frac_per_layer"] else None),
              "gate_rate_tasks_max_layer": max(gate["beta_gt1_frac_per_layer"], default=None), **lm,
              "eval_s": round(time.time() - t0, 1)}
        emit(ev)

    def run_test(step, path=None):
        t0 = time.time()
        with autocast():
            res, gate = eval_tasks(model, V, args.test_len, args.test_n, args.eval_batch, device, rank, world,
                                   args.test_seed)
            nll = chunk_nll(model, nll_chunks, device, rank, world)
        wm = window_means(res, args.test_len)
        if rank == 0:
            atomic_json({"arm": args.arm, "seed": args.seed, "step": step, "test_seed": args.test_seed,
                         "test_n": args.test_n, "test_len": args.test_len, "tasks": res, "nll": nll,
                         "negeig_stats": gate, "windows": wm}, path or test_path(out, step))
        emit({"event": "test", "step": step, "windows": wm, "nll_mean": sum(nll) / len(nll),
              "gate_max_layer": max(gate["beta_gt1_frac_per_layer"], default=None),
              "test_s": round(time.time() - t0, 1)})

    if args.eval_only:
        if args.eval_only != "none":
            sd = torch.load(args.eval_only, map_location="cpu")
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            assert set(sd) == set(params), f"trainable file does not match --arm/--rank: {sorted(set(sd) ^ set(params))[:4]}"
            with torch.no_grad():
                for k, v in sd.items():
                    params[k].copy_(v.to(params[k].dtype))
        emit({"event": "config", "eval_only": args.eval_only, "args": vars(args), "world": world})
        run_val(-1)
        run_test(-1, out / f"test_{Path(args.eval_only).stem}.json")
        dist.destroy_process_group()
        return

    start = 0
    if args.resume:
        path, ck, skipped = T27.pick_resume(out) if rank == 0 else (None, None, [])
        path = T27.bcast(str(path) if path is not None else None, world)
        if path is not None:
            if ck is None:
                ck = torch.load(path, map_location="cpu", weights_only=False)
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            assert set(ck["trainable"]) == set(params), (
                f"checkpoint does not match this model's trainable set: {sorted(set(ck['trainable']) ^ set(params))[:4]}")
            with torch.no_grad():
                for k, v in ck["trainable"].items():
                    params[k].copy_(v.to(params[k].dtype))
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start = ck["step"]
            del ck
            if rank == 0:
                print(json.dumps({"resumed_from": path, "step": start, "skipped_unloadable": skipped}), flush=True)

    emit({"event": "config", "start": start, "args": vars(args), "world": world, "n_gates": n_gates,
          "n_trainable": sum(p.numel() for p in trainable), "n_lora": sum(p.numel() for p in lora_p),
          "n_w": sum(p.numel() for p in w_p), "replay_train_chunks": lm_train.shape[0], "torch": torch.__version__,
          "task_tokens": {t: len(tasks.make(t, args.train_len, random.Random(0), V).ids) for t, _ in counts}})

    if args.mem_probe:
        mbs, cost = plan_step(stream.batch(0), step_chunks(lm_train, perm, 0, args.replay_batch), args.micro_tokens,
                              args.replay_micro)
        kind, b = mbs[max(range(len(mbs)), key=lambda i: cost[i])]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with autocast():
            tot, _ = t1_loss(model, b, device) if kind == "t1" else T27.seq_loss(model, [(x, [1] * len(x)) for x in b], device)
        tot.backward()
        opt.zero_grad(set_to_none=True)
        emit({"event": "mem_probe", "kind": kind, "padded_tokens": max(cost), "grad_ckpt": args.grad_ckpt,
              "train_mode": T27.train_mode(model), "s": round(time.time() - t0, 2),
              "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0})
        if args.mem_probe == 2:
            dist.destroy_process_group()
            return

    # Evals of the step we start from: step 0 on a fresh run; on a resume, whatever the checkpoint step lacks (the save
    # comes before its evals). Rank 0 decides and broadcasts, so every rank enters the same collectives.
    if args.eval_every and start % args.eval_every == 0 and not T27.bcast(rank == 0 and T27.eval_logged(out, start), world):
        run_val(start)
    if start in test_at and not T27.bcast(rank == 0 and test_path(out, start).exists(), world):
        run_test(start)

    n_task = {t: n * args.train_len for t, n in counts}
    n_labels = sum(n_task.values())
    n_lm = args.replay_batch * (args.replay_len - 1)
    keys = [t for t, _ in counts] + ["lm"]
    t_start, t_train, tokens = time.time(), 0.0, 0
    for step in range(start, total):
        t0 = time.time()
        groups_s = stream.batch(step)
        mbs, cost = plan_step(groups_s, step_chunks(lm_train, perm, step, args.replay_batch), args.micro_tokens,
                              args.replay_micro)
        owner, load = T27.lpt(cost, world)
        sums = {k: torch.zeros((), dtype=torch.float64, device=device) for k in keys}
        for (kind, b), o in zip(mbs, owner):
            if o != rank:
                continue
            with autocast():
                if kind == "t1":
                    tot, n = t1_loss(model, b, device)
                    loss = tot / n_labels
                else:
                    tot, n = T27.seq_loss(model, [(x, [1] * len(x)) for x in b], device)
                    loss = tot * args.w_lm / n_lm
            loss.backward()
            sums[b[0].task if kind == "t1" else "lm"] += tot.detach().double()
            tokens += sum(len(e.ids) if kind == "t1" else len(e) for e in b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_compute = time.time() - t0
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in trainable]
        flat = torch.cat([g.reshape(-1).float() for g in grads])
        stats = torch.stack([sums[k] for k in keys])
        if world > 1:
            dist.all_reduce(flat)
            dist.all_reduce(stats)
        o = 0
        for p in trainable:
            p.grad = flat[o : o + p.numel()].view_as(p).to(p.dtype)
            o += p.numel()
        if args.clip_joint:
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.clip).item()
            gnorm_lora = gnorm_w = None
        else:
            gnorm_lora = torch.nn.utils.clip_grad_norm_(lora_p, args.clip).item()
            gnorm_w = torch.nn.utils.clip_grad_norm_(w_p, args.clip).item() if w_p else 0.0
            gnorm = (gnorm_lora ** 2 + gnorm_w ** 2) ** 0.5
        lr_step = [g["lr"] for g in opt.param_groups]
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        t_train += time.time() - t0
        if step % 10 == 0 or step == total - 1:
            agg = torch.tensor([tokens], device=device, dtype=torch.float64)
            per = T27.all_gather(t_compute, world)
            if world > 1:
                dist.all_reduce(agg)
            loss_t = {t: stats[j].item() / n_task[t] for j, t in enumerate(keys[:-1])}
            emit({"event": "step", "step": step + 1, "loss": {**{k: round(v, 5) for k, v in loss_t.items()},
                                                               "tasks": round(sum(stats[j].item() for j in range(len(keys) - 1)) / n_labels, 5),
                                                               "lm": round(stats[-1].item() / n_lm, 5) if n_lm else None},
                  "gnorm": round(gnorm, 4), "gnorm_lora": gnorm_lora, "gnorm_w": gnorm_w if w_p else None,
                  "lr": lr_step[0], "lr_w": lr_step[1] if len(lr_step) > 1 else None, "train_mode": T27.train_mode(model),
                  "train_tok_per_s": round(agg[0].item() / max(t_train, 1e-9), 1),
                  "step_s": round(t_train / (step + 1 - start), 3), "rank_compute_s": [round(x, 2) for x in per],
                  "lpt_load_tokens": [int(x) for x in load], "micro_batches": len(mbs),
                  "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0,
                  "w_norm": (sum(p.float().norm().item() ** 2 for p in w_p) ** 0.5) if w_p else None})
        s1 = step + 1
        if (args.save_every and s1 % args.save_every == 0) or s1 == total:
            if T27.replicas_differ(trainable, world, device):
                emit({"event": "replica_mismatch", "step": s1, "action": "broadcast from rank 0"})
                T27.sync_trainable(trainable, world)
            if rank == 0:
                T27.save_ckpt(out, s1, model, opt, sched, args.keep_ckpts)
            if world > 1:  # at world 1 this would be the first NCCL call of the run (seen to fail on a lone GPU)
                dist.barrier()
        if args.eval_every and s1 % args.eval_every == 0:
            run_val(s1)
        if s1 in test_at:
            run_test(s1)
    if rank == 0:
        atomic_json({"steps": total, "elapsed_s": time.time() - t_start}, out / "complete.json")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
