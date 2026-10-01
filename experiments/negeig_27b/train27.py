#!/usr/bin/env python3
"""Stage A trainer for the Qwen3.8-27B retrofit: supervised state-tracking sessions plus replay.

One full bf16 replica per GPU (data parallel, gradients all-reduced once per step), LoRA on the Gated DeltaNet
projections, plus the zero-init widened-beta gate for the `wide` arm (patch.py, the same code the 4B/9B lane used).

Each optimizer step uses one GLOBAL batch that does not depend on the world size (all counts are global):
  --s1_per_step       S1 sessions (results/negeig-27b/DATA.md): loss on each assistant answer and its <|im_end|>
  --chat_per_step     self-distilled chat replay (the untouched model's own answers): loss on assistant turns
  --tool_per_step     self-distilled tool-use replay (its own score-1.0 agentic runs): loss on assistant turns
  --lm_per_step       clean LM replay chunks (512 tokens, English web, code, Hebrew web): loss on every token
Rows are drawn by step from per-epoch permutations seeded by (seed, source, epoch), so step t has the same batch on
any number of GPUs and after any resume. Every rank knows every row's token length (a table built once at start),
assigns the step's rows to ranks longest-first to the least-loaded rank (LPT), and runs each row of at least
--single_min tokens as its own unpadded micro-batch; shorter rows share padded micro-batches of at most
--short_tokens. The step loss is the weighted sum of per-source global token means (--w_s1, --w_chat, --w_tool,
--w_lm), so the rank assignment changes no gradient.

Checkpoint evals (every --eval_every steps and at step 0), distributed over ranks:
  S1 teacher-forced exact match on a fixed subset: an answer counts when every answer token and the closing
  <|im_end|> is the argmax given the gold history, which equals greedy decoding of that answer. Also the answer NLL
  (continuous, moves before the exact match does). Split at event 256 (the training maximum) and per length.
  Clean-replay test NLL, and KL(untouched || current) per token on the same chunks (adapter off, gates zeroed).
  Tool-replay agreement: argmax of the current model against the untouched one on assistant tokens of fixed tool
  rows held out of training (token rate and whole-turn rate).
  Gate firing rate (beta > 1), token-weighted over every eval forward on every rank, on S1 and on text.
  Every --recall_every steps and at step 0: a long-context recall guard (needles at 16k and 32k tokens).
Every --save_every steps and at the last step rank 0 writes an atomic, fsynced resumable checkpoint (trainable
tensors, optimizer, scheduler, step), keeps the newest --keep_ckpts, and writes the adapter-plus-gate tensors of every
save; the save comes before that step's eval. --resume continues from the newest checkpoint that loads (a truncated
newest file falls back to the one before), re-runs that step's eval when the log lacks it, and may change the world
size (no per-rank state). --lr_override applies a new --lr/--w_lr after loading (the optimizer and scheduler state
would otherwise restore the old one). Rank 0's trainable tensors are broadcast at start, so the replicas start equal.

  torchrun --standalone --nproc_per_node 4 train27.py --model M --arm wide --seed 0 --out RUN \
      --s1 DATA/s1 --lm_replay DATA/s4/replay_clean_q38.pt [--chat_replay F] [--tool_replay F]
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import zlib
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "negeig_retrofit"))
import patch  # noqa: E402

sys.path.insert(0, str(HERE / "data"))
from common import Session, to_messages  # noqa: E402

LORA_TARGETS = r".*layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_a|in_proj_b|out_proj)"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
THINK_OFF = "<think>\n\n</think>\n\n"


# ------------------------------------------------------------------ tokenization with assistant-only labels
def encode_chat(tok, messages, tools=None):
    """Token ids and a label mask for a chat rendered by the model's own template (thinking off).

    The mask covers each assistant turn's content and its closing <|im_end|>, not the role header and not the
    empty think block the template inserts. Spans are found on the rendered text and mapped through offsets, so
    tool-call markup inside assistant turns is covered exactly as the template renders it."""
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, enable_thinking=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    spans = []
    for m in re.finditer(re.escape(IM_START + "assistant\n"), text):
        start = m.end()
        if text.startswith(THINK_OFF, start):
            start += len(THINK_OFF)
        end = text.find(IM_END, start)
        assert end >= 0, "unterminated assistant turn"
        spans.append((start, end + len(IM_END)))
    mask = [0] * len(ids)
    j = 0
    for i, (a, b) in enumerate(offs):
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        if j < len(spans) and a >= spans[j][0] and b <= spans[j][1] and b > a:
            mask[i] = 1
    return ids, mask


def row_messages(row):
    """S1 rows are Session records (data/common.py); replay rows carry `messages` directly."""
    if "messages" in row:
        return row["messages"]
    return to_messages(Session.from_json(json.dumps(row)))


def spans_of(mask):
    """The answer spans in order: runs of consecutive labelled positions, one per assistant turn."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def load_jsonl(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f]
    return rows


def all_gather(obj, world):
    if world == 1:
        return [obj]
    out = [None] * world
    dist.all_gather_object(out, obj)
    return out


def bcast(obj, world):
    """Rank 0's value on every rank. Every decision that guards a collective (a cache hit, the checkpoint to resume,
    an eval to re-run) goes through here, so no rank can take a branch the others skip."""
    if world == 1:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def atomic_save(obj, path):
    """torch.save to a temp name, fsync, rename, fsync the directory: a preempted VM never leaves a named but empty
    checkpoint behind (the rename can reach the disk before the data without the fsync)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def lpt(costs, world):
    """Longest-first assignment of items to the least-loaded rank; the same answer on every rank."""
    order = sorted(range(len(costs)), key=lambda i: (-costs[i], i))
    load, owner = [0] * world, [0] * len(costs)
    for i in order:
        r = min(range(world), key=lambda k: (load[k], k))
        owner[i] = r
        load[r] += costs[i]
    return owner, load


# ------------------------------------------------------------------ data pools
LENGTH_TABLE_VERSION = 2  # bump when encode_chat or the label rule changes: the cache key then misses


def length_table_key(name, paths, tok, tools_key):
    """Cache key of a length table: the files (path, size, mtime), the tokenizer path AND its chat template (a changed
    template changes every row), the tools field and the encoder version. Not the world size: the table is whole."""
    tmpl = hashlib.sha256(str(getattr(tok, "chat_template", "") or "").encode()).hexdigest()
    files = [(str(p), os.path.getsize(p), int(os.path.getmtime(p))) for p in paths]
    blob = json.dumps([LENGTH_TABLE_VERSION, name, tools_key, files, getattr(tok, "name_or_path", ""), tmpl])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Pool:
    """Chat rows of one source with their token lengths and loss-label counts, drawn by step."""

    def __init__(self, name, rows, tok, per_step, weight, max_len, tools_key, rank, world, cache_dir, paths):
        self.name, self.rows, self.tok, self.k, self.w, self.tools_key = name, rows, tok, per_step, weight, tools_key
        cache = Path(cache_dir) / f"lengths_{name}_{length_table_key(name, paths, tok, tools_key)}.json"
        table = None
        if rank == 0 and cache.exists():  # rank 0 reads and validates, then every rank gets rank 0's answer
            try:
                t = json.loads(cache.read_text())
                if len(t) == len(rows) and all(isinstance(v, list) and len(v) == 2 for v in t):
                    table = t
            except (OSError, ValueError):
                table = None
        table = bcast(table, world)
        self.cache_hit = table is not None
        if table is None:  # each rank encodes a slice, then everyone has the whole table
            mine = {i: self._measure(i) for i in range(rank, len(rows), world)}
            table = [None] * len(rows)
            for part in all_gather(mine, world):
                for i, v in part.items():
                    table[int(i)] = v
            if rank == 0:
                cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(f".{os.getpid()}.tmp")  # two runs may share one cache dir
                tmp.write_text(json.dumps(table))
                os.replace(tmp, cache)
        self.length = [t[0] for t in table]
        self.labels = [t[1] for t in table]
        self.keep = [i for i in range(len(rows)) if 0 < self.length[i] <= max_len and self.labels[i] > 0]
        self.dropped = len(rows) - len(self.keep)
        self.held_out = 0
        assert self.keep or per_step == 0, f"{name}: no usable rows"
        self._perm = {}

    def hold_out(self, ids):
        """Remove rows from the training draw (eval rows must not be trained on). Call before the first draw."""
        ids = set(ids)
        self.keep = [i for i in self.keep if i not in ids]
        self.held_out += len(ids)
        self._perm = {}
        assert self.keep or self.k == 0, f"{self.name}: no rows left after the hold-out"

    def _measure(self, i):
        ids, mask = self.encode(i)
        return (len(ids), sum(mask[1:]))

    def encode(self, i):
        r = self.rows[i]
        return encode_chat(self.tok, row_messages(r), r.get(self.tools_key) if self.tools_key else None)

    def draw(self, step, seed):
        """The global row indices of this source for one step: epoch permutations without replacement."""
        n, out = len(self.keep), []
        for j in range(self.k):
            p = step * self.k + j
            e = p // n
            if e not in self._perm:
                g = random.Random(seed * 1_000_003 + zlib.crc32(self.name.encode()) * 7919 + e)
                perm = self.keep[:]
                g.shuffle(perm)
                self._perm = {k: v for k, v in self._perm.items() if k >= e - 1}
                self._perm[e] = perm
            out.append(self._perm[e][p % n])
        return out


# ------------------------------------------------------------------ model
def build_model(args, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, info = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map={"": device},
                                                       output_loading_info=True)
    # a checkpoint key the text model did not map would load as random init and still run fluently
    assert not info["missing_keys"], f"weights not loaded: {sorted(info['missing_keys'])[:5]}"
    # The gate Linears draw their default init from the CPU generator before they are zeroed. Without the fork the
    # wide arm's LoRA A init would differ from the ctrl arm's at the same seed, and the pair would not be paired.
    with torch.random.fork_rng(devices=[]):
        new = patch.install_for_arm(model, args.arm)
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none")
    model = get_peft_model(model, cfg)
    for n, p in model.named_parameters():
        if "negeig_w" in n:
            p.requires_grad_(True)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    # from_pretrained leaves every submodule in eval mode (the peft wrapper reports training=True, which hides it), and
    # HF checkpoints a decoder layer only when self.training is set. Without this, everything before the first eval
    # ends (the mem_probe, a whole --eval_every 0 run, and the steps after a resume whose eval is already logged) would
    # train with checkpointing off: a wrong mem_probe and smoke rate, and an OOM loop after a preemption when the
    # grad_ckpt decision is "on". eval27 calls model.eval() itself.
    model.train()
    return model, len(new)


def train_mode(model):
    """True when every Gated DeltaNet layer is in train mode (train() and eval() recurse, so this is the decoder
    layers' flag, the condition for HF gradient checkpointing); logged with each step and mem_probe."""
    layers = patch.gdn_layers(model)
    return bool(layers) and all(layer.training for layer in layers)


def core(model):
    """(decoder, lm_head) under peft wrapping; the decoder returns final hidden states."""
    base = model.get_base_model()
    return base.model, base.lm_head


@contextlib.contextmanager
def untouched(model):
    """The untouched model inside the trained one: LoRA disabled and every gate zeroed (restored on exit)."""
    saved = [(p, p.detach().clone()) for n, p in model.named_parameters() if "negeig_w" in n]
    with torch.no_grad():
        for p, _ in saved:
            p.zero_()
    try:
        with model.disable_adapter():
            yield
    finally:
        with torch.no_grad():
            for p, v in saved:
                p.copy_(v)


def gate_rate(model):
    """Mean over GDN layers of the fraction of (token, head) with beta > 1 since stats were enabled; None for ctrl.
    Per forward, unweighted, this rank only; the evals use gate_rate_global."""
    fr = patch.collect_stats(model)["beta_gt1_frac_per_layer"]
    return sum(fr) / len(fr) if fr else None


def gate_rate_global(model, tokens, device, world):
    """Token-weighted beta > 1 rate over every eval forward on every rank since enable_stats, averaged over GDN
    layers; None for an arm without the beta gate. tokens[i] is forward i's token count: patch.py appends one
    per-forward mean per layer, in call order. The all-reduce runs on every rank (same arm, same layer count)."""
    sums = []
    for layer in patch.gdn_layers(model):
        st = getattr(layer, "_negeig_stats", None)
        if st is not None:
            assert len(st) == len(tokens), (len(st), len(tokens))
            sums.append(sum(f * w for (f, _), w in zip(st, tokens)))
    t = torch.tensor(sums + [float(sum(tokens))], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(t)
    if not sums or t[-1].item() == 0:
        return None
    return (t[:-1] / t[-1]).mean().item()


@contextlib.contextmanager
def stats_paused(model):
    """No gate stats inside (the untouched forward of an eval would add zeros); the lists come back intact."""
    layers = patch.gdn_layers(model)
    keep = [getattr(layer, "_negeig_stats", None) for layer in layers]
    for layer in layers:
        if hasattr(layer, "negeig_w"):
            layer._negeig_stats = None
    try:
        yield
    finally:
        for layer, st in zip(layers, keep):
            if hasattr(layer, "negeig_w"):
                layer._negeig_stats = st


def fwd(dec, x, att=None):
    """Final hidden states. use_cache=False: in eval mode (and in training without grad_ckpt) the config default
    would build a KV and recurrent cache on every forward, memory the mem_probe would then count."""
    return dec(input_ids=x, attention_mask=att, use_cache=False).last_hidden_state


def hidden(dec, ids_list, device):
    """Final hidden states for a list of sequences: a lone sequence runs unpadded with no mask (the SDPA causal fast
    path); several run right-padded with an attention mask."""
    if len(ids_list) == 1:
        x = torch.tensor([ids_list[0]], device=device)
        return fwd(dec, x)
    L = max(len(x) for x in ids_list)
    ids = torch.zeros((len(ids_list), L), dtype=torch.long)
    att = torch.zeros((len(ids_list), L), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, : len(x)] = torch.tensor(x)
        att[i, : len(x)] = 1
    return fwd(dec, ids.to(device), att.to(device))


def seq_loss(model, batch, device):
    """Sum of token losses on labelled positions and their count, for one micro-batch of (ids, mask).

    Position t predicts token t+1 and counts when mask[t+1] is set; right padding carries no label. The labelled
    positions are chosen on the CPU (no device sync for a boolean index) and the head runs on them only, in
    checkpointed chunks of 2048 rows, so the full-vocabulary logits of one chunk are the only ones alive."""
    dec, head = core(model)
    h = hidden(dec, [x for x, _ in batch], device)
    L = h.shape[1]
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (x, m) in enumerate(batch):
        t = torch.tensor(x[1:] + [0])
        mm = torch.tensor(m[1:] + [0]).bool()
        lab[i, : len(x)] = torch.where(mm, t, torch.full_like(t, -100))
    idx = (lab.view(-1) != -100).nonzero().squeeze(1)
    n = idx.numel()
    hs = h.reshape(-1, h.shape[-1]).index_select(0, idx.to(device))
    ys = lab.view(-1)[idx].to(device)

    def ce(x, y):
        return F.cross_entropy(head(x).float(), y, reduction="sum")

    tot = sum(checkpoint(ce, hs[s : s + 2048], ys[s : s + 2048], use_reentrant=False)
              for s in range(0, n, 2048)) if n else h.sum() * 0.0
    return tot, n


def plan_micro(items, single_min, short_tokens):
    """items: (key, ids, mask). Long rows alone; short rows grouped by length under a padded-token cap."""
    long = [[it] for it in items if len(it[1]) >= single_min]
    short = sorted((it for it in items if len(it[1]) < single_min), key=lambda it: len(it[1]))
    groups, cur = [], []
    for it in short:
        if cur and (len(cur) + 1) * len(it[1]) > short_tokens:
            groups.append(cur)
            cur = []
        cur.append(it)
    if cur:
        groups.append(cur)
    return long + groups


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def eval_s1(model, mine, device, world):
    """Exact match and answer NLL per answer on this rank's pre-encoded eval sessions; summed over ranks."""
    model.eval()
    dec, head = core(model)
    stats, fw_tokens = {}, []
    patch.enable_stats(model, True)
    for r, ids, spans in mine:
        x = torch.tensor([ids], device=device)
        h = fwd(dec, x)[0]
        fw_tokens.append(len(ids))
        for (a, b), t in zip(spans, r["turns"]):
            lp = F.log_softmax(head(h[a - 1 : b - 1]).float(), -1)
            y = x[0, a:b]
            ok = bool((lp.argmax(-1) == y).all())
            nll = -lp.gather(-1, y[:, None]).sum().item()
            where = "le256" if t["meta"]["after_events"] <= 256 else "gt256"
            for k in ("all", where, "dom:" + r["domain"], "lang:" + r["lang"], f"len{r['n_events']}:{where}"):
                c, n, s, tk = stats.get(k, (0, 0, 0.0, 0))
                stats[k] = (c + ok, n + 1, s + nll, tk + (b - a))
    gr = gate_rate_global(model, fw_tokens, device, world)
    patch.enable_stats(model, False)
    agg = {}
    for part in all_gather(stats, world):
        for k, v in part.items():
            agg[k] = tuple(a + b for a, b in zip(agg.get(k, (0, 0, 0.0, 0)), v))
    model.train()
    out = {"gate_rate_s1": gr}
    for k, (c, n, s, tk) in sorted(agg.items()):
        out["s1:" + k] = c / n if n else None
        out["nll:" + k] = s / tk if tk else None
        out["n:" + k] = n
    return out


@torch.no_grad()
def eval_lm(model, chunks, device, rank, world):
    """Clean-replay test NLL per token, and KL(untouched || current) per token, summed over ranks."""
    model.eval()
    dec, head = core(model)
    nll = kl = n = 0.0
    fw_tokens = []
    patch.enable_stats(model, True)
    for i in range(rank, chunks.shape[0], world):
        x = chunks[i : i + 1].long().to(device)
        h = fwd(dec, x)[0, :-1]
        fw_tokens.append(x.shape[1])
        with untouched(model), stats_paused(model):
            h0 = fwd(dec, x)[0, :-1]
        for s in range(0, h.shape[0], 512):
            lp = F.log_softmax(head(h[s : s + 512]).float(), -1)
            lp0 = F.log_softmax(head(h0[s : s + 512]).float(), -1)
            y = x[0, 1 + s : 1 + s + lp.shape[0]]
            nll += -lp.gather(-1, y[:, None]).sum().item()
            kl += (lp0.exp() * (lp0 - lp)).sum().item()
            n += lp.shape[0]
    gr = gate_rate_global(model, fw_tokens, device, world)
    patch.enable_stats(model, False)
    t = torch.tensor([nll, kl, n], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(t)
    model.train()
    return {"replay_nll": (t[0] / t[2]).item(), "kl_to_untouched": (t[1] / t[2]).item(), "gate_rate_text": gr}


@torch.no_grad()
def eval_agree(model, mine, device, world, prefix):
    """Argmax agreement with the untouched model on labelled tokens: token rate and whole-turn rate."""
    model.eval()
    dec, head = core(model)
    tok_ok = tok_n = turn_ok = turn_n = 0
    for ids, spans in mine:
        x = torch.tensor([ids], device=device)
        h = fwd(dec, x)[0]
        with untouched(model):
            h0 = fwd(dec, x)[0]
        for a, b in spans:
            p = head(h[a - 1 : b - 1]).argmax(-1)
            p0 = head(h0[a - 1 : b - 1]).argmax(-1)
            eq = p == p0
            tok_ok += int(eq.sum())
            tok_n += b - a
            turn_ok += bool(eq.all())
            turn_n += 1
    t = torch.tensor([tok_ok, tok_n, turn_ok, turn_n], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(t)
    model.train()
    return {f"{prefix}_token_agree": (t[0] / t[1]).item() if t[1] else None,
            f"{prefix}_turn_agree": (t[2] / t[3]).item() if t[3] else None}


@torch.no_grad()
def eval_recall(model, mine, device, world):
    """Teacher-forced exact match on needle answers inside long replay text, per context length."""
    model.eval()
    dec, head = core(model)
    stats = {}
    for L, ids, (a, b) in mine:
        x = torch.tensor([ids], device=device)
        h = fwd(dec, x)[0]
        ok = bool((head(h[a - 1 : b - 1]).argmax(-1) == x[0, a:b]).all())
        c, n = stats.get(L, (0, 0))
        stats[L] = (c + ok, n + 1)
    agg = {}
    for part in all_gather(stats, world):
        for k, (c, n) in part.items():
            c0, n0 = agg.get(k, (0, 0))
            agg[k] = (c0 + c, n0 + n)
    model.train()
    return {f"recall:{k}": c / n for k, (c, n) in sorted(agg.items())}


def recall_items(tok, chunks, n_per_len, lengths, seed=777):
    """Needle items: a code sentence at a fixed depth inside decoded replay text, then a question about it."""
    g = random.Random(seed)
    names = ["cedar", "harbor", "quartz", "falcon", "meadow", "copper", "lantern", "orchid", "summit", "willow"]
    pieces = [tok.decode(c.tolist(), skip_special_tokens=True) for c in chunks]
    plen = [len(tok(t + "\n\n", add_special_tokens=False)["input_ids"]) for t in pieces]
    out = []
    for L in lengths:
        for i in range(n_per_len):
            name, code = g.choice(names), str(g.randrange(100000, 999999))
            needle = f"\nThe access code for the {name} archive is {code}.\n"
            depth = 0.1 + 0.8 * i / max(1, n_per_len - 1)
            parts, n, j = [], 0, g.randrange(len(pieces))
            while n < L - 200:  # piece token counts add up to within a few tokens of the joined text
                parts.append(pieces[j % len(pieces)] + "\n\n")
                n += plen[j % len(pieces)]
                j += 1
            text = "".join(parts)
            cut = int(len(text) * depth)
            sp_at = text.find(" ", cut)  # at a word boundary, not inside a word
            cut = sp_at if 0 <= sp_at < cut + 200 else cut
            doc = text[:cut] + needle + text[cut:]
            q = f"{doc}\n\nWhat is the access code for the {name} archive? Answer with the number only."
            ids, mask = encode_chat(tok, [{"role": "user", "content": q}, {"role": "assistant", "content": code}])
            sp = spans_of(mask)
            assert len(sp) == 1
            out.append((L, ids, sp[0]))
    return out


# ------------------------------------------------------------------ checkpoints
def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def save_ckpt(out, step, model, opt, sched, keep):
    """Resumable checkpoint (keep <= 0 keeps all), then the adapter-plus-gate tensors (every save kept, for evals
    and repair restarts). Both atomic and fsynced."""
    atomic_save({"step": step, "trainable": trainable_state(model), "opt": opt.state_dict(),
                 "sched": sched.state_dict()}, out / f"ckpt_{step:06d}.pt")
    if keep > 0:
        for old in sorted(out.glob("ckpt_*.pt"))[:-keep]:
            old.unlink()
    atomic_save(trainable_state(model), out / f"trainable_{step:06d}.pt")


def pick_resume(out):
    """(path, checkpoint, skipped): the newest checkpoint that loads, trying newest first, and the ones that did not;
    path and checkpoint are None when none loads. Called on rank 0 only; the path is broadcast so every rank resumes
    from the same step."""
    skipped = []
    for p in sorted(out.glob("ckpt_*.pt"), reverse=True):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(ck, dict) and {"step", "trainable", "opt", "sched"} <= set(ck):
                return p, ck, skipped
            skipped.append(str(p))
        except Exception as e:  # a truncated or empty file: fall back to the one before it
            skipped.append(f"{p}: {type(e).__name__}")
    return None, None, skipped


def eval_logged(out, step):
    """True when log.jsonl already holds an eval record for this step (rank 0 reads, callers broadcast)."""
    f = out / "log.jsonl"
    if not f.exists():
        return False
    with open(f) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:  # a line cut by a preemption
                continue
            if r.get("event") == "eval" and r.get("step") == step:
                return True
    return False


def sync_trainable(trainable, world):
    """Broadcast rank 0's trainable tensors. The manual all-reduce keeps replicas equal only if they start equal, and
    nothing else guarantees that the LoRA init consumed the same random stream on every rank."""
    if world > 1:
        with torch.no_grad():
            for p in trainable:
                dist.broadcast(p.data, src=0)


def replicas_differ(trainable, world, device):
    """Per-tensor float64 sum and sum of squares, compared exactly across ranks (two all-reduces)."""
    if world == 1:
        return False
    with torch.no_grad():
        d = torch.stack([s for p in trainable for s in (p.double().sum(), p.double().pow(2).sum())]).to(device)
        lo, hi = d.clone(), d.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        return bool((lo != hi).any())


def set_lrs(opt, sched, lrs):
    """Make new base learning rates stick after a resume: group lr and initial_lr, and the scheduler's base_lrs."""
    sched.base_lrs = list(lrs)
    for g, lr, fn in zip(opt.param_groups, lrs, sched.lr_lambdas):
        g["initial_lr"] = lr
        g["lr"] = lr * fn(sched.last_epoch)
    sched._last_lr = [g["lr"] for g in opt.param_groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=sorted(patch.ARMS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--s1", required=True, help="dir with <domain>.train.jsonl and <domain>.eval.jsonl")
    ap.add_argument("--lm_replay", required=True)
    ap.add_argument("--chat_replay", default="")
    ap.add_argument("--tool_replay", default="")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w_lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--sched", choices=["constant", "cosine"], default="constant")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--s1_per_step", type=int, default=64, help="global")
    ap.add_argument("--chat_per_step", type=int, default=8)
    ap.add_argument("--tool_per_step", type=int, default=8)
    ap.add_argument("--lm_per_step", type=int, default=8)
    ap.add_argument("--w_s1", type=float, default=1.0)
    ap.add_argument("--w_chat", type=float, default=0.5)
    ap.add_argument("--w_tool", type=float, default=0.5)
    ap.add_argument("--w_lm", type=float, default=0.15)
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip, applied to each group separately")
    ap.add_argument("--max_len", type=int, default=12288)
    ap.add_argument("--cache_dir", default="", help="token-length tables; default the run dir, share one across runs")
    ap.add_argument("--single_min", type=int, default=1024, help="rows this long run alone, unpadded")
    ap.add_argument("--short_tokens", type=int, default=8192, help="padded-token cap for a micro-batch of short rows")
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="fp32 only for CPU smokes")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_per_domain", type=int, default=16, help="S1 eval sessions per domain and length")
    ap.add_argument("--eval_lengths", default="32,256,512")
    ap.add_argument("--eval_lm_chunks", type=int, default=64)
    ap.add_argument("--eval_tool_rows", type=int, default=64)
    ap.add_argument("--recall_every", type=int, default=200)
    ap.add_argument("--recall_per_len", type=int, default=8)
    ap.add_argument("--recall_lengths", default="16384,32768")
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--keep_ckpts", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lr_override", action="store_true", help="after --resume, apply --lr/--w_lr")
    ap.add_argument("--mem_probe", type=int, default=0, help="1: fwd+bwd on the longest row of each pool first; "
                                                                  "2: then exit")
    ap.add_argument("--profile_step", type=int, default=-1, help="profile steps N..N+2 on rank 0")
    ap.add_argument("--max_steps_debug", type=int, default=0)
    ap.add_argument("--dist_timeout_min", type=int, default=30,
                    help="collective timeout; the first collective waits for the slowest rank's 54 GB model load")
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
    autocast = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.dtype == "bf16")  # noqa: E731

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model, n_gates = build_model(args, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    sync_trainable(trainable, world)
    lora_p = [p for n, p in model.named_parameters() if p.requires_grad and "negeig_w" not in n]
    w_p = [p for n, p in model.named_parameters() if p.requires_grad and "negeig_w" in n]
    groups = [{"params": lora_p, "lr": args.lr}] + ([{"params": w_p, "lr": args.w_lr}] if w_p else [])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    total = args.max_steps_debug or args.steps
    decay = (lambda s: 1.0) if args.sched == "constant" else (lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)) * decay(s))

    # ---------------------------------------------------------------- data
    t_data = time.time()
    s1_paths = sorted(glob.glob(os.path.join(args.s1, "*.train.jsonl")))
    cache_dir = Path(args.cache_dir) if args.cache_dir else out  # the key holds file sizes, mtimes, template and tools
    pools = [Pool("s1", load_jsonl(s1_paths), tok, args.s1_per_step, args.w_s1, args.max_len, None, rank, world, cache_dir,
                  s1_paths)]
    if args.chat_replay and args.chat_per_step:
        pools.append(Pool("chat", load_jsonl([args.chat_replay]), tok, args.chat_per_step, args.w_chat, args.max_len,
                          None, rank, world, cache_dir, [args.chat_replay]))
    if args.tool_replay and args.tool_per_step:
        pools.append(Pool("tool", load_jsonl([args.tool_replay]), tok, args.tool_per_step, args.w_tool, args.max_len,
                          "tools", rank, world, cache_dir, [args.tool_replay]))
    rep = torch.load(args.lm_replay)
    lm_train, lm_test = rep["train"], rep["test"]
    lm_offset = random.Random(args.seed).randrange(lm_train.shape[0])

    # fixed eval sets, identical on every rank, split by LPT on token length
    lengths = {int(x) for x in args.eval_lengths.split(",")}
    ev_rng = random.Random(12345)
    s1_eval = []
    for p in sorted(glob.glob(os.path.join(args.s1, "*.eval.jsonl"))):
        by_len = {}
        for r in load_jsonl([p]):
            if r["n_events"] in lengths:
                by_len.setdefault(r["n_events"], []).append(r)
        for L in sorted(by_len):
            s1_eval += ev_rng.sample(by_len[L], min(args.eval_per_domain, len(by_len[L])))
    enc = {i: encode_chat(tok, row_messages(r)) for i, r in enumerate(s1_eval) if i % world == rank}
    lens = [None] * len(s1_eval)
    for part in all_gather({i: len(v[0]) for i, v in enc.items()}, world):
        for i, n in part.items():
            lens[int(i)] = n
    owner, _ = lpt(lens, world)
    s1_mine = []
    for i, r in enumerate(s1_eval):
        if owner[i] == rank:
            ids, mask = enc[i] if i in enc else encode_chat(tok, row_messages(r))
            sp = spans_of(mask)
            assert len(sp) == len(r["turns"]), (len(sp), len(r["turns"]))
            s1_mine.append((r, ids, sp))
    del enc
    lm_eval = lm_test[: args.eval_lm_chunks]
    tool_mine = []
    tool_pool = next((p for p in pools if p.name == "tool"), None)
    tool_heldout = False
    if tool_pool is not None and args.eval_tool_rows:
        # The agreement rows are held out of training (at most a fifth of the pool): measured on rows the adapter is
        # trained on, agreement with the untouched model's own answers would pass the safety gate by construction.
        # A pool too small to spare a row (CPU smokes) is scored on training rows, and the config says so.
        n_hold = min(args.eval_tool_rows, len(tool_pool.keep) // 5)
        tool_heldout = n_hold > 0
        pick = random.Random(999).sample(tool_pool.keep, n_hold if tool_heldout
                                         else min(args.eval_tool_rows, len(tool_pool.keep)))
        if tool_heldout:
            tool_pool.hold_out(pick)
        owner, _ = lpt([tool_pool.length[i] for i in pick], world)
        for i, o in zip(pick, owner):
            if o == rank:
                ids, mask = tool_pool.encode(i)
                tool_mine.append((ids, spans_of(mask)))
    recall_mine = []
    if args.recall_every:
        rl = [int(x) for x in args.recall_lengths.split(",")]
        items = recall_items(tok, lm_test[args.eval_lm_chunks:], args.recall_per_len, rl)
        owner, _ = lpt([len(it[1]) for it in items], world)
        recall_mine = [it for it, o in zip(items, owner) if o == rank]
    t_data = time.time() - t_data

    start = 0
    if args.resume:
        path, ck, skipped = pick_resume(out) if rank == 0 else (None, None, [])
        path = bcast(str(path) if path is not None else None, world)
        if path is not None:
            if ck is None:  # ranks other than 0 load the file rank 0 chose
                ck = torch.load(path, map_location="cpu", weights_only=False)
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            assert set(ck["trainable"]) == set(params), (
                f"checkpoint does not match this model's trainable set (arm or LoRA rank changed?): "
                f"{sorted(set(ck['trainable']) ^ set(params))[:4]}")
            with torch.no_grad():
                for k, v in ck["trainable"].items():
                    params[k].copy_(v.to(params[k].dtype))
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            if args.lr_override:
                set_lrs(opt, sched, [args.lr] + ([args.w_lr] if w_p else []))
            start = ck["step"]
            del ck
            if rank == 0:
                print(json.dumps({"resumed_from": path, "step": start, "skipped_unloadable": skipped,
                                  "lrs": [g["lr"] for g in opt.param_groups]}), flush=True)

    log = open(out / "log.jsonl", "a") if rank == 0 else None

    def emit(rec):
        if log:
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)

    eval_items = all_gather([len(s1_mine), len(tool_mine), len(recall_mine)], world)
    emit({"event": "config", "start": start, "args": vars(args), "world": world, "n_gates": n_gates,
          "n_trainable": sum(p.numel() for p in trainable), "n_s1_eval": len(s1_eval), "data_s": round(t_data, 1),
          "eval_items_per_rank": {"s1": [e[0] for e in eval_items], "tool": [e[1] for e in eval_items],
                                  "recall": [e[2] for e in eval_items]}, "tool_eval_heldout": tool_heldout,
          "pools": {p.name: {"rows": len(p.rows), "usable": len(p.keep), "dropped_over_max_len": p.dropped,
                             "held_out_for_eval": p.held_out, "length_cache_hit": p.cache_hit,
                             "mean_len": sum(p.length[i] for i in p.keep) / max(1, len(p.keep)),
                             "max_len": max((p.length[i] for i in p.keep), default=0)} for p in pools}})

    def run_eval(step):
        t0 = time.time()
        with autocast():
            ev = {"event": "eval", "step": step, **eval_s1(model, s1_mine, device, world),
                  **eval_lm(model, lm_eval, device, rank, world)}
            if tool_pool is not None and args.eval_tool_rows:
                ev |= eval_agree(model, tool_mine, device, world, "tool")
            if args.recall_every and step % args.recall_every == 0:
                ev |= eval_recall(model, recall_mine, device, world)
        ev["eval_s"] = round(time.time() - t0, 1)
        emit(ev)

    if args.mem_probe:
        # One fwd+bwd on the longest usable row of EVERY pool (a micro-batch is one session, so a row is the unit of
        # peak memory). The s1 pool alone would miss a tool row near --max_len, which decides grad_ckpt.
        for p0 in pools:
            if not p0.keep:
                continue
            i = max(p0.keep, key=lambda j: p0.length[j])
            ids, mask = p0.encode(i)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with autocast():
                tot, _ = seq_loss(model, [(ids, mask)], device)
            tot.backward()
            opt.zero_grad(set_to_none=True)
            emit({"event": "mem_probe", "pool": p0.name, "tokens": len(ids), "grad_ckpt": args.grad_ckpt,
                  "train_mode": train_mode(model), "s": round(time.time() - t0, 2),
                  "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0})
        if args.mem_probe == 2:
            dist.destroy_process_group()
            return

    # The eval of the step we start from: step 0 on a fresh run, and on a resume the eval of the checkpoint step when
    # the process died after the save and before that eval was logged (saves come first, see the loop end). Rank 0
    # reads the log and broadcasts, so every rank enters the eval collectives or none does.
    if args.eval_every and start % args.eval_every == 0 and not bcast(rank == 0 and eval_logged(out, start), world):
        run_eval(start)
    t_start, tokens, spread = time.time(), 0, []
    t_train = 0.0  # step time only (no evals, no saves): the throughput a run pays for
    prof = None
    for step in range(start, total):
        if rank == 0 and step == args.profile_step and device.type == "cuda":
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        t_step0 = time.time()
        # the step's global rows: (pool index or -1 for LM, row index, length); identical on every rank
        glob_items = []
        for pi, p in enumerate(pools):
            glob_items += [(pi, i, p.length[i]) for i in p.draw(step, args.seed)]
        for j in range(args.lm_per_step):
            glob_items.append((-1, (lm_offset + step * args.lm_per_step + j) % lm_train.shape[0], lm_train.shape[1]))
        owner, load = lpt([it[2] for it in glob_items], world)
        counts = {}
        for pi, i, _ in glob_items:
            key = "lm" if pi < 0 else pools[pi].name
            counts[key] = counts.get(key, 0) + (lm_train.shape[1] - 1 if pi < 0 else pools[pi].labels[i])
        weight = {p.name: p.w for p in pools} | {"lm": args.w_lm}
        items = []
        for (pi, i, _), o in zip(glob_items, owner):
            if o != rank:
                continue
            if pi < 0:
                x = lm_train[i].tolist()
                items.append(("lm", x, [1] * len(x)))
            else:
                ids, mask = pools[pi].encode(i)
                assert sum(mask[1:]) == pools[pi].labels[i] and len(ids) == pools[pi].length[i], "length table stale"
                items.append((pools[pi].name, ids, mask))
        t0 = time.time()
        losses = {}
        for mb in plan_micro(items, args.single_min, args.short_tokens):
            by = {}
            for key, ids, mask in mb:
                by.setdefault(key, []).append((ids, mask))
            for key, batch in by.items():  # plan groups can mix sources; each source keeps its own normaliser
                with autocast():
                    tot, n = seq_loss(model, batch, device)
                    loss = tot * weight[key] / max(counts[key], 1)  # this rank's share of the global token mean
                loss.backward()
                # kept on the device: a .item() per micro-batch would stall the launch queue on every backward
                losses[key] = losses.get(key, 0.0) + tot.detach().double() / max(counts[key], 1)
                tokens += sum(len(x) for x, _ in batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_compute = time.time() - t0
        # one all-reduce of the flattened gradients (LoRA and gate only), summed: the loss is already the global mean
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in trainable]
        flat = torch.cat([g.reshape(-1).float() for g in grads])
        keys = sorted(weight)
        stats = torch.stack([torch.as_tensor(losses.get(k, 0.0), dtype=torch.float64, device=device) for k in keys])
        if world > 1:
            dist.all_reduce(flat)
            dist.all_reduce(stats)
        o = 0
        for p in trainable:
            p.grad = flat[o : o + p.numel()].view_as(p).to(p.dtype)
            o += p.numel()
        losses = {k: stats[j].item() for j, k in enumerate(keys) if k in counts}
        # Clip per group: the zero-init gate's first gradients are about 60x the LoRA norm, and a joint clip would
        # rescale the wide arm's LoRA step by a factor the ctrl arm never sees (the 4B/9B trainer clipped jointly).
        gnorm_lora = torch.nn.utils.clip_grad_norm_(lora_p, args.clip).item()
        gnorm_w = torch.nn.utils.clip_grad_norm_(w_p, args.clip).item() if w_p else 0.0
        gnorm = (gnorm_lora ** 2 + gnorm_w ** 2) ** 0.5
        lr_step = [g["lr"] for g in opt.param_groups]  # the rates this step used (get_last_lr is the next step's)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        spread.append(t_compute)
        t_train += time.time() - t_step0
        if prof is not None and step == args.profile_step + 2:
            prof.__exit__(None, None, None)
            ka = prof.key_averages()
            (out / "profile_rank0.txt").write_text(ka.table(sort_by="self_device_time_total", row_limit=60))
            dev = {e.key: getattr(e, "self_device_time_total", 0) for e in ka}
            tot_dev = sum(dev.values()) or 1
            cls = {"gdn": ("chunk", "delta", "wy", "solve_tril", "kkt", "l2norm", "cumsum", "conv1d", "recompute_w_u"),
                   "attention": ("flash", "attention", "fmha", "sdpa", "efficient"),
                   "gemm": ("gemm", "cutlass", "nvjet", "cublas", "matmul", "sm90_", "sm100_")}
            share = {c: sum(v for k, v in dev.items() if any(s in k.lower() for s in pats)) / tot_dev
                     for c, pats in cls.items()}
            (out / "profile_rank0.json").write_text(json.dumps({"steps": [args.profile_step, args.profile_step + 2],
                                                                 "device_share": share}))
            prof = None
        if step % 10 == 0 or step == total - 1:
            el = time.time() - t_start
            agg = torch.tensor([tokens, t_compute], device=device, dtype=torch.float64)
            per = all_gather(t_compute, world)
            if world > 1:
                dist.all_reduce(agg)
            emit({"event": "step", "step": step + 1, "loss": {k: round(v, 4) for k, v in losses.items()},
                  "gnorm": round(gnorm, 4), "gnorm_lora": round(gnorm_lora, 4), "gnorm_w": round(gnorm_w, 4) if w_p else None, "lr": lr_step[0], "lr_w": lr_step[1] if len(lr_step) > 1 else None,
                  "train_mode": train_mode(model),
                  "tok_per_s": round(agg[0].item() / max(el, 1e-9), 1),
                  "train_tok_per_s": round(agg[0].item() / max(t_train, 1e-9), 1),
                  "elapsed_s": round(el, 1), "rank_compute_s": [round(x, 2) for x in per],
                  "lpt_load_tokens": [int(x) for x in load],
                  "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0,
                  "w_norm": (sum(p.float().norm().item() ** 2 for p in w_p) ** 0.5) if w_p else None})
        # Save before the eval: an eval that crashes (or a preemption during it) then costs the eval, not 50 steps,
        # and the resume re-runs the eval of the checkpoint step (see eval_logged above). The last step always saves.
        if (args.save_every and (step + 1) % args.save_every == 0) or step == total - 1:
            if replicas_differ(trainable, world, device):  # never expected; rank 0's copy is the one saved anyway
                emit({"event": "replica_mismatch", "step": step + 1, "action": "broadcast from rank 0"})
                sync_trainable(trainable, world)
            if rank == 0:
                save_ckpt(out, step + 1, model, opt, sched, args.keep_ckpts)
            dist.barrier()
        if args.eval_every and (step + 1) % args.eval_every == 0:
            run_eval(step + 1)
    if rank == 0:
        tmp = out / ".complete.tmp"
        tmp.write_text(json.dumps({"steps": total, "elapsed_s": time.time() - t_start}))
        os.replace(tmp, out / "complete.json")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
