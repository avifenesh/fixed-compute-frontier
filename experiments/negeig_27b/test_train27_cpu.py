#!/usr/bin/env python3
"""CPU tests for train27.py and eval27.py. No GPU: a tiny random qwen3_5 stands in for the 27B.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 taskset -c 8-15 \\
        ~/.venvs/negeig/bin/python test_train27_cpu.py [--only a,b] [--skip-dist] [--keep]

Builds a fixture in a temp dir (TMPDIR): a 4-layer qwen3_5 (3 Gated DeltaNet + 1 attention, hidden 64, the real
248,320 vocab, random weights), 12 S1 train and 3 S1 eval sessions from each of two domains of the rig's data, 128-token
LM chunks cut from the real replay file, and hand-written chat and tool rows. fla is blocked, so transformers runs its
torch GDN path. Needs --tokenizer (the Qwen3.8 tokenizer) and --data (the negeig-27b data dir with s1/ and s4/).

Unit tests run in-process at world 1; the dist_* tests run train27.py and eval27.py under torchrun (gloo) and compare
runs. Every test has a red arm: a deliberately wrong variant that the same check must reject, so no check is vacuous.
What this cannot cover: bf16 numerics, fla and SDPA kernels, NCCL, the 27B's memory. That is the box checklist
(results/negeig-27b/assess-speed.md section 10).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = []
FX: Path | None = None  # fixture dir
T = None  # train27 module, imported after fla is blocked


# ------------------------------------------------------------------ runner
def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def run_test(name, fn):
    t0 = time.time()
    try:
        fn()
        RESULTS.append((name, "PASS", round(time.time() - t0, 1), ""))
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, "FAIL", round(time.time() - t0, 1), f"{type(e).__name__}: {e}"))
        traceback.print_exc()
    print(f"{RESULTS[-1][1]} {name} ({RESULTS[-1][2]} s) {RESULTS[-1][3]}", flush=True)


# ------------------------------------------------------------------ fixture
CHAT = [
    ("Name three rivers.", "The Nile, the Amazon and the Danube."),
    ("What is 17 times 3?", "17 times 3 is 51."),
    ("מה בירת צרפת?", "פריז."),
    ("Explain what a hash map is.", "A hash map stores key and value pairs. " * 12),
    ("Write a haiku about rain.", "Soft rain on the roof\nthe gutter hums a low tune\nnight drinks it all in"),
    ("Give me a packing list for a hike.", "Water, a map, a rain jacket, snacks, sunscreen and a first aid kit. " * 5),
    ("תן לי שלושה צבעים.", "אדום, כחול וירוק."),
    ("Summarize the plot of a heist film in two sentences.",
     "A crew plans to rob a casino vault. The plan goes wrong, and the planner still walks away with the money."),
]


def tool_rows():
    tools = [{"type": "function", "function": {
        "name": "get_order", "description": "Look up an order by id",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}}]
    rows = []
    for k in range(10):
        oid = str(100 + 7 * k)
        status = ["shipped", "packed", "delayed"][k % 3]
        extra = " The courier note says: " + "left at the front desk. " * (k * 2) if k % 2 else ""
        rows.append({"tools": tools, "messages": [
            {"role": "user", "content": f"Where is order {oid}?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "get_order", "arguments": {"id": oid}}}]},
            {"role": "tool", "content": json.dumps({"status": status, "note": extra})},
            {"role": "assistant", "content": f"Order {oid} is {status}.{extra}"}]})
    return rows


def build_fixture(tok_dir, data_dir):
    """Everything the in-process tests and the torchrun runs share."""
    fx = Path(tempfile.mkdtemp(prefix="t27_"))
    (fx / "nofla" / "fla").mkdir(parents=True)
    (fx / "nofla" / "fla" / "__init__.py").write_text('raise ImportError("fla blocked for the CPU tests")\n')
    sys.path.insert(0, str(fx / "nofla"))

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=248320, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], full_attention_interval=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64, attn_output_gate=True,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, tie_word_embeddings=True, eos_token_id=248044,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True})
    torch.manual_seed(1234)
    Qwen3_5ForCausalLM(cfg).save_pretrained(fx / "model")
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"):
        if (Path(tok_dir) / f).exists():
            shutil.copy(Path(tok_dir) / f, fx / "model" / f)

    (fx / "s1").mkdir()
    for dom in ("custody", "codetrace"):
        for split, n in (("train", 12), ("eval", 3)):
            rows = []
            with open(Path(data_dir) / "s1" / f"{dom}.{split}.jsonl") as f:
                for line in f:
                    if json.loads(line)["n_events"] == 32:
                        rows.append(line)
                    if len(rows) == n:
                        break
            (fx / "s1" / f"{dom}.{split}.jsonl").write_text("".join(rows))
    rep = torch.load(Path(data_dir) / "s4" / "replay_clean_q38.pt")
    torch.save({"train": rep["train"][:200, :128].clone(), "test": rep["test"][:48, :128].clone()}, fx / "replay.pt")
    del rep
    with open(fx / "chat.jsonl", "w") as f:
        for q, a in CHAT:
            f.write(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]},
                               ensure_ascii=False) + "\n")
    with open(fx / "tool.jsonl", "w") as f:
        for r in tool_rows():
            f.write(json.dumps(r) + "\n")
    AutoTokenizer.from_pretrained(fx / "model")  # fails here, not inside a test, when the copy is incomplete
    return fx


def import_train27():
    global T
    sys.path.insert(0, str(HERE))
    import train27
    T = train27
    return train27


def build(arm, seed=0, rank=4):
    import torch
    torch.manual_seed(seed)
    a = types.SimpleNamespace(model=str(FX / "model"), dtype="fp32", arm=arm, rank=rank, grad_ckpt=1)
    model, _ = T.build_model(a, torch.device("cpu"))
    return model


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(FX / "model")


def perturb(model, seed=7, w_scale=0.5, b_scale=0.05):
    """Nonzero gates and LoRA B, so every trainable tensor gets a gradient and the model is not the untouched one."""
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "negeig_w" in n:
                p.copy_(torch.randn(p.shape, generator=g) * w_scale)
            elif "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * b_scale)


def fixture_items(tok, n_s1=5):
    """(key, ids, mask) for S1, chat, tool and LM rows, as the step loop builds them."""
    import torch
    items = []
    s1 = T.load_jsonl([FX / "s1" / "custody.train.jsonl", FX / "s1" / "codetrace.train.jsonl"])
    for j in (0, 5, 9, 15, 13)[:n_s1]:  # short and long rows (about 580, 940, 1045, 760, 510 tokens)
        items.append(("s1", *T.encode_chat(tok, T.row_messages(s1[j]))))
    for r in T.load_jsonl([FX / "chat.jsonl"])[:3]:
        items.append(("chat", *T.encode_chat(tok, T.row_messages(r))))
    for r in T.load_jsonl([FX / "tool.jsonl"])[1:4]:
        items.append(("tool", *T.encode_chat(tok, T.row_messages(r), r["tools"])))
    lm = torch.load(FX / "replay.pt")["train"]
    for i in range(3):
        x = lm[i].tolist()
        items.append(("lm", x, [1] * len(x)))
    return items


WEIGHT = {"s1": 1.0, "chat": 0.5, "tool": 0.5, "lm": 0.15}


def global_counts(items):
    c = {}
    for k, ids, m in items:
        c[k] = c.get(k, 0) + sum(m[1:])
    return c


def grads_of(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def max_rel_diff(ga, gb):
    import torch
    assert set(ga) == set(gb), (sorted(set(ga) ^ set(gb)))[:3]
    num = max(float((ga[k] - gb[k]).abs().max()) for k in ga)
    den = max(float(gb[k].abs().max()) for k in gb)
    return num / den


# ------------------------------------------------------------------ unit tests
def test_draw_epochs_and_process_determinism():
    """Pool.draw: each epoch is a permutation of the usable rows (no repeat, full coverage); the same (seed, source,
    step) gives the same rows in another process whatever PYTHONHASHSEED is. Red arm: a draw seeded by hash(name)
    disagrees across hash seeds, so the cross-process check can see a nondeterministic draw."""
    P = T.Pool.__new__(T.Pool)
    P.name, P.k, P.keep, P._perm = "s1", 5, list(range(3, 40)), {}
    n = len(P.keep)
    seq = [i for s in range(3 * n // P.k + 1) for i in P.draw(s, seed=0)]
    for e in range(3):
        chunk = seq[e * n:(e + 1) * n]
        check(sorted(chunk) == sorted(P.keep), f"epoch {e} is not a permutation of keep")
    check(seq[:n] != seq[n:2 * n], "epochs repeat the same order")
    Q = copy.copy(P)
    Q._perm = {}
    check([Q.draw(s, 1) for s in range(8)] != [P.draw(s, 0) for s in range(8)], "seed does not change the draw")
    # random access equals sequential access (a resume at step t draws what an unbroken run drew at t)
    R = copy.copy(P)
    R._perm = {}
    check(R.draw(17, 0) == [seq[17 * P.k + j] for j in range(P.k)], "draw(t) depends on the call history")

    prog = ("import sys, json, zlib, random; sys.path.insert(0, %r); sys.path.insert(0, %r); import train27 as T\n"
            "P = T.Pool.__new__(T.Pool); P.k, P.keep, P._perm = 5, list(range(3, 40)), {}\n"
            "out = {}\n"
            "for name in ('s1', 'chat', 'tool'):\n"
            "    P.name, P._perm = name, {}\n"
            "    out[name] = [P.draw(s, 3) for s in range(20)]\n"
            "    if %s:\n"
            "        g = random.Random(hash(name)); perm = P.keep[:]; g.shuffle(perm); out[name] = perm\n"
            "print(json.dumps(out))\n")
    env = dict(os.environ, PYTHONPATH=str(FX / "nofla"))

    def run(hashseed, broken):
        r = subprocess.run([sys.executable, "-c", prog % (str(FX / "nofla"), str(HERE), broken)],
                           env=dict(env, PYTHONHASHSEED=str(hashseed)), capture_output=True, text=True, timeout=300)
        check(r.returncode == 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    a, b = run(1, False), run(2, False)
    check(a == b, "the draw differs between processes with different PYTHONHASHSEED")
    here = {}
    for name in ("s1", "chat", "tool"):
        P.name, P._perm = name, {}
        here[name] = [P.draw(s, 3) for s in range(20)]
    check(here == a, "the draw differs between this process and a fresh one")
    ra, rb = run(1, True), run(2, True)
    check(ra != rb, "red arm: a hash()-seeded draw agreed across hash seeds, the cross-process check is blind")


def test_lpt_assignment():
    """lpt: every item has an owner, loads are the owners' sums, equal costs spread by index, the makespan on the
    fixture's real S1 lengths is within 5% of the lower bound. Red arm: a contiguous-block split of the same list is
    not."""
    tok = tokenizer()
    s1 = T.load_jsonl([FX / "s1" / "custody.train.jsonl", FX / "s1" / "codetrace.train.jsonl"])
    costs = [len(T.encode_chat(tok, T.row_messages(r))[0]) for r in s1] + [128] * 8 + [60, 200, 90]
    for world in (2, 3, 4):
        owner, load = T.lpt(costs, world)
        check(owner == T.lpt(costs, world)[0], "lpt not deterministic")
        check(all(0 <= o < world for o in owner), "bad owner")
        check(load == [sum(c for c, o in zip(costs, owner) if o == r) for r in range(world)], "loads wrong")
        lb = max(max(costs), sum(costs) / world)
        check(max(load) <= 1.05 * lb, f"world {world}: makespan {max(load)} vs bound {lb:.0f}")
        srt = sorted(costs, reverse=True)
        blk = [sum(srt[r * len(srt) // world:(r + 1) * len(srt) // world]) for r in range(world)]
        check(max(blk) > 1.05 * lb, "red arm: a block split also met the bound, the bound check is blind")
    check(T.lpt([5] * 6, 3)[0] == [0, 1, 2, 0, 1, 2], "equal costs are not spread by index")


def test_seq_loss_alignment_single_and_padded():
    """seq_loss on one unpadded row equals full-logit cross entropy with position t predicting token t+1 where
    mask[t+1] is set; a right-padded batch equals the sum of its rows. Red arm: the unshifted labels differ."""
    import torch
    import torch.nn.functional as F
    tok = tokenizer()
    model = build("wide")
    perturb(model)
    model.eval()
    items = fixture_items(tok, n_s1=2)
    rows = [items[1][1:], items[2][1:], items[5][1:], items[-1][1:]]  # s1 (long), chat, tool, lm
    dec, head = T.core(model)
    singles = []
    with torch.no_grad():
        for ids, m in rows:
            tot, n = T.seq_loss(model, [(ids, m)], torch.device("cpu"))
            logits = head(T.fwd(dec, torch.tensor([ids])))[0].float()
            lab = torch.tensor(ids[1:])
            keep = torch.tensor(m[1:]).bool()
            ref = F.cross_entropy(logits[:-1][keep], lab[keep], reduction="sum")
            bad = F.cross_entropy(logits[:-1][keep], torch.tensor(ids[:-1])[keep], reduction="sum")  # no shift
            check(n == int(keep.sum()), f"label count {n} != {int(keep.sum())}")
            check(rel(float(tot), float(ref)) < 1e-5, f"single row: seq_loss {float(tot)} vs reference {float(ref)}")
            check(rel(float(bad), float(ref)) > 1e-4, "red arm: unshifted labels gave the same loss")
            singles.append((float(tot), n))
        tot_b, n_b = T.seq_loss(model, rows, torch.device("cpu"))
    check(n_b == sum(n for _, n in singles), "padded batch label count")
    check(rel(float(tot_b), sum(t for t, _ in singles)) < 2e-5,
          f"padded batch {float(tot_b)} vs sum of rows {sum(t for t, _ in singles)}")


def _trainer_grads(model, items, counts, single_min, short_tokens, local_mean=False):
    import torch
    model.zero_grad(set_to_none=True)
    for mb in T.plan_micro(items, single_min, short_tokens):
        by = {}
        for key, ids, mask in mb:
            by.setdefault(key, []).append((ids, mask))
        for key, batch in by.items():
            tot, n = T.seq_loss(model, batch, torch.device("cpu"))
            den = n if local_mean else counts[key]
            (tot * WEIGHT[key] / max(den, 1)).backward()
    return grads_of(model)


def test_grad_equivalence_micro_batching():
    """The trainer's gradient (plan_micro groups, the per-source split inside a group, each loss over the GLOBAL
    per-source label count) equals the per-row reference, equals one big padded batch with per-token weights, and
    the sum over a two-rank LPT split equals the whole (what the all-reduce computes). Red arms: a per-micro-batch
    mean, and per-rank counts, both change the gradient."""
    import torch
    import torch.nn.functional as F
    tok = tokenizer()
    model = build("wide")
    perturb(model)
    model.train()
    items = fixture_items(tok, n_s1=5)
    counts = global_counts(items)
    plan = T.plan_micro(items, 700, 1600)
    check(any(len(g) == 1 and len(g[0][1]) >= 700 for g in plan), "no single long row in the plan")
    check(any(len({k for k, _, _ in g}) > 1 for g in plan), "no group mixes sources: the split is not exercised")
    ga = _trainer_grads(model, items, counts, 700, 1600)
    check(len(ga) == sum(1 for p in model.parameters() if p.requires_grad), "some trainable tensor got no grad")
    check(any("negeig_w" in k for k in ga) and any("lora_A" in k for k in ga), "gate or LoRA A has no grad")
    # per-row reference
    model.zero_grad(set_to_none=True)
    for key, ids, m in items:
        tot, _ = T.seq_loss(model, [(ids, m)], torch.device("cpu"))
        (tot * WEIGHT[key] / counts[key]).backward()
    gb = grads_of(model)
    # one big padded batch, per-token weights
    model.zero_grad(set_to_none=True)
    dec, head = T.core(model)
    h = T.hidden(dec, [ids for _, ids, _ in items], torch.device("cpu"))
    loss = 0.0
    for i, (key, ids, m) in enumerate(items):
        lab = torch.tensor(ids[1:])
        keep = torch.tensor(m[1:]).bool()
        ce = F.cross_entropy(head(h[i, : len(ids) - 1][keep]).float(), lab[keep], reduction="sum")
        loss = loss + ce * WEIGHT[key] / counts[key]
    loss.backward()
    gd = grads_of(model)
    # two ranks: LPT split, each rank's own plan, summed
    owner, _ = T.lpt([len(ids) for _, ids, _ in items], 2)
    gc = None
    for r in range(2):
        g = _trainer_grads(model, [it for it, o in zip(items, owner) if o == r], counts, 700, 1600)
        gc = g if gc is None else {k: gc[k] + g[k] for k in gc}
    for name, g in (("per-row", gb), ("big batch", gd), ("two-rank sum", gc)):
        d = max_rel_diff(ga, g)
        check(d < 1e-4, f"trainer gradient vs {name}: max rel diff {d:.2e}")
    d_red = max_rel_diff(_trainer_grads(model, items, counts, 700, 1600, local_mean=True), gb)
    check(d_red > 1e-2, f"red arm: a per-micro-batch mean matched the reference ({d_red:.2e})")
    gl = None
    for r in range(2):
        mine = [it for it, o in zip(items, owner) if o == r]
        g = _trainer_grads(model, mine, global_counts(mine), 700, 1600)
        gl = g if gl is None else {k: gl[k] + g[k] for k in gl}
    d_red2 = max_rel_diff(gl, gb)
    check(d_red2 > 1e-2, f"red arm: per-rank counts matched the global normaliser ({d_red2:.2e})")


def test_untouched_context():
    """untouched(): LoRA off and every gate zeroed gives exactly a fresh untouched model; the trained state comes back
    after the block and after an exception inside it. Red arm: the same context without try/finally leaves the gates
    zeroed after an exception."""
    import torch
    x = torch.tensor([list(range(1000, 1100))])
    ref = build("wide")
    ref.eval()
    with torch.no_grad():
        h_ref = T.fwd(T.core(ref)[0], x)
    model = build("wide")
    perturb(model)
    model.eval()
    dec = T.core(model)[0]
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        h_cur = T.fwd(dec, x)
        check(float((h_cur - h_ref).abs().max()) > 1e-3, "perturbation did not change the model")
        with T.untouched(model):
            h_u = T.fwd(dec, x)
        check(float((h_u - h_ref).abs().max()) == 0.0, "untouched() is not the untouched model")
        check(all(torch.equal(p, saved[n]) for n, p in model.named_parameters() if p.requires_grad), "not restored")
        check(torch.equal(T.fwd(dec, x), h_cur), "forward after the block differs")
        with contextlib.suppress(RuntimeError):
            with T.untouched(model):
                raise RuntimeError("boom")
        check(all(torch.equal(p, saved[n]) for n, p in model.named_parameters() if p.requires_grad),
              "not restored after an exception")
        check(torch.equal(T.fwd(dec, x), h_cur), "forward after an exception differs")

        @contextlib.contextmanager
        def no_finally(m):
            keep = [(p, p.detach().clone()) for n, p in m.named_parameters() if "negeig_w" in n]
            for p, _ in keep:
                p.zero_()
            with m.disable_adapter():
                yield
            for p, v in keep:
                p.copy_(v)

        with contextlib.suppress(RuntimeError):
            with no_finally(model):
                raise RuntimeError("boom")
        w = [p for n, p in model.named_parameters() if "negeig_w" in n]
        check(all(float(p.abs().max()) == 0 for p in w), "red arm: the broken context restored the gates anyway")
        for n, p in model.named_parameters():
            if n in saved:
                p.copy_(saved[n])


class _Oracle(__import__("torch").nn.Module):
    """A fake peft model whose decoder writes the NEXT token id into a 1-wide hidden state and whose head turns it
    into a peaked one-hot over the real vocab: every answer is the argmax. shift=0 writes the CURRENT token (wrong).
    wrong=<id> predicts token 0 wherever the next token is <id> (a wrong end-of-turn token with the <|im_end|> id)."""

    V = 248320

    def __init__(self, shift=1, base_shift=None, wrong=None):
        super().__init__()
        self.shift, self.base_shift, self.off, self.wrong = shift, base_shift, False, wrong
        me = self

        class Dec:
            def __call__(self, input_ids, attention_mask=None, use_cache=None):
                import torch
                s = me.base_shift if (me.off and me.base_shift is not None) else me.shift
                x = input_ids.float()
                h = torch.cat([x[:, s:], torch.zeros_like(x[:, :s])], 1) if s else x
                if me.wrong is not None:
                    h = torch.where(h == float(me.wrong), torch.zeros_like(h), h)
                return types.SimpleNamespace(last_hidden_state=h[..., None])

        def head(z):
            import torch
            return torch.nn.functional.one_hot(z[..., 0].round().long(), self.V).float() * 30.0

        self._base = types.SimpleNamespace(model=Dec(), lm_head=head)

    def get_base_model(self):
        return self._base

    @contextlib.contextmanager
    def disable_adapter(self):
        self.off = True
        try:
            yield
        finally:
            self.off = False


def test_eval_indexing_oracle_and_nll():
    """eval_s1, eval_recall and eval_agree read the logits at a-1..b-2 for the tokens at a..b-1: an oracle that knows
    the next token scores 1.0 everywhere, and eval_s1's summed answer NLL on the real model equals seq_loss on the
    same session. Red arms: a current-token oracle scores 0, an untouched copy that disagrees lowers agreement, and
    an off-by-one slice changes the NLL."""
    import torch
    import torch.nn.functional as F
    tok = tokenizer()
    ev = T.load_jsonl([FX / "s1" / "custody.eval.jsonl"])[:2]
    mine = []
    for r in ev:
        ids, m = T.encode_chat(tok, T.row_messages(r))
        mine.append((r, ids, T.spans_of(m)))
    rec = T.recall_items(tok, torch.load(FX / "replay.pt")["test"][8:], 2, [300])
    agree = [(ids, [sp]) for _, ids, sp in rec]
    cpu = torch.device("cpu")
    good, bad = _Oracle(1), _Oracle(0)
    check(T.eval_s1(good, mine, cpu, 1)["s1:all"] == 1.0, "oracle did not score 1.0 on S1")
    check(T.eval_s1(bad, mine, cpu, 1)["s1:all"] == 0.0, "red arm: a current-token oracle scored on S1")
    check(T.eval_recall(good, rec, cpu, 1)["recall:300"] == 1.0, "oracle did not pass recall")
    check(T.eval_recall(bad, rec, cpu, 1)["recall:300"] == 0.0, "red arm: current-token oracle passed recall")
    check(T.eval_agree(_Oracle(1, 1), agree, cpu, 1, "t")["t_turn_agree"] == 1.0, "self agreement below 1")
    check(T.eval_agree(_Oracle(1, 0), agree, cpu, 1, "t")["t_turn_agree"] == 0.0,
          "red arm: a disagreeing untouched model still agreed")
    # NLL indexing on the real model
    model = build("wide")
    perturb(model)
    out = T.eval_s1(model, mine[:1], cpu, 1)
    r, ids, sp = mine[0]
    tk = sum(b - a for a, b in sp)
    model.eval()
    with torch.no_grad():
        tot, n = T.seq_loss(model, [T.encode_chat(tok, T.row_messages(r))], cpu)
        h = T.fwd(T.core(model)[0], torch.tensor([ids]))[0]
        head = T.core(model)[1]
        off = sum(-F.log_softmax(head(h[a:b]).float(), -1).gather(-1, torch.tensor(ids[a:b])[:, None]).sum().item()
                  for a, b in sp)
    check(n == tk and out["n:all"] == len(sp), "answer token or answer count mismatch")
    check(rel(out["nll:all"] * tk, float(tot)) < 1e-5, f"eval NLL {out['nll:all'] * tk} vs seq_loss {float(tot)}")
    check(rel(off, float(tot)) > 1e-4, "red arm: the off-by-one slice gave the same NLL")


def test_gate_stats_per_call():
    """The gate rate belongs to one eval call: W = 0 after a call with large W reads exactly 0 (beta = sigmoid < 1),
    and stats are off afterwards (training forwards record nothing). Red arm: stats enabled once across both
    settings read above 0 for the W = 0 part."""
    import torch
    tok = tokenizer()
    r = T.load_jsonl([FX / "s1" / "codetrace.eval.jsonl"])[0]
    ids, m = T.encode_chat(tok, T.row_messages(r))
    mine = [(r, ids, T.spans_of(m))]
    model = build("wide")
    perturb(model, w_scale=2.0)
    cpu = torch.device("cpu")
    r1 = T.eval_s1(model, mine, cpu, 1)["gate_rate_s1"]
    check(r1 is not None and r1 > 0.05, f"large W gave gate rate {r1}")
    check(all(getattr(l, "_negeig_stats", None) is None for l in T.patch.gdn_layers(model)), "stats left on")
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if "negeig_w" in n}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "negeig_w" in n:
                p.zero_()
    r2 = T.eval_s1(model, mine, cpu, 1)["gate_rate_s1"]
    check(r2 == 0.0, f"W = 0 gave gate rate {r2}: stats leak across calls")
    # red arm: one enable across both settings mixes them
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in saved:
                p.copy_(saved[n])
        model.eval()
        T.patch.enable_stats(model, True)
        T.fwd(T.core(model)[0], torch.tensor([ids]))
        for n, p in model.named_parameters():
            if n in saved:
                p.zero_()
        T.fwd(T.core(model)[0], torch.tensor([ids]))
        mixed = T.gate_rate(model)
        T.patch.enable_stats(model, False)
    check(mixed > 0.0, "red arm: accumulated stats read 0, the per-call check is blind")
    ctrl = build("ctrl")
    check(T.eval_s1(ctrl, mine, cpu, 1)["gate_rate_s1"] is None, "ctrl arm reported a gate rate")


def test_gate_rate_token_weighted_and_text():
    """eval_s1's gate rate is the token-weighted mean over sessions of different lengths, and eval_lm's text rate
    counts the current model's forwards only (the untouched forwards are paused, not averaged in as zeros). Red arms:
    the unweighted per-session mean, and a rate that counts the untouched forwards, are different numbers."""
    import torch
    tok = tokenizer()
    ev = T.load_jsonl([FX / "s1" / "custody.eval.jsonl"])  # 568, 569, 637 tokens: the weighting matters
    mine = []
    for r in ev:
        ids, m = T.encode_chat(tok, T.row_messages(r))
        mine.append((r, ids, T.spans_of(m)))
    model = build("wide")
    perturb(model, w_scale=0.3)
    cpu = torch.device("cpu")
    got = T.eval_s1(model, mine, cpu, 1)["gate_rate_s1"]
    model.eval()
    per, lens = [], []
    with torch.no_grad():
        for _, ids, _ in mine:
            T.patch.enable_stats(model, True)
            T.fwd(T.core(model)[0], torch.tensor([ids]))
            per.append(T.gate_rate(model))
            lens.append(len(ids))
        T.patch.enable_stats(model, False)
    want = sum(p * n for p, n in zip(per, lens)) / sum(lens)
    check(rel(got, want) < 1e-9, f"S1 gate rate {got} vs token-weighted {want}")
    unweighted = sum(per) / len(per)
    check(rel(unweighted, want) > 1e-6, "red arm: the unweighted mean is the same number")
    chunks = torch.load(FX / "replay.pt")["test"][:3]
    text = T.eval_lm(model, chunks, cpu, 0, 1)["gate_rate_text"]
    model.eval()
    with torch.no_grad():
        T.patch.enable_stats(model, True)
        for i in range(3):
            T.fwd(T.core(model)[0], chunks[i : i + 1].long())
        cur = T.gate_rate(model)
        for i in range(3):
            with T.untouched(model):
                T.fwd(T.core(model)[0], chunks[i : i + 1].long())
        mixed = T.gate_rate(model)
        T.patch.enable_stats(model, False)
    check(rel(text, cur) < 1e-9, f"text gate rate {text} vs current-model forwards {cur}")
    check(cur > 0 and rel(mixed, cur) > 0.1, "red arm: counting the untouched forwards did not change the rate")


def test_eval_lm_kl_direction():
    """eval_lm's KL is KL(untouched || current) per token, computed on the untouched() model, and exactly 0 before
    training. Red arm: KL(current || untouched) is a different number."""
    import torch
    import torch.nn.functional as F
    chunks = torch.load(FX / "replay.pt")["test"][:3]
    cpu = torch.device("cpu")
    fresh = build("wide")
    z = T.eval_lm(fresh, chunks, cpu, 0, 1)
    check(z["kl_to_untouched"] == 0.0, f"KL at init is {z['kl_to_untouched']}")
    model = build("wide")
    perturb(model, w_scale=3.0, b_scale=1.0)  # far from the untouched model: KL is symmetric to second order
    with torch.no_grad():  # peaked distributions (same base change in both, so fresh stays model's untouched copy)
        for m in (model, fresh):
            T.core(m)[1].weight.mul_(20.0)
    out = T.eval_lm(model, chunks, cpu, 0, 1)
    fresh.eval()
    model.eval()
    fwd_kl = bwd_kl = nll = n = 0.0
    with torch.no_grad():
        for i in range(chunks.shape[0]):
            x = chunks[i : i + 1].long()
            lp = F.log_softmax(T.core(model)[1](T.fwd(T.core(model)[0], x)[0, :-1]).float(), -1)
            lp0 = F.log_softmax(T.core(fresh)[1](T.fwd(T.core(fresh)[0], x)[0, :-1]).float(), -1)
            fwd_kl += float((lp0.exp() * (lp0 - lp)).sum())
            bwd_kl += float((lp.exp() * (lp - lp0)).sum())
            nll += float(-lp.gather(-1, x[0, 1:, None]).sum())
            n += lp.shape[0]
    check(rel(out["kl_to_untouched"], fwd_kl / n) < 1e-4, f"KL {out['kl_to_untouched']} vs {fwd_kl / n}")
    check(rel(out["replay_nll"], nll / n) < 1e-5, "replay NLL mismatch")
    check(rel(bwd_kl / n, fwd_kl / n) > 1e-3, "red arm: the reverse KL is the same number")


def test_set_lrs_after_resume():
    """After loading optimizer and scheduler state, set_lrs makes the new base rates stick (group lr, initial_lr,
    base_lrs) for the following steps, with the warmup lambda at the restored step. Red arm: without it the old rate
    comes back."""
    import torch

    def make(lrs):
        ps = [torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(2))]
        opt = torch.optim.AdamW([{"params": [ps[0]], "lr": lrs[0]}, {"params": [ps[1]], "lr": lrs[1]}])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 4))
        return ps, opt, sched

    ps, opt, sched = make([1e-3, 2e-3])
    for _ in range(2):
        for p in ps:
            p.grad = torch.ones_like(p)
        opt.step()
        sched.step()
    so, ss = copy.deepcopy(opt.state_dict()), copy.deepcopy(sched.state_dict())
    for override in (False, True):
        _, o2, s2 = make([5e-4, 1e-4])
        o2.load_state_dict(copy.deepcopy(so))
        s2.load_state_dict(copy.deepcopy(ss))
        if override:
            T.set_lrs(o2, s2, [5e-4, 1e-4])
        for k in range(3):
            want = [b * min(1.0, (s2.last_epoch + 1) / 4) for b in ([5e-4, 1e-4] if override else [1e-3, 2e-3])]
            got = [g["lr"] for g in o2.param_groups]
            check(all(rel(a, b) < 1e-12 for a, b in zip(got, want)), f"override={override} step {k}: {got} {want}")
            if not override and k == 0:
                check(all(rel(a, b) > 0.1 for a, b in zip(got, [5e-4 * 0.75, 1e-4 * 0.75])),
                      "red arm: the new rate stuck without set_lrs")
            for p in o2.param_groups:
                for q in p["params"]:
                    q.grad = torch.ones_like(q)
            o2.step()
            s2.step()


def test_checkpoints_retention_atomic_and_fallback():
    """save_ckpt keeps the newest --keep_ckpts resumable files and every trainable file, leaves no temp file, keeps all
    at keep <= 0; pick_resume skips a truncated newest file. Red arm: loading that file directly raises."""
    import torch
    d = Path(tempfile.mkdtemp(prefix="ck_", dir=FX))
    m = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(m.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    for s in range(1, 6):
        T.save_ckpt(d, s, m, opt, sched, 2)
    names = sorted(p.name for p in d.iterdir())
    check([n for n in names if n.startswith("ckpt_")] == ["ckpt_000004.pt", "ckpt_000005.pt"], names)
    check(len([n for n in names if n.startswith("trainable_")]) == 5, names)
    check(not [n for n in names if n.startswith(".")], f"temp files left: {names}")
    d0 = Path(tempfile.mkdtemp(prefix="ck0_", dir=FX))
    for s in range(1, 4):
        T.save_ckpt(d0, s, m, opt, sched, 0)
    check(len(list(d0.glob("ckpt_*.pt"))) == 3, "keep 0 did not keep all")
    (d / "ckpt_000005.pt").write_bytes((d / "ckpt_000005.pt").read_bytes()[:64])
    p, ck, skipped = T.pick_resume(d)
    check(p is not None and p.name == "ckpt_000004.pt" and ck["step"] == 4, f"picked {p}")
    check(len(skipped) == 1 and "ckpt_000005" in skipped[0], f"skipped {skipped}")
    try:
        torch.load(d / "ckpt_000005.pt", map_location="cpu", weights_only=False)
        raise AssertionError("red arm: the truncated checkpoint loaded")
    except AssertionError:
        raise
    except Exception:
        pass
    check(T.pick_resume(Path(tempfile.mkdtemp(prefix="ck_empty_", dir=FX)))[0] is None, "empty dir picked a file")


def test_length_table_cache():
    """The length-table key moves with the chat template, the tools field and the file mtime (a stale table would
    train on wrong label counts); a second Pool reads the cache and agrees. Red arm: the old key (files and tokenizer
    path only) does not move with the template."""
    import hashlib
    tok = tokenizer()
    paths = [FX / "tool.jsonl"]
    k0 = T.length_table_key("tool", paths, tok, "tools")
    check(k0 == T.length_table_key("tool", paths, tok, "tools"), "key not stable")
    check(k0 != T.length_table_key("tool", paths, tok, None), "tools field not in the key")

    def old_key(t):
        return hashlib.sha256(json.dumps(["tool", [(str(p), os.path.getsize(p), int(os.path.getmtime(p)))
                                                   for p in paths], getattr(t, "name_or_path", "")]).encode()).hexdigest()
    t2 = copy.deepcopy(tok)
    t2.chat_template = tok.chat_template.replace("<tool_call>", "<tool_call >")
    check(T.length_table_key("tool", paths, t2, "tools") != k0, "template change did not move the key")
    check(old_key(t2) == old_key(tok), "red arm: the old key moved with the template")
    d = Path(tempfile.mkdtemp(prefix="lt_", dir=FX))
    rows = T.load_jsonl(paths)
    a = T.Pool("tool", rows, tok, 2, 0.5, 12288, "tools", 0, 1, d, paths)
    b = T.Pool("tool", rows, tok, 2, 0.5, 12288, "tools", 0, 1, d, paths)
    check(not a.cache_hit and b.cache_hit, "cache not written or not read")
    check(a.length == b.length and a.labels == b.labels, "cached table differs")
    check(all(sum(a.encode(i)[1][1:]) == a.labels[i] for i in range(len(rows))), "labels do not match encode")
    c = T.Pool("tool", rows, t2, 2, 0.5, 12288, "tools", 0, 1, d, paths)
    check(not c.cache_hit, "a changed template hit the old table")
    os.utime(paths[0], (time.time() + 5, time.time() + 5))
    check(T.length_table_key("tool", paths, tok, "tools") != k0, "mtime change did not move the key")


def test_tool_hold_out():
    """Held-out tool rows leave the training draw for good. Red arm: without hold_out they are drawn."""
    tok = tokenizer()
    paths = [FX / "tool.jsonl"]
    d = Path(tempfile.mkdtemp(prefix="ho_", dir=FX))
    P = T.Pool("tool", T.load_jsonl(paths), tok, 3, 0.5, 12288, "tools", 0, 1, d, paths)
    drawn = {i for s in range(20) for i in P.draw(s, 0)}
    held = [P.keep[0], P.keep[3]]
    check(set(held) <= drawn, "red arm: the rows to hold out were never drawn anyway")
    P.hold_out(held)
    drawn = {i for s in range(20) for i in P.draw(s, 0)}
    check(not (set(held) & drawn), "held-out rows were drawn")
    check(drawn == set(P.keep), "the remaining rows are not all drawn")


def test_arm_pairing_lora_init():
    """At one seed the wide and ctrl arms start from the same LoRA tensors (the gate init is forked off the global
    generator). Red arm: installing the gates without the fork changes the wide arm's LoRA A."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    w, c = build("wide"), build("ctrl")
    lw = {n: p for n, p in w.named_parameters() if "lora_" in n}
    lc = {n: p for n, p in c.named_parameters() if "lora_" in n}
    check(set(lw) == set(lc) and lw, "LoRA sets differ")
    check(all(torch.equal(lw[n], lc[n]) for n in lw), "wide and ctrl LoRA init differ at the same seed")
    def manual(install):  # build_model's steps, with the gate install unforked
        torch.manual_seed(0)
        m, _ = AutoModelForCausalLM.from_pretrained(FX / "model", dtype=torch.float32,
                                                    device_map={"": torch.device("cpu")}, output_loading_info=True)
        if install:
            T.patch.install_for_arm(m, "wide")
        m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=T.LORA_TARGETS,
                                         bias="none"))
        return {n: p for n, p in m.named_parameters() if "lora_A" in n}

    same = manual(False)
    check(all(torch.equal(same[n], lc[n]) for n in same), "the manual path does not reproduce build_model's init")
    lr = manual(True)
    check(any(not torch.equal(lr[n], lc[n]) for n in lr), "red arm: the unforked install left LoRA A unchanged")


def test_train_mode_checkpointing():
    """build_model returns the model in train mode, so HF gradient checkpointing runs in every training forward,
    including the ones before the first eval (mem_probe, an --eval_every 0 run, the steps after a resume whose eval is
    logged). Red arms: from_pretrained plus peft leaves the layers in eval mode while the peft wrapper says training,
    and in eval mode the same counter sees no checkpointed layer."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from transformers.modeling_layers import GradientCheckpointingLayer
    model = build("wide")
    layers = [m for m in model.modules() if isinstance(m, GradientCheckpointingLayer) and m.gradient_checkpointing]
    check(len(layers) == 4, f"checkpointed decoder layers: {len(layers)}")
    calls = [0]
    for m in layers:
        f = m._gradient_checkpointing_func

        def wrap(*a, _f=f, **kw):
            calls[0] += 1
            return _f(*a, **kw)
        m._gradient_checkpointing_func = wrap
    ids, mask = list(range(1000, 1200)), [0] * 100 + [1] * 100
    check(T.train_mode(model), "build_model did not return the model in train mode")
    tot, _ = T.seq_loss(model, [(ids, mask)], torch.device("cpu"))
    tot.backward()
    check(calls[0] == 4, f"checkpointed layer calls in a training forward: {calls[0]}")
    model.eval()
    calls[0] = 0
    tot, _ = T.seq_loss(model, [(ids, mask)], torch.device("cpu"))
    tot.backward()
    check(calls[0] == 0 and not T.train_mode(model), "red arm: eval mode still checkpointed or reported train mode")
    raw, _ = AutoModelForCausalLM.from_pretrained(FX / "model", dtype=torch.float32,
                                                  device_map={"": torch.device("cpu")}, output_loading_info=True)
    raw = get_peft_model(raw, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=T.LORA_TARGETS,
                                         bias="none"))
    check(raw.training and not T.train_mode(raw),
          "red arm: from_pretrained + peft is not the eval-mode trap this test guards (wrapper train, layers eval)")


def test_recall_items():
    """Needle items: the answer span is the 6 digits and <|im_end|> (one token per digit), the needle appears once at
    a word boundary, the length is near the target. Red arm: a span shifted by one token is not the answer."""
    import torch
    tok = tokenizer()
    items = T.recall_items(tok, torch.load(FX / "replay.pt")["test"][8:], 3, [400, 800])
    check(len(items) == 6, "item count")
    for L, ids, (a, b) in items:
        ans = tok.decode(ids[a:b])
        check(b - a == 7 and ans[:6].isdigit() and ans[6:] == "<|im_end|>", f"answer span {ans!r}")
        text = tok.decode(ids)
        code = ans[:6]
        check(text.count(f"archive is {code}.") == 1, "needle not exactly once")
        post = text[text.index(f"archive is {code}.\n") + len(f"archive is {code}.\n"):]
        check(post.startswith(" "), f"needle not at a word boundary: {post[:20]!r}")
        check(L - 250 <= len(ids) <= L + 150, f"length {len(ids)} for target {L}")
        check(tok.decode(ids[a - 1:b - 1]) != ans, "red arm: a shifted span decoded to the answer")


# ------------------------------------------------------------------ distributed tests (torchrun)
COMMON = ["--arm", "wide", "--s1", "{fx}/s1", "--lm_replay", "{fx}/replay.pt", "--chat_replay", "{fx}/chat.jsonl",
          "--tool_replay", "{fx}/tool.jsonl", "--model", "{fx}/model", "--steps", "6", "--s1_per_step", "4",
          "--chat_per_step", "2", "--tool_per_step", "2", "--lm_per_step", "2", "--eval_every", "3", "--save_every", "3",
          "--eval_per_domain", "8", "--eval_lengths", "32", "--eval_lm_chunks", "4", "--eval_tool_rows", "2",
          "--recall_every", "3", "--recall_per_len", "1", "--recall_lengths", "300", "--warmup", "2", "--dtype", "fp32",
          "--lr", "1e-3", "--w_lr", "1e-3", "--single_min", "700", "--short_tokens", "1600", "--rank", "4"]


def torchrun(nproc, out, extra=(), seed=0, script="train27.py", common=True, timeout=900):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (FX / "tmp").mkdir(exist_ok=True)  # torchelastic_* and inductor dirs land in the fixture, removed with it
    env = dict(os.environ, PYTHONPATH=str(FX / "nofla"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               TMPDIR=str(FX / "tmp"))
    args = [a.format(fx=FX) for a in COMMON] + ["--seed", str(seed)] if common else []
    cmd = [str(Path(sys.executable).parent / "torchrun"), "--standalone", "--nproc_per_node", str(nproc),
           str(HERE / script), "--out", str(out), *args, *extra]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    (out / "test_stdout.log").write_text(r.stdout + "\n" + r.stderr)
    check(r.returncode == 0, f"{script} x{nproc} exit {r.returncode}: {r.stderr[-3000:]}")
    return r


def log_of(out):
    recs = []
    for line in open(Path(out) / "log.jsonl"):
        with contextlib.suppress(ValueError):
            recs.append(json.loads(line))
    return recs


def evals(out):
    return {r["step"]: r for r in log_of(out) if r.get("event") == "eval"}


def steps(out):
    return {r["step"]: r for r in log_of(out) if r.get("event") == "step"}


def trainable_diff(a, b):
    import torch
    x, y = torch.load(a), torch.load(b)
    check(set(x) == set(y), "trainable key sets differ")
    return max(float((x[k] - y[k]).abs().max()) for k in x) / max(float(y[k].abs().max()) for k in y)


EVAL_TOL = {"s1:all": 1e-9, "nll:all": 1e-6, "replay_nll": 1e-6, "tool_token_agree": 1e-9, "tool_turn_agree": 1e-9,
            "recall:300": 1e-9, "kl_to_untouched": 3e-8,  # absolute; KL of near-equal fp32 distributions: ~1e-8 floor
            "gate_rate_s1": 1e-6, "gate_rate_text": 1e-6}


def same_evals(ea, eb):
    """Every eval key of eb at every step of eb, within EVAL_TOL. Measured on the fixture: 1 vs 2 ranks differ by at
    most 2.4e-7 in an NLL and 1.2e-8 in KL; another seed moves nll:all by 6.7e-4."""
    for s in eb:
        check(s in ea, f"eval step {s} missing")
        for k, tol in EVAL_TOL.items():
            if k in eb[s] or k in ea[s]:
                a, b = ea[s].get(k), eb[s].get(k)
                check(a is not None and b is not None, f"step {s} {k}: {a} vs {b}")
                check(abs(a - b) <= tol, f"step {s} {k}: {a} vs {b}")


DIST = {}


def dist_dir(name):
    return FX / "runs" / name


def test_dist_world_size_equality():
    """Steps, evals and the final adapter agree between 1 and 2 ranks (the global draw, LPT and global normalisers
    make the world size invisible up to float summation order). Also: the tool eval rows are held out, and the replicas
    agreed at every save. Red arm: seed 1 on 2 ranks is far outside the same tolerance."""
    a, b, c = dist_dir("w1"), dist_dir("w2"), dist_dir("w2_seed1")
    torchrun(1, a)
    torchrun(2, b)
    torchrun(2, c, seed=1)
    DIST["w1"], DIST["w2"] = a, b
    sa, sb = steps(a), steps(b)
    check(set(sa) == set(sb) == {1, 6}, f"logged steps {sorted(sa)} {sorted(sb)}")
    for s in sa:
        for k, v in sa[s]["loss"].items():
            check(rel(v, sb[s]["loss"][k]) < 1e-4, f"step {s} {k} loss {v} vs {sb[s]['loss'][k]}")
    same_evals(evals(a), evals(b))
    check(set(evals(a)) == {0, 3, 6}, f"eval steps {sorted(evals(a))}")
    d = trainable_diff(a / "trainable_000006.pt", b / "trainable_000006.pt")
    check(d < 1e-4, f"1 vs 2 ranks: final adapter max rel diff {d:.2e}")
    cfg = [r for r in log_of(b) if r.get("event") == "config"][0]
    check(cfg["tool_eval_heldout"] and cfg["pools"]["tool"]["held_out_for_eval"] == 2, "tool eval rows not held out")
    check(not [r for r in log_of(b) if r.get("event") == "replica_mismatch"], "replicas diverged")
    # main()'s update path goes downhill: the fixed S1 eval answers and the held-out replay get likelier. The gradient
    # tests above replicate the loop rather than call it, and the run comparisons here compare one code path with
    # itself, so a sign or ordering error in main()'s all-reduce, clip or step would pass every other check.
    # (S1 answers carry weight 1.0 and fall by more than 1e-3 here; the near-uniform tiny model's replay NLL moves by
    # about 2e-5 in 6 steps at weight 0.15, above the zero-LR floor of exactly 0 but not by a fixed margin)
    ea = evals(a)
    check(ea[6]["nll:all"] < ea[0]["nll:all"] - 1e-3,
          f"S1 answer NLL did not fall: step 0 {ea[0]['nll:all']:.5f}, step 6 {ea[6]['nll:all']:.5f}")
    check(ea[6]["replay_nll"] < ea[0]["replay_nll"],
          f"replay NLL rose: step 0 {ea[0]['replay_nll']:.6f}, step 6 {ea[6]['replay_nll']:.6f}")
    z = dist_dir("w1_lr0")
    torchrun(1, z, ["--lr", "0", "--w_lr", "0"])
    ez = evals(z)
    check(all(abs(ez[6][k] - ez[0][k]) < 1e-9 for k in ("nll:all", "replay_nll")),
          "red arm: a zero learning rate still moved the evals, so the fall above is not the update's doing")
    d_red = trainable_diff(c / "trainable_000006.pt", b / "trainable_000006.pt")
    check(d_red > 1e-2, f"red arm: another seed matched within {d_red:.2e}")
    try:
        same_evals(evals(c), {s: e for s, e in evals(b).items() if s > 0})
        raise RuntimeError("red arm: another seed's evals passed same_evals")
    except AssertionError:
        pass


def test_dist_resume_equality_and_eval_rerun():
    """3 steps, stop, resume to 6 (2 ranks) equals the unbroken 2-rank run; a 1-rank resume of the 2-rank checkpoint
    equals the unbroken 1-rank run (resume may change the world size); a bogus newer checkpoint is skipped; an eval
    missing from the log at the resume step is re-run, a present one is not. Red arm: --lr_override to another rate
    diverges from the unbroken run, and its step log shows the new rate."""
    b1 = DIST.get("w1") or dist_dir("w1")
    b2 = DIST.get("w2") or dist_dir("w2")
    if not (b2 / "trainable_000006.pt").exists():
        torchrun(1, b1)
        torchrun(2, b2)
    r = dist_dir("res2")
    torchrun(2, r, ["--max_steps_debug", "3"])
    check((r / "complete.json").exists() and (r / "ckpt_000003.pt").exists(), "3-step run did not finish")
    x = dist_dir("res1")
    shutil.copytree(r, x)
    y = dist_dir("res_lr")
    shutil.copytree(r, y)
    for d in (r, x, y):
        (d / "complete.json").unlink()
    torchrun(2, r, ["--resume", "--mem_probe", "1"])
    d2 = trainable_diff(r / "trainable_000006.pt", b2 / "trainable_000006.pt")
    check(d2 < 1e-6, f"2-rank resume vs unbroken 2-rank run: {d2:.2e}")
    check(sum(1 for e in log_of(r) if e.get("event") == "eval" and e["step"] == 3) == 1, "eval 3 was re-run")
    # eval 3 is logged, so no eval ran before steps 4 to 6: they and the mem_probe must still train in train mode
    # (gradient checkpointing on). The pre-fix trainer logged train_mode false here.
    probes = [e for e in log_of(r) if e.get("event") == "mem_probe"]
    check({e["pool"] for e in probes} == {"s1", "chat", "tool"} and all(e["train_mode"] for e in probes),
          f"mem_probe after resume: {[(e['pool'], e.get('train_mode')) for e in probes]}")
    check(steps(r)[6]["train_mode"] is True, "resumed steps ran with the layers in eval mode (no checkpointing)")
    # 1-rank resume of the 2-rank checkpoint, with a bogus newer checkpoint and eval 3 stripped from the log
    (x / "ckpt_000005.pt").write_bytes(b"not a checkpoint")
    lines = [l for l in open(x / "log.jsonl") if not ('"event": "eval"' in l and '"step": 3,' in l)]
    (x / "log.jsonl").write_text("".join(lines))
    run = torchrun(1, x, ["--resume"])
    check('"step": 3' in run.stdout and "ckpt_000005.pt" in run.stdout, "resume did not skip the bogus file")
    d1 = trainable_diff(x / "trainable_000006.pt", b1 / "trainable_000006.pt")
    check(d1 < 1e-4, f"1-rank resume of a 2-rank checkpoint vs unbroken 1-rank run: {d1:.2e}")
    check(sum(1 for e in log_of(x) if e.get("event") == "eval" and e["step"] == 3) == 1, "eval 3 was not re-run")
    same_evals(evals(x), {6: evals(b1)[6], 3: evals(b1)[3]})
    torchrun(2, y, ["--resume", "--lr_override", "--lr", "3e-4", "--w_lr", "3e-4"])
    check(abs(steps(y)[6]["lr"] - 3e-4) < 1e-12 and abs(steps(y)[6]["lr_w"] - 3e-4) < 1e-12, "override not in effect")
    d_red = trainable_diff(y / "trainable_000006.pt", b2 / "trainable_000006.pt")
    check(d_red > 1e-3, f"red arm: a different rate after resume matched ({d_red:.2e})")


def test_dist_uneven_eval_sharding_world3():
    """World 3 with 2 S1 eval sessions, 1 held-out tool row and 1 recall item: some ranks own no eval item, every
    collective still matches (the run finishes inside the timeout), and the evals equal a 1-rank run. Red arm
    (non-vacuity): the config must show a rank with no S1 and no recall item, or the empty branch was not run, and the
    gate must fire (--w_lr 5e-2), or the cross-rank gate-rate aggregation is compared at 0 = 0."""
    extra = ["--eval_per_domain", "1", "--eval_tool_rows", "1", "--steps", "3", "--w_lr", "5e-2"]
    a, b = dist_dir("u3"), dist_dir("u1")
    torchrun(3, a, extra, timeout=900)
    torchrun(1, b, extra)
    cfg = [r for r in log_of(a) if r.get("event") == "config"][0]["eval_items_per_rank"]
    check(0 in cfg["s1"] and 0 in cfg["recall"] and 0 in cfg["tool"], f"no empty rank: {cfg}")
    check(sum(cfg["s1"]) == 2 and sum(cfg["recall"]) == 1 and sum(cfg["tool"]) == 1, f"item counts {cfg}")
    check(evals(a)[3]["gate_rate_s1"] > 0 and evals(a)[3]["gate_rate_text"] > 0, "the gate never fired")
    same_evals(evals(a), evals(b))


def test_dist_eval27_matches_trainer():
    """eval27.py on a step-3 adapter (the gate firing, --w_lr 5e-2) scores the same sessions exactly as the trainer's
    step-3 eval (same answers, same NLL, same token-weighted gate rate). Red arm: the untouched model
    (--trainable none) gives a different NLL and a zero gate rate."""
    b2 = dist_dir("g2")
    torchrun(2, b2, ["--steps", "3", "--w_lr", "5e-2"])
    base = ["--model", str(FX / "model"), "--arm", "wide", "--s1", str(FX / "s1"), "--lengths", "32", "--rank", "4",
            "--dtype", "fp32"]
    o1, o0 = dist_dir("ev27"), dist_dir("ev27_none")
    torchrun(2, o1, [*base, "--trainable", str(b2 / "trainable_000003.pt")], script="eval27.py", common=False)
    torchrun(2, o0, [*base, "--trainable", "none"], script="eval27.py", common=False)
    s1, s0 = json.loads((o1 / "summary.json").read_text()), json.loads((o0 / "summary.json").read_text())
    tr = evals(b2)[3]
    check(s1["n_answers"] == tr["n:all"], f"answers {s1['n_answers']} vs {tr['n:all']}")
    check(s1["all"]["acc"] == tr["s1:all"], "exact match differs")
    check(rel(s1["all"]["answer_nll"], tr["nll:all"]) < 1e-5, f"NLL {s1['all']['answer_nll']} vs {tr['nll:all']}")
    check(rel(s0["all"]["answer_nll"], evals(b2)[0]["nll:all"]) < 1e-5, "untouched eval27 differs from step 0")
    check(tr["gate_rate_s1"] > 0 and rel(s1["gate_rate"], tr["gate_rate_s1"]) < 1e-6,
          f"gate rate {s1.get('gate_rate')} vs trainer {tr['gate_rate_s1']}")
    check(s0["gate_rate"] == 0.0, "untouched model fired the gate")
    check(rel(s0["all"]["answer_nll"], tr["nll:all"]) > 1e-4, "red arm: the untouched model matched the adapter")


def test_dist_train_mode_eval_every_0():
    """An --eval_every 0 run (the accept smoke's 8-rank timing part has this shape) trains and probes memory in train
    mode, so its tok/s and mem_probe are taken with gradient checkpointing as configured. Red arm: the unit test
    test_train_mode_checkpointing shows train_mode reads false in eval mode, so this field is not constant."""
    d = dist_dir("ev0")
    torchrun(1, d, ["--eval_every", "0", "--recall_every", "0", "--steps", "1", "--mem_probe", "1"])
    probes = [e for e in log_of(d) if e.get("event") == "mem_probe"]
    check(probes and all(e["train_mode"] for e in probes), f"mem_probe train_mode {[e.get('train_mode') for e in probes]}")
    check(steps(d)[1]["train_mode"] is True, "an --eval_every 0 run trained with the layers in eval mode")
    check(not evals(d), "an --eval_every 0 run logged an eval")


EVAL27_DRIVER = """
import sys
sys.path.insert(0, {here!r})
import eval27
import test_train27_cpu as TT
args = sys.argv[1:]
mode = args.pop(2)  # the torchrun helper puts --out OUT first, then the mode, then eval27's flags
oracle = {{"good": lambda: TT._Oracle(1), "current": lambda: TT._Oracle(0),
          "bad_end": lambda: TT._Oracle(1, wrong={im_end})}}[mode]()
eval27.build_model = lambda args, device: (oracle, 0)
sys.argv = ["eval27.py"] + args
eval27.main()
"""


def test_dist_eval27_oracle():
    """eval27's own exact match, not 0 = 0 on a random model: an oracle that knows every next token scores every answer
    of every session, an oracle that is wrong only on <|im_end|> fails every answer (PLAN.md's eval27 check), and a
    current-token oracle scores nothing. The answer count is the number of assistant turns."""
    tok = tokenizer()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    drv = FX / "eval27_oracle_driver.py"
    drv.write_text(EVAL27_DRIVER.format(here=str(HERE), im_end=im_end))
    sessions = T.load_jsonl(sorted((FX / "s1").glob("*.eval.jsonl")))
    n_turns = sum(len(r["turns"]) for r in sessions)
    out = {}
    for mode in ("good", "bad_end", "current"):
        o = dist_dir(f"ev27_oracle_{mode}")
        torchrun(1, o, [mode, "--model", str(FX / "model"), "--arm", "ctrl", "--trainable", "none",
                        "--s1", str(FX / "s1"), "--dtype", "fp32"], script=str(drv), common=False)
        out[mode] = json.loads((o / "summary.json").read_text())
    g = out["good"]
    check(g["n_answers"] == n_turns and g["n_sessions"] == len(sessions), f"{g['n_answers']} answers, {n_turns} turns")
    check(g["all"]["acc"] == 1.0, f"oracle accuracy {g['all']['acc']}")
    check(out["bad_end"]["all"]["acc"] == 0.0, f"a wrong <|im_end|> scored {out['bad_end']['all']['acc']}")
    check(out["current"]["all"]["acc"] == 0.0, "red arm: a current-token oracle scored")


UNIT = [test_draw_epochs_and_process_determinism, test_lpt_assignment, test_seq_loss_alignment_single_and_padded,
        test_grad_equivalence_micro_batching, test_untouched_context, test_eval_indexing_oracle_and_nll,
        test_gate_stats_per_call, test_gate_rate_token_weighted_and_text, test_eval_lm_kl_direction,
        test_set_lrs_after_resume,
        test_checkpoints_retention_atomic_and_fallback, test_length_table_cache, test_tool_hold_out,
        test_arm_pairing_lora_init, test_train_mode_checkpointing, test_recall_items]
DISTRIBUTED = [test_dist_world_size_equality, test_dist_resume_equality_and_eval_rerun,
               test_dist_uneven_eval_sharding_world3, test_dist_eval27_matches_trainer, test_dist_train_mode_eval_every_0,
               test_dist_eval27_oracle]


def main():
    global FX
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/data/ai-ml/hf-models/qwen3.8-27b-tokenizer")
    ap.add_argument("--data", default="/data/ai-ml/models/_runs/negeig-27b/data")
    ap.add_argument("--only", default="", help="comma list of test name substrings")
    ap.add_argument("--skip-dist", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the fixture dir")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    FX = build_fixture(a.tokenizer, a.data)
    print(f"fixture {FX}", flush=True)
    (FX / "tmp").mkdir(exist_ok=True)  # this process's temp files (torch inductor) also go away with the fixture
    os.environ["TMPDIR"] = tempfile.tempdir = str(FX / "tmp")
    try:
        import_train27()
        tests = UNIT + ([] if a.skip_dist else DISTRIBUTED)
        if a.only:
            tests = [t for t in tests if any(s in t.__name__ for s in a.only.split(","))]
        for t in tests:
            run_test(t.__name__, t)
    finally:
        if not a.keep:
            shutil.rmtree(FX, ignore_errors=True)
    fails = [r for r in RESULTS if r[1] != "PASS"]
    print(json.dumps({"passed": len(RESULTS) - len(fails), "failed": [r[0] for r in fails]}))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
