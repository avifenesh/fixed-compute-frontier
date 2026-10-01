#!/usr/bin/env python3
"""CPU tests for the tier-1 probe (train_t1.py, decide.py, run_t1.sh). No GPU: the tiny random qwen3_5 of the train27
smoke assets stands in for the 27B, with fla blocked so transformers runs its torch Gated DeltaNet path.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 taskset -c 8-15 \\
        ~/.venvs/negeig/bin/python test_t1_cpu.py --assets DIR [--only a,b] [--skip-dist] [--keep]

--assets is a dir with model/ (tiny qwen3_5 with the Qwen3.8 tokenizer), nofla/ (an fla package that refuses to import)
and replay.pt ({'train','test'} token chunks). Unit tests run in-process at world 1; the dist_* tests run train_t1.py
under torchrun (gloo) and compare runs; the box test runs run_t1.sh end to end on CPU. Every check has a red arm (a
deliberately wrong variant the same check rejects). Not covered here: bf16 numerics, fla kernels, NCCL, the 27B's
memory and speed (the box's first steps and mem probe measure those).
"""

from __future__ import annotations

import argparse
import contextlib
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
E27 = HERE.parent.parent
RETRO = E27.parent / "negeig_retrofit"
PYBIN = Path(sys.executable).parent
RESULTS = []
A: Path | None = None  # assets
FX: Path | None = None  # scratch fixture dir
T1 = TR = tasks = None  # modules, imported after fla is blocked


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


def build(arm, seed=0):
    import torch
    torch.manual_seed(seed)
    ns = types.SimpleNamespace(model=str(A / "model"), dtype="fp32", arm=arm, rank=4, grad_ckpt=1)
    model, _ = T1.T27.build_model(ns, torch.device("cpu"))
    return model


def perturb(model, seed=7, w_scale=0.5, b_scale=0.05):
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "negeig_w" in n:
                p.copy_(torch.randn(p.shape, generator=g) * w_scale)
            elif "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * b_scale)


def vocab():
    from transformers import AutoTokenizer
    return tasks.Vocab(AutoTokenizer.from_pretrained(A / "model"))


@contextlib.contextmanager
def cuda_is_cpu():
    """train.py calls .cuda() on every tensor; on this CPU-only machine make it a no-op so train.py's own functions run."""
    import torch
    old = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self
    try:
        yield
    finally:
        torch.Tensor.cuda = old


def grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def max_rel_diff(ga, gb):
    check(set(ga) == set(gb), f"grad key sets differ: {sorted(set(ga) ^ set(gb))[:3]}")
    return max(float((ga[k] - gb[k]).abs().max()) for k in ga) / max(float(gb[k].abs().max()) for k in gb)


def same_ex(a, b):
    return (a.task, a.ids, a.label_pos, a.labels, a.n_steps) == (b.task, b.ids, b.label_pos, b.labels, b.n_steps)


# ------------------------------------------------------------------ unit tests
def test_stream_and_replay_match_train_py():
    """Stream reproduces train.py's task stream example for example (one random.Random(seed), counts order), random
    access equals sequential access (resume), and the replay chunks are train.py's replay_chunks rows. Red arm: another
    seed gives another stream."""
    import random
    import torch
    V = vocab()
    counts = [("parity", 3), ("swap", 4), ("codeswap", 2)]
    rng = random.Random(5)
    ref = [[(t, [tasks.make(t, 16, rng, V) for _ in range(n)]) for t, n in counts] for _ in range(6)]
    s = T1.Stream(V, 5, counts, 16)
    seq = [s.batch(k) for k in range(6)]
    for k in range(6):
        for (ta, ea), (tb, eb) in zip(seq[k], ref[k]):
            check(ta == tb and len(ea) == len(eb) and all(same_ex(x, y) for x, y in zip(ea, eb)), f"step {k} differs")
    fresh = T1.Stream(V, 5, counts, 16)
    check(all(same_ex(x, y) for x, y in zip(fresh.batch(4)[2][1], ref[4][2][1])), "fast-forward to step 4 differs")
    check(all(same_ex(x, y) for x, y in zip(fresh.batch(1)[0][1], ref[1][0][1])), "going back to step 1 differs")
    other = T1.Stream(V, 6, counts, 16).batch(0)
    check(not all(same_ex(x, y) for x, y in zip(other[1][1], ref[0][1][1])), "red arm: seed 6 gave seed 5's stream")
    # replay: train.py takes randperm(seed)[:steps * k] and slices k per step
    with cuda_is_cpu():
        tr_chunks = TR.replay_chunks(str(A / "replay.pt"), "train", 5 * 3, 64, 2)
    train = torch.load(A / "replay.pt")["train"][:, :64]
    perm = T1.replay_perm(train.shape[0], 2)
    for k in range(5):
        mine = T1.step_chunks(train, perm, k, 3)
        check(mine == tr_chunks[k * 3:(k + 1) * 3].tolist(), f"replay chunks of step {k} differ from train.py")
    check(T1.step_chunks(train, T1.replay_perm(train.shape[0], 3), 0, 3) != tr_chunks[:3].tolist(),
          "red arm: another seed picked the same replay chunks")


def test_plan_world_invariant_and_lpt_cover():
    """plan_step: single-task groups of max(1, micro_tokens // length) (train.py's rule), replay groups of replay_micro,
    nothing lost or duplicated; the plan has no world-size input, and lpt gives every micro-batch exactly one owner on
    1 to 4 ranks. Red arm: micro_tokens 0 gives one group per task (fewer micro-batches)."""
    V = vocab()
    groups = T1.Stream(V, 0, [("parity", 16), ("swap", 64), ("codeswap", 64)], 64).batch(0)
    chunks = [[1] * 512 for _ in range(12)]
    mbs, cost = T1.plan_step(groups, chunks, 4096, 4)
    per_task = {}
    for kind, b in mbs:
        if kind == "t1":
            check(len({e.task for e in b}) == 1, "a micro-batch mixes tasks")
            per_task[b[0].task] = per_task.get(b[0].task, 0) + len(b)
            check(len(b) <= max(1, 4096 // len(b[0].ids)), "group above the micro_tokens rule")
    check(per_task == {"parity": 16, "swap": 64, "codeswap": 64}, f"examples lost or duplicated: {per_task}")
    check(sum(len(b) for k, b in mbs if k == "lm") == 12, "replay chunks lost")
    check(sum(cost) == sum(len(e.ids) for _, g in groups for e in g) + 12 * 512, "costs are not the padded tokens")
    for world in (1, 2, 3, 4):
        owner, load = T1.T27.lpt(cost, world)
        check(sorted(set(owner)) == list(range(min(world, len(mbs)))) and len(owner) == len(mbs), f"world {world} owners")
        check(sum(load) == sum(cost), "load lost")
    one, _ = T1.plan_step(groups, chunks, 0, 12)
    check(len(one) == 4 and len(mbs) > len(one), "red arm: micro_tokens 0 did not collapse to one group per task")


def test_loss_and_grad_match_train_py():
    """One step's objective and gradient equal train.py's (task_loss per micro-batch weighted by its share of the
    sequences, plus lm_loss on the replay chunks), on the tiny model with nonzero LoRA and gate. Red arm: labels read one
    position early give another loss."""
    import torch
    V = vocab()
    model = build("wide")
    perturb(model)
    groups = T1.Stream(V, 3, [("parity", 3), ("swap", 4), ("codeswap", 3)], 12).batch(0)
    train = torch.load(A / "replay.pt")["train"][:, :64]
    chunks = T1.step_chunks(train, T1.replay_perm(train.shape[0], 3), 0, 4)
    exs = [e for _, g in groups for e in g]
    # train.py: micro-batches of one task, (l_mb * len(g) / len(exs)).backward(), then lm_loss over the chunks
    model.zero_grad(set_to_none=True)
    ref = 0.0
    with cuda_is_cpu():
        for _, g in groups:
            for a in range(0, len(g), 2):
                sub = g[a:a + 2]
                l_mb, _ = TR.task_loss(model, sub, 0)
                (l_mb * len(sub) / len(exs)).backward()
                ref += l_mb.item() * len(sub) / len(exs)
        l_r = TR.lm_loss(model, torch.tensor(chunks).long())
        l_r.backward()
        ref += l_r.item()
    g_ref = grads(model)
    # train_t1: global micro-batches, summed losses over the step's global counts
    model.zero_grad(set_to_none=True)
    mbs, _ = T1.plan_step(groups, chunks, 60, 3)
    n_labels = sum(len(e.labels) for e in exs)
    n_lm = len(chunks) * (len(chunks[0]) - 1)
    mine = 0.0
    for kind, b in mbs:
        if kind == "t1":
            tot, n = T1.t1_loss(model, b, torch.device("cpu"))
            loss = tot / n_labels
        else:
            tot, n = T1.T27.seq_loss(model, [(x, [1] * len(x)) for x in b], torch.device("cpu"))
            loss = tot / n_lm
        loss.backward()
        mine += loss.item()
    g_mine = grads(model)
    check(rel(mine, ref) < 1e-5, f"objective {mine} vs train.py {ref}")
    d = max_rel_diff(g_mine, g_ref)
    check(d < 1e-4, f"gradient max rel diff {d:.2e}")
    check(any("negeig_w" in k for k in g_mine), "the gate got no gradient")
    # Red arms for the objective comparison itself: the two weighting slips a data-parallel rewrite makes, each far
    # outside the 1e-5 tolerance above (measured 1.3e-3 and 7.8e-3 on this model). (a) every task micro-batch weighted
    # equally (its mean label loss) instead of by its share of the step's labels; the groups are uneven here (60 padded
    # tokens hold 3 parity, 3 or 1 swap, 1 codeswap). (b) the replay divided by replay_len tokens a chunk instead of
    # replay_len - 1 targets.
    t1_means, t1_good, lm_good, lm_bad = [], 0.0, 0.0, 0.0
    with torch.no_grad():
        for kind, b in mbs:
            if kind == "t1":
                tot, n = T1.t1_loss(model, b, torch.device("cpu"))
                t1_means.append(float(tot) / n)
                t1_good += float(tot) / n_labels
            else:
                tot, n = T1.T27.seq_loss(model, [(x, [1] * len(x)) for x in b], torch.device("cpu"))
                lm_good += float(tot) / n_lm
                lm_bad += float(tot) / (len(chunks) * len(chunks[0]))
    check(rel(t1_good + lm_good, ref) < 1e-5, "the no-grad recomputation of the objective drifted")
    check(len({len(b) for k, b in mbs if k == "t1"}) > 1, "red arm needs uneven task micro-batches")
    check(rel(sum(t1_means) / len(t1_means) + lm_good, ref) > 1e-4,
          "red arm: equal weight per micro-batch passed as train.py's objective")
    check(rel(t1_good + lm_bad, ref) > 1e-4, "red arm: replay normalised by replay_len passed as train.py's objective")
    shifted = [tasks.Example(e.task, e.ids, [p - 1 for p in e.label_pos], e.labels, e.n_steps) for e in exs[:3]]
    with torch.no_grad():
        a, _ = T1.t1_loss(model, exs[:3], torch.device("cpu"))
        b, _ = T1.t1_loss(model, shifted, torch.device("cpu"))
    check(rel(float(a), float(b)) > 1e-3, "red arm: labels one position early gave the same loss")


def test_eval_matches_train_py():
    """eval_tasks equals train.py's eval_tasks key for key (per-step, per-sequence and window accuracy) for all five
    tasks at two lengths, with a final batch shorter than the rest. Red arm: another seed's sets differ."""
    import torch
    V = vocab()
    model = build("wide")
    perturb(model)
    for L in (70, 300):
        with cuda_is_cpu():
            ref = TR.eval_tasks(model, V, [L], 6, 4, 0, seed=10_000)
        mine, gate = T1.eval_tasks(model, V, L, 6, 4, torch.device("cpu"), 0, 1, 10_000)
        for t in tasks.ALL_TASKS:
            for suf in ("", "/seq", "/win"):
                k = f"{t}@{L}{suf}"
                check(mine[k] == ref[k], f"{k}: {str(mine[k])[:120]} vs {str(ref[k])[:120]}")
        check(len(gate["beta_gt1_frac_per_layer"]) == 3, "gate stats missing for the 3 GDN layers")
    other, _ = T1.eval_tasks(model, V, 70, 6, 4, torch.device("cpu"), 0, 1, 10_001)
    with cuda_is_cpu():
        ref70 = TR.eval_tasks(model, V, [70], 6, 4, 0, seed=10_000)
    check(any(other[f"{t}@70"] != ref70[f"{t}@70"] for t in tasks.ALL_TASKS), "red arm: seed 10,001 gave seed 10,000's scores")


def oracle_states(task, ids, V, wrong_steps=(), shift=False):
    """Independent recomputation of the label index after every token from the token stream alone (not tasks.make's
    code). wrong_steps: step indices (0-based) whose label is deliberately wrong; shift: emit the state BEFORE the
    token (one step late)."""
    import itertools
    h = len(V.header[task])
    body = ids[h:]
    out = [0] * h
    inv_letter = {t: k for k, t in enumerate(V.letter)}
    cups = "abcde"
    pairs = [a + b for a, b in itertools.combinations(cups, 2)]
    perms = list(itertools.permutations(range(5)))
    m = {"parity": 2, "z3": 3}.get(task, 5)
    step = 0

    def emit(state, prev):
        nonlocal step
        s = prev if shift else state
        out.append((s + 1) % m if step in wrong_steps else s)
        step += 1

    if task in ("parity", "z3"):
        syms = V.bit if task == "parity" else V.tri
        s = 0
        for t in body:
            p, s = s, (s + syms.index(t)) % m
            emit(s, p)
    elif task in ("swap", "s5full"):
        ball = inv_letter[body[0]]
        out += [0, 0]
        inv_pair = {t: k for k, t in enumerate(V.pair)}
        inv_perm = {t: k for k, t in enumerate(V.perm_word)}
        for t in body[2:]:
            p = ball
            if task == "swap":
                a, b = (cups.index(c) for c in pairs[inv_pair[t]])
                ball = b if ball == a else a if ball == b else ball
            else:
                ball = perms[inv_perm[t]][ball]
            emit(ball, p)
    else:
        stmt = {tuple(v): k for k, v in V.code_stmt.items()}
        n0 = len(V.code_init)
        out += [0] * n0
        val, buf = list(range(5)), []
        for t in body[n0:]:
            buf.append(t)
            if tuple(buf) in stmt:
                x, y = stmt[tuple(buf)]
                p = val[0]
                i, j = cups.index(x), cups.index(y)
                val[i], val[j] = val[j], val[i]
                emit(val[0], p)
                buf = []
            else:
                out.append(0)
    return out


def oracle_model(V, **kw):
    """A fake (decoder, head): the decoder emits e_k (dim 5) where k = oracle_states at every position, the head maps
    the five letter tokens to e_0..e_4, so the restricted argmax returns the oracle's label."""
    import torch
    import torch.nn as nn

    class Dec:
        def __call__(self, input_ids, attention_mask=None, use_cache=False):
            B, Lx = input_ids.shape
            h = torch.zeros(B, Lx, 5)
            for i in range(B):
                ids = input_ids[i].tolist()
                n = int(attention_mask[i].sum()) if attention_mask is not None else Lx
                task = next(t for t in tasks.ALL_TASKS if ids[:len(V.header[t])] == V.header[t])
                st = oracle_states(task, ids[:n], V, **kw)
                h[i, torch.arange(n), torch.tensor(st)] = 1.0
            return types.SimpleNamespace(last_hidden_state=h)

    w = torch.zeros(248320, 5)
    for k, t in enumerate(V.letter):
        w[t, k] = 1.0
    return nn.Module(), Dec(), types.SimpleNamespace(weight=w)


def test_eval_oracle_and_windows():
    """With an oracle decoder every window is exact; an oracle wrong exactly on steps 65-256 gives w1 = 1, w4 = 0,
    w16 = 1, which pins the windows to steps 1-64 / 65-256 / 257+. Red arm: the oracle one step late is far from exact
    on the tier-1 tasks."""
    import torch
    V = vocab()
    old = T1.T27.core
    try:
        for kw, want in (({}, (1.0, 1.0, 1.0)), ({"wrong_steps": set(range(64, 256))}, (1.0, 0.0, 1.0))):
            fake, dec, head = oracle_model(V, **kw)
            T1.T27.core = lambda m, d=dec, h=head: (d, h)
            res, _ = T1.eval_tasks(fake, V, 300, 5, 2, torch.device("cpu"), 0, 1, 10_000)
            wm = T1.window_means(res, 300)
            for t in tasks.ALL_TASKS:
                got = tuple(round(wm[t][k], 6) for k in ("w1", "w4", "w16"))
                check(got == want, f"oracle {kw and 'wrong 65-256' or 'exact'} {t}: {got} vs {want}")
        fake, dec, head = oracle_model(V, shift=True)
        T1.T27.core = lambda m, d=dec, h=head: (d, h)
        res, _ = T1.eval_tasks(fake, V, 300, 5, 2, torch.device("cpu"), 0, 1, 10_000)
        wm = T1.window_means(res, 300)
        check(all(wm[t]["w1"] < 0.8 for t in tasks.TIER1), f"red arm: a one-step-late oracle scored {wm}")
    finally:
        T1.T27.core = old


def test_untouched_identity_and_gate_rate():
    """At init (LoRA B = 0, W = 0) wide and ctrl score identically (G0 through eval_tasks) and the gate never fires; with
    W perturbed the gate fires and the scores change (red arm). gate_layers is the token-weighted mean of patch.py's
    per-forward means."""
    import torch
    V = vocab()
    w, c = build("wide"), build("ctrl")
    rw, gw = T1.eval_tasks(w, V, 70, 4, 4, torch.device("cpu"), 0, 1, 10_000)
    rc, gc = T1.eval_tasks(c, V, 70, 4, 4, torch.device("cpu"), 0, 1, 10_000)
    check(rw == rc, "wide at W = 0 scores differently from ctrl")
    check(gw["beta_gt1_frac_per_layer"] == [0.0] * 3 and gc["beta_gt1_frac_per_layer"] == [], f"gate stats {gw} {gc}")
    perturb(w)
    rp, gp = T1.eval_tasks(w, V, 70, 4, 4, torch.device("cpu"), 0, 1, 10_000)
    check(max(gp["beta_gt1_frac_per_layer"]) > 0.05, f"perturbed gate never fired: {gp}")
    check(rp != rc, "red arm: a perturbed wide model scored exactly like ctrl")
    # token weighting: two forwards of different lengths
    T1.patch.enable_stats(w, True)
    with torch.no_grad():
        dec, _ = T1.T27.core(w)
        for n in (40, 130):
            T1.T27.fwd(dec, torch.randint(0, 1000, (1, n)))
    raw = [layer._negeig_stats for layer in T1.patch.gdn_layers(w)]
    frac, _ = T1.gate_layers(w, [40, 130], torch.device("cpu"), 1)
    want = [(st[0][0] * 40 + st[1][0] * 130) / 170 for st in raw]
    check(all(abs(a - b) < 1e-12 for a, b in zip(frac, want)), f"gate_layers {frac} vs {want}")
    T1.patch.enable_stats(w, False)


def _rec(acc_w4, acc_other, gate, n=8, L=300, arm="wide"):
    """A synthetic test record: every sequence has the given window accuracies for each task."""
    tasks_ = {}
    for t in ("parity", "swap", "codeswap", "s5full", "z3"):
        a4 = acc_w4[t] if isinstance(acc_w4, dict) else acc_w4
        tasks_[f"{t}@{L}/win"] = [[acc_other, a4, acc_other] for _ in range(n)]
        tasks_[f"{t}@{L}/seq"] = [a4] * n
        tasks_[f"{t}@{L}"] = [a4] * L
    return {"arm": arm, "test_len": L, "tasks": tasks_, "nll": [2.0], "negeig_stats": {"beta_gt1_frac_per_layer": gate}}


def _decide(root, step, final, extra=()):
    r = subprocess.run([sys.executable, str(HERE / "decide.py"), "--root", str(root), "--seed", "0", "--step", str(step),
                        "--final", str(final), "--B", "200", *extra], capture_output=True, text=True, timeout=300)
    return r.returncode, (json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}), r.stderr


def test_decide_rule_synthetic():
    """decide.py applies the written rule: GO (D >= 10, gate used), REVIEW (D >= 10, gate unused; or configs differ),
    EXTEND (D < 10 before the final rung), KILL (D < 10 at the final rung; or ctrl >= 0.60 at 4x). Missing records exit
    2. Each verdict is a red arm for the others."""
    root = FX / "decide"

    def setup(wide4, ctrl4, gate, cfg_seed_ctrl=0):
        shutil.rmtree(root, ignore_errors=True)
        for arm, a4, g in (("wide", wide4, gate), ("ctrl", ctrl4, [])):
            d = root / f"{arm}_s0"
            d.mkdir(parents=True)
            (d / "test_step000000.json").write_text(json.dumps(_rec(0.3, 0.5, g if arm == "wide" else [], arm=arm)))
            (d / "test_step001200.json").write_text(json.dumps(_rec(a4, 0.9, g, arm=arm)))
            seed = cfg_seed_ctrl if arm == "ctrl" else 0
            (d / "log.jsonl").write_text(json.dumps({"event": "config", "args": {"arm": arm, "seed": seed, "lr": 1e-4}})
                                         + "\n" + json.dumps({"event": "eval", "step": 50, "tier1": {},
                                                              **{t: {"w1": 0.97, "w4": 0.5} for t in
                                                                 ("parity", "swap", "codeswap")}}) + "\n")

    cases = [((0.8, 0.3, [0.02, 0.2]), 0, "GO"), ((0.8, 0.3, [0.001, 0.002]), 0, "REVIEW"),
             ((0.35, 0.3, [0.2]), 0, "EXTEND"), ((0.35, 0.3, [0.2]), 1, "KILL"), ((0.75, 0.7, [0.2]), 0, "KILL")]
    for (w4, c4, g), final, want in cases:
        setup(w4, c4, g)
        rc, out, err = _decide(root, 1200, final)
        check(rc == 0 and out.get("verdict") == want, f"w4 {w4} c4 {c4} gate {g} final {final}: {out or err[-500:]}")
    rec = json.loads((root / "decision_step001200.json").read_text())
    check(abs(rec["accuracy"]["tier1/w4"]["diff_points"] - 5.0) < 1e-9, "D is not 100 x (wide - ctrl)")
    check(rec["step0_identity"]["identical"] and rec["switch_step_val_w1_ge_0.95"]["wide"]["parity"] == 50, "extras")
    setup(0.8, 0.3, [0.2], cfg_seed_ctrl=1)
    rc, out, _ = _decide(root, 1200, 0)
    check(out.get("verdict") == "REVIEW", f"red arm: arms with different seeds were judged: {out}")
    (root / "ctrl_s0" / "test_step001200.json").unlink()
    rc, out, _ = _decide(root, 1200, 0)
    check(rc == 2, "a missing record did not exit 2")


def test_decide_paired_bootstrap():
    """decide.py's interval is paired by test sequence (both arms score the same sequences): wide = ctrl + 0.1 on every
    sequence, with ctrl spread from 0 to 0.9 across sequences, gives D = 10 with a zero-width interval. Red arm: the same
    two score sets with ctrl's sequence order shuffled (pairing lost, same means) give an interval tens of points wide."""
    import random
    sys.path.insert(0, str(HERE))
    import decide
    n, L = 40, 300

    def rec(vals):
        return {"test_len": L, "tasks": {f"{t}@{L}/win": [[v, v, v] for v in vals] for t in decide.TIER1}}

    ctrl = [(i % 10) / 10 for i in range(n)]
    wide = [c + 0.1 for c in ctrl]
    check(abs(100 * (decide.mean_acc(rec(wide), decide.TIER1, "w4") - decide.mean_acc(rec(ctrl), decide.TIER1, "w4"))
              - 10) < 1e-9, "D is not 10")
    lo, hi = decide.paired_ci(rec(wide), rec(ctrl), decide.TIER1, "w4", 2000)
    check(abs(lo - 10) < 1e-9 and abs(hi - 10) < 1e-9, f"paired interval [{lo}, {hi}], expected [10, 10]")
    shuf = ctrl[:]
    random.Random(1).shuffle(shuf)
    lo2, hi2 = decide.paired_ci(rec(wide), rec(shuf), decide.TIER1, "w4", 2000)
    check(lo2 < 10 < hi2 and hi2 - lo2 > 5, f"red arm: unpaired scores gave the interval [{lo2}, {hi2}]")


# ------------------------------------------------------------------ distributed tests (torchrun, gloo)
COMMON = ["--arm", "wide", "--model", "{a}/model", "--lm_replay", "{a}/replay.pt", "--steps", "6",
          "--task_counts", "parity:2,swap:3,codeswap:3", "--train_len", "8", "--replay_batch", "3", "--replay_len", "64",
          "--replay_micro", "2", "--micro_tokens", "40", "--eval_every", "3", "--save_every", "3", "--val_n", "4",
          "--val_len", "70", "--test_n", "4", "--test_len", "300", "--eval_batch", "2", "--lm_eval_chunks", "2",
          "--nll_n", "3", "--warmup", "2", "--lr", "1e-3", "--w_lr", "1e-3", "--dtype", "fp32", "--rank", "4"]


def torchrun(nproc, out, extra=(), seed=0, common=True, timeout=900):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(A / "nofla"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               TMPDIR=str(FX / "tmp"))
    args = ([x.format(a=A) for x in COMMON] + ["--seed", str(seed)]) if common else []
    cmd = [str(PYBIN / "torchrun"), "--standalone", "--nproc_per_node", str(nproc), str(HERE / "train_t1.py"),
           "--out", str(out), *args, *extra]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    (out / "test_stdout.log").write_text(r.stdout + "\n" + r.stderr)
    check(r.returncode == 0, f"train_t1 x{nproc} exit {r.returncode}: {r.stderr[-3000:]}")
    return r


def log_of(out):
    recs = []
    for line in open(Path(out) / "log.jsonl"):
        with contextlib.suppress(ValueError):
            recs.append(json.loads(line))
    return recs


def events(out, kind):
    return {r["step"]: r for r in log_of(out) if r.get("event") == kind}


def trainable_diff(a, b):
    import torch
    x, y = torch.load(a), torch.load(b)
    check(set(x) == set(y), "trainable key sets differ")
    return max(float((x[k] - y[k]).abs().max()) for k in x) / max(float(y[k].abs().max()) for k in y)


def test_record(out, step):
    return json.loads((Path(out) / f"test_step{step:06d}.json").read_text())


def same_scores(ra, rb, what):
    for k in rb["tasks"]:
        check(ra["tasks"][k] == rb["tasks"][k], f"{what}: {k} differs")


def same_val(ea, eb, what, nll_tol=1e-5):
    for s in eb:
        check(s in ea, f"{what}: val step {s} missing")
        for t in tasks.ALL_TASKS:
            for k in ("w1", "w4", "first8"):
                check(ea[s][t][k] == eb[s][t][k], f"{what}: step {s} {t} {k} {ea[s][t][k]} vs {eb[s][t][k]}")
            check(rel(ea[s][t]["nll"], eb[s][t]["nll"]) < nll_tol, f"{what}: step {s} {t} nll")
        for k in ("replay_nll", "gate_rate_tasks", "gate_rate_text"):
            if ea[s].get(k) is not None or eb[s].get(k) is not None:
                check(rel(ea[s][k], eb[s][k]) < 1e-5, f"{what}: step {s} {k} {ea[s][k]} vs {eb[s][k]}")
        # KL of near-equal fp32 distributions over 248k tokens: measured 1 vs 2 ranks at step 3, 5.674e-5 vs 5.678e-5
        # (4.4e-8 apart, from the all-reduce order); another seed or a zero LR moves it by orders of magnitude more
        ka, kb = ea[s]["kl_to_untouched"], eb[s]["kl_to_untouched"]
        check(abs(ka - kb) <= max(1e-7, 2e-3 * max(abs(ka), abs(kb))), f"{what}: step {s} kl {ka} vs {kb}")


DIST = {}


def run_dir(name):
    return FX / "runs" / name


def test_dist_world_size_equality():
    """1 and 2 ranks agree: the step-1 gradient norms, step losses, every val eval, the test records (step 0 and the
    final step) and the final adapter, up to float summation order. The update goes downhill on the held-out val label NLL and a zero LR leaves
    it exactly in place. Red arm: seed 1 on 2 ranks ends far from seed 0."""
    a, b, c, z = run_dir("w1"), run_dir("w2"), run_dir("w2_seed1"), run_dir("w1_lr0")
    torchrun(1, a)
    torchrun(2, b)
    torchrun(2, c, seed=1)
    DIST["w1"], DIST["w2"] = a, b
    sa, sb = events(a, "step"), events(b, "step")
    check(set(sa) == set(sb) == {1, 6}, f"logged steps {sorted(sa)} {sorted(sb)}")
    for s in sa:
        for k, v in sa[s]["loss"].items():
            check(rel(v, sb[s]["loss"][k]) < 1e-4, f"step {s} loss {k}: {v} vs {sb[s]['loss'][k]}")
    va, vb = events(a, "eval"), events(b, "eval")
    check(set(va) == {0, 3, 6}, f"val steps {sorted(va)}")
    same_val(va, vb, "1 vs 2 ranks")
    check(set(events(a, "test")) == {0, 6} and (a / "test_step000000.json").exists(), "test evals at 0 and 6")
    same_scores(test_record(a, 0), test_record(b, 0), "test step 0, 1 vs 2 ranks")
    same_scores(test_record(a, 6), test_record(b, 6), "test step 6, 1 vs 2 ranks")
    check(rel(sum(test_record(a, 6)["nll"]), sum(test_record(b, 6)["nll"])) < 1e-5, "test nll 1 vs 2 ranks")
    for k in ("gnorm_lora", "gnorm_w"):  # the step-1 gradient, before Adam has amplified anything
        check(rel(sa[1][k], sb[1][k]) < 1e-6, f"step-1 {k}: {sa[1][k]} vs {sb[1][k]}")
    # Measured 2.3e-4 after 6 steps, all of it in a few gate entries whose gradient is near zero: Adam scales an update by
    # 1/sqrt(v), so a summation-order difference there becomes a visible step. Seed 1 is 1.9 away (red arm below).
    d = trainable_diff(a / "trainable_000006.pt", b / "trainable_000006.pt")
    check(d < 1e-3, f"1 vs 2 ranks: final adapter max rel diff {d:.2e}")
    check(not [r for r in log_of(b) if r.get("event") == "replica_mismatch"], "replicas diverged")
    check(json.loads((a / "complete.json").read_text())["steps"] == 6, "complete.json")
    cfg = [r for r in log_of(a) if r.get("event") == "config"][0]
    check(cfg["n_gates"] == 3 and cfg["task_tokens"] == {"parity": 12, "swap": 14, "codeswap": 60}, f"config {cfg}")
    mean_nll = lambda e: sum(e[t]["nll"] for t in tasks.TIER1)  # noqa: E731
    check(mean_nll(va[6]) < mean_nll(va[0]) - 1e-3, f"tier-1 val NLL did not fall: {mean_nll(va[0])} -> {mean_nll(va[6])}")
    torchrun(1, z, ["--lr", "0", "--w_lr", "0"])
    ez = events(z, "eval")
    check(all(ez[6][t]["nll"] == ez[0][t]["nll"] for t in tasks.ALL_TASKS),
          "red arm: a zero learning rate still moved the val NLL, so the fall above is not the update's doing")
    d_red = trainable_diff(c / "trainable_000006.pt", b / "trainable_000006.pt")
    check(d_red > 1e-2, f"red arm: seed 1 matched seed 0 within {d_red:.2e}")


def test_dist_resume_equality_and_eval_rerun():
    """Stop at step 3 and resume to 6 on 2 ranks: bit-identical to the unbroken 2-rank run; a 1-rank resume of the 2-rank checkpoint equals the unbroken 1-rank run; a val eval missing from the log at
    the resume step is re-run. Red arm: resuming with another seed (another task stream after the checkpoint) diverges."""
    b1, b2 = DIST.get("w1") or run_dir("w1"), DIST.get("w2") or run_dir("w2")
    if not (b2 / "trainable_000006.pt").exists():
        torchrun(1, b1)
        torchrun(2, b2)
    r = run_dir("res2")
    torchrun(2, r, ["--max_steps_debug", "3"])
    check(json.loads((r / "complete.json").read_text())["steps"] == 3, "the debug stop did not record step 3")
    lines = [line for line in open(r / "log.jsonl") if not ('"event": "eval"' in line and '"step": 3,' in line)]
    (r / "log.jsonl").write_text("".join(lines))  # the eval of the checkpoint step "never ran"
    x, red = run_dir("res1"), run_dir("res_red")
    for d in (x, red):
        d.mkdir(parents=True, exist_ok=True)
        for f in ("ckpt_000003.pt", "test_step000000.json"):
            shutil.copy(r / f, d / f)
        (d / "log.jsonl").write_text("".join(lines))
    torchrun(2, r, ["--resume"])
    check(3 in events(r, "eval"), "the missing step-3 eval was not re-run")
    d = trainable_diff(r / "trainable_000006.pt", b2 / "trainable_000006.pt")
    check(d == 0.0, f"2-rank resume vs unbroken 2-rank run: {d:.2e} (same world, same order: expected bit-identical)")
    same_scores(test_record(r, 6), test_record(b2, 6), "resumed test step 6")
    torchrun(1, x, ["--resume"])
    d1 = trainable_diff(x / "trainable_000006.pt", b1 / "trainable_000006.pt")
    check(d1 < 1e-3, f"1-rank resume of a 2-rank checkpoint vs unbroken 1-rank run: {d1:.2e} (the 1 vs 2 rank bound)")
    torchrun(2, red, ["--resume"], seed=1)
    dr = trainable_diff(red / "trainable_000006.pt", b2 / "trainable_000006.pt")
    check(dr > 1e-3, f"red arm: a resume on another seed's stream matched within {dr:.2e}")


def test_dist_pairing_eval_only_and_decide():
    """ctrl and wide at step 0 give identical test records (G0 through the trainer); --eval_only none reproduces the
    step-0 record (the untouched model) and --eval_only trainable_000006.pt reproduces the step-6 record; decide.py on
    the pair finds no config difference and the step-0 identity. Red arm: eval_only of the trained adapter is not the
    untouched record."""
    w = DIST.get("w1") or run_dir("w1")
    if not (w / "trainable_000006.pt").exists():
        torchrun(1, w)
    root = FX / "runs" / "pairtag"
    shutil.rmtree(root, ignore_errors=True)
    (root / "wide_s0").mkdir(parents=True)
    for f in ("test_step000000.json", "test_step000006.json", "log.jsonl"):
        shutil.copy(w / f, root / "wide_s0" / f)
    torchrun(1, root / "ctrl_s0", ["--arm", "ctrl"])
    same_scores(test_record(root / "wide_s0", 0), test_record(root / "ctrl_s0", 0), "wide vs ctrl at step 0")
    eo = run_dir("eval_only")
    torchrun(1, eo, ["--eval_only", "none"])
    torchrun(1, eo, ["--eval_only", str(w / "trainable_000006.pt")])
    untouched = json.loads((eo / "test_none.json").read_text())
    trained = json.loads((eo / "test_trainable_000006.json").read_text())
    same_scores(untouched, test_record(w, 0), "eval_only none vs the step-0 test")
    same_scores(trained, test_record(w, 6), "eval_only trainable_000006 vs the step-6 test")
    check(untouched["tasks"] != trained["tasks"], "red arm: the trained adapter scored exactly like the untouched model")
    rc, out, err = _decide(root, 6, 0)
    check(rc == 0, f"decide.py failed: {err[-1500:]}")
    rec = json.loads((root / "decision_step000006.json").read_text())
    check(rec["config_differences"] == {}, f"config differences {rec['config_differences']}")
    check(rec["step0_identity"]["identical"], "step-0 identity not seen by decide.py")
    check(out["verdict"] in ("EXTEND", "KILL", "GO", "REVIEW"), f"verdict {out}")


def test_dist_step1_grad_matches_train_py():
    """The trainer's OWN backward (main(): loss = tot / n_labels and tot * w_lm / n_lm per micro-batch, the flat
    all-reduce, the grad restore) gives train.py's step-1 gradient: the pre-clip LoRA and gate gradient norms logged at
    step 1 by a 1-rank run equal the norms of train.py's task_loss + lm_loss gradient on the same init and batch.
    test_loss_and_grad_match_train_py checks the formula on a copy in the test; this checks the copy in main(). Red arm:
    the equal-weight-per-micro-batch slip moves the norms far outside the tolerance."""
    import torch
    w1 = DIST.get("w1") or run_dir("w1")
    if not (w1 / "trainable_000006.pt").exists():
        torchrun(1, w1)
    s1 = events(w1, "step")[1]
    cfg = [r for r in log_of(w1) if r.get("event") == "config"][0]["args"]
    V = vocab()
    model = build("wide", seed=cfg["seed"])  # main(): torch.manual_seed(seed), then train27.build_model
    counts = T1.parse_counts(cfg["task_counts"])
    groups = T1.Stream(V, cfg["seed"], counts, cfg["train_len"]).batch(0)
    train = torch.load(A / "replay.pt")["train"][:, : cfg["replay_len"]]
    chunks = T1.step_chunks(train, T1.replay_perm(train.shape[0], cfg["seed"]), 0, cfg["replay_batch"])
    exs = [e for _, g in groups for e in g]

    def norms(equal_mb=False):
        model.zero_grad(set_to_none=True)
        with cuda_is_cpu():
            mbs = [g[i:i + max(1, cfg["micro_tokens"] // len(g[0].ids))] for _, g in groups
                   for i in range(0, len(g), max(1, cfg["micro_tokens"] // len(g[0].ids)))]
            for sub in mbs:  # train.py: (l_mb * len(g) / len(exs)).backward() per single-task micro-batch
                l_mb, _ = TR.task_loss(model, sub, 0)
                (l_mb * (1 / len(mbs) if equal_mb else len(sub) / len(exs))).backward()
            (cfg["w_lm"] * TR.lm_loss(model, torch.tensor(chunks).long())).backward()
        g_l = sum(float(p.grad.double().pow(2).sum()) for n, p in model.named_parameters()
                  if p.requires_grad and "negeig_w" not in n and p.grad is not None) ** 0.5
        g_w = sum(float(p.grad.double().pow(2).sum()) for n, p in model.named_parameters()
                  if "negeig_w" in n and p.grad is not None) ** 0.5
        return g_l, g_w

    g_l, g_w = norms()
    check(g_l > 0 and g_w > 0, f"reference gradient is zero: {g_l} {g_w}")
    check(rel(s1["gnorm_lora"], g_l) < 1e-4 and rel(s1["gnorm_w"], g_w) < 1e-4,
          f"step-1 gradient norms: trainer {s1['gnorm_lora']}, {s1['gnorm_w']} vs train.py {g_l}, {g_w}")
    b_l, b_w = norms(equal_mb=True)
    check(max(rel(s1["gnorm_lora"], b_l), rel(s1["gnorm_w"], b_w)) > 1e-3,
          f"red arm: equal weight per micro-batch matched the trainer: {b_l}, {b_w}")


def test_dist_replay_len_guard():
    """--replay_len wider than the replay file's chunks is refused (the lm loss would be divided by replay_len - 1
    targets a chunk that do not exist); the 64-token slices every other test trains on are the green arm."""
    out = run_dir("replay_len_guard")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(A / "nofla"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               TMPDIR=str(FX / "tmp"))
    args = [x.format(a=A) for x in COMMON] + ["--seed", "0", "--replay_len", "600"]
    r = subprocess.run([str(PYBIN / "torchrun"), "--standalone", "--nproc_per_node", "1", str(HERE / "train_t1.py"),
                        "--out", str(out), *args], env=env, capture_output=True, text=True, timeout=600)
    check(r.returncode != 0 and "--replay_len 600 but the chunks" in r.stderr,
          f"replay_len 600 on 512-token chunks was not refused: rc {r.returncode} {r.stderr[-800:]}")
    check(not list(out.glob("ckpt_*.pt")), "a refused run trained")


def test_box_script_end_to_end():
    """run_t1.sh on CPU through two rungs (3 and 6) with the real trainer: mem probe -> grad_ckpt decision, both arms,
    decide.py at each rung, PROBE_DONE, and a second invocation that launches nothing. The wide arm's first launch is
    made to crash after 2 steps (a torchrun shim), so the retry path and the resume run; the final wide adapter equals
    the unbroken 1-rank run of test_dist_world_size_equality (same flags). Red arms: an unknown GRAD_CKPT and a passed
    --steps are refused before anything starts."""
    w1 = DIST.get("w1") or run_dir("w1")
    if not (w1 / "trainable_000006.pt").exists():
        torchrun(1, w1)
    W = FX / "box"
    shutil.rmtree(W, ignore_errors=True)
    (W / "data" / "t1").mkdir(parents=True)
    os.symlink(A / "model", W / "model")
    os.symlink(A / "replay.pt", W / "data" / "t1" / "replay_wikitext103.pt")
    shim = FX / "shimbin"
    shim.mkdir(exist_ok=True)
    (shim / "python").write_text(f'#!/usr/bin/env bash\nexec "{PYBIN}/python" "$@"\n')
    # The shim records the GPU list run_t1.sh assigned, then BLANKS it: on this rig CUDA_VISIBLE_DEVICES=0 is the owner's
    # GPU, and the trainer would take it. First wide training launch: stop after 2 steps (the debug stop saves its last
    # step, ckpt_000002), then exit 1.
    (shim / "torchrun").write_text(f"""#!/usr/bin/env bash
echo "gpus=${{CUDA_VISIBLE_DEVICES-unset}} $*" >>"{shim}/launches.log"
export CUDA_VISIBLE_DEVICES=
if [[ " $* " == *" --arm wide "* ]] && [[ " $* " != *" --mem_probe "* ]] && [ ! -f "{shim}/crashed" ]; then
  touch "{shim}/crashed"
  "{PYBIN}/torchrun" "$@" --max_steps_debug 2 || exit 3
  exit 1
fi
exec "{PYBIN}/torchrun" "$@"
""")
    for f in ("python", "torchrun"):
        (shim / f).chmod(0o755)
    for f in ("crashed", "launches.log"):
        with contextlib.suppress(FileNotFoundError):
            (shim / f).unlink()
    flags = [x.format(a=A) for x in COMMON]
    drop = {"--arm", "--model", "--lm_replay", "--steps"}
    extra, i = [], 0
    while i < len(flags):
        if flags[i] in drop:
            i += 2
            continue
        extra.append(flags[i])
        i += 1
    env = dict(os.environ, NEGEIG_W=str(W), PYBIN=str(shim), EXPECT_GPUS="2", GPUS_PER_RUN="1", RUNGS="3 6",
               ARM_BACKOFF_S="1", PYTHONPATH=str(A / "nofla"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               TMPDIR=str(FX / "tmp"), PORT_BASE="29611")
    sh = ["bash", str(HERE / "run_t1.sh")]
    bad = subprocess.run(sh + ["tb", "0", *extra], env=dict(env, GRAD_CKPT="2"), capture_output=True, text=True, timeout=120)
    check(bad.returncode == 2 and "GRAD_CKPT" in bad.stderr, f"red arm: GRAD_CKPT=2 accepted: {bad.stderr[-400:]}")
    bad = subprocess.run(sh + ["tb", "0", "--steps", "9"], env=env, capture_output=True, text=True, timeout=120)
    check(bad.returncode == 2 and "--steps" in bad.stderr, "red arm: --steps passed through")
    for flag in ("--eval_only", "--max_steps_debug", "--mem_probe"):  # each would end every launch before the rung
        bad = subprocess.run(sh + ["tb", "0", flag, "1"], env=env, capture_output=True, text=True, timeout=120)
        check(bad.returncode == 2 and flag in bad.stderr, f"red arm: {flag} passed through")
    check(not (W / "runs" / "tb").exists(), "a refused invocation created the run directory")
    r = subprocess.run(sh + ["tb", "0", *extra], env=env, capture_output=True, text=True, timeout=2400)
    (FX / "box_run.log").write_text(r.stdout + "\n" + r.stderr)
    check(r.returncode == 0, f"run_t1.sh exit {r.returncode}: {(r.stdout + r.stderr)[-3000:]}")
    out = W / "runs" / "tb"
    check((out / "grad_ckpt.env").read_text().strip() == "GRAD_CKPT=0", "mem probe on CPU (peak 0) should choose 0")
    for s in (3, 6):
        check((out / f"decision_step{s:06d}.json").exists(), f"no decision at rung {s}")
    d3 = json.loads((out / "decision_step000003.json").read_text())
    done = (out / "PROBE_DONE").read_text()
    check(d3["verdict"] in ("EXTEND", "KILL", "GO", "REVIEW"), f"rung 3 verdict {d3['verdict']}")
    if d3["verdict"] == "EXTEND":
        check(done.startswith(("KILL at step 6", "GO at step 6", "REVIEW at step 6")), f"PROBE_DONE {done}")
    launch = (out / "run_t1.log").read_text()
    check("wide attempt 1 exited 1" in launch and "wide launch 2" in launch, "the crash and retry are not in the log")
    shim_log = (shim / "launches.log").read_text().splitlines()
    gpus = {("probe" if "--mem_probe" in line else "wide" if "--arm wide" in line else "ctrl"): line.split()[0]
            for line in shim_log}
    check(gpus == {"probe": "gpus=0", "wide": "gpus=0", "ctrl": "gpus=1"}, f"GPU assignment {gpus}")
    check(launch.count(" launch ") == 5, f"expected 5 launches (wide 3, ctrl 2): {launch.count(' launch ')}")
    for arm in ("wide", "ctrl"):
        check(json.loads((out / f"{arm}_s0" / "complete.json").read_text())["steps"] == 6, f"{arm} not at step 6")
    d = trainable_diff(out / "wide_s0" / "trainable_000006.pt", w1 / "trainable_000006.pt")
    check(d < 1e-5, f"box wide (crash at 2, rung 3, rung 6, grad_ckpt 0) vs the unbroken run: {d:.2e}")
    n_launch = launch.count(" launch ")
    r2 = subprocess.run(sh + ["tb", "0", *extra], env=env, capture_output=True, text=True, timeout=300)
    check(r2.returncode == 0 and "already decided" in r2.stdout, f"second invocation: {r2.stdout[-500:]}")
    check((out / "run_t1.log").read_text().count(" launch ") == n_launch, "the second invocation launched a run")
    # a grad_ckpt.env cut by a preemption (no GRAD_CKPT=0|1 line) is refused before any launch, not passed on as
    # --grad_ckpt auto to every attempt of both arms
    cut = W / "runs" / "tcut"
    cut.mkdir(parents=True)
    (cut / "grad_ckpt.env").write_text("GRAD_CK")
    n_shim = len((shim / "launches.log").read_text().splitlines())
    r3 = subprocess.run(sh + ["tcut", "0", *extra], env=env, capture_output=True, text=True, timeout=120)
    check(r3.returncode == 2 and "grad_ckpt.env" in r3.stderr, f"red arm: a cut grad_ckpt.env was used: {r3.stderr[-400:]}")
    check(len((shim / "launches.log").read_text().splitlines()) == n_shim, "a launch happened with a cut grad_ckpt.env")


UNIT = [test_stream_and_replay_match_train_py, test_plan_world_invariant_and_lpt_cover, test_loss_and_grad_match_train_py,
        test_eval_matches_train_py, test_eval_oracle_and_windows, test_untouched_identity_and_gate_rate,
        test_decide_rule_synthetic, test_decide_paired_bootstrap]
DISTRIBUTED = [test_dist_world_size_equality, test_dist_resume_equality_and_eval_rerun,
               test_dist_pairing_eval_only_and_decide, test_dist_step1_grad_matches_train_py, test_dist_replay_len_guard,
               test_box_script_end_to_end]


def main():
    global A, FX, T1, TR, tasks
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True, help="the train27 smoke assets dir (model/, nofla/, replay.pt)")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-dist", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    A = Path(a.assets)
    for p in ("model", "nofla", "replay.pt"):
        check((A / p).exists(), f"missing asset {A / p}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    FX = Path(tempfile.mkdtemp(prefix="t1_", dir=os.environ.get("T1_SCRATCH") or None))
    (FX / "tmp").mkdir()
    os.environ["TMPDIR"] = tempfile.tempdir = str(FX / "tmp")
    sys.path.insert(0, str(A / "nofla"))
    print(f"fixture {FX}", flush=True)
    try:
        sys.path.insert(0, str(HERE))
        import train_t1
        T1 = train_t1
        tasks = train_t1.tasks
        sys.path.insert(0, str(RETRO))
        import train as tr_mod
        TR = tr_mod
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
