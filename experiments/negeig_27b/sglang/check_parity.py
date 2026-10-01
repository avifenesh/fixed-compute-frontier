"""GPU-box parity gates for the negeig SGLang patch (Qwen3.8-27B, SGLang 0.5.20).

Runs on the box, against live servers, through the public /generate endpoint. Only `collect` needs a server; the rest
is CPU work on JSON files. The gates, in the order the README runs them:

    G0  the stock server repeats itself bit for bit (two collect passes). If not, G1 cannot say anything: INCONCLUSIVE.
    G1  NEGEIG_GATES=zeros server vs the unpatched server: every logprob, every top-k entry, every greedy token and
        every decode logprob bit-identical on 32 prompts. Needs the patch-engaged evidence from the server log.
    G2  random W: SGLang(W) vs the HF patch (patch.py, transformers 5.17.0) inside the gap that stock SGLang already
        has to stock HF, and the effect of W (random vs stock) equal in both implementations. Vacuous if W does not
        move the logprobs far above that gap: INCONCLUSIVE, not PASS.
    G3  decode equals prefill: the logprob a decode step gives a token equals the logprob a prefill of the same
        sequence gives it, inside the gap the stock server has between its own decode and prefill.

Subcommands:
    prepare            write the 32 prompts (token ids) used by every other step
    make-random-gates  write a random fp32 gate file (safetensors) of the right keys and shape
    collect            query a running server, write one record file (prefill logprobs, top-k, decode, replay)
    hf-ref             the same record from the HF model with patch.py (stock, zeros, or a gate file)
    compare            apply G0..G3 to record files, write a report, exit 0 only when nothing failed
    selftest           CPU: mock servers with red arms for the compare logic, optional real hf-ref on a tiny model

Exit codes of compare: 0 all requested gates PASS, 1 a gate FAILED, 2 a gate is INCONCLUSIVE (and none failed).
Logprobs are the server's own float32 numbers read from JSON, so equality means equality of the floats.
"""

import argparse
import hashlib
import http.server
import inspect
import json
import math
import os
import random
import re
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

FORMAT = "negeig-parity-v1"
HERE = Path(__file__).resolve().parent
ACTIVE_RE = re.compile(r"negeig: active v(\d+): (\d+) gated GDN layers on this rank, gates from (.*?), max\|W\|=(\S+)")
ANY_NEGEIG_RE = re.compile(r"negeig:")

# text prompts (English, Hebrew, code, digits): tokenized when a tokenizer is given; ids are random otherwise
TEXTS = [
    "The delta rule updates a fast weight matrix by the difference between the value and what the matrix predicts.",
    "Gated DeltaNet keeps a recurrent state per head and reads it with the query at every token.",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "הערכת המודל נעשית על סמך שאלות אמיתיות ולא על דוגמאות מנוסות.",
    "3.14159 26535 89793 23846 26433 83279 50288 41971 69399 37510 58209 74944 59230 78164 06286",
    "Q: What is the capital of France?\nA: Paris.\nQ: What is the capital of Japan?\nA:",
    "SELECT name, count(*) FROM orders JOIN users ON users.id = orders.user_id GROUP BY name ORDER BY 2 DESC;",
    "Once upon a time, in a small village by the sea, a fisherman found a bottle with a map inside.",
]
EDGE_LENGTHS = [1, 2, 7, 63, 64, 65, 127, 128, 129, 191, 256, 257, 511, 513, 700, 1025]


class ParityError(SystemExit):
    def __init__(self, msg: str):
        super().__init__("check_parity: " + msg)


def jdump(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def jload(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def sha_of(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------------------------------------------
def prepare(out: str, n: int, seed: int, vocab: int, tokenizer: Optional[str]) -> dict:
    rng = random.Random(seed)
    prompts: List[dict] = []
    if tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer)
        for i, text in enumerate(TEXTS):
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if ids:
                prompts.append({"id": f"text{i:02d}", "kind": "text", "input_ids": [int(x) for x in ids]})
    lo, hi = 1000, max(1001, min(vocab, 150000))  # stay inside ordinary text ids: no specials, no vision tokens
    k = 0
    while len(prompts) < n:
        length = EDGE_LENGTHS[k % len(EDGE_LENGTHS)] if k < len(EDGE_LENGTHS) else rng.randint(3, 600)
        prompts.append({"id": f"rand{k:02d}", "kind": "random",
                        "input_ids": [rng.randrange(lo, hi) for _ in range(length)]})
        k += 1
    prompts = prompts[:n]
    doc = {"format": FORMAT, "seed": seed, "vocab": vocab, "prompts": prompts}
    doc["sha256"] = sha_of(prompts)
    jdump(doc, out)
    return {"out": out, "n": len(prompts), "sha256": doc["sha256"], "lengths": [len(p["input_ids"]) for p in prompts]}


def load_prompts(path: str) -> dict:
    doc = jload(path)
    if doc.get("format") != FORMAT or sha_of(doc["prompts"]) != doc.get("sha256"):
        raise ParityError(f"{path}: not a prompt file of this tool, or edited after prepare")
    return doc


# ----------------------------------------------------------------------------------------------------------
# make-random-gates
# ----------------------------------------------------------------------------------------------------------
def make_random_gates(config: str, out: str, target_t: float, seed: int) -> dict:
    """W ~ N(0, (target_t / sqrt(hidden))^2): for a unit-rms x the pre-tanh value has std target_t, so beta moves a lot."""
    import torch
    from safetensors.torch import save_file

    cfg = jload(config)
    cfg = cfg.get("text_config", cfg)
    layers = [i for i, t in enumerate(cfg["layer_types"]) if t == "linear_attention"]
    nv, hidden = cfg["linear_num_value_heads"], cfg["hidden_size"]
    g = torch.Generator().manual_seed(seed)
    std = target_t / math.sqrt(hidden)
    tensors = {f"model.language_model.layers.{i}.linear_attn.negeig_w":
               (torch.randn(nv, hidden, generator=g, dtype=torch.float32) * std).contiguous() for i in layers}
    save_file(tensors, out, metadata={"format": "negeig-gates-v1", "note": "random, for check_parity G2/G3 only",
                                      "layers": ",".join(map(str, layers)), "std": repr(std)})
    return {"out": out, "layers": len(layers), "shape": [nv, hidden], "std": std, "target_t": target_t}


# ----------------------------------------------------------------------------------------------------------
# collect (server side)
# ----------------------------------------------------------------------------------------------------------
def post_json(url: str, payload: dict, timeout: float) -> Any:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ParityError(f"{url}: HTTP {e.code}: {e.read()[:400]!r}") from None


def _lp_list(entries: Optional[Sequence]) -> List[Optional[float]]:
    """(logprob, token_id, text) tuples (lists after JSON) to floats. SGLang 0.5.20 puts (None, first_id, None) at
    position 0 of the input logprobs (scheduler_components/logprob_result_processor.py), so the float is None there."""
    return [None if e is None or e[0] is None else float(e[0]) for e in (entries or [])]


def _top_list(tops: Optional[Sequence]) -> List[Optional[List[List[float]]]]:
    out: List[Optional[List[List[float]]]] = []
    for pos in tops or []:
        out.append(None if not pos else [[float(e[0]), int(e[1])] for e in pos])
    return out


def generate(base: str, ids: List[int], new: int, topk: int, timeout: float) -> dict:
    body = {"input_ids": ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": new, "ignore_eos": True},
            "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": topk}
    res = post_json(base.rstrip("/") + "/generate", body, timeout)
    if isinstance(res, list):
        res = res[0]
    return res["meta_info"]


def parse_log(path: Optional[str]) -> dict:
    if not path:
        return {"log": None}
    text = Path(path).read_text(errors="replace")
    active = ACTIVE_RE.findall(text)
    return {"log": os.path.basename(path), "negeig_lines": len(ANY_NEGEIG_RE.findall(text)),
            "active_lines": len(active), "active": [list(a) for a in active[:8]],
            "max_w": sorted({a[3] for a in active})}


def collect(url: str, prompts_path: str, out: str, tag: str, topk: int, decode_tokens: int, server_log: Optional[str],
            timeout: float) -> dict:
    doc = load_prompts(prompts_path)
    recs = []
    t0 = time.time()
    for p in doc["prompts"]:
        ids = p["input_ids"]
        mi = generate(url, ids, 1, topk, timeout)  # prefill only
        pre_lp = _lp_list(mi.get("input_token_logprobs"))
        if len(pre_lp) != len(ids):
            raise ParityError(f"{p['id']}: got {len(pre_lp)} input logprobs for {len(ids)} tokens "
                              "(start the server with --disable-radix-cache and no chunked-prefill logprob limits)")
        rec = {"id": p["id"], "n": len(ids),
               "prefill": {"lp": pre_lp, "top": _top_list(mi.get("input_top_logprobs"))}}
        if decode_tokens > 1:
            md = generate(url, ids, decode_tokens, topk, timeout)
            olp = md.get("output_token_logprobs") or []
            toks = [int(e[1]) for e in olp]
            rec["decode"] = {"tokens": toks, "lp": _lp_list(olp), "top": _top_list(md.get("output_top_logprobs"))}
            mr = generate(url, ids + toks, 1, topk, timeout)  # the same tokens through prefill
            rp = _lp_list(mr.get("input_token_logprobs"))
            if len(rp) != len(ids) + len(toks):
                raise ParityError(f"{p['id']}: replay returned {len(rp)} logprobs for {len(ids) + len(toks)} tokens")
            rec["replay"] = {"lp": rp[len(ids):]}
        recs.append(rec)
        print(f"collect {tag}: {p['id']} n={len(ids)} done", file=sys.stderr, flush=True)
    meta = {"url": url, "topk": topk, "decode_tokens": decode_tokens, "seconds": round(time.time() - t0, 1),
            **parse_log(server_log)}
    rec_doc = {"format": FORMAT, "kind": "sglang", "tag": tag, "prompts_sha256": doc["sha256"], "meta": meta,
               "records": recs}
    jdump(rec_doc, out)
    return {"out": out, "tag": tag, "records": len(recs), "meta": meta}


# ----------------------------------------------------------------------------------------------------------
# hf-ref (the HF model with patch.py)
# ----------------------------------------------------------------------------------------------------------
def load_patch(path: Optional[str]):
    import importlib.util

    p = Path(path) if path else HERE.parent.parent / "negeig_retrofit" / "patch.py"
    if not p.is_file():
        raise ParityError(f"HF patch {p} not found")
    spec = importlib.util.spec_from_file_location("hf_negeig_patch", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hf_ref(model_dir: str, prompts_path: str, out: str, tag: str, mode: str, topk: int, device: str, dtype: str,
           patch_path: Optional[str], torch_gdn: bool) -> dict:
    """mode: stock (library forward, no patch), zeros (patch installed, W = 0), or a gate safetensors path."""
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    patch = load_patch(patch_path)
    if torch_gdn:  # CPU only: the library's reference delta rule instead of the fla Triton kernels
        M.torch_chunk_gated_delta_rule = inspect.unwrap(M.torch_chunk_gated_delta_rule)
    doc = load_prompts(prompts_path)
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    model, info = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt, device_map={"": device},
                                                       output_loading_info=True)
    if info["missing_keys"]:
        raise ParityError(f"weights not loaded: {sorted(info['missing_keys'])[:5]}")
    model.eval()
    gate_sha = None
    if mode != "stock":
        params = patch.install_for_arm(model, "wide")
        if mode != "zeros":
            from safetensors import safe_open

            gates: Dict[int, Any] = {}
            with safe_open(mode, framework="pt", device="cpu") as f:
                for k in f.keys():  # noqa: SIM118
                    m = re.search(r"layers\.(\d+)\.linear_attn\.negeig_w(?:\.weight)?$", k)
                    if not m:
                        raise ParityError(f"{mode}: unexpected key {k}")
                    gates[int(m.group(1))] = f.get_tensor(k)
            layers = patch.gdn_layers(model)
            missing = sorted(set(l.layer_idx for l in layers) - set(gates))
            if missing:
                raise ParityError(f"{mode}: no gate for GDN layers {missing}")
            with torch.no_grad():
                for l in layers:
                    l.negeig_w.weight.copy_(gates[l.layer_idx].to(torch.float32))
            gate_sha = hashlib.sha256(Path(mode).read_bytes()).hexdigest()
        if not params:
            raise ParityError("patch installed no parameters")
    recs = []
    for p in doc["prompts"]:
        ids = torch.tensor([p["input_ids"]], device=device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0]  # [L, V] in the model dtype, as SGLang's lm_head output
        L = logits.shape[0]
        lp: List[Optional[float]] = [None]
        top: List[Optional[List[List[float]]]] = [None]
        # position i predicts token i+1; SGLang reports, for input position i+1, the logprob of that token
        for s in range(0, L - 1, 64):
            e = min(L - 1, s + 64)
            lsm = torch.log_softmax(logits[s:e].float(), dim=-1)
            tgt = ids[0, s + 1:e + 1]
            lp += [float(v) for v in lsm.gather(1, tgt[:, None])[:, 0]]
            tv, ti = lsm.topk(topk, dim=-1)
            top += [[[float(v), int(i)] for v, i in zip(rv, ri)] for rv, ri in zip(tv, ti)]
        # entry j of both lists describes the distribution that predicted token j (logits at position j-1), entry 0 is None
        recs.append({"id": p["id"], "n": L, "prefill": {"lp": lp, "top": top}})
    meta = {"model": model_dir, "mode": "gates" if gate_sha else mode, "gate_sha256": gate_sha, "dtype": dtype,
            "device": device, "torch_gdn": torch_gdn, "topk": topk}
    jdump({"format": FORMAT, "kind": "hf", "tag": tag, "prompts_sha256": doc["sha256"], "meta": meta,
           "records": recs}, out)
    return {"out": out, "tag": tag, "records": len(recs), "meta": meta}


# ----------------------------------------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------------------------------------
def load_rec(path: str) -> dict:
    d = jload(path)
    if d.get("format") != FORMAT or "records" not in d:
        raise ParityError(f"{path}: not a record file of this tool")
    return d


def aligned(docs: Dict[str, dict]) -> None:
    shas = {k: d["prompts_sha256"] for k, d in docs.items()}
    if len(set(shas.values())) != 1:
        raise ParityError(f"record files come from different prompt files: {shas}")
    ids = {k: [r["id"] for r in d["records"]] for k, d in docs.items()}
    first = next(iter(ids.values()))
    for k, v in ids.items():
        if v != first:
            raise ParityError(f"{k}: prompt ids differ from the others")


def pct(vals: Sequence[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))]


def summarize(vals: Sequence[float]) -> dict:
    a = [abs(v) for v in vals]
    return {"n": len(a), "max": max(a) if a else float("nan"), "p99": pct(a, 0.99),
            "mean": (sum(a) / len(a)) if a else float("nan")}


def exact(a: dict, b: dict) -> dict:
    """Bit equality of every number two record files share. Returns counts and the first few mismatches."""
    n = 0
    bad: List[str] = []

    def eq(where: str, x: Any, y: Any) -> None:
        nonlocal n
        n += 1
        if x != y and len(bad) < 8:
            bad.append(f"{where}: {x!r} vs {y!r}")
        elif x != y:
            bad.append("")  # count only

    for ra, rb in zip(a["records"], b["records"]):
        pid = ra["id"]
        for sec in ("prefill", "decode", "replay"):
            sa, sb = ra.get(sec), rb.get(sec)
            if sa is None and sb is None:
                continue
            if (sa is None) != (sb is None):
                eq(f"{pid}.{sec}.present", sa is None, sb is None)
                continue
            for key in ("tokens", "lp", "top"):
                if key not in sa and key not in sb:
                    continue
                xa, xb = sa.get(key), sb.get(key)
                if xa is None or xb is None or len(xa) != len(xb):
                    eq(f"{pid}.{sec}.{key}.len", None if xa is None else len(xa), None if xb is None else len(xb))
                    continue
                for i, (u, v) in enumerate(zip(xa, xb)):
                    eq(f"{pid}.{sec}.{key}[{i}]", u, v)
    return {"compared": n, "mismatches": sum(1 for x in bad), "first": [x for x in bad if x][:8]}


def lp_pairs(a: dict, b: dict) -> List[Tuple[float, float]]:
    out = []
    for ra, rb in zip(a["records"], b["records"]):
        for x, y in zip(ra["prefill"]["lp"], rb["prefill"]["lp"]):
            if x is not None and y is not None:
                out.append((x, y))
    return out


def top_pairs(a: dict, b: dict) -> Tuple[List[Tuple[float, float]], Tuple[int, int]]:
    """Top-k logprobs of the ids both servers list at a position, and (top-1 agreements, positions)."""
    out: List[Tuple[float, float]] = []
    agree = tot = 0
    for ra, rb in zip(a["records"], b["records"]):
        for ta, tb in zip(ra["prefill"].get("top") or [], rb["prefill"].get("top") or []):
            if not ta or not tb:
                continue
            da, db = {i: v for v, i in ta}, {i: v for v, i in tb}
            for i in da.keys() & db.keys():
                out.append((da[i], db[i]))
            tot += 1
            agree += int(ta[0][1] == tb[0][1])
    return out, (agree, tot)


def gap(a: dict, b: dict) -> dict:
    lps = lp_pairs(a, b)
    tps, (ag, tot) = top_pairs(a, b)
    s = summarize([x - y for x, y in lps + tps])
    s["top1_agree"] = (ag / tot) if tot else float("nan")
    s["lp_n"], s["top_n"] = len(lps), len(tps)
    return s


def effect_vectors(w: dict, s: dict, w2: dict, s2: dict) -> Tuple[List[float], List[float]]:
    """(w - s) per prompt token, for the first pair and for the second pair, in the same order."""
    return [x - y for x, y in lp_pairs(w, s)], [x - y for x, y in lp_pairs(w2, s2)]


def rel_err_corr(e_a: Sequence[float], e_ref: Sequence[float]) -> Tuple[float, float]:
    den = math.sqrt(sum(v * v for v in e_ref)) or float("nan")
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(e_a, e_ref)))
    ma, mr = sum(e_a) / len(e_a), sum(e_ref) / len(e_ref)
    sa = math.sqrt(sum((x - ma) ** 2 for x in e_a))
    sr = math.sqrt(sum((y - mr) ** 2 for y in e_ref))
    cov = sum((x - ma) * (y - mr) for x, y in zip(e_a, e_ref))
    corr = cov / (sa * sr) if sa > 0 and sr > 0 else float("nan")
    return num / den, corr


def decode_gap(d: dict) -> dict:
    diffs: List[float] = []
    first: List[float] = []
    for r in d["records"]:
        dec, rep = r.get("decode"), r.get("replay")
        if not dec or not rep:
            continue
        for j, (x, y) in enumerate(zip(dec["lp"], rep["lp"])):
            if x is None or y is None:
                continue
            (diffs if j >= 1 else first).append(x - y)  # j = 0 came out of the prefill pass, not a decode step
    s = summarize(diffs)
    s["first_token_prefill_vs_replay"] = summarize(first)
    return s


def engaged(d: dict) -> Optional[bool]:
    m = d.get("meta", {})
    if m.get("log") is None:
        return None
    return m.get("active_lines", 0) >= 1


def verdict_line(name: str, v: str, why: str) -> dict:
    return {"gate": name, "verdict": v, "why": why}


def compare(a: argparse.Namespace) -> dict:
    files = {"stock": a.stock, "stock_repeat": a.stock_repeat, "zero": a.zero, "rand": a.rand,
             "hf_stock": a.hf_stock, "hf_rand": a.hf_rand}
    docs = {k: load_rec(v) for k, v in files.items() if v}
    if "stock" not in docs:
        raise ParityError("--stock is required")
    aligned(docs)
    skip = {s.strip().upper() for s in (a.skip or "").split(",") if s.strip()}
    rep: Dict[str, Any] = {"files": files, "params": {k: getattr(a, k) for k in (
        "gap_ratio", "gap_floor", "effect_min", "noise_factor", "effect_rel_tol", "corr_min", "top1_slack",
        "decode_ratio", "decode_floor", "min_n")}, "gates": []}
    g = rep["gates"]

    # G0
    g0 = None
    if "stock_repeat" in docs:
        g0 = exact(docs["stock"], docs["stock_repeat"])
        rep["g0"] = g0
        g.append(verdict_line("G0", "PASS" if g0["mismatches"] == 0 and g0["compared"] >= a.min_n else "FAIL",
                              f"stock vs stock again: {g0['compared']} numbers, {g0['mismatches']} differ"))
    else:
        g.append(verdict_line("G0", "SKIPPED", "no --stock-repeat file"))

    # G1
    if "G1" in skip:
        g.append(verdict_line("G1", "WAIVED", "listed in --skip"))
    elif "zero" not in docs:
        g.append(verdict_line("G1", "SKIPPED", "no --zero file"))
    else:
        g1 = exact(docs["stock"], docs["zero"])
        rep["g1"] = g1
        eng_zero, eng_stock = engaged(docs["zero"]), engaged(docs["stock"])
        stock_clean = docs["stock"].get("meta", {}).get("negeig_lines", 0) == 0 if eng_stock is not None else None
        rep["g1"]["engaged_zero"], rep["g1"]["stock_log_clean"] = eng_zero, stock_clean
        what = f"{g1['compared']} numbers compared, {g1['mismatches']} differ"
        if g1["compared"] < a.min_n:
            g.append(verdict_line("G1", "INCONCLUSIVE", f"only {g1['compared']} numbers compared; " + what))
        elif g1["mismatches"] == 0 and eng_zero is True and stock_clean is not False:
            g.append(verdict_line("G1", "PASS", what + "; patch engaged in the zeros server log"))
        elif g1["mismatches"] == 0:
            g.append(verdict_line("G1", "INCONCLUSIVE", what + "; patch engagement not evidenced (pass --server-log "
                                  "to collect, it must hold a 'negeig: active' line; the stock log must hold none)"))
        elif g0 is not None and g0["mismatches"] > 0:
            g.append(verdict_line("G1", "INCONCLUSIVE", what + "; the stock server does not repeat itself (G0)"))
        else:
            g.append(verdict_line("G1", "FAIL", what + ("" if g0 is not None else "; stock repeatability unknown")))

    # G2
    vacuous = None
    if "G2" in skip:
        g.append(verdict_line("G2", "WAIVED", "listed in --skip"))
    elif not all(k in docs for k in ("rand", "hf_stock", "hf_rand")):
        g.append(verdict_line("G2", "SKIPPED", "needs --rand, --hf-stock and --hf-rand"))
    else:
        gs = gap(docs["stock"], docs["hf_stock"])
        gr = gap(docs["rand"], docs["hf_rand"])
        e_sgl, e_hf = effect_vectors(docs["rand"], docs["stock"], docs["hf_rand"], docs["hf_stock"])
        eff_hf, eff_sgl = summarize(e_hf), summarize(e_sgl)
        rel, corr = rel_err_corr(e_sgl, e_hf)
        need = max(a.effect_min, a.noise_factor * gs["mean"])
        rep["g2"] = {"gap_stock": gs, "gap_rand": gr, "effect_hf": eff_hf, "effect_sgl": eff_sgl,
                     "effect_rel_err": rel, "effect_corr": corr, "effect_needed": need}
        vacuous = not (eff_hf["mean"] >= need)
        why = (f"gap p99 stock {gs['p99']:.4g} rand {gr['p99']:.4g}; effect mean hf {eff_hf['mean']:.4g} "
               f"sglang {eff_sgl['mean']:.4g}; effect rel err {rel:.3g} corr {corr:.4f}; "
               f"top1 agree stock {gs['top1_agree']:.4f} rand {gr['top1_agree']:.4f}")
        if vacuous:
            g.append(verdict_line("G2", "INCONCLUSIVE", f"W barely moves the HF logprobs (mean {eff_hf['mean']:.4g} < "
                                  f"{need:.4g}); make gates with a larger --target-t; " + why))
        else:
            fails = []
            if not gr["p99"] <= a.gap_ratio * gs["p99"] + a.gap_floor:
                fails.append("gap to HF grew beyond stock gap")
            if not rel <= a.effect_rel_tol:
                fails.append("effect of W differs between SGLang and HF")
            if not corr >= a.corr_min:
                fails.append("effect correlation low")
            if not gr["top1_agree"] >= gs["top1_agree"] - a.top1_slack:
                fails.append("top-1 agreement with HF dropped")
            g.append(verdict_line("G2", "FAIL" if fails else "PASS", ("; ".join(fails) + ": " if fails else "") + why))

    # G3
    if "G3" in skip:
        g.append(verdict_line("G3", "WAIVED", "listed in --skip"))
    elif "rand" not in docs:
        g.append(verdict_line("G3", "SKIPPED", "needs --rand"))
    else:
        ds, dr = decode_gap(docs["stock"]), decode_gap(docs["rand"])
        rep["g3"] = {"stock": ds, "rand": dr}
        if "zero" in docs:
            rep["g3"]["zero"] = decode_gap(docs["zero"])
        why = f"decode vs prefill |lp| p99 stock {ds['p99']:.4g} rand {dr['p99']:.4g} (n {ds['n']}, {dr['n']})"
        if ds["n"] < a.min_n or dr["n"] < a.min_n:
            g.append(verdict_line("G3", "INCONCLUSIVE", "too few decode steps compared; " + why))
        elif vacuous is None or vacuous:
            g.append(verdict_line("G3", "INCONCLUSIVE", "G2 did not show that W moves the logprobs, so a decode path "
                                  "that ignored W would pass; " + why))
        elif dr["p99"] <= a.decode_ratio * ds["p99"] + a.decode_floor:
            g.append(verdict_line("G3", "PASS", why))
        else:
            g.append(verdict_line("G3", "FAIL", "decode drifts from prefill beyond the stock gap; " + why))

    vs = {x["gate"]: x["verdict"] for x in g}
    if any(vs.get(k) == "FAIL" for k in ("G1", "G2", "G3")):
        rep["exit"] = 1
    elif any(v in ("INCONCLUSIVE", "SKIPPED") for k, v in vs.items() if k != "G0") or vs.get("G0") == "FAIL":
        rep["exit"] = 2
    else:
        rep["exit"] = 0
    rep["summary"] = vs
    if a.out:
        jdump(rep, a.out)
    return rep


# ----------------------------------------------------------------------------------------------------------
# selftest: a mock /generate server with the SGLang response shape, and red arms for every gate
# ----------------------------------------------------------------------------------------------------------
def _u(*parts: Any) -> float:
    d = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(d[:8], "big") / 2.0 ** 64


class MockModel:
    """Deterministic pseudo-logprobs with the structure the gates test: a base value per prefix, an effect of W per
    prefix (same in the HF reference and in the server), implementation noise per path (prefill or decode)."""

    def __init__(self, path: str = "sp", has_w: bool = False, wscale: float = 1.0, sigma: float = 2e-3,
                 sigma_w: Optional[float] = None, dec_ignores_w: bool = False, flaky: bool = False,
                 perturb: bool = False, short_logprobs: bool = False):
        self.path, self.has_w, self.wscale, self.sigma = path, has_w, wscale, sigma
        self.sigma_w = sigma if sigma_w is None else sigma_w
        self.dec_ignores_w, self.flaky, self.perturb, self.short = dec_ignores_w, flaky, perturb, short_logprobs
        self.calls = 0

    def _shift(self, path: str, key: str, slot: str) -> float:
        """effect of W plus implementation noise"""
        eff = 0.0
        if self.has_w and not (path == "sd" and self.dec_ignores_w):
            eff = self.wscale * 0.6 * (_u("eff", key, slot) - 0.5)
        noise = 0.0
        if path != "hf":
            sg = self.sigma_w if self.has_w else self.sigma
            noise = sg * (2 * _u(path, key, slot, self.has_w, self.calls if self.flaky else 0) - 1)  # W changes the rounding pattern
        return eff + noise

    def _val(self, path: str, key: str, slot: str) -> float:
        return -(0.3 + 11 * _u("base", key, slot)) + self._shift(path, key, slot)

    @staticmethod
    def _hashes(ids: Sequence[int]) -> List[str]:
        h = hashlib.sha256()
        out = [h.hexdigest()]
        for t in ids:
            h.update(int(t).to_bytes(4, "big"))
            out.append(h.hexdigest())
        return out

    def _top(self, path: str, key: str, k: int) -> List[list]:
        ents = []
        for j in range(k):
            tid = 1000 + int(_u("topid", key, j) * 50000)
            lp = -(0.5 + 1.3 * j + _u("topj", key, j)) + self._shift(path, key, f"t{j}")
            ents.append([lp, tid, None])
        ents.sort(key=lambda e: -e[0])
        return ents

    def prefill(self, ids: Sequence[int], k: int, path: Optional[str] = None) -> Tuple[list, list]:
        path = path or self.path
        H = self._hashes(ids)
        lps: list = [[None, int(ids[0]), None]]  # the shape SGLang returns at position 0
        tops: list = [None]
        for i in range(1, len(ids)):
            v = self._val(path, H[i + 1], "lp")
            if self.perturb and i == 2:
                v += 1e-6
            lps.append([v, int(ids[i]), None])
            tops.append(self._top(path, H[i], k))
        return lps, tops

    def next_token(self, prefix: Sequence[int]) -> int:
        key = self._hashes(prefix)[-1]
        return 1000 + int(_u("tok", key, self.has_w) * 50000)

    def decode(self, ids: Sequence[int], n: int, k: int) -> Tuple[list, list]:
        seq = list(ids)
        outs: list = []
        tops: list = []
        for j in range(n):
            tok = self.next_token(seq)
            H = self._hashes(seq + [tok])
            path = "sp" if j == 0 else "sd"  # the first token comes out of the prefill pass
            outs.append([self._val(path, H[-1], "lp"), tok, None])
            tops.append(self._top(path, H[-2], k))
            seq.append(tok)
        return outs, tops


def make_handler(model: MockModel):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model.calls += 1
            ids, sp = body["input_ids"], body["sampling_params"]
            k = int(body.get("top_logprobs_num", 0))
            lps, tops = model.prefill(ids, k)
            if model.short and len(ids) > 3:
                lps = lps[:-1]
            meta = {"input_token_logprobs": lps, "input_top_logprobs": tops}
            outs, otops = model.decode(ids, int(sp["max_new_tokens"]), k)
            meta["output_token_logprobs"], meta["output_top_logprobs"] = outs, otops
            raw = json.dumps({"text": "", "meta_info": meta}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return H


class MockServer:
    def __init__(self, model: MockModel):
        self.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), make_handler(model))
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self) -> "MockServer":
        self.thread.start()
        return self

    def __exit__(self, *a: Any) -> None:
        self.srv.shutdown()
        self.srv.server_close()


def selftest(tiny: Optional[str], tmp: str) -> dict:
    """CPU. Mock servers exercise collect over HTTP and every branch of compare, red arms included."""
    tmpd = Path(tmp)
    results: List[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})
        print(("ok   " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)

    def raises(name: str, fn: Any, frag: str) -> None:
        try:
            fn()
        except SystemExit as e:
            check(name, frag in str(e), f"message {e}")
            return
        check(name, False, "did not raise")

    pp = str(tmpd / "prompts.json")
    info = prepare(pp, 32, 7, 248320, None)
    check("prepare writes 32 prompts with edge lengths", info["n"] == 32 and 64 in info["lengths"] and 129 in info["lengths"])
    doc = jload(pp)
    doc["prompts"][0]["input_ids"][0] += 1
    bad = str(tmpd / "prompts_edited.json")
    jdump(doc, bad)
    raises("an edited prompt file is refused", lambda: load_prompts(bad), "edited")

    stock_log = "INFO server started\nINFO ready\n"
    zero_log = ("INFO server started\nnegeig: active v1: 48 gated GDN layers on this rank, gates from zeros "
                "(layers absent from the stream), max|W|=0\n")
    rand_log = ("negeig: active v1: 48 gated GDN layers on this rank, gates from file /x/g.safetensors sha256=ab, "
                "max|W|=0.0312\n")
    logs: Dict[str, str] = {}

    def run(name: str, model: MockModel, log: Optional[str] = None, decode: int = 16, prompts: str = pp) -> str:
        logp = None
        if log is not None:
            logp = str(tmpd / f"{name}.log")
            Path(logp).write_text(log)
        with MockServer(model) as srv:
            collect(srv.url, prompts, str(tmpd / f"{name}.json"), name, 6, decode, logp, 60)
        return str(tmpd / f"{name}.json")

    def hf(name: str, model: MockModel) -> str:
        p = run(name, model, None, decode=1)
        d = jload(p)
        d["kind"] = "hf"
        jdump(d, p)
        return p

    f: Dict[str, str] = {}
    stock_model = MockModel()
    f["stock"] = run("stock", stock_model, stock_log)
    f["stock2"] = run("stock2", stock_model, stock_log)
    f["zero"] = run("zero", MockModel(), zero_log)
    f["zero_nolog"] = run("zero_nolog", MockModel(), None)
    f["zero_bad"] = run("zero_bad", MockModel(perturb=True), zero_log)
    f["rand"] = run("rand", MockModel(has_w=True), rand_log)
    f["hf_stock"] = hf("hf_stock", MockModel(path="hf"))
    f["hf_rand"] = hf("hf_rand", MockModel(path="hf", has_w=True))
    d = jload(f["stock"])
    check("collect parses prefill logprobs, top-k, decode and replay",
          d["records"][5]["prefill"]["lp"][0] is None and len(d["records"][5]["prefill"]["lp"]) == d["records"][5]["n"]
          and len(d["records"][5]["decode"]["tokens"]) == 16 and len(d["records"][5]["replay"]["lp"]) == 16
          and len(d["records"][5]["prefill"]["top"][3]) == 6 and d["meta"]["active_lines"] == 0,
          "record shape")
    check("collect reads the engagement line of the patched log",
          jload(f["zero"])["meta"]["active_lines"] == 1 and jload(f["rand"])["meta"]["max_w"] == ["0.0312"])
    with MockServer(MockModel(short_logprobs=True)) as srv:
        raises("a server that returns fewer logprobs than tokens stops collect",
               lambda: collect(srv.url, pp, str(tmpd / "x.json"), "short", 6, 4, None, 60), "logprobs for")

    def cmp(skip: Optional[str] = None, **over: str) -> dict:
        args = ["compare", "--stock", f["stock"]]
        names = {"stock_repeat": "stock2", "zero": "zero", "rand": "rand", "hf_stock": "hf_stock", "hf_rand": "hf_rand"}
        for opt, key in names.items():
            path = over.pop(opt, f[key] if key in f else None)
            if path:
                args += ["--" + opt.replace("_", "-"), path]
        if "stock" in over:
            args[2] = over.pop("stock")
        if skip:
            args += ["--skip", skip]
        return compare(build_parser().parse_args(args))

    def verdicts(rep: dict) -> str:
        return " ".join(f"{k}={v}" for k, v in sorted(rep["summary"].items()))

    rep = cmp()
    check("green path: G0..G3 PASS, exit 0", rep["exit"] == 0 and set(rep["summary"].values()) == {"PASS"}, verdicts(rep))
    print("     " + "; ".join(x["why"] for x in rep["gates"] if x["gate"] in ("G2", "G3")), flush=True)
    check("non-vacuity: the random W moves HF logprobs far above the stock gap",
          rep["g2"]["effect_hf"]["mean"] > 20 * rep["g2"]["gap_stock"]["mean"], f"{rep['g2']['effect_hf']['mean']:.4g}")

    # G1 red arms
    rep = cmp(zero=f["zero_bad"])
    check("red arm G1: one float off by 1e-6 fails bit identity", rep["summary"]["G1"] == "FAIL" and rep["exit"] == 1, verdicts(rep))
    rep = cmp(zero=f["zero_nolog"])
    check("G1: equal numbers without engagement evidence are INCONCLUSIVE", rep["summary"]["G1"] == "INCONCLUSIVE" and rep["exit"] == 2, verdicts(rep))
    rep = cmp(zero=f["stock2"])
    check("G1: an unpatched server posing as the zeros server has no engagement line", rep["summary"]["G1"] == "INCONCLUSIVE", verdicts(rep))
    flaky = MockModel(flaky=True)
    f["fl1"], f["fl2"] = run("fl1", flaky, stock_log), run("fl2", flaky, stock_log)
    rep = cmp(stock=f["fl1"], stock_repeat=f["fl2"])
    check("G0 red arm: a stock server that does not repeat fails G0 and G1 becomes INCONCLUSIVE",
          rep["summary"]["G0"] == "FAIL" and rep["summary"]["G1"] == "INCONCLUSIVE" and rep["exit"] == 2, verdicts(rep))
    rep = compare(build_parser().parse_args(["compare", "--stock", f["stock"], "--zero", f["zero_bad"]]))
    check("G1 fails without a repeat file too, and says so", rep["summary"]["G1"] == "FAIL" and "unknown" in rep["gates"][1]["why"], verdicts(rep))

    # G2 and G3 red arms
    arms = {
        "W ignored by the server": MockModel(has_w=False),
        "effect halved": MockModel(has_w=True, wscale=0.5),
        "effect sign flipped": MockModel(has_w=True, wscale=-1.0),
        "noise far above the stock gap": MockModel(has_w=True, sigma_w=0.2),
    }
    for label, model in arms.items():
        p = run("arm", model, rand_log)
        rep = cmp(rand=p)
        check(f"red arm G2: {label}", rep["summary"]["G2"] == "FAIL" and rep["exit"] == 1, verdicts(rep) + " | " + rep["gates"][3]["why"][:120])
    p = run("arm_dec", MockModel(has_w=True, dec_ignores_w=True), rand_log)
    rep = cmp(rand=p)
    check("red arm G3: a decode path that ignores W passes G2 and fails G3",
          rep["summary"]["G2"] == "PASS" and rep["summary"]["G3"] == "FAIL" and rep["exit"] == 1, verdicts(rep))
    tiny_rand = run("tiny_rand", MockModel(has_w=True, wscale=1e-4), rand_log)
    tiny_hf = hf("tiny_hf", MockModel(path="hf", has_w=True, wscale=1e-4))
    rep = cmp(rand=tiny_rand, hf_rand=tiny_hf)
    check("vacuity: a W that moves nothing makes G2 and G3 INCONCLUSIVE, never PASS",
          rep["summary"]["G2"] == "INCONCLUSIVE" and rep["summary"]["G3"] == "INCONCLUSIVE" and rep["exit"] == 2, verdicts(rep))
    rep = compare(build_parser().parse_args(["compare", "--stock", f["stock"], "--zero", f["zero"], "--skip", "G2,G3"]))
    check("--skip waives a gate and does not block the exit (a missing G0 is not a blocker)",
          rep["summary"]["G2"] == "WAIVED" and rep["exit"] == 0 and rep["summary"]["G0"] == "SKIPPED", verdicts(rep))
    rep = compare(build_parser().parse_args(["compare", "--stock", f["stock"], "--zero", f["zero"]]))
    check("without --skip a gate whose files are missing is SKIPPED and blocks the exit",
          rep["summary"]["G2"] == "SKIPPED" and rep["exit"] == 2, verdicts(rep))
    rep = compare(build_parser().parse_args(["compare", "--stock", f["stock"], "--stock-repeat", f["stock2"],
                                             "--zero", f["zero"], "--skip", "G2,G3"]))
    check("G0 and G1 only, G2 and G3 waived: exit 0", rep["exit"] == 0, verdicts(rep))
    other = tmpd / "prompts_b.json"
    prepare(str(other), 32, 8, 248320, None)
    pb = run("other_prompts", MockModel(), zero_log, prompts=str(other))
    raises("record files from different prompt files are refused", lambda: cmp(zero=pb), "different prompt files")

    # small pure checks
    check("pct and summarize", pct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.99) == 10 and summarize([-2.0, 1.0])["max"] == 2.0)
    r, c = rel_err_corr([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    check("rel_err_corr on equal vectors", r == 0.0 and abs(c - 1.0) < 1e-12)
    r, c = rel_err_corr([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    check("rel_err_corr on a vector of zeros: error 1, correlation undefined", r == 1.0 and math.isnan(c))

    if tiny:
        selftest_hf(tiny, tmpd, check)
    failed = [r for r in results if not r["ok"]]
    return {"passed": len(results) - len(failed), "failed": len(failed), "failures": failed}


def selftest_hf(tiny: str, tmpd: Path, check: Any) -> None:
    """The real hf-ref on a tiny qwen3_5 checkpoint (CPU): stock == patched at W = 0, a random W moves the logprobs."""
    pp = str(tmpd / "tiny_prompts.json")
    prepare(pp, 8, 3, 248320, None)
    gates = str(tmpd / "tiny_gates.safetensors")
    mk = make_random_gates(str(Path(tiny) / "config.json"), gates, 2.0, 11)
    common = dict(model_dir=tiny, prompts_path=pp, topk=6, device="cpu", dtype="bf16", patch_path=None, torch_gdn=True)
    out = {}
    for tag, mode in (("stock", "stock"), ("zeros", "zeros"), ("rand", gates)):
        out[tag] = str(tmpd / f"tiny_hf_{tag}.json")
        hf_ref(out=out[tag], tag=tag, mode=mode, **common)
    a, b, c = (jload(out[k]) for k in ("stock", "zeros", "rand"))
    r0 = a["records"][4]["prefill"]
    check("hf-ref record shape", r0["lp"][0] is None and len(r0["lp"]) == a["records"][4]["n"] and r0["top"][0] is None
          and len(r0["top"][1]) == 6 and all(math.isfinite(v) for v in r0["lp"][1:]))
    check("hf-ref: the patch at W = 0 is bit-identical to the library forward", exact(a, b)["mismatches"] == 0 and exact(a, b)["compared"] > 100)
    eff = summarize([x - y for x, y in lp_pairs(c, a)])
    check("hf-ref: a random W (std target_t 2) moves the logprobs", eff["mean"] > 1e-3 and exact(a, c)["mismatches"] > 0,
          f"mean |effect| {eff['mean']:.4g} ({mk['layers']} layers)")


# ----------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--vocab", type=int, default=248320)
    p.add_argument("--tokenizer", help="model dir: adds tokenized text prompts (English, Hebrew, code) before the random-id ones")
    p = sub.add_parser("make-random-gates")
    p.add_argument("--config", required=True, help="model config.json")
    p.add_argument("--out", required=True)
    p.add_argument("--target-t", type=float, default=0.5, help="std of the pre-tanh gate value for a unit-rms input")
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("collect")
    p.add_argument("--url", required=True, help="server base url, e.g. http://127.0.0.1:30000")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--decode-tokens", type=int, default=16, help="greedy tokens for G1/G3 (0 or 1 disables the decode part)")
    p.add_argument("--server-log", help="the server's stdout/stderr file: read for 'negeig: active' lines (engagement evidence)")
    p.add_argument("--timeout", type=float, default=600.0)
    p = sub.add_parser("hf-ref")
    p.add_argument("--model", required=True, help="HF model dir (same revision the server loads)")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", required=True, help="stock | zeros | path of a gates safetensors file")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--patch", help="patch.py of the HF retrofit (default: experiments/negeig_retrofit/patch.py)")
    p.add_argument("--torch-gdn", action="store_true", help="CPU only: library reference delta rule instead of fla Triton")
    p = sub.add_parser("compare")
    for k in ("stock", "stock-repeat", "zero", "rand", "hf-stock", "hf-rand"):
        p.add_argument("--" + k, required=(k == "stock"))
    p.add_argument("--out")
    p.add_argument("--skip", help="comma list of gates to waive, e.g. G2,G3")
    p.add_argument("--gap-ratio", type=float, default=2.0, help="G2: rand gap p99 <= ratio * stock gap p99 + floor")
    p.add_argument("--gap-floor", type=float, default=0.02)
    p.add_argument("--effect-min", type=float, default=0.02, help="G2 non-vacuity: mean |HF effect of W| at least this")
    p.add_argument("--noise-factor", type=float, default=3.0, help="G2 non-vacuity: and at least this many stock-gap means")
    p.add_argument("--effect-rel-tol", type=float, default=0.5, help="G2: ||effect_sglang - effect_hf|| / ||effect_hf|| at most")
    p.add_argument("--corr-min", type=float, default=0.9)
    p.add_argument("--top1-slack", type=float, default=0.03)
    p.add_argument("--decode-ratio", type=float, default=2.0, help="G3: rand decode-vs-prefill p99 <= ratio * stock + floor")
    p.add_argument("--decode-floor", type=float, default=0.02)
    p.add_argument("--min-n", type=int, default=100, help="fewest numbers a gate may rest on")
    p = sub.add_parser("selftest")
    p.add_argument("--tinyqwen", help="tiny qwen3_5 checkpoint dir: also run the real hf-ref on CPU")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    a = build_parser().parse_args(argv)
    if a.cmd == "prepare":
        res: Any = prepare(a.out, a.n, a.seed, a.vocab, a.tokenizer)
    elif a.cmd == "make-random-gates":
        res = make_random_gates(a.config, a.out, a.target_t, a.seed)
    elif a.cmd == "collect":
        res = collect(a.url, a.prompts, a.out, a.tag, a.topk, a.decode_tokens, a.server_log, a.timeout)
    elif a.cmd == "hf-ref":
        res = hf_ref(a.model, a.prompts, a.out, a.tag, a.mode, a.topk, a.device, a.dtype, a.patch, a.torch_gdn)
    elif a.cmd == "compare":
        res = compare(a)
        for gline in res["gates"]:
            print(f"{gline['gate']}: {gline['verdict']}  {gline['why']}")
        print(json.dumps(res, indent=2))
        sys.exit(res["exit"])
    else:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="check_parity_selftest_") as td:
            res = selftest(a.tinyqwen, td)
        print(json.dumps({k: v for k, v in res.items() if k != "failures"}))
        sys.exit(1 if res["failed"] else 0)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
