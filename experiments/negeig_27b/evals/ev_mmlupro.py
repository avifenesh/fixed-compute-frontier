#!/usr/bin/env python3
"""MMLU-Pro through the served model, scored two ways from the same server (the merged adapter exactly as it serves).

cloze (the gate, band 1 point): per question one batch request, the option texts as continuations of
    "The following is a question about {category}.\\n\\nQuestion: {question}\\nAnswer:"
  score = mean per-token log-probability of the option's tokens, argmax over options (ties to the first), as
  experiments/negeig_retrofit/mmlu_cloze.py scored it on HF. Context = last 2000 tokens, option = first 200 tokens of
  " " + text, tokenized without special tokens.

letter (report only): the next-token distribution after
    "The following is a multiple choice question about {category}. Answer with the letter of the correct option.
     \\n\\nQuestion: ...\\nOptions:\\nA. ...\\nAnswer:"
  restricted to the letter tokens " A".." J" of the question's options, argmax (experiments/negeig_retrofit/
  general_eval.py and mmlu_letters.py). Last 3000 tokens of the prompt.

Why SGLang and not HF for this port: the merged adapter must be scored as it is served (the plugin gate, the merged
bf16 weights), eight DP replicas give eight times the throughput without eight more 54 GB HF loads, one server then
serves every eval of the arm, and HF is slower on Gated DeltaNet layers. The risk that comes with the port is the
meaning of SGLang's logprob fields, which `--selfcheck N` pins on the box before the run: batch against single
sequence scores, logprob_start_len len(ctx)-1 against 0, the tail's token ids against the option ids, letter scores by
token id against top-k, and the predicted letter against the greedy token.

Logprob requests need the radix cache off (serve_arm.sh's default). Needs transformers for the tokenizer.

    python ev_mmlupro.py --arm base --out RUN/base --tokenizer $MODEL_DIR [--modes cloze,letter] [--stride 1]
    python ev_mmlupro.py --arm base --out RUN/base --tokenizer $MODEL_DIR --selfcheck 40
Exit codes: 0 complete, 2 usage or data error, 3 items failed, 4 server down, 5 selfcheck failed.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalcommon import (EXIT_OK, EXIT_USAGE, EvalRun, ItemFailed, Server, add_common_args, eval_pins,  # noqa: E402
                        finish, load_normalized, load_pins, load_tokenizer, log, make_server, metric, resolve_out,
                        sha256_file, write_json)

EXIT_SELFCHECK = 5
LETTERS = list("ABCDEFGHIJ")
CTX_TAIL, OPT_CAP, LETTER_TAIL = 2000, 200, 3000
TOPK = 64


class AlignmentError(ItemFailed):
    """The logprob list did not hold the option's own tokens: the score would belong to other tokens."""


# ----------------------------------------------------------------------------------------------------------
# prompts and tokens
def cloze_context(tok, row: dict) -> List[int]:
    return tok.encode(f"The following is a question about {row['category']}.\n\nQuestion: {row['question']}\nAnswer:",
                      add_special_tokens=False)[-CTX_TAIL:]


def cloze_options(tok, row: dict) -> List[List[int]]:
    return [tok.encode(" " + o, add_special_tokens=False)[:OPT_CAP] for o in row["options"]]


def letter_prompt_ids(tok, row: dict) -> List[int]:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))
    prompt = (f"The following is a multiple choice question about {row['category']}. "
              f"Answer with the letter of the correct option.\n\nQuestion: {row['question']}\n"
              f"Options:\n{opts}\nAnswer:")
    return tok.encode(prompt, add_special_tokens=False)[-LETTER_TAIL:]


def letter_token_ids(tok) -> List[int]:
    ids = []
    for c in LETTERS:
        e = tok.encode(" " + c, add_special_tokens=False)
        if len(e) != 1:
            raise SystemExit(f"the letter ' {c}' is {len(e)} tokens in this tokenizer; letter scoring needs one")
        ids.append(e[0])
    if len(set(ids)) != len(ids):
        raise SystemExit("letter token ids are not distinct")
    return ids


# ----------------------------------------------------------------------------------------------------------
# /generate parsing
def _meta(res) -> dict:
    if isinstance(res, list):
        res = res[0]
    return res["meta_info"]


def tail_logprob(meta: dict, opt: Sequence[int]) -> float:
    """Mean logprob of the option's tokens: the last len(opt) entries of input_token_logprobs, each checked against
    the option's own token id. Offset-agnostic: it does not matter whether the list starts at logprob_start_len or
    one later."""
    ents = meta.get("input_token_logprobs")
    if not ents or len(ents) < len(opt):
        raise AlignmentError(f"{len(ents or [])} input logprob entries for an option of {len(opt)} tokens")
    tail = ents[-len(opt):]
    got = [int(e[1]) for e in tail]
    if got != list(opt):
        raise AlignmentError(f"tail token ids {got[:6]} != option token ids {list(opt)[:6]}")
    lps = [e[0] for e in tail]
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in lps):
        raise AlignmentError("a None or NaN logprob inside the option's tokens")
    return float(sum(lps)) / len(lps)


def cloze_scores(server: Server, ctx: List[int], opts: List[List[int]], start: Optional[int] = None) -> List[float]:
    seqs = [ctx + o for o in opts]
    start = len(ctx) - 1 if start is None else start
    res = server.generate({"input_ids": seqs, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                           "return_logprob": True, "logprob_start_len": start})
    if isinstance(res, dict):
        res = [res]
    if len(res) != len(seqs):
        raise AlignmentError(f"{len(res)} results for {len(seqs)} sequences")
    return [tail_logprob(_meta(r), o) for r, o in zip(res, opts)]


def cloze_single(server: Server, ctx: List[int], opt: List[int], start: Optional[int] = None) -> float:
    start = len(ctx) - 1 if start is None else start
    res = server.generate({"input_ids": ctx + opt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                           "return_logprob": True, "logprob_start_len": start})
    return tail_logprob(_meta(res), opt)


def argmax_first(xs: Sequence[float]) -> int:
    best, bi = -math.inf, 0
    for i, x in enumerate(xs):
        if x > best:
            best, bi = x, i
    return bi


class LetterScorer:
    """Letter log-probabilities of the next token. method ids: token_ids_logprob on the listed ids; topk: the top
    TOPK next tokens, missing letters are -inf; auto: ids, falling back to topk if the server does not return them."""

    def __init__(self, server: Server, ids: List[int], method: str = "auto") -> None:
        self.server, self.ids, self.method = server, ids, method
        self.used = {"ids": 0, "topk": 0}
        self._lock = threading.Lock()

    def _call(self, prompt_ids: List[int], how: str):
        body = {"input_ids": prompt_ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                "return_logprob": True, "logprob_start_len": len(prompt_ids) - 1}
        if how == "ids":
            body["token_ids_logprob"] = self.ids
        else:
            body["top_logprobs_num"] = TOPK
        return self.server.generate(body)

    def scores(self, prompt_ids: List[int], n_opts: int, how: Optional[str] = None) -> Tuple[List[float], str, int]:
        """(letter logprobs for the first n_opts letters, method used, greedy first output token id or -1)."""
        order = [how] if how else (["ids", "topk"] if self.method == "auto" else [self.method])
        last = ""
        for h in order:
            res = self._call(prompt_ids, h)
            meta = _meta(res)
            table: Dict[int, float] = {}
            if h == "ids":
                rows = meta.get("output_token_ids_logprobs")
                if rows and rows[0]:
                    table = {int(e[1]): float(e[0]) for e in rows[0] if e[0] is not None}
            else:
                rows = meta.get("output_top_logprobs")
                if rows and rows[0]:
                    table = {int(e[1]): float(e[0]) for e in rows[0] if e[0] is not None}
            if table and (h != "ids" or all(i in table for i in self.ids[:n_opts])):
                with self._lock:
                    self.used[h] += 1
                out = [table.get(self.ids[i], -math.inf) for i in range(n_opts)]
                greedy = -1
                oids = res[0].get("output_ids") if isinstance(res, list) else res.get("output_ids")
                if oids:
                    greedy = int(oids[0])
                return out, h, greedy
            last = f"{h}: no letter logprobs in the response"
        raise ItemFailed("letter scoring: " + last)


# ----------------------------------------------------------------------------------------------------------
def selfcheck(server: Server, tok, rows: List[dict], n: int, method: str, tol: float, out: Path) -> int:
    """Pin the logprob semantics on the live server. Writes receipts/mmlupro.selfcheck.json; returns the exit code."""
    step = max(1, len(rows) // max(n, 1))
    pick = rows[::step][:n]
    lids = letter_token_ids(tok)
    scorer = LetterScorer(server, lids, "ids")
    checks: Dict[str, dict] = {k: {"n": 0, "max_abs_diff": 0.0, "bad": 0} for k in
                               ("batch_vs_single", "start_end_vs_zero", "ids_vs_topk")}
    argmax_same = {"n": 0, "agree": 0}
    greedy = {"n": 0, "greedy_is_letter": 0, "agree_when_letter": 0}
    problems: List[str] = []
    for r in pick:
        ctx, opts = cloze_context(tok, r), cloze_options(tok, r)
        try:
            batch = cloze_scores(server, ctx, opts)
            single = [cloze_single(server, ctx, o) for o in opts]
            zero = [cloze_single(server, ctx, o, start=0) for o in opts]
        except ItemFailed as e:
            problems.append(f"{r['id']}: {e}")
            continue
        for name, other in (("batch_vs_single", single), ("start_end_vs_zero", zero)):
            d = max(abs(x - y) for x, y in zip(batch, other))
            c = checks[name]
            c["n"] += 1
            c["max_abs_diff"] = max(c["max_abs_diff"], d)
            c["bad"] += int(d > tol)
        argmax_same["n"] += 1
        argmax_same["agree"] += int(argmax_first(batch) == argmax_first(single))
        pids = letter_prompt_ids(tok, r)
        k = len(r["options"])
        try:
            a, _, g = scorer.scores(pids, k, "ids")
            b, _, _ = scorer.scores(pids, k, "topk")
        except ItemFailed as e:
            problems.append(f"{r['id']}: letters: {e}")
            continue
        finite = [i for i in range(k) if math.isfinite(a[i]) and math.isfinite(b[i])]
        if finite:
            d = max(abs(a[i] - b[i]) for i in finite)
            c = checks["ids_vs_topk"]
            c["n"] += 1
            c["max_abs_diff"] = max(c["max_abs_diff"], d)
            c["bad"] += int(d > tol or argmax_first(a) != argmax_first(b))
        greedy["n"] += 1
        if g in lids[:k]:
            greedy["greedy_is_letter"] += 1
            greedy["agree_when_letter"] += int(lids[argmax_first(a)] == g)
    ok = (not problems and all(c["n"] > 0 and c["bad"] == 0 for c in checks.values())
          and argmax_same["agree"] == argmax_same["n"]
          and greedy["agree_when_letter"] == greedy["greedy_is_letter"])
    doc = {"n_items": len(pick), "tol": tol, "checks": checks, "cloze_argmax_batch_vs_single": argmax_same,
           "letter_predicted_vs_greedy": greedy, "problems": problems[:20], "ok": ok}
    write_json(out, doc)
    log(f"selfcheck {'OK' if ok else 'FAILED'}: {doc}")
    return EXIT_OK if ok else EXIT_SELFCHECK


# ----------------------------------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap, workers=96)
    ap.add_argument("--tokenizer", default=os.environ.get("EVAL_TOKENIZER"),
                    help="model directory or tokenizer directory (the same for every arm)")
    ap.add_argument("--modes", default="cloze,letter", help="comma list of cloze, letter")
    ap.add_argument("--letter-method", choices=("ids", "topk", "auto"), default="auto")
    ap.add_argument("--selfcheck", type=int, default=0, metavar="N",
                    help="check the logprob semantics on N items against the live server and exit")
    ap.add_argument("--selfcheck-tol", type=float, default=0.25,
                    help="max abs logprob difference (bf16 batch-shape noise) before a check counts as bad")
    a = ap.parse_args(argv)
    resolve_out(a)
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    if not modes or any(m not in ("cloze", "letter") for m in modes):
        print("--modes: cloze, letter or both", file=sys.stderr)
        return EXIT_USAGE
    if not a.tokenizer:
        print("--tokenizer (or EVAL_TOKENIZER) is required", file=sys.stderr)
        return EXIT_USAGE
    pins = load_pins()
    rows, sha = load_normalized(a, "mmlupro_test", pins)
    if not a.allow_unpinned and len(rows) != pins["datasets"]["mmlupro"]["rows"]:
        print(f"mmlupro: {len(rows)} rows, pinned {pins['datasets']['mmlupro']['rows']}", file=sys.stderr)
        return EXIT_USAGE
    tok = load_tokenizer(a.tokenizer)
    lids = letter_token_ids(tok) if "letter" in modes else []
    tj = Path(a.tokenizer) / "tokenizer.json"
    if a.selfcheck:
        return selfcheck(make_server(a), tok, rows, a.selfcheck, a.letter_method, a.selfcheck_tol,
                         Path(a.out) / "receipts" / "mmlupro.selfcheck.json")
    by_id = {r["id"]: r for r in rows}
    config = {"eval": "mmlupro", "v": 1, "modes": sorted(modes), "ctx_tail": CTX_TAIL, "opt_cap": OPT_CAP,
              "letter_tail": LETTER_TAIL, "letter_ids": lids, "score": "mean_token_logprob",
              "tokenizer_json_sha256": sha256_file(tj) if tj.is_file() else None}
    run = EvalRun("mmlupro", a, config, eval_pins(a, pins, "mmlupro", {"mmlupro_test.jsonl": sha}),
                  [r["id"] for r in rows])
    scorer = LetterScorer(run.server, lids, a.letter_method) if "letter" in modes else None

    def work(item_id: str) -> dict:
        r = by_id[item_id]
        out: dict = {"gold": r["answer_index"], "category": r["category"], "n_opts": len(r["options"])}
        if "cloze" in modes:
            sc = cloze_scores(run.server, cloze_context(tok, r), cloze_options(tok, r))
            out["cp"] = argmax_first(sc)
            out["c"] = int(out["cp"] == r["answer_index"])
        if scorer is not None:
            lp, how, _ = scorer.scores(letter_prompt_ids(tok, r), len(r["options"]))
            finite = any(math.isfinite(x) for x in lp)
            out["lp"] = argmax_first(lp) if finite else -1
            out["l"] = int(out["lp"] == r["answer_index"])
            out["lm"] = how
        return out

    status = run.execute(work)

    def build():
        recs = run.records()
        m: Dict[str, dict] = {}
        diag: dict = {"modes": modes}
        if "cloze" in modes:
            m["mmlupro_cloze"] = metric({r["id"]: r["c"] for r in recs}, n_full=len(rows))
            per: Dict[str, List[int]] = {}
            for r in recs:
                per.setdefault(r["category"], []).append(r["c"])
            diag["cloze_per_category"] = {k: [sum(v), len(v)] for k, v in sorted(per.items())}
        if "letter" in modes:
            m["mmlupro_letter"] = metric({r["id"]: r["l"] for r in recs}, n_full=len(rows))
            diag["letter_pred_hist"] = [sum(1 for r in recs if r["lp"] == i) for i in range(10)]
            diag["letter_gold_hist"] = [sum(1 for r in recs if r["gold"] == i) for i in range(10)]
            diag["letter_method_used"] = {k: sum(1 for r in recs if r.get("lm") == k) for k in ("ids", "topk")}
            diag["letter_no_letter_logprobs"] = sum(1 for r in recs if r["lp"] == -1)
        return m, diag

    return finish(run, status, build)


if __name__ == "__main__":
    sys.exit(main())
