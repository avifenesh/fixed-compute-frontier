#!/usr/bin/env python3
"""A deterministic fake of `python -m sglang.launch_server` for the CPU tests of the Stage A general eval runner.
Standard library only. Never used on a box.

    fake_server.py --port 30000 [--served-model-name qwen3.8-27b] [--dp-size 8] [--tp-size 1]
                   [--special-ids 1,2,...,10] [--humaneval FILE.jsonl] [--gmmlu FILE.jsonl] [other launch_server flags]

Unknown arguments are ignored, so serve_arm.sh's whole launch command line can be handed to it unchanged.

Endpoints (all JSON)
  GET  /health, /v1/models                      200 once --delay seconds have passed (FAKE_DELAY), else 503
  POST /v1/chat/completions                     the 17*23 probe, Global-MMLU letters, IFEval text, plain text
  POST /v1/completions                          HumanEval completions, BFCL prompts, plain text
  POST /generate                                single and batch input_ids with the logprob fields ev_mmlupro.py reads
  POST /_fake/control {...}                     failure injection and behaviour switches, see CONTROL below
  GET  /_fake/stats                             request counts, max in-flight, last bodies, last params and last prompt tail per path

What makes it useful as a test oracle: every answer is a pure function of the request, and the functions are importable
(`letter_for`, `pair_logprob`, `letter_logprobs`, `option_scores`), so a test recomputes the expected metric on its own
instead of reading it back from the system under test.

CONTROL (POST /_fake/control, merged into the state, reply = the state)
  fail_status N + fail_next K   the next K work requests answer HTTP N (500, 503, 429, 400, ...)
  down true                     work requests are dropped without an answer (the client sees a connection error)
  latency_ms                    sleep before answering a work request
  no_ids_logprob true           /generate ignores token_ids_logprob (the client must fall back to top-k)
  greedy_wrong true             /generate's output_ids is not the argmax letter (selfcheck must fail)
  misaligned true               /generate's logprob list holds shifted token ids (the alignment check must fail)
  batch_noise X                 batch /generate adds X*(index+1) to the logprobs (selfcheck batch-vs-single must fail)
  salt S                        changes the hash behind every answer (so two arms differ by a known amount)
  gmmlu_err P                   probability (0..1, by hash) that a Global-MMLU answer is wrong
  he_corrupt_mod M              every M-th HumanEval task (by index + salt) gets a wrong body
  probe_answer TEXT             what the 17*23 probe answers (default 391)
  bfcl_mode text|call           plain text, or a tool call to the first tool in the prompt
  ctx_chars N                   a /v1/completions prompt longer than N chars gets the 400 "longer than the model's
                                context length"

Environment: FAKE_DELAY (seconds before /health is 200), FAKE_CRASH=1 (exit 3 at once), FAKE_SALT, FAKE_LETTER_IDS
(comma list, same as --special-ids), FAKE_LOGPROB_START (0 or 1: where input_token_logprobs starts relative to
logprob_start_len), FAKE_HUMANEVAL, FAKE_GMMLU, and for the negeig engagement lines in the log: with SGLANG_PLUGINS=negeig
set the server prints one `negeig: active ...` line per rank (dp * tp) reading the file named by NEGEIG_GATES;
FAKE_NEGEIG_RANKS (print this many instead), FAKE_NEGEIG_LAYERS, FAKE_NEGEIG_SHA (a wrong sha256), FAKE_NEGEIG_MAXW
(default 0.0123), FAKE_NEGEIG_FORCE=1 (print the lines even though the plugin is not loaded: the red arm of base/ctrl).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence

LETTERS = list("ABCDEFGHIJ")
G_LETTERS = list("ABCD")
TOPK_FILLER = 20
HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------------------------------------
# pure answer functions (importable by tests)
def h64(*parts) -> int:
    return int(hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12], 16)


def pair_logprob(prev: int, tok: int, salt: str = "") -> float:
    """Log-probability of token `tok` after token `prev`. A function of the pair only, so a batch request, a single
    request and a request that starts scoring earlier all agree."""
    return -0.5 - (h64("pair", salt, prev, tok) % 977) / 200.0


def option_scores(ctx: Sequence[int], opts: Sequence[Sequence[int]], salt: str = "") -> List[float]:
    """What ev_mmlupro's cloze scorer must compute: the mean logprob of each option's tokens after the context."""
    out = []
    for o in opts:
        seq = list(ctx) + list(o)
        n = len(ctx)
        lps = [pair_logprob(seq[i - 1], seq[i], salt) for i in range(n, len(seq))]
        out.append(sum(lps) / len(lps))
    return out


def letter_logprobs(prompt_ids: Sequence[int], letter_ids: Sequence[int], salt: str = "") -> Dict[int, float]:
    """Next-token log-probabilities of the letter tokens after a prompt: a function of the whole prompt."""
    key = h64("prompt", salt, *prompt_ids)
    return {lid: -0.2 - (h64("letter", key, lid) % 1000) / 250.0 for lid in letter_ids}


def letter_for(prompt: str, n: int = 4, salt: str = "") -> str:
    return LETTERS[h64("letter-for", salt, prompt) % n]


def gmmlu_block(prompt: str) -> str:
    """The question block at the end of a Global-MMLU prompt (what follows the last solved shot)."""
    last = None
    for m in re.finditer(r"Answer: [A-D]\n\n", prompt):
        last = m
    return prompt[last.end():] if last else prompt


def wrong_letter(gold: str, key: int) -> str:
    others = [x for x in G_LETTERS if x != gold]
    return others[key % len(others)]


def he_index_hash(idx: int, salt: str) -> int:
    return idx + h64("he", salt) % 997


# ----------------------------------------------------------------------------------------------------------
# state
class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.cfg: dict = {
            "fail_status": 0, "fail_next": 0, "down": False, "latency_ms": 0, "no_ids_logprob": False,
            "greedy_wrong": False, "misaligned": False, "batch_noise": 0.0, "salt": os.environ.get("FAKE_SALT", ""),
            "gmmlu_err": 0.0, "he_corrupt_mod": 0, "probe_answer": "391", "bfcl_mode": "text", "ctx_chars": 0,
        }
        self.counts: Dict[str, int] = {}
        self.inflight = 0
        self.max_inflight = 0
        self.bodies: Dict[str, List[str]] = {}
        self.params: Dict[str, dict] = {}
        self.tails: Dict[str, str] = {}  # the last 120 chars of the last text prompt per path (the prompt's end shows the template)
        self.served = "qwen3.8-27b"
        self.letter_ids: List[int] = []
        self.logprob_start = int(os.environ.get("FAKE_LOGPROB_START", "0"))
        self.delay = float(os.environ.get("FAKE_DELAY", "0"))
        self.he: Dict[str, dict] = {}
        self.gm: Dict[str, str] = {}


S = State()


def load_data(he_path: Optional[str], gm_path: Optional[str]) -> None:
    if he_path:
        for i, line in enumerate(open(he_path, encoding="utf-8")):
            if line.strip():
                r = json.loads(line)
                S.he[r["prompt"]] = {"idx": i, "canonical": r["canonical_solution"]}
    if gm_path:
        sys.path.insert(0, str(HERE))
        from ev_gmmlu import block  # noqa: E402  the very function that builds the prompts
        for line in open(gm_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                S.gm[block(r, False)] = r["answer"]


# ----------------------------------------------------------------------------------------------------------
def answer_chat(body: dict) -> dict:
    msgs = body.get("messages") or []
    user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
    system = next((m["content"] for m in msgs if m.get("role") == "system"), "")
    salt = S.cfg["salt"]
    if "17 times 23" in user:
        return {"text": str(S.cfg["probe_answer"]), "finish": "stop"}
    if system.strip() == "Answer with only the letter":
        block = gmmlu_block(user)
        gold = S.gm.get(block)
        if gold is None:
            return {"text": letter_for(user, 4, salt), "finish": "length"}
        err = float(S.cfg["gmmlu_err"])
        wrong = (h64("gm-err", salt, block) % 10000) / 10000.0 < err
        return {"text": (wrong_letter(gold, h64("gm-w", salt, block)) if wrong else gold), "finish": "length"}
    # IFEval and everything else: a short fixed answer that depends on the prompt (so the scorer has something real)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    k = h64("chat", salt, user) % len(words)
    return {"text": f"Here is a short answer. {words[k]} {words[(k + 1) % len(words)]}.\n\nThat is all.",
            "finish": "stop"}


def answer_completion(body: dict) -> dict:
    prompt = body.get("prompt") or ""
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    he = S.he.get(prompt)
    if he is not None:
        mod = int(S.cfg["he_corrupt_mod"])
        bad = mod > 0 and he_index_hash(he["idx"], S.cfg["salt"]) % mod == 0
        text = "    return None\n" if bad else he["canonical"]
        return {"text": text + "\n\ndef helper():\n    pass\n", "finish": "stop"}
    if "<tools>" in prompt and S.cfg["bfcl_mode"] == "call":
        m = re.search(r'"name":\s*"([^"]+)"', prompt[prompt.index("<tools>"):])
        if m:
            return {"text": '<tool_call>\n{"name": "%s", "arguments": {}}\n</tool_call>' % m.group(1), "finish": "stop"}
    return {"text": "I cannot help with that request.", "finish": "stop"}


def meta_for(ids: Sequence[int], body: dict, index: int, batch: bool) -> dict:
    salt = S.cfg["salt"]
    n = len(ids)
    meta: dict = {"id": "fake", "finish_reason": {"type": "length", "length": 1}, "prompt_tokens": n,
                  "completion_tokens": 1}
    start = body.get("logprob_start_len")
    start = 0 if start is None or start < 0 else int(start)
    if body.get("return_logprob"):
        noise = float(S.cfg["batch_noise"]) * (index + 1) if batch else 0.0
        first = min(n, start + S.logprob_start)
        ents = []
        for p in range(first, n):
            lp = None if p == 0 else pair_logprob(ids[p - 1], ids[p], salt) + noise
            tid = ids[p]
            if S.cfg["misaligned"]:
                tid = tid + 1
            ents.append([lp, tid, None])
        meta["input_token_logprobs"] = ents
        lids = list(S.letter_ids)
        table = letter_logprobs(ids, lids, salt) if lids else {}
        want_ids = body.get("token_ids_logprob")
        if want_ids and not S.cfg["no_ids_logprob"]:
            row = [[table.get(int(i), -20.0 - (int(i) % 7)), int(i), None] for i in want_ids]
            meta["output_token_ids_logprobs"] = [row]
        topk = body.get("top_logprobs_num")
        if topk:
            rows = [[lp, lid, None] for lid, lp in table.items()]
            rows += [[-12.0 - j * 0.1, 50000 + j, None] for j in range(TOPK_FILLER)]
            rows.sort(key=lambda r: -r[0])
            meta["output_top_logprobs"] = [rows[: int(topk)]]
        meta["output_token_logprobs"] = [[max(table.values()) if table else -1.0, 0, None]]
    return meta


def answer_generate(body: dict) -> object:
    ids = body.get("input_ids")
    if ids is None:
        return {"text": "fake", "meta_info": {"id": "fake"}}
    batch = bool(ids) and isinstance(ids[0], list)
    seqs = ids if batch else [ids]
    out = []
    for i, seq in enumerate(seqs):
        meta = meta_for(seq, body, i, batch)
        lids = list(S.letter_ids)
        table = letter_logprobs(seq, lids, S.cfg["salt"]) if lids else {}
        if table:
            best = max(table, key=lambda k: table[k])
            if S.cfg["greedy_wrong"]:
                best = min(table, key=lambda k: table[k])
            oid = best
        else:
            oid = 1
        out.append({"text": "x", "output_ids": [oid], "meta_info": meta})
    return out if batch else out[0]


# ----------------------------------------------------------------------------------------------------------
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a) -> None:
        pass

    def _send(self, code: int, obj) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _drop(self) -> None:
        self.close_connection = True
        try:
            self.connection.shutdown(2)
        except OSError:
            pass
        self.connection.close()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/_fake/stats":
            with S.lock:
                self._send(200, {"counts": dict(S.counts), "max_inflight": S.max_inflight, "bodies": S.bodies,
                                 "params": S.params, "tails": S.tails, "cfg": S.cfg})
            return
        if path in ("/health", "/health_generate", "/v1/models", "/get_model_info"):
            if time.time() - S.t0 < S.delay:
                self._send(503, {"error": "starting"})
                return
            if path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": S.served, "object": "model"}]})
            else:
                self._send(200, {})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n).decode("utf-8", "replace") if n else ""
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            self._send(400, {"error": "bad json"})
            return
        if path == "/_fake/control":
            with S.lock:
                S.cfg.update(body)
                cfg = dict(S.cfg)
            self._send(200, cfg)
            return
        with S.lock:
            S.counts[path] = S.counts.get(path, 0) + 1
            S.inflight += 1
            S.max_inflight = max(S.max_inflight, S.inflight)
            tail = S.bodies.setdefault(path, [])
            tail.append(raw[:600])
            del tail[:-5]
            S.params[path] = {k: v for k, v in body.items() if k not in ("messages", "prompt", "text", "input_ids")}
            if isinstance(body.get("prompt"), str):
                S.tails[path] = body["prompt"][-120:]
            down = bool(S.cfg["down"])
            fail = 0
            if S.cfg["fail_next"] > 0:
                S.cfg["fail_next"] -= 1
                fail = int(S.cfg["fail_status"])
            lat = float(S.cfg["latency_ms"]) / 1000.0
        try:
            if down:
                self._drop()
                return
            if lat:
                time.sleep(lat)
            if fail:
                self._send(fail, {"error": f"injected {fail}"})
                return
            if path == "/v1/chat/completions":
                a = answer_chat(body)
                self._send(200, {"id": "fake", "object": "chat.completion", "created": 0, "model": S.served,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": a["text"]},
                                              "finish_reason": a["finish"]}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}})
            elif path == "/v1/completions":
                prompt = body.get("prompt") or ""
                lim = int(S.cfg["ctx_chars"])
                if isinstance(prompt, str) and lim and len(prompt) > lim:
                    self._send(400, {"object": "error", "message": (
                        f"The input ({len(prompt) // 4} tokens) is longer than the model's context length ({lim // 4} "
                        "tokens)."), "type": "BadRequestError", "code": 400})
                    return
                a = answer_completion(body)
                self._send(200, {"id": "fake", "object": "text_completion", "created": 0, "model": S.served,
                                 "choices": [{"index": 0, "text": a["text"], "finish_reason": a["finish"],
                                              "logprobs": None}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}})
            elif path == "/generate":
                self._send(200, answer_generate(body))
            else:
                self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with S.lock:
                S.inflight -= 1


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 512
    allow_reuse_address = True


def negeig_lines(args: argparse.Namespace) -> None:
    plugin = os.environ.get("SGLANG_PLUGINS", "") == "negeig"
    force = os.environ.get("FAKE_NEGEIG_FORCE") == "1"
    if not plugin and not force:
        return
    gates = os.environ.get("NEGEIG_GATES", "")
    sha = os.environ.get("FAKE_NEGEIG_SHA", "")
    if not sha and gates and os.path.isfile(gates):
        sha = hashlib.sha256(open(gates, "rb").read()).hexdigest()
    ranks = int(os.environ.get("FAKE_NEGEIG_RANKS", str(args.dp_size * args.tp_size)))
    layers = os.environ.get("FAKE_NEGEIG_LAYERS", str(48 // max(1, args.tp_size)))
    maxw = os.environ.get("FAKE_NEGEIG_MAXW", "0.0123")
    # what the real plugin's _say() writes: a bare line and the same line through logging, per rank
    line = f"negeig: installed v1 on sglang 0.5.20 (pid {os.getpid()})"
    print(line, file=sys.stderr, flush=True)
    print(f"[2026-10-01 00:00:00] {line}", file=sys.stderr, flush=True)
    for r in range(ranks):
        line = (f"negeig: active v1: {layers} gated GDN layers on this rank, "
                f"gates from file {gates} sha256={sha}, max|W|={maxw}")
        print(line, file=sys.stderr, flush=True)
        print(f"[2026-10-01 00:00:0{r % 10} TP{r}] {line}", file=sys.stderr, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--served-model-name", default="qwen3.8-27b")
    ap.add_argument("--dp-size", type=int, default=1)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--special-ids", default=os.environ.get("FAKE_LETTER_IDS", ""))
    ap.add_argument("--humaneval", default=os.environ.get("FAKE_HUMANEVAL", ""))
    ap.add_argument("--gmmlu", default=os.environ.get("FAKE_GMMLU", ""))
    a, _unknown = ap.parse_known_args(argv)
    if os.environ.get("FAKE_CRASH") == "1":
        print("fake server: planned crash", file=sys.stderr, flush=True)
        return 3
    S.served = a.served_model_name
    S.letter_ids = [int(x) for x in a.special_ids.split(",") if x.strip()]
    load_data(a.humaneval or None, a.gmmlu or None)
    print(f"fake server: listening on {a.host}:{a.port} served={S.served} letters={S.letter_ids[:3]}...",
          file=sys.stderr, flush=True)
    negeig_lines(a)
    srv = Server((a.host, a.port), H)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
