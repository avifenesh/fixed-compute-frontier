#!/usr/bin/env python3
"""Shared plumbing for the Stage A general-eval runner. Standard library only (the tokenizer helper imports
transformers lazily), so it runs on the GPU box and on the rig.

What every eval module gets from here:

* `Server`        a stdlib HTTP client with retry and backoff on transient statuses and a circuit breaker. Items
                  that cannot be scored stay undone. They are never written down as wrong answers.
* `Journal`       a resumable jsonl journal: one record per finished item, a header that pins the configuration,
                  and run records that accumulate the wall time across preemptions.
* `EvalRun`       selection (full set or a stride subset), the bounded worker pool, progress lines, the status
                  receipt, and the standard result file.
* `write_result`  the one result schema compare_general.py reads.

Exit codes of every ev_*.py: 0 complete, 2 usage or data error, 3 some items failed (resume re-runs them),
4 server down (circuit breaker tripped).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import http.client
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "negeig-general-eval/1"
RETRY_STATUS = (0, 408, 429, 500, 502, 503, 504)
THINK_OFF = {"enable_thinking": False}  # chat_template_kwargs: every eval here is think-off (PLAN.md)
# The 17 BFCL v4 categories this lane gates, with their full-set item counts (measured on the Hebrew lane's real base
# run: 4,441 items). Shared by ev_bfcl.py (what to generate) and compare_general.py (coverage, coarse categories).
BFCL_FULL = {
    "live_irrelevance": 884, "live_multiple": 1053, "live_parallel_multiple": 24, "live_parallel": 16,
    "live_relevance": 16, "live_simple": 258, "multi_turn_base": 200, "multi_turn_long_context": 200,
    "multi_turn_miss_func": 200, "multi_turn_miss_param": 200, "irrelevance": 240, "multiple": 200,
    "parallel_multiple": 200, "parallel": 200, "simple_java": 100, "simple_javascript": 50, "simple_python": 400,
}
EXIT_OK, EXIT_USAGE, EXIT_INCOMPLETE, EXIT_DOWN = 0, 2, 3, 4
HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------------------------------------
# small helpers
def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", file=sys.stderr, flush=True)


def sha256_file(path: os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def canon_sha(obj: Any) -> str:
    return hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()


def write_json(path: os.PathLike, obj: Any, indent: Optional[int] = 1) -> None:
    """Atomic: a reader never sees a half-written file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)


def read_json(path: os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: os.PathLike) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_pins() -> dict:
    return read_json(HERE / "pins.json")


def load_tokenizer(path: str):
    from transformers import AutoTokenizer  # lazy: only the MMLU-Pro module needs it

    return AutoTokenizer.from_pretrained(path, trust_remote_code=False)


# ----------------------------------------------------------------------------------------------------------
# HTTP client
class ServerDown(RuntimeError):
    """Consecutive requests exhausted their retries: the server is gone. Stop, do not score anything."""


class ItemFailed(RuntimeError):
    """One item could not be scored (non-retryable status, or retries exhausted). The item stays undone."""


class Server:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: float = 900.0,
                 attempts: int = 11, breaker: int = 8,
                 sleep: Callable[[float], None] = time.sleep,
                 backoff: Callable[[int], float] = lambda n: float(min(30, 2 ** n))) -> None:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        self.root = root
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.attempts = attempts
        self.breaker = breaker
        self._sleep = sleep
        self._backoff = backoff
        self._lock = threading.Lock()
        self._consecutive = 0
        self.down = threading.Event()
        self.stats = {"requests": 0, "retries": 0, "exhausted": 0}

    # one HTTP round trip: (status, bytes). status 0 means no HTTP answer (connect, reset, timeout).
    def _once(self, method: str, path: str, body: Optional[dict], timeout: float) -> Tuple[int, bytes]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.root + path, data=data, method=method, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            try:
                return e.code, e.read()
            except Exception:  # noqa: BLE001
                return e.code, b""
        except (urllib.error.URLError, socket.timeout, ConnectionError, http.client.HTTPException, OSError) as e:
            return 0, str(e).encode("utf-8", "replace")

    def call(self, path: str, body: Optional[dict] = None, method: Optional[str] = None,
             timeout: Optional[float] = None) -> Any:
        if self.down.is_set():
            raise ServerDown("server marked down by the circuit breaker")
        method = method or ("POST" if body is not None else "GET")
        last = ""
        for attempt in range(self.attempts):
            with self._lock:
                self.stats["requests"] += 1
            status, raw = self._once(method, path, body, timeout or self.timeout)
            if status == 200:
                try:
                    res = json.loads(raw)
                except ValueError:
                    status, raw = 0, b"200 with a body that is not json: " + raw[:120]
                else:
                    with self._lock:
                        self._consecutive = 0
                    return res
            if status not in RETRY_STATUS:
                raise ItemFailed(f"{method} {path}: HTTP {status}: {raw[:300].decode('utf-8', 'replace')}")
            last = f"HTTP {status}: {raw[:200].decode('utf-8', 'replace')}"
            with self._lock:
                self.stats["retries"] += 1
            if attempt + 1 < self.attempts:
                self._sleep(self._backoff(attempt))
        with self._lock:
            self.stats["exhausted"] += 1
            self._consecutive += 1
            tripped = self._consecutive >= self.breaker
            if tripped:
                self.down.set()
        if tripped:
            raise ServerDown(f"{self.breaker} consecutive requests exhausted {self.attempts} attempts; last: {last}")
        raise ItemFailed(f"{method} {path}: retries exhausted; last: {last}")

    def chat(self, messages: List[dict], **params: Any) -> dict:
        body = {"model": self.model, "messages": messages}
        body.update(params)
        return self.call("/v1/chat/completions", body)

    def completions(self, prompt: str, **params: Any) -> dict:
        body = {"model": self.model, "prompt": prompt}
        body.update(params)
        return self.call("/v1/completions", body)

    def generate(self, body: dict) -> Any:
        return self.call("/generate", body)

    def wait_ready(self, seconds: float, poll: float = 2.0) -> bool:
        end = time.time() + seconds
        while time.time() < end:
            status, _ = self._once("GET", "/v1/models", None, 10.0)
            if status == 200:
                return True
            time.sleep(poll)
        return False


def chat_text(res: dict) -> Tuple[str, str]:
    """(content, finish_reason) of the first choice of a chat completion."""
    ch = res["choices"][0]
    return (ch.get("message") or {}).get("content") or "", ch.get("finish_reason") or ""


def completion_text(res: dict) -> Tuple[str, str]:
    ch = res["choices"][0]
    return ch.get("text") or "", ch.get("finish_reason") or ""


# ----------------------------------------------------------------------------------------------------------
# journal
class JournalMismatch(SystemExit):
    pass


class Journal:
    """Resumable jsonl. Line 1 is {"_header": {...}}, then one {"id": ..., ...} per finished item and {"_run": ...}
    records. A configuration change refuses to resume (pass fresh=True or a new --out)."""

    def __init__(self, path: os.PathLike, config_sha: str, config_note: Optional[dict] = None,
                 fresh: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rec: Dict[str, dict] = {}
        self.runs: List[dict] = []
        self._lock = threading.Lock()
        if fresh and self.path.exists():
            self.path.unlink()
        header_ok = False
        if self.path.exists():
            with open(self.path, "rb") as f:
                raw = f.read()
            for line in raw.decode("utf-8", "replace").split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue  # a line cut by a kill mid-write; the item is simply redone
                if "_header" in obj:
                    if obj["_header"].get("config_sha") != config_sha:
                        raise JournalMismatch(
                            f"{self.path}: journal was written with a different configuration "
                            f"({obj['_header'].get('config_sha', '')[:12]} vs {config_sha[:12]}). "
                            "Use --fresh to discard it or a new --out.")
                    header_ok = True
                elif "_run" in obj:
                    self.runs.append(obj["_run"])
                elif "id" in obj:
                    self.rec[str(obj["id"])] = obj
            if self.rec and not header_ok:
                raise JournalMismatch(f"{self.path}: records without a header; use --fresh")
            self._fh = open(self.path, "a", encoding="utf-8")
            if raw and not raw.endswith(b"\n"):
                self._fh.write("\n")
        else:
            self._fh = open(self.path, "a", encoding="utf-8")
        if not header_ok:
            self._fh.write(json.dumps({"_header": {"config_sha": config_sha, "created": utc(),
                                                   "note": config_note or {}}}, ensure_ascii=False) + "\n")
            self._fh.flush()

    def __contains__(self, item_id: str) -> bool:
        return str(item_id) in self.rec

    def add(self, item_id: str, **fields: Any) -> None:
        rec = {"id": str(item_id)}
        rec.update(fields)
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            self.rec[str(item_id)] = rec

    def add_run(self, info: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps({"_run": info}, ensure_ascii=False) + "\n")
            self._fh.flush()
            self.runs.append(info)

    def total_seconds(self) -> float:
        return float(sum(r.get("seconds", 0.0) for r in self.runs))

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------------------------------------
# selection
def select_ids(ids: Sequence[str], stride: int = 1, offset: int = 0, limit: int = 0) -> List[str]:
    """Every stride-th id of the dataset's own order starting at offset, then the first `limit` of those.
    The order is the normalized file's order, pinned by MANIFEST.json, so arm and base pick the same items."""
    if stride < 1 or not 0 <= offset < stride:
        raise ValueError(f"stride {stride} offset {offset}")
    sel = list(ids)[offset::stride]
    if limit:
        sel = sel[:limit]
    return sel


def subset_info(full_ids: Sequence[str], sel_ids: Sequence[str], stride: int, offset: int, limit: int) -> dict:
    return {"stride": stride, "offset": offset, "limit": limit, "n_full": len(full_ids),
            "n_selected": len(sel_ids), "ids_sha256": canon_sha(list(sel_ids)),
            "full_ids_sha256": canon_sha(list(full_ids))}


# ----------------------------------------------------------------------------------------------------------
# arguments
def add_common_args(ap: argparse.ArgumentParser, workers: int = 64) -> None:
    ap.add_argument("--arm", required=True, help="arm name, recorded in the result (base, wide_s1, ctrl_s1, ...)")
    ap.add_argument("--out", required=True, help="arm output directory (journal/, results/, receipts/ live here)")
    ap.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:30000"),
                    help="server root, with or without /v1")
    ap.add_argument("--served-model", default=os.environ.get("EVAL_SERVED_MODEL", "qwen3.8-27b"))
    ap.add_argument("--api-key", default=os.environ.get("EVAL_API_KEY", "EMPTY"))
    ap.add_argument("--data", default=os.environ.get("EVAL_DATA", str(HERE / "data")),
                    help="directory written by prepare_data.py")
    ap.add_argument("--workers", type=int, default=workers, help="concurrent requests")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every stride-th item (1 = the full set)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N selected items only (smoke tests)")
    ap.add_argument("--fresh", action="store_true", help="discard this eval's journal and start over")
    ap.add_argument("--timeout", type=float, default=900.0, help="per request seconds")
    ap.add_argument("--attempts", type=int, default=11)
    ap.add_argument("--breaker", type=int, default=8,
                    help="consecutive requests that exhausted their retries before the server is declared down")
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="tests only: skip the check of the data files against pins.json (the result says so)")


def make_server(a: argparse.Namespace) -> Server:
    return Server(a.base_url, a.served_model, a.api_key, a.timeout, a.attempts, a.breaker)


def load_normalized(a: argparse.Namespace, name: str, pins: dict) -> Tuple[List[dict], str]:
    """(rows, sha256) of <data>/<name>.jsonl, checked against pins.json (and MANIFEST.json when that exists)."""
    p = Path(a.data) / f"{name}.jsonl"
    if not p.is_file():
        raise SystemExit(f"{p} is missing: run prepare_box.sh data (or prepare_data.py) first")
    sha = sha256_file(p)
    want = pins.get("normalized_sha256", {}).get(f"{name}.jsonl")
    if want and want != sha and not getattr(a, "allow_unpinned", False):
        raise SystemExit(f"{p}: sha256 {sha[:12]} is not the pinned {want[:12]}; re-run prepare_data.py")
    man = Path(a.data) / "MANIFEST.json"
    if man.is_file():
        mwant = read_json(man).get("files", {}).get(f"{name}.jsonl", {}).get("sha256")
        if mwant and mwant != sha:
            raise SystemExit(f"{p}: sha256 {sha[:12]} does not match MANIFEST.json {mwant[:12]}")
    return read_jsonl(p), sha


def eval_pins(a: argparse.Namespace, pins: dict, dataset_key: str, normalized: Dict[str, str]) -> dict:
    """The pin block stored in a result and compared between arm and base: model, dataset revision, file hashes."""
    return {"model": pins["model"], "dataset": pins["datasets"][dataset_key], "normalized_sha256": normalized,
            "unpinned": bool(getattr(a, "allow_unpinned", False))}


# ----------------------------------------------------------------------------------------------------------
# the run object
class Status:
    def __init__(self, total: int, already: int) -> None:
        self.total = total
        self.already = already
        self.done_now = 0
        self.failed: Dict[str, str] = {}
        self.down = False
        self.seconds = 0.0

    @property
    def complete(self) -> bool:
        return not self.failed and not self.down and (self.already + self.done_now) >= self.total

    @property
    def exit_code(self) -> int:
        if self.down:
            return EXIT_DOWN
        return EXIT_OK if self.complete else EXIT_INCOMPLETE

    def as_dict(self) -> dict:
        return {"total": self.total, "already": self.already, "done_now": self.done_now, "down": self.down,
                "n_failed": len(self.failed), "failed_first20": dict(list(self.failed.items())[:20]),
                "complete": self.complete, "seconds": round(self.seconds, 1)}


class EvalRun:
    """One eval on one arm. `config` and `pins` decide the journal's identity."""

    def __init__(self, eval_name: str, a: argparse.Namespace, config: dict, pins: dict,
                 all_ids: Sequence[str], server: Optional[Server] = None,
                 sel: Optional[Sequence[str]] = None, subset_extra: Optional[dict] = None) -> None:
        """`sel` replaces the default flat stride/offset/limit selection (BFCL selects per category); the
        journal identity then follows the ids actually selected."""
        self.name = eval_name
        self.a = a
        self.out = Path(a.out)
        self.config = config
        self.pins = pins
        self.all_ids = [str(i) for i in all_ids]
        if sel is None:
            try:
                self.sel = select_ids(self.all_ids, a.stride, a.offset, a.limit)
            except ValueError as e:
                raise SystemExit(f"bad subset arguments: {e}") from e
        else:
            self.sel = [str(i) for i in sel]
            if not set(self.sel) <= set(self.all_ids):
                raise SystemExit("selection has ids outside the full set")
        self.subset = subset_info(self.all_ids, self.sel, a.stride, a.offset, a.limit)
        if subset_extra:
            self.subset.update(subset_extra)
        self.config_sha = canon_sha({"eval": eval_name, "config": config, "pins": pins,
                                     "sel": self.subset["ids_sha256"]})
        self.server = server or make_server(a)
        self.journal = Journal(self.out / "journal" / f"{eval_name}.jsonl", self.config_sha,
                               {"eval": eval_name, "config": config}, fresh=a.fresh)
        self.status: Optional[Status] = None

    def todo(self) -> List[str]:
        return [i for i in self.sel if i not in self.journal]

    def execute(self, work: Callable[[str], dict], progress_every: float = 30.0) -> Status:
        """Run work(item_id) -> fields for every undone selected item on a bounded pool."""
        todo = self.todo()
        st = Status(len(self.sel), len(self.sel) - len(todo))
        self.status = st
        t0 = time.time()
        log(f"[{self.name}] arm={self.a.arm} selected={len(self.sel)} already={st.already} todo={len(todo)} "
            f"workers={self.a.workers}")
        lock = threading.Lock()
        last = [t0]

        def one(item_id: str) -> None:
            if self.server.down.is_set():
                return
            try:
                fields = work(item_id)
            except ServerDown:
                st.down = True
                return
            except ItemFailed as e:
                with lock:
                    st.failed[item_id] = str(e)[:300]
                return
            except Exception as e:  # noqa: BLE001  a bug in a scorer must show up as a failed item, not vanish
                with lock:
                    st.failed[item_id] = f"{type(e).__name__}: {str(e)[:280]}"
                return
            self.journal.add(item_id, **fields)
            with lock:
                st.done_now += 1
                now = time.time()
                if now - last[0] >= progress_every:
                    last[0] = now
                    rate = st.done_now / max(now - t0, 1e-9)
                    left = len(todo) - st.done_now - len(st.failed)
                    log(f"[{self.name}] {st.already + st.done_now}/{st.total} done, {rate:.1f} items/s, "
                        f"eta {left / max(rate, 1e-9):.0f}s, failed {len(st.failed)}")

        window = max(1, self.a.workers) * 4
        with cf.ThreadPoolExecutor(max_workers=max(1, self.a.workers)) as pool:
            pending: set = set()
            it = iter(todo)
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < window and not self.server.down.is_set():
                    try:
                        item_id = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(pool.submit(one, item_id))
                if self.server.down.is_set():
                    exhausted = True
                if not pending:
                    break
                done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
        if self.server.down.is_set():
            st.down = True
        st.seconds = time.time() - t0
        self.journal.add_run({"start": utc(), "seconds": round(st.seconds, 2), "done": st.done_now,
                              "failed": len(st.failed), "down": st.down})
        self.write_status()
        log(f"[{self.name}] finished: done_now={st.done_now} failed={len(st.failed)} down={st.down} "
            f"in {st.seconds:.0f}s")
        return st

    def write_status(self) -> None:
        if self.status is None:
            return
        doc = {"schema": SCHEMA, "eval": self.name, "arm": self.a.arm, "utc": utc(), **self.status.as_dict(),
               "server_stats": dict(self.server.stats), "subset": self.subset}
        write_json(self.out / "receipts" / f"{self.name}.status.json", doc)

    def records(self) -> List[dict]:
        return [self.journal.rec[i] for i in self.sel if i in self.journal]

    def write_result(self, metrics: Dict[str, dict], diagnostics: Optional[dict] = None) -> Path:
        doc = {
            "schema": SCHEMA, "eval": self.name, "arm": self.a.arm, "utc": utc(), "complete": True,
            "subset": self.subset, "config": self.config, "config_sha": self.config_sha, "pins": self.pins,
            "seconds": round(self.journal.total_seconds(), 1), "diagnostics": diagnostics or {},
            "server_stats": dict(self.server.stats), "metrics": metrics,
        }
        path = self.out / "results" / f"{self.name}.json"
        write_json(path, doc, indent=None)
        return path


def metric(items: Dict[str, int], level: str = "item", n_full: Optional[int] = None) -> dict:
    """One metric of a result file: per-item 0/1 outcomes keyed by item id (level "instruction" keys are
    "<prompt id>:<index>"), so compare_general.py can pair arm and base item by item."""
    n = len(items)
    out = {"n": n, "score": (sum(items.values()) / n) if n else None, "level": level, "items": items}
    if n_full is not None:
        out["n_full"] = n_full
    return out


def finish(run: EvalRun, status: Status, build: Callable[[], Tuple[Dict[str, dict], dict]]) -> int:
    """Shared tail of every ev_*.py main(): write the result only when every selected item is done."""
    if not status.complete:
        log(f"[{run.name}] incomplete: re-run the same command to resume (exit {status.exit_code})")
        return status.exit_code
    metrics, diag = build()
    path = run.write_result(metrics, diag)
    summary = {k: (round(v["score"], 4) if v["score"] is not None else None, v["n"]) for k, v in metrics.items()
               if not k.endswith("_inst")}
    log(f"[{run.name}] wrote {path}: {summary}")
    return EXIT_OK


def resolve_out(a: argparse.Namespace) -> None:
    Path(a.out).mkdir(parents=True, exist_ok=True)
