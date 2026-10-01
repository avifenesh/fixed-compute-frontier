#!/usr/bin/env python3
"""BFCL v4 (17 categories, 4,441 items) through the served model, think-off, scored by BFCL's own checker.

This module does not reimplement BFCL. It drives the pinned gorilla checkout (pins.json bfcl.commit, patched by
bfcl/bfcl_patch.py) the way the Hebrew lane did, `bfcl generate` then `bfcl evaluate`, and turns the score files into
per-item outcomes so compare_general.py can pair arm and base item by item:

* handler    bfcl/tiyuvta_hermes.py renders the tokenizer's own chat template locally (enable_thinking false) and posts
             to /v1/completions. The tokenizer directory is the BASE model's for every arm (an adapter changes weights,
             never the template).
* generation `bfcl generate --model negeig/qwen3.8-27b-FC --skip-server-setup`, stock temperature 0.001, cap
             BFCL_MAX_TOKENS (default 4096, the stock value), one project root per arm so arms cannot mix.
* resume     the BFCL result files are the resumable state: a re-run generates only the ids with no result.
* errors     a result row "Error during inference: ..." is a transport failure and is NOT a wrong answer. It is moved
             to receipts/bfcl.errors.jsonl and regenerated on the next pass, up to --error-passes tries per id, after
             which it stays and scores wrong (the id is listed in the diagnostics). The one exception is a request
             longer than the server's context window: that is the model's own trajectory overflowing, deterministic,
             scored wrong like BFCL does, and counted separately (n_context_overflow). Serve with a window that makes
             it rare; the Hebrew lane's base run had 44 of 200 multi_turn_long_context items overflow 32,768.
* scoring    `bfcl evaluate --partial-eval`, then correct = the ids in the result minus the ids listed as failures.
             The summary line's total_count and correct_count must agree with that, or the run stops.

Categories are the 17 the lane gates (single_turn plus multi_turn of BFCL v4 as the Hebrew lane ran them). The agentic
categories (web search, memory) need external search keys and are not part of this gate. --stride and --limit select
per category (the data file's own order); --limit is per category too.

    python ev_bfcl.py --arm base --out RUN/base --bfcl-venv $BFCL_VENV --tokenizer $MODEL_DIR [--stride 4]

Needs network to nothing but the served endpoint. Exit codes as the other evals: 0 complete, 2 usage/data/tool error,
3 incomplete (re-run resumes), 4 server down.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalcommon import (BFCL_FULL, EXIT_DOWN, EXIT_INCOMPLETE, EXIT_USAGE, THINK_OFF, EvalRun, Server, Status,  # noqa: E402
                        add_common_args, canon_sha, finish, load_pins, log, metric, read_json, resolve_out,
                        sha256_file, utc, write_json)

DEFAULT_TEMPERATURE = 0.001  # stock BFCL
DEFAULT_MAX_TOKENS = 4096    # stock BFCL generation cap
CATEGORIES = list(BFCL_FULL)
ERROR_PREFIX = "Error during inference"
# SGLang's 400 for a request over the window: "The input (35565 tokens) is longer than the model's context length (...)";
# BFCL stores str(e), whose repr escapes the apostrophe (model\'s).
CONTEXT_RE = re.compile(r"is longer than the model\W{0,3}s context length")


# ----------------------------------------------------------------------------------------------------------
# layout
def group_of(cat: str) -> str:
    if cat.startswith("live_"):
        return "live"
    if cat.startswith("multi_turn"):
        return "multi_turn"
    return "non_live"


def key_dir(key: str) -> str:
    return key.replace("/", "_")


def result_file(root: Path, key: str, cat: str) -> Path:
    return root / "result" / key_dir(key) / group_of(cat) / f"BFCL_v4_{cat}_result.json"


def score_file(root: Path, key: str, cat: str) -> Path:
    return root / "score" / key_dir(key) / group_of(cat) / f"BFCL_v4_{cat}_score.json"


def category_ids(data_dir: Path, cat: str) -> List[str]:
    """The ids of one category in the data file's own order (jsonl, one entry per line)."""
    p = data_dir / f"BFCL_v4_{cat}.json"
    if not p.is_file():
        raise SystemExit(f"{p} is missing: is --bfcl-data the bfcl_eval/data directory?")
    ids = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)["id"]))
    return ids


def select_per_category(full: Dict[str, List[str]], stride: int, offset: int, limit: int) -> Dict[str, List[str]]:
    if stride < 1 or not 0 <= offset < stride:
        raise SystemExit(f"bad subset arguments: stride {stride} offset {offset}")
    out = {}
    for cat, ids in full.items():
        sel = ids[offset::stride]
        out[cat] = sel[:limit] if limit else sel
    return out


# ----------------------------------------------------------------------------------------------------------
# result files
def classify_result(res) -> str:
    """'ok', 'context' (deterministic overflow, a scored failure) or 'error' (transport, to be regenerated)."""
    if isinstance(res, str) and res.startswith(ERROR_PREFIX):
        return "context" if CONTEXT_RE.search(res) else "error"
    return "ok"


def read_rows(path: Path) -> Tuple[List[dict], int]:
    """(rows, number of unreadable lines). A line cut by a kill mid-write is dropped, not fatal."""
    rows, bad = [], 0
    if not path.is_file():
        return rows, bad
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append({"id": str(obj["id"]), "_raw": line, "result": obj.get("result")})
            except (ValueError, KeyError, TypeError):
                bad += 1
    return rows, bad


def scan_results(root: Path, key: str, sel: Dict[str, List[str]], counts: dict, error_passes: int,
                 receipts_path: Optional[Path] = None) -> dict:
    """Normalize every selected category's result file and say what is still missing.

    Drops unreadable lines and duplicate ids (the last row of an id wins), keeps rows of ids outside the selection
    untouched out of the picture (they are removed from the file only when the file is rewritten anyway), moves transport
    errors out of the result file while they have tries left, and records a try per id in `counts` (mutated).
    Returns {"missing": {cat: [ids]}, "kinds": {id: kind}, "accepted": {id: msg}, "stripped": n, "repaired": n}."""
    missing: Dict[str, List[str]] = {}
    kinds: Dict[str, str] = {}
    accepted: Dict[str, str] = {}
    stripped = repaired = 0
    for cat, ids in sel.items():
        want = set(ids)
        path = result_file(root, key, cat)
        rows, bad = read_rows(path)
        last: Dict[str, dict] = {}
        for r in rows:
            last[r["id"]] = r
        keep: List[dict] = []
        for i, r in last.items():
            if i not in want:
                continue  # an id this run did not select (an earlier, wider subset): not scored, not kept
            kind = classify_result(r["result"])
            if kind == "error":
                rec = counts.setdefault(i, {"n": 0, "msg": ""})
                rec["n"] += 1
                rec["msg"] = str(r["result"])[:240]
                if rec["n"] >= error_passes:
                    accepted[i] = rec["msg"]
                    keep.append(r)
                    kinds[i] = "accepted_error"
                else:
                    stripped += 1
                    if receipts_path is not None:
                        receipts_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(receipts_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"utc": utc(), "id": i, "try": rec["n"], "of": error_passes,
                                                "message": rec["msg"]}, ensure_ascii=False) + "\n")
                continue
            kinds[i] = kind
            keep.append(r)
        dirty = bool(bad) or len(last) != len(rows) or len(keep) != len(rows) or any(r["id"] not in want for r in rows)
        if dirty and path.is_file():
            repaired += 1 if (bad or len(last) != len(rows)) else 0
            tmp = path.with_name(path.name + f".tmp{os.getpid()}")
            with open(tmp, "w", encoding="utf-8") as f:
                for r in keep:
                    f.write(r["_raw"] + "\n")
            os.replace(tmp, path)
        have = {r["id"] for r in keep}
        gone = [i for i in ids if i not in have]
        if gone:
            missing[cat] = gone
    return {"missing": missing, "kinds": kinds, "accepted": accepted, "stripped": stripped, "repaired": repaired}


def parse_scores(root: Path, key: str, sel: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """{cat: {id: 0/1}} from BFCL's score files. correct = the selected ids minus the listed failures, and the
    summary line must say the same (total_count, correct_count), or this raises SystemExit."""
    out: Dict[str, Dict[str, int]] = {}
    for cat, ids in sel.items():
        if not ids:
            continue
        p = score_file(root, key, cat)
        if not p.is_file():
            raise SystemExit(f"{p} missing: bfcl evaluate did not score {cat}")
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            raise SystemExit(f"{p} is empty")
        head = json.loads(lines[0])
        fails = set()
        for ln in lines[1:]:
            obj = json.loads(ln)
            if "id" in obj:
                fails.add(str(obj["id"]))
        want = set(ids)
        stray = sorted(fails - want)
        if stray:
            raise SystemExit(f"{p}: failure ids outside the selection: {stray[:5]}")
        if int(head["total_count"]) != len(ids):
            raise SystemExit(f"{p}: BFCL scored {head['total_count']} items of {cat}, the selection has {len(ids)}")
        correct = len(ids) - len(fails)
        if int(head["correct_count"]) != correct:
            raise SystemExit(f"{p}: summary says {head['correct_count']} correct, the failure list implies {correct}")
        out[cat] = {i: (0 if i in fails else 1) for i in ids}
    return out


# ----------------------------------------------------------------------------------------------------------
# the installed BFCL
def locate_bfcl(venv: Path, env: dict) -> dict:
    """{dir, data, handler_sha256, patched, cli} of the bfcl_eval the venv (plus PYTHONPATH) imports."""
    py = venv / "bin" / "python"
    cli = venv / "bin" / "bfcl"
    for p in (py, cli):
        if not p.exists():
            raise SystemExit(f"{p} not found: run prepare_box.sh bfcl, or pass --bfcl-venv")
    r = subprocess.run([str(py), "-c", "import bfcl_eval, os; print(os.path.dirname(bfcl_eval.__file__))"],
                       env=env, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise SystemExit(f"import bfcl_eval failed in {venv}: {r.stderr.strip()[-400:]}")
    d = Path(r.stdout.strip().splitlines()[-1])
    handler = d / "model_handler" / "local_inference" / "tiyuvta_hermes.py"
    mc = (d / "constants" / "model_config.py").read_text(encoding="utf-8")
    bo = (d / "model_handler" / "local_inference" / "base_oss_handler.py").read_text(encoding="utf-8")
    pins = load_pins()["bfcl"]
    patched = {"handler": handler.is_file(), "registry_key": f'"{pins["registry_key"]}": ModelConfig(' in mc,
               "cap_env": 'os.getenv("BFCL_MAX_TOKENS"' in bo}
    return {"dir": str(d), "data": str(d / "data"), "cli": str(cli), "patched": patched,
            "handler_sha256": sha256_file(handler) if handler.is_file() else None}


def bfcl_env(a: argparse.Namespace, root: Path) -> dict:
    env = dict(os.environ)
    base = a.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    env.update({"BFCL_PROJECT_ROOT": str(root), "REMOTE_OPENAI_BASE_URL": base, "REMOTE_OPENAI_API_KEY": a.api_key,
                "REMOTE_OPENAI_TOKENIZER_PATH": a.tokenizer, "BFCL_SERVED_MODEL": a.served_model,
                "BFCL_TEMPLATE_KWARGS": json.dumps(THINK_OFF), "BFCL_MAX_TOKENS": str(a.max_tokens),
                "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    if a.bfcl_pythonpath:
        env["PYTHONPATH"] = a.bfcl_pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _kill(p: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            p.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            continue


def run_generate(cmd: List[str], env: dict, cwd: Path, logfile: Path, server: Server, deadline_s: float,
                 root: Path, health_every: float = 30.0, health_fails: int = 6) -> Tuple[Optional[int], str]:
    """Run bfcl generate; (returncode or None, why). why is '' | 'deadline' | 'server_down'."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "ab") as lf:
        lf.write(f"\n=== {utc()} {' '.join(cmd)}\n".encode())
        lf.flush()
        p = subprocess.Popen(cmd, env=env, cwd=str(cwd), stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        last_health = t0
        last_log = t0
        fails = 0
        try:
            while p.poll() is None:
                time.sleep(2.0)
                now = time.time()
                if now - t0 > deadline_s:
                    log(f"[bfcl] generate passed its {deadline_s:.0f}s deadline, killing it")
                    _kill(p)
                    return None, "deadline"
                if now - last_health >= health_every:
                    last_health = now
                    status, _ = server._once("GET", "/v1/models", None, 20.0)
                    fails = 0 if status == 200 else fails + 1
                    if fails >= health_fails:
                        log(f"[bfcl] server stopped answering ({fails} probes), killing generate")
                        _kill(p)
                        return None, "server_down"
                if now - last_log >= 60.0:
                    last_log = now
                    n = sum(len(read_rows(f)[0]) for f in (root / "result").rglob("*_result.json"))
                    log(f"[bfcl] generating, {n} result rows on disk, {now - t0:.0f}s")
            return p.returncode, ""
        finally:
            if p.poll() is None:
                _kill(p)


# ----------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap, workers=64)
    ap.add_argument("--bfcl-venv", default=os.environ.get("BFCL_VENV", ""),
                    help="venv with the patched gorilla install (prepare_box.sh bfcl); env BFCL_VENV")
    ap.add_argument("--tokenizer", default=os.environ.get("EVAL_TOKENIZER", ""),
                    help="directory with config.json and the tokenizer: the BASE model dir for every arm; env EVAL_TOKENIZER")
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("BFCL_MAX_TOKENS", DEFAULT_MAX_TOKENS)))
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--error-passes", type=int, default=3,
                    help="tries per id before a transport error is accepted as a scored failure")
    ap.add_argument("--deadline-s", type=float, default=4 * 3600.0, help="wall limit of one bfcl generate pass")
    ap.add_argument("--ready-wait", type=float, default=60.0, help="seconds to wait for /v1/models before exit 4")
    ap.add_argument("--include-input-log", action="store_true",
                    help="debug: keep the rendered prompt in the BFCL inference log (large)")
    ap.add_argument("--bfcl-data", default="", help="override the bfcl_eval/data directory (tests)")
    ap.add_argument("--bfcl-pythonpath", default="", help="tests only: directory put first on PYTHONPATH for bfcl")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    resolve_out(a)
    pins = load_pins()
    bp = pins["bfcl"]
    if not a.bfcl_venv:
        print("--bfcl-venv (or env BFCL_VENV) is required", file=sys.stderr)
        return EXIT_USAGE
    if not a.tokenizer or not (Path(a.tokenizer) / "config.json").is_file():
        print(f"--tokenizer {a.tokenizer!r}: needs a directory holding config.json and the tokenizer files", file=sys.stderr)
        return EXIT_USAGE
    out = Path(a.out)
    root = out / "bfcl"
    env = bfcl_env(a, root)
    inst = locate_bfcl(Path(a.bfcl_venv), env)
    if not all(inst["patched"].values()):
        print(f"the BFCL install is not patched ({inst['patched']}): run prepare_box.sh bfcl", file=sys.stderr)
        return EXIT_USAGE
    if inst["handler_sha256"] != bp["handler_sha256"] and not a.allow_unpinned:
        print(f"installed handler sha256 {str(inst['handler_sha256'])[:12]} is not the pinned {bp['handler_sha256'][:12]}",
              file=sys.stderr)
        return EXIT_USAGE
    data_dir = Path(a.bfcl_data or inst["data"])
    full = {c: category_ids(data_dir, c) for c in CATEGORIES}
    bad = {c: (len(ids), BFCL_FULL[c]) for c, ids in full.items() if len(ids) != BFCL_FULL[c]}
    if bad and not a.allow_unpinned:
        print(f"BFCL data counts differ from the pinned ones (found, pinned): {bad}", file=sys.stderr)
        return EXIT_USAGE
    sel = select_per_category(full, a.stride, a.offset, a.limit)
    is_subset = any(len(sel[c]) != len(full[c]) for c in CATEGORIES)
    flat_full = [i for c in CATEGORIES for i in full[c]]
    flat_sel = [i for c in CATEGORIES for i in sel[c]]
    config = {"eval": "bfcl", "v": 1, "protocol": "bfcl generate/evaluate, FC handler tiyuvta_hermes, local chat template",
              "registry_key": bp["registry_key"], "temperature": a.temperature, "max_tokens": a.max_tokens,
              "chat_template_kwargs": THINK_OFF, "categories": CATEGORIES, "gorilla_commit": bp["commit"],
              "handler_sha256": bp["handler_sha256"], "tokenizer_config_sha256": sha256_file(Path(a.tokenizer) / "config.json"),
              "select": "per category: ids[offset::stride][:limit]"}
    server = Server(a.base_url, a.served_model, a.api_key, a.timeout, a.attempts, a.breaker)
    pin_block = {"model": pins["model"], "bfcl": {k: bp[k] for k in ("repo", "commit", "registry_key", "handler_sha256",
                                                                       "requirements_sha256")},
                 "category_counts_sha256": canon_sha({c: len(full[c]) for c in CATEGORIES}),
                 "category_ids_sha256": canon_sha(full), "unpinned": bool(a.allow_unpinned)}
    if len(set(flat_full)) != len(flat_full):
        print("BFCL ids are not unique across categories", file=sys.stderr)
        return EXIT_USAGE
    if a.fresh:
        for d in (root / "result", root / "score", root / "state"):
            shutil.rmtree(d, ignore_errors=True)
    run = EvalRun("bfcl", a, config, pin_block, flat_full, server=server, sel=flat_sel,
                  subset_extra={"limit_per_category": a.limit, "select": "per category: ids[offset::stride][:limit]",
                                "per_category": {c: [len(sel[c]), len(full[c])] for c in CATEGORIES}})
    todo = run.todo()
    status = Status(len(flat_sel), len(flat_sel) - len(todo))
    run.status = status
    log(f"[bfcl] arm={a.arm} selected={len(flat_sel)} of {len(flat_full)} ({'subset' if is_subset else 'full'}), "
        f"already={status.already}, todo={len(todo)}, workers={a.workers}")
    diag: dict = {"installed": inst, "subset": run.subset["per_category"], "generate_passes": []}
    if todo:
        t0 = time.time()
        if not server.wait_ready(a.ready_wait):
            log(f"[bfcl] {a.base_url} did not answer /v1/models within {a.ready_wait:.0f}s")
            status.down = True
            run.write_status()
            return EXIT_DOWN
        root.mkdir(parents=True, exist_ok=True)
        state = root / "state"
        counts_path = state / "error_counts.json"
        counts = read_json(counts_path) if counts_path.is_file() else {}
        cmd = [inst["cli"], "generate", "--model", bp["registry_key"], "--temperature", str(a.temperature),
               "--num-threads", str(a.workers), "--skip-server-setup"]
        if a.include_input_log:
            cmd.append("--include-input-log")
        if is_subset:
            write_json(root / "test_case_ids_to_generate.json", {c: sel[c] for c in CATEGORIES if sel[c]})
            cmd.append("--run-ids")
        else:
            cmd += ["--test-category", ",".join(CATEGORIES)]
        logfile = out / "logs" / "bfcl-generate.log"
        rc, why = run_generate(cmd, env, root, logfile, server, a.deadline_s, root)
        diag["generate_passes"].append({"returncode": rc, "why": why, "utc": utc()})
        if why == "server_down" or not server.wait_ready(5.0):
            status.down = True
            run.write_status()
            return EXIT_DOWN
        sc = scan_results(root, bp["registry_key"], sel, counts, a.error_passes, out / "receipts" / "bfcl.errors.jsonl")
        write_json(counts_path, counts)
        n_missing = sum(len(v) for v in sc["missing"].values())
        if n_missing:
            status.failed = {i: ("no result after generate" if i not in counts else counts[i]["msg"])
                             for v in sc["missing"].values() for i in v}
            status.done_now = len(todo) - n_missing
            run.journal.add_run({"start": utc(), "seconds": round(time.time() - t0, 2), "done": 0,
                                 "failed": n_missing, "down": False})
            run.write_status()
            log(f"[bfcl] {n_missing} items have no result (generate rc={rc} {why}); transport errors were moved to "
                f"receipts/bfcl.errors.jsonl. Re-run the same command to resume (exit {EXIT_INCOMPLETE})")
            return EXIT_INCOMPLETE
        shutil.rmtree(root / "score", ignore_errors=True)
        ecmd = [inst["cli"], "evaluate", "--model", bp["registry_key"], "--test-category", ",".join(CATEGORIES),
                "--partial-eval"]
        elog = out / "logs" / "bfcl-evaluate.log"
        with open(elog, "ab") as lf:
            lf.write(f"\n=== {utc()} {' '.join(ecmd)}\n".encode())
            lf.flush()
            er = subprocess.run(ecmd, env=env, cwd=str(root), stdout=lf, stderr=subprocess.STDOUT)
        if er.returncode != 0:
            print(f"bfcl evaluate exited {er.returncode}; see {elog}", file=sys.stderr)
            return EXIT_USAGE
        try:
            scored = parse_scores(root, bp["registry_key"], sel)
        except SystemExit as e:
            print(f"cannot read the BFCL scores: {e}", file=sys.stderr)
            return EXIT_USAGE
        for cat in CATEGORIES:
            for i, ok in scored.get(cat, {}).items():
                if i in run.journal:
                    continue
                run.journal.add(i, cat=cat, ok=ok, kind=sc["kinds"].get(i, "ok"))
        status.done_now = len(todo)
        run.journal.add_run({"start": utc(), "seconds": round(time.time() - t0, 2), "done": len(todo), "failed": 0,
                             "down": False})
        diag["transport_errors_stripped_this_pass"] = sc["stripped"]
        diag["files_repaired_this_pass"] = sc["repaired"]
    run.write_status()

    def build():
        recs = run.records()
        by_cat: Dict[str, Dict[str, int]] = {c: {} for c in CATEGORIES}
        for r in recs:
            by_cat[r["cat"]][r["id"]] = int(r["ok"])
        metrics = {f"bfcl/{c}": metric(v, n_full=BFCL_FULL[c]) for c, v in by_cat.items() if v}
        allitems = {i: v for d in by_cat.values() for i, v in d.items()}
        metrics["bfcl/all"] = metric(allitems, n_full=sum(BFCL_FULL.values()))
        d = dict(diag)
        d["n_context_overflow"] = sum(1 for r in recs if r.get("kind") == "context")
        acc = {r["id"]: 1 for r in recs if r.get("kind") == "accepted_error"}
        d["n_accepted_transport_errors"] = len(acc)
        d["accepted_transport_error_ids_first20"] = sorted(acc)[:20]
        d["per_category"] = {c: [sum(v.values()), len(v)] for c, v in by_cat.items() if v}
        return metrics, d

    return finish(run, status, build)


if __name__ == "__main__":
    sys.exit(main())
