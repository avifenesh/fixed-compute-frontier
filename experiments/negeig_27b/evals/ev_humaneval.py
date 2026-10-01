#!/usr/bin/env python3
"""HumanEval (openai/openai_humaneval, 164 tasks) pass@1 through the served model, greedy.

Protocol = experiments/negeig_retrofit/general_eval.py: the raw prompt as a plain completion (no chat template, no
special tokens), temperature 0, the completion cut at the earliest of the stop strings below (done client side, like
the HF run), program = prompt + completion + test + check(entry_point). The token budget is 512 here (the HF run used
384); it is the same for every arm, and a longer budget only removes truncation failures.

Each program runs in its own process: RLIMIT_AS 2 GiB, RLIMIT_CPU 10 s, a 30 s wall timeout that kills the whole
process group, an empty environment but PATH, a scratch working directory that is deleted after. A semaphore keeps the
number of concurrent programs at or below the core count so a loaded box does not turn slow into wrong. A program that
exits early with status 0 counts as passing, as in the original harness.

    python ev_humaneval.py --arm base --out RUN/base --data evals/data
    python ev_humaneval.py --check-canonical --data evals/data      # no server: the sandbox against the reference solutions
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalcommon import (EXIT_OK, EXIT_USAGE, EvalRun, add_common_args, completion_text, eval_pins,  # noqa: E402
                        finish, load_normalized, load_pins, log, metric, resolve_out)

STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n```"]
MAX_TOKENS = 512
WALL_S, CPU_S, MEM_BYTES = 30.0, 10, 2 << 30
_WRAP = ("import resource, runpy, sys\n"
         f"resource.setrlimit(resource.RLIMIT_AS, ({MEM_BYTES}, {MEM_BYTES}))\n"
         f"resource.setrlimit(resource.RLIMIT_CPU, ({CPU_S}, {CPU_S}))\n"
         "runpy.run_path(sys.argv[1], run_name='__main__')\n")
_SLOTS = threading.BoundedSemaphore(max(1, min(8, os.cpu_count() or 1)))


def cut_completion(comp: str) -> str:
    cut = min([comp.find(st) for st in STOPS if st in comp] + [len(comp)])
    return comp[:cut]


def program_for(row: dict, completion: str) -> str:
    return row["prompt"] + completion + "\n\n" + row["test"] + f"\n\ncheck({row['entry_point']})\n"


def run_program(code: str, wall: float = WALL_S) -> Tuple[bool, str]:
    """(passed, short reason). Never raises for a bad program: a crash, a timeout and a memory bomb are all 'failed'."""
    with _SLOTS, tempfile.TemporaryDirectory(prefix="he_") as d:
        prog = os.path.join(d, "prog.py")
        with open(prog, "w", encoding="utf-8") as f:
            f.write(code)
        p = subprocess.Popen([sys.executable, "-c", _WRAP, prog], cwd=d, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env={"PATH": "/usr/bin:/bin"}, start_new_session=True)
        try:
            _, err = p.communicate(timeout=wall)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            p.communicate()
            return False, "timeout"
        finally:
            try:
                os.killpg(p.pid, signal.SIGKILL)  # stragglers the program forked
            except (ProcessLookupError, PermissionError):
                pass
        if p.returncode == 0:
            return True, ""
        tail = (err or b"").decode("utf-8", "replace").strip().splitlines()[-1:] or [f"exit {p.returncode}"]
        return False, tail[0][:200]


def check_canonical(rows: List[dict]) -> dict:
    """The sandbox against known answers: every reference solution must pass, and an empty body must fail."""
    good = [run_program(program_for(r, r["canonical_solution"]))[0] for r in rows]
    empty = [run_program(program_for(r, ""))[0] for r in rows]
    bombs = {
        "infinite_loop": run_program("while True:\n    pass\n", wall=5.0)[0],
        "memory_bomb": run_program("x = bytearray(8 << 30)\n")[0],
        "raise": run_program("raise ValueError('x')\n")[0],
    }
    ok = all(good) and sum(empty) <= 2 and not any(bombs.values())
    return {"n": len(rows), "canonical_pass": sum(good), "canonical_failed": [r["id"] for r, g in zip(rows, good) if not g],
            "empty_body_pass": sum(empty), "bombs_pass": bombs, "ok": ok}


def main(argv: Optional[List[str]] = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--check-canonical", action="store_true")
    pre.add_argument("--data", default=os.environ.get("EVAL_DATA", str(Path(__file__).resolve().parent / "data")))
    pre.add_argument("--allow-unpinned", action="store_true")
    pa, _ = pre.parse_known_args(argv)
    pins = load_pins()
    if pa.check_canonical:
        rows, _ = load_normalized(pa, "humaneval_test", pins)
        res = check_canonical(rows)
        log(f"canonical check: {res}")
        print(json.dumps(res, sort_keys=True))  # stdout is the receipt run_general.sh keeps as humaneval.canonical.json
        return EXIT_OK if res["ok"] else EXIT_USAGE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap, workers=32)
    ap.add_argument("--check-canonical", action="store_true",
                    help="run the reference solutions through the sandbox and exit (no server needed)")
    a = ap.parse_args(argv)
    resolve_out(a)
    rows, sha = load_normalized(a, "humaneval_test", pins)
    if not a.allow_unpinned and len(rows) != pins["datasets"]["humaneval"]["rows"]:
        print(f"humaneval: {len(rows)} rows, pinned {pins['datasets']['humaneval']['rows']}", file=sys.stderr)
        return EXIT_USAGE
    by_id = {r["id"]: r for r in rows}
    config = {"eval": "humaneval", "v": 1, "endpoint": "completions", "max_tokens": MAX_TOKENS, "temperature": 0,
              "stops": STOPS, "rlimit_as": MEM_BYTES, "rlimit_cpu_s": CPU_S, "wall_s": WALL_S}
    run = EvalRun("humaneval", a, config, eval_pins(a, pins, "humaneval", {"humaneval_test.jsonl": sha}),
                  [r["id"] for r in rows])

    def work(item_id: str) -> dict:
        r = by_id[item_id]
        res = run.server.completions(r["prompt"], max_tokens=MAX_TOKENS, temperature=0)
        raw, fin = completion_text(res)
        comp = cut_completion(raw)
        ok, why = run_program(program_for(r, comp))
        return {"ok": int(ok), "why": why, "finish": fin, "truncated": int(fin == "length" and len(comp) == len(raw)),
                "comp": comp}

    status = run.execute(work)

    def build():
        recs = run.records()
        diag = {"n_truncated_at_budget": sum(r["truncated"] for r in recs),
                "n_timeout": sum(1 for r in recs if r["why"] == "timeout"),
                "n_empty_completion": sum(1 for r in recs if not r["comp"].strip())}
        return {"humaneval": metric({r["id"]: r["ok"] for r in recs}, n_full=len(rows))}, diag

    return finish(run, status, build)


if __name__ == "__main__":
    sys.exit(main())
