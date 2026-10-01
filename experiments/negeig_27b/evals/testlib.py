"""Shared helpers of the CPU tests for the Stage A general eval runner (standard library only).

What is here: a scratch directory that deletes itself, a subprocess runner, the fake server wrapper
(fake_server.py on a free port), small synthetic datasets in the normalized format prepare_data.py writes, and the
switches that tell a test where the optional real inputs are:

  EVAL_TEST_DATA        a directory written by prepare_data.py (real pinned data). Tests that need it skip without it.
  EVAL_TEST_TOKENIZER   the Qwen3.8-27B tokenizer directory (default /data/ai-ml/hf-models/qwen3.8-27b-tokenizer)
  EVAL_TEST_IFEVAL_PY   a python with nltk, langdetect, immutabledict and absl-py (default: this interpreter if it has them)
  EVAL_TEST_NLTK_DATA   punkt and punkt_tab (default: <EVAL_TEST_DATA>/nltk_data)
  EVAL_TEST_BFCL        a dir holding bfcl-venv/, gorilla/ and bfcl-runs/ of a real BFCL run (the BFCL tests skip without;
                        default: the Hebrew lane's ~/evals/agentic-tool-evals-20260907 when it exists)
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
PY = sys.executable
sys.path.insert(0, str(HERE))

TOKENIZER_DIR = os.environ.get("EVAL_TEST_TOKENIZER", "/data/ai-ml/hf-models/qwen3.8-27b-tokenizer")
REAL_DATA = os.environ.get("EVAL_TEST_DATA", "")


def have_tokenizer() -> bool:
    return (Path(TOKENIZER_DIR) / "tokenizer.json").is_file() and importlib.util.find_spec("transformers") is not None


def have_real_data(*names: str) -> bool:
    return bool(REAL_DATA) and all((Path(REAL_DATA) / n).exists() for n in names)


def ifeval_python() -> Optional[str]:
    cand = os.environ.get("EVAL_TEST_IFEVAL_PY")
    if cand:
        return cand if Path(cand).exists() else None
    if all(importlib.util.find_spec(m) for m in ("nltk", "langdetect", "immutabledict", "absl")):
        return PY
    return None


def nltk_data_dir() -> Optional[str]:
    cand = os.environ.get("EVAL_TEST_NLTK_DATA") or (str(Path(REAL_DATA) / "nltk_data") if REAL_DATA else "")
    return cand if cand and Path(cand).is_dir() else None


def bfcl_assets() -> Optional[Path]:
    """The directory with bfcl-venv/, gorilla/ and bfcl-runs/ (env EVAL_TEST_BFCL), or None when any of them is missing."""
    cand = Path(os.environ.get("EVAL_TEST_BFCL") or Path.home() / "evals" / "agentic-tool-evals-20260907")
    ok = (cand / "bfcl-venv" / "bin" / "bfcl").exists() and (cand / "gorilla" / ".git").exists() and (cand / "bfcl-runs").is_dir()
    return cand if ok else None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def scratch(prefix: str = "evt_") -> Iterator[Path]:
    d = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class ScratchCase(unittest.TestCase):
    """A TestCase with self.tmp, a fresh directory per test that is removed afterwards."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="evt_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)


def run(cmd: Sequence[str], env: Optional[Dict[str, str]] = None, timeout: float = 600.0,
        cwd: Optional[Path] = None, stdin: Optional[str] = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.update({"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": "",
              "TOKENIZERS_PARALLELISM": "false"})
    if env:
        e.update(env)
    return subprocess.run([str(c) for c in cmd], env=e, capture_output=True, text=True, timeout=timeout,
                          cwd=str(cwd) if cwd else None, input=stdin)


def run_py(script: str, args: Sequence[str], env: Optional[Dict[str, str]] = None, timeout: float = 600.0,
           python: Optional[str] = None) -> subprocess.CompletedProcess:
    return run([python or PY, str(HERE / script), *args], env=env, timeout=timeout)


def in_process(main, argv: Sequence[str]):
    """Call a module's main(argv) with stderr captured: (exit code, stderr text). SystemExit with a message is code 1
    (the interpreter's rule) and its message lands in the text, like the shell would show it."""
    buf = io.StringIO()
    code: object
    with contextlib.redirect_stderr(buf):
        try:
            code = main(list(argv))
        except SystemExit as e:
            code = e.code
            if isinstance(code, str):
                print(code, file=buf)
                code = 1
    return (0 if code is None else code), buf.getvalue()


# ----------------------------------------------------------------------------------------------------------
class FakeServer:
    """fake_server.py on a free port. Use as a context manager; control() and stats() talk to /_fake/*."""

    def __init__(self, extra_args: Sequence[str] = (), env: Optional[Dict[str, str]] = None, ready_s: float = 20.0,
                 wait: bool = True, log_path: Optional[Path] = None) -> None:
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        e = dict(os.environ)
        for k in list(e):
            if k.startswith("FAKE_") or k in ("SGLANG_PLUGINS", "NEGEIG_GATES"):
                del e[k]
        e.update(env or {})
        e["PYTHONDONTWRITEBYTECODE"] = "1"
        self.log = open(log_path, "w") if log_path else tempfile.TemporaryFile("w+")
        self.proc = subprocess.Popen([PY, str(HERE / "fake_server.py"), "--port", str(self.port), *extra_args],
                                     env=e, stdout=self.log, stderr=subprocess.STDOUT)
        if wait:
            self.wait_ready(ready_s)

    def wait_ready(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            if self.proc.poll() is not None:
                raise RuntimeError(f"fake server exited with {self.proc.returncode}")
            try:
                with urllib.request.urlopen(self.base + "/v1/models", timeout=2) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            time.sleep(0.1)
        raise RuntimeError("fake server did not become ready")

    def control(self, **kw) -> dict:
        req = urllib.request.Request(self.base + "/_fake/control", data=json.dumps(kw).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def stats(self) -> dict:
        with urllib.request.urlopen(self.base + "/_fake/stats", timeout=10) as r:
            return json.loads(r.read())

    def count(self, path: str) -> int:
        return int(self.stats()["counts"].get(path, 0))

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
        try:
            self.log.close()
        except OSError:
            pass

    def __enter__(self) -> "FakeServer":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# ----------------------------------------------------------------------------------------------------------
# synthetic datasets in the normalized format
def write_jsonl(path: Path, rows: List[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def mmlupro_rows(n: int = 24) -> List[dict]:
    cats = ["math", "law", "biology", "history"]
    rows = []
    for i in range(n):
        k = 4 + i % 7  # 4..10 options
        rows.append({"id": str(70000 + i), "category": cats[i % len(cats)],
                     "question": f"Synthetic question number {i}: which statement about item {i * 7} holds?",
                     "options": [f"option {j} about thing {i}-{j}" for j in range(k)],
                     "answer_index": (i * 3) % k, "src": "synthetic"})
    return rows


def gmmlu_rows(n_test: int = 24, n_dev: int = 10) -> "tuple[List[dict], List[dict]]":
    subjects = ["anatomy", "astronomy", "virology"]
    dev, test = [], []
    for i in range(n_dev):
        s = subjects[i % 3]
        dev.append({"id": f"{s}/dev/{i}", "subject": s, "question": f"dev question {i} שלום",
                    "options": [f"dev {i} option {x}" for x in "abcd"], "answer": "ABCD"[i % 4]})
    for i in range(n_test):
        s = subjects[i % 3]
        test.append({"id": f"{s}/test/{i}", "subject": s, "question": f"test question {i} בדיקה",
                     "options": [f"test {i} option {x}" for x in "abcd"], "answer": "ABCD"[(i * 5) % 4]})
    return test, dev


def humaneval_rows(n: int = 8) -> List[dict]:
    rows = []
    for i in range(n):
        rows.append({"id": f"HumanEval/{i}", "entry_point": f"f{i}",
                     "prompt": f"def f{i}(x):\n    \"\"\"return x plus {i}\"\"\"\n",
                     "canonical_solution": f"    return x + {i}\n",
                     "test": f"def check(candidate):\n    assert candidate(1) == {1 + i}\n    assert candidate(10) == {10 + i}\n"})
    return rows


def write_data_dir(d: Path, mmlupro: int = 24, gm_test: int = 24, gm_dev: int = 10, he: int = 8) -> Dict[str, str]:
    """The four normalized files an eval run reads, plus the sha256 of each."""
    test, dev = gmmlu_rows(gm_test, gm_dev)
    return {"mmlupro_test.jsonl": write_jsonl(d / "mmlupro_test.jsonl", mmlupro_rows(mmlupro)),
            "gmmlu_he_test.jsonl": write_jsonl(d / "gmmlu_he_test.jsonl", test),
            "gmmlu_he_dev.jsonl": write_jsonl(d / "gmmlu_he_dev.jsonl", dev),
            "humaneval_test.jsonl": write_jsonl(d / "humaneval_test.jsonl", humaneval_rows(he))}
