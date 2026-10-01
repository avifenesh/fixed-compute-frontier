#!/usr/bin/env python3
"""IFEval (google/IFEval, 541 prompts) through the served model, scored by the official google-research harness.

Chat, the prompt as the only user message, temperature 0, max_tokens 4096, thinking off. Scoring is the pinned
`instruction_following_eval` package (strict and loose, prompt level and instruction level), unchanged except that the
langdetect seed is fixed to 0 so the same response always gets the same verdict (the harness leaves langdetect
unseeded, so two scorings of one response can differ on short or mixed-language text; arm and base need one verdict
per response to be comparable). The package files are checked against their pinned sha256 before they are imported.

Gate: ifeval_strict, prompt-level strict accuracy, band 2 points. Reported beside it: ifeval_loose (prompt level) and
both instruction-level rows (834 instructions, resampled by prompt in compare_general.py).

    python ev_ifeval.py --arm base --out RUN/base --data evals/data [--stride 2]

Needs nltk, langdetect, immutabledict, absl-py (pins.json ifeval_harness.python_deps) and the nltk data punkt and
punkt_tab. NLTK_DATA is used when set, else <data>/nltk_data when that exists (prepare_box.sh nltk puts it there).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalcommon import (EXIT_USAGE, THINK_OFF, EvalRun, add_common_args, chat_text, eval_pins, finish,  # noqa: E402
                        load_pins, log, metric, read_jsonl, resolve_out, sha256_file)

MAX_TOKENS = 4096
_SCORE_LOCK = threading.Lock()


class Harness:
    """The imported official package plus the parsed input examples."""

    def __init__(self, data_dir: Path, pins: dict, verify: bool = True) -> None:
        pkg_dir = data_dir / "ifeval_harness" / "instruction_following_eval"
        want = pins["ifeval_harness"]["files"]
        self.file_sha256: Dict[str, str] = {}
        for name, sha in want.items():
            p = pkg_dir / name
            if not p.is_file():
                raise SystemExit(f"{p} is missing: run prepare_box.sh data (ifeval_harness) first")
            got = sha256_file(p)
            self.file_sha256[name] = got
            if verify and got != sha:
                raise SystemExit(f"{p}: sha256 {got[:12]} is not the pinned {sha[:12]}")
        nltk_dir = os.environ.get("NLTK_DATA") or (str(data_dir / "nltk_data") if (data_dir / "nltk_data").is_dir() else "")
        if nltk_dir:
            os.environ["NLTK_DATA"] = nltk_dir
        sys.path.insert(0, str(data_dir / "ifeval_harness"))
        try:
            import langdetect  # noqa: F401
            import nltk  # noqa: F401
            from langdetect import DetectorFactory
            from instruction_following_eval import evaluation_lib
        except ImportError as e:
            raise SystemExit(f"IFEval harness import failed ({e}). Needs nltk, langdetect, immutabledict, absl-py; "
                             "see prepare_box.sh evals") from e
        if nltk_dir and nltk_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_dir)
        DetectorFactory.seed = 0
        langdetect.detect("hello world, this primes the profile load")  # load profiles once, before any thread
        self.lib = evaluation_lib
        self.versions = {"nltk": nltk.__version__, "langdetect": getattr(langdetect, "__version__", "?")}
        rows = read_jsonl(data_dir / "ifeval_input_data.jsonl")
        self.examples = evaluation_lib.read_prompt_list(str(data_dir / "ifeval_input_data.jsonl"))
        if len(rows) != len(self.examples):
            raise SystemExit("IFEval input rows differ from parsed examples")
        self.by_key = {str(e.key): e for e in self.examples}
        if len(self.by_key) != len(self.examples):
            raise SystemExit("IFEval keys are not unique")
        if len({e.prompt for e in self.examples}) != len(self.examples):
            raise SystemExit("IFEval prompts are not unique (the harness keys responses by prompt)")

    def score(self, key: str, response: str) -> dict:
        ex = self.by_key[key]
        with _SCORE_LOCK:
            strict = self.lib.test_instruction_following_strict(ex, {ex.prompt: response})
            loose = self.lib.test_instruction_following_loose(ex, {ex.prompt: response})
        return {"strict": [int(x) for x in strict.follow_instruction_list],
                "loose": [int(x) for x in loose.follow_instruction_list]}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap, workers=96)
    a = ap.parse_args(argv)
    resolve_out(a)
    pins = load_pins()
    data = Path(a.data)
    inp = data / "ifeval_input_data.jsonl"
    if not inp.is_file():
        print(f"{inp} is missing: run prepare_box.sh data first", file=sys.stderr)
        return EXIT_USAGE
    want = pins["datasets"]["ifeval"]
    got = sha256_file(inp)
    if got != want["input"]["sha256"] and not a.allow_unpinned:
        print(f"{inp}: sha256 {got[:12]} is not the pinned {want['input']['sha256'][:12]}", file=sys.stderr)
        return EXIT_USAGE
    h = Harness(data, pins, verify=not a.allow_unpinned)
    if not a.allow_unpinned and len(h.examples) != want["rows"]:
        print(f"ifeval: {len(h.examples)} rows, pinned {want['rows']}", file=sys.stderr)
        return EXIT_USAGE
    n_inst = {k: len(e.instruction_id_list) for k, e in h.by_key.items()}
    config = {"eval": "ifeval", "v": 1, "protocol": "chat", "max_tokens": MAX_TOKENS, "temperature": 0,
              "chat_template_kwargs": THINK_OFF, "langdetect_seed": 0,
              "harness_commit": pins["ifeval_harness"]["commit"], "harness_files": h.file_sha256}
    run = EvalRun("ifeval", a, config, eval_pins(a, pins, "ifeval", {"ifeval_input_data.jsonl": got}),
                  [str(e.key) for e in h.examples])
    log(f"[ifeval] harness {h.versions}")

    def work(key: str) -> dict:
        ex = h.by_key[key]
        res = run.server.chat([{"role": "user", "content": ex.prompt}], max_tokens=MAX_TOKENS, temperature=0,
                              chat_template_kwargs=THINK_OFF)
        text, fin = chat_text(res)
        sc = h.score(key, text)
        return {"response": text, "finish": fin, "strict": sc["strict"], "loose": sc["loose"]}

    status = run.execute(work)

    def build():
        recs = run.records()
        strict = {r["id"]: int(all(r["strict"])) for r in recs}
        loose = {r["id"]: int(all(r["loose"])) for r in recs}
        strict_i = {f"{r['id']}:{i}": v for r in recs for i, v in enumerate(r["strict"])}
        loose_i = {f"{r['id']}:{i}": v for r in recs for i, v in enumerate(r["loose"])}
        full_i = sum(n_inst.values())
        by_type: Dict[str, List[int]] = {}
        for r in recs:
            for iid, v in zip(h.by_key[r["id"]].instruction_id_list, r["strict"]):
                by_type.setdefault(iid.split(":")[0], []).append(v)
        diag = {"n_truncated_at_budget": sum(1 for r in recs if r["finish"] == "length"),
                "n_empty_response": sum(1 for r in recs if not r["response"].strip()),
                "strict_by_instruction_group": {k: [sum(v), len(v)] for k, v in sorted(by_type.items())},
                "harness_versions": h.versions}
        return ({"ifeval_strict": metric(strict, n_full=len(h.examples)),
                 "ifeval_loose": metric(loose, n_full=len(h.examples)),
                 "ifeval_strict_inst": metric(strict_i, level="instruction", n_full=full_i),
                 "ifeval_loose_inst": metric(loose_i, level="instruction", n_full=full_i)}, diag)

    return finish(run, status, build)


if __name__ == "__main__":
    sys.exit(main())
