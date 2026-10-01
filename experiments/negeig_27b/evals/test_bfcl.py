#!/usr/bin/env python3
"""CPU tests of the BFCL leg: the result/score parsers, the gorilla patch and a dry run through the real bfcl CLI.

Three layers, each skipped on its own when its input is missing:

  synthetic    select_per_category, classify_result, read_rows, scan_results, parse_scores on hand-made files
               (standard library only, always runs)
  real run     the same parsers on the Hebrew lane's finished base run (4,441 items, 17 categories, 44 context
               overflows), read only; needs the BFCL assets (testlib.bfcl_assets)
  real bfcl    bfcl_patch.py on a clean `git archive` of the pinned commit, then ev_bfcl.py end to end against
               fake_server.py with the real bfcl-venv, the real tokenizer and the patched tree on PYTHONPATH: a first
               pass, a resume, a transport-error pass, a context-overflow pass. This is the same call run_general.sh
               makes on the box, minus the GPU.

    python -m unittest test_bfcl -v          (run_tests.sh does this with the CPU caps)
"""
from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import testlib
from testlib import HERE, PY, FakeServer, ScratchCase, run

import ev_bfcl as eb
from evalcommon import BFCL_FULL, read_json, sha256_file

KEY = "negeig/qwen3.8-27b-FC"
PINS = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))["bfcl"]
PATCH = HERE / "bfcl" / "bfcl_patch.py"
ASSETS = testlib.bfcl_assets()
IRRELEVANCE = ("irrelevance", "live_irrelevance")
IMPORT_ANCHOR = "from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler"
IMPORT_NEW = "from bfcl_eval.model_handler.local_inference.tiyuvta_hermes import TiyuvtaHermesHandler"


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")


def ok_row(i: str) -> dict:
    return {"id": i, "result": f"answer {i}", "input_token_count": 5, "output_token_count": 3, "latency": 0.1}


def err_row(i: str, msg: str = "Error code: 500 - {'error': 'boom'}") -> dict:
    return {"id": i, "result": f"{eb.ERROR_PREFIX}: {msg}", "input_token_count": 0, "output_token_count": 0}


CONTEXT_MSG = ("Error code: 400 - {'object': 'error', 'message': \"The input (35565 tokens) is longer than the model\\'s "
               "context length (32768 tokens).\", 'type': 'BadRequestError', 'param': None, 'code': 400}")


# ----------------------------------------------------------------------------------------------------------
class TestSelection(ScratchCase):
    def test_layout_paths(self):
        root = Path("/r")
        self.assertEqual(eb.group_of("live_simple"), "live")
        self.assertEqual(eb.group_of("multi_turn_base"), "multi_turn")
        self.assertEqual(eb.group_of("simple_python"), "non_live")
        self.assertEqual(eb.group_of("irrelevance"), "non_live")
        self.assertEqual(eb.key_dir(KEY), "negeig_qwen3.8-27b-FC")
        self.assertEqual(eb.result_file(root, KEY, "live_simple"),
                         Path("/r/result/negeig_qwen3.8-27b-FC/live/BFCL_v4_live_simple_result.json"))
        self.assertEqual(eb.score_file(root, KEY, "multi_turn_miss_func"),
                         Path("/r/score/negeig_qwen3.8-27b-FC/multi_turn/BFCL_v4_multi_turn_miss_func_score.json"))

    def test_the_gated_categories_are_the_seventeen_of_the_plan(self):
        self.assertEqual(len(eb.CATEGORIES), 17)
        self.assertEqual(sum(BFCL_FULL.values()), 4441)
        self.assertEqual(set(eb.CATEGORIES), set(BFCL_FULL))
        self.assertFalse([c for c in eb.CATEGORIES if "web_search" in c or "memory" in c or "format_sens" in c])

    def test_select_per_category(self):
        full = {"a": [f"a{i}" for i in range(10)], "b": ["b0", "b1", "b2"]}
        self.assertEqual(eb.select_per_category(full, 1, 0, 0), full)
        self.assertEqual(eb.select_per_category(full, 4, 1, 0), {"a": ["a1", "a5", "a9"], "b": ["b1"]})
        self.assertEqual(eb.select_per_category(full, 1, 0, 2), {"a": ["a0", "a1"], "b": ["b0", "b1"]})
        self.assertEqual(eb.select_per_category(full, 3, 2, 2), {"a": ["a2", "a5"], "b": ["b2"]})
        self.assertEqual(eb.select_per_category({"a": []}, 2, 1, 5), {"a": []})

    def test_the_offsets_of_one_stride_partition_the_set(self):
        full = {"a": [f"a{i}" for i in range(23)], "b": [f"b{i}" for i in range(7)]}
        for stride in (2, 3, 5):
            got = [i for off in range(stride) for ids in eb.select_per_category(full, stride, off, 0).values() for i in ids]
            self.assertEqual(sorted(got), sorted(i for ids in full.values() for i in ids), stride)

    def test_bad_subset_arguments_stop(self):
        for stride, offset in ((0, 0), (-1, 0), (3, 3), (3, -1), (1, 1)):
            with self.subTest(stride=stride, offset=offset):
                with self.assertRaises(SystemExit):
                    eb.select_per_category({"a": ["x"]}, stride, offset, 0)

    def test_category_ids_keep_the_file_order_and_skip_blank_lines(self):
        write_rows(self.tmp / "BFCL_v4_simple_python.json", [{"id": "simple_python_0"}, "", {"id": "simple_python_2"},
                                                             {"id": "simple_python_1"}])
        self.assertEqual(eb.category_ids(self.tmp, "simple_python"),
                         ["simple_python_0", "simple_python_2", "simple_python_1"])

    def test_a_missing_data_file_names_the_flag(self):
        with self.assertRaises(SystemExit) as cm:
            eb.category_ids(self.tmp, "simple_python")
        self.assertIn("--bfcl-data", str(cm.exception))


class TestClassify(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(eb.classify_result("fine"), "ok")
        self.assertEqual(eb.classify_result([["f(x=1)"]]), "ok")
        self.assertEqual(eb.classify_result(None), "ok")
        self.assertEqual(eb.classify_result(f"{eb.ERROR_PREFIX}: Error code: 500 - x"), "error")
        self.assertEqual(eb.classify_result(f"{eb.ERROR_PREFIX}: timed out"), "error")
        self.assertEqual(eb.classify_result(f"{eb.ERROR_PREFIX}: {CONTEXT_MSG}"), "context")

    def test_the_context_message_in_every_escaping_bfcl_may_store(self):
        for apos in ("'", "\\'", "\\\\'", "’"):
            with self.subTest(apos=apos):
                msg = f"{eb.ERROR_PREFIX}: x is longer than the model{apos}s context length (32768 tokens)."
                self.assertEqual(eb.classify_result(msg), "context")

    def test_only_a_leading_error_prefix_counts(self):
        self.assertEqual(eb.classify_result("the text says: Error during inference: boom"), "ok")
        self.assertEqual(eb.classify_result("it is longer than the model's context length"), "ok")
        self.assertEqual(eb.classify_result(f"{eb.ERROR_PREFIX}: the answer is longer than the models context"), "error")


class TestReadRows(ScratchCase):
    def test_a_cut_line_and_garbage_are_dropped_and_counted(self):
        p = self.tmp / "r.json"
        write_rows(p, [ok_row("a"), "", '{"id": "b", "result": "cut here', "not json", '{"no_id": 1}', '[1, 2]', ok_row("c")])
        rows, bad = eb.read_rows(p)
        self.assertEqual([r["id"] for r in rows], ["a", "c"])
        self.assertEqual(bad, 4)  # the cut line, the text, the row without an id, the list

    def test_a_missing_file_is_empty(self):
        self.assertEqual(eb.read_rows(self.tmp / "none.json"), ([], 0))

    def test_ids_are_strings_and_the_raw_line_is_kept(self):
        p = self.tmp / "r.json"
        write_rows(p, ['{"id": 7, "result": "x", "latency": 1}'])
        rows, _ = eb.read_rows(p)
        self.assertEqual(rows[0]["id"], "7")
        self.assertEqual(rows[0]["_raw"], '{"id": 7, "result": "x", "latency": 1}')


# ----------------------------------------------------------------------------------------------------------
class TestScanResults(ScratchCase):
    def setUp(self):
        super().setUp()
        self.sel = {"simple_python": ["a", "b", "c"], "irrelevance": ["i1", "i2"]}
        self.sp = eb.result_file(self.tmp, KEY, "simple_python")
        self.ir = eb.result_file(self.tmp, KEY, "irrelevance")
        self.receipts = self.tmp / "receipts" / "bfcl.errors.jsonl"

    def scan(self, counts=None, passes=3):
        counts = {} if counts is None else counts
        return eb.scan_results(self.tmp, KEY, self.sel, counts, passes, self.receipts), counts

    def test_a_complete_clean_run_is_untouched(self):
        write_rows(self.sp, [ok_row("a"), ok_row("b"), ok_row("c")])
        write_rows(self.ir, [ok_row("i1"), ok_row("i2")])
        before = (self.sp.read_bytes(), self.ir.read_bytes())
        sc, counts = self.scan()
        self.assertEqual(sc["missing"], {})
        self.assertEqual(set(sc["kinds"].values()), {"ok"})
        self.assertEqual((sc["stripped"], sc["repaired"], sc["accepted"], counts), (0, 0, {}, {}))
        self.assertEqual((self.sp.read_bytes(), self.ir.read_bytes()), before)
        self.assertFalse(self.receipts.exists())

    def test_missing_ids_and_missing_files(self):
        write_rows(self.sp, [ok_row("a"), ok_row("c")])
        sc, _ = self.scan()
        self.assertEqual(sc["missing"], {"simple_python": ["b"], "irrelevance": ["i1", "i2"]})

    def test_a_transport_error_is_stripped_while_it_has_tries_left_then_accepted(self):
        counts: dict = {}
        write_rows(self.sp, [ok_row("a"), err_row("b"), ok_row("c")])
        write_rows(self.ir, [ok_row("i1"), ok_row("i2")])
        for attempt in (1, 2):
            sc, _ = self.scan(counts, passes=3)
            self.assertEqual(sc["missing"], {"simple_python": ["b"]}, attempt)
            self.assertEqual((sc["stripped"], counts["b"]["n"], sc["accepted"]), (1, attempt, {}))
            self.assertEqual([r["id"] for r in eb.read_rows(self.sp)[0]], ["a", "c"])
            with open(self.sp, "a") as f:  # the generate pass writes the error again
                f.write(json.dumps(err_row("b")) + "\n")
        sc, _ = self.scan(counts, passes=3)
        self.assertEqual(sc["missing"], {})
        self.assertEqual(sc["kinds"]["b"], "accepted_error")
        self.assertEqual(list(sc["accepted"]), ["b"])
        self.assertIn("boom", sc["accepted"]["b"])
        self.assertEqual(sorted(r["id"] for r in eb.read_rows(self.sp)[0]), ["a", "b", "c"])
        receipt = [json.loads(ln) for ln in self.receipts.read_text().splitlines()]
        self.assertEqual([(r["id"], r["try"], r["of"]) for r in receipt], [("b", 1, 3), ("b", 2, 3)])

    def test_one_pass_accepts_at_once(self):
        write_rows(self.sp, [ok_row("a"), err_row("b"), ok_row("c")])
        write_rows(self.ir, [ok_row("i1"), ok_row("i2")])
        sc, _ = self.scan(passes=1)
        self.assertEqual((sc["missing"], sc["stripped"], list(sc["accepted"])), ({}, 0, ["b"]))

    def test_a_context_overflow_is_a_scored_failure_not_a_retry(self):
        write_rows(self.sp, [ok_row("a"), {"id": "b", "result": f"{eb.ERROR_PREFIX}: {CONTEXT_MSG}"}, ok_row("c")])
        write_rows(self.ir, [ok_row("i1"), ok_row("i2")])
        counts: dict = {}
        sc, _ = self.scan(counts)
        self.assertEqual((sc["missing"], sc["stripped"], sc["accepted"], counts), ({}, 0, {}, {}))
        self.assertEqual(sc["kinds"]["b"], "context")
        self.assertEqual(len(eb.read_rows(self.sp)[0]), 3)

    def test_duplicates_collapse_to_the_last_row_and_the_file_is_repaired(self):
        newer = dict(ok_row("a"), result="newer")
        write_rows(self.sp, [ok_row("a"), ok_row("b"), newer, ok_row("c")])
        write_rows(self.ir, [ok_row("i1"), ok_row("i2")])
        sc, _ = self.scan()
        self.assertEqual((sc["missing"], sc["repaired"]), ({}, 1))
        rows = eb.read_rows(self.sp)[0]
        self.assertEqual(sorted(r["id"] for r in rows), ["a", "b", "c"])
        self.assertEqual(json.loads([r for r in rows if r["id"] == "a"][0]["_raw"])["result"], "newer")

    def test_a_cut_last_line_is_dropped_and_its_id_regenerated(self):
        write_rows(self.sp, [ok_row("a"), ok_row("b"), '{"id": "c", "result": "half a li'])
        write_rows(self.ir, [ok_row("i1"), ok_row("i2")])
        sc, _ = self.scan()
        self.assertEqual((sc["missing"], sc["repaired"]), ({"simple_python": ["c"]}, 1))
        self.assertEqual(eb.read_rows(self.sp)[1], 0)  # the file no longer holds the cut line

    def test_ids_outside_the_selection_are_removed_and_never_missing(self):
        write_rows(self.sp, [ok_row("a"), ok_row("b"), ok_row("c"), ok_row("zzz")])
        write_rows(self.ir, [ok_row("i1"), ok_row("i2")])
        sc, _ = self.scan()
        self.assertEqual(sc["missing"], {})
        self.assertNotIn("zzz", sc["kinds"])
        self.assertEqual(sorted(r["id"] for r in eb.read_rows(self.sp)[0]), ["a", "b", "c"])

    def test_the_stored_message_is_cut_to_240_characters(self):
        write_rows(self.sp, [ok_row("a"), err_row("b", "x" * 1000), ok_row("c")])
        write_rows(self.ir, [ok_row("i1"), ok_row("i2")])
        _, counts = self.scan()
        self.assertEqual(len(counts["b"]["msg"]), 240)

    def test_counts_survive_between_passes_through_the_state_file(self):
        write_rows(self.sp, [ok_row("a"), err_row("b"), ok_row("c")])
        write_rows(self.ir, [ok_row("i1"), ok_row("i2")])
        _, counts = self.scan(passes=2)
        again = json.loads(json.dumps(counts))
        with open(self.sp, "a") as f:
            f.write(json.dumps(err_row("b")) + "\n")
        sc, _ = self.scan(again, passes=2)
        self.assertEqual(list(sc["accepted"]), ["b"])


# ----------------------------------------------------------------------------------------------------------
class TestParseScores(ScratchCase):
    def setUp(self):
        super().setUp()
        self.sel = {"simple_python": ["a", "b", "c"], "irrelevance": ["i1"], "simple_java": []}

    def put(self, cat, total, correct, fails):
        p = eb.score_file(self.tmp, KEY, cat)
        write_rows(p, [{"accuracy": correct / max(total, 1), "correct_count": correct, "total_count": total},
                       *[{"id": i, "valid": False, "error": {"error_message": "x"}} for i in fails]])

    def test_correct_is_the_selection_minus_the_listed_failures(self):
        self.put("simple_python", 3, 2, ["b"])
        self.put("irrelevance", 1, 1, [])
        out = eb.parse_scores(self.tmp, KEY, self.sel)
        self.assertEqual(out, {"simple_python": {"a": 1, "b": 0, "c": 1}, "irrelevance": {"i1": 1}})

    def test_an_empty_selection_needs_no_score_file(self):
        self.put("simple_python", 3, 3, [])
        self.put("irrelevance", 1, 0, ["i1"])
        self.assertNotIn("simple_java", eb.parse_scores(self.tmp, KEY, self.sel))

    def test_the_summary_must_agree(self):
        self.put("irrelevance", 1, 1, [])
        for total, correct, why in ((4, 3, "total"), (3, 3, "correct")):
            with self.subTest(why=why):
                self.put("simple_python", total, correct, ["b"])
                with self.assertRaises(SystemExit) as cm:
                    eb.parse_scores(self.tmp, KEY, self.sel)
                self.assertIn("simple_python", str(cm.exception))

    def test_a_failure_outside_the_selection_stops(self):
        self.put("irrelevance", 1, 1, [])
        self.put("simple_python", 3, 1, ["b", "elsewhere"])
        with self.assertRaises(SystemExit) as cm:
            eb.parse_scores(self.tmp, KEY, self.sel)
        self.assertIn("outside the selection", str(cm.exception))

    def test_a_missing_or_empty_score_file_stops(self):
        self.put("irrelevance", 1, 1, [])
        with self.assertRaises(SystemExit) as cm:
            eb.parse_scores(self.tmp, KEY, self.sel)
        self.assertIn("did not score simple_python", str(cm.exception))
        p = eb.score_file(self.tmp, KEY, "simple_python")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n\n")
        with self.assertRaises(SystemExit) as cm:
            eb.parse_scores(self.tmp, KEY, self.sel)
        self.assertIn("empty", str(cm.exception))


# ----------------------------------------------------------------------------------------------------------
def clean_gorilla(dest: Path) -> Path:
    """The pinned commit's berkeley-function-call-leaderboard in dest (git archive, so the owner's working-tree edits
    in the lane checkout are not copied). Returns the gorilla root (the --gorilla argument of bfcl_patch.py)."""
    repo = ASSETS / "gorilla"
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != PINS["commit"]:
        raise unittest.SkipTest(f"{repo} is at {head[:12]}, the pin is {PINS['commit'][:12]}")
    dest.mkdir(parents=True, exist_ok=True)
    arc = subprocess.Popen(["git", "-C", str(repo), "archive", "HEAD", "berkeley-function-call-leaderboard"],
                           stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=arc.stdout, check=True)
    arc.stdout.close()
    if arc.wait() != 0:
        raise RuntimeError("git archive failed")
    return dest


def patch(gorilla: Path, *args: str) -> subprocess.CompletedProcess:
    return run([PY, str(PATCH), "--gorilla", str(gorilla), *args])


@unittest.skipUnless(ASSETS, "no BFCL assets (EVAL_TEST_BFCL)")
class TestRealRunParsers(unittest.TestCase):
    """The parsers on the Hebrew lane's finished base run. Read only: nothing here writes under bfcl-runs."""

    RUN = "tiyuvta/qwen3.8-27b-base-FC"

    @classmethod
    def setUpClass(cls):
        cls.root = ASSETS / "bfcl-runs"
        if not eb.result_file(cls.root, cls.RUN, "simple_python").is_file():
            raise unittest.SkipTest("the real base run is not in bfcl-runs")
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="evt_bfclreal_"))
        cls.addClassCleanup(shutil.rmtree, cls.tmpdir, True)
        cls.data = clean_gorilla(cls.tmpdir / "g") / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
        cls.full = {c: eb.category_ids(cls.data, c) for c in eb.CATEGORIES}

    def test_the_pinned_counts_are_the_real_data_counts(self):
        self.assertEqual({c: len(v) for c, v in self.full.items()}, BFCL_FULL)
        self.assertEqual(sum(len(v) for v in self.full.values()), 4441)

    def test_every_result_row_reads_and_the_44_context_overflows_are_found(self):
        kinds = {"ok": 0, "context": 0, "error": 0}
        per_cat = {}
        for c in eb.CATEGORIES:
            rows, bad = eb.read_rows(eb.result_file(self.root, self.RUN, c))
            self.assertEqual(bad, 0, c)
            self.assertEqual(sorted(r["id"] for r in rows), sorted(self.full[c]), c)
            per_cat[c] = 0
            for r in rows:
                k = eb.classify_result(r["result"])
                kinds[k] += 1
                per_cat[c] += k == "context"
        self.assertEqual(kinds["ok"] + kinds["context"], 4441)
        self.assertEqual(kinds["error"], 0)
        self.assertEqual(kinds["context"], 44)
        self.assertEqual({c: n for c, n in per_cat.items() if n}, {"multi_turn_long_context": 44})

    def test_the_score_files_parse_and_agree_with_their_summary_lines(self):
        scored = eb.parse_scores(self.root, self.RUN, self.full)
        self.assertEqual({c: len(v) for c, v in scored.items()}, BFCL_FULL)
        self.assertEqual(sum(scored["multi_turn_long_context"].values()), 97)  # summary line: 97 of 200
        self.assertEqual(sum(len(v) for v in scored.values()), 4441)

    def test_scan_results_leaves_clean_real_rows_alone(self):
        sel = {}
        for c in eb.CATEGORIES:
            src = eb.result_file(self.root, self.RUN, c)
            dst = eb.result_file(self.tmpdir / "scan", self.RUN, c)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
                head = list(itertools.islice(f, 10))  # live_relevance has only 16 rows
                g.writelines(head)
            sel[c] = [json.loads(ln)["id"] for ln in head]
        before = {c: eb.result_file(self.tmpdir / "scan", self.RUN, c).read_bytes() for c in sel}
        sc = eb.scan_results(self.tmpdir / "scan", self.RUN, sel, {}, 3)
        self.assertEqual((sc["missing"], sc["stripped"], sc["repaired"], sc["accepted"]), ({}, 0, 0, {}))
        self.assertEqual({c: eb.result_file(self.tmpdir / "scan", self.RUN, c).read_bytes() for c in sel}, before)


# ----------------------------------------------------------------------------------------------------------
@unittest.skipUnless(ASSETS, "no BFCL assets (EVAL_TEST_BFCL)")
class TestGorillaPatch(ScratchCase):
    def tree(self) -> Path:
        return clean_gorilla(self.tmp / "g")

    def bfcl_dir(self, g: Path) -> Path:
        return g / "berkeley-function-call-leaderboard" / "bfcl_eval"

    def test_check_apply_reapply(self):
        g = self.tree()
        p = patch(g, "--check")
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertEqual(json.loads(p.stdout), {"model_config": "applied", "base_oss_handler": "applied", "handler": "missing"})
        p = patch(g, "--receipt", str(self.tmp / "receipt.json"))
        self.assertEqual(p.returncode, 0, p.stderr)
        d = self.bfcl_dir(g)
        handler = d / "model_handler" / "local_inference" / "tiyuvta_hermes.py"
        self.assertEqual(sha256_file(handler), PINS["handler_sha256"])
        mc = (d / "constants" / "model_config.py").read_text()
        self.assertEqual(mc.count(f'"{KEY}": ModelConfig('), 1)
        self.assertEqual(mc.count("TiyuvtaHermesHandler"), 2)  # the import and the entry
        bo = (d / "model_handler" / "local_inference" / "base_oss_handler.py").read_text()
        self.assertIn('int(os.getenv("BFCL_MAX_TOKENS", "4096"))', bo)
        self.assertNotIn("\n                4096,\n", bo)
        rc = read_json(self.tmp / "receipt.json")
        self.assertEqual((rc["commit"], rc["registry_key"]), (PINS["commit"], KEY))
        self.assertNotEqual(rc["sha256_before"]["model_config"], rc["sha256_after"]["model_config"])
        for f in (d / "constants" / "model_config.py", d / "model_handler" / "local_inference" / "base_oss_handler.py"):
            r = run([PY, "-m", "py_compile", str(f)])
            self.assertEqual(r.returncode, 0, r.stderr)
        snap = {f: f.read_bytes() for f in (handler, d / "constants" / "model_config.py")}
        p = patch(g)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual({f: f.read_bytes() for f in snap}, snap, "a second apply must change nothing")
        p = patch(g, "--check")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(set(json.loads(p.stdout).values()), {"patched"})

    def test_an_anchor_that_is_not_unique_fails_and_writes_nothing(self):
        g = self.tree()
        mc = self.bfcl_dir(g) / "constants" / "model_config.py"
        text = mc.read_text()
        self.assertEqual(text.count("\nlocal_inference_model_map = {\n"), 1)
        mc.write_text(text + "\nlocal_inference_model_map = {\n}\n")
        before = mc.read_bytes()
        p = patch(g)
        self.assertEqual(p.returncode, 1)
        self.assertIn("found 2 times, expected exactly 1", p.stderr)
        self.assertEqual(mc.read_bytes(), before)
        self.assertFalse((self.bfcl_dir(g) / "model_handler" / "local_inference" / "tiyuvta_hermes.py").exists())

    def test_a_missing_anchor_and_a_half_patched_file_fail_loudly(self):
        g = self.tree()
        bo = self.bfcl_dir(g) / "model_handler" / "local_inference" / "base_oss_handler.py"
        bo.write_text(bo.read_text().replace("\n                4096,\n", "\n                8192,\n"))
        p = patch(g)
        self.assertEqual(p.returncode, 1)
        self.assertIn("found 0 times", p.stderr)
        g2 = clean_gorilla(self.tmp / "g2")
        mc = self.bfcl_dir(g2) / "constants" / "model_config.py"
        text = mc.read_text()
        mc.write_text(text.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + "\n" + IMPORT_NEW, 1))
        p = patch(g2)
        self.assertEqual(p.returncode, 1)
        self.assertIn("half patched", p.stderr)

    def test_a_handler_that_is_not_the_pinned_one_is_refused(self):
        g = self.tree()
        other = self.tmp / "other_handler.py"
        other.write_text((HERE / "bfcl" / "tiyuvta_hermes.py").read_text() + "\n# edited\n")
        p = patch(g, "--handler", str(other))
        self.assertEqual(p.returncode, 1)
        self.assertIn("is not the pinned", p.stderr)
        self.assertFalse((self.bfcl_dir(g) / "model_handler" / "local_inference" / "tiyuvta_hermes.py").exists())

    def test_an_installed_handler_that_differs_is_refused_and_reported(self):
        g = self.tree()
        self.assertEqual(patch(g).returncode, 0)
        h = self.bfcl_dir(g) / "model_handler" / "local_inference" / "tiyuvta_hermes.py"
        h.write_text(h.read_text() + "\n# local edit\n")
        p = patch(g, "--check")
        self.assertEqual(p.returncode, 2)
        self.assertEqual(json.loads(p.stdout)["handler"], "different")
        p = patch(g)
        self.assertEqual(p.returncode, 1)
        self.assertIn("differs from the pinned handler", p.stderr)

    def test_not_a_gorilla_checkout(self):
        p = patch(self.tmp)
        self.assertEqual(p.returncode, 1)
        self.assertIn("bfcl_eval not found", p.stderr)


# ----------------------------------------------------------------------------------------------------------
@unittest.skipUnless(ASSETS and testlib.have_tokenizer(), "needs the BFCL assets and the Qwen3.8 tokenizer")
class TestBfclEndToEnd(ScratchCase):
    """ev_bfcl.py through the real bfcl CLI against the fake server: --limit 1 is 17 items, one per category."""

    @classmethod
    def setUpClass(cls):
        cls.shared = Path(tempfile.mkdtemp(prefix="evt_bfclclone_"))
        cls.addClassCleanup(shutil.rmtree, cls.shared, True)
        g = clean_gorilla(cls.shared / "g")
        r = patch(g)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        cls.pkg = g / "berkeley-function-call-leaderboard"
        cls.venv = ASSETS / "bfcl-venv"

    def args(self, out: Path, srv: FakeServer, *extra: str):
        return ["--arm", "t", "--out", str(out), "--base-url", srv.base, "--bfcl-venv", str(self.venv),
                "--tokenizer", testlib.TOKENIZER_DIR, "--bfcl-pythonpath", str(self.pkg),
                "--bfcl-data", str(self.pkg / "bfcl_eval" / "data"), "--workers", "4", "--limit", "1",
                "--ready-wait", "10", *extra]

    def go(self, out: Path, srv: FakeServer, *extra: str, timeout: float = 600.0):
        return testlib.run_py("ev_bfcl.py", self.args(out, srv, *extra), timeout=timeout)

    @staticmethod
    def assertRun(p, out: Path, code: int = 0) -> None:
        """The exit code, with the run's stderr and the bfcl logs in the message when it is not the wanted one."""
        if p.returncode == code:
            return
        logs = ""
        for name in ("bfcl-generate.log", "bfcl-evaluate.log"):
            f = out / "logs" / name
            if f.is_file():
                logs += f"\n--- {name}\n" + f.read_text(errors="replace")[-2500:]
        err = out / "receipts" / "bfcl.errors.jsonl"
        if err.is_file():
            logs += "\n--- bfcl.errors.jsonl\n" + err.read_text(errors="replace")[-1500:]
        raise AssertionError(f"exit {p.returncode}, wanted {code}\n{p.stderr[-2500:]}{logs}")

    def doc(self, out: Path) -> dict:
        return read_json(out / "results" / "bfcl.json")

    @staticmethod
    def scores(doc: dict) -> dict:
        return {k.split("/", 1)[1]: v["score"] for k, v in doc["metrics"].items()}

    def test_a_text_answer_scores_only_the_irrelevance_categories_and_a_rerun_is_a_no_op(self):
        out = self.tmp / "out"
        with FakeServer() as srv:
            p = self.go(out, srv)
            self.assertRun(p, out, 0)
            doc = self.doc(out)
            self.assertTrue(doc["complete"])
            self.assertEqual(doc["eval"], "bfcl")
            sc = self.scores(doc)
            self.assertEqual({c for c in eb.CATEGORIES if sc[c] == 1.0}, set(IRRELEVANCE) | set(), sc)
            self.assertEqual({c: sc[c] for c in eb.CATEGORIES if c not in IRRELEVANCE},
                             {c: 0.0 for c in eb.CATEGORIES if c not in IRRELEVANCE})
            self.assertAlmostEqual(sc["all"], 2 / 17)
            self.assertEqual(doc["metrics"]["bfcl/all"]["n"], 17)
            self.assertEqual(doc["metrics"]["bfcl/all"]["n_full"], 4441)
            self.assertEqual(doc["subset"]["limit"], 1)
            d = doc["diagnostics"]
            self.assertEqual((d["n_context_overflow"], d["n_accepted_transport_errors"]), (0, 0))
            self.assertEqual(set(d["installed"]["patched"].values()), {True})
            self.assertEqual(d["installed"]["handler_sha256"], PINS["handler_sha256"])
            self.assertEqual(d["per_category"]["irrelevance"], [1, 1])
            self.assertEqual(doc["pins"]["bfcl"]["commit"], PINS["commit"])
            self.assertEqual(doc["config"]["registry_key"], KEY)
            self.assertEqual(doc["config"]["chat_template_kwargs"], {"enable_thinking": False})
            # the wire: stock temperature, the stock cap, the template rendered locally with thinking off
            prm = srv.stats()["params"]["/v1/completions"]
            self.assertEqual((prm["temperature"], prm["max_tokens"]), (0.001, 4096))
            self.assertEqual(prm["model"], "qwen3.8-27b")
            self.assertEqual(srv.stats()["counts"].get("/v1/chat/completions", 0), 0)
            self.assertTrue(srv.stats()["tails"]["/v1/completions"].endswith("<think>\n\n</think>\n\n"),
                            repr(srv.stats()["tails"]["/v1/completions"]))
            n1 = srv.count("/v1/completions")
            self.assertGreaterEqual(n1, 17)
            result_files = sorted((out / "bfcl" / "result").rglob("*_result.json"))
            self.assertEqual(len(result_files), 17)
            snap = {f: f.read_bytes() for f in result_files}
            first = (out / "results" / "bfcl.json").read_bytes()
            p = self.go(out, srv)
            self.assertRun(p, out, 0)
            self.assertEqual(srv.count("/v1/completions"), n1, "a finished run must send no request")
            self.assertEqual({f: f.read_bytes() for f in result_files}, snap)
            self.assertEqual(self.scores(self.doc(out)), sc)
            self.assertTrue(first)

    def test_a_different_selection_on_the_same_out_is_refused_and_fresh_redoes_it(self):
        out = self.tmp / "out"

        def snapshot():
            return {f: f.read_bytes() for f in (out / "bfcl" / "result").rglob("*_result.json")}

        with FakeServer() as srv:
            self.assertRun(self.go(out, srv), out, 0)
            n1 = srv.count("/v1/completions")
            snap = snapshot()
            # the journal's identity is the selection: a wider subset (or the full set) needs its own --out or --fresh
            p = self.go(out, srv, "--limit", "2")
            self.assertRun(p, out, 1)
            self.assertIn("different configuration", p.stderr)
            self.assertIn("--fresh", p.stderr)
            self.assertEqual(srv.count("/v1/completions"), n1)
            self.assertEqual(snapshot(), snap)
            p = self.go(out, srv, "--limit", "2", "--fresh")
            self.assertRun(p, out, 0)
            self.assertEqual(self.doc(out)["metrics"]["bfcl/all"]["n"], 34)
            for f in (out / "bfcl" / "result").rglob("*_result.json"):
                self.assertEqual(len({r["id"] for r in eb.read_rows(f)[0]}), 2, f.name)

    def test_a_tool_call_scores_only_the_relevance_category(self):
        out = self.tmp / "out"
        with FakeServer() as srv:
            srv.control(bfcl_mode="call")
            p = self.go(out, srv)
            self.assertRun(p, out, 0)
            sc = self.scores(self.doc(out))
            self.assertEqual({c for c in eb.CATEGORIES if sc[c] == 1.0}, {"live_relevance"}, sc)
            self.assertAlmostEqual(sc["all"], 1 / 17)

    def test_transport_errors_are_regenerated_then_accepted_and_reported(self):
        out = self.tmp / "out"
        with FakeServer() as srv:
            srv.control(fail_status=400, fail_next=100000)  # a 400 is not retried by the client, so this is fast
            p = self.go(out, srv, "--error-passes", "2")
            self.assertRun(p, out, 3)
            self.assertFalse((out / "results" / "bfcl.json").exists(), "an incomplete run writes no result")
            rec = [json.loads(x) for x in (out / "receipts" / "bfcl.errors.jsonl").read_text().splitlines()]
            self.assertEqual({(r["try"], r["of"]) for r in rec}, {(1, 2)})
            self.assertEqual(len({r["id"] for r in rec}), 17)
            self.assertIn("injected", rec[0]["message"])
            left = sum(len(eb.read_rows(f)[0]) for f in (out / "bfcl" / "result").rglob("*_result.json"))
            self.assertEqual(left, 0, "stripped errors must not stay in the result files")
            st = read_json(out / "receipts" / "bfcl.status.json")
            self.assertEqual((st["complete"], st["n_failed"], st["done_now"]), (False, 17, 0))
            # the second pass is the last try: the errors stay and are scored
            p = self.go(out, srv, "--error-passes", "2")
            self.assertRun(p, out, 0)
            doc = self.doc(out)
            d = doc["diagnostics"]
            self.assertEqual(d["n_accepted_transport_errors"], 17)
            self.assertEqual(len(d["accepted_transport_error_ids_first20"]), 17)
            sc = self.scores(doc)
            # BFCL's checker reads an error as no call: right on the irrelevance categories, wrong elsewhere
            self.assertEqual({c for c in eb.CATEGORIES if sc[c] == 1.0}, set(IRRELEVANCE), sc)

    def test_a_request_over_the_window_is_a_scored_failure_not_a_retry(self):
        out = self.tmp / "out"
        with FakeServer() as srv:
            srv.control(ctx_chars=8000)
            p = self.go(out, srv)
            self.assertRun(p, out, 0)
            doc = self.doc(out)
            d = doc["diagnostics"]
            self.assertGreater(d["n_context_overflow"], 0)
            self.assertEqual(d["n_accepted_transport_errors"], 0)
            self.assertFalse((out / "receipts" / "bfcl.errors.jsonl").exists(), "an overflow is never regenerated")
            rows = [json.loads(x) for x in (out / "journal" / "bfcl.jsonl").read_text().splitlines()]
            ctx = [r for r in rows if r.get("kind") == "context"]
            self.assertEqual(len(ctx), d["n_context_overflow"])
            for r in ctx:
                if r["cat"] not in IRRELEVANCE:
                    self.assertEqual(r["ok"], 0, r)

    def test_an_unpatched_install_is_refused(self):
        raw = clean_gorilla(self.tmp / "raw") / "berkeley-function-call-leaderboard"
        out = self.tmp / "out"
        with FakeServer() as srv:
            a = self.args(out, srv)
            a[a.index("--bfcl-pythonpath") + 1] = str(raw)
            a[a.index("--bfcl-data") + 1] = str(raw / "bfcl_eval" / "data")
            p = testlib.run_py("ev_bfcl.py", a)
            self.assertRun(p, out, 2)
            self.assertIn("not patched", p.stderr)
            self.assertEqual(srv.count("/v1/completions"), 0)

    def test_a_handler_that_is_not_the_pinned_one_is_refused_unless_unpinned(self):
        pkg = clean_gorilla(self.tmp / "g") / "berkeley-function-call-leaderboard"
        self.assertEqual(patch(pkg.parent).returncode, 0)
        h = pkg / "bfcl_eval" / "model_handler" / "local_inference" / "tiyuvta_hermes.py"
        h.write_text(h.read_text() + "\n# edited on the box\n")
        out = self.tmp / "out"
        with FakeServer() as srv:
            a = self.args(out, srv)
            a[a.index("--bfcl-pythonpath") + 1] = str(pkg)
            a[a.index("--bfcl-data") + 1] = str(pkg / "bfcl_eval" / "data")
            p = testlib.run_py("ev_bfcl.py", a)
            self.assertRun(p, out, 2)
            self.assertIn("is not the pinned", p.stderr)
            self.assertEqual(srv.count("/v1/completions"), 0)
            p = testlib.run_py("ev_bfcl.py", [*a, "--allow-unpinned"])
            self.assertRun(p, out, 0)
            self.assertTrue(self.doc(out)["pins"]["unpinned"])

    def test_a_dead_server_exits_4_and_writes_no_result(self):
        out = self.tmp / "out"
        srv = FakeServer()
        base = srv.base
        srv.stop()
        a = self.args(out, srv)
        a[a.index("--ready-wait") + 1] = "3"
        p = testlib.run_py("ev_bfcl.py", a)
        self.assertRun(p, out, 4)
        self.assertIn(base.split("//")[1], p.stderr)
        self.assertFalse((out / "results" / "bfcl.json").exists())

    def test_fresh_discards_the_earlier_results(self):
        out = self.tmp / "out"
        with FakeServer() as srv:
            p = self.go(out, srv)
            self.assertRun(p, out, 0)
            n1 = srv.count("/v1/completions")
            p = self.go(out, srv, "--fresh")
            self.assertRun(p, out, 0)
            self.assertGreaterEqual(srv.count("/v1/completions"), 2 * n1 - 1)


if __name__ == "__main__":
    unittest.main()
