#!/usr/bin/env python3
"""CPU dry runs of the four logprob/chat evals (MMLU-Pro, Global-MMLU he, HumanEval, IFEval) against fake_server.py.

Each test starts the fake OpenAI-compatible server on a loopback port, runs the real ev_*.py entry point against it, and
compares the result file with an expectation recomputed from fake_server's pure answer functions, never read back from
the code under test. The red arms (a misaligned logprob list, batch noise, a wrong greedy letter, wrong answers, a dead
server, a failing request) must fail the way the box run needs them to fail. Also here: the Global-MMLU prompt
byte-equality oracle against the Hebrew lane's original block() source.

    python -m unittest test_evals -v          (run_tests.sh does this with the CPU caps)
"""
from __future__ import annotations

import ast
import collections
import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import testlib
from testlib import HERE, REAL_DATA, FakeServer, ScratchCase, in_process

import evalcommon as ec
import ev_gmmlu
import ev_humaneval
import ev_mmlupro
import fake_server as fs
import prepare_data

# The Hebrew lane's block(), copied verbatim from agentic-tool-evals-20260907/gmmlu/gmmlu-original.py (lines 33 to 40).
# test_original_source_is_still_the_copy checks the copy against that file when GMMLU_ORIGINAL points at it.
ORIGINAL_BLOCK_SRC = '''def block(row, with_answer):
    q = f"{row.question.strip()}\\n"
    for L, col in zip(LETTERS, ["option_a", "option_b", "option_c", "option_d"]):
        q += f"{L}. {str(getattr(row, col)).strip()}\\n"
    q += "Answer:"
    if with_answer:
        q += f" {row.answer.strip()}\\n\\n"
    return q
'''
ORIGINAL_FILE = Path(os.environ.get("GMMLU_ORIGINAL", "gmmlu-original.py"))
GMMLU_PARQUET = Path("/data/ai-ml/datasets/gmmlu")


def canonical_main(argv):
    """ev_humaneval.py --check-canonical in-process: (exit code, stderr text, the JSON receipt it printed on stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code, err = in_process(ev_humaneval.main, ["--check-canonical", *argv])
    return code, err, buf.getvalue()


def fast_server(a):
    """The real client with the retry sleeps removed, so a dead-server test takes milliseconds."""
    return ec.Server(a.base_url, a.served_model, a.api_key, a.timeout, a.attempts, a.breaker, sleep=lambda s: None)


class EvalCase(ScratchCase):
    """self.data (synthetic normalized files) and a helper that runs one entry point in-process against a server."""

    def setUp(self) -> None:
        super().setUp()
        self.data = self.tmp / "data"
        self.shas = testlib.write_data_dir(self.data)
        self.out = self.tmp / "out"

    def argv(self, srv, *extra, out=None, data=None):
        return ["--arm", "t", "--out", str(out or self.out), "--data", str(data or self.data), "--base-url", srv.base,
                "--allow-unpinned", "--workers", "4", "--timeout", "30", "--attempts", "2", "--breaker", "3", *extra]

    def go(self, main, srv, *extra, **kw):
        with mock.patch.object(ec, "make_server", fast_server):
            return in_process(main, self.argv(srv, *extra, **kw))

    def result(self, name, out=None):
        return json.loads((Path(out or self.out) / "results" / f"{name}.json").read_text())

    def status(self, name, out=None):
        return json.loads((Path(out or self.out) / "receipts" / f"{name}.status.json").read_text())

    def has_result(self, name, out=None):
        return (Path(out or self.out) / "results" / f"{name}.json").exists()


# ----------------------------------------------------------------------------------------------------------
@unittest.skipUnless(testlib.have_tokenizer(), "needs the Qwen3.8 tokenizer")
class TestMmluproDryRun(EvalCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ec.load_tokenizer(testlib.TOKENIZER_DIR)
        cls.lids = ev_mmlupro.letter_token_ids(cls.tok)
        cls.rows = testlib.mmlupro_rows(24)
        cls.exp_cloze, cls.exp_letter = {}, {}
        for r in cls.rows:
            sc = fs.option_scores(ev_mmlupro.cloze_context(cls.tok, r), ev_mmlupro.cloze_options(cls.tok, r))
            cls.exp_cloze[r["id"]] = int(ev_mmlupro.argmax_first(sc) == r["answer_index"])
            table = fs.letter_logprobs(ev_mmlupro.letter_prompt_ids(cls.tok, r), cls.lids)
            lp = [table[cls.lids[i]] for i in range(len(r["options"]))]
            cls.exp_letter[r["id"]] = int(ev_mmlupro.argmax_first(lp) == r["answer_index"])

    def server(self, **env):
        return FakeServer(["--special-ids", ",".join(str(i) for i in self.lids)], env=env)

    def run_main(self, srv, *extra, **kw):
        return self.go(ev_mmlupro.main, srv, "--tokenizer", testlib.TOKENIZER_DIR, *extra, **kw)

    def test_expectations_are_not_degenerate(self):
        # a run that always answered the same would pass the oracle comparison vacuously
        self.assertGreater(sum(self.exp_cloze.values()), 0)
        self.assertLess(sum(self.exp_cloze.values()), len(self.rows))
        self.assertGreater(sum(self.exp_letter.values()), 0)
        self.assertLess(sum(self.exp_letter.values()), len(self.rows))
        self.assertNotEqual(self.exp_cloze, self.exp_letter)

    def check_oracle(self, logprob_start):
        with self.server(FAKE_LOGPROB_START=str(logprob_start)) as srv:
            code, err = self.run_main(srv)
            self.assertEqual(code, 0, err)
            doc = self.result("mmlupro")
            self.assertEqual(doc["metrics"]["mmlupro_cloze"]["items"], self.exp_cloze)
            self.assertEqual(doc["metrics"]["mmlupro_letter"]["items"], self.exp_letter)
            self.assertEqual(doc["metrics"]["mmlupro_cloze"]["n_full"], 24)
            self.assertEqual(doc["diagnostics"]["letter_method_used"], {"ids": 24, "topk": 0})
            self.assertTrue(doc["pins"]["unpinned"])
            self.assertEqual(doc["pins"]["normalized_sha256"], {"mmlupro_test.jsonl": self.shas["mmlupro_test.jsonl"]})
            self.assertEqual(self.status("mmlupro")["complete"], True)
            self.assertGreater(srv.count("/generate"), 0)

    def test_oracle_when_logprobs_start_at_the_requested_position(self):
        self.check_oracle(0)

    def test_oracle_when_the_server_drops_the_first_logprob_position(self):
        self.check_oracle(1)

    def test_single_modes(self):
        with self.server() as srv:
            self.assertEqual(self.run_main(srv, "--modes", "cloze")[0], 0)
            m = self.result("mmlupro")["metrics"]
            self.assertEqual(list(m), ["mmlupro_cloze"])
            self.assertEqual(srv.count("/generate") > 0, True)
            out2 = self.tmp / "out2"
            self.assertEqual(self.run_main(srv, "--modes", "letter", out=out2)[0], 0)
            self.assertEqual(list(self.result("mmlupro", out2)["metrics"]), ["mmlupro_letter"])
            self.assertEqual(self.result("mmlupro", out2)["metrics"]["mmlupro_letter"]["items"], self.exp_letter)

    def test_subset_run_is_a_subset_of_the_full_run(self):
        with self.server() as srv:
            self.assertEqual(self.run_main(srv, "--stride", "3", "--offset", "1")[0], 0)
            m = self.result("mmlupro")["metrics"]["mmlupro_cloze"]
            want = {r["id"]: self.exp_cloze[r["id"]] for r in self.rows[1::3]}
            self.assertEqual(m["items"], want)
            self.assertEqual((m["n"], m["n_full"]), (8, 24))
            self.assertEqual(self.result("mmlupro")["subset"]["stride"], 3)

    def test_selfcheck_passes_on_a_consistent_server(self):
        for start in (0, 1):
            with self.server(FAKE_LOGPROB_START=str(start)) as srv:
                out = self.tmp / f"sc{start}"
                code, err = self.run_main(srv, "--selfcheck", "8", out=out)
                self.assertEqual(code, 0, err)
                rec = json.loads((out / "receipts" / "mmlupro.selfcheck.json").read_text())
                self.assertTrue(rec["ok"])
                self.assertEqual(rec["problems"], [])
                self.assertFalse(self.has_result("mmlupro", out))

    def test_selfcheck_red_arms(self):
        for knob, value in (("misaligned", True), ("batch_noise", 1.5), ("greedy_wrong", True)):
            with self.subTest(knob=knob), self.server() as srv:
                srv.control(**{knob: value})
                out = self.tmp / f"red_{knob}"
                code, err = self.run_main(srv, "--selfcheck", "8", out=out)
                self.assertEqual(code, ev_mmlupro.EXIT_SELFCHECK, err)
                rec = json.loads((out / "receipts" / "mmlupro.selfcheck.json").read_text())
                self.assertFalse(rec["ok"])

    def test_a_misaligned_server_makes_every_item_fail_not_score_garbage(self):
        with self.server() as srv:
            srv.control(misaligned=True)
            code, err = self.run_main(srv, "--modes", "cloze")
            self.assertEqual(code, ec.EXIT_INCOMPLETE, err)
            self.assertFalse(self.has_result("mmlupro"))
            st = self.status("mmlupro")
            self.assertEqual((st["done_now"], st["n_failed"], st["complete"]), (0, 24, False))

    def test_no_ids_logprob_falls_back_to_topk_with_the_same_answers(self):
        with self.server() as srv:
            srv.control(no_ids_logprob=True)
            code, err = self.run_main(srv, "--modes", "letter")
            self.assertEqual(code, 0, err)
            doc = self.result("mmlupro")
            self.assertEqual(doc["diagnostics"]["letter_method_used"], {"ids": 0, "topk": 24})
            self.assertEqual(doc["metrics"]["mmlupro_letter"]["items"], self.exp_letter)

    def test_forcing_ids_on_a_server_without_them_fails_the_items(self):
        with self.server() as srv:
            srv.control(no_ids_logprob=True)
            code, _ = self.run_main(srv, "--modes", "letter", "--letter-method", "ids")
            self.assertEqual(code, ec.EXIT_INCOMPLETE)
            self.assertFalse(self.has_result("mmlupro"))

    def test_failed_items_resume_to_the_same_result(self):
        with self.server() as srv:
            srv.control(fail_status=400, fail_next=7)
            code, err = self.run_main(srv, "--modes", "cloze")
            self.assertEqual(code, ec.EXIT_INCOMPLETE, err)
            self.assertFalse(self.has_result("mmlupro"))
            st = self.status("mmlupro")
            self.assertEqual(st["n_failed"] > 0, True)
            done = st["done_now"]
            code, err = self.run_main(srv, "--modes", "cloze")
            self.assertEqual(code, 0, err)
            st2 = self.status("mmlupro")
            self.assertEqual(st2["already"], done)
            self.assertEqual(st2["already"] + st2["done_now"], 24)
            self.assertEqual(self.result("mmlupro")["metrics"]["mmlupro_cloze"]["items"], self.exp_cloze)

    def test_a_dead_server_exits_4_and_scores_nothing(self):
        with self.server() as srv:
            srv.control(down=True)
            code, err = self.run_main(srv, "--modes", "cloze")
            self.assertEqual(code, ec.EXIT_DOWN, err)
            self.assertFalse(self.has_result("mmlupro"))
            self.assertTrue(self.status("mmlupro")["down"])

    def test_unpinned_data_is_refused_without_the_flag(self):
        with self.server() as srv:
            argv = [x for x in self.argv(srv, "--tokenizer", testlib.TOKENIZER_DIR) if x != "--allow-unpinned"]
            code, err = in_process(ev_mmlupro.main, argv)
            self.assertNotEqual(code, 0)
            self.assertIn("not the pinned", err)
            self.assertEqual(srv.count("/generate"), 0)

    def test_changed_dataset_is_a_journal_mismatch_until_fresh(self):
        with self.server() as srv:
            self.assertEqual(self.run_main(srv, "--modes", "cloze")[0], 0)
            code, err = self.run_main(srv, "--modes", "cloze", "--stride", "2")
            self.assertEqual(code, 1)
            self.assertIn("--fresh", err)
            code, err = self.run_main(srv, "--modes", "cloze", "--stride", "2", "--fresh")
            self.assertEqual(code, 0, err)
            self.assertEqual(self.result("mmlupro")["metrics"]["mmlupro_cloze"]["n"], 12)

    def test_usage_errors_exit_2(self):
        with self.server() as srv:
            self.assertEqual(self.go(ev_mmlupro.main, srv, "--modes", "nonsense",
                                     "--tokenizer", testlib.TOKENIZER_DIR)[0], 2)
            with mock.patch.dict("os.environ", {}, clear=False):
                import os
                os.environ.pop("EVAL_TOKENIZER", None)
                self.assertEqual(self.go(ev_mmlupro.main, srv)[0], 2)


# ----------------------------------------------------------------------------------------------------------
class TestGmmluDryRun(EvalCase):
    def server(self, **env):
        return FakeServer(["--gmmlu", str(self.data / "gmmlu_he_test.jsonl")], env=env)

    def expected(self, err, salt=""):
        test, dev = testlib.gmmlu_rows(24, 10)
        out = {}
        for r in test:
            p = ev_gmmlu.block(r, False)
            wrong = (fs.h64("gm-err", salt, p) % 10000) / 10000.0 < err
            out[r["id"]] = 0 if wrong else 1
        return out, test, dev

    def test_all_correct_server_scores_one(self):
        with self.server() as srv:
            code, err = self.go(ev_gmmlu.main, srv)
            self.assertEqual(code, 0, err)
            doc = self.result("gmmlu_he")
            m = doc["metrics"]
            self.assertEqual(m["gmmlu_he"]["score"], 1.0)
            self.assertEqual(m["gmmlu_he"]["n"], 24)
            self.assertEqual(m["gmmlu_he_pop2000"]["n"], 24)  # the synthetic set is smaller than the 2000 population
            self.assertEqual(doc["diagnostics"]["unparsed"], 0)
            self.assertEqual(sum(doc["diagnostics"]["pred_counts"].values()), 24)
            prm = srv.stats()["params"]["/v1/chat/completions"]
            self.assertEqual((prm["max_tokens"], prm["temperature"]), (1, 0))
            self.assertEqual(prm["chat_template_kwargs"], {"enable_thinking": False})

    def test_a_server_that_errs_on_a_known_set_scores_exactly_that(self):
        want, test, dev = self.expected(0.4)
        self.assertTrue(0 < sum(want.values()) < 24, "the error rate must hit some and not all items")
        with self.server() as srv:
            srv.control(gmmlu_err=0.4)
            code, err = self.go(ev_gmmlu.main, srv)
            self.assertEqual(code, 0, err)
            doc = self.result("gmmlu_he")
            self.assertEqual(doc["metrics"]["gmmlu_he"]["items"], want)
            self.assertAlmostEqual(doc["metrics"]["gmmlu_he"]["score"], sum(want.values()) / 24)
            per = doc["diagnostics"]["per_subject"]
            self.assertEqual(sum(v[1] for v in per.values()), 24)
            self.assertEqual(sum(v[0] for v in per.values()), sum(want.values()))

    def test_prompts_sent_are_the_shots_plus_the_question(self):
        want, test, dev = self.expected(0.0)
        shots = ev_gmmlu.build_shots(dev)
        digest = ec.canon_sha(sorted((r["id"], ec.hashlib.sha256(ev_gmmlu.prompt_for(r, shots).encode()).hexdigest()[:16])
                                     for r in test))
        with self.server() as srv:
            self.assertEqual(self.go(ev_gmmlu.main, srv)[0], 0)
            self.assertEqual(self.result("gmmlu_he")["diagnostics"]["prompts_sha256"], digest)
            sent = srv.stats()["bodies"]["/v1/chat/completions"]
            self.assertTrue(any("Answer: " in b and "dev question" in b for b in sent))

    def test_stride_subset(self):
        want, test, dev = self.expected(0.0)
        with self.server() as srv:
            self.assertEqual(self.go(ev_gmmlu.main, srv, "--stride", "3")[0], 0)
            m = self.result("gmmlu_he")["metrics"]["gmmlu_he"]
            self.assertEqual(set(m["items"]), {r["id"] for r in test[::3]})
            self.assertEqual((m["n"], m["n_full"]), (8, 24))

    def test_failure_then_resume(self):
        with self.server() as srv:
            srv.control(fail_status=400, fail_next=5)
            code, err = self.go(ev_gmmlu.main, srv)
            self.assertEqual(code, ec.EXIT_INCOMPLETE, err)
            self.assertFalse(self.has_result("gmmlu_he"))
            st = self.status("gmmlu_he")
            self.assertEqual((st["n_failed"], st["done_now"]), (5, 19))
            code, err = self.go(ev_gmmlu.main, srv)
            self.assertEqual(code, 0, err)
            st = self.status("gmmlu_he")
            self.assertEqual((st["already"], st["done_now"], st["complete"]), (19, 5, True))
            self.assertEqual(self.result("gmmlu_he")["metrics"]["gmmlu_he"]["score"], 1.0)
            self.assertEqual(srv.count("/v1/chat/completions"), 24 + 5)  # 5 refused, then only the 5 undone ones

    def test_a_third_run_does_no_requests(self):
        with self.server() as srv:
            self.assertEqual(self.go(ev_gmmlu.main, srv)[0], 0)
            n = srv.count("/v1/chat/completions")
            self.assertEqual(self.go(ev_gmmlu.main, srv)[0], 0)
            self.assertEqual(srv.count("/v1/chat/completions"), n)
            self.assertEqual(self.status("gmmlu_he")["done_now"], 0)

    def test_dead_server_exit_4(self):
        with self.server() as srv:
            srv.control(down=True)
            code, err = self.go(ev_gmmlu.main, srv)
            self.assertEqual(code, ec.EXIT_DOWN, err)
            self.assertFalse(self.has_result("gmmlu_he"))

    def test_server_that_comes_back_resumes(self):
        with self.server() as srv:
            srv.control(down=True)
            self.assertEqual(self.go(ev_gmmlu.main, srv)[0], ec.EXIT_DOWN)
            srv.control(down=False)
            self.assertEqual(self.go(ev_gmmlu.main, srv)[0], 0)
            self.assertEqual(self.result("gmmlu_he")["metrics"]["gmmlu_he"]["n"], 24)

    def test_missing_data_file_names_the_fix(self):
        (self.data / "gmmlu_he_dev.jsonl").unlink()
        with self.server() as srv:
            code, err = self.go(ev_gmmlu.main, srv)
            self.assertNotEqual(code, 0)
            self.assertIn("prepare_box.sh data", err)

    def test_manifest_mismatch_is_refused(self):
        (self.data / "MANIFEST.json").write_text(json.dumps({"files": {"gmmlu_he_test.jsonl": {"sha256": "0" * 64}}}))
        with self.server() as srv:
            code, err = self.go(ev_gmmlu.main, srv)
            self.assertNotEqual(code, 0)
            self.assertIn("MANIFEST", err)

    def test_pinned_row_counts_are_enforced(self):
        with self.server() as srv:
            argv = [x for x in self.argv(srv) if x != "--allow-unpinned"]
            code, err = in_process(ev_gmmlu.main, argv)
            self.assertNotEqual(code, 0)  # the synthetic files are not the pinned sha and not the pinned row counts

    def test_a_different_arm_name_does_not_change_the_answers(self):
        want, _, _ = self.expected(0.3)
        with self.server() as srv:
            srv.control(gmmlu_err=0.3)
            self.assertEqual(self.go(ev_gmmlu.main, srv)[0], 0)
            out2 = self.tmp / "out_b"
            argv = self.argv(srv, out=out2)
            argv[argv.index("t")] = "other"
            with mock.patch.object(ec, "make_server", fast_server):
                self.assertEqual(in_process(ev_gmmlu.main, argv)[0], 0)
            self.assertEqual(self.result("gmmlu_he", out2)["metrics"]["gmmlu_he"]["items"], want)
            self.assertEqual(self.result("gmmlu_he", out2)["arm"], "other")

    def test_salt_moves_the_wrong_set_so_a_paired_comparison_has_something_to_find(self):
        with self.server() as srv:
            srv.control(gmmlu_err=0.5)
            self.assertEqual(self.go(ev_gmmlu.main, srv)[0], 0)
            a = self.result("gmmlu_he")["metrics"]["gmmlu_he"]["items"]
            srv.control(salt="other")
            out2 = self.tmp / "out_s"
            self.assertEqual(self.go(ev_gmmlu.main, srv, out=out2)[0], 0)
            b = self.result("gmmlu_he", out2)["metrics"]["gmmlu_he"]["items"]
            self.assertEqual(set(a), set(b))
            self.assertNotEqual(a, b)
            self.assertEqual(b, self.expected(0.5, "other")[0])


# ----------------------------------------------------------------------------------------------------------
class TestHumanEvalDryRun(EvalCase):
    def server(self, **env):
        return FakeServer(["--humaneval", str(self.data / "humaneval_test.jsonl")], env=env)

    def test_canonical_answers_pass(self):
        with self.server() as srv:
            code, err = self.go(ev_humaneval.main, srv)
            self.assertEqual(code, 0, err)
            doc = self.result("humaneval")
            m = doc["metrics"]["humaneval"]
            self.assertEqual((m["n"], m["score"]), (8, 1.0))
            self.assertEqual(set(m["items"]), {f"HumanEval/{i}" for i in range(8)})
            self.assertEqual(doc["diagnostics"]["n_timeout"], 0)
            prm = srv.stats()["params"]["/v1/completions"]
            self.assertEqual((prm["max_tokens"], prm["temperature"]), (512, 0))
            self.assertNotIn("stop", prm)  # the cut is client side, so the raw completion is journaled and re-scorable

    def test_corrupted_tasks_fail_exactly_those(self):
        rows = testlib.humaneval_rows(8)
        want = {r["id"]: (0 if fs.he_index_hash(i, "") % 3 == 0 else 1) for i, r in enumerate(rows)}
        self.assertTrue(0 < sum(want.values()) < 8)
        with self.server() as srv:
            srv.control(he_corrupt_mod=3)
            self.assertEqual(self.go(ev_humaneval.main, srv)[0], 0)
            self.assertEqual(self.result("humaneval")["metrics"]["humaneval"]["items"], want)

    def test_an_infinite_loop_completion_is_a_timeout_failure_not_a_hang(self):
        # a completion that loops forever must cost WALL_S at most and count as a fail; shrink the wall for the test
        rows = testlib.humaneval_rows(1)
        with mock.patch.object(ev_humaneval, "WALL_S", 1.5):
            ok, why = ev_humaneval.run_program(ev_humaneval.program_for(rows[0], "    while True:\n        pass\n"),
                                               wall=1.5)
        self.assertEqual((ok, why), (False, "timeout"))

    def test_failure_then_resume(self):
        with self.server() as srv:
            srv.control(fail_status=400, fail_next=3)
            self.assertEqual(self.go(ev_humaneval.main, srv)[0], ec.EXIT_INCOMPLETE)
            self.assertFalse(self.has_result("humaneval"))
            self.assertEqual(self.go(ev_humaneval.main, srv)[0], 0)
            self.assertEqual(self.result("humaneval")["metrics"]["humaneval"]["score"], 1.0)

    def test_dead_server_exit_4(self):
        with self.server() as srv:
            srv.control(down=True)
            self.assertEqual(self.go(ev_humaneval.main, srv)[0], ec.EXIT_DOWN)

    def test_check_canonical_needs_no_server(self):
        code, err, out = canonical_main(["--data", str(self.data), "--allow-unpinned"])
        self.assertEqual(code, 0, err)
        self.assertIn("'ok': True", err)
        rec = json.loads(out)  # run_general.sh keeps stdout as receipts/humaneval.canonical.json
        self.assertEqual((rec["ok"], rec["n"], rec["canonical_pass"], rec["canonical_failed"]), (True, 8, 8, []))

    def test_check_canonical_exits_2_on_a_broken_reference(self):
        rows = testlib.humaneval_rows(4)
        rows[1]["canonical_solution"] = "    return -5\n"
        testlib.write_jsonl(self.data / "humaneval_test.jsonl", rows)
        code, err, out = canonical_main(["--data", str(self.data), "--allow-unpinned"])
        self.assertEqual(code, 2)
        self.assertIn("HumanEval/1", err)
        rec = json.loads(out)
        self.assertEqual((rec["ok"], rec["canonical_failed"]), (False, ["HumanEval/1"]))

    @unittest.skipUnless(testlib.have_real_data("humaneval_test.jsonl"), "needs EVAL_TEST_DATA")
    def test_check_canonical_on_the_real_164_tasks(self):
        code, err, out = canonical_main(["--data", REAL_DATA])
        self.assertEqual(code, 0, err)
        rec = json.loads(out)
        self.assertEqual((rec["n"], rec["canonical_pass"], rec["ok"]), (164, 164, True))

    def test_row_count_pin_is_enforced(self):
        with self.server() as srv:
            argv = [x for x in self.argv(srv) if x != "--allow-unpinned"]
            code, _ = in_process(ev_humaneval.main, argv)
            self.assertNotEqual(code, 0)


# ----------------------------------------------------------------------------------------------------------
IFEVAL_EXPECT = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import fake_server as fs, ev_ifeval, evalcommon as ec
keys = json.loads(sys.argv[3])
h = ev_ifeval.Harness(Path(sys.argv[2]), ec.load_pins(), verify=True)
out = {}
for k in keys:
    ex = h.by_key[k]
    text = fs.answer_chat({"messages": [{"role": "user", "content": ex.prompt}]})["text"]
    out[k] = {"text": text, **h.score(k, text), "inst": list(ex.instruction_id_list)}
print(json.dumps(out))
'''


@unittest.skipUnless(testlib.ifeval_python() and testlib.nltk_data_dir()
                     and testlib.have_real_data("ifeval_input_data.jsonl", "ifeval_harness"),
                     "needs EVAL_TEST_DATA with ifeval_harness, EVAL_TEST_IFEVAL_PY and EVAL_TEST_NLTK_DATA")
class TestIfevalDryRun(ScratchCase):
    STRIDE = 27

    def setUp(self) -> None:
        super().setUp()
        self.py = testlib.ifeval_python()
        self.env = {"NLTK_DATA": testlib.nltk_data_dir()}
        self.out = self.tmp / "out"
        rows = ec.read_jsonl(Path(REAL_DATA) / "ifeval_input_data.jsonl")
        self.keys = ec.select_ids([str(r["key"]) for r in rows], self.STRIDE)

    def run_eval(self, srv, *extra, out=None):
        return testlib.run([self.py, HERE / "ev_ifeval.py", "--arm", "t", "--out", out or self.out, "--data", REAL_DATA,
                            "--base-url", srv.base, "--workers", "4", "--stride", str(self.STRIDE), "--attempts", "2",
                            "--breaker", "3", *extra], env=self.env, timeout=300)

    def expectation(self):
        p = testlib.run([self.py, "-c", IFEVAL_EXPECT, HERE, REAL_DATA, json.dumps(self.keys)], env=self.env)
        self.assertEqual(p.returncode, 0, p.stderr)
        return json.loads(p.stdout)

    def test_scores_match_the_harness_applied_to_the_fake_answers(self):
        exp = self.expectation()
        with FakeServer() as srv:
            p = self.run_eval(srv)
            self.assertEqual(p.returncode, 0, p.stderr)
            doc = json.loads((self.out / "results" / "ifeval.json").read_text())
            m = doc["metrics"]
            self.assertEqual(set(m), {"ifeval_strict", "ifeval_loose", "ifeval_strict_inst", "ifeval_loose_inst"})
            self.assertEqual(m["ifeval_strict"]["items"], {k: int(all(v["strict"])) for k, v in exp.items()})
            self.assertEqual(m["ifeval_loose"]["items"], {k: int(all(v["loose"])) for k, v in exp.items()})
            self.assertEqual(m["ifeval_strict_inst"]["items"],
                             {f"{k}:{i}": x for k, v in exp.items() for i, x in enumerate(v["strict"])})
            self.assertEqual(m["ifeval_strict_inst"]["level"], "instruction")
            self.assertEqual(m["ifeval_strict"]["n_full"], 541)
            self.assertEqual(m["ifeval_strict"]["n"], len(self.keys))
            # loose is never stricter than strict, on a prompt and on an instruction
            for k in self.keys:
                self.assertLessEqual(m["ifeval_strict"]["items"][k], m["ifeval_loose"]["items"][k])
            self.assertFalse(doc["pins"]["unpinned"])
            journal = ec.read_jsonl(self.out / "journal" / "ifeval.jsonl")
            resp = {r["id"]: r["response"] for r in journal if "id" in r}
            self.assertEqual(resp, {k: v["text"] for k, v in exp.items()})
            prm = srv.stats()["params"]["/v1/chat/completions"]
            self.assertEqual((prm["max_tokens"], prm["temperature"]), (4096, 0))
            self.assertEqual(prm["chat_template_kwargs"], {"enable_thinking": False})

    def test_the_canned_answer_fails_most_prompts_so_the_score_is_not_vacuous(self):
        exp = self.expectation()
        total = sum(len(v["strict"]) for v in exp.values())
        passed = sum(sum(v["strict"]) for v in exp.values())
        self.assertGreater(total, 0)
        self.assertLess(passed, total)

    def test_resume_and_dead_server(self):
        with FakeServer() as srv:
            srv.control(fail_status=400, fail_next=4)
            p = self.run_eval(srv)
            self.assertEqual(p.returncode, ec.EXIT_INCOMPLETE, p.stderr)
            self.assertFalse((self.out / "results" / "ifeval.json").exists())
            p = self.run_eval(srv)
            self.assertEqual(p.returncode, 0, p.stderr)
            st = json.loads((self.out / "receipts" / "ifeval.status.json").read_text())
            self.assertEqual((st["already"] + st["done_now"], st["complete"]), (len(self.keys), True))
        with FakeServer() as srv:
            srv.control(down=True)
            p = self.run_eval(srv, out=self.tmp / "dead")
            self.assertEqual(p.returncode, ec.EXIT_DOWN, p.stderr)

    def test_a_changed_harness_file_is_refused(self):
        import shutil
        d = self.tmp / "data"
        d.mkdir()
        shutil.copy(Path(REAL_DATA) / "ifeval_input_data.jsonl", d / "ifeval_input_data.jsonl")
        shutil.copytree(Path(REAL_DATA) / "ifeval_harness", d / "ifeval_harness")
        victim = d / "ifeval_harness" / "instruction_following_eval" / "instructions.py"
        victim.write_text(victim.read_text() + "\n# edited\n")
        with FakeServer() as srv:
            p = testlib.run([self.py, HERE / "ev_ifeval.py", "--arm", "t", "--out", self.out, "--data", d,
                             "--base-url", srv.base, "--stride", "100"], env=self.env, timeout=120)
            self.assertEqual(p.returncode, 1)
            self.assertIn("is not the pinned", p.stderr)
            self.assertEqual(srv.count("/v1/chat/completions"), 0)


# ----------------------------------------------------------------------------------------------------------
class Row(collections.namedtuple("Row", "question option_a option_b option_c option_d answer subject sample_id")):
    pass


class TestGmmluPromptOracle(unittest.TestCase):
    """The prompts ev_gmmlu.py sends are byte-identical to the Hebrew lane's: same block(), same five shots per subject."""

    @staticmethod
    def original_block():
        ns = {"LETTERS": ["A", "B", "C", "D"]}
        exec(compile(ORIGINAL_BLOCK_SRC, "gmmlu-original-block", "exec"), ns)  # noqa: S102  the lane's own 8 lines
        return ns["block"]

    def test_original_source_is_still_the_copy(self):
        if not ORIGINAL_FILE.is_file():
            self.skipTest("the original script is not on this machine")
        tree = ast.parse(ORIGINAL_FILE.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "block")
        self.assertEqual(ast.dump(fn), ast.dump(ast.parse(ORIGINAL_BLOCK_SRC).body[0]))

    def test_synthetic_rows(self):
        block = self.original_block()
        test, dev = testlib.gmmlu_rows(30, 20)
        # include awkward text: stripped spaces, a non-string option, Hebrew, a trailing newline in the question
        test[0]["question"] = "  padded \n"
        test[1]["options"][2] = 7
        test[2]["answer"] = "C"

        def row(r):
            o = r["options"]
            return Row(r["question"], o[0], o[1], o[2], o[3], r["answer"], r["subject"], r["id"])

        by_subject = collections.defaultdict(list)
        for r in dev:
            by_subject[r["subject"]].append(row(r))
        orig_shots = {s: "".join(block(x, True) for x in rows[:5]) for s, rows in by_subject.items()}
        mine = ev_gmmlu.build_shots(dev)
        self.assertEqual(mine, orig_shots)
        for r in test:
            self.assertEqual(ev_gmmlu.prompt_for(r, mine), orig_shots.get(r["subject"], "") + block(row(r), False))

    @unittest.skipUnless((GMMLU_PARQUET / "he-test.parquet").is_file() and (GMMLU_PARQUET / "he-dev.parquet").is_file(),
                         "needs the local Global-MMLU parquet files")
    def test_real_he_test_and_dev_rows(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("needs pandas and pyarrow")
        block = self.original_block()
        test_df = pd.read_parquet(GMMLU_PARQUET / "he-test.parquet")
        dev_df = pd.read_parquet(GMMLU_PARQUET / "he-dev.parquet")
        orig_shots = {s: "".join(block(r, True) for r in list(g.itertuples())[:5]) for s, g in dev_df.groupby("subject")}
        norm_test = prepare_data.norm_gmmlu(test_df.to_dict("records"), "test")
        norm_dev = prepare_data.norm_gmmlu(dev_df.to_dict("records"), "dev")
        mine = ev_gmmlu.build_shots(norm_dev)
        self.assertEqual(mine, orig_shots)
        n = 0
        for orig_row, norm_row in zip(test_df.itertuples(), norm_test):
            self.assertEqual(str(orig_row.sample_id), norm_row["id"])
            want = orig_shots.get(orig_row.subject, "") + block(orig_row, False)
            self.assertEqual(ev_gmmlu.prompt_for(norm_row, mine), want, norm_row["id"])
            self.assertEqual(norm_row["answer"], orig_row.answer.strip())
            n += 1
        self.assertEqual(n, 14042)
        # the files prepare_data.py wrote are these same rows
        if testlib.have_real_data("gmmlu_he_test.jsonl", "gmmlu_he_dev.jsonl"):
            self.assertEqual(ec.read_jsonl(Path(REAL_DATA) / "gmmlu_he_test.jsonl"), norm_test)
            self.assertEqual(ec.read_jsonl(Path(REAL_DATA) / "gmmlu_he_dev.jsonl"), norm_dev)

    @unittest.skipUnless(testlib.have_real_data("gmmlu_he_test.jsonl", "gmmlu_he_dev.jsonl"), "needs EVAL_TEST_DATA")
    def test_population_of_2000_is_the_lanes_pinned_sample(self):
        import random
        test = ec.read_jsonl(Path(REAL_DATA) / "gmmlu_he_test.jsonl")
        g = ec.load_pins()["datasets"]["gmmlu_he"]
        ids = sorted({r["id"] for r in test})
        random.Random(g["pop2000_seed"]).shuffle(ids)
        self.assertEqual(ev_gmmlu.pop_ids(test, g["pop2000_seed"], g["pop2000_n"]), ids[: g["pop2000_n"]])
        self.assertEqual(len(ev_gmmlu.pop_ids(test, g["pop2000_seed"], g["pop2000_n"])), 2000)


if __name__ == "__main__":
    sys.exit(unittest.main())
