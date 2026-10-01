#!/usr/bin/env python3
"""CPU dry runs of run_general.sh against fake_server.py: the arm-identity guard, argument handling, a full run, resume
and skip, bounded retries, a dead server, a selfcheck or canonical failure, the pass timeout, and --overlap.

The script runs for real (bash, curl, timeout, the ev_*.py entry points); only the server is fake. serving.json and
arm.json are written by hand here in the shape serve_arm.sh writes them, so no GPU, no SGLang and no network are involved.

    python -m unittest test_run_general -v          (run_tests.sh does this with the CPU caps)
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Optional

import testlib
from testlib import HERE, REAL_DATA, FakeServer, ScratchCase

import ev_mmlupro
import evalcommon as ec

SCRIPT = HERE / "run_general.sh"
ARM = "t"


class GeneralCase(ScratchCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = self.tmp / "eval"
        self.arm_dir = self.root / ARM
        self.data = self.tmp / "data"
        testlib.write_data_dir(self.data)
        self.tok = testlib.TOKENIZER_DIR if testlib.have_tokenizer() else None
        self.fake_tok = self.tmp / "tok"
        self.fake_tok.mkdir()
        (self.fake_tok / "config.json").write_text("{}")
        self.bfcl_venv = self.tmp / "bfcl-venv"
        (self.bfcl_venv / "bin").mkdir(parents=True)
        (self.bfcl_venv / "bin" / "python").symlink_to(sys.executable)

    # -- the state serve_arm.sh leaves behind ------------------------------------------------------------------
    def write_state(self, port: int, arm: str = ARM, engagement_ok: object = True, dp: int = 2, ctx: int = 131072,
                    serving: bool = True, arm_json: bool = True) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.arm_dir.mkdir(parents=True, exist_ok=True)
        if serving:
            (self.root / "serving.json").write_text(json.dumps({"arm": arm, "port": port, "kind": "base"}))
        if arm_json:
            (self.arm_dir / "arm.json").write_text(json.dumps({"arm": ARM, "engagement": {"ok": engagement_ok},
                                                              "serve": {"dp_size": dp, "context_length": ctx}}))

    def server(self, **env) -> FakeServer:
        args = ["--gmmlu", str(self.data / "gmmlu_he_test.jsonl"), "--humaneval", str(self.data / "humaneval_test.jsonl")]
        if self.tok:
            tok = ec.load_tokenizer(self.tok)
            args += ["--special-ids", ",".join(str(i) for i in ev_mmlupro.letter_token_ids(tok))]
        return FakeServer(args, env=env)

    def sh(self, *args: str, py: Optional[str] = None, data: Optional[Path] = None, timeout: float = 400.0,
           **env: str):
        e = {"EVAL_ROOT": str(self.root), "EVAL_PY": py or sys.executable, "EVAL_DATA": str(data or self.data),
             "EVAL_TOKENIZER": self.tok or str(self.fake_tok), "BFCL_VENV": str(self.bfcl_venv),
             "NLTK_DATA": os.environ.get("EVAL_TEST_NLTK_DATA", str(self.tmp / "no-nltk")), "PASS_TIMEOUT_S": "300"}
        e.update(env)
        return testlib.run(["bash", SCRIPT, *args], env=e, timeout=timeout)

    def report(self) -> dict:
        return json.loads((self.arm_dir / "receipts" / "run_general.json").read_text())

    def result(self, name: str) -> dict:
        return json.loads((self.arm_dir / "results" / f"{name}.json").read_text())

    def results_present(self) -> set:
        d = self.arm_dir / "results"
        return {p.stem for p in d.glob("*.json")} if d.is_dir() else set()

    def assert_no_state_dir(self) -> None:
        self.assertEqual(list((self.arm_dir / "receipts").glob(".run.*")), [])


# ----------------------------------------------------------------------------------------------------------
class TestCommandLine(GeneralCase):
    def test_usage_errors_exit_2(self):
        for args in ([], ["--arm"], ["--arm", "bad name"], ["--arm", "a/b"], ["--arm", ARM, "--bogus"],
                     ["--arm", ARM, "--evals", "mmlupro,nope"], ["--arm", ARM, "--evals", ""],
                     ["--arm", ARM, "--stride-mmlupro", "0"], ["--arm", ARM, "--stride-mmlupro", "x"],
                     ["--arm", ARM, "--stride-nope", "2"], ["--arm", ARM, "--extra-nope", "x"],
                     ["--arm", ARM, "--retries", "0"], ["--arm", ARM, "--retries", "two"],
                     ["--arm", ARM, "--limit", "-1"], ["--arm", ARM, "--selfcheck", "x"], ["--help"]):
            with self.subTest(args=args):
                p = self.sh(*args)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertTrue(p.stderr.strip())

    def test_bad_pass_timeout_env_exits_2(self):
        p = self.sh("--arm", ARM, "--print-cmd", PASS_TIMEOUT_S="0")
        self.assertEqual(p.returncode, 2, p.stderr)

    def test_help_prints_the_option_list(self):
        p = self.sh("--help")
        self.assertEqual(p.returncode, 2)
        for word in ("--arm", "--evals", "--stride-<eval>", "--overlap", "--fresh", "--print-cmd", "EVAL_PY"):
            self.assertIn(word, p.stderr)

    def test_print_cmd_lists_every_eval_in_order_with_its_options(self):
        p = self.sh("--arm", ARM, "--print-cmd", "--stride-humaneval", "2", "--limit", "7", "--allow-unpinned",
                    "--extra-bfcl", "--workers 256", "--extra-ifeval", "--timeout 99", "--base-url", "http://h:1")
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = p.stdout.splitlines()
        self.assertTrue(lines[0].startswith(f"arm={ARM} out={self.root / ARM} base_url=http://h:1 "), lines[0])
        self.assertIn("evals=mmlupro humaneval ifeval gmmlu bfcl", lines[0])
        cmds = [l for l in lines if l.startswith("cmd ")]
        self.assertEqual([l.split(":")[0] for l in cmds],
                         ["cmd mmlupro", "cmd humaneval", "cmd ifeval", "cmd gmmlu", "cmd bfcl"])
        by = {l.split(":")[0][4:]: l for l in cmds}
        self.assertIn("ev_humaneval.py", by["humaneval"])
        self.assertIn("--stride 2", by["humaneval"])
        self.assertIn("--stride 1", by["mmlupro"])
        for n in by:
            self.assertIn("--limit 7", by[n])
            self.assertIn("--allow-unpinned", by[n])
            self.assertIn("--base-url http://h:1", by[n])
            self.assertIn(f"--data {self.data}", by[n])
            self.assertNotIn("--fresh", by[n])
        self.assertIn("--tokenizer", by["mmlupro"])
        self.assertIn("--bfcl-venv", by["bfcl"])
        self.assertTrue(by["bfcl"].rstrip().endswith("--workers 256"), by["bfcl"])  # an --extra comes last, so it wins
        self.assertIn("--workers 128", by["bfcl"])  # 16 x the default dp of 8 when there is no arm.json to read
        self.assertTrue(by["ifeval"].rstrip().endswith("--timeout 99"))
        self.assertIn("pre mmlupro: selfcheck 48 items", lines)
        self.assertIn("pre humaneval: ev_humaneval.py --check-canonical", lines)
        self.assertFalse((self.root).exists(), "--print-cmd must not create anything")

    def test_print_cmd_subset_keeps_the_canonical_order_and_drops_the_pre_steps(self):
        p = self.sh("--arm", ARM, "--print-cmd", "--evals", "gmmlu,mmlupro", "--selfcheck", "0", "--fresh")
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = p.stdout.splitlines()
        self.assertIn("evals=mmlupro gmmlu", lines[0])
        self.assertEqual([l.split(":")[0] for l in lines if l.startswith("cmd ")], ["cmd mmlupro", "cmd gmmlu"])
        self.assertFalse([l for l in lines if l.startswith("pre ")])
        self.assertTrue(all("--fresh" in l for l in lines if l.startswith("cmd ")))


# ----------------------------------------------------------------------------------------------------------
class TestArmGuard(GeneralCase):
    def refused(self, needle: str, **state):
        p = self.sh("--arm", ARM, "--evals", "gmmlu", "--base-url", "http://127.0.0.1:9")
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn(needle, p.stderr)
        self.assertEqual(self.results_present(), set())

    def test_no_serving_json(self):
        self.write_state(1, serving=False)
        self.refused("serve_arm.sh up --arm")

    def test_serving_a_different_arm(self):
        self.write_state(1, arm="ctrl")
        self.refused("serving arm 'ctrl', not 't'")

    def test_no_arm_json(self):
        self.write_state(1, arm_json=False)
        self.refused("has no engagement record")

    def test_engagement_not_ok(self):
        for val in (False, None):
            with self.subTest(val=val):
                self.write_state(1, engagement_ok=val)
                self.refused("not proven to be arm 't'")

    def test_engagement_missing_key(self):
        self.write_state(1)
        (self.arm_dir / "arm.json").write_text(json.dumps({"arm": ARM}))
        self.refused("engagement.ok=missing")

    def test_eval_py_that_is_not_an_interpreter(self):
        self.write_state(1)
        p = self.sh("--arm", ARM, "--evals", "gmmlu", py=str(self.tmp / "nope" / "python"))
        self.assertEqual(p.returncode, 1)
        self.assertIn("is not an interpreter", p.stderr)

    def test_missing_tokenizer_config_is_named_for_mmlupro_and_bfcl_only(self):
        self.write_state(1)
        empty = self.tmp / "emptytok"
        empty.mkdir()
        for evals in ("mmlupro", "bfcl"):
            p = self.sh("--arm", ARM, "--evals", evals, EVAL_TOKENIZER=str(empty))
            self.assertEqual(p.returncode, 1, p.stderr)
            self.assertIn("has no config.json", p.stderr)
        p = self.sh("--arm", ARM, "--evals", "gmmlu", "--base-url", "http://127.0.0.1:9", EVAL_TOKENIZER=str(empty))
        self.assertNotIn("has no config.json", p.stderr)

    def test_missing_bfcl_venv(self):
        self.write_state(1)
        p = self.sh("--arm", ARM, "--evals", "bfcl", BFCL_VENV=str(self.tmp / "nope"))
        self.assertEqual(p.returncode, 1)
        self.assertIn("prepare_box.sh bfcl", p.stderr)

    def test_short_context_is_warned_about_but_not_refused(self):
        with self.server() as srv:
            self.write_state(srv.port, ctx=32768)
            p = self.sh("--arm", ARM, "--evals", "gmmlu", "--allow-unpinned")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("context_length is 32768", p.stderr)

    def test_out_dir_option_and_env_are_the_arm_dir(self):
        with self.server() as srv:
            self.write_state(srv.port)
            other = self.tmp / "elsewhere"
            other.mkdir()
            (other / "arm.json").write_text(json.dumps({"engagement": {"ok": True}, "serve": {"dp_size": 1}}))
            p = self.sh("--arm", ARM, "--evals", "gmmlu", "--allow-unpinned", "--out", str(other))
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue((other / "results" / "gmmlu_he.json").is_file())
            self.assertFalse((self.arm_dir / "results").exists())


# ----------------------------------------------------------------------------------------------------------
@unittest.skipUnless(testlib.have_tokenizer(), "needs the Qwen3.8 tokenizer for the MMLU-Pro leg")
class TestFullRun(GeneralCase):
    EVALS = "mmlupro,humaneval,gmmlu"

    def run_all(self, *extra, **env):
        return self.sh("--arm", ARM, "--evals", self.EVALS, "--allow-unpinned", "--selfcheck", "8", "--backoff-s", "0",
                       *extra, **env)

    def test_complete_run_resume_skip_and_fresh(self):
        with self.server() as srv:
            self.write_state(srv.port)
            p = self.run_all()
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(self.results_present(), {"mmlupro", "humaneval", "gmmlu_he"})
            rep = self.report()
            self.assertEqual((rep["schema"], rep["arm"], rep["exit"], rep["limit"]), ("negeig-general-run/1", ARM, 0, 0))
            self.assertEqual(rep["base_url"], srv.base)
            self.assertEqual({n: (e["exit"], e["passes"], e["how"]) for n, e in rep["evals"].items()},
                             {n: (0, 1, "run") for n in ("mmlupro", "humaneval", "gmmlu")})
            self.assertTrue(json.loads((self.arm_dir / "receipts" / "mmlupro.selfcheck.json").read_text())["ok"])
            self.assertTrue(json.loads((self.arm_dir / "receipts" / "humaneval.canonical.json").read_text())["ok"])
            for n in ("mmlupro", "humaneval", "gmmlu"):
                self.assertTrue((self.arm_dir / "logs" / f"{n}.log").is_file())
                self.assertTrue((self.arm_dir / "journal" / f"{'gmmlu_he' if n == 'gmmlu' else n}.jsonl").is_file())
            for name in ("mmlupro", "humaneval", "gmmlu_he"):
                doc = self.result(name)
                self.assertEqual(doc["arm"], ARM)
                self.assertTrue(doc["complete"])
            self.assertEqual(self.result("humaneval")["metrics"]["humaneval"]["score"], 1.0)
            self.assertIn("complete in", p.stderr)
            self.assertIn("compare_general.py --base", p.stderr)
            self.assert_no_state_dir()

            before = srv.stats()["counts"]
            p = self.run_all()
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(srv.stats()["counts"], before, "a complete result must not cost one request")
            rep = self.report()
            self.assertEqual({e["how"] for e in rep["evals"].values()}, {"skipped"})
            self.assertEqual({e["passes"] for e in rep["evals"].values()}, {0})

            p = self.run_all("--fresh")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual({e["how"] for e in self.report()["evals"].values()}, {"run"})
            self.assertGreater(srv.stats()["counts"]["/v1/chat/completions"], before["/v1/chat/completions"])
            self.assert_no_state_dir()

    def test_a_subset_run_is_recorded_and_a_later_full_run_needs_fresh(self):
        with self.server() as srv:
            self.write_state(srv.port)
            p = self.run_all("--limit", "5")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(self.report()["limit"], 5)
            self.assertEqual(self.result("gmmlu_he")["subset"]["limit"], 5)
            self.assertEqual(self.result("gmmlu_he")["metrics"]["gmmlu_he"]["n"], 5)
            # same out dir, no limit: the journals refuse to mix a 5-item run with the full one
            p = self.run_all()
            self.assertEqual(p.returncode, 1, p.stderr)
            self.assertIn("--fresh", (self.arm_dir / "logs" / "gmmlu.log").read_text())
            p = self.run_all("--fresh")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(self.result("gmmlu_he")["metrics"]["gmmlu_he"]["n"], 24)

    def test_selfcheck_failure_stops_that_eval_with_exit_5_and_the_others_still_run(self):
        with self.server() as srv:
            srv.control(misaligned=True)
            self.write_state(srv.port)
            p = self.run_all()
            self.assertEqual(p.returncode, 1, p.stderr)
            rep = self.report()
            self.assertEqual((rep["evals"]["mmlupro"]["exit"], rep["evals"]["mmlupro"]["how"]), (ev_mmlupro.EXIT_SELFCHECK, "pre"))
            self.assertEqual(rep["evals"]["gmmlu"]["exit"], 0)
            self.assertNotIn("mmlupro", self.results_present())
            self.assertFalse(json.loads((self.arm_dir / "receipts" / "mmlupro.selfcheck.json").read_text())["ok"])
            self.assertIn("logprob semantics", p.stderr)

    def test_skipping_the_selfcheck(self):
        with self.server() as srv:
            self.write_state(srv.port)
            p = self.sh("--arm", ARM, "--evals", "mmlupro", "--allow-unpinned", "--selfcheck", "0")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertFalse((self.arm_dir / "receipts" / "mmlupro.selfcheck.json").exists())


# ----------------------------------------------------------------------------------------------------------
class TestFailureHandling(GeneralCase):
    def run_gm(self, *extra, **env):
        return self.sh("--arm", ARM, "--evals", "gmmlu", "--allow-unpinned", "--backoff-s", "0", *extra, **env)

    def test_incomplete_pass_is_retried_and_resumes_where_it_stopped(self):
        with self.server() as srv:
            srv.control(fail_status=400, fail_next=5)
            self.write_state(srv.port)
            p = self.run_gm("--retries", "3")
            self.assertEqual(p.returncode, 0, p.stderr)
            e = self.report()["evals"]["gmmlu"]
            self.assertEqual((e["exit"], e["passes"], e["how"]), (0, 2, "run"))
            self.assertEqual(self.result("gmmlu_he")["metrics"]["gmmlu_he"]["n"], 24)
            self.assertEqual(srv.count("/v1/chat/completions"), 24 + 5)
            self.assertIn("retrying", p.stderr)
            self.assert_no_state_dir()

    def test_retries_are_bounded_and_the_exit_is_3(self):
        with self.server() as srv:
            srv.control(fail_status=400, fail_next=100000)
            self.write_state(srv.port)
            p = self.run_gm("--retries", "2")
            self.assertEqual(p.returncode, 3, p.stderr)
            e = self.report()["evals"]["gmmlu"]
            self.assertEqual((e["exit"], e["passes"]), (3, 2))
            self.assertEqual(self.report()["exit"], 3)
            self.assertEqual(self.results_present(), set())
            self.assertIn("re-run the same command to resume", p.stderr)
            self.assertEqual(srv.count("/v1/chat/completions"), 48)
            # the server recovers: the same command finishes it
            srv.control(fail_next=0)
            p = self.run_gm("--retries", "2")
            self.assertEqual(p.returncode, 0, p.stderr)

    def test_a_server_that_is_not_up_exits_4_without_running_the_eval(self):
        srv = self.server()
        port = srv.port
        srv.stop()
        self.write_state(port)
        p = self.run_gm()
        self.assertEqual(p.returncode, 4, p.stderr)
        e = self.report()["evals"]["gmmlu"]
        self.assertEqual((e["exit"], e["how"], e["passes"]), (4, "down", 1))
        self.assertIn("not 200", p.stderr)
        self.assertIn("serve_arm.sh up --replace", p.stderr)

    def test_a_server_that_dies_mid_eval_stops_the_whole_run(self):
        with self.server() as srv:
            srv.control(down=True)
            self.write_state(srv.port)
            p = self.sh("--arm", ARM, "--evals", "humaneval,gmmlu", "--allow-unpinned", "--backoff-s", "0",
                        "--extra-humaneval", "--attempts 2 --breaker 2")
            self.assertEqual(p.returncode, 4, p.stderr)
            ev = self.report()["evals"]
            self.assertEqual(ev["humaneval"]["exit"], ec.EXIT_DOWN)
            self.assertEqual((ev["gmmlu"]["exit"], ev["gmmlu"]["how"]), (-1, "notrun"))
            self.assertEqual(srv.count("/v1/chat/completions"), 0)

    def test_the_pass_timeout_kills_a_stuck_eval_and_counts_as_incomplete(self):
        with self.server() as srv:
            srv.control(latency_ms=6000)
            self.write_state(srv.port)
            p = self.run_gm("--retries", "1", PASS_TIMEOUT_S="2")
            self.assertEqual(p.returncode, 3, p.stderr)
            self.assertEqual(self.report()["evals"]["gmmlu"]["exit"], 124)

    def test_a_broken_canonical_solution_stops_humaneval_before_any_request(self):
        rows = testlib.humaneval_rows(4)
        rows[2]["canonical_solution"] = "    return None\n"
        testlib.write_jsonl(self.data / "humaneval_test.jsonl", rows)
        with self.server() as srv:
            self.write_state(srv.port)
            p = self.sh("--arm", ARM, "--evals", "humaneval,gmmlu", "--allow-unpinned")
            self.assertEqual(p.returncode, 1, p.stderr)
            ev = self.report()["evals"]
            self.assertEqual((ev["humaneval"]["exit"], ev["humaneval"]["how"]), (2, "pre"))
            self.assertEqual(ev["gmmlu"]["exit"], 0)
            self.assertFalse(json.loads((self.arm_dir / "receipts" / "humaneval.canonical.json").read_text())["ok"])
            self.assertEqual(srv.count("/v1/completions"), 0)

    def test_overlap_runs_bfcl_in_the_background_without_coupling_the_exits(self):
        with self.server() as srv:
            self.write_state(srv.port)
            p = self.sh("--arm", ARM, "--evals", "gmmlu,bfcl", "--allow-unpinned", "--overlap",
                        "--extra-bfcl", "--no-such-flag")
            self.assertEqual(p.returncode, 1, p.stderr)  # bfcl exit 2 is an eval failing for good
            ev = self.report()["evals"]
            self.assertEqual(ev["gmmlu"]["exit"], 0)
            self.assertEqual(ev["bfcl"]["exit"], 2)
            self.assertIn("gmmlu_he", self.results_present())
            self.assertIn("--no-such-flag", (self.arm_dir / "logs" / "bfcl.log").read_text())
            self.assert_no_state_dir()

    def test_stride_option_selects_every_nth_item(self):
        with self.server() as srv:
            self.write_state(srv.port)
            p = self.run_gm("--stride-gmmlu", "4")
            self.assertEqual(p.returncode, 0, p.stderr)
            m = self.result("gmmlu_he")["metrics"]["gmmlu_he"]
            self.assertEqual((m["n"], m["n_full"]), (6, 24))


# ----------------------------------------------------------------------------------------------------------
@unittest.skipUnless(testlib.ifeval_python() and testlib.nltk_data_dir()
                     and testlib.have_real_data("ifeval_input_data.jsonl", "ifeval_harness"),
                     "needs EVAL_TEST_DATA with ifeval_harness, EVAL_TEST_IFEVAL_PY and EVAL_TEST_NLTK_DATA")
class TestIfevalThroughTheRunner(GeneralCase):
    def test_ifeval_leg_with_the_pinned_harness(self):
        with FakeServer() as srv:
            self.write_state(srv.port)
            p = self.sh("--arm", ARM, "--evals", "ifeval", "--stride-ifeval", "40", py=testlib.ifeval_python(),
                        data=Path(REAL_DATA), NLTK_DATA=testlib.nltk_data_dir())
            self.assertEqual(p.returncode, 0, p.stderr)
            doc = self.result("ifeval")
            self.assertFalse(doc["pins"]["unpinned"], "the pinned harness check must have run")
            m = doc["metrics"]["ifeval_strict"]
            self.assertEqual((m["n"], m["n_full"]), (14, 541))
            self.assertEqual(self.report()["evals"]["ifeval"]["exit"], 0)


if __name__ == "__main__":
    sys.exit(unittest.main())
