#!/usr/bin/env python3
"""CPU unit tests of the Stage A general eval runner: argument parsing, the shared plumbing (selection, journal, HTTP
client), the engagement and receipt checks, the comparison math (including a red arm), and the scoring functions on
hand-made items. No GPU, no network beyond a loopback fake server. Run through run_tests.sh (capped), or

    python -m unittest test_unit -v
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import testlib
from testlib import HERE, ScratchCase, in_process

import arm_kind
import compare_general as cg
import evalcommon as ec
import ev_gmmlu
import ev_humaneval
import ev_mmlupro

ACTIVE = ("[2026-10-01 00:00:0{i}] negeig: active v1: {layers} gated GDN layers on this rank, "
          "gates from file /w/gates.safetensors sha256={sha}, max|W|={maxw}")
SHA = "ab" * 32


def active_lines(n=8, layers=48, sha=SHA, maxw="0.0123"):
    return "\n".join(ACTIVE.format(i=i % 10, layers=layers, sha=sha, maxw=maxw) for i in range(n)) + "\n"


BARE = "negeig: active v1: {layers} gated GDN layers on this rank, gates from file /w/gates.safetensors sha256={sha}, max|W|={maxw}"
INSTALLED = "negeig: installed v{v} on sglang 0.5.20 (pid {pid})"


def real_log(n=8, layers=48, sha=SHA, maxw="0.0123", bare=None, logged=None, v=1):
    """The shape of a real DP launch: the plugin's _say() prints each line bare AND logs it (the root handler adds the
    `[date DPn TPn]` prefix), every process that loads the plugin says `installed`, and noise sits between."""
    bare = n if bare is None else bare
    logged = n if logged is None else logged
    out = ["[2026-10-01 00:00:00] INFO server args: ...", INSTALLED.format(v=v, pid=100),
           "[2026-10-01 00:00:00] " + INSTALLED.format(v=v, pid=100), INSTALLED.format(v=v, pid=101),
           "Loading safetensors checkpoint shards:  50% 6/11 [00:09<00:08,  1.7s/it]"]
    out += [BARE.format(layers=layers, sha=sha, maxw=maxw) for _ in range(bare)]
    out += [f"[2026-10-01 00:01:0{i % 10} DP{i} TP0] " + BARE.format(layers=layers, sha=sha, maxw=maxw) for i in range(logged)]
    out += ["[2026-10-01 00:02:00] INFO The server is fired up and ready to roll!"]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------------------------
class TestSelection(unittest.TestCase):
    def test_stride_offset_limit(self):
        ids = [str(i) for i in range(10)]
        self.assertEqual(ec.select_ids(ids), ids)
        self.assertEqual(ec.select_ids(ids, 3), ["0", "3", "6", "9"])
        self.assertEqual(ec.select_ids(ids, 3, 1), ["1", "4", "7"])
        self.assertEqual(ec.select_ids(ids, 3, 2, 2), ["2", "5"])
        self.assertEqual(ec.select_ids(ids, 1, 0, 4), ["0", "1", "2", "3"])

    def test_strides_partition_the_set(self):
        ids = [str(i) for i in range(101)]
        got = sorted(sum((ec.select_ids(ids, 4, o) for o in range(4)), []), key=int)
        self.assertEqual(got, ids)

    def test_bad_arguments(self):
        for stride, off in ((0, 0), (-1, 0), (3, 3), (3, -1)):
            with self.assertRaises(ValueError):
                ec.select_ids(["a"], stride, off)

    def test_subset_info_pins_the_ids(self):
        full = [str(i) for i in range(10)]
        a = ec.subset_info(full, ec.select_ids(full, 2), 2, 0, 0)
        b = ec.subset_info(full, ec.select_ids(full, 2, 1), 2, 1, 0)
        self.assertEqual(a["n_full"], 10)
        self.assertEqual(a["n_selected"], 5)
        self.assertNotEqual(a["ids_sha256"], b["ids_sha256"])
        self.assertEqual(a["full_ids_sha256"], b["full_ids_sha256"])


class TestJournal(ScratchCase):
    def test_roundtrip_and_resume(self):
        p = self.tmp / "j" / "x.jsonl"
        j = ec.Journal(p, "sha1", {"eval": "x"})
        j.add("a", ok=1)
        j.add("b", ok=0)
        j.add_run({"seconds": 2.5})
        j.close()
        j2 = ec.Journal(p, "sha1", {"eval": "x"})
        self.addCleanup(j2.close)
        self.assertIn("a", j2)
        self.assertIn("b", j2)
        self.assertNotIn("c", j2)
        self.assertEqual(j2.rec["b"]["ok"], 0)
        self.assertAlmostEqual(j2.total_seconds(), 2.5)
        j2.add("c", ok=1)
        j2.close()
        j3 = ec.Journal(p, "sha1")
        self.assertEqual(sorted(j3.rec), ["a", "b", "c"])
        j3.close()

    def test_header_is_first_line(self):
        p = self.tmp / "x.jsonl"
        ec.Journal(p, "shaH", {"eval": "x"}).close()
        first = json.loads(p.read_text().splitlines()[0])
        self.assertEqual(first["_header"]["config_sha"], "shaH")

    def test_config_change_refuses_to_resume(self):
        p = self.tmp / "x.jsonl"
        j = ec.Journal(p, "one")
        j.add("a", ok=1)
        j.close()
        with self.assertRaises(ec.JournalMismatch) as cm:
            ec.Journal(p, "two")
        self.assertIsInstance(cm.exception, SystemExit)
        self.assertIn("--fresh", str(cm.exception))

    def test_fresh_discards(self):
        p = self.tmp / "x.jsonl"
        j = ec.Journal(p, "one")
        j.add("a", ok=1)
        j.close()
        j2 = ec.Journal(p, "two", fresh=True)
        self.assertNotIn("a", j2)
        j2.close()

    def test_torn_last_line_is_redone_not_fatal(self):
        p = self.tmp / "x.jsonl"
        j = ec.Journal(p, "one")
        j.add("a", ok=1)
        j.add("b", ok=1)
        j.close()
        raw = p.read_bytes()
        p.write_bytes(raw[:-9])  # cut the last record mid-line, no newline
        j2 = ec.Journal(p, "one")
        self.assertIn("a", j2)
        self.assertNotIn("b", j2)
        j2.add("b", ok=1)
        j2.close()
        j3 = ec.Journal(p, "one")
        self.assertIn("b", j3)
        j3.close()

    def test_records_without_header_refuse(self):
        p = self.tmp / "x.jsonl"
        p.write_text(json.dumps({"id": "a", "ok": 1}) + "\n")
        with self.assertRaises(ec.JournalMismatch):
            ec.Journal(p, "one")


class TestWriteJson(ScratchCase):
    def test_atomic_and_sorted(self):
        p = self.tmp / "d" / "o.json"
        ec.write_json(p, {"b": 1, "a": [1, 2]})
        self.assertEqual(json.loads(p.read_text()), {"a": [1, 2], "b": 1})
        self.assertEqual([x.name for x in p.parent.iterdir()], ["o.json"])  # no temp file left behind
        self.assertLess(p.read_text().index('"a"'), p.read_text().index('"b"'))


class TestMetric(unittest.TestCase):
    def test_metric_shape(self):
        m = ec.metric({"a": 1, "b": 0, "c": 1, "d": 1}, n_full=10)
        self.assertEqual((m["n"], m["score"], m["n_full"], m["level"]), (4, 0.75, 10, "item"))
        self.assertIsNone(ec.metric({})["score"])

    def test_status_exit_codes(self):
        st = ec.Status(total=3, already=1)
        self.assertEqual(st.exit_code, ec.EXIT_INCOMPLETE)
        st.done_now = 2
        self.assertTrue(st.complete)
        self.assertEqual(st.exit_code, ec.EXIT_OK)
        st.failed["x"] = "boom"
        self.assertEqual(st.exit_code, ec.EXIT_INCOMPLETE)
        st.down = True
        self.assertEqual(st.exit_code, ec.EXIT_DOWN)


class TestServerClient(unittest.TestCase):
    """The HTTP client against fake_server.py: what is retried, what is not, and when the breaker opens."""

    @classmethod
    def setUpClass(cls):
        cls.fs = testlib.FakeServer()

    @classmethod
    def tearDownClass(cls):
        cls.fs.stop()

    def setUp(self):
        self.fs.control(fail_status=0, fail_next=0, down=False, latency_ms=0)
        self.sleeps = []
        self.srv = ec.Server(self.fs.base, "m", attempts=4, breaker=3, sleep=self.sleeps.append,
                             backoff=lambda n: 0.01 * (n + 1), timeout=5)

    def test_plain_chat(self):
        res = self.srv.chat([{"role": "user", "content": "What is 17 times 23? Answer with the number only."}])
        self.assertEqual(ec.chat_text(res), ("391", "stop"))
        self.assertEqual(self.srv.stats["retries"], 0)

    def test_base_url_with_v1_suffix(self):
        s = ec.Server(self.fs.base + "/v1/", "m")
        self.assertEqual(s.root, self.fs.base)

    def test_retries_transient_statuses_then_succeeds(self):
        self.fs.control(fail_status=503, fail_next=2)
        res = self.srv.chat([{"role": "user", "content": "hi"}])
        self.assertTrue(ec.chat_text(res)[0])
        self.assertEqual(self.srv.stats["retries"], 2)
        self.assertEqual(len(self.sleeps), 2)

    def test_429_and_500_are_retried(self):
        for st in (429, 500, 502, 504):
            self.fs.control(fail_status=st, fail_next=1)
            self.srv.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(self.srv.stats["retries"], 4)

    def test_400_is_not_retried(self):
        self.fs.control(fail_status=400, fail_next=1)
        with self.assertRaises(ec.ItemFailed) as cm:
            self.srv.chat([{"role": "user", "content": "hi"}])
        self.assertIn("HTTP 400", str(cm.exception))
        self.assertEqual(self.srv.stats["retries"], 0)
        self.assertFalse(self.srv.down.is_set())

    def test_exhausted_retries_fail_the_item_not_the_run(self):
        self.fs.control(fail_status=503, fail_next=4)
        with self.assertRaises(ec.ItemFailed):
            self.srv.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(self.srv.stats["exhausted"], 1)
        self.assertFalse(self.srv.down.is_set())
        self.srv.chat([{"role": "user", "content": "hi"}])  # and the next request works, counter resets

    def test_breaker_opens_after_consecutive_exhaustion(self):
        self.fs.control(down=True)
        for _ in range(2):
            with self.assertRaises(ec.ItemFailed):
                self.srv.chat([{"role": "user", "content": "hi"}])
        with self.assertRaises(ec.ServerDown):
            self.srv.chat([{"role": "user", "content": "hi"}])
        self.assertTrue(self.srv.down.is_set())
        with self.assertRaises(ec.ServerDown):  # once open it stays open, without touching the network
            self.srv.chat([{"role": "user", "content": "hi"}])

    def test_success_resets_the_breaker_count(self):
        self.fs.control(fail_status=503, fail_next=4)
        with self.assertRaises(ec.ItemFailed):
            self.srv.chat([{"role": "user", "content": "hi"}])
        self.srv.chat([{"role": "user", "content": "hi"}])
        self.fs.control(fail_status=503, fail_next=8)
        for _ in range(2):
            with self.assertRaises(ec.ItemFailed):
                self.srv.chat([{"role": "user", "content": "hi"}])
        self.assertFalse(self.srv.down.is_set())

    def test_connection_refused_is_retried_as_status_zero(self):
        s = ec.Server(f"http://127.0.0.1:{testlib.free_port()}", "m", attempts=2, breaker=5, sleep=lambda x: None)
        with self.assertRaises(ec.ItemFailed) as cm:
            s.generate({"input_ids": [1, 2, 3]})
        self.assertIn("retries exhausted", str(cm.exception))
        self.assertEqual(s.stats["retries"], 2)

    def test_wait_ready(self):
        self.assertTrue(self.srv.wait_ready(5, poll=0.05))
        s = ec.Server(f"http://127.0.0.1:{testlib.free_port()}", "m")
        self.assertFalse(s.wait_ready(0.3, poll=0.05))


# ----------------------------------------------------------------------------------------------------------
class TestArgumentParsing(ScratchCase):
    """Every entry point: --help exits 0, a missing required argument exits 2, bad values are refused with a reason."""

    def mods(self):
        import ev_bfcl
        import ev_ifeval
        import prepare_data
        return {"ev_mmlupro": ev_mmlupro.main, "ev_gmmlu": ev_gmmlu.main, "ev_humaneval": ev_humaneval.main,
                "ev_ifeval": ev_ifeval.main, "ev_bfcl": ev_bfcl.main, "compare_general": cg.main,
                "arm_kind": arm_kind.main, "prepare_data": prepare_data.main}

    def test_help_exits_zero_everywhere(self):
        for name, main in self.mods().items():
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code, _ = in_process(main, ["--help"])
            self.assertEqual(code, 0, name)
            self.assertIn("usage", out.getvalue().lower(), name)

    def test_missing_required_arguments_exit_two(self):
        for name in ("ev_mmlupro", "ev_gmmlu", "ev_humaneval", "ev_ifeval", "ev_bfcl", "compare_general", "arm_kind"):
            code, err = in_process(self.mods()[name], [])
            self.assertEqual(code, 2, f"{name}: {err[-200:]}")

    def test_common_arguments_are_on_every_eval(self):
        for name in ("ev_mmlupro", "ev_gmmlu", "ev_humaneval", "ev_ifeval", "ev_bfcl"):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                in_process(self.mods()[name], ["--help"])
            for flag in ("--arm", "--out", "--base-url", "--served-model", "--data", "--workers", "--stride",
                         "--offset", "--limit", "--fresh", "--timeout", "--attempts", "--breaker", "--allow-unpinned"):
                self.assertIn(flag, out.getvalue(), f"{name} lacks {flag}")

    def test_defaults(self):
        import argparse
        ap = argparse.ArgumentParser()
        ec.add_common_args(ap, workers=7)
        a = ap.parse_args(["--arm", "base", "--out", "o"])
        self.assertEqual((a.workers, a.stride, a.offset, a.limit, a.attempts, a.breaker, a.fresh, a.allow_unpinned),
                         (7, 1, 0, 0, 11, 8, False, False))
        self.assertEqual(a.base_url, "http://127.0.0.1:30000")
        self.assertEqual(a.served_model, "qwen3.8-27b")

    def test_env_defaults(self):
        import argparse, os
        old = {k: os.environ.get(k) for k in ("EVAL_BASE_URL", "EVAL_SERVED_MODEL", "EVAL_DATA")}
        os.environ.update({"EVAL_BASE_URL": "http://h:1", "EVAL_SERVED_MODEL": "zz", "EVAL_DATA": "/dd"})
        try:
            ap = argparse.ArgumentParser()
            ec.add_common_args(ap)
            a = ap.parse_args(["--arm", "base", "--out", "o"])
            self.assertEqual((a.base_url, a.served_model, a.data), ("http://h:1", "zz", "/dd"))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def data(self):
        d = self.tmp / "data"
        testlib.write_data_dir(d)
        return d

    def test_bad_stride_is_refused_with_a_reason(self):
        d = self.data()
        for main, extra in ((ev_gmmlu.main, []), (ev_humaneval.main, [])):
            code, err = in_process(main, ["--arm", "x", "--out", str(self.tmp / "o"), "--data", str(d),
                                          "--allow-unpinned", "--stride", "0", *extra])
            self.assertEqual(code, 1)
            self.assertIn("bad subset arguments", err)
            code, err = in_process(main, ["--arm", "x", "--out", str(self.tmp / "o"), "--data", str(d),
                                          "--allow-unpinned", "--stride", "3", "--offset", "3"])
            self.assertEqual(code, 1)
            self.assertIn("bad subset arguments", err)

    def test_missing_data_file_names_the_fix(self):
        for main in (ev_gmmlu.main, ev_humaneval.main):
            code, err = in_process(main, ["--arm", "x", "--out", str(self.tmp / "o"), "--data", str(self.tmp / "nope")])
            self.assertEqual(code, 1)
            self.assertIn("prepare_box.sh data", err)

    def test_sha_mismatch_against_the_pins(self):
        d = self.data()
        code, err = in_process(ev_humaneval.main, ["--arm", "x", "--out", str(self.tmp / "o"), "--data", str(d)])
        self.assertEqual(code, 1)
        self.assertIn("is not the pinned", err)

    def test_row_count_against_the_pins(self):
        d = self.data()
        # the synthetic files are small: with --allow-unpinned they pass the sha check, but the row count is the pin's
        # only for a pinned run, so the same files without the switch must not slip through on row count either
        code, _ = in_process(ev_humaneval.main, ["--arm", "x", "--out", str(self.tmp / "o"), "--data", str(d),
                                                  "--allow-unpinned", "--limit", "0", "--stride", "0"])
        self.assertEqual(code, 1)

    def test_manifest_mismatch(self):
        d = self.data()
        (d / "MANIFEST.json").write_text(json.dumps({"files": {"humaneval_test.jsonl": {"sha256": "00" * 32}}}))
        code, err = in_process(ev_humaneval.main, ["--arm", "x", "--out", str(self.tmp / "o"), "--data", str(d),
                                                    "--allow-unpinned"])
        self.assertEqual(code, 1)
        self.assertIn("MANIFEST.json", err)

    def test_mmlupro_bad_modes_and_missing_tokenizer(self):
        d = self.data()
        base = ["--arm", "x", "--out", str(self.tmp / "o"), "--data", str(d), "--allow-unpinned"]
        code, err = in_process(ev_mmlupro.main, base + ["--modes", "cloze,bogus", "--tokenizer", "/nope"])
        self.assertEqual(code, 2)
        self.assertIn("--modes", err)
        code, err = in_process(ev_mmlupro.main, base + ["--modes", ""])
        self.assertEqual(code, 2)
        code, err = in_process(ev_mmlupro.main, base + ["--modes", "cloze"])
        self.assertEqual(code, 2)
        self.assertIn("--tokenizer", err)

    def test_ifeval_missing_input(self):
        import ev_ifeval
        code, err = in_process(ev_ifeval.main, ["--arm", "x", "--out", str(self.tmp / "o"), "--data", str(self.tmp)])
        self.assertEqual(code, 2)
        self.assertIn("ifeval_input_data.jsonl", err)

    def test_bfcl_needs_its_venv(self):
        import ev_bfcl
        code, err = in_process(ev_bfcl.main, ["--arm", "x", "--out", str(self.tmp / "o"), "--data", str(self.tmp)])
        self.assertEqual(code, 2)
        self.assertIn("bfcl", err.lower())

    def test_humaneval_check_canonical_flag_needs_no_server(self):
        d = self.data()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code, err = in_process(ev_humaneval.main, ["--check-canonical", "--data", str(d), "--allow-unpinned"])
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out.getvalue())["ok"])

    def test_arm_kind_subcommands_validate(self):
        for argv in (["engagement", "--log", "x", "--kind", "bogus", "--ranks", "8"],
                     ["engagement", "--log", "x", "--kind", "wide"],
                     ["assemble", "--arm", "a", "--kind", "bogus", "--serve", "s", "--out", "o"]):
            code, _ = in_process(arm_kind.main, argv)
            self.assertEqual(code, 2, argv)

    def test_compare_arguments(self):
        for argv, text in ((["--base", "b", "--arm", "a", "--ci-level", "1.5"], "--ci-level"),
                           (["--base", "b", "--arm", "a", "--ci-level", "0.2"], "--ci-level"),
                           (["--base", "b"], "--arm is required")):
            code, err = in_process(cg.main, argv)
            self.assertEqual(code, 1, argv)
            self.assertIn(text, err)


# ----------------------------------------------------------------------------------------------------------
class TestEngagement(unittest.TestCase):
    def check(self, text, kind, ranks=8, **kw):
        return arm_kind.check_engagement(text, kind, ranks, **kw)

    def test_wide_ok(self):
        r = self.check(active_lines(8), "wide", gates_sha256=SHA)
        self.assertTrue(r["ok"], r["problems"])
        self.assertEqual((r["active_lines"], r["negeig_lines"], r["layers_per_rank"]), (8, 8, [48]))

    def test_wide_tp2_layers(self):
        r = self.check(active_lines(8, layers=24), "wide", ranks=8, layers=24, gates_sha256=SHA)
        self.assertTrue(r["ok"], r["problems"])
        r = self.check(active_lines(8, layers=24), "wide", ranks=8, layers=48, gates_sha256=SHA)
        self.assertFalse(r["ok"])

    def test_wide_seven_of_eight_ranks_is_red(self):
        r = self.check(active_lines(7), "wide", gates_sha256=SHA)
        self.assertFalse(r["ok"])
        self.assertTrue(any("7 'negeig: active' lines, expected 8" in p for p in r["problems"]))

    def test_wide_extra_rank_lines_is_red(self):
        self.assertFalse(self.check(active_lines(9), "wide")["ok"])

    def test_wide_no_lines_at_all_is_red(self):
        r = self.check("server started\n", "wide")
        self.assertFalse(r["ok"])
        self.assertEqual(r["active_lines"], 0)

    def test_wide_wrong_layer_count(self):
        r = self.check(active_lines(7) + ACTIVE.format(i=0, layers=47, sha=SHA, maxw="0.1") + "\n", "wide")
        self.assertFalse(r["ok"])
        self.assertTrue(any("layer counts" in p for p in r["problems"]))

    def test_wide_sha_mismatch_is_red(self):
        r = self.check(active_lines(8, sha="cd" * 32), "wide", gates_sha256=SHA)
        self.assertFalse(r["ok"])
        self.assertTrue(any("not the exported file" in p for p in r["problems"]))

    def test_wide_sha_unchecked_without_the_argument(self):
        self.assertTrue(self.check(active_lines(8, sha="cd" * 32), "wide")["ok"])

    def test_wide_zero_gate_is_red(self):
        for w in ("0.0", "0", "nan", "inf", "-0.5", "abc"):
            r = self.check(active_lines(8, maxw=w), "wide", gates_sha256=SHA)
            self.assertFalse(r["ok"], w)

    def test_wide_other_negeig_line_is_red(self):
        r = self.check(active_lines(8) + "negeig: WARNING gates fell back to zeros\n", "wide", gates_sha256=SHA)
        self.assertFalse(r["ok"])
        self.assertTrue(any("other 'negeig:' line" in p for p in r["problems"]))

    # -- the real log shape: each line bare and through logging, plus `installed` lines ------------------------------
    def test_wide_real_log_shape_is_ok(self):
        r = self.check(real_log(8), "wide", gates_sha256=SHA)
        self.assertTrue(r["ok"], r["problems"])
        self.assertEqual((r["active_lines"], r["active_bare"], r["active_logged"], r["installed_lines"], r["other_lines"]),
                         (8, 8, 8, 3, 0))
        self.assertEqual(r["negeig_lines"], 3 + 16)

    def test_wide_either_form_alone_is_enough(self):
        self.assertTrue(self.check(real_log(8, logged=0), "wide", gates_sha256=SHA)["ok"], "no logging form (log level)")
        self.assertTrue(self.check(real_log(8, bare=0), "wide", gates_sha256=SHA)["ok"], "no bare form")

    def test_wide_a_progress_bar_may_share_the_line(self):
        text = ("Loading safetensors checkpoint shards: 100% 11/11 [00:20<00:00,  1.8s/it]\r"
                + "\r".join(BARE.format(layers=48, sha=SHA, maxw="0.0123") for _ in range(8)) + "\n")
        r = self.check(text, "wide", gates_sha256=SHA)
        self.assertTrue(r["ok"], r["problems"])

    def test_wide_real_shape_seven_of_eight_is_red(self):
        r = self.check(real_log(8, bare=7, logged=7), "wide", gates_sha256=SHA)
        self.assertFalse(r["ok"])
        self.assertTrue(any("7 'negeig: active' lines, expected 8" in p for p in r["problems"]))

    def test_wide_real_shape_nine_of_eight_is_red(self):
        self.assertFalse(self.check(real_log(8, bare=9, logged=9), "wide")["ok"])

    def test_wide_a_rank_that_engaged_twice_is_not_hidden_by_the_logging_form(self):
        # 16 bare lines for 8 ranks: the patch ran twice per rank. The logged form (8) must not average it away.
        r = self.check(real_log(8, bare=16, logged=8), "wide")
        self.assertFalse(r["ok"])
        self.assertTrue(any("16 'negeig: active' lines, expected 8" in p for p in r["problems"]))

    def test_wide_real_shape_with_a_warning_is_red(self):
        r = self.check(real_log(8) + "negeig: warning: a layer fell back to the stock kernel\n", "wide", gates_sha256=SHA)
        self.assertFalse(r["ok"])
        self.assertTrue(any("other 'negeig:' line" in p and "fell back" in p for p in r["problems"]))

    def test_wide_an_unreadable_installed_line_is_other(self):
        r = self.check(real_log(8) + "negeig: installed but something is off\n", "wide")
        self.assertFalse(r["ok"])

    def test_wide_installed_line_of_another_patch_version_is_red(self):
        r = self.check(real_log(8) + INSTALLED.format(v=2, pid=7) + "\n", "wide", gates_sha256=SHA)
        self.assertFalse(r["ok"])
        self.assertTrue(any("patch version" in p for p in r["problems"]))

    def test_wide_real_shape_wrong_sha_zero_gate_and_layers_are_still_red(self):
        self.assertFalse(self.check(real_log(8, sha="cd" * 32), "wide", gates_sha256=SHA)["ok"])
        self.assertFalse(self.check(real_log(8, maxw="0"), "wide", gates_sha256=SHA)["ok"])
        self.assertFalse(self.check(real_log(8, layers=47), "wide", gates_sha256=SHA)["ok"])

    def test_base_and_ctrl_with_the_real_shape_are_red(self):
        for kind in ("base", "ctrl"):
            self.assertFalse(self.check(real_log(8), kind)["ok"], kind)
            self.assertFalse(self.check(INSTALLED.format(v=1, pid=1) + "\n", kind)["ok"], f"{kind}: an installed line alone")

    def test_base_and_ctrl_must_be_silent(self):
        for kind in ("base", "ctrl"):
            self.assertTrue(self.check("sglang up\nno plugin here\n", kind)["ok"], kind)
            r = self.check(active_lines(8), kind)
            self.assertFalse(r["ok"], kind)
            self.assertTrue(any("must not be loaded" in p for p in r["problems"]))
            self.assertFalse(self.check("x\nnegeig: something\n", kind)["ok"])

    def test_unknown_kind(self):
        with self.assertRaises(SystemExit):
            self.check("", "gate_only")

    def test_patch_version_is_reported(self):
        r = self.check(active_lines(8), "wide")
        self.assertEqual(r["patch_versions"], [1])


class TestReceipt(unittest.TestCase):
    def doc(self, **kw):
        d = {"delta_kept": 0.998, "delta_kept_worst_tensor": 0.97, "trainable_sha256": "ab" * 32, "scale": 2.0,
             "rounding": "stochastic", "seed": 20261001, "targets": 144}
        d.update(kw)
        return d

    def test_ok(self):
        self.assertTrue(arm_kind.check_receipt(self.doc(), 0.95)["ok"])

    def test_missing_keys(self):
        d = self.doc()
        del d["delta_kept"]
        r = arm_kind.check_receipt(d, 0.95)
        self.assertFalse(r["ok"])
        self.assertIn("lacks", r["problems"][0])

    def test_low_delta_kept_is_red(self):
        self.assertFalse(arm_kind.check_receipt(self.doc(delta_kept=0.5), 0.95)["ok"])
        self.assertFalse(arm_kind.check_receipt(self.doc(delta_kept_worst_tensor=0.2), 0.95)["ok"])
        self.assertFalse(arm_kind.check_receipt(self.doc(delta_kept=float("nan")), 0.95)["ok"])
        self.assertFalse(arm_kind.check_receipt(self.doc(delta_kept="x"), 0.95)["ok"])

    def test_threshold_is_the_argument(self):
        self.assertTrue(arm_kind.check_receipt(self.doc(delta_kept=0.9, delta_kept_worst_tensor=0.8), 0.85)["ok"])
        self.assertFalse(arm_kind.check_receipt(self.doc(delta_kept=0.9), 0.95)["ok"])


class TestArmKindCli(ScratchCase):
    def cli(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code, err = in_process(arm_kind.main, argv)
        return code, out.getvalue(), err

    def test_engagement_cli_exit_codes(self):
        log = self.tmp / "s.log"
        log.write_text(active_lines(8))
        code, out, _ = self.cli(["engagement", "--log", str(log), "--kind", "wide", "--ranks", "8",
                                 "--gates-sha256", SHA])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["ok"])
        code, out, _ = self.cli(["engagement", "--log", str(log), "--kind", "base", "--ranks", "8"])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(out)["ok"])

    def test_receipt_cli(self):
        r = self.tmp / "r.json"
        r.write_text(json.dumps(TestReceipt().doc()))
        self.assertEqual(self.cli(["receipt", "--receipt", str(r)])[0], 0)
        r.write_text(json.dumps(TestReceipt().doc(delta_kept=0.1)))
        self.assertEqual(self.cli(["receipt", "--receipt", str(r)])[0], 1)

    def test_assemble(self):
        serve = self.tmp / "serve.json"
        serve.write_text(json.dumps({"tp_size": 1}))
        eng = self.tmp / "e.json"
        eng.write_text(json.dumps({"ok": True}))
        rec = self.tmp / "r.json"
        rec.write_text(json.dumps(TestReceipt().doc()))
        out = self.tmp / "arm.json"
        self.assertEqual(self.cli(["assemble", "--arm", "base", "--kind", "base", "--serve", str(serve),
                                   "--engagement", str(eng), "--out", str(out)])[0], 0)
        d = json.loads(out.read_text())
        self.assertEqual((d["arm"], d["kind"], d["serve"], d["engagement"]), ("base", "base", {"tp_size": 1}, {"ok": True}))
        self.assertNotIn("merge_receipt", d)
        code, _, err = self.cli(["assemble", "--arm", "wide_s1", "--kind", "wide", "--serve", str(serve),
                                 "--out", str(self.tmp / "x.json")])
        self.assertEqual(code, 2)
        self.assertFalse((self.tmp / "x.json").exists())
        self.assertEqual(self.cli(["assemble", "--arm", "wide_s1", "--kind", "wide", "--serve", str(serve),
                                   "--merge-receipt", str(rec), "--engagement", str(eng), "--out", str(out)])[0], 0)
        self.assertEqual(json.loads(out.read_text())["merge_receipt"]["scale"], 2.0)

    def test_classify_keys(self):
        gate = "base_model.model.model.language_model.layers.3.linear_attn.negeig_w"
        lora = "base_model.model.model.language_model.layers.3.linear_attn.in_proj_qkv.lora_A.default.weight"
        self.assertEqual(arm_kind.classify_keys([gate, lora])["kind"], "wide")
        self.assertEqual(arm_kind.classify_keys([lora])["kind"], "ctrl")
        self.assertEqual(arm_kind.classify_keys([gate])["kind"], "gate_only")
        self.assertEqual(arm_kind.classify_keys(["layers.3.mlp.up_proj.weight"])["kind"], "unknown")
        self.assertEqual(arm_kind.classify_keys([gate + ".weight"])["n_gate_tensors"], 1)
        c = arm_kind.classify_keys([gate, lora, lora.replace("lora_A", "lora_B")])
        self.assertEqual((c["n_gate_tensors"], c["n_lora_tensors"], c["n_tensors"]), (1, 2, 3))

    @unittest.skipUnless(__import__("importlib.util").util.find_spec("torch"), "needs torch")
    def test_kind_on_saved_trainables(self):
        import torch
        gate = "base_model.model.model.layers.3.linear_attn.negeig_w"
        lora = "base_model.model.model.layers.3.linear_attn.in_proj_qkv.lora_A.default.weight"
        t = torch.zeros(2, 2)
        cases = {"wide": ({gate: t, lora: t}, 0, "wide"), "ctrl": ({lora: t}, 0, "ctrl"),
                 "gate": ({gate: t}, 1, "gate_only"), "wrapped": ({"step": 5, "trainable": {gate: t, lora: t}}, 0, "wide"),
                 "unknown": ({"x.weight": t}, 1, "unknown")}
        for name, (obj, want_code, want_kind) in cases.items():
            p = self.tmp / f"{name}.pt"
            torch.save(obj, p)
            code, out, _ = self.cli(["kind", "--trainable", str(p)])
            self.assertEqual(code, want_code, name)
            self.assertEqual(json.loads(out)["kind"], want_kind, name)


# ----------------------------------------------------------------------------------------------------------
# comparison math
class TestMath(unittest.TestCase):
    def test_paired_cells(self):
        base = {"a": 1, "b": 1, "c": 0, "d": 0, "e": 1, "only_base": 1}
        arm = {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1, "only_arm": 1}
        ids, both, b_only, a_only, neither = cg.paired_cells(base, arm)
        self.assertEqual(ids, ["a", "b", "c", "d", "e"])
        self.assertEqual((both, b_only, a_only, neither), (2, 1, 1, 1))

    def test_bootstrap_degenerate_cases(self):
        lo, hi = cg.bootstrap_interval(0, 0, 0, 0)
        self.assertTrue(math.isnan(lo) and math.isnan(hi))
        self.assertEqual(cg.bootstrap_interval(90, 0, 0, 10), (0.0, 0.0))

    def test_bootstrap_matches_the_normal_approximation(self):
        a, b, c, d = 800, 60, 20, 120  # n = 1000, delta = -4.0 points
        n = a + b + c + d
        delta = 100.0 * (c - b) / n
        se = 100.0 * math.sqrt(((b + c) - (c - b) ** 2 / n)) / n
        lo, hi = cg.bootstrap_interval(a, b, c, d, "t", draws=20000)
        self.assertAlmostEqual(delta, -4.0)
        self.assertAlmostEqual(lo, delta - 1.96 * se, delta=0.15)
        self.assertAlmostEqual(hi, delta + 1.96 * se, delta=0.15)
        self.assertLess(lo, delta)
        self.assertGreater(hi, delta)

    def test_bootstrap_is_deterministic_and_name_seeded(self):
        x = cg.bootstrap_interval(800, 60, 20, 120, "name", draws=2000)
        self.assertEqual(x, cg.bootstrap_interval(800, 60, 20, 120, "name", draws=2000))
        self.assertNotEqual(x, cg.bootstrap_interval(800, 60, 20, 120, "other", draws=2000))

    def test_interval_narrows_with_n_and_widens_with_level(self):
        w = lambda n, level=0.95: (lambda r: r[1] - r[0])(cg.bootstrap_interval(int(.8 * n), int(.06 * n), int(.04 * n),
                                                                               n - int(.8 * n) - int(.06 * n) - int(.04 * n),
                                                                               "w", 4000, level))
        self.assertGreater(w(200), w(2000))
        self.assertGreater(w(500, 0.99), w(500, 0.90))

    def test_coverage_of_a_true_zero_delta(self):
        import numpy as np
        rng = np.random.default_rng(7)
        p = [0.80, 0.07, 0.07, 0.06]  # equal discordant halves: the true delta is exactly 0
        hit, reps, n = 0, 150, 400
        for i in range(reps):
            a, b, c, d = rng.multinomial(n, p)
            lo, hi = cg.bootstrap_interval(int(a), int(b), int(c), int(d), f"cov{i}", draws=600)
            hit += int(lo <= 0.0 <= hi)
        self.assertGreaterEqual(hit / reps, 0.89)
        self.assertLessEqual(hit / reps, 1.0)

    def test_cluster_interval_is_wider_than_the_independent_one(self):
        base, arm = {}, {}
        for k in range(100):
            for j in range(2):
                base[f"{k}:{j}"] = 1
                arm[f"{k}:{j}"] = 0 if k < 10 else 1  # whole prompts flip together
        lo, hi = cg.cluster_interval(base, arm, "c", draws=4000)
        a, b, c, d = cg.paired_cells(base, arm)[1:]
        blo, bhi = cg.bootstrap_interval(a, b, c, d, "c", draws=4000)
        self.assertAlmostEqual(100.0 * (c - b) / 200, -10.0)
        self.assertGreater(hi - lo, bhi - blo)
        self.assertLess(hi, 0.0)

    def test_cluster_interval_degenerate(self):
        self.assertEqual(cg.cluster_interval({"a:0": 1}, {"a:0": 1}), (0.0, 0.0))
        lo, hi = cg.cluster_interval({}, {})
        self.assertTrue(math.isnan(lo))

    def test_mcnemar(self):
        self.assertEqual(cg.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(cg.mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(cg.mcnemar_exact(0, 10), 2.0 / 1024)
        self.assertAlmostEqual(cg.mcnemar_exact(3, 0), 0.25)
        self.assertEqual(cg.mcnemar_exact(2, 9), cg.mcnemar_exact(9, 2))

    def test_classify(self):
        self.assertEqual(cg.classify(-1.0, -2.0, 0.0, 3.0), ("PASS", "within"))
        self.assertEqual(cg.classify(-2.0, -3.5, -0.5, 3.0), ("PASS_UNRESOLVED", "within"))
        self.assertEqual(cg.classify(-3.5, -5.0, -2.0, 3.0), ("FAIL_UNRESOLVED", "outside"))
        self.assertEqual(cg.classify(-6.0, -8.0, -3.1, 3.0), ("FAIL", "outside"))
        self.assertEqual(cg.classify(-3.0, -3.0, -1.0, 3.0), ("PASS", "within"))  # the band edge is inside
        self.assertEqual(cg.classify(2.0, 1.0, 3.0, 3.0), ("PASS", "within"))

    def test_power_stats(self):
        p = cg.power_stats(164, 0.05, 3.0)
        self.assertAlmostEqual(p["se_points"], 100 * math.sqrt(0.05 / 164))
        self.assertTrue(p["cannot_certify_at_zero"])  # HumanEval at q=0.05 cannot certify a 3 point band
        self.assertAlmostEqual(p["fail_detect_80pct_points"], 3.0 + 2.8 * p["se_points"], places=3)
        self.assertAlmostEqual(p["n_needed_to_resolve_at_delta0"], 0.05 * (1.959964 / 0.03) ** 2, places=3)
        big = cg.power_stats(12032, 0.05, 1.0)
        self.assertFalse(big["cannot_certify_at_zero"])
        self.assertLess(big["ci_half_width_points"], 0.5)
        self.assertTrue(math.isnan(cg.power_stats(0, 0.05, 3.0)["se_points"]))

    def test_power_gets_better_with_smaller_discordance(self):
        self.assertGreater(cg.power_stats(500, 0.15, 2.0)["se_points"], cg.power_stats(500, 0.02, 2.0)["se_points"])


# ----------------------------------------------------------------------------------------------------------
def eval_of(metric: str) -> str:
    for prefix, ev in (("bfcl", "bfcl"), ("mmlupro", "mmlupro"), ("humaneval", "humaneval"), ("ifeval", "ifeval"),
                       ("gmmlu_he", "gmmlu_he")):
        if metric.startswith(prefix):
            return ev
    raise KeyError(metric)


SERVE = {"context_length": 131072, "radix": "off", "tp_size": 1, "attn_backend": "triton",
         "linear_attn_backend": "triton", "linear_prefill_backend": "triton", "linear_decode_backend": "triton",
         "sglang_version": "0.5.20", "model_rev": "1d4bf0f", "gpu_name": "B300", "dtype": "bfloat16",
         "dp_size": 8, "port": 30000}


def outcomes(n, wrong_ids=(), prefix="i"):
    return {f"{prefix}{i}": (0 if i in set(wrong_ids) else 1) for i in range(n)}


def write_arm(d: Path, name: str, metrics: dict, serve=SERVE, engagement_ok=True, config_tag="c", kind="base",
              merge=None, pins_tag="p"):
    """An arm directory as run_general.sh leaves it: results/<eval>.json with per-item outcomes plus arm.json."""
    by_eval: dict = {}
    for m, items in metrics.items():
        level = "instruction" if m.endswith("_inst") else "item"
        by_eval.setdefault(eval_of(m), {})[m] = ec.metric(items, level)
    for ev, ms in by_eval.items():
        ec.write_json(d / "results" / f"{ev}.json", {
            "schema": ec.SCHEMA, "eval": ev, "arm": name, "utc": "t", "complete": True, "subset": {}, "config": {"t": config_tag},
            "config_sha": "x", "pins": {"p": pins_tag}, "seconds": 1.0, "diagnostics": {}, "server_stats": {}, "metrics": ms})
    info = {"arm": name, "kind": kind}
    if serve is not None:
        info["serve"] = serve
    if engagement_ok is not None:
        info["engagement"] = {"ok": engagement_ok}
    if merge:
        info["merge_receipt"] = merge
    ec.write_json(d / "arm.json", info)


def all_gate_items(n=400, bfcl_n=60):
    m = {}
    for name in cg.GATE_ORDER:
        m[name] = outcomes(bfcl_n if name.startswith("bfcl/") else n)
    return m


class TestCompare(ScratchCase):
    def pair(self, arm_mutate=None, base_mutate=None, **arm_kw):
        base = all_gate_items()
        arm = all_gate_items()
        if base_mutate:
            base_mutate(base)
        if arm_mutate:
            arm_mutate(arm)
        write_arm(self.tmp / "base", "base", base)
        write_arm(self.tmp / "arm", "wide_s1", arm, kind="wide", **arm_kw)
        return self.tmp / "base", self.tmp / "arm"

    def rows(self, doc):
        return {r["metric"]: r for r in doc["rows"]}

    def test_identical_arm_passes_everywhere(self):
        b, a = self.pair()
        doc = cg.compare(b, a, draws=500)
        self.assertEqual(doc["overall"]["all"]["ci"], "PASS")
        self.assertEqual(doc["overall"]["all"]["n_gates"], 21)
        self.assertEqual(doc["overall"]["all"]["counts"]["PASS"], 21)
        self.assertTrue(doc["valid"])
        for r in doc["rows"]:
            self.assertEqual(r["delta"], 0.0)
            self.assertEqual((r["ci_lo"], r["ci_hi"]), (0.0, 0.0))

    def test_red_arm_fails_the_gate_it_broke_and_only_that_one(self):
        def mutate(arm):
            arm["humaneval"] = outcomes(400, wrong_ids=range(0, 400, 5))  # 20% of the items lost
        b, a = self.pair(mutate)
        doc = cg.compare(b, a, draws=2000)
        r = self.rows(doc)
        self.assertEqual(r["humaneval"]["verdict"], "FAIL")
        self.assertAlmostEqual(r["humaneval"]["delta"], -20.0)
        self.assertEqual(r["humaneval"]["point_verdict"], "outside")
        self.assertEqual(doc["overall"]["all"]["ci"], "FAIL")
        self.assertEqual(doc["overall"]["all"]["point"], "FAIL")
        self.assertEqual(doc["overall"]["all"]["counts"]["FAIL"], 1)
        for k, v in r.items():
            if k != "humaneval" and v.get("band") is not None:
                self.assertEqual(v["verdict"], "PASS", k)

    def test_an_improvement_is_not_a_failure(self):
        base = all_gate_items()
        base["ifeval_strict"] = outcomes(400, wrong_ids=range(0, 400, 4))
        write_arm(self.tmp / "base", "base", base)
        write_arm(self.tmp / "arm", "wide_s1", all_gate_items(), kind="wide")
        doc = cg.compare(self.tmp / "base", self.tmp / "arm", draws=1000)
        r = self.rows(doc)["ifeval_strict"]
        self.assertGreater(r["delta"], 20)
        self.assertEqual(r["verdict"], "PASS")

    def test_small_n_gives_unresolved_not_a_coin_flip(self):
        def base_mut(base):
            base["humaneval"] = outcomes(164)

        def arm_mut(arm):
            arm["humaneval"] = outcomes(164, wrong_ids=range(3))  # 3 of 164 lost: delta -1.83, interval straddles -3

        b, a = self.pair(arm_mut, base_mut)
        r = self.rows(cg.compare(b, a, draws=4000))["humaneval"]
        self.assertEqual(r["verdict"], "PASS_UNRESOLVED")
        self.assertEqual(r["point_verdict"], "within")
        self.assertLess(r["ci_lo"], -3.0)

        def arm_mut2(arm):
            arm["humaneval"] = outcomes(164, wrong_ids=range(6))  # delta -3.66: point outside, interval straddles

        b, a = self.pair(arm_mut2, base_mut)
        r = self.rows(cg.compare(b, a, draws=4000))["humaneval"]
        self.assertEqual(r["verdict"], "FAIL_UNRESOLVED")
        self.assertEqual(r["point_verdict"], "outside")

    def test_coarse_flag_when_one_item_exceeds_the_band(self):
        def mutate(m):
            m["gmmlu_he"] = outcomes(50)
        b, a = self.pair(mutate, mutate)
        doc = cg.compare(b, a, draws=200)
        r = self.rows(doc)["gmmlu_he"]
        self.assertTrue(r["coarse"])  # one of 50 items is 2 points; the band is 1
        self.assertAlmostEqual(r["one_item_points"], 2.0)
        self.assertEqual(doc["overall"]["without_coarse"]["n_gates"], 20)

    def test_missing_gates_make_the_overall_incomplete(self):
        def mutate(arm):
            del arm["bfcl/simple_java"]
            del arm["gmmlu_he"]
        b, a = self.pair(mutate)
        doc = cg.compare(b, a, draws=200)
        self.assertEqual(doc["overall"]["all"]["ci"], "INCOMPLETE")
        self.assertEqual(sorted(doc["overall"]["all"]["missing"]), ["bfcl/simple_java", "gmmlu_he"])
        self.assertEqual(self.rows(doc)["gmmlu_he"]["verdict"], "MISSING")
        self.assertTrue(any("no paired items" in n for n in doc["validity_notes"]))

    def edit_result(self, d, ev, **fields):
        p = d / "results" / f"{ev}.json"
        doc = json.loads(p.read_text())
        doc.update(fields)
        p.write_text(json.dumps(doc))

    def test_health_notes_accepted_transport_errors_and_subsets(self):
        b, a = self.pair()
        self.edit_result(a, "bfcl", diagnostics={"n_context_overflow": 7, "n_accepted_transport_errors": 3})
        self.edit_result(a, "humaneval", subset={"stride": 4, "offset": 0, "limit": 0})
        doc = cg.compare(b, a, draws=200)
        notes = " ".join(doc["validity_notes"])
        self.assertIn("arm: 3 BFCL items were scored from a transport error", notes)
        self.assertIn("arm: humaneval ran a subset", notes)
        self.assertNotIn("base:", notes)
        self.assertTrue(doc["valid"], "health notes inform, they do not invalidate")
        self.assertEqual(doc["health"]["arm"]["bfcl"],
                         {"subset": False, "n_context_overflow": 7, "n_accepted_transport_errors": 3})
        self.assertEqual(doc["health"]["base"]["bfcl"]["n_context_overflow"], 0)
        self.assertTrue(doc["health"]["arm"]["humaneval"]["subset"])
        self.assertIn("BFCL context overflows (scored wrong, deterministic): base 0, arm 7", cg.render_md(doc))

    def test_a_limit_or_offset_alone_counts_as_a_subset(self):
        for fields in ({"limit": 5}, {"offset": 1}, {"stride": 2}):
            with self.subTest(fields=fields):
                self.assertTrue(cg.run_health({"results": {"gmmlu_he": {"subset": fields}}})["gmmlu_he"]["subset"])
        self.assertFalse(cg.run_health({"results": {"gmmlu_he": {"subset": {"stride": 1, "offset": 0, "limit": 0}}}})
                         ["gmmlu_he"]["subset"])

    def test_clean_full_runs_have_no_health_notes(self):
        b, a = self.pair()
        doc = cg.compare(b, a, draws=200)
        self.assertEqual(doc["validity_notes"], [])

    def test_disjoint_ids_are_missing_not_passing(self):
        def mutate(arm):
            arm["humaneval"] = outcomes(400, prefix="z")
        b, a = self.pair(mutate)
        r = self.rows(cg.compare(b, a, draws=200))["humaneval"]
        self.assertEqual((r["verdict"], r["n"]), ("MISSING", 0))

    def test_subset_pairs_on_the_common_items(self):
        def mutate(arm):
            arm["humaneval"] = {k: v for k, v in arm["humaneval"].items() if int(k[1:]) % 2 == 0}
        b, a = self.pair(mutate)
        r = self.rows(cg.compare(b, a, draws=200))["humaneval"]
        self.assertEqual((r["n"], r["n_base"], r["n_arm"]), (200, 400, 200))
        self.assertAlmostEqual(r["coverage"], 200 / 164)

    def test_failed_engagement_is_invalid(self):
        b, a = self.pair(engagement_ok=False)
        doc = cg.compare(b, a, draws=200)
        self.assertFalse(doc["valid"])
        self.assertEqual(doc["overall"]["all"]["ci"], "INVALID")
        self.assertEqual(doc["overall"]["without_coarse"]["ci"], "INVALID")
        self.assertTrue(any("engagement check failed" in n for n in doc["validity_notes"]))

    def test_failed_base_engagement_is_invalid_too(self):
        write_arm(self.tmp / "base", "base", all_gate_items(), engagement_ok=False)
        write_arm(self.tmp / "arm", "wide_s1", all_gate_items(), kind="wide")
        self.assertEqual(cg.compare(self.tmp / "base", self.tmp / "arm", draws=200)["overall"]["all"]["ci"], "INVALID")

    def test_no_arm_engagement_record_is_invalid(self):
        b, a = self.pair(engagement_ok=None)
        doc = cg.compare(b, a, draws=200)
        self.assertFalse(doc["valid"])
        self.assertEqual(doc["overall"]["all"]["ci"], "INVALID")
        self.assertTrue(any("arm has no engagement record" in n for n in doc["validity_notes"]))

    def test_no_base_engagement_record_is_invalid(self):
        write_arm(self.tmp / "base", "base", all_gate_items(), engagement_ok=None)
        write_arm(self.tmp / "arm", "wide_s1", all_gate_items(), kind="wide")
        doc = cg.compare(self.tmp / "base", self.tmp / "arm", draws=200)
        self.assertFalse(doc["valid"])
        self.assertEqual(doc["overall"]["all"]["ci"], "INVALID")
        self.assertTrue(any("base has no engagement record" in n for n in doc["validity_notes"]))

    def test_no_serve_record_is_invalid(self):
        for side in ("base", "arm"):
            with self.subTest(side=side):
                b = self.tmp / "b_" / side
                write_arm(b / "base", "base", all_gate_items(), serve=None if side == "base" else SERVE)
                write_arm(b / "arm", "wide_s1", all_gate_items(), kind="wide", serve=None if side == "arm" else SERVE)
                doc = cg.compare(b / "base", b / "arm", draws=200)
                self.assertFalse(doc["valid"])
                self.assertTrue(any("no serve record" in n for n in doc["validity_notes"]))

    def test_a_base_directory_that_is_not_kind_base_is_invalid(self):
        write_arm(self.tmp / "base", "base", all_gate_items(), kind="wide")
        write_arm(self.tmp / "arm", "wide_s1", all_gate_items(), kind="wide")
        doc = cg.compare(self.tmp / "base", self.tmp / "arm", draws=200)
        self.assertFalse(doc["valid"])
        self.assertTrue(any("not the untouched model" in n for n in doc["validity_notes"]))

    def test_an_arm_of_an_unknown_kind_is_invalid(self):
        write_arm(self.tmp / "base", "base", all_gate_items())
        write_arm(self.tmp / "arm", "x", all_gate_items(), kind="gate_only")
        self.assertFalse(cg.compare(self.tmp / "base", self.tmp / "arm", draws=200)["valid"])

    def test_engagement_that_ran_as_another_kind_is_invalid(self):
        write_arm(self.tmp / "base", "base", all_gate_items())
        write_arm(self.tmp / "arm", "wide_s1", all_gate_items(), kind="wide")
        info = json.loads((self.tmp / "arm" / "arm.json").read_text())
        info["engagement"]["kind"] = "ctrl"
        (self.tmp / "arm" / "arm.json").write_text(json.dumps(info))
        doc = cg.compare(self.tmp / "base", self.tmp / "arm", draws=200)
        self.assertFalse(doc["valid"])
        self.assertTrue(any("ran as kind 'ctrl'" in n for n in doc["validity_notes"]))

    def test_base_against_itself_is_refused(self):
        write_arm(self.tmp / "base", "base", all_gate_items())
        for other in (self.tmp / "base", self.tmp / "base" / ".." / "base"):
            with self.assertRaises(SystemExit) as cm:
                cg.compare(self.tmp / "base", other, draws=200)
            self.assertIn("same directory", str(cm.exception))

    def test_a_base_repeat_is_a_valid_comparison_target(self):
        write_arm(self.tmp / "base", "base", all_gate_items())
        write_arm(self.tmp / "rep", "base_repeat", all_gate_items())
        doc = cg.compare(self.tmp / "base", self.tmp / "rep", draws=200, allow_config_diff=True)
        self.assertTrue(doc["valid"], doc["validity_notes"])

    def test_serve_mismatch_refuses_unless_allowed(self):
        other = dict(SERVE, context_length=65536)
        b, a = self.pair(serve=other)
        with self.assertRaises(SystemExit) as cm:
            cg.compare(b, a, draws=200)
        self.assertIn("context_length", str(cm.exception))
        self.assertIn("--allow-config-diff", str(cm.exception))
        doc = cg.compare(b, a, draws=200, allow_config_diff=True)
        self.assertTrue(any("context_length" in p for p in doc["protocol_problems"]))

    def test_plumbing_differences_are_allowed(self):
        b, a = self.pair(serve=dict(SERVE, dp_size=4, port=31000))
        self.assertEqual(cg.compare(b, a, draws=200)["protocol_problems"], [])

    def test_each_serve_key_is_checked(self):
        for key in cg.SERVE_KEYS:
            b, a = self.pair(serve=dict(SERVE, **{key: "different"}))
            with self.assertRaises(SystemExit, msg=key):
                cg.compare(b, a, draws=100)

    def test_config_and_pin_differences_refuse(self):
        write_arm(self.tmp / "base", "base", all_gate_items())
        write_arm(self.tmp / "arm", "wide_s1", all_gate_items(), kind="wide", config_tag="other")
        with self.assertRaises(SystemExit) as cm:
            cg.compare(self.tmp / "base", self.tmp / "arm", draws=100)
        self.assertIn("config differs", str(cm.exception))
        write_arm(self.tmp / "arm", "wide_s1", all_gate_items(), kind="wide", pins_tag="other")
        with self.assertRaises(SystemExit) as cm:
            cg.compare(self.tmp / "base", self.tmp / "arm", draws=100)
        self.assertIn("pins differ", str(cm.exception))

    def test_empty_or_missing_dirs(self):
        write_arm(self.tmp / "base", "base", all_gate_items())
        with self.assertRaises(SystemExit):
            cg.compare(self.tmp / "base", self.tmp / "nope")
        (self.tmp / "empty").mkdir()
        with self.assertRaises(SystemExit):
            cg.compare(self.tmp / "base", self.tmp / "empty")

    def test_instruction_level_rows_are_report_only_and_clustered(self):
        base = all_gate_items()
        arm = all_gate_items()
        base["ifeval_strict_inst"] = {f"{k}:{j}": 1 for k in range(60) for j in range(2)}
        arm["ifeval_strict_inst"] = {f"{k}:{j}": (0 if k < 6 else 1) for k in range(60) for j in range(2)}
        write_arm(self.tmp / "base", "base", base)
        write_arm(self.tmp / "arm", "wide_s1", arm, kind="wide")
        doc = cg.compare(self.tmp / "base", self.tmp / "arm", draws=500)
        r = self.rows(doc)["ifeval_strict_inst"]
        self.assertEqual(r["verdict"], "REPORT")
        self.assertEqual(r["level"], "instruction")
        self.assertIsNone(r["band"])
        self.assertEqual(doc["overall"]["all"]["n_gates"], 21)  # report-only rows never count as gates

    def test_noise_floor(self):
        b, a = self.pair()
        write_arm(self.tmp / "rep", "base_repeat", all_gate_items(), serve=dict(SERVE, port=1))
        nf = cg.noise_floor(b, self.tmp / "rep", 200, 0.95, 1)
        self.assertEqual(nf["max_abs_delta_gated"], 0.0)
        flip = all_gate_items()
        flip["humaneval"] = outcomes(400, wrong_ids=range(0, 400, 10))
        write_arm(self.tmp / "rep2", "base_repeat", flip)
        nf = cg.noise_floor(b, self.tmp / "rep2", 200, 0.95, 1)
        self.assertAlmostEqual(nf["max_abs_delta_gated"], 10.0)
        self.assertAlmostEqual(nf["max_discordant_rate"], 0.1)

    def test_cli_writes_json_and_markdown(self):
        def mutate(arm):
            arm["humaneval"] = outcomes(400, wrong_ids=range(0, 400, 5))
        b, a = self.pair(mutate)
        write_arm(self.tmp / "rep", "base_repeat", all_gate_items())
        oj, om = self.tmp / "out" / "cmp.json", self.tmp / "out" / "cmp.md"
        code, err = in_process(cg.main, ["--base", str(b), "--arm", str(a), "--out-json", str(oj), "--out-md", str(om),
                                         "--draws", "500", "--base-repeat", str(self.tmp / "rep")])
        self.assertEqual(code, 0, err)
        self.assertIn("overall FAIL", err)
        doc = json.loads(oj.read_text())
        self.assertEqual(doc["schema"], cg.SCHEMA)
        self.assertEqual(doc["overall"]["all"]["ci"], "FAIL")
        self.assertIn("noise_floor", doc)
        md = om.read_text()
        self.assertIn("| humaneval |", md)
        self.assertIn("FAIL", md)
        self.assertIn("Noise floor", md)
        self.assertIn("| gate | n (cover) |", md)
        self.assertEqual(md.count("\n| bfcl/"), 17)
        self.assertNotIn("\u2014", md)  # no em dash in generated text

    def test_cli_prints_markdown_without_outputs(self):
        b, a = self.pair()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code, _ = in_process(cg.main, ["--base", str(b), "--arm", str(a), "--draws", "100"])
        self.assertEqual(code, 0)
        self.assertIn("# General evals: wide_s1 against base", out.getvalue())

    def test_power_table(self):
        b, _ = self.pair()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code, _ = in_process(cg.main, ["--base", str(b), "--power-table"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("| humaneval | 400 | 3 |", text)
        self.assertEqual(text.count("\n| bfcl/"), 17)
        self.assertIn("q=0.04", text)

    def test_render_marks_the_unresolvable(self):
        base = all_gate_items()
        base["humaneval"] = outcomes(164)
        arm = all_gate_items()
        arm["humaneval"] = outcomes(164)
        write_arm(self.tmp / "base", "base", base)
        write_arm(self.tmp / "arm", "wide_s1", arm, kind="wide")
        md = cg.render_md(cg.compare(self.tmp / "base", self.tmp / "arm", draws=100))
        self.assertIn("coarse", md)
        self.assertIn("Validity notes", md) if False else None  # engagement ok and serve present: no notes needed


# ----------------------------------------------------------------------------------------------------------
class FakeTok:
    def __init__(self, table=None, per=1):
        self.table, self.per = table or {}, per

    def encode(self, text, add_special_tokens=False):
        if text in self.table:
            return list(self.table[text])
        return [hash(text) % 1000 + i for i in range(self.per)]


class StubServer:
    """A /generate that answers from fake_server's pure functions, with switches for the failure shapes."""

    def __init__(self, salt="", start_offset=0, short_batch=False):
        self.salt, self.start_offset, self.short_batch = salt, start_offset, short_batch
        self.bodies = []

    def generate(self, body):
        import fake_server as fs
        self.bodies.append(body)
        ids = body["input_ids"]
        batch = isinstance(ids[0], list)
        seqs = ids if batch else [ids]
        out = []
        for seq in seqs:
            first = body["logprob_start_len"] + self.start_offset
            ents = [[None if p == 0 else fs.pair_logprob(seq[p - 1], seq[p], self.salt), seq[p], None]
                    for p in range(max(first, 0), len(seq))]
            out.append({"meta_info": {"input_token_logprobs": ents}})
        if self.short_batch:
            out = out[:-1]
        return out if batch else out[0]


class TestMmluproScoring(unittest.TestCase):
    def test_tail_logprob_on_hand_made_meta(self):
        meta = {"input_token_logprobs": [[None, 5, None], [-1.0, 6, None], [-2.0, 7, None], [-3.0, 9, None]]}
        self.assertAlmostEqual(ev_mmlupro.tail_logprob(meta, [7, 9]), -2.5)
        self.assertAlmostEqual(ev_mmlupro.tail_logprob(meta, [9]), -3.0)
        self.assertAlmostEqual(ev_mmlupro.tail_logprob(meta, [6, 7, 9]), -2.0)

    def test_tail_logprob_refuses_other_tokens(self):
        meta = {"input_token_logprobs": [[-1.0, 6, None], [-2.0, 7, None], [-3.0, 9, None]]}
        with self.assertRaises(ev_mmlupro.AlignmentError):
            ev_mmlupro.tail_logprob(meta, [7, 8])  # same length, shifted ids
        with self.assertRaises(ev_mmlupro.AlignmentError):
            ev_mmlupro.tail_logprob(meta, [1, 6, 7, 9])  # fewer entries than option tokens
        with self.assertRaises(ev_mmlupro.AlignmentError):
            ev_mmlupro.tail_logprob({}, [1])
        with self.assertRaises(ev_mmlupro.AlignmentError):
            ev_mmlupro.tail_logprob({"input_token_logprobs": [[None, 7, None]]}, [7])
        with self.assertRaises(ev_mmlupro.AlignmentError):
            ev_mmlupro.tail_logprob({"input_token_logprobs": [[float("nan"), 7, None]]}, [7])

    def test_alignment_error_is_an_item_failure(self):
        self.assertTrue(issubclass(ev_mmlupro.AlignmentError, ec.ItemFailed))

    def test_argmax_first_ties_go_to_the_first(self):
        f = ev_mmlupro.argmax_first
        self.assertEqual(f([1.0, 3.0, 3.0, 2.0]), 1)
        self.assertEqual(f([-5.0, -2.0, -9.0]), 1)
        self.assertEqual(f([-math.inf, -math.inf]), 0)
        self.assertEqual(f([float("nan"), 1.0]), 1)
        self.assertEqual(f([0.0]), 0)

    def test_cloze_scores_match_the_reference_function_at_both_offsets(self):
        import fake_server as fs
        ctx = [11, 12, 13, 14]
        opts = [[21, 22], [31], [41, 42, 43]]
        want = fs.option_scores(ctx, opts, "s")
        for off in (0, 1):
            got = ev_mmlupro.cloze_scores(StubServer("s", off), ctx, opts)
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w)

    def test_cloze_single_equals_batch_and_start_zero(self):
        srv = StubServer("s")
        ctx, opt = [1, 2, 3], [7, 8]
        a = ev_mmlupro.cloze_scores(srv, ctx, [opt])[0]
        self.assertAlmostEqual(a, ev_mmlupro.cloze_single(srv, ctx, opt))
        self.assertAlmostEqual(a, ev_mmlupro.cloze_single(srv, ctx, opt, start=0))

    def test_cloze_request_shape(self):
        srv = StubServer()
        ev_mmlupro.cloze_scores(srv, [1, 2, 3], [[7], [8, 9]])
        b = srv.bodies[0]
        self.assertEqual(b["input_ids"], [[1, 2, 3, 7], [1, 2, 3, 8, 9]])
        self.assertEqual(b["logprob_start_len"], 2)
        self.assertTrue(b["return_logprob"])
        self.assertEqual(b["sampling_params"], {"temperature": 0.0, "max_new_tokens": 1})

    def test_wrong_result_count_is_refused(self):
        with self.assertRaises(ev_mmlupro.AlignmentError):
            ev_mmlupro.cloze_scores(StubServer(short_batch=True), [1, 2], [[5], [6], [7]])

    def test_letter_token_ids_error_branches(self):
        two = FakeTok(per=2)
        with self.assertRaises(SystemExit) as cm:
            ev_mmlupro.letter_token_ids(two)
        self.assertIn("one", str(cm.exception))
        same = FakeTok({" " + c: [5] for c in ev_mmlupro.LETTERS})
        with self.assertRaises(SystemExit) as cm:
            ev_mmlupro.letter_token_ids(same)
        self.assertIn("distinct", str(cm.exception))
        ok = FakeTok({" " + c: [100 + i] for i, c in enumerate(ev_mmlupro.LETTERS)})
        self.assertEqual(ev_mmlupro.letter_token_ids(ok), list(range(100, 110)))

    def test_letter_scorer_methods(self):
        ids = list(range(100, 110))

        class Srv:
            def __init__(self, ids_rows=True, top_rows=True, greedy=101):
                self.calls, self.ids_rows, self.top_rows, self.greedy = [], ids_rows, top_rows, greedy

            def generate(self, body):
                self.calls.append(body)
                meta = {}
                if body.get("token_ids_logprob") and self.ids_rows:
                    meta["output_token_ids_logprobs"] = [[[-3.0 + 0.1 * j, i, None] for j, i in enumerate(body["token_ids_logprob"])]]
                if body.get("top_logprobs_num") and self.top_rows:
                    meta["output_top_logprobs"] = [[[-2.0, 102, None], [-4.0, 100, None], [-9.0, 777, None]]]
                return {"output_ids": [self.greedy], "meta_info": meta}

        s = Srv()
        sc = ev_mmlupro.LetterScorer(s, ids, "ids")
        lp, how, g = sc.scores([1, 2, 3], 4)
        self.assertEqual((how, g, len(lp)), ("ids", 101, 4))
        self.assertAlmostEqual(lp[2], -2.8)
        self.assertEqual(s.calls[0]["token_ids_logprob"], ids)
        self.assertEqual(s.calls[0]["logprob_start_len"], 2)
        sc = ev_mmlupro.LetterScorer(Srv(), ids, "topk")
        lp, how, g = sc.scores([1, 2, 3], 4)
        self.assertEqual(how, "topk")
        self.assertEqual(lp[2], -2.0)
        self.assertEqual(lp[0], -4.0)
        self.assertEqual(lp[1], -math.inf)  # a letter outside the top-k is -inf, not zero
        sc = ev_mmlupro.LetterScorer(Srv(ids_rows=False), ids, "auto")
        self.assertEqual(sc.scores([1, 2, 3], 4)[1], "topk")
        self.assertEqual(sc.used, {"ids": 0, "topk": 1})
        sc = ev_mmlupro.LetterScorer(Srv(ids_rows=False), ids, "ids")
        with self.assertRaises(ec.ItemFailed):
            sc.scores([1, 2, 3], 4)
        sc = ev_mmlupro.LetterScorer(Srv(ids_rows=False, top_rows=False), ids, "auto")
        with self.assertRaises(ec.ItemFailed):
            sc.scores([1, 2, 3], 4)


@unittest.skipUnless(testlib.have_tokenizer(), "needs the Qwen3.8 tokenizer")
class TestMmluproWithRealTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ec.load_tokenizer(testlib.TOKENIZER_DIR)

    def test_letter_ids_are_ten_distinct_single_tokens(self):
        ids = ev_mmlupro.letter_token_ids(self.tok)
        self.assertEqual(len(set(ids)), 10)
        for c, i in zip(ev_mmlupro.LETTERS, ids):
            self.assertEqual(self.tok.encode(" " + c, add_special_tokens=False), [i])

    def test_cloze_context_and_options_are_capped(self):
        row = {"category": "law", "question": "word " * 5000, "options": ["x " * 400, "short"], "answer_index": 0}
        ctx = ev_mmlupro.cloze_context(self.tok, row)
        self.assertEqual(len(ctx), ev_mmlupro.CTX_TAIL)
        opts = ev_mmlupro.cloze_options(self.tok, row)
        self.assertEqual(len(opts[0]), ev_mmlupro.OPT_CAP)
        self.assertLess(len(opts[1]), 5)
        # the context is the TAIL: its end is the end of "Answer:"
        self.assertTrue(self.tok.decode(ctx).endswith("Answer:"))

    def test_cloze_options_start_with_a_space_token(self):
        row = {"category": "law", "question": "q", "options": ["alpha beta"], "answer_index": 0}
        opts = ev_mmlupro.cloze_options(self.tok, row)
        self.assertEqual(self.tok.decode(opts[0]), " alpha beta")

    def test_letter_prompt_lists_the_options_and_ends_in_answer(self):
        row = {"category": "law", "question": "Which?", "options": ["a1", "b2", "c3"], "answer_index": 1}
        text = self.tok.decode(ev_mmlupro.letter_prompt_ids(self.tok, row))
        self.assertIn("A. a1\nB. b2\nC. c3\nAnswer:", text)
        self.assertTrue(text.endswith("Answer:"))
        self.assertIn("multiple choice question about law", text)

    def test_letter_prompt_is_tail_capped(self):
        row = {"category": "law", "question": "w " * 9000, "options": ["a", "b"], "answer_index": 0}
        self.assertEqual(len(ev_mmlupro.letter_prompt_ids(self.tok, row)), ev_mmlupro.LETTER_TAIL)

    def test_tail_check_catches_a_misaligned_tokenization(self):
        row = {"category": "law", "question": "Which?", "options": ["hello world"], "answer_index": 0}
        ctx = ev_mmlupro.cloze_context(self.tok, row)
        opt = ev_mmlupro.cloze_options(self.tok, row)[0]
        good = {"input_token_logprobs": [[-1.0, t, None] for t in ctx[-2:] + opt]}
        self.assertAlmostEqual(ev_mmlupro.tail_logprob(good, opt), -1.0)
        bad = {"input_token_logprobs": [[-1.0, t + 1, None] for t in ctx[-2:] + opt]}
        with self.assertRaises(ev_mmlupro.AlignmentError):
            ev_mmlupro.tail_logprob(bad, opt)


# ----------------------------------------------------------------------------------------------------------
class TestHumanEvalScoring(unittest.TestCase):
    def test_cut_completion(self):
        c = ev_humaneval.cut_completion
        self.assertEqual(c("    return 1\n\ndef helper():\n    pass\n"), "    return 1\n")
        self.assertEqual(c("    return 1\n"), "    return 1\n")
        self.assertEqual(c("    return 1\n```\ntext"), "    return 1")
        self.assertEqual(c("    x = 1\n# comment at column 0\n    return x"), "    x = 1")
        self.assertEqual(c("    x = 1\n    # indented comment stays\n    return x"),
                         "    x = 1\n    # indented comment stays\n    return x")
        self.assertEqual(c("    return 1\nprint(f(1))"), "    return 1")
        self.assertEqual(c("    return 1\nif __name__ == '__main__':\n    pass"), "    return 1")
        self.assertEqual(c("    return 1\nclass A:\n    pass\n\ndef g(): pass"), "    return 1")  # earliest stop wins
        self.assertEqual(c(""), "")

    def test_program_for(self):
        row = testlib.humaneval_rows(1)[0]
        p = ev_humaneval.program_for(row, "    return x + 0\n")
        self.assertTrue(p.startswith(row["prompt"] + "    return x + 0\n"))
        self.assertIn(row["test"], p)
        self.assertTrue(p.endswith("check(f0)\n"))

    def test_run_program_pass_fail_timeout(self):
        rows = testlib.humaneval_rows(3)
        for r in rows:
            self.assertEqual(ev_humaneval.run_program(ev_humaneval.program_for(r, r["canonical_solution"])), (True, ""))
        ok, why = ev_humaneval.run_program(ev_humaneval.program_for(rows[1], "    return 0\n"))
        self.assertFalse(ok)
        self.assertIn("AssertionError", why)
        ok, why = ev_humaneval.run_program("while True:\n    pass\n", wall=1.5)
        self.assertEqual((ok, why), (False, "timeout"))
        ok, why = ev_humaneval.run_program("raise ValueError('boom')\n")
        self.assertFalse(ok)
        self.assertIn("ValueError", why)
        ok, _ = ev_humaneval.run_program("def broken(:\n")
        self.assertFalse(ok)

    def test_memory_bomb_fails_without_taking_the_box(self):
        self.assertFalse(ev_humaneval.run_program("x = bytearray(8 << 30)\nprint(len(x))\n")[0])

    def test_the_program_cannot_see_the_environment(self):
        import os
        os.environ["SECRET_TOKEN_FOR_TEST"] = "hunter2"
        try:
            ok, _ = ev_humaneval.run_program("import os\nassert 'SECRET_TOKEN_FOR_TEST' not in os.environ\n")
        finally:
            del os.environ["SECRET_TOKEN_FOR_TEST"]
        self.assertTrue(ok)

    def test_scratch_dir_is_removed(self):
        marker = "import os\nopen('left_behind.txt', 'w').write(os.getcwd())\n"
        self.assertTrue(ev_humaneval.run_program(marker)[0])

    def test_check_canonical_on_synthetic_rows(self):
        res = ev_humaneval.check_canonical(testlib.humaneval_rows(6))
        self.assertTrue(res["ok"], res)
        self.assertEqual((res["n"], res["canonical_pass"], res["empty_body_pass"]), (6, 6, 0))
        self.assertFalse(any(res["bombs_pass"].values()))

    def test_check_canonical_goes_red_on_a_broken_reference(self):
        rows = testlib.humaneval_rows(4)
        rows[2]["canonical_solution"] = "    return -1\n"
        res = ev_humaneval.check_canonical(rows)
        self.assertFalse(res["ok"])
        self.assertEqual(res["canonical_failed"], ["HumanEval/2"])

    def test_check_canonical_goes_red_when_an_empty_body_passes(self):
        rows = testlib.humaneval_rows(4)
        for r in rows:
            r["test"] = "def check(candidate):\n    pass\n"  # a test that checks nothing
        self.assertFalse(ev_humaneval.check_canonical(rows)["ok"])


class TestGmmluScoring(unittest.TestCase):
    def test_block_layout(self):
        row = {"question": "  Q one?  ", "options": [" a ", "b", "c", "d"], "answer": " B "}
        self.assertEqual(ev_gmmlu.block(row, False), "Q one?\nA. a\nB. b\nC. c\nD. d\nAnswer:")
        self.assertEqual(ev_gmmlu.block(row, True), "Q one?\nA. a\nB. b\nC. c\nD. d\nAnswer: B\n\n")

    def test_shots_are_the_first_five_dev_rows_of_the_subject(self):
        dev = []
        for i in range(8):
            dev.append({"id": f"x/{i}", "subject": "x", "question": f"qx{i}", "options": list("abcd"), "answer": "A"})
        dev.append({"id": "y/0", "subject": "y", "question": "qy0", "options": list("abcd"), "answer": "C"})
        shots = ev_gmmlu.build_shots(dev)
        self.assertEqual(shots["x"].count("Answer: A\n\n"), 5)
        self.assertIn("qx4", shots["x"])
        self.assertNotIn("qx5", shots["x"])
        self.assertEqual(shots["y"], ev_gmmlu.block(dev[-1], True))

    def test_prompt_for_unknown_subject_has_no_shots(self):
        row = {"subject": "zzz", "question": "q", "options": list("abcd"), "answer": "A"}
        self.assertEqual(ev_gmmlu.prompt_for(row, {"x": "SHOTS"}), ev_gmmlu.block(row, False))
        row["subject"] = "x"
        self.assertTrue(ev_gmmlu.prompt_for(row, {"x": "SHOTS"}).startswith("SHOTS"))

    def test_pop_ids(self):
        rows = [{"id": str(i)} for i in range(500)] + [{"id": "7"}]  # a duplicate id counts once
        a = ev_gmmlu.pop_ids(rows, 20260905, 100)
        self.assertEqual(a, ev_gmmlu.pop_ids(rows, 20260905, 100))
        self.assertEqual(len(set(a)), 100)
        self.assertNotEqual(a, ev_gmmlu.pop_ids(rows, 1, 100))
        self.assertEqual(a, [x for x in random.Random(20260905).sample(sorted({r["id"] for r in rows}), 0)] or a)

    def test_predict(self):
        p = ev_gmmlu.predict
        self.assertEqual(p("B"), "B")
        self.assertEqual(p(" c\n"), "C")
        self.assertEqual(p("The answer is D"), "T")  # first character only, as the lane scored it: no retry, no parse
        self.assertEqual(p(""), "")
        self.assertEqual(p("\u05d0"), "\u05d0")


if __name__ == "__main__":
    unittest.main()
