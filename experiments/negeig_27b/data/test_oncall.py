"""CPU tests for the long on-call session generator (oncall.py). stdlib unittest, no GPU, no server.

    cd experiments/negeig_27b/data
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 taskset -c 8-15 python3 -B -m unittest -v test_oncall

What is covered
  verify     every built session passes the independent replay; the replay reads only the user turns (blanking
             every expectation, result and meta field leaves its answers unchanged)
  red        every corruption kind fires and every corrupted copy fails verify()
  bugs       generator bugs injected by monkeypatching (agent actions not applied to the simulator; page_role
             paging the other role; rollback naming the current version; ages off by one; a decoy label that
             disagrees with the stack) are caught by verify() on the built set; a meta key verify cannot check fails
  episode    the items run through selfdistill.tool_gen.ToolEpisode (the message-level agentic_env loop) with the
             lane's own agentic_env.turn_score and call_key: an oracle policy scores 1.0 on every turn, a policy that
             drops one call or never calls scores exactly what turn_score says (skipped when the lane is absent)
  format     call_key is agentic_env's; ids unique; seeds deterministic; eval sessions use eval-pool names only;
             action turns are spread over the session (one per 32-event stratum)
"""
from __future__ import annotations

import copy
import os
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import oncall  # noqa: E402
import ops  # noqa: E402
from common import split_pool  # noqa: E402
from pools import PEOPLE_EN  # noqa: E402

LANE = Path(os.environ.get("SELFDISTILL_LANE_DIR", "hebrew-rl-20260923"))
SMALL = None


def small_set():
    """24 sessions: 8 each at 64, 256 and 1024 events (cached across tests)."""
    global SMALL
    if SMALL is None:
        SMALL = oncall.build("eval", 8, [64, 256, 1024])
    return SMALL


def lane_modules():
    if not (LANE / "agentic_env.py").exists():
        return None
    if str(LANE) not in sys.path:
        sys.path.append(str(LANE))
    sys.path.insert(0, str(HERE.parent / "selfdistill"))
    import agentic_env
    import office_env
    import tool_gen
    return agentic_env, office_env, tool_gen


class Verify(unittest.TestCase):
    def test_all_verify(self):
        for it in small_set():
            oncall.verify(it)

    def test_replay_reads_user_turns_only(self):
        it = small_set()[11]
        want = oncall.replay(it)
        blank = copy.deepcopy(it)
        blank["meta"] = {}
        for t in blank["turns"]:
            t["expect"], t["results"], t["meta"] = {}, {}, {}
        self.assertEqual(oncall.replay(blank), want)

    def test_unparsed_text_fails(self):
        it = copy.deepcopy(small_set()[3])
        t = next(t for t in it["turns"] if t["meta"]["type"] == "events")
        t["user"] = t["user"].replace("Events:\n", "Events:\nSomething unusual happens.\n", 1)
        with self.assertRaises(oncall.ReplayError):
            oncall.verify(it)

    def test_agent_line_out_of_place_fails(self):
        it = copy.deepcopy(small_set()[5])
        t = next(t for t in it["turns"][2:] if t["meta"]["type"] == "events" and not t["meta"]["agent_lines"])
        t["user"] += "\nThe on-call agent pages Omer about INC-1000."
        with self.assertRaises(oncall.ReplayError):
            oncall.verify(it)


class RedTest(unittest.TestCase):
    def test_every_corruption_caught(self):
        stats = oncall.redtest(small_set(), seed=7, per_item=2)
        for kind, (applied, caught) in stats.items():
            self.assertGreater(applied, 0, f"{kind} never applied")
            self.assertEqual(applied, caught, f"{kind}: caught {caught} of {applied}")


class GeneratorBugs(unittest.TestCase):
    def failures(self):
        n = 0
        for it in oncall.build("eval", 6, [256, 1024]):
            try:
                oncall.verify(it)
            except oncall.ReplayError:
                n += 1
        return n

    def test_actions_not_applied_to_state(self):
        real = oncall.apply_actions

        def lines_only(sim, calls):
            return real(copy.deepcopy(sim), calls)  # report the actions, leave the simulator state untouched
        with mock.patch.object(oncall, "apply_actions", lines_only):
            self.assertGreater(self.failures(), 0)

    def test_page_role_pages_the_other_role(self):
        real = oncall.choose_action

        def wrong(rng, sim, tracker):
            text, calls, meta = real(rng, sim, tracker)
            if meta["action"] == "page_role" and calls and sim.primary != sim.secondary:
                other = sim.secondary if meta["role"] == "primary" else sim.primary
                calls = [oncall.c("page_oncall", person=other, incident_id=calls[0]["arguments"]["incident_id"])]
            return text, calls, meta
        with mock.patch.object(oncall, "choose_action", wrong):
            self.assertGreater(self.failures(), 0)

    def test_rollback_names_current_version(self):
        real = oncall.choose_action

        def wrong(rng, sim, tracker):
            text, calls, meta = real(rng, sim, tracker)
            if calls and calls[0]["name"] == "rollback_service":
                svc = calls[0]["arguments"]["service"]
                calls = [oncall.c("rollback_service", service=svc, version=ops._vs(sim.hist[svc][-1]))]
            return text, calls, meta

        def apply_lenient(sim, calls):  # let the wrong expectation through the generator's own assert
            fixed = [oncall.c("rollback_service", service=x["arguments"]["service"],
                              version=ops._vs(sim.hist[x["arguments"]["service"]][-2]))
                     if x["name"] == "rollback_service" else x for x in calls]
            return real_apply(sim, fixed)
        real_apply = oncall.apply_actions
        with mock.patch.object(oncall, "choose_action", wrong), mock.patch.object(oncall, "apply_actions",
                                                                                  apply_lenient):
            self.assertGreater(self.failures(), 0)

    def test_age_label_off_by_one(self):
        real = oncall.Tracker.age
        with mock.patch.object(oncall.Tracker, "age", lambda self, key, step: real(self, key, step) + 1):
            self.assertGreater(self.failures(), 0)

    def test_decoy_label_by_draw_branch(self):
        """The first build labelled the decoy by the branch that drew it, and a drawn older tag or proposed version
        can be the previous one. Any label that disagrees with the version's place in the stack must fail."""
        real = oncall.choose_action

        def wrong(rng, sim, tracker):
            text, calls, meta = real(rng, sim, tracker)
            if meta.get("decoy") == "previous":
                meta["decoy"] = "older"
            return text, calls, meta
        with mock.patch.object(oncall, "choose_action", wrong):
            self.assertGreater(self.failures(), 0)

    def test_unchecked_label_rejected(self):
        it = copy.deepcopy(small_set()[2])
        next(t for t in it["turns"] if t["meta"]["type"] == "action")["meta"]["note"] = "x"
        with self.assertRaises(oncall.ReplayError):
            oncall.verify(it)


class Format(unittest.TestCase):
    def test_deterministic(self):
        a = oncall.make_item("eval", 256, 3, 90_000_003)
        b = oncall.make_item("eval", 256, 3, 90_000_003)
        self.assertEqual(a, b)

    def test_ids_unique_and_lengths(self):
        items = small_set()
        self.assertEqual(len({i["id"] for i in items}), len(items))
        for it in items:
            self.assertEqual(it["turns"][-1]["meta"]["after_events"], it["meta"]["n_events"])

    def test_eval_pools_only(self):
        ev_people = set(split_pool(PEOPLE_EN, "eval"))
        ev_svc = set(split_pool(ops.SERVICES, "eval"))
        ev_inc = set(split_pool(ops.INC_IDS, "eval"))
        for it in small_set():
            text = "\n".join(t["user"] for t in it["turns"])
            people = set(re.findall(r"\b(" + "|".join(PEOPLE_EN) + r")\b", text))
            self.assertTrue(people <= ev_people, people - ev_people)
            svcs = {x for x in ops.SERVICES if re.search(r"(?<![\w-])" + re.escape(x) + r"(?![\w-])", text)}
            self.assertTrue(svcs <= ev_svc, svcs - ev_svc)
            self.assertTrue(set(re.findall(r"INC-\d{4}", text)) <= ev_inc)

    def test_actions_spread(self):
        for it in small_set():
            pos = [t["meta"]["after_events"] for t in it["turns"] if t["meta"]["type"] == "action"]
            self.assertEqual(len(pos), it["meta"]["n_events"] // oncall.ACTION_EVERY)
            for i, p in enumerate(pos):
                self.assertTrue(i * 32 < p <= (i + 1) * 32, (i, p))

    def test_call_key_is_agentic_env(self):
        mods = lane_modules()
        if mods is None:
            self.skipTest("lane not on this machine")
        ag = mods[0]
        for it in small_set()[:6]:
            for t in it["turns"]:
                for x in t["expect"].get("calls", []):
                    self.assertEqual(oncall.call_key(x["name"], x["arguments"]), ag.call_key(x["name"], x["arguments"]))
                    self.assertIn(ag.call_key(x["name"], x["arguments"]), t["results"])


def gen(content="", calls=None):
    tcs = [{"id": f"call_{j}", "type": "function",
            "function": {"name": x["name"], "arguments": oncall.json.dumps(x["arguments"])}}
           for j, x in enumerate(calls or [])]
    return types.SimpleNamespace(content=content, reasoning="", tool_calls=tcs,
                                 finish_reason="tool_calls" if tcs else "stop",
                                 usage={"completion_tokens": 3, "prompt_tokens": 10})


class Episode(unittest.TestCase):
    def drive(self, item, policy):
        agentic_env, office_env, tool_gen = lane_modules()
        ag = types.SimpleNamespace(**{k: getattr(agentic_env, k) for k in
                                      ("turn_score", "call_key", "ERROR_UNKNOWN", "ERROR_NO_MATCH",
                                       "MAX_STEPS_PER_TURN")}, MAX_GENERATIONS=10_000)
        ep = tool_gen.ToolEpisode(item, office_env, ag)
        while ep.next_request() is not None:
            t = item["turns"][ep.turn_index]
            ep.observe(policy(ep.turn_index, t, len(ep.steps)))
        return ep.result(), ep

    def setUp(self):
        if lane_modules() is None:
            self.skipTest("lane not on this machine")

    def test_oracle_scores_one(self):
        for it in small_set()[::3]:
            res, ep = self.drive(it, lambda i, t, k: gen(calls=t["expect"]["calls"]) if t["expect"].get("calls")
                                 and k == 0 else gen("Noted."))
            self.assertEqual(res["turn_scores"], [1.0] * len(it["turns"]), it["id"])
            self.assertTrue(res["closed"])
            self.assertFalse(any("error" in m["content"] for m in ep.messages if m["role"] == "tool"))

    def test_drop_one_call(self):
        it = next(i for i in small_set() if any(len(t["expect"].get("calls", [])) >= 2 for t in i["turns"]))
        ti = next(j for j, t in enumerate(it["turns"]) if len(t["expect"].get("calls", [])) >= 2)
        n = len(it["turns"][ti]["expect"]["calls"])

        def policy(i, t, k):
            if t["expect"].get("calls") and k == 0:
                return gen(calls=t["expect"]["calls"][1:] if i == ti else t["expect"]["calls"])
            return gen("Noted.")
        res, _ = self.drive(it, policy)
        self.assertAlmostEqual(res["turn_scores"][ti], (n - 1) / n)
        self.assertEqual(sum(s < 1.0 for s in res["turn_scores"]), 1)

    def test_never_call(self):
        it = small_set()[20]
        res, _ = self.drive(it, lambda i, t, k: gen("Noted."))
        want = [0.0 if t["expect"].get("calls") else 1.0 for t in it["turns"]]
        self.assertEqual(res["turn_scores"], want)

    def test_wrong_call_on_event_turn(self):
        it = small_set()[4]
        bad = oncall.c("acknowledge_incident", incident_id="INC-1000")

        def policy(i, t, k):
            if i == 1 and k == 0:
                return gen(calls=[bad])
            if t["expect"].get("calls") and k == 0:
                return gen(calls=t["expect"]["calls"])
            return gen("Noted.")
        res, ep = self.drive(it, policy)
        self.assertEqual(res["turn_scores"][1], 0.0)
        self.assertIn("No record matches", next(m["content"] for m in ep.messages if m["role"] == "tool"))


if __name__ == "__main__":
    unittest.main()
