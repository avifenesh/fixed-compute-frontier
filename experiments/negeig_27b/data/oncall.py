#!/usr/bin/env python3
"""oncall: long on-call agent sessions in the Hebrew lane's agentic_env multi-turn item format.

One item is one shift of an on-call agent with action tools. The first user turn gives the platform state; every
later user turn streams a batch of 1 to MAX_BATCH events from the ops simulator (ops.Sim: incidents opened,
acknowledged, mitigated, resolved, reopened; deploys and rollbacks on a per-service version stack; on-call handoffs
and swaps; alerts, notes, pages, canaries, maintenance and failed deploys that change nothing). About once every
ACTION_EVERY events (one position drawn in each 32-event stratum) the turn ends with a request from the team whose
correct calls depend on the state at that moment:

  ack_open           acknowledge every incident whose status is open                    (set, may be empty)
  resolve_mitigated  resolve every mitigated incident                                    (set, may be empty)
  page_role          page the current primary/secondary about INC-x unless it is resolved (role + status)
  page_sev           page the current primary/secondary about every unresolved SEVn       (role + status + severity)
  rollback           roll service S back; the call names the version it returns to       (version stack)
  rollback_if        roll S back only if it runs vX.Y.Z now (else no call)               (version stack, condition)

Turns that only deliver events expect a short acknowledgement and no call; an action turn whose request matches
nothing expects no call. The agent's own actions take effect: the next delivery starts with lines that report them
("The on-call agent acknowledges INC-4821."), and the simulator state changes with them, so later requests see
the post-action state. The world is canned (as agentic_env's results are): the report lines say what the correct
calls did, so a missed call does not leave the rest of the session inconsistent with the text.

Item format (agentic_pool.item / agentic_env): id, env="agentic", kind, task_type, lang, domain, system, tools,
turns, split, meta; a turn is {user, expect: {calls, parallel} | {no_call}, results: {call_key: result}, meta}.
turn meta: type (initial / events / action), after_events (simulator events delivered up to and including this
turn), batch_events and agent_lines (the turn's simulator and agent report lines), and for action turns the request
kind and phrasing (action, variant), call count, the state sizes, and the ages to read accuracy against: role_age and
entity_age are events since that role, incident or service last changed; for rollback_if, decoy is what the asked
version is when the condition is false (previous, older = deeper in the stack, other = not in the stack).

Expected calls come from the simulator state. verify() recomputes every turn from the rendered user turns ONLY
(replay(): its own parser of the log text and the requests, no simulator, no templates), including that the agent
report lines repeat the previous turn's expected calls, and requires every meta label to equal the replay's;
redtest() corrupts expectations and labels and must catch every one.

  oncall.py --split eval --n 40 --out DIR [--tokenizer DIR]     (n sessions per length; census with --tokenizer)
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import random
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ops  # noqa: E402
from common import EVENTS_HEADER, INITIAL_HEADER  # noqa: E402
from pools import PEOPLE_EN  # noqa: E402
from s2 import GENERIC, c, call_key, fn, s  # noqa: E402

LENGTHS = [64, 256, 1024, 2048]
ACTION_EVERY = 32
MAX_BATCH = 16
SPLIT_BASE = {"train": 86_000_000, "eval": 90_000_000}
CONTEXT = 131072
AGENT_PREFIX = "The on-call agent "

ONCALL = (
    "You are the on-call operations agent for a platform team. The first message gives the current state of the "
    "platform; after that, messages stream the operations log in order, and some of them end with a request from the "
    "team. Each incident has a status: open (opened or reopened, and nobody has acknowledged it since), acknowledged, "
    "mitigated (impact stopped, not yet resolved) or resolved; a reopened incident goes back to open. An incident "
    "keeps the severity it was opened with. Unresolved means any status other than resolved. Each service keeps a "
    "history of deployed versions, newest last: a deploy appends its version, and a rollback drops the newest entry "
    "so the service runs the one before it (further rollbacks keep going back). Swaps exchange who is primary and "
    "who is secondary on-call. When someone takes over a role from another person, they replace that person in that "
    "role and the other role does not change. Alerts, notes, pages, canaries, maintenance windows and failed deploys "
    "change nothing. Your own actions appear in the log too, as lines that start with \"The on-call agent\". When a "
    "message only delivers events, reply with a short acknowledgement and do not call any tool. When a message asks "
    "you to act, act on the state at that moment, one call per action; if nothing qualifies, call no tool and say so.")
SYSTEM = GENERIC + "\n\n" + ONCALL

TOOLS = [
    fn("acknowledge_incident", "Acknowledge an open incident.", {"incident_id": s("Incident ID, e.g. INC-1234.")}),
    fn("resolve_incident", "Mark an incident resolved.", {"incident_id": s("Incident ID.")}),
    fn("page_oncall", "Page a person about an incident.",
       {"person": s("Name of the person to page."), "incident_id": s("Incident ID.")}),
    fn("rollback_service", "Roll a service back one deploy, to the version before its newest one.",
       {"service": s("Service name, as in the log."),
        "version": s("The version the service runs after the rollback, e.g. v2.3.1.")}),
]

AGENT = {
    "acknowledge_incident": AGENT_PREFIX + "acknowledges {incident_id}.",
    "resolve_incident": AGENT_PREFIX + "resolves {incident_id}.",
    "page_oncall": AGENT_PREFIX + "pages {person} about {incident_id}.",
    "rollback_service": AGENT_PREFIX + "rolls back {service} to {version}.",
}

REQ = {
    "ack_open": ["Acknowledge every incident whose status is open right now.",
                 "Please ack all incidents that are currently open."],
    "resolve_mitigated": ["Mark every incident that is currently mitigated as resolved.",
                          "Close out the mitigated incidents: resolve each one that is mitigated right now."],
    "page_role": ["If {inc} is still unresolved, page whoever is {role} on-call about it; if it is already resolved, "
                  "do nothing.",
                  "Page the current {role} on-call about {inc}, unless {inc} has been resolved."],
    "page_sev": ["Page whoever is {role} on-call about every unresolved {sev} incident.",
                 "For each {sev} incident that is not resolved yet, page the current {role} on-call."],
    "rollback": ["Roll back {svc} to its previous version.",
                 "{svc} needs to go back to the version before its latest deploy. Roll it back."],
    "rollback_if": ["If {svc} is running {ver} right now, roll it back to its previous version; otherwise leave it "
                    "alone.",
                    "Check {svc}: if it is on {ver}, roll it back to the previous version, and if not, do nothing."],
}
KIND_WEIGHTS = {"ack_open": 18, "resolve_mitigated": 14, "page_role": 24, "page_sev": 14, "rollback": 15,
                "rollback_if": 15}
ACK = {"events": "Noted.", "action_none": "Nothing qualifies right now, so I made no calls.", "action_done": "Done."}


def result_for(name, a):
    """The tool's answer to a correct call (agentic_env answers any other call with ERROR_NO_MATCH)."""
    if name == "acknowledge_incident":
        return {"ok": True, "incident_id": a["incident_id"], "status": "acknowledged"}
    if name == "resolve_incident":
        return {"ok": True, "incident_id": a["incident_id"], "status": "resolved"}
    if name == "page_oncall":
        return {"ok": True, "paged": a["person"], "incident_id": a["incident_id"]}
    return {"ok": True, "service": a["service"], "running": a["version"]}


def make_turn(text, calls, meta):
    expect = {"calls": calls, "parallel": False} if calls else {"no_call": True}
    return {"user": text, "expect": expect,
            "results": {call_key(x["name"], x["arguments"]): result_for(x["name"], x["arguments"]) for x in calls},
            "meta": meta}


# ---------------------------------------------------------------- generator (simulator state)
class Tracker:
    """Step of the latest real change of each entity (incident status, service version stack, on-call role)."""

    def __init__(self, sim):
        self.sim, self.changed = sim, {}
        self.snap = self._snap()

    def _snap(self):
        sim = self.sim
        return (dict(sim.status), {v: tuple(h) for v, h in sim.hist.items()},
                {"primary": sim.primary, "secondary": sim.secondary})

    def update(self, step):
        st, hist, roles = self._snap()
        ost, ohist, oroles = self.snap
        for k in st:
            if ost.get(k) != st[k]:
                self.changed[k] = step
        for k in hist:
            if ohist[k] != hist[k]:
                self.changed[k] = step
        for k in roles:
            if oroles[k] != roles[k]:
                self.changed[k] = step
        self.snap = (st, hist, roles)

    def age(self, key, step):
        return step - self.changed.get(key, 0)


def apply_actions(sim, calls):
    """Carry out the expected calls on the simulator state; return the log lines that report them."""
    lines = []
    for x in calls:
        a = x["arguments"]
        name = x["name"]
        if name == "acknowledge_incident":
            assert sim.status[a["incident_id"]] == "open"
            sim.status[a["incident_id"]] = "ack"
            sim._touch(a["incident_id"])
        elif name == "resolve_incident":
            assert sim.status[a["incident_id"]] != "res"
            sim.status[a["incident_id"]] = "res"
            sim._touch(a["incident_id"])
        elif name == "page_oncall":
            sim._touch(a["incident_id"], changed=False)
        else:
            h = sim.hist[a["service"]]
            assert len(h) >= 2 and ops._vs(h[-2]) == a["version"]
            h.pop()
            sim._touch(a["service"])
        lines.append(AGENT[name].format(**a))
    return lines


def choose_action(rng, sim, tracker):
    """(request text, expected calls, meta) for the state the simulator is in now."""
    st = sim.status
    deep = sorted(v for v in sim.svcs if len(sim.hist[v]) >= 2)
    weights = {k: w for k, w in KIND_WEIGHTS.items() if deep or not k.startswith("rollback")}
    kind = rng.choices(list(weights), weights=list(weights.values()))[0]
    variant = rng.randrange(len(REQ[kind]))
    holder = {"primary": sim.primary, "secondary": sim.secondary}
    step = sim.step
    meta = {"action": kind, "variant": variant}
    if kind == "ack_open":
        calls = [c("acknowledge_incident", incident_id=i) for i in sorted(st) if st[i] == "open"]
        text = REQ[kind][variant]
    elif kind == "resolve_mitigated":
        calls = [c("resolve_incident", incident_id=i) for i in sorted(st) if st[i] == "mit"]
        text = REQ[kind][variant]
    elif kind == "page_role":
        role = rng.choice(["primary", "secondary"])
        unres = sorted(i for i in st if st[i] != "res")
        res = sorted(i for i in st if st[i] == "res")
        pool = res if res and (not unres or rng.random() < 0.25) else unres
        inc = sim._pick_q(pool)
        calls = [] if st[inc] == "res" else [c("page_oncall", person=holder[role], incident_id=inc)]
        text = REQ[kind][variant].format(inc=inc, role=role)
        meta.update(role=role, entity=inc, role_age=tracker.age(role, step), entity_age=tracker.age(inc, step))
    elif kind == "page_sev":
        role = rng.choice(["primary", "secondary"])
        sev = rng.choices(["SEV1", "SEV2", "SEV3"], weights=[2, 2, 1])[0]
        calls = [c("page_oncall", person=holder[role], incident_id=i) for i in sorted(st)
                 if st[i] != "res" and sim.sev[i] == sev]
        text = REQ[kind][variant].format(role=role, sev=sev)
        meta.update(role=role, sev=sev, role_age=tracker.age(role, step))
    else:
        svc = sim._pick_q(deep)
        h = sim.hist[svc]
        prev = ops._vs(h[-2])
        if kind == "rollback":
            holds = True
            text = REQ[kind][variant].format(svc=svc)
        else:
            if rng.random() < 0.5:
                ver = h[-1]
            else:
                older = [v for v in h[:-2] if v != h[-1]]
                r = rng.random()
                if r < 0.5:
                    ver = h[-2]
                elif r < 0.75 and older:
                    ver = rng.choice(older)
                else:
                    ver = sim._next_version(svc)  # can be an older tag, h[-2] included, or a never-deployed one
            holds = ver == h[-1]
            # the label is what the asked version IS in the stack now, not which branch drew it (a drawn older tag or
            # proposed version can equal the previous one): previous, older (deeper in the stack), or other (not in
            # the stack: never deployed, or deployed and since rolled back)
            decoy = (None if holds else "previous" if ver == h[-2] else "older" if ver in h[:-2] else "other")
            text = REQ[kind][variant].format(svc=svc, ver=ops._vs(ver))
            meta.update(decoy=decoy)
        calls = [c("rollback_service", service=svc, version=prev)] if holds else []
        meta.update(entity=svc, depth=len(h), entity_age=tracker.age(svc, step))
    meta.update(n_calls=len(calls), n_incidents=len(st), n_open=sum(v == "open" for v in st.values()),
                n_mitigated=sum(v == "mit" for v in st.values()), n_unresolved=sum(v != "res" for v in st.values()))
    return text, calls, meta


def make_item(split, n_events, idx, seed, max_batch=MAX_BATCH, every=ACTION_EVERY):
    rng = random.Random(seed)
    sim = ops.Sim(rng, "en", split)
    tracker = Tracker(sim)
    initial = sim.initial()
    pos = [rng.randint(i * every + 1, (i + 1) * every) for i in range(n_events // every)]
    turns = [make_turn(INITIAL_HEADER["en"] + "\n" + "\n".join(initial), [], {"type": "initial", "after_events": 0})]
    done, ai, pending = 0, 0, []
    while done < n_events:
        nxt = pos[ai] if ai < len(pos) else n_events
        k = min(nxt - done, rng.randint(1, max_batch))
        evs = []
        for _ in range(k):
            evs.append(sim.event())
            tracker.update(sim.step)
        done += k
        block = EVENTS_HEADER["en"] + "\n" + "\n".join(pending + evs)
        meta = {"after_events": done, "batch_events": k, "agent_lines": len(pending)}
        pending = []
        if ai < len(pos) and done == pos[ai]:
            ai += 1
            text, calls, ameta = choose_action(rng, sim, tracker)
            turns.append(make_turn(block + "\n\n" + text, calls, {"type": "action", **meta, **ameta}))
            pending = apply_actions(sim, calls)
            tracker.update(sim.step)
        else:
            turns.append(make_turn(block, [], {"type": "events", **meta}))
    acts = [t for t in turns if t["meta"]["type"] == "action"]
    return {"id": f"oncall-{split}-L{n_events:04d}-{idx:03d}", "env": "agentic", "kind": "oncall_long",
            "task_type": f"oncall-L{n_events}", "lang": "en", "domain": "oncall", "system": SYSTEM,
            "tools": TOOLS, "turns": turns, "split": split,
            "meta": {"n_events": n_events, "seed": seed, "max_batch": max_batch, "action_every": every,
                     "n_turns": len(turns), "n_action_turns": len(acts),
                     "n_call_turns": sum(bool(t["expect"].get("calls")) for t in acts),
                     "n_calls": sum(t["meta"]["n_calls"] for t in acts)}}


# ---------------------------------------------------------------- independent replay (rendered user turns only)
def _patterns():
    INC, VER, SEV = r"(INC-\d{4})", r"(v\d+\.\d+\.\d+)", r"(SEV[123])"
    SVC = "(" + "|".join(sorted((re.escape(x) for x in ops.SERVICES), key=len, reverse=True)) + ")"
    AL = "(" + "|".join(re.escape(x) for x in ops.ALERTS) + ")"
    N = r"([^\s,.]+)"
    ROLE = "(primary|secondary)"
    ev = {
        "i_svc": rf"{SVC} runs {VER}\.",
        "i_on": rf"On-call: {N} is primary, {N} is secondary\.",
        "i_inc": rf"{INC} on {SVC} \({SEV}\) is (open|acknowledged|mitigated|resolved)\.",
        "open": rf"{INC} opened on {SVC} \({SEV}\)\.",
        "ack": rf"{N} acknowledges {INC}\.",
        "mit_a": rf"{N} mitigates {INC}\.", "mit_p": rf"{INC} is mitigated\.",
        "res_a": rf"{N} resolves {INC}\.", "res_p": rf"{INC} is resolved\.",
        "reopen_a": rf"{N} reopens {INC}\.", "reopen_p": rf"{INC} is reopened\.",
        "deploy": rf"{SVC} is deployed at {VER}\.",
        "rollback": rf"{SVC} is rolled back to its previous version\.",
        "hand_p": rf"{N} takes over as primary on-call from {N}\.",
        "hand_s": rf"{N} takes over as secondary on-call from {N}\.",
        "swap_n": rf"{N} and {N} swap on-call roles\.",
        "swap_u": r"The primary and secondary on-call swap roles\.",
        "alert": rf"Alert {AL} on {SVC} auto-clears\.",
        "note": rf"{N} adds a note to {INC}\.",
        "page": rf"{N} is paged for {INC}\.",
        "fail": rf"Deploy of {SVC} {VER} failed its health check and was not applied\.",
        "canary": rf"Canary of {SVC} {VER} starts at 5% of traffic\.",
        "maint": rf"Maintenance window scheduled for {SVC}\.",
        "ag_ack": rf"The on-call agent acknowledges {INC}\.",
        "ag_res": rf"The on-call agent resolves {INC}\.",
        "ag_page": rf"The on-call agent pages {N} about {INC}\.",
        "ag_rb": rf"The on-call agent rolls back {SVC} to {VER}\.",
        "q_ack1": r"Acknowledge every incident whose status is open right now\.",
        "q_ack2": r"Please ack all incidents that are currently open\.",
        "q_res1": r"Mark every incident that is currently mitigated as resolved\.",
        "q_res2": r"Close out the mitigated incidents: resolve each one that is mitigated right now\.",
        "q_role1": rf"If {INC} is still unresolved, page whoever is {ROLE} on-call about it; if it is already "
                   rf"resolved, do nothing\.",
        "q_role2": rf"Page the current {ROLE} on-call about {INC}, unless {INC} has been resolved\.",
        "q_sev1": rf"Page whoever is {ROLE} on-call about every unresolved {SEV} incident\.",
        "q_sev2": rf"For each {SEV} incident that is not resolved yet, page the current {ROLE} on-call\.",
        "q_rb1": rf"Roll back {SVC} to its previous version\.",
        "q_rb2": rf"{SVC} needs to go back to the version before its latest deploy\. Roll it back\.",
        "q_rbif1": rf"If {SVC} is running {VER} right now, roll it back to its previous version; otherwise leave "
                   rf"it alone\.",
        "q_rbif2": rf"Check {SVC}: if it is on {VER}, roll it back to the previous version, and if not, do nothing\.",
    }
    return {k: re.compile("^(?:" + v + ")$") for k, v in ev.items()}


_R = _patterns()
_EVENT_KEYS = [k for k in _R if not k.startswith(("i_", "ag_", "q_"))]
_AGENT_KEYS = [k for k in _R if k.startswith("ag_")]
_REQ_KEYS = [k for k in _R if k.startswith("q_")]


class ReplayError(AssertionError):
    pass


def _check(cond, msg):
    if not cond:
        raise ReplayError(msg)


def replay(item, trace=False):
    """Every turn's expectation recomputed from the user turns' text alone.

    Returns one dict per turn: type (initial / events / action), after_events (simulator event lines so far, agent
    report lines excluded) and calls (list of (name, arguments); empty means no call). With trace, each dict also
    carries the state after the turn's events (roles, current and previous version per service)."""
    people = set(PEOPLE_EN)
    status, sev, stack = {}, {}, {}
    roles = {"primary": None, "secondary": None}
    sw = {"open": "open", "acknowledged": "ack", "mitigated": "mit", "resolved": "res"}

    def person(x, line):
        _check(x in people, f"unknown person {x!r} in {line!r}")

    def known(inc, line):
        _check(inc in status, f"unknown incident in {line!r}")

    def match(keys, line):
        hits = [(k, m) for k in keys if (m := _R[k].match(line))]
        _check(len(hits) == 1, f"{len(hits)} patterns match {line!r}")
        return hits[0]

    # step of the latest real change of each incident, service and role (0: unchanged since the initial state). An
    # event line takes its own step (its 1-based index among the simulator lines); the agent's report lines take the
    # step of the action turn's last event, when the actions happened.
    changed = {}
    out, n_events, owed = [], 0, []
    for ti, t in enumerate(item["turns"]):
        blocks = t["user"].split("\n\n")
        if ti == 0:
            _check(len(blocks) == 1 and blocks[0].startswith(INITIAL_HEADER["en"] + "\n"), "bad first turn")
            for line in blocks[0].split("\n")[1:]:
                k, m = match(["i_svc", "i_on", "i_inc"], line)
                if k == "i_svc":
                    _check(m.group(1) not in stack, line)
                    stack[m.group(1)] = [m.group(2)]
                elif k == "i_on":
                    person(m.group(1), line), person(m.group(2), line)
                    _check(m.group(1) != m.group(2) and roles["primary"] is None, line)
                    roles["primary"], roles["secondary"] = m.group(1), m.group(2)
                else:
                    inc, svc, sv, st = m.groups()
                    _check(svc in stack and inc not in status, line)
                    status[inc], sev[inc] = sw[st], sv
            _check(roles["primary"] is not None and stack and status, "incomplete initial state")
            out.append({"type": "initial", "after_events": 0, "calls": []})
            continue
        _check(blocks[0].startswith(EVENTS_HEADER["en"] + "\n") and len(blocks) in (1, 2), f"turn {ti}: bad layout")
        lines = blocks[0].split("\n")[1:]
        # the agent's report lines: first in the turn, exactly the calls the previous turn needed
        reported = []
        while lines and lines[0].startswith(AGENT_PREFIX):
            line = lines.pop(0)
            k, m = match(_AGENT_KEYS, line)
            if k == "ag_ack":
                known(m.group(1), line)
                _check(status[m.group(1)] == "open", f"agent acks a non-open incident: {line!r}")
                status[m.group(1)] = "ack"
                changed[m.group(1)] = n_events
                reported.append(("acknowledge_incident", {"incident_id": m.group(1)}))
            elif k == "ag_res":
                known(m.group(1), line)
                _check(status[m.group(1)] != "res", f"agent resolves a resolved incident: {line!r}")
                status[m.group(1)] = "res"
                changed[m.group(1)] = n_events
                reported.append(("resolve_incident", {"incident_id": m.group(1)}))
            elif k == "ag_page":
                person(m.group(1), line), known(m.group(2), line)
                reported.append(("page_oncall", {"person": m.group(1), "incident_id": m.group(2)}))
            else:
                svc, ver = m.groups()
                _check(svc in stack and len(stack[svc]) >= 2 and stack[svc][-2] == ver, f"bad rollback: {line!r}")
                stack[svc].pop()
                changed[svc] = n_events
                reported.append(("rollback_service", {"service": svc, "version": ver}))
        _check(_keys(reported) == _keys(owed), f"turn {ti}: agent lines {reported} do not report {owed}")
        _check(lines, f"turn {ti}: no events")
        for j, e in enumerate(lines):
            step = n_events + j + 1
            k, m = match(_EVENT_KEYS, e)
            g = m.groups()
            if k == "open":
                inc, svc, sv = g
                _check(inc not in status and svc in stack, e)
                status[inc], sev[inc] = "open", sv
                changed[inc] = step
            elif k in ("ack", "mit_a", "mit_p", "res_a", "res_p", "reopen_a", "reopen_p"):
                inc = g[-1]
                known(inc, e)
                if len(g) == 2:
                    person(g[0], e)
                src = {"ack": ("open",), "mit": ("open", "ack"), "res": ("open", "ack", "mit"),
                       "reopen": ("res",)}[k.split("_")[0]]
                _check(status[inc] in src, f"illegal transition: {e!r}")
                status[inc] = {"ack": "ack", "mit": "mit", "res": "res", "reopen": "open"}[k.split("_")[0]]
                changed[inc] = step
            elif k == "deploy":
                svc, ver = g
                _check(svc in stack and stack[svc][-1] != ver, e)
                stack[svc].append(ver)
                changed[svc] = step
            elif k == "rollback":
                _check(g[0] in stack and len(stack[g[0]]) >= 2, e)
                stack[g[0]].pop()
                changed[g[0]] = step
            elif k in ("hand_p", "hand_s"):
                new, old = g
                role = "primary" if k == "hand_p" else "secondary"
                person(new, e), person(old, e)
                _check(roles[role] == old and new not in roles.values(), e)
                roles[role] = new
                changed[role] = step
            elif k in ("swap_n", "swap_u"):
                if k == "swap_n":
                    _check(set(g) == set(roles.values()), e)
                roles["primary"], roles["secondary"] = roles["secondary"], roles["primary"]
                changed["primary"] = changed["secondary"] = step
            else:  # no-change lines: their entities must exist
                for x in g:
                    if x.startswith("INC-"):
                        known(x, e)
                    elif not (x in stack or x in ops.ALERTS or re.fullmatch(r"v\d+\.\d+\.\d+", x)):
                        person(x, e)
                if k in ("alert", "fail", "canary", "maint"):
                    _check(g[1 if k == "alert" else 0] in stack, e)
        n_events += len(lines)
        snap = {"state": {"roles": dict(roles), "top": {v: h[-1] for v, h in stack.items()},
                          "prev": {v: h[-2] for v, h in stack.items() if len(h) >= 2}}} if trace else {}
        counts = {"batch_events": len(lines), "agent_lines": len(reported)}
        if len(blocks) == 1:
            out.append({"type": "events", "after_events": n_events, "calls": [], "labels": counts, **snap})
            owed = []
            continue
        q = blocks[1]
        k, m = match(_REQ_KEYS, q)
        g = m.groups()
        lab = {"action": _REQ_KIND[k[:-1]], "variant": int(k[-1]) - 1}
        if k in ("q_ack1", "q_ack2"):
            calls = [("acknowledge_incident", {"incident_id": i}) for i in sorted(status) if status[i] == "open"]
        elif k in ("q_res1", "q_res2"):
            calls = [("resolve_incident", {"incident_id": i}) for i in sorted(status) if status[i] == "mit"]
        elif k in ("q_role1", "q_role2"):
            inc, role = (g[0], g[1]) if k == "q_role1" else (g[1], g[0])
            if k == "q_role2":
                _check(g[1] == g[2], q)
            known(inc, q)
            calls = [] if status[inc] == "res" else [("page_oncall", {"person": roles[role], "incident_id": inc})]
            lab.update(role=role, entity=inc, role_age=n_events - changed.get(role, 0),
                       entity_age=n_events - changed.get(inc, 0))
        elif k in ("q_sev1", "q_sev2"):
            role, sv = g if k == "q_sev1" else (g[1], g[0])
            calls = [("page_oncall", {"person": roles[role], "incident_id": i}) for i in sorted(status)
                     if status[i] != "res" and sev[i] == sv]
            lab.update(role=role, sev=sv, role_age=n_events - changed.get(role, 0))
        elif k in ("q_rb1", "q_rb2"):
            svc = g[0]
            _check(svc in stack and len(stack[svc]) >= 2, f"nothing to roll back: {q!r}")
            calls = [("rollback_service", {"service": svc, "version": stack[svc][-2]})]
            lab.update(entity=svc, depth=len(stack[svc]), entity_age=n_events - changed.get(svc, 0))
        else:
            svc, ver = g
            _check(svc in stack and len(stack[svc]) >= 2, f"nothing to roll back: {q!r}")
            h = stack[svc]
            calls = [("rollback_service", {"service": svc, "version": h[-2]})] if h[-1] == ver else []
            lab.update(entity=svc, depth=len(h), entity_age=n_events - changed.get(svc, 0),
                       decoy=None if h[-1] == ver else "previous" if ver == h[-2] else "older" if ver in h[:-2]
                       else "other")
        lab.update(n_calls=len(calls), n_incidents=len(status), n_open=sum(v == "open" for v in status.values()),
                   n_mitigated=sum(v == "mit" for v in status.values()),
                   n_unresolved=sum(v != "res" for v in status.values()), **counts)
        out.append({"type": "action", "after_events": n_events, "calls": calls, "labels": lab, **snap})
        owed = calls
    return out


_REQ_KIND = {"q_ack": "ack_open", "q_res": "resolve_mitigated", "q_role": "page_role", "q_sev": "page_sev",
             "q_rb": "rollback", "q_rbif": "rollback_if"}


def _keys(calls):
    return collections.Counter(call_key(n, a) for n, a in calls)


def verify(item):
    """Raise unless every turn's expectation, results and meta agree with the replay of the rendered text."""
    iid = item.get("id")
    _check(item["system"] == SYSTEM and item["tools"] == TOOLS and item["env"] == "agentic", f"{iid}: header")
    got = replay(item)
    _check(len(got) == len(item["turns"]), f"{iid}: turn count")
    for ti, (t, g) in enumerate(zip(item["turns"], got)):
        m = t["meta"]
        _check(m["type"] == g["type"] and m["after_events"] == g["after_events"],
               f"{iid} turn {ti}: meta {m['type']}/{m['after_events']} vs replay {g['type']}/{g['after_events']}")
        if g["calls"]:
            ex = t["expect"]
            _check(set(ex) == {"calls", "parallel"} and ex["parallel"] is False and ex["calls"],
                   f"{iid} turn {ti}: expected calls, found {ex}")
            want = collections.Counter(call_key(x["name"], x["arguments"]) for x in ex["calls"])
            _check(want == _keys(g["calls"]), f"{iid} turn {ti}: only-expected {list((want - _keys(g['calls'])))} "
                                              f"only-replay {list(_keys(g['calls']) - want)}")
            _check(m.get("n_calls") == len(g["calls"]), f"{iid} turn {ti}: n_calls")
        else:
            _check(t["expect"] == {"no_call": True}, f"{iid} turn {ti}: expected no call, found {t['expect']}")
        _check(set(t["results"]) == set(_keys(g["calls"])), f"{iid} turn {ti}: results keys")
        for name, a in g["calls"]:
            _check(t["results"][call_key(name, a)] == result_for(name, a), f"{iid} turn {ti}: result body")
        # every analysis label (request kind and phrasing, role, severity, entity, decoy, stack depth, state sizes,
        # ages since the last change, line counts) equals the replay's, and the meta carries no label it cannot check
        want_meta = {"type": g["type"], "after_events": g["after_events"], **g.get("labels", {})}
        if m != want_meta:
            diff = {k: (m.get(k, "<absent>"), want_meta.get(k, "<absent>")) for k in sorted(set(m) | set(want_meta))
                    if m.get(k, "<absent>") != want_meta.get(k, "<absent>")}
            raise ReplayError(f"{iid} turn {ti}: meta labels (meta, replay) differ: {diff}")
    _check(got[-1]["after_events"] == item["meta"]["n_events"], f"{iid}: session length")
    acts = [g for g in got if g["type"] == "action"]
    _check(item["meta"]["n_action_turns"] == len(acts) and item["meta"]["n_calls"] == sum(len(g["calls"]) for g in acts),
           f"{iid}: item meta")


# ---------------------------------------------------------------- red test: corrupted expectations must all fail
def _session_names(item):
    text = "\n".join(t["user"] for t in item["turns"])
    return {"inc": sorted(set(re.findall(r"INC-\d{4}", text))),
            "ver": sorted(set(re.findall(r"v\d+\.\d+\.\d+", text))),
            "svc": sorted(x for x in ops.SERVICES if re.search(r"(?<![\w-])" + re.escape(x) + r"(?![\w-])", text)),
            "person": sorted(set(re.findall(r"\b(" + "|".join(PEOPLE_EN) + r")\b", text)))}


def _set_calls(t, calls):
    t["expect"] = {"calls": calls, "parallel": False} if calls else {"no_call": True}
    t["results"] = {call_key(x["name"], x["arguments"]): result_for(x["name"], x["arguments"]) for x in calls}
    if t["meta"]["type"] == "action":
        t["meta"]["n_calls"] = len(calls)  # keep the bookkeeping consistent so only the semantics can fail


def _other(rng, pool, cur):
    alt = [x for x in pool if x != cur]
    return rng.choice(alt) if alt else None


def _plausible_call(rng, names):
    k = rng.randrange(4)
    if k == 0:
        return c("acknowledge_incident", incident_id=rng.choice(names["inc"]))
    if k == 1:
        return c("resolve_incident", incident_id=rng.choice(names["inc"]))
    if k == 2:
        return c("page_oncall", person=rng.choice(names["person"]), incident_id=rng.choice(names["inc"]))
    return c("rollback_service", service=rng.choice(names["svc"]), version=rng.choice(names["ver"]))


def corrupt(item, kind, rng, truth=None):
    """A copy of item with one expectation corrupted (consistent results and counts), or None if not applicable.
    truth: replay(item, trace=True) of the clean item, for the corruptions that need the state at a turn."""
    it = copy.deepcopy(item)
    names = _session_names(it)
    turns = it["turns"]
    truth = truth or replay(item, trace=True)
    call_t = [t for t in turns if t["expect"].get("calls")]
    none_act = [t for t in turns if t["meta"]["type"] == "action" and not t["expect"].get("calls")]
    ev_t = [t for t in turns if t["meta"]["type"] in ("events", "initial")]
    if kind == "drop_call":
        cand = [t for t in call_t if len(t["expect"]["calls"]) >= 2]
        if not cand:
            return None
        t = rng.choice(cand)
        calls = list(t["expect"]["calls"])
        calls.pop(rng.randrange(len(calls)))
        _set_calls(t, calls)
    elif kind == "call_to_nocall":
        if not call_t:
            return None
        _set_calls(rng.choice(call_t), [])
    elif kind == "extra_call":
        if not call_t:
            return None
        t = rng.choice(call_t)
        calls = list(t["expect"]["calls"])
        extra = copy.deepcopy(rng.choice(calls)) if rng.random() < 0.3 else _plausible_call(rng, names)
        _set_calls(t, calls + [extra])
    elif kind == "wrong_arg":
        if not call_t:
            return None
        t = rng.choice(call_t)
        calls = copy.deepcopy(t["expect"]["calls"])
        x = rng.choice(calls)
        key = rng.choice(sorted(x["arguments"]))
        pool = {"incident_id": names["inc"], "person": names["person"], "service": names["svc"],
                "version": names["ver"]}[key]
        alt = _other(rng, pool, x["arguments"][key])
        if alt is None:
            return None
        x["arguments"][key] = alt
        _set_calls(t, calls)
    elif kind == "wrong_tool":
        cand = [t for t in call_t if t["expect"]["calls"][0]["name"] in ("acknowledge_incident", "resolve_incident")]
        if not cand:
            return None
        t = rng.choice(cand)
        calls = copy.deepcopy(t["expect"]["calls"])
        x = rng.choice(calls)
        x["name"] = "resolve_incident" if x["name"] == "acknowledge_incident" else "acknowledge_incident"
        _set_calls(t, calls)
    elif kind == "role_swap":
        cand = [t for t in call_t if t["expect"]["calls"][0]["name"] == "page_oncall"]
        if not cand:
            return None
        t = rng.choice(cand)
        role = t["meta"]["role"]
        other = "secondary" if role == "primary" else "primary"
        holder = truth[turns.index(t)]["state"]["roles"][other]  # the other role's holder at that turn
        calls = copy.deepcopy(t["expect"]["calls"])
        for x in calls:
            x["arguments"]["person"] = holder
        _set_calls(t, calls)
    elif kind == "stale_version":
        cand = [t for t in call_t if t["expect"]["calls"][0]["name"] == "rollback_service"]
        if not cand:
            return None
        t = rng.choice(cand)
        calls = copy.deepcopy(t["expect"]["calls"])
        svc = calls[0]["arguments"]["service"]
        calls[0]["arguments"]["version"] = truth[turns.index(t)]["state"]["top"][svc]  # the version it runs now
        _set_calls(t, calls)
    elif kind == "nocall_to_call":
        if not none_act:
            return None
        t = rng.choice(none_act)
        _set_calls(t, [_tempting_call(truth[turns.index(t)]["state"], t["meta"], rng, names)])
    elif kind == "event_to_call":
        t = rng.choice(ev_t)
        _set_calls(t, [_plausible_call(rng, names)])
    elif kind == "after_events":
        t = rng.choice(turns[1:])
        t["meta"]["after_events"] += rng.choice([-1, 1])
    elif kind == "result_drop":
        if not call_t:
            return None
        t = rng.choice(call_t)
        t["results"].pop(rng.choice(sorted(t["results"])))
    elif kind == "result_wrong":
        if not call_t:
            return None
        t = rng.choice(call_t)
        k = rng.choice(sorted(t["results"]))
        t["results"][k] = {"error": "No record matches these arguments."}
    elif kind == "agent_line_drop":
        cand = [t for t in turns if t["meta"].get("agent_lines")]
        if not cand:
            return None
        t = rng.choice(cand)
        head, rest = t["user"].split("\n", 1)
        lines = rest.split("\n")
        lines.pop(rng.randrange(t["meta"]["agent_lines"]))
        t["user"] = head + "\n" + "\n".join(lines)
    elif kind == "agent_line_wrong":
        cand = [t for t in turns if t["meta"].get("agent_lines")]
        if not cand:
            return None
        t = rng.choice(cand)
        head, rest = t["user"].split("\n", 1)
        lines = rest.split("\n")
        j = rng.randrange(t["meta"]["agent_lines"])
        toks = re.findall(r"INC-\d{4}|v\d+\.\d+\.\d+", lines[j])
        if not toks:
            return None
        tok = toks[-1]
        alt = _other(rng, names["inc"] if tok.startswith("INC-") else names["ver"], tok)
        if alt is None:
            return None
        lines[j] = lines[j].replace(tok, alt)
        t["user"] = head + "\n" + "\n".join(lines)
    # analysis labels: the expectations stay right, only a label the runner or the analysis reads is wrong
    elif kind == "label_kind":
        t = rng.choice([t for t in turns if t["meta"]["type"] == "action"])
        t["meta"]["action"] = _other(rng, sorted(KIND_WEIGHTS), t["meta"]["action"])
    elif kind == "label_age":
        cand = [t for t in turns if "entity_age" in t["meta"] or "role_age" in t["meta"]]
        if not cand:
            return None
        t = rng.choice(cand)
        key = rng.choice(sorted(k for k in ("entity_age", "role_age") if k in t["meta"]))
        t["meta"][key] += rng.choice([-1, 1])
    elif kind == "label_decoy":
        cand = [t for t in turns if t["meta"].get("action") == "rollback_if"]
        if not cand:
            return None
        t = rng.choice(cand)
        t["meta"]["decoy"] = rng.choice([d for d in (None, "previous", "older", "other") if d != t["meta"]["decoy"]])
    elif kind == "label_count":
        t = rng.choice(turns[1:])
        key = rng.choice(["batch_events", "agent_lines"] + (["n_open", "n_unresolved", "depth"] if
                                                            t["meta"]["type"] == "action" else []))
        if key not in t["meta"]:
            return None
        t["meta"][key] += 1
    else:
        raise ValueError(kind)
    return it


CORRUPTIONS = ["drop_call", "call_to_nocall", "extra_call", "wrong_arg", "wrong_tool", "role_swap", "stale_version",
               "nocall_to_call", "event_to_call", "after_events", "result_drop", "result_wrong", "agent_line_drop",
               "agent_line_wrong", "label_kind", "label_age", "label_decoy", "label_count"]


def _tempting_call(state, meta, rng, names):
    """The call a model is most likely to make wrongly on a no-call action turn (state: the clean replay's)."""
    if meta["action"] == "page_role":
        return c("page_oncall", person=state["roles"][meta["role"]], incident_id=meta["entity"])
    if meta["action"] == "rollback_if":
        return c("rollback_service", service=meta["entity"], version=state["prev"][meta["entity"]])
    return _plausible_call(rng, names)


def redtest(items, seed=20261001, per_item=1):
    """Apply every corruption kind to every item (per_item times); every corrupted copy must fail verify().
    Returns {kind: [applied, caught]}; a clean copy of every item must still pass."""
    rng = random.Random(seed)
    stats = {k: [0, 0] for k in CORRUPTIONS}
    for it in items:
        verify(it)
        truth = replay(it, trace=True)
        for kind in CORRUPTIONS:
            for _ in range(per_item):
                bad = corrupt(it, kind, rng, truth)
                if bad is None:
                    continue
                stats[kind][0] += 1
                try:
                    verify(bad)
                except ReplayError:
                    stats[kind][1] += 1
    return stats


# ---------------------------------------------------------------- token census (real tokenizer and template)
def transcript(item):
    """The ideal conversation (expected calls, canned results, short replies) as wire messages."""
    msgs = [{"role": "system", "content": item["system"]}]
    for t in item["turns"]:
        msgs.append({"role": "user", "content": t["user"]})
        calls = t["expect"].get("calls")
        if calls:
            tcs = [{"id": f"call_{j}", "type": "function",
                    "function": {"name": x["name"], "arguments": json.dumps(x["arguments"], ensure_ascii=False)}}
                   for j, x in enumerate(calls)]
            msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
            for j, x in enumerate(calls):
                msgs.append({"role": "tool", "tool_call_id": f"call_{j}",
                             "content": json.dumps(t["results"][call_key(x["name"], x["arguments"])],
                                                   ensure_ascii=False)})
            msgs.append({"role": "assistant", "content": ACK["action_done"]})
        else:
            msgs.append({"role": "assistant",
                         "content": ACK["action_none"] if t["meta"]["type"] == "action" else ACK["events"]})
    return msgs


def census(items, tokenizer_path, lane_dir):
    """Token counts through the Qwen3.8 chat template exactly as SGLang renders the request (office_env's
    server_messages and template_tools), thinking off and on. Prompt tokens are taken at every turn's first
    generation; history keeps no reasoning (the episode loops send content only), so reasoning is extra."""
    sys.dont_write_bytecode = True
    if lane_dir not in sys.path:
        sys.path.append(lane_dir)
    import office_env
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    office_env.check_tokenizer(tok)
    tools = office_env.template_tools(TOOLS)
    gen_prompt = {}
    rows = []
    for it in items:
        msgs = office_env.server_messages(transcript(it))
        row = {"id": it["id"], "n_events": it["meta"]["n_events"], "turns": len(it["turns"]),
               "generations_ideal": len(it["turns"]) + it["meta"]["n_call_turns"]}
        for mode, think in (("think_off", False), ("think_on", True)):
            text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False,
                                           enable_thinking=think)
            if mode not in gen_prompt:
                one = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
                a = tok.apply_chat_template(one, tokenize=False, add_generation_prompt=True, enable_thinking=think)
                b = tok.apply_chat_template(one, tokenize=False, add_generation_prompt=False, enable_thinking=think)
                gen_prompt[mode] = len(tok(a[len(b):], add_special_tokens=False).input_ids)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            starts = [o[0] for o in enc["offset_mapping"]]
            cur, at = 0, []
            for t in it["turns"]:
                seg = "<|im_start|>user\n" + t["user"].strip() + "<|im_end|>\n"
                p = text.index(seg, cur)
                cur = p + len(seg)
                at.append(_count_lt(starts, cur) + gen_prompt[mode])
            if mode == "think_off":
                row["prefill_no_cache"] = sum(at)  # every turn re-prefilled from scratch (call steps not counted)
            head = _count_lt(starts, text.index("<|im_start|>user\n"))
            row[mode] = {"final": len(enc["input_ids"]), "prompt_at_turn_max": max(at), "system_and_tools": head,
                         "first_prompt": at[0],
                         "prompt_at_action": [a for a, t in zip(at, it["turns"]) if t["meta"]["type"] == "action"]}
        rows.append(row)
    by_len = {}
    for L in sorted({r["n_events"] for r in rows}):
        rs = [r for r in rows if r["n_events"] == L]
        d = {"sessions": len(rs), "turns_mean": statistics.mean(r["turns"] for r in rs),
             "turns_max": max(r["turns"] for r in rs),
             "generations_ideal_mean": statistics.mean(r["generations_ideal"] for r in rs),
             "generations_ideal_max": max(r["generations_ideal"] for r in rs)}
        for mode in ("think_off", "think_on"):
            fin = [r[mode]["final"] for r in rs]
            d[mode] = {"final_mean": round(statistics.mean(fin)), "final_max": max(fin),
                       "room_min": CONTEXT - max(fin),
                       # reasoning tokens per generation (mean) that would still fit if the client sent reasoning
                       # back and the template kept it (preserve_thinking defaults to true), worst session
                       "kept_reasoning_fit_per_generation": min((CONTEXT - r[mode]["final"]) // r["generations_ideal"]
                                                                for r in rs)}
        d["generations_ideal_total"] = sum(r["generations_ideal"] for r in rs)
        d["prefill_without_prefix_cache_tokens"] = sum(r["prefill_no_cache"] for r in rs)
        by_len[L] = d
    return {"context": CONTEXT, "generation_prompt_tokens": gen_prompt, "by_length": by_len, "rows": rows}


def _count_lt(starts, pos):
    lo, hi = 0, len(starts)
    while lo < hi:
        mid = (lo + hi) // 2
        if starts[mid] < pos:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------- build
def build(split, n, lengths=LENGTHS, max_batch=MAX_BATCH, every=ACTION_EVERY):
    items = []
    for li, L in enumerate(lengths):
        for k in range(n):
            seed = SPLIT_BASE[split] + 1_000_003 * li + k
            items.append(make_item(split, L, k, seed, max_batch, every))
    return items


def summarize(items):
    out = {}
    for L in sorted({it["meta"]["n_events"] for it in items}):
        its = [it for it in items if it["meta"]["n_events"] == L]
        acts = [t for it in its for t in it["turns"] if t["meta"]["type"] == "action"]
        out[L] = {"sessions": len(its), "turns_mean": round(statistics.mean(it["meta"]["n_turns"] for it in its), 1),
                  "action_turns": len(acts), "no_call_action_turns": sum(not t["expect"].get("calls") for t in acts),
                  "calls": sum(t["meta"]["n_calls"] for t in acts),
                  "calls_per_call_turn_max": max((t["meta"]["n_calls"] for t in acts), default=0),
                  "kinds": dict(sorted(collections.Counter(t["meta"]["action"] for t in acts).items())),
                  "no_call_by_kind": dict(sorted(collections.Counter(
                      t["meta"]["action"] for t in acts if not t["expect"].get("calls")).items())),
                  "incidents_at_end_mean": round(statistics.mean(
                      len(set(re.findall(r"INC-\d{4}", "\n".join(t["user"] for t in it["turns"])))) for it in its), 1)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    ap.add_argument("--n", type=int, default=40, help="sessions per length")
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    ap.add_argument("--max-batch", type=int, default=MAX_BATCH, help="most events in one delivery turn")
    ap.add_argument("--action-every", type=int, default=ACTION_EVERY, help="one action request per this many events")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="", help="run the token census with this tokenizer dir")
    ap.add_argument("--lane-dir", default=os.environ.get("SELFDISTILL_LANE_DIR", ""),
                    help="the Hebrew RL lane dir (office_env), needed with --tokenizer")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = build(a.split, a.n, [int(x) for x in a.lengths.split(",")], a.max_batch, a.action_every)
    for it in items:
        verify(it)
    print(f"built and verified {len(items)} sessions", flush=True)
    red = redtest(items)
    applied, caught = sum(v[0] for v in red.values()), sum(v[1] for v in red.values())
    print(f"red test: caught {caught} of {applied} corrupted copies; by kind {red}", flush=True)
    if caught != applied:
        raise SystemExit("red test missed a corruption")
    path = os.path.join(a.out, f"oncall.{a.split}.jsonl")
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    summ = summarize(items)
    print(json.dumps({"file": path, "sha256": sha, "summary": summ}, indent=1), flush=True)
    report = {"file": path, "sha256": sha, "summary": summ, "redtest": red}
    if a.tokenizer:
        if not a.lane_dir:
            ap.error("--tokenizer needs --lane-dir or SELFDISTILL_LANE_DIR")
        cen = census(items, a.tokenizer, a.lane_dir)
        report["census"] = {k: v for k, v in cen.items() if k != "rows"}
        with open(os.path.join(a.out, f"oncall.{a.split}.census.json"), "w") as f:
            json.dump(cen, f, indent=1)
        print(json.dumps(report["census"], indent=1), flush=True)
    with open(os.path.join(a.out, f"oncall.{a.split}.report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
