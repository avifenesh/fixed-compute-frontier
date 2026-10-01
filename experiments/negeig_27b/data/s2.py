#!/usr/bin/env python3
"""S2: agentic episodes that need tracked state, in the Hebrew lane's agentic_env item format.

An episode gives the model a log tool and action tools. The log tool returns a long event log (the initial state and
events of an S1 simulator run); the request asks for actions whose correct set depends on the state at the END of the
log (acknowledge every incident still open, refund every item in a returned order, ask whoever holds an item now to
return it). The expected calls are the log read plus exactly those actions, keyed canonically as agentic_env scores
them (turn score = matched / max(expected, made) over canonical calls). An empty action set is a real case: read the
log, act on nothing.

Item format (agentic_pool.item/turn): id, env="agentic", kind, task_type, lang, domain, system, tools, turns,
split; a turn is {user, expect: {calls, parallel} | {no_call}, results: {call_key: result}}.

  s2.py --split train --n 1500 --out DIR     (per domain; eval pools for --split eval)
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import custody  # noqa: E402
import ops  # noqa: E402
import orders  # noqa: E402
import toggles  # noqa: E402


# ---- canonical call keys: verbatim copy of agentic_env._norm / call_key (checked against it in --selftest)
def _norm(value):
    if isinstance(value, str):
        return unicodedata.normalize('NFKC', value).strip().casefold()
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        items = [_norm(v) for v in value]
        if all(not isinstance(v, (dict, list)) for v in items):
            return sorted(items, key=lambda v: (str(type(v)), str(v)))
        return items
    return value


def call_key(name, arguments):
    return name + ' ' + json.dumps(_norm(arguments), ensure_ascii=False, sort_keys=True)


GENERIC = ("You are an assistant with access to business tools. Call a tool only when the request needs it and every "
           "required value is known; if a value is missing, ask for it instead of guessing. When several independent "
           "actions are needed, request them together. Answer in the language of the user.")
LENGTHS = [32, 64, 96, 128]
HE_SHARE = 0.25


def fn(name, desc, props=None, required=None):
    props = props or {}
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required if required is not None else list(props),
        "additionalProperties": False}}}


def s(desc, enum=None):
    d = {"type": "string", "description": desc}
    if enum:
        d["enum"] = enum
    return d


def run_log(sim, n):
    lines = list(sim.initial())
    header = {"en": ("Initial state:", "Events:"), "he": ("מצב התחלתי:", "אירועים:")}[sim.lang]
    evs = [sim.event() for _ in range(n)]
    return header[0] + "\n" + "\n".join(lines) + "\n\n" + header[1] + "\n" + "\n".join(evs)


def turn(text, calls, results):
    expect = {"calls": calls, "parallel": False} if calls else {"no_call": True}
    return {"user": text, "expect": expect, "results": {call_key(c["name"], c["arguments"]): r for c, r in zip(calls, results)}}


def c(name, **args):
    return {"name": name, "arguments": args}


# ---------------------------------------------------------------- domains
def ep_ops(rng, lang, split, n):
    sim = ops.Sim(rng, lang, split)
    log = run_log(sim, n)
    he = lang == "he"
    nm = (lambda p: p) if not he else (lambda p: p[0])
    tools = [fn("get_ops_log", "The on-call operations log: the initial state, then every event in order."),
             fn("acknowledge_incident", "Acknowledge an incident.", {"incident_id": s("Incident ID, e.g. INC-4821.")}),
             fn("resolve_incident", "Mark an incident resolved.", {"incident_id": s("Incident ID.")}),
             fn("page_oncall", "Page a person about an incident.",
                {"person": s("Name of the person to page."), "incident_id": s("Incident ID.")})]
    kind = rng.choice(["ack_open", "resolve_mitigated", "page_primary"])
    unresolved = [i for i, st in sim.status.items() if st != "res"]
    if kind == "page_primary" and not unresolved:
        kind = "ack_open"
    if kind == "ack_open":
        acts = [c("acknowledge_incident", incident_id=i) for i, st in sorted(sim.status.items()) if st == "open"]
        text = ("Go through the ops log and acknowledge every incident that nobody has acknowledged yet (still open)."
                if not he else "תעבור על יומן הכוננות ותאשר קבלה לכל תקלה שעוד לא אישרו לה קבלה (עדיין פתוחה).")
    elif kind == "resolve_mitigated":
        acts = [c("resolve_incident", incident_id=i) for i, st in sorted(sim.status.items()) if st == "mit"]
        text = ("Check the ops log and mark resolved every incident that is currently mitigated."
                if not he else "תבדוק ביומן הכוננות ותסמן כנפתרה כל תקלה שנמצאת עכשיו במצב בשליטה.")
    else:
        inc = rng.choice(sorted(unresolved))
        acts = [c("page_oncall", person=nm(sim.primary), incident_id=inc)]
        text = (f"Page whoever is primary on call right now about {inc}. Check the ops log first."
                if not he else f"תקפיץ את מי שנמצא עכשיו בכוננות הראשית בנוגע ל-{inc}. קודם תבדוק ביומן הכוננות.")
    system = GENERIC + "\n\n" + ops.Sim.SYSTEM[lang]
    return kind, tools, system, text, log, "get_ops_log", acts


def ep_toggles(rng, lang, split, n):
    sim = toggles.Sim(rng, lang, split)
    log = run_log(sim, n)
    he = lang == "he"
    nm = (lambda p: p) if not he else (lambda p: p[0])
    tools = [fn("get_flag_audit_log", "The audit log of flags, services and locks: initial state, then every event."),
             fn("set_flag", "Set a feature flag on or off.", {"flag": s("Flag name."), "state": s("on or off.", ["on", "off"])}),
             fn("set_service", "Start (on) or stop (off) a service.", {"service": s("Service name."), "state": s("on or off.", ["on", "off"])}),
             fn("request_lock_release", "Ask the current holder of a lock to release it.",
                {"lock": s("Lock name."), "holder": s("Name of the person holding it.")})]
    kind = rng.choice(["flags_off", "services_on", "release"])
    held = [l for l in sim.locks if sim.holder[l] is not None]
    if kind == "release" and not held:
        kind = "flags_off"
    if kind == "flags_off":
        acts = [c("set_flag", flag=f, state="off") for f in sorted(sim.flags) if sim.on[("flag", f)]]
        text = ("Using the audit log, turn off every feature flag that is currently on."
                if not he else "לפי יומן הביקורת, תכבה כל דגל שכרגע פועל.")
    elif kind == "services_on":
        acts = [c("set_service", service=v, state="on") for v in sorted(sim.services) if not sim.on[("service", v)]]
        text = ("Using the audit log, start every service that is currently off."
                if not he else "לפי יומן הביקורת, תפעיל כל שירות שכרגע כבוי.")
    else:
        lock = rng.choice(sorted(held))
        acts = [c("request_lock_release", lock=lock, holder=nm(sim.holder[lock]))]
        text = (f"Ask whoever holds the {lock} lock right now to release it. Check the audit log first."
                if not he else f"תבקש ממי שמחזיק עכשיו את הנעילה {lock} לשחרר אותה. קודם תבדוק ביומן הביקורת.")
    system = GENERIC + "\n\n" + toggles.Sim.SYSTEM[lang]
    return kind, tools, system, text, log, "get_flag_audit_log", acts


def ep_custody(rng, lang, split, n):
    sim = custody.Sim(rng, lang, split)
    log = run_log(sim, n)
    he = lang == "he"
    tools = [fn("get_handover_log", "The log of who holds which item: initial state, then every handover."),
             fn("request_return", "Ask a person to bring back an item they hold.",
                {"item": s("Item, as named in the log."), "person": s("Name of the person holding it.")}),
             fn("collect_item", "Send a runner to collect an item from a place.",
                {"item": s("Item, as named in the log."), "place": s("Place, as named in the log.")})]
    k = min(len(sim.items), rng.randint(1, 3))
    chosen = sorted(rng.sample(sim.items, k))
    acts = []
    for it in chosen:
        kind_h, h = sim.holder[it]
        if kind_h == "p":
            acts.append(c("request_return", item=it, person=h if not he else h[0]))
        else:
            acts.append(c("collect_item", item=it, place=h if not he else h[0]))
    if he:
        names = " ו".join(chosen) if len(chosen) <= 2 else ", ".join(chosen[:-1]) + " ו" + chosen[-1]
        text = f"אני צריך בחזרה את {names}. לפי יומן המסירות, תבקש ממי שמחזיק כל אחד מהם להחזיר אותו, ומה שנמצא במקום, תשלח מישהו לאסוף."
    else:
        names = " and ".join("the " + x for x in chosen) if len(chosen) <= 2 else \
            ", ".join("the " + x for x in chosen[:-1]) + " and the " + chosen[-1]
        text = (f"I need {names} back. Using the handover log, ask whoever holds each one to return it, and send someone "
                f"to collect anything that is sitting in a place.")
    system = GENERIC + "\n\n" + custody.Sim.SYSTEM[lang]
    return "return_items", tools, system, text, log, "get_handover_log", acts


def ep_orders(rng, lang, split, n):
    sim = orders.Sim(rng, lang, split)
    log = run_log(sim, n)
    he = lang == "he"
    tools = [fn("get_order_history", "The order desk history: initial state of the orders, then every event in order."),
             fn("refund_item", "Refund one item of an order.",
                {"order_id": s('Order number as shown in the history, e.g. "#4817".'), "item": s("Item name as shown.")}),
             fn("ship_order", "Hand a paid order to the courier.", {"order_id": s('Order number, e.g. "#4817".')})]
    kind = rng.choice(["refund_returned", "ship_paid"])
    if kind == "refund_returned":
        acts = [c("refund_item", order_id=f"#{o}", item=x) for o in sorted(sim.status) if sim.status[o] == "returned"
                for x in sorted(sim.items[o])]
        text = ("Using the order history, refund every item that is currently in a returned order."
                if not he else "לפי היסטוריית ההזמנות, תזכה כל פריט שנמצא עכשיו בהזמנה שהוחזרה.")
    else:
        acts = [c("ship_order", order_id=f"#{o}") for o in sorted(sim.status) if sim.status[o] == "paid"]
        text = ("Using the order history, ship every order that is currently paid and not yet shipped."
                if not he else "לפי היסטוריית ההזמנות, תשלח כל הזמנה ששולמה ועדיין לא נשלחה.")
    system = GENERIC + "\n\n" + orders.Sim.SYSTEM[lang]
    return kind, tools, system, text, log, "get_order_history", acts


EPISODES = {"ops": ep_ops, "toggles": ep_toggles, "custody": ep_custody, "orders": ep_orders}


# ---------------------------------------------------------------- independent verification
def _session(domain, lang, log, questions):
    """An S1 Session from the log text alone: initial lines, all events, then one turn per question."""
    from common import Session, Turn
    head, _, body = log.partition("\n\n")
    initial = head.split("\n")[1:]
    events = body.split("\n")[1:]
    turns = [Turn(events=events if i == 0 else [], question=q, answer="") for i, q in enumerate(questions)]
    return Session(domain=domain, lang=lang, split="", seed=0, n_events=len(events), system="",
                   initial=initial, turns=turns)


def derive(item):
    """Re-derive the expected actions from the log text through the domain's S1 replay() checker."""
    import re
    t = item["turns"][0]
    lang, domain, kind = item["lang"], item["domain"][3:], item["kind"][3:]
    he = lang == "he"
    log = next(v["log"] for k, v in t["results"].items() if "log" in v)
    if domain == "ops":
        incs = sorted(set(re.findall(r"INC-\d+", log)))
        qs = [ops.T[lang]["q_status"].format(inc=i) for i in incs]
        if kind == "page_primary":
            qs.append(ops.T[lang]["q_pri"])
        ans = ops.replay(_session("ops", lang, log, qs))
        inv = {v: k for k, v in (ops.STATUS_HE if he else ops.STATUS_EN).items()}
        st = {i: inv[a] for i, a in zip(incs, ans)}
        if kind == "ack_open":
            return [c("acknowledge_incident", incident_id=i) for i in incs if st[i] == "open"]
        if kind == "resolve_mitigated":
            return [c("resolve_incident", incident_id=i) for i in incs if st[i] == "mit"]
        (inc,) = re.findall(r"INC-\d+", t["user"])  # the entity comes from the request text, not the expectation
        assert st[inc] != "res", "paged about a resolved incident"
        return [c("page_oncall", person=ans[-1], incident_id=inc)]
    if domain == "toggles":
        def present(names):
            return sorted(n for n in names if re.search(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])", log))
        if kind == "release":
            (lock,) = [l for l in toggles.LOCKS if re.search(r"(?<![\w-])" + re.escape(l) + r"(?![\w-])", t["user"])]
            q = f"Who holds the {lock} lock now?" if not he else f"מי מחזיק בנעילה {lock} עכשיו?"
            (h,) = toggles.replay(_session("toggles", lang, log, [q]))
            assert h != toggles.NOBODY[lang]
            return [c("request_lock_release", lock=lock, holder=h)]
        flags = kind == "flags_off"
        names = present(toggles.FLAGS if flags else toggles.SERVICES)
        ref = (lambda n: f"the flag {n}" if flags else f"the {n} service") if not he else \
              (lambda n: f"הדגל {n}" if flags else f"השירות {n}")
        qs = [f"Is {ref(n)} on or off now?" if not he else f"האם {ref(n)} פועל או כבוי עכשיו?" for n in names]
        ans = toggles.replay(_session("toggles", lang, log, qs))
        on_word = "on" if not he else toggles.ON_HE
        if flags:
            return [c("set_flag", flag=n, state="off") for n, a in zip(names, ans) if a == on_word]
        return [c("set_service", service=n, state="on") for n, a in zip(names, ans) if a != on_word]
    if domain == "custody":
        pool = custody.ITEMS_HE if he else custody.ITEMS_EN
        found, rest = [], t["user"]
        for x in sorted(pool, key=len, reverse=True):  # longest names first, so "safe key" is not read as "key"
            pat = re.escape(x) if he else r"\bthe " + re.escape(x) + r"\b"
            if re.search(pat, rest):
                found.append(x)
                rest = re.sub(pat, " ", rest)
        items = sorted(found)
        qs = [f"Where is the {x} now?" if not he else f"איפה {x} עכשיו?" for x in items]
        ans = custody.replay(_session("custody", lang, log, qs))
        places = {p[0] for p in custody.PLACES_HE} if he else {"the " + p for p in custody.PLACES_EN}
        out = []
        for x, a in zip(items, ans):
            if a in places:
                out.append(c("collect_item", item=x, place=a if he else a[4:]))
            else:
                out.append(c("request_return", item=x, person=a))
        return out
    if domain == "orders":
        oids = sorted(set(int(o) for o in re.findall(r"#(\d+)", log)))
        tq = orders.TEMPLATES[lang]
        ans = orders.replay(_session("orders", lang, log, [tq["q_status"][0].format(o=o) for o in oids]))
        inv = {v: k for k, v in orders.STATUS_WORD[lang].items()}
        st = {o: inv[a] for o, a in zip(oids, ans)}
        if kind == "ship_paid":
            return [c("ship_order", order_id=f"#{o}") for o in oids if st[o] == "paid"]
        ret = [o for o in oids if st[o] == "returned"]
        if not ret:
            return []
        items_ans = orders.replay(_session("orders", lang, log, [tq["q_items"][0].format(o=o) for o in ret]))
        none_word = "none" if not he else "אין"
        return [c("refund_item", order_id=f"#{o}", item=x) for o, a in zip(ret, items_ans)
                if a != none_word for x in a.split(", ")]
    raise ValueError(domain)


def verify(item):
    t = item["turns"][0]
    want = collections.Counter(call_key(x["name"], x["arguments"]) for x in t["expect"].get("calls", [])[1:])
    try:
        got = collections.Counter(call_key(x["name"], x["arguments"]) for x in derive(item))
    except (KeyError, ValueError, IndexError) as e:
        raise AssertionError(f"{item['id']}: replay could not derive the actions: {e!r}")
    if want != got:
        raise AssertionError(f"{item['id']}: expected actions disagree with replay: only-expected "
                             f"{list((want - got).elements())[:3]}, only-replay {list((got - want).elements())[:3]}")
    for x in t["expect"].get("calls", []):
        assert call_key(x["name"], x["arguments"]) in t["results"], item["id"]


def build(domain, split, n, base_seed):
    items = []
    rng0 = random.Random(base_seed)
    for k in range(n):
        seed = base_seed + k
        rng = random.Random(seed)
        L = rng0.choice(LENGTHS)
        lang = "he" if rng0.random() < HE_SHARE else "en"
        kind, tools, system, text, log, log_tool, acts = EPISODES[domain](rng, lang, split, L)
        calls = [c(log_tool)] + acts
        results = [{"log": log}] + [{"ok": True}] * len(acts)
        items.append({"id": f"s2-{domain}-{split}-{k:05d}", "env": "agentic", "kind": f"s2_{kind}",
                      "task_type": f"s2-{domain}-{kind}", "lang": lang, "domain": f"s2_{domain}", "system": system,
                      "tools": tools, "turns": [turn(text, calls, results)], "split": split,
                      "meta": {"n_events": L, "n_actions": len(acts), "seed": seed}})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--n", type=int, required=True, help="episodes per domain")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = {"train": 70_000_000, "eval": 80_000_000}[a.split]
    for di, d in enumerate(EPISODES):
        items = build(d, a.split, a.n, base + 1_000_003 * di)
        for it in items:
            verify(it)
        path = os.path.join(a.out, f"s2_{d}.{a.split}.jsonl")
        with open(path, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        na = collections.Counter(min(it["meta"]["n_actions"], 5) for it in items)
        kinds = collections.Counter(it["kind"] for it in items)
        print(f"{d} {a.split}: {len(items)} episodes verified -> {path}; actions per episode (5 = 5+) "
              f"{dict(sorted(na.items()))}; kinds {dict(kinds)}", flush=True)


if __name__ == "__main__":
    main()
