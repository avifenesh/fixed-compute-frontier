"""toggles: feature flags, services and locks; questions: is X on or off, who holds lock L.

State: every flag and service is on or off; every lock is held by one person or is free.
Events:
  toggle   a flag or service is flipped (on becomes off, off becomes on)       (parity: the key mechanism)
  on/off   a flag or service is turned on or off (often redundant)              (set, overwrite)
  set      a flag or service is set to an explicit value                        (set, overwrite)
  acquire  a person takes a free lock                                           (move)
  release  the holder releases a lock                                           (move)
  handoff  the holder hands a lock to another person                            (move)
  swap     two holders exchange the locks they hold                             (transposition of holders)
  noop     dashboard check, reading a flag, a note, viewing logs, restarting a running service,
           a failed attempt to take a busy lock                                 (no change)
Implicit-effect events change many entities WITHOUT naming them, so the answer cannot be read off the last line that
mentions an entity; it has to be carried as state:
  flipall  a person flips every flag                                           (parity on all flags, flags unnamed)
  swapall  two holders swap all the locks they hold                            (permutation, locks unnamed)
  handall  a holder hands all the locks they hold to another person            (move, locks unnamed)
One service per session is cold: no event ever names it, so a question about it (and about any entity that no event
touched) is answered from the initial state, and untouched questions exist at every session length.
English and Hebrew; Hebrew verbs agree with the actor's gender. Flag, service and lock names are Latin identifiers.
Answers: "on" / "off" (Hebrew "פועל" / "כבוי"); a holder's name, or "nobody" (Hebrew "אף אחד").
"""

from __future__ import annotations

import re

from common import Session, split_pool
from pools import PEOPLE_EN, PEOPLE_HE

FLAGS = [
    "checkout_v2", "dark_mode", "new_onboarding", "fast_search", "beta_dashboard", "smart_reorder",
    "guest_checkout", "express_shipping", "wishlist_share", "voice_search", "ai_summaries", "bulk_export",
    "two_step_login", "promo_banner", "live_chat", "offline_sync", "compact_view", "price_alerts", "gift_wrap",
    "referral_credit", "auto_retry", "lazy_images", "new_navbar", "inline_edit", "csv_import", "saved_carts",
    "quick_reply", "smart_tags", "usage_meter", "team_invites", "audit_trail", "sso_login", "mobile_push",
    "dense_tables", "split_payments", "order_tracking", "fraud_check", "recs_carousel", "slim_footer",
    "multi_currency", "auto_save", "read_receipts", "night_shift", "draft_mode", "bulk_delete", "api_v3",
    "fast_upload", "image_zoom", "coupon_stack", "loyalty_points", "smart_filters", "vat_invoices",
    "email_digest", "pin_items", "quiet_hours", "card_vault", "theme_picker", "export_pdf", "deep_links",
    "weekly_report", "auto_assign", "soft_delete", "rate_hints", "cohort_view",
]

SERVICES = [
    "billing-api", "search-indexer", "auth-gateway", "email-worker", "image-resizer", "report-builder",
    "notif-relay", "cart-api", "ledger-sync", "webhook-sender", "pdf-renderer", "geo-lookup", "cache-warmer",
    "audit-logger", "queue-runner", "session-store", "rate-limiter", "feed-ranker", "export-worker",
    "metrics-agent", "order-router", "stock-sync", "invoice-gen", "tax-engine", "mail-relay", "chat-gateway",
    "video-encoder", "fraud-scorer", "pricing-api", "profile-api", "sms-sender", "task-scheduler",
    "log-shipper", "config-server", "upload-proxy", "recs-engine", "payout-batch", "user-sync", "map-tiles",
    "event-bus",
]

LOCKS = [
    "deploy", "migrations", "release-train", "hotfix", "schema-change", "cache-flush", "config-edit",
    "db-failover", "cert-rotation", "backup-window", "maintenance", "index-rebuild", "traffic-shift",
    "secrets-rotation", "dns-change", "batch-window", "load-test", "feature-freeze", "log-rotation",
    "node-drain", "queue-purge", "rollback", "data-fix", "canary",
]

for _pool in (FLAGS, SERVICES, LOCKS):
    assert len(set(_pool)) == len(_pool)
assert not (set(FLAGS) & set(SERVICES)) and not (set(FLAGS) & set(LOCKS)) and not (set(SERVICES) & set(LOCKS))

ON_HE, OFF_HE = "פועל", "כבוי"
NOBODY = {"en": "nobody", "he": "אף אחד"}

# Hebrew verb forms as (masculine, feminine) for the actor's gender; identical forms are listed twice.
HE_ON = {"flag": [("מפעיל", "מפעילה"), ("מדליק", "מדליקה")],
         "service": [("מפעיל", "מפעילה"), ("מעלה", "מעלה")]}
HE_OFF = {"flag": [("מכבה", "מכבה"), ("מנטרל", "מנטרלת")],
          "service": [("מכבה", "מכבה"), ("עוצר", "עוצרת")]}
HE_TOGGLE = [("הופך", "הופכת"), ("מחליף", "מחליפה")]
HE_SET = ("מגדיר", "מגדירה")
HE_ACQUIRE = ("תופס", "תופסת")
HE_RELEASE = ("משחרר", "משחררת")
HE_HAND = ("מעביר", "מעבירה")
HE_DASH = ("בודק", "בודקת")
HE_READ = ("קורא", "קוראת")
HE_NOTE = ("מוסיף", "מוסיפה")
HE_LOGS = ("צופה", "צופה")
HE_RESTART = ("מפעיל מחדש", "מפעילה מחדש")
HE_TRY = ("מנסה", "מנסה")

EN_ON = {"flag": ["{a} turns on {r}.", "{a} enables {r}.", "{a} switches {r} on."],
         "service": ["{a} starts {r}.", "{a} turns on {r}.", "{a} brings {r} up."]}
EN_OFF = {"flag": ["{a} turns off {r}.", "{a} disables {r}.", "{a} switches {r} off."],
          "service": ["{a} stops {r}.", "{a} turns off {r}.", "{a} takes {r} down."]}
EN_TOGGLE = ["{a} toggles {r}.", "{a} flips {r}."]


class Sim:
    SYSTEM = {
        "en": "You are tracking feature flags, services and locks for a software team. You get the initial state, "
              "then events in order. Every flag or service is either on or off; toggling or flipping it reverses its "
              "state, and turning on something that is already on (or turning off something that is already off) "
              "changes nothing. Each lock is held by one person or is free. Checking the dashboard, reading a flag, "
              "adding a note, viewing logs, restarting a running service and failing to take a busy lock change "
              "nothing. After each batch of events, answer the question with only the current value: \"on\" or "
              "\"off\" for a flag or service, or the holder's name for a lock (\"nobody\" if the lock is free).",
        "he": "את/ה עוקב/ת אחרי דגלים, שירותים ונעילות בצוות תוכנה. מקבלים מצב התחלתי ואחריו אירועים לפי הסדר. "
              "כל דגל או שירות הוא פועל או כבוי; הפיכה או החלפה של המצב הופכת פועל לכבוי וכבוי לפועל, והפעלה של "
              "משהו שכבר פועל (או כיבוי של משהו שכבר כבוי) לא משנה דבר. כל נעילה מוחזקת על ידי אדם אחד או פנויה. "
              "בדיקת הדשבורד, קריאת דגל, הוספת הערה, צפייה בלוגים, הפעלה מחדש של שירות שפועל וניסיון כושל לתפוס "
              "נעילה תפוסה לא משנים דבר. אחרי כל קבוצת אירועים עונים על השאלה רק בערך הנוכחי: \"פועל\" או "
              "\"כבוי\" עבור דגל או שירות, או שם המחזיק עבור נעילה (\"אף אחד\" אם הנעילה פנויה).",
    }

    def __init__(self, rng, lang, split):
        self.rng, self.lang = rng, lang
        people = split_pool(PEOPLE_EN if lang == "en" else PEOPLE_HE, split)
        self.people = rng.sample(people, rng.randint(4, 7))
        self.flags = rng.sample(split_pool(FLAGS, split), rng.randint(5, 9))
        self.services = rng.sample(split_pool(SERVICES, split), rng.randint(3, 6))
        self.locks = rng.sample(split_pool(LOCKS, split), rng.randint(2, 4))
        self.switches = [("flag", f) for f in self.flags] + [("service", s) for s in self.services]
        self.cold = ("service", rng.choice(self.services))  # never named by an event: answered from the initial state
        self.live = [sw for sw in self.switches if sw != self.cold]
        self.lock_ents = [("lock", l) for l in self.locks]
        self.on = {sw: rng.random() < 0.5 for sw in self.switches}
        self.holder = {l: (rng.choice(self.people) if rng.random() < 0.75 else None) for l in self.locks}
        self.recent: list = []  # entities mentioned by events, oldest first
        self.nchg = {e: 0 for e in self.switches + self.lock_ents}

    # ---------- rendering helpers
    def _nm(self, p):
        return p if self.lang == "en" else p[0]

    def _vb(self, pair, p):
        return pair[0] if p[1] == "m" else pair[1]

    def _val(self, on):
        return ("on" if on else "off") if self.lang == "en" else (ON_HE if on else OFF_HE)

    def _ref(self, ent):
        kind, name = ent
        if self.lang == "en":
            return {"flag": f"the flag {name}", "service": f"the {name} service", "lock": f"the {name} lock"}[kind]
        return {"flag": f"הדגל {name}", "service": f"השירות {name}", "lock": f"הנעילה {name}"}[kind]

    def _touch(self, ent):
        self.recent.append(ent)
        if len(self.recent) > 64:
            del self.recent[:32]

    def _recent_of(self, pool, n):
        """The n most recently mentioned distinct entities that belong to pool."""
        out = []
        for e in reversed(self.recent):
            if e in pool and e not in out:
                out.append(e)
                if len(out) == n:
                    break
        return out

    def _pick_switch(self, pool):
        rec = self._recent_of(pool, 5)
        if rec and self.rng.random() < 0.5:
            return self.rng.choice(rec)
        return self.rng.choice(pool)

    def _set(self, sw, val):
        if self.on[sw] != val:
            self.nchg[sw] += 1
        self.on[sw] = val

    def _give(self, lock, who):
        if self.holder[lock] != who:
            self.nchg[("lock", lock)] += 1
        self.holder[lock] = who

    # ---------- Sim interface
    def initial(self):
        out = []
        for kind, name in self.switches:
            v = self._val(self.on[(kind, name)])
            if self.lang == "en":
                out.append(f"The flag {name} is {v}." if kind == "flag" else f"The {name} service is {v}.")
            else:
                out.append(f"הדגל {name} {v}." if kind == "flag" else f"השירות {name} {v}.")
        for l in self.locks:
            h = self.holder[l]
            if self.lang == "en":
                out.append(f"The {l} lock is held by {h}." if h else f"The {l} lock is free.")
            else:
                out.append(f"הנעילה {l} אצל {h[0]}." if h else f"הנעילה {l} פנויה.")
        return out

    def event(self):
        rng, he = self.rng, self.lang == "he"
        held = [l for l in self.locks if self.holder[l] is not None]
        free = [l for l in self.locks if self.holder[l] is None]
        swap_pairs = [(x, y) for i, x in enumerate(held) for y in held[i + 1:] if self.holder[x] != self.holder[y]]
        up = [sw for sw in self.live if sw[0] == "service" and self.on[sw]]
        opts = ["toggle"] * 28 + ["on"] * 7 + ["off"] * 7 + ["set"] * 6 + ["dash"] * 3 + ["read"] * 3 \
            + ["note"] * 2 + ["logs"] * 2 + ["flipall"] * 2
        if free:
            opts += ["acquire"] * 8
        if held:
            opts += ["release"] * 3 + ["handoff"] * 7 + ["tryfail"] * 3 + ["handall"] * 2
        if swap_pairs:
            opts += ["swap"] * 5 + ["swapall"] * 3
        if up:
            opts += ["restart"] * 2
        kind = rng.choice(opts)

        if kind in ("toggle", "on", "off", "set"):
            pool = self.live
            if kind == "on":
                cand = [sw for sw in self.live if not self.on[sw]]
                pool = cand if cand and rng.random() < 0.7 else self.live
            elif kind == "off":
                cand = [sw for sw in self.live if self.on[sw]]
                pool = cand if cand and rng.random() < 0.7 else self.live
            sw = self._pick_switch(pool)
            ek = sw[0]
            a = rng.choice(self.people)
            r = self._ref(sw)
            self._touch(sw)
            if kind == "toggle":
                self._set(sw, not self.on[sw])
                if not he:
                    return rng.choice(EN_TOGGLE).format(a=a, r=r)
                return f"{a[0]} {self._vb(rng.choice(HE_TOGGLE), a)} את מצב {r}."
            if kind == "set":
                val = rng.random() < 0.5
                self._set(sw, val)
                if not he:
                    return f"{a} sets {r} to {self._val(val)}."
                return f"{a[0]} {self._vb(HE_SET, a)} את {r} למצב {self._val(val)}."
            val = kind == "on"
            self._set(sw, val)
            if not he:
                return rng.choice((EN_ON if val else EN_OFF)[ek]).format(a=a, r=r)
            return f"{a[0]} {self._vb(rng.choice((HE_ON if val else HE_OFF)[ek]), a)} את {r}."

        if kind == "acquire":
            l = rng.choice(free)
            a = rng.choice(self.people)
            self._give(l, a)
            self._touch(("lock", l))
            if not he:
                return f"{a} {rng.choice(['takes', 'acquires'])} the {l} lock."
            return f"{a[0]} {self._vb(HE_ACQUIRE, a)} את הנעילה {l}."
        if kind == "release":
            l = rng.choice(held)
            a = self.holder[l]
            self._give(l, None)
            self._touch(("lock", l))
            return f"{a} releases the {l} lock." if not he else f"{a[0]} {self._vb(HE_RELEASE, a)} את הנעילה {l}."
        if kind == "handoff":
            l = rng.choice(held)
            a = self.holder[l]
            b = rng.choice([p for p in self.people if p != a])
            self._give(l, b)
            self._touch(("lock", l))
            if not he:
                return f"{a} {rng.choice(['hands', 'passes'])} the {l} lock to {b}."
            return f"{a[0]} {self._vb(HE_HAND, a)} את הנעילה {l} ל{b[0]}."
        if kind == "swap":
            x, y = rng.choice(swap_pairs)
            a, b = self.holder[x], self.holder[y]
            self._give(x, b)
            self._give(y, a)
            self._touch(("lock", x))
            self._touch(("lock", y))
            if not he:
                return f"{a} and {b} swap the {x} lock and the {y} lock."
            fem = a[1] == "f" and b[1] == "f"
            return f"{a[0]} ו{b[0]} {'מחליפות ביניהן' if fem else 'מחליפים ביניהם'} את הנעילה {x} ואת הנעילה {y}."
        if kind == "flipall":
            a = rng.choice(self.people)
            flags = [sw for sw in self.switches if sw[0] == "flag"]
            for sw in flags:
                self._set(sw, not self.on[sw])
            for sw in rng.sample(flags, min(3, len(flags))):
                self._touch(sw)
            if not he:
                return f"{a} flips every flag."
            return f"{a[0]} {self._vb(rng.choice(HE_TOGGLE), a)} את מצב כל הדגלים."
        if kind == "swapall":
            x, y = rng.choice(swap_pairs)
            a, b = self.holder[x], self.holder[y]
            mine_a = [l for l in self.locks if self.holder[l] == a]
            mine_b = [l for l in self.locks if self.holder[l] == b]
            for l in mine_a:
                self._give(l, b)
            for l in mine_b:
                self._give(l, a)
            for l in rng.sample(mine_a + mine_b, min(3, len(mine_a + mine_b))):
                self._touch(("lock", l))
            if not he:
                return f"{a} and {b} swap all the locks they hold."
            fem = a[1] == "f" and b[1] == "f"
            return f"{a[0]} ו{b[0]} {'מחליפות ביניהן את כל הנעילות שהן מחזיקות' if fem else 'מחליפים ביניהם את כל הנעילות שהם מחזיקים'}."
        if kind == "handall":
            a = self.holder[rng.choice(held)]
            b = rng.choice([p for p in self.people if p != a])
            mine = [l for l in self.locks if self.holder[l] == a]
            for l in mine:
                self._give(l, b)
            for l in rng.sample(mine, min(3, len(mine))):
                self._touch(("lock", l))
            if not he:
                return f"{a} hands all the locks they hold to {b}."
            return f"{a[0]} {self._vb(HE_HAND, a)} ל{b[0]} את כל הנעילות ש{'הוא מחזיק' if a[1] == 'm' else 'היא מחזיקה'}."
        if kind == "tryfail":
            l = rng.choice(held)
            a = rng.choice([p for p in self.people if p != self.holder[l]])
            self._touch(("lock", l))
            if not he:
                return f"{a} tries to take the {l} lock, but it is busy."
            return f"{a[0]} {self._vb(HE_TRY, a)} לתפוס את הנעילה {l}, אבל היא תפוסה."
        if kind == "restart":
            sw = rng.choice(up)
            a = rng.choice(self.people)
            self._touch(sw)
            if not he:
                return f"{a} restarts {self._ref(sw)}."
            return f"{a[0]} {self._vb(HE_RESTART, a)} את {self._ref(sw)}."
        a = rng.choice(self.people)
        if kind == "dash":
            return f"{a} checks the dashboard." if not he else f"{a[0]} {self._vb(HE_DASH, a)} את הדשבורד."
        if kind == "logs":
            sw = rng.choice([s for s in self.live if s[0] == "service"])
            self._touch(sw)
            if not he:
                return f"{a} views the logs of {self._ref(sw)}."
            return f"{a[0]} {self._vb(HE_LOGS, a)} בלוגים של {self._ref(sw)}."
        sw = self._pick_switch([s for s in self.live if s[0] == "flag"])  # read / note: flags
        self._touch(sw)
        if kind == "read":
            if not he:
                return f"{a} reads the value of {self._ref(sw)}."
            return f"{a[0]} {self._vb(HE_READ, a)} את ערך {self._ref(sw)}."
        if not he:
            return f"{a} adds a note to {self._ref(sw)}."
        return f"{a[0]} {self._vb(HE_NOTE, a)} הערה לדגל {sw[1]}."  # ל + הדגל merges into לדגל

    def ask(self):
        rng, he = self.rng, self.lang == "he"
        pool = self.lock_ents if rng.random() < 0.3 else self.switches
        r = rng.random()
        cand, pick = [], "any"
        if r < 0.06:  # the cold service is always a candidate, so this holds at every session length
            cand, pick = [e for e in self.switches + self.lock_ents if self.nchg[e] == 0], "untouched"
        elif r < 0.56:
            cand, pick = self._recent_of(pool, 4), "recent"
        elif r < 0.86:
            cand, pick = [e for e in pool if self.nchg[e] >= 2], "several"
        if not cand:
            cand, pick = pool, "any"
        ent = rng.choice(cand)
        kind, name = ent
        meta = {"kind": kind, "entity": name, "changes": self.nchg[ent], "pick": pick}
        v = rng.randint(0, 1)
        if kind == "lock":
            h = self.holder[name]
            if not he:
                q = [f"Who holds the {name} lock now?", f"Who currently holds the {name} lock?"][v]
                return q, h if h else NOBODY["en"], meta
            q = [f"מי מחזיק בנעילה {name} עכשיו?", f"אצל מי נמצאת הנעילה {name} עכשיו?"][v]
            return q, h[0] if h else NOBODY["he"], meta
        ans = self._val(self.on[ent])
        if not he:
            ref = f"the flag {name}" if kind == "flag" else f"the {name} service"
            q = [f"Is {ref} on or off now?", f"What is the state of {ref} now, on or off?"][v]
        else:
            ref = f"הדגל {name}" if kind == "flag" else f"השירות {name}"
            q = [f"האם {ref} פועל או כבוי עכשיו?", f"מה מצב {ref} עכשיו, פועל או כבוי?"][v]
        return q, ans, meta


# ---------- independent checker: recompute every answer from the rendered text only
def _alt(xs):
    return "|".join(sorted((re.escape(x) for x in xs), key=len, reverse=True))


def _he_verb(*pairs):
    """Regex for Hebrew verb forms; gendered forms go to groups gm / gf, identical forms stay neutral."""
    neutral = [m for m, f in pairs if m == f]
    ms = [m for m, f in pairs if m != f]
    fs = [f for m, f in pairs if m != f]
    parts = [re.escape(x) for x in neutral]
    if ms:
        parts.append("(?P<gm>" + "|".join(re.escape(x) for x in ms) + ")")
        parts.append("(?P<gf>" + "|".join(re.escape(x) for x in fs) + ")")
    return "(?:" + "|".join(parts) + ")"


def _rules(he: bool):
    """(operation, entity kind, regex template) for events. <a> <b> actors, <f> flag, <s> service, <l> <x> <y> locks,
    <v> on/off value."""
    if not he:
        return [
            ("on", "flag", r"<a> turns on the flag <f>\."), ("on", "flag", r"<a> enables the flag <f>\."),
            ("on", "flag", r"<a> switches the flag <f> on\."),
            ("off", "flag", r"<a> turns off the flag <f>\."), ("off", "flag", r"<a> disables the flag <f>\."),
            ("off", "flag", r"<a> switches the flag <f> off\."),
            ("toggle", "flag", r"<a> toggles the flag <f>\."), ("toggle", "flag", r"<a> flips the flag <f>\."),
            ("set", "flag", r"<a> sets the flag <f> to <v>\."),
            ("on", "service", r"<a> starts the <s> service\."), ("on", "service", r"<a> turns on the <s> service\."),
            ("on", "service", r"<a> brings the <s> service up\."),
            ("off", "service", r"<a> stops the <s> service\."), ("off", "service", r"<a> turns off the <s> service\."),
            ("off", "service", r"<a> takes the <s> service down\."),
            ("toggle", "service", r"<a> toggles the <s> service\."), ("toggle", "service", r"<a> flips the <s> service\."),
            ("set", "service", r"<a> sets the <s> service to <v>\."),
            ("acquire", "", r"<a> (?:takes|acquires) the <l> lock\."),
            ("release", "", r"<a> releases the <l> lock\."),
            ("handoff", "", r"<a> (?:hands|passes) the <l> lock to <b>\."),
            ("swap", "", r"<a> and <b> swap the <x> lock and the <y> lock\."),
            ("tryfail", "", r"<a> tries to take the <l> lock, but it is busy\."),
            ("flipall", "", r"<a> flips every flag\."),
            ("swapall", "", r"<a> and <b> swap all the locks they hold\."),
            ("handall", "", r"<a> hands all the locks they hold to <b>\."),
            ("restart", "service", r"<a> restarts the <s> service\."),
            ("dash", "", r"<a> checks the dashboard\."),
            ("readflag", "flag", r"<a> reads the value of the flag <f>\."),
            ("readflag", "flag", r"<a> adds a note to the flag <f>\."),
            ("readsvc", "service", r"<a> views the logs of the <s> service\."),
        ]
    V = _he_verb
    return [
        ("on", "flag", rf"<a> {V(('מפעיל', 'מפעילה'), ('מדליק', 'מדליקה'))} את הדגל <f>\."),
        ("off", "flag", rf"<a> {V(('מכבה', 'מכבה'), ('מנטרל', 'מנטרלת'))} את הדגל <f>\."),
        ("toggle", "flag", rf"<a> {V(('הופך', 'הופכת'), ('מחליף', 'מחליפה'))} את מצב הדגל <f>\."),
        ("set", "flag", rf"<a> {V(('מגדיר', 'מגדירה'))} את הדגל <f> למצב <v>\."),
        ("on", "service", rf"<a> {V(('מפעיל', 'מפעילה'), ('מעלה', 'מעלה'))} את השירות <s>\."),
        ("off", "service", rf"<a> {V(('מכבה', 'מכבה'), ('עוצר', 'עוצרת'))} את השירות <s>\."),
        ("toggle", "service", rf"<a> {V(('הופך', 'הופכת'), ('מחליף', 'מחליפה'))} את מצב השירות <s>\."),
        ("set", "service", rf"<a> {V(('מגדיר', 'מגדירה'))} את השירות <s> למצב <v>\."),
        ("acquire", "", rf"<a> {V(('תופס', 'תופסת'))} את הנעילה <l>\."),
        ("release", "", rf"<a> {V(('משחרר', 'משחררת'))} את הנעילה <l>\."),
        ("handoff", "", rf"<a> {V(('מעביר', 'מעבירה'))} את הנעילה <l> ל<b>\."),
        ("swap", "", r"<a> ו<b> (?:(?P<sm>מחליפים ביניהם)|(?P<sf>מחליפות ביניהן)) את הנעילה <x> ואת הנעילה <y>\."),
        ("tryfail", "", r"<a> מנסה לתפוס את הנעילה <l>, אבל היא תפוסה\."),
        ("flipall", "", rf"<a> {V(('הופך', 'הופכת'), ('מחליף', 'מחליפה'))} את מצב כל הדגלים\."),
        ("swapall", "", r"<a> ו<b> (?:(?P<sm>מחליפים ביניהם את כל הנעילות שהם מחזיקים)|"
                         r"(?P<sf>מחליפות ביניהן את כל הנעילות שהן מחזיקות))\."),
        ("handall", "", r"<a> (?:(?P<gm>מעביר)|(?P<gf>מעבירה)) ל<b> את כל הנעילות ש(?:(?P<hm>הוא מחזיק)|(?P<hf>היא מחזיקה))\."),
        ("restart", "service", rf"<a> {V(('מפעיל מחדש', 'מפעילה מחדש'))} את השירות <s>\."),
        ("dash", "", rf"<a> {V(('בודק', 'בודקת'))} את הדשבורד\."),
        ("readflag", "flag", rf"<a> {V(('קורא', 'קוראת'))} את ערך הדגל <f>\."),
        ("readflag", "flag", rf"<a> {V(('מוסיף', 'מוסיפה'))} הערה לדגל <f>\."),
        ("readsvc", "service", r"<a> צופה בלוגים של השירות <s>\."),
    ]


def replay(s: Session) -> list[str]:
    he = s.lang == "he"
    people = [p[0] for p in PEOPLE_HE] if he else list(PEOPLE_EN)
    gender = {p[0]: p[1] for p in PEOPLE_HE} if he else {}
    vals = {"he": {ON_HE: True, OFF_HE: False}, "en": {"on": True, "off": False}}[s.lang]
    groups = {"a": _alt(people), "b": _alt(people), "f": _alt(FLAGS), "s": _alt(SERVICES), "l": _alt(LOCKS),
              "x": _alt(LOCKS), "y": _alt(LOCKS), "v": _alt(vals)}

    def rx(template: str):
        for k, v in groups.items():
            template = template.replace(f"<{k}>", f"(?P<{k}>{v})")
        return re.compile("^" + template + "$")

    if he:
        init_pats = [("flag", rx(r"הדגל <f> <v>\.")), ("service", rx(r"השירות <s> <v>\.")),
                     ("held", rx(r"הנעילה <l> אצל <a>\.")), ("free", rx(r"הנעילה <l> פנויה\."))]
        q_pats = [("flag", rx(r"האם הדגל <f> פועל או כבוי עכשיו\?")), ("flag", rx(r"מה מצב הדגל <f> עכשיו, פועל או כבוי\?")),
                  ("service", rx(r"האם השירות <s> פועל או כבוי עכשיו\?")),
                  ("service", rx(r"מה מצב השירות <s> עכשיו, פועל או כבוי\?")),
                  ("lock", rx(r"מי מחזיק בנעילה <l> עכשיו\?")), ("lock", rx(r"אצל מי נמצאת הנעילה <l> עכשיו\?"))]
        nobody = "אף אחד"
    else:
        init_pats = [("flag", rx(r"The flag <f> is <v>\.")), ("service", rx(r"The <s> service is <v>\.")),
                     ("held", rx(r"The <l> lock is held by <a>\.")), ("free", rx(r"The <l> lock is free\."))]
        q_pats = [("flag", rx(r"Is the flag <f> on or off now\?")),
                  ("flag", rx(r"What is the state of the flag <f> now, on or off\?")),
                  ("service", rx(r"Is the <s> service on or off now\?")),
                  ("service", rx(r"What is the state of the <s> service now, on or off\?")),
                  ("lock", rx(r"Who holds the <l> lock now\?")), ("lock", rx(r"Who currently holds the <l> lock\?"))]
        nobody = "nobody"
    ev_pats = [(op, ek, rx(t)) for op, ek, t in _rules(he)]
    word = {True: ON_HE if he else "on", False: OFF_HE if he else "off"}

    state: dict = {}  # ("flag", name) / ("service", name) -> bool
    lock: dict = {}   # lock name -> holder name or None
    for line in s.initial:
        for kind, p in init_pats:
            m = p.match(line)
            if not m:
                continue
            d = m.groupdict()
            if kind in ("flag", "service"):
                key = (kind, d["f"] if kind == "flag" else d["s"])
                assert key not in state, f"duplicate initial line: {line!r}"
                state[key] = vals[d["v"]]
            else:
                assert d["l"] not in lock, f"duplicate initial line: {line!r}"
                lock[d["l"]] = d["a"] if kind == "held" else None
            break
        else:
            raise AssertionError(f"unparsed initial line: {line!r}")

    def check_gender(d, line):
        a = d.get("a")
        if he and a is not None:
            if d.get("gm"):
                assert gender[a] == "m", f"gender mismatch: {line!r}"
            if d.get("gf"):
                assert gender[a] == "f", f"gender mismatch: {line!r}"
            if d.get("hm"):
                assert gender[a] == "m", f"gender mismatch: {line!r}"
            if d.get("hf"):
                assert gender[a] == "f", f"gender mismatch: {line!r}"

    answers = []
    for t in s.turns:
        for e in t.events:
            for op, ek, p in ev_pats:
                m = p.match(e)
                if m:
                    break
            else:
                raise AssertionError(f"unparsed event: {e!r}")
            d = m.groupdict()
            check_gender(d, e)
            key = (ek, d["f"] if ek == "flag" else d.get("s")) if ek else None
            if key is not None:
                assert key in state, f"unknown entity: {e!r}"
            if op == "on":
                state[key] = True
            elif op == "off":
                state[key] = False
            elif op == "toggle":
                state[key] = not state[key]
            elif op == "set":
                state[key] = vals[d["v"]]
            elif op == "restart":
                assert state[key] is True, f"restart of a stopped service: {e!r}"
            elif op in ("readflag", "readsvc", "dash"):
                pass
            elif op == "acquire":
                assert d["l"] in lock and lock[d["l"]] is None, f"lock not free: {e!r}"
                lock[d["l"]] = d["a"]
            elif op == "release":
                assert lock.get(d["l"]) == d["a"], f"actor does not hold the lock: {e!r}"
                lock[d["l"]] = None
            elif op == "handoff":
                assert lock.get(d["l"]) == d["a"] and d["b"] != d["a"], f"bad handoff: {e!r}"
                lock[d["l"]] = d["b"]
            elif op == "swap":
                x, y, a, b = d["x"], d["y"], d["a"], d["b"]
                assert x != y and a != b and x in lock and y in lock, f"bad swap: {e!r}"
                assert {lock[x], lock[y]} == {a, b}, f"swap actors do not hold the locks: {e!r}"
                if he:
                    assert (d["sf"] is not None) == (gender[a] == "f" and gender[b] == "f"), f"plural gender: {e!r}"
                lock[x], lock[y] = lock[y], lock[x]
            elif op == "tryfail":
                assert lock.get(d["l"]) not in (None, d["a"]), f"lock not busy for the actor: {e!r}"
            elif op == "flipall":
                for k in state:
                    if k[0] == "flag":
                        state[k] = not state[k]
            elif op == "swapall":
                a, b = d["a"], d["b"]
                mine_a = [l for l, h in lock.items() if h == a]
                mine_b = [l for l, h in lock.items() if h == b]
                assert a != b and mine_a and mine_b, f"bad swapall: {e!r}"
                if he:
                    assert (d["sf"] is not None) == (gender[a] == "f" and gender[b] == "f"), f"plural gender: {e!r}"
                for l in mine_a:
                    lock[l] = b
                for l in mine_b:
                    lock[l] = a
            elif op == "handall":
                a, b = d["a"], d["b"]
                mine = [l for l, h in lock.items() if h == a]
                assert a != b and mine, f"bad handall: {e!r}"
                for l in mine:
                    lock[l] = b
            else:
                raise AssertionError(f"unhandled op {op}")
        for kind, p in q_pats:
            m = p.match(t.question)
            if not m:
                continue
            d = m.groupdict()
            if kind == "lock":
                assert d["l"] in lock, t.question
                answers.append(lock[d["l"]] or nobody)
            else:
                key = (kind, d["f"] if kind == "flag" else d["s"])
                assert key in state, t.question
                answers.append(word[state[key]])
            break
        else:
            raise AssertionError(f"unparsed question: {t.question!r}")
    return answers
