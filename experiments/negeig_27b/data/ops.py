"""ops: an on-call operations log; questions: incident status, unresolved incidents, service version, on-call.

State: incidents (status open / acknowledged / mitigated / resolved), services with a per-service version
history (a stack), a primary and a secondary on-call person.
Events:
  open      a new incident is opened on a service with a severity            (new entity)
  ack       an open incident is acknowledged by a person                     (open -> acknowledged)
  mit       an open or acknowledged incident is mitigated                    (-> mitigated)
  res       an unresolved incident is resolved                               (-> resolved)
  reopen    a resolved incident is reopened                                  (resolved -> open)
  deploy    a service deploys a version (pushed on its history)              (overwrite + push)
  rollback  a service returns to the version before its latest deploy        (pop)
  hand_p    the primary on-call is replaced by someone off the rota           (overwrite)
  hand_s    the secondary on-call is replaced by someone off the rota         (overwrite)
  swap      primary and secondary exchange roles, named or unnamed            (transposition)
  alert / note / page / fail / canary / maint                                 (no change; fail and canary carry a
                                                                              version that is NOT deployed)
English and Hebrew; Hebrew verbs agree with the actor's gender; IDs, versions and service names stay Latin.
"""

from __future__ import annotations

import random as _random
import re

from common import Session, split_pool
from pools import PEOPLE_EN, PEOPLE_HE

SERVICES = [
    "payments-api", "auth-service", "search-indexer", "billing-worker", "api-gateway", "notification-hub",
    "inventory-api", "checkout-web", "reports-batch", "media-transcoder", "user-profile", "email-relay",
    "order-router", "ledger-sync", "catalog-api", "cart-service", "session-store", "webhook-dispatcher",
    "fraud-scorer", "image-resizer", "geo-lookup", "pricing-engine", "shipping-calc", "audit-logger",
    "config-server", "feature-flags", "rate-limiter", "job-scheduler", "metrics-collector", "log-shipper",
    "cdn-purger", "sms-gateway", "recommendation-api", "tax-service", "refund-worker", "export-service",
    "import-pipeline", "identity-proxy", "profile-cache", "stream-ingest", "doc-renderer", "pdf-builder",
    "chat-relay", "voice-bridge", "sync-engine", "vault-proxy", "queue-broker", "health-probe",
]
ALERTS = [
    "high-latency", "disk-usage", "error-rate", "queue-depth", "cert-expiry", "memory-pressure", "cpu-spike",
    "replica-lag", "conn-pool-full", "gc-pause", "timeout-burst", "retry-storm", "packet-loss", "slow-query",
    "oom-kill", "backlog-growth", "sla-breach", "dns-failure", "token-expiry", "lock-wait",
]


def _id_pool() -> list[str]:
    ids = [f"INC-{n}" for n in range(1000, 10000) if n not in (1234, 2345)]  # the SYSTEM examples are never data
    _random.Random(20260930).shuffle(ids)  # fixed shuffle: the train/eval halves are not parity classes
    return ids


INC_IDS = _id_pool()

STATUS_EN = {"open": "open", "ack": "acknowledged", "mit": "mitigated", "res": "resolved"}
STATUS_HE = {"open": "פתוח", "ack": "בטיפול", "mit": "בשליטה", "res": "נפתר"}
NONE_WORD = {"en": "none", "he": "אין"}

# Templates. Hebrew "<m/f>" marks a gender-dependent form, resolved with the actor's gender.
T = {
    "en": {
        "open": "{inc} opened on {svc} ({sev}).",
        "ack": "{p} acknowledges {inc}.",
        "mit_a": "{p} mitigates {inc}.", "mit_p": "{inc} is mitigated.",
        "res_a": "{p} resolves {inc}.", "res_p": "{inc} is resolved.",
        "reopen_a": "{p} reopens {inc}.", "reopen_p": "{inc} is reopened.",
        "deploy": "{svc} is deployed at {ver}.",
        "rollback": "{svc} is rolled back to its previous version.",
        "hand_p": "{new} takes over as primary on-call from {old}.",
        "hand_s": "{new} takes over as secondary on-call from {old}.",
        "swap_n": "{a} and {b} swap on-call roles.",
        "swap_u": "The primary and secondary on-call swap roles.",
        "alert": "Alert {al} on {svc} auto-clears.",
        "note": "{p} adds a note to {inc}.",
        "page": "{p} is paged for {inc}.",
        "fail": "Deploy of {svc} {ver} failed its health check and was not applied.",
        "canary": "Canary of {svc} {ver} starts at 5% of traffic.",
        "maint": "Maintenance window scheduled for {svc}.",
        "init_svc": "{svc} runs {ver}.",
        "init_oncall": "On-call: {a} is primary, {b} is secondary.",
        "init_inc": "{inc} on {svc} ({sev}) is {st}.",
        "q_status": "What is the status of {inc}?",
        "q_unres": "Which incidents are unresolved right now?",
        "q_ver": "What version is {svc} running?",
        "q_pri": "Who is primary on-call?",
        "q_sec": "Who is secondary on-call?",
    },
    "he": {
        "open": "נפתחה תקלה {inc} בשירות {svc} (חומרה {sev}).",
        "ack": "{p} <מאשר/מאשרת> קבלה של התקלה {inc}.",
        "mit_a": "{p} <מביא/מביאה> את התקלה {inc} תחת שליטה.", "mit_p": "התקלה {inc} הובאה תחת שליטה.",
        "res_a": "{p} <פותר/פותרת> את התקלה {inc}.", "res_p": "התקלה {inc} נפתרה.",
        "reopen_a": "{p} <פותח/פותחת> מחדש את התקלה {inc}.", "reopen_p": "התקלה {inc} נפתחה מחדש.",
        "deploy": "נפרסה גרסה {ver} בשירות {svc}.",
        "rollback": "בוצעה חזרה לגרסה הקודמת בשירות {svc}.",
        "hand_p": "{new} <מקבל/מקבלת> את הכוננות הראשית מ{old}.",
        "hand_s": "{new} <מקבל/מקבלת> את הכוננות המשנית מ{old}.",
        "swap_n": "{a} ו{b} <מחליפים ביניהם/מחליפות ביניהן> את תפקידי הכוננות.",
        "swap_u": "תפקידי הכוננות הראשית והמשנית מוחלפים.",
        "alert": "ההתראה {al} בשירות {svc} נסגרה מעצמה.",
        "note": "{p} <מוסיף/מוסיפה> הערה לתקלה {inc}.",
        "page": "{p} <מקבל/מקבלת> זימון בגלל התקלה {inc}.",
        "fail": "הפריסה של גרסה {ver} בשירות {svc} נכשלה בבדיקת התקינות ולא יושמה.",
        "canary": "פריסת קנרי של גרסה {ver} בשירות {svc} מתחילה ב-5% מהתנועה.",
        "maint": "נקבע חלון תחזוקה לשירות {svc}.",
        "init_svc": "השירות {svc} רץ על גרסה {ver}.",
        "init_oncall": "הכוננות הראשית: {a}, הכוננות המשנית: {b}.",
        "init_inc": "התקלה {inc} בשירות {svc} (חומרה {sev}) בסטטוס {st}.",
        "q_status": "מה הסטטוס של {inc}?",
        "q_unres": "אילו תקלות עדיין לא נפתרו?",
        "q_ver": "איזו גרסה רצה כרגע בשירות {svc}?",
        "q_pri": "מי בכוננות הראשית כרגע?",
        "q_sec": "מי בכוננות המשנית כרגע?",
    },
}


def _vs(v: tuple) -> str:
    return f"v{v[0]}.{v[1]}.{v[2]}"


class Sim:
    SYSTEM = {
        "en": "You are tracking an on-call operations log for a platform team. You get the initial state, then "
              "events in order. Each incident has a status: open (nobody has acknowledged it yet), acknowledged, "
              "mitigated (impact stopped, not yet resolved) or resolved; a reopened incident goes back to open. "
              "Each service keeps a history of deployed versions, newest last: a deploy appends its version, and "
              "a rollback drops the newest entry so the service runs the one before it (further rollbacks keep "
              "going back). Swaps exchange who is primary and who is secondary on-call. When someone takes over a "
              "role from another person, they replace that person in that role and the other role does not "
              "change. Unresolved means any status other than resolved. Alerts, notes, pages, canaries, "
              "maintenance windows and failed deploys change nothing. After each batch, answer the "
              "question with only the value: a status word (open, acknowledged, mitigated or resolved), a "
              "version such as v2.3.1, a person's name, or incident IDs sorted and joined by ', ' (for example "
              "INC-1234, INC-2345; answer none if there are no such incidents).",
        "he": "את/ה עוקב/ת אחרי יומן כוננות (on-call) של צוות פלטפורמה. מקבלים מצב התחלתי ואחריו אירועים לפי "
              "הסדר. לכל תקלה יש סטטוס: פתוח (עוד לא אישרו קבלה), בטיפול (אישרו קבלה), בשליטה (ההשפעה נבלמה אך "
              "התקלה עוד לא נפתרה) או נפתר; תקלה שנפתחה מחדש חוזרת להיות פתוחה. לכל שירות יש היסטוריית גרסאות, "
              "החדשה בסוף: פריסה מוסיפה את הגרסה שנפרסה לסוף ההיסטוריה, וחזרה לגרסה הקודמת מסירה את הגרסה "
              "האחרונה, כך שהשירות רץ על זו שלפניה (חזרות נוספות ממשיכות אחורה). החלפת תפקידים מחליפה בין מי "
              "שבכוננות הראשית למי שבכוננות המשנית. מי שמקבל תפקיד כוננות ממישהו אחר תופס את מקומו באותו "
              "תפקיד, והתפקיד השני לא משתנה. תקלות שלא נפתרו הן כל התקלות שהסטטוס שלהן אינו נפתר. התראות, "
              "הערות, זימונים, קנרי, חלונות תחזוקה ופריסות שנכשלו לא משנים דבר. אחרי כל קבוצת אירועים עונים על השאלה רק בערך עצמו: מילת סטטוס (פתוח, "
              "בטיפול, בשליטה או נפתר), גרסה כמו v2.3.1, שם של אדם, או מספרי תקלות ממוינים ומופרדים בפסיק וברווח, "
              "לדוגמה INC-1234, INC-2345 (אם אין תקלות כאלה עונים: אין).",
    }

    def __init__(self, rng, lang, split):
        self.rng, self.lang = rng, lang
        people = split_pool(PEOPLE_EN if lang == "en" else PEOPLE_HE, split)
        self.people = rng.sample(people, rng.randint(6, 9))
        self.primary, self.secondary = self.people[0], self.people[1]
        self.off = list(self.people[2:])  # on the roster but not on call
        self.svcs = rng.sample(split_pool(SERVICES, split), rng.randint(4, 7))
        self.alerts = split_pool(ALERTS, split)
        self.idpool = list(split_pool(INC_IDS, split))
        self.hist = {s: [(rng.randint(1, 4), rng.randint(0, 9), rng.randint(0, 9))] for s in self.svcs}
        self.status, self.isvc, self.sev = {}, {}, {}
        for _ in range(rng.randint(2, 5)):
            inc = self._new_id()
            self.status[inc] = rng.choices(["open", "ack", "mit", "res"], weights=[25, 25, 20, 30])[0]
            self.isvc[inc] = rng.choice(self.svcs)
            self.sev[inc] = rng.choices(["SEV1", "SEV2", "SEV3"], weights=[1, 3, 3])[0]
        # per-session load: calm sessions often drain to zero unresolved incidents, busy ones keep many open
        self.load = rng.choices(["calm", "mid", "busy"], weights=[35, 40, 25])[0]
        self.step = 0
        self.last: dict[str, int] = {}   # entity -> step of its latest mention (change or distractor)
        self.nchg: dict[str, int] = {}   # entity -> number of real changes

    # ---------- helpers
    def _new_id(self):
        return self.idpool.pop(self.rng.randrange(len(self.idpool)))

    def _nm(self, p):
        return p if self.lang == "en" else p[0]

    def _r(self, key, g="m", **kw):
        s = T[self.lang][key].format(**kw)
        if self.lang == "he":
            s = re.sub(r"<([^/>]+)/([^>]+)>", lambda m: m.group(1) if g == "m" else m.group(2), s)
        return s

    def _g(self, p):
        return p[1] if self.lang == "he" else "m"

    def _touch(self, key, changed=True):
        self.last[key] = self.step
        if changed:
            self.nchg[key] = self.nchg.get(key, 0) + 1

    def _pick_inc(self, pool):
        rec = [k for k in pool if self.step - self.last.get(k, -99) <= 10]
        if rec and self.rng.random() < 0.5:
            return self.rng.choice(rec)
        return self.rng.choice(pool)

    def _vers(self, svc):
        return _vs(self.hist[svc][-1])

    def _next_version(self, svc):
        rng, h = self.rng, self.hist[svc]
        older = [v for v in h[:-1] if v != h[-1]]
        if older and rng.random() < 0.12:  # redeploy of an older tag: a push, not a rollback
            return rng.choice(older)
        ma, mi, pa = h[-1]
        r = rng.random()
        if r < 0.6:
            return (ma, mi, pa + rng.randint(1, 2))
        if r < 0.9:
            return (ma, mi + 1, 0)
        return (ma + 1, 0, 0)

    # ---------- Sim interface
    def initial(self):
        out = [self._r("init_svc", svc=s, ver=self._vers(s)) for s in self.svcs]
        out.append(self._r("init_oncall", a=self._nm(self.primary), b=self._nm(self.secondary)))
        he = self.lang == "he"
        for inc, st in self.status.items():
            out.append(self._r("init_inc", inc=inc, svc=self.isvc[inc], sev=self.sev[inc],
                               st=(STATUS_HE if he else STATUS_EN)[st]))
        return out

    def event(self):
        rng = self.rng
        self.step += 1
        st = self.status
        unres = [i for i in st if st[i] != "res"]
        nu = len(unres)
        om, rm = {"calm": (0.35, 1.8), "mid": (1.0, 1.0), "busy": (1.3, 0.7)}[self.load]
        w = {"open": om * (12 if nu < 2 else 10 if nu < 4 else 6 if nu < 6 else 2 if nu < 8 else 0)}
        if any(s == "open" for s in st.values()):
            w["ack"] = 10
        if any(s in ("open", "ack") for s in st.values()):
            w["mit"] = 8
        if unres:
            w["res"] = rm * (8 if nu <= 3 else 10 if nu < 6 else 16)
        if any(s == "res" for s in st.values()):
            w["reopen"] = 4
        extra = sum(max(0, len(h) - 3) for h in self.hist.values())  # keeps version stacks shallow on average
        w["deploy"] = 16 if extra < 4 else 12
        if any(len(h) >= 2 for h in self.hist.values()):
            w["rollback"] = min(7 + 2 * extra, 20)
        if self.off:
            w["hand_p"], w["hand_s"] = 4, 3
        w["swap"] = 13
        w.update(alert=5, note=5, page=3, fail=4, canary=2, maint=2)
        kind = rng.choices(list(w), weights=list(w.values()))[0]

        if kind == "open":
            inc, svc = self._new_id(), rng.choice(self.svcs)
            sev = rng.choices(["SEV1", "SEV2", "SEV3"], weights=[1, 3, 3])[0]
            st[inc], self.isvc[inc], self.sev[inc] = "open", svc, sev
            self._touch(inc)
            return self._r("open", inc=inc, svc=svc, sev=sev)
        if kind in ("ack", "mit", "res", "reopen"):
            src = {"ack": ("open",), "mit": ("open", "ack"), "res": ("open", "ack", "mit"), "reopen": ("res",)}[kind]
            inc = self._pick_inc([i for i in st if st[i] in src])
            st[inc] = {"ack": "ack", "mit": "mit", "res": "res", "reopen": "open"}[kind]
            self._touch(inc)
            p = rng.choice(self.people)
            if kind == "ack":
                return self._r("ack", self._g(p), p=self._nm(p), inc=inc)
            key = kind + ("_a" if rng.random() < 0.5 else "_p")
            return self._r(key, self._g(p), p=self._nm(p), inc=inc)
        if kind == "deploy":
            svc = rng.choices(self.svcs, weights=[1 / len(self.hist[s]) for s in self.svcs])[0]
            v = self._next_version(svc)
            self.hist[svc].append(v)
            self._touch(svc)
            return self._r("deploy", svc=svc, ver=_vs(v))
        if kind == "rollback":
            cand = [s for s in self.svcs if len(self.hist[s]) >= 2]
            svc = rng.choices(cand, weights=[len(self.hist[s]) - 1 for s in cand])[0]
            self.hist[svc].pop()
            self._touch(svc)
            return self._r("rollback", svc=svc)
        if kind in ("hand_p", "hand_s"):
            new = rng.choice(self.off)
            self.off.remove(new)
            if kind == "hand_p":
                old, self.primary = self.primary, new
            else:
                old, self.secondary = self.secondary, new
            self.off.append(old)
            self._touch("primary" if kind == "hand_p" else "secondary")
            return self._r(kind, self._g(new), new=self._nm(new), old=self._nm(old))
        if kind == "swap":
            self.primary, self.secondary = self.secondary, self.primary
            self._touch("primary")
            self._touch("secondary")
            if rng.random() < 0.5:
                return self._r("swap_u")
            a, b = self.primary, self.secondary
            if rng.random() < 0.5:
                a, b = b, a
            g = "f" if self._g(a) == "f" and self._g(b) == "f" else "m"
            return self._r("swap_n", g, a=self._nm(a), b=self._nm(b))
        # no-change distractors
        if kind == "alert":
            svc = rng.choice(self.svcs)
            self._touch(svc, changed=False)
            return self._r("alert", al=rng.choice(self.alerts), svc=svc)
        if kind in ("note", "page"):
            inc, p = self._pick_inc(list(st)), rng.choice(self.people)
            self._touch(inc, changed=False)
            return self._r(kind, self._g(p), p=self._nm(p), inc=inc)
        svc = rng.choice(self.svcs)
        self._touch(svc, changed=False)
        if kind == "maint":
            return self._r("maint", svc=svc)
        ver = _vs(self._next_version(svc))  # a version that is proposed, never deployed
        return self._r(kind, svc=svc, ver=ver)

    def _pick_q(self, keys):
        """Mostly recently touched or multiply changed entities; about 4% untouched; the rest uniform."""
        rng = self.rng
        untouched = [k for k in keys if k not in self.last]
        r = rng.random()
        if untouched and r < 0.04:
            return rng.choice(untouched)
        hot = [k for k in keys if k in self.last and (self.step - self.last[k] <= 8 or self.nchg.get(k, 0) >= 2)]
        if hot and r < 0.72:
            return rng.choice(hot)
        return rng.choice(keys)

    def ask(self):
        rng = self.rng
        kind = rng.choices(["status", "unres", "ver", "role"], weights=[28, 17, 27, 28])[0]
        he = self.lang == "he"
        if kind == "status":
            keys = list(self.status)
            if rng.random() < 0.9:  # balance the answer classes: pick a status first, then an incident in it
                present = sorted(set(self.status.values()))
                cls = rng.choices(present, weights=[0.35 if c == "res" else 1.0 for c in present])[0]
                keys = [i for i in keys if self.status[i] == cls]
            inc = self._pick_q(keys)
            return (self._r("q_status", inc=inc), (STATUS_HE if he else STATUS_EN)[self.status[inc]],
                    {"kind": "status", "entity": inc})
        if kind == "unres":
            ids = sorted(i for i, s in self.status.items() if s != "res")
            return self._r("q_unres"), (", ".join(ids) if ids else NONE_WORD[self.lang]), {"kind": "unres", "n": len(ids)}
        if kind == "ver":
            svc = self._pick_q(self.svcs)
            return (self._r("q_ver", svc=svc), self._vers(svc),
                    {"kind": "ver", "entity": svc, "depth": len(self.hist[svc])})
        roles = ["primary", "secondary"]
        seen = [r for r in roles if r in self.last]
        role = max(seen, key=lambda r: (self.last[r], rng.random())) if seen and rng.random() < 0.7 else rng.choice(roles)
        who = self.primary if role == "primary" else self.secondary
        return self._r("q_pri" if role == "primary" else "q_sec"), self._nm(who), {"kind": "role", "entity": role}


# ---------- independent checker: recompute every answer from the rendered text only
def replay(s: Session) -> list[str]:
    he = s.lang == "he"
    people = {p[0] for p in PEOPLE_HE} if he else set(PEOPLE_EN)
    INC, VER, SEV = r"(INC-\d{4})", r"(v\d+\.\d+\.\d+)", r"(SEV[123])"
    SVC = "(" + "|".join(sorted((re.escape(x) for x in SERVICES), key=len, reverse=True)) + ")"
    AL = "(" + "|".join(re.escape(x) for x in ALERTS) + ")"
    N = r"([^\s,.]+)"
    if he:
        sw = {"פתוח": "open", "בטיפול": "ack", "בשליטה": "mit", "נפתר": "res"}
        ST = "(" + "|".join(sw) + ")"
        none = "אין"
        P = {
            "i_svc": rf"השירות {SVC} רץ על גרסה {VER}\.",
            "i_on": rf"הכוננות הראשית: {N}, הכוננות המשנית: {N}\.",
            "i_inc": rf"התקלה {INC} בשירות {SVC} \(חומרה {SEV}\) בסטטוס {ST}\.",
            "open": rf"נפתחה תקלה {INC} בשירות {SVC} \(חומרה {SEV}\)\.",
            "ack": rf"{N} (?:מאשר|מאשרת) קבלה של התקלה {INC}\.",
            "mit": rf"{N} (?:מביא|מביאה) את התקלה {INC} תחת שליטה\.|התקלה {INC} הובאה תחת שליטה\.",
            "res": rf"{N} (?:פותר|פותרת) את התקלה {INC}\.|התקלה {INC} נפתרה\.",
            "reopen": rf"{N} (?:פותח|פותחת) מחדש את התקלה {INC}\.|התקלה {INC} נפתחה מחדש\.",
            "deploy": rf"נפרסה גרסה {VER} בשירות {SVC}\.",
            "rollback": rf"בוצעה חזרה לגרסה הקודמת בשירות {SVC}\.",
            "hand_p": rf"{N} (?:מקבל|מקבלת) את הכוננות הראשית מ{N}\.",
            "hand_s": rf"{N} (?:מקבל|מקבלת) את הכוננות המשנית מ{N}\.",
            "swap_n": rf"{N} ו{N} (?:מחליפים ביניהם|מחליפות ביניהן) את תפקידי הכוננות\.",
            "swap_u": r"תפקידי הכוננות הראשית והמשנית מוחלפים\.",
            "noop": "|".join([
                rf"ההתראה {AL} בשירות {SVC} נסגרה מעצמה\.",
                rf"{N} (?:מוסיף|מוסיפה) הערה לתקלה {INC}\.",
                rf"{N} (?:מקבל|מקבלת) זימון בגלל התקלה {INC}\.",
                rf"הפריסה של גרסה {VER} בשירות {SVC} נכשלה בבדיקת התקינות ולא יושמה\.",
                rf"פריסת קנרי של גרסה {VER} בשירות {SVC} מתחילה ב-5% מהתנועה\.",
                rf"נקבע חלון תחזוקה לשירות {SVC}\.",
            ]),
            "q_status": rf"מה הסטטוס של {INC}\?",
            "q_unres": r"אילו תקלות עדיין לא נפתרו\?",
            "q_ver": rf"איזו גרסה רצה כרגע בשירות {SVC}\?",
            "q_pri": r"מי בכוננות הראשית כרגע\?",
            "q_sec": r"מי בכוננות המשנית כרגע\?",
        }
    else:
        sw = {"open": "open", "acknowledged": "ack", "mitigated": "mit", "resolved": "res"}
        ST = "(" + "|".join(sw) + ")"
        none = "none"
        P = {
            "i_svc": rf"{SVC} runs {VER}\.",
            "i_on": rf"On-call: {N} is primary, {N} is secondary\.",
            "i_inc": rf"{INC} on {SVC} \({SEV}\) is {ST}\.",
            "open": rf"{INC} opened on {SVC} \({SEV}\)\.",
            "ack": rf"{N} acknowledges {INC}\.",
            "mit": rf"{N} mitigates {INC}\.|{INC} is mitigated\.",
            "res": rf"{N} resolves {INC}\.|{INC} is resolved\.",
            "reopen": rf"{N} reopens {INC}\.|{INC} is reopened\.",
            "deploy": rf"{SVC} is deployed at {VER}\.",
            "rollback": rf"{SVC} is rolled back to its previous version\.",
            "hand_p": rf"{N} takes over as primary on-call from {N}\.",
            "hand_s": rf"{N} takes over as secondary on-call from {N}\.",
            "swap_n": rf"{N} and {N} swap on-call roles\.",
            "swap_u": r"The primary and secondary on-call swap roles\.",
            "noop": "|".join([
                rf"Alert {AL} on {SVC} auto-clears\.",
                rf"{N} adds a note to {INC}\.",
                rf"{N} is paged for {INC}\.",
                rf"Deploy of {SVC} {VER} failed its health check and was not applied\.",
                rf"Canary of {SVC} {VER} starts at 5% of traffic\.",
                rf"Maintenance window scheduled for {SVC}\.",
            ]),
            "q_status": rf"What is the status of {INC}\?",
            "q_unres": r"Which incidents are unresolved right now\?",
            "q_ver": rf"What version is {SVC} running\?",
            "q_pri": r"Who is primary on-call\?",
            "q_sec": r"Who is secondary on-call\?",
        }
    R = {k: re.compile("^(?:" + v + ")$") for k, v in P.items()}

    def last(m):  # the one group that matched in an alternation
        return [g for g in m.groups() if g is not None]

    status: dict[str, str] = {}
    stack: dict[str, list[str]] = {}
    roles = {"primary": None, "secondary": None}

    def person(x, line):
        assert x in people, f"unknown person {x!r} in {line!r}"

    gen = dict(PEOPLE_HE)
    FEM = {"מאשרת", "מביאה", "פותרת", "פותחת", "מקבלת", "מוסיפה", "מחליפות"}
    MASC = {"מאשר", "מביא", "פותר", "פותח", "מקבל", "מוסיף", "מחליפים"}

    def agree(line):
        """Hebrew verb gender must match the actor: one actor by their own gender, a swap pair feminine only
        when both are feminine."""
        tok = line.split(" ")
        if tok[0] in gen and tok[1] in FEM | MASC:
            assert (tok[1] in FEM) == (gen[tok[0]] == "f"), f"gender disagreement in {line!r}"
        elif tok[0] in gen and len(tok) > 2 and tok[1].startswith("ו") and tok[1][1:] in gen:
            fem = gen[tok[0]] == "f" and gen[tok[1][1:]] == "f"
            assert tok[2] in ("מחליפות" if fem else "מחליפים"), f"gender disagreement in {line!r}"

    for line in s.initial:
        if (m := R["i_svc"].match(line)):
            assert m.group(1) not in stack, line
            stack[m.group(1)] = [m.group(2)]
        elif (m := R["i_on"].match(line)):
            person(m.group(1), line), person(m.group(2), line)
            assert m.group(1) != m.group(2), line
            roles["primary"], roles["secondary"] = m.group(1), m.group(2)
        elif (m := R["i_inc"].match(line)):
            assert m.group(2) in stack and m.group(1) not in status, line
            status[m.group(1)] = sw[m.group(4)]
        else:
            raise AssertionError(f"unparsed initial line: {line!r}")
    assert roles["primary"] is not None, "no on-call line in the initial state"

    answers = []
    for t in s.turns:
        for e in t.events:
            if he:
                agree(e)
            if (m := R["open"].match(e)):
                inc, svc, _sev = m.groups()
                assert inc not in status and svc in stack, e
                status[inc] = "open"
            elif (m := R["ack"].match(e)):
                who, inc = m.groups()
                person(who, e)
                assert status.get(inc) == "open", e
                status[inc] = "ack"
            elif (m := R["mit"].match(e)):
                g = last(m)
                inc = g[-1]
                if len(g) == 2:
                    person(g[0], e)
                assert status.get(inc) in ("open", "ack"), e
                status[inc] = "mit"
            elif (m := R["res"].match(e)):
                g = last(m)
                inc = g[-1]
                if len(g) == 2:
                    person(g[0], e)
                assert status.get(inc) in ("open", "ack", "mit"), e
                status[inc] = "res"
            elif (m := R["reopen"].match(e)):
                g = last(m)
                inc = g[-1]
                if len(g) == 2:
                    person(g[0], e)
                assert status.get(inc) == "res", e
                status[inc] = "open"
            elif (m := R["deploy"].match(e)):
                svc, ver = (m.group(1), m.group(2)) if not he else (m.group(2), m.group(1))
                assert svc in stack and stack[svc][-1] != ver, e
                stack[svc].append(ver)
            elif (m := R["rollback"].match(e)):
                svc = m.group(1)
                assert svc in stack and len(stack[svc]) >= 2, e
                stack[svc].pop()
            elif (m := R["hand_p"].match(e) or R["hand_s"].match(e)):
                new, old = m.groups()
                role = "primary" if R["hand_p"].match(e) else "secondary"
                person(new, e), person(old, e)
                assert roles[role] == old and new not in roles.values(), e
                roles[role] = new
            elif (m := R["swap_n"].match(e)):
                a, b = m.groups()
                assert {a, b} == set(roles.values()), e
                roles["primary"], roles["secondary"] = roles["secondary"], roles["primary"]
            elif R["swap_u"].match(e):
                roles["primary"], roles["secondary"] = roles["secondary"], roles["primary"]
            elif (m := R["noop"].match(e)):
                for g in last(m):
                    if g.startswith("INC-"):
                        assert g in status, e
                    elif g in stack or re.fullmatch(r"v\d+\.\d+\.\d+", g) or g in ALERTS:
                        pass
                    else:
                        person(g, e)
            else:
                raise AssertionError(f"unparsed event: {e!r}")
        q = t.question
        if (m := R["q_status"].match(q)):
            inc = m.group(1)
            assert inc in status, q
            answers.append({v: k for k, v in sw.items()}[status[inc]])
        elif R["q_unres"].match(q):
            ids = sorted(i for i, v in status.items() if v != "res")
            answers.append(", ".join(ids) if ids else none)
        elif (m := R["q_ver"].match(q)):
            assert m.group(1) in stack, q
            answers.append(stack[m.group(1)][-1])
        elif R["q_pri"].match(q):
            answers.append(roles["primary"])
        elif R["q_sec"].match(q):
            answers.append(roles["secondary"])
        else:
            raise AssertionError(f"unparsed question: {q!r}")
    return answers
