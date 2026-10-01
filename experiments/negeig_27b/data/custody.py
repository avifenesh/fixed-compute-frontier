"""custody: items handed between people and places; question: where is item X now (a person or a place).

Events (each keeps every item with exactly one holder):
  give    a person hands an item they hold to another person                 (move)
  swap    two people exchange the items they hold                            (transposition of holders)
  leave   a person leaves an item they hold at a place                       (move)
  take    a person picks up an item from the place it is in                  (move)
  walk    a person walks somewhere, or looks at an item                      (no change)
Implicit-effect events change holders WITHOUT naming the items, so the answer cannot be read off the last line
that mentions an item (a lookup attention can do); it has to be carried as state:
  swapall  two people swap everything they are carrying                     (permutation, items unnamed)
  handall  a person gives everything they carry to another person           (move, items unnamed)
  moveall  everything in one place is moved to another place                (move, items unnamed)
English and Hebrew; Hebrew verbs agree with the actor's gender.
"""

from __future__ import annotations

import re

from common import Session, split_pool
from pools import ITEMS_EN, ITEMS_HE, PEOPLE_EN, PEOPLE_HE, PLACES_EN, PLACES_HE

V_HE = {  # verb forms by gender
    "give": {"m": "מעביר", "f": "מעבירה"}, "leave": {"m": "משאיר", "f": "משאירה"},
    "take": {"m": "לוקח", "f": "לוקחת"}, "walk": {"m": "הולך", "f": "הולכת"},
}


class Sim:
    SYSTEM = {
        "en": "You are tracking where items are in an office. You get the initial state, then events in order. "
              "After each batch, answer the question with only the current holder: a person's name or a place "
              "(for example: Omer, or the storage room).",
        "he": "את/ה עוקב/ת אחרי המיקום של חפצים במשרד. מקבלים מצב התחלתי ואחריו אירועים לפי הסדר. "
              "אחרי כל קבוצת אירועים עונים על השאלה רק במחזיק הנוכחי: שם של אדם או מקום (למשל: עומר, או המחסן).",
    }

    def __init__(self, rng, lang, split):
        self.rng, self.lang = rng, lang
        people = split_pool(PEOPLE_EN if lang == "en" else PEOPLE_HE, split)
        items = split_pool(ITEMS_EN if lang == "en" else ITEMS_HE, split)
        places = split_pool(PLACES_EN if lang == "en" else PLACES_HE, split)
        self.people = rng.sample(people, rng.randint(4, 7))
        self.items = rng.sample(items, rng.randint(4, 8))
        self.places = rng.sample(places, rng.randint(2, 4))
        # holder: ("p", person) or ("l", place)
        self.holder = {}
        for it in self.items:
            self.holder[it] = ("p", rng.choice(self.people)) if rng.random() < 0.7 else ("l", rng.choice(self.places))

    # ---------- rendering helpers
    def _pname(self, p):
        return p if self.lang == "en" else p[0]

    def _g(self, p):
        return p[1]

    def _place(self, l, form):  # form: def / in / to / from
        if self.lang == "en":
            return {"def": "the " + l, "in": "in the " + l, "to": "to the " + l, "from": "from the " + l}[form]
        return l[{"def": 0, "in": 1, "to": 2, "from": 3}[form]]

    def _holder_str(self, h):
        return self._pname(h[1]) if h[0] == "p" else self._place(h[1], "def")

    # ---------- Sim interface
    def initial(self):
        out = []
        for it in self.items:
            kind, h = self.holder[it]
            if self.lang == "en":
                out.append(f"{h} has the {it}." if kind == "p" else f"The {it} is in the {h}.")
            else:
                out.append(f"{it} אצל {h[0]}." if kind == "p" else f"{it} {h[1]}.")
        return out

    def event(self):
        rng = self.rng
        held = [it for it in self.items if self.holder[it][0] == "p"]
        placed = [it for it in self.items if self.holder[it][0] == "l"]
        opts = ["walk"]
        if held:
            opts += ["give"] * 4 + ["leave"] * 2
        if placed:
            opts += ["take"] * 2
        pairs = [(a, b) for a in held for b in held if self.holder[a][1] != self.holder[b][1]]
        if pairs:
            opts += ["swap"] * 3
        carriers = sorted({self.holder[it][1] for it in held}, key=str)
        if carriers:
            opts += ["swapall"] * 4 + ["handall"]
        if placed:
            opts += ["moveall"]
        kind = rng.choice(opts)
        he = self.lang == "he"
        if kind == "swapall":
            a = rng.choice(carriers)
            b = rng.choice([p for p in self.people if p != a])
            for it in self.items:
                if self.holder[it] == ("p", a):
                    self.holder[it] = ("p", b)
                elif self.holder[it] == ("p", b):
                    self.holder[it] = ("p", a)
            if not he:
                return f"{a} and {b} swap everything they are carrying."
            fem = a[1] == "f" and b[1] == "f"
            return f"{a[0]} ו{b[0]} {'מחליפות ביניהן את כל מה שהן מחזיקות' if fem else 'מחליפים ביניהם את כל מה שהם מחזיקים'}."
        if kind == "handall":
            a = rng.choice(carriers)
            b = rng.choice([p for p in self.people if p != a])
            for it in self.items:
                if self.holder[it] == ("p", a):
                    self.holder[it] = ("p", b)
            return (f"{a} gives everything they are carrying to {b}." if not he
                    else f"{a[0]} {'נותן' if a[1] == 'm' else 'נותנת'} ל{b[0]} את כל מה {'שהוא מחזיק' if a[1] == 'm' else 'שהיא מחזיקה'}.")
        if kind == "moveall":
            src = self.holder[rng.choice(placed)][1]
            dst = rng.choice([l for l in self.places if l != src]) if len(self.places) > 1 else None
            if dst is not None:
                for it in self.items:
                    if self.holder[it] == ("l", src):
                        self.holder[it] = ("l", dst)
                if not he:
                    return f"Everything {self._place(src, 'in')} is moved {self._place(dst, 'to')}."
                return f"כל מה שנמצא {self._place(src, 'in')} מועבר {self._place(dst, 'to')}."
            kind = "walk"
        if kind == "give":
            it = rng.choice(held)
            a = self.holder[it][1]
            b = rng.choice([p for p in self.people if p != a])
            self.holder[it] = ("p", b)
            return (f"{a} hands the {it} to {b}." if not he else f"{a[0]} {V_HE['give'][a[1]]} את {it} ל{b[0]}.")
        if kind == "swap":
            x, y = rng.choice(pairs)
            a, b = self.holder[x][1], self.holder[y][1]
            self.holder[x], self.holder[y] = ("p", b), ("p", a)
            if not he:
                return f"{a} and {b} swap the {x} and the {y}."
            fem = a[1] == "f" and b[1] == "f"
            return f"{a[0]} ו{b[0]} {'מחליפות ביניהן' if fem else 'מחליפים ביניהם'} את {x} ואת {y}."
        if kind == "leave":
            it = rng.choice(held)
            a = self.holder[it][1]
            l = rng.choice(self.places)
            self.holder[it] = ("l", l)
            return (f"{a} leaves the {it} {self._place(l, 'in')}." if not he
                    else f"{a[0]} {V_HE['leave'][a[1]]} את {it} {self._place(l, 'in')}.")
        if kind == "take":
            it = rng.choice(placed)
            l = self.holder[it][1]
            a = rng.choice(self.people)
            self.holder[it] = ("p", a)
            return (f"{a} picks up the {it} {self._place(l, 'from')}." if not he
                    else f"{a[0]} {V_HE['take'][a[1]]} את {it} {self._place(l, 'from')}.")
        a = rng.choice(self.people)  # walk: no state change
        if rng.random() < 0.5:
            l = rng.choice(self.places)
            return f"{a} walks {self._place(l, 'to')}." if not he else f"{a[0]} {V_HE['walk'][a[1]]} {self._place(l, 'to')}."
        it = rng.choice(self.items)
        return f"{a} asks about the {it}." if not he else f"{a[0]} {'שואל' if a[1] == 'm' else 'שואלת'} על {it}."

    def ask(self):
        it = self.rng.choice(self.items)
        q = (f"Where is the {it} now?" if self.lang == "en" else f"איפה {it} עכשיו?")
        return q, self._holder_str(self.holder[it]), {"item": it}


# ---------- independent checker: recompute every answer from the rendered text only
def replay(s: Session) -> list[str]:
    he = s.lang == "he"
    people = {p[0] for p in PEOPLE_HE} if he else set(PEOPLE_EN)
    items = ITEMS_HE if he else ITEMS_EN
    places = PLACES_HE if he else PLACES_EN
    place_by_form = {}
    for l in places:
        if he:
            for i, f in enumerate(("def", "in", "to", "from")):
                place_by_form[l[i]] = l[0]
        else:
            place_by_form["the " + l] = "the " + l
            for pre in ("in", "to", "from"):
                place_by_form[f"{pre} the {l}"] = "the " + l
    item_alt = "|".join(sorted((re.escape(i) for i in items), key=len, reverse=True))
    form_alt = "|".join(sorted((re.escape(f) for f in place_by_form), key=len, reverse=True))
    holder: dict[str, str] = {}
    if he:
        pat = {
            "init_p": re.compile(rf"^({item_alt}) אצל (\S+)\.$"),
            "init_l": re.compile(rf"^({item_alt}) ({form_alt})\.$"),
            "give": re.compile(rf"^(\S+) (?:מעביר|מעבירה) את ({item_alt}) ל(\S+)\.$"),
            "swap": re.compile(rf"^(\S+) ו(\S+) (?:מחליפים ביניהם|מחליפות ביניהן) את ({item_alt}) ואת ({item_alt})\.$"),
            "leave": re.compile(rf"^(\S+) (?:משאיר|משאירה) את ({item_alt}) ({form_alt})\.$"),
            "take": re.compile(rf"^(\S+) (?:לוקח|לוקחת) את ({item_alt}) ({form_alt})\.$"),
            "noop": re.compile(rf"^(\S+) (?:הולך|הולכת) ({form_alt})\.$|^(\S+) (?:שואל|שואלת) על ({item_alt})\.$"),
            "swapall": re.compile(r"^(\S+) ו(\S+) (?:מחליפים ביניהם את כל מה שהם מחזיקים|מחליפות ביניהן את כל מה שהן מחזיקות)\.$"),
            "handall": re.compile(r"^(\S+) (?:נותן|נותנת) ל(\S+) את כל מה (?:שהוא מחזיק|שהיא מחזיקה)\.$"),
            "moveall": re.compile(rf"^כל מה שנמצא ({form_alt}) מועבר ({form_alt})\.$"),
            "q": re.compile(rf"^איפה ({item_alt}) עכשיו\?$"),
        }
    else:
        pat = {
            "init_p": re.compile(rf"^(\S+) has the ({item_alt})\.$"),
            "init_l": re.compile(rf"^The ({item_alt}) is ({form_alt})\.$"),
            "give": re.compile(rf"^(\S+) hands the ({item_alt}) to (\S+)\.$"),
            "swap": re.compile(rf"^(\S+) and (\S+) swap the ({item_alt}) and the ({item_alt})\.$"),
            "leave": re.compile(rf"^(\S+) leaves the ({item_alt}) ({form_alt})\.$"),
            "take": re.compile(rf"^(\S+) picks up the ({item_alt}) ({form_alt})\.$"),
            "noop": re.compile(rf"^(\S+) walks ({form_alt})\.$|^(\S+) asks about the ({item_alt})\.$"),
            "swapall": re.compile(r"^(\S+) and (\S+) swap everything they are carrying\.$"),
            "handall": re.compile(r"^(\S+) gives everything they are carrying to (\S+)\.$"),
            "moveall": re.compile(rf"^Everything ({form_alt}) is moved ({form_alt})\.$"),
            "q": re.compile(rf"^Where is the ({item_alt}) now\?$"),
        }
    for line in s.initial:
        m = pat["init_p"].match(line)
        if m:
            it, who = (m.group(1), m.group(2)) if he else (m.group(2), m.group(1))
            assert who in people, line
            holder[it] = who
            continue
        m = pat["init_l"].match(line)
        assert m, f"unparsed initial line: {line!r}"
        holder[m.group(1)] = place_by_form[m.group(2)]
    answers = []
    for t in s.turns:
        for e in t.events:
            if (m := pat["give"].match(e)):
                a, it, b = m.groups()
                assert holder[it] == a and b in people and b != a, e
                holder[it] = b
            elif (m := pat["swap"].match(e)):
                a, b, x, y = m.groups()
                assert holder[x] == a and holder[y] == b, e
                holder[x], holder[y] = b, a
            elif (m := pat["leave"].match(e)):
                a, it, f = m.groups()
                assert holder[it] == a, e
                holder[it] = place_by_form[f]
            elif (m := pat["take"].match(e)):
                a, it, f = m.groups()
                assert holder[it] == place_by_form[f] and a in people, e
                holder[it] = a
            elif (m := pat["swapall"].match(e)):
                a, b = m.groups()
                assert a in people and b in people and a != b, e
                assert any(h in (a, b) for h in holder.values()), e
                for it, h in list(holder.items()):
                    holder[it] = b if h == a else a if h == b else h
            elif (m := pat["handall"].match(e)):
                a, b = m.groups()
                assert a in people and b in people and a != b and a in holder.values(), e
                for it, h in list(holder.items()):
                    if h == a:
                        holder[it] = b
            elif (m := pat["moveall"].match(e)):
                src, dst = place_by_form[m.group(1)], place_by_form[m.group(2)]
                assert src != dst and src in holder.values(), e
                for it, h in list(holder.items()):
                    if h == src:
                        holder[it] = dst
            elif pat["noop"].match(e):
                pass
            else:
                raise AssertionError(f"unparsed event: {e!r}")
        m = pat["q"].match(t.question)
        assert m, f"unparsed question: {t.question!r}"
        answers.append(holder[m.group(1)])
    return answers
