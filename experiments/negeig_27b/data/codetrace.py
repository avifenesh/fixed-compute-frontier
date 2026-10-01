"""codetrace: a Python program delivered a few statements at a time; question: value of a variable now.

State: lowercase-named variables of four types (int, bool, str, list of ints). Events are single statements:
  swap    a, b = b, a                       two same-typed variables exchange values        (transposition)
  perm3   a, b, c = b, c, a                 three same-typed variables permuted              (cycle or swap + fixed)
  flip    flag = not flag                   boolean toggle                                   (parity)
  lswap   items[1], items[3] = items[3], items[1]    two elements of one list exchanged      (transposition)
  xswap   a[0], b[2] = b[2], a[0]           elements of two lists exchanged                  (transposition)
  lset    items[2] = 7                      list element overwrite
  rev     items.reverse()                   list reversal                                    (permutation)
  lit     x = 5 / name = 'maple' / flag = True / items = [4, 1, 6]      overwrite with a literal
  copy    x = y (lists: x = y[:], so lists never alias)                 overwrite from another variable
  inc     n += 3 / n -= 3 / n += m                                       increments (ints stay in 0..199)
  logic   a = not b / a = a and b / a = a or b                           boolean overwrite from another boolean
  noop    pass, print(x), x = x, a, b = a, b, x = not not x, n += 0      distractors: nothing changes
English only. Answers are Python repr: 7, True, 'orion', [3, 1, 2].
"""

from __future__ import annotations

import builtins
import contextlib
import keyword
import re

from common import Session, split_pool

# Type-specific name pools (natural code names). Disjoint from each other and from the string values.
BOOL_NAMES = [
    "ready", "done", "active", "enabled", "visible", "locked", "paused", "dirty", "valid", "debug", "verbose",
    "online", "admin", "cached", "muted", "hidden", "loaded", "synced", "armed", "busy", "stale", "public",
    "secure", "dry_run", "is_open", "has_data", "is_empty", "can_edit", "use_cache", "auto_save", "is_ready",
    "has_error", "show_help", "in_stock", "is_admin", "is_done", "is_valid", "found", "failed", "passed",
    "finished", "running", "selected", "checked", "frozen", "expired", "retry_on", "is_live", "has_key",
    "can_undo", "is_new",
]
INT_NAMES = [
    "count", "total", "score", "level", "retries", "offset", "width", "height", "depth", "limit", "index",
    "size", "step", "speed", "rank", "attempts", "balance", "budget", "stock", "volume", "weight", "age",
    "year", "lives", "points", "ticks", "delay", "timeout", "port", "quota", "margin", "stride", "pos", "turn",
    "wave", "batch", "epoch", "hits", "misses", "rounds", "fuel", "gold", "temp", "price", "units", "tally",
    "seats", "max_len", "min_val", "max_size", "retry_count", "page_no", "line_no", "col_no",
]
STR_NAMES = [
    "name", "title", "label", "city", "color", "status", "mode", "owner", "author", "topic", "prefix", "suffix",
    "tag", "word", "greeting", "user", "team", "planet", "fruit", "animal", "team_name", "city_name", "nickname",
    "channel", "region", "project", "shape", "flavor", "season", "genre", "brand", "role", "state", "stage",
    "phase", "species", "lang", "drink", "device", "theme", "dest", "origin", "login_name", "file_name",
    "host_name", "last_name", "first_name", "pet", "dish", "street", "course", "badge_text", "motto",
]
LIST_NAMES = [
    "items", "queue", "stack", "scores", "values", "data", "nums", "buffer", "row", "cols", "rows", "heights",
    "ranks", "lengths", "weights", "levels", "counts", "totals", "prices", "ages", "marks", "slots", "deck",
    "hand", "pile", "line", "lane", "track", "samples", "readings", "digits", "bins", "cells", "shelf",
    "window_vals", "history", "results", "stats", "pixels", "order", "batch_ids", "temps", "speeds", "tokens",
    "ids", "sizes", "layers", "chain", "ring", "tape", "board", "path",
]
WORDS = [
    "orion", "maple", "cedar", "amber", "falcon", "harbor", "willow", "comet", "ember", "lotus", "marble",
    "nimbus", "otter", "pebble", "quartz", "raven", "saffron", "tundra", "velvet", "walnut", "yarrow", "zephyr",
    "acorn", "birch", "cobalt", "dune", "fjord", "glacier", "heron", "indigo", "jasper", "kelp", "lagoon",
    "meadow", "nectar", "onyx", "pepper", "quill", "ridge", "sable", "thistle", "umber", "violet", "wren",
    "xenon", "yonder", "zinc", "basil", "clover", "dahlia", "elm", "fern", "garnet", "hazel", "iris", "juniper",
    "kiwi", "lemon", "mango", "nutmeg", "olive", "plum", "quince", "radish", "sorrel", "tulip", "urchin",
    "vanilla", "wheat", "yam", "zucchini", "anchor", "beacon", "canyon", "drift", "eagle", "flint", "grove",
    "helix", "island", "jungle", "koala", "lantern", "mirror", "nebula", "oasis", "prairie", "quasar", "river",
    "summit", "timber", "umbra", "valley", "whisper", "yucca", "zenith", "atlas", "breeze", "coral", "delta",
]

_POOLS = {"bool": BOOL_NAMES, "int": INT_NAMES, "str": STR_NAMES, "list": LIST_NAMES}
_ALL = BOOL_NAMES + INT_NAMES + STR_NAMES + LIST_NAMES + WORDS
assert len(set(_ALL)) == len(_ALL), "codetrace pools overlap"
assert all(re.fullmatch(r"[a-z][a-z_]*", w) for w in _ALL)
assert all(not keyword.iskeyword(w) and not hasattr(builtins, w) for w in _ALL)
assert all(len(p) >= 48 for p in _POOLS.values()) and len(WORDS) >= 96


def _stem(w):
    w = re.sub(r"_name$", "", w)
    return w.rstrip("s") if len(w) > 3 else w


def _check_hygiene(system):
    """A pool word in the SYSTEM prompt would put an eval name (or a train name) into every session's text;
    names that differ only by _name or a plural s must land in the same split."""
    sys_words = set(re.findall(r"[a-z_]+", system.lower()))
    assert not sys_words & set(_ALL), f"pool words in SYSTEM: {sorted(sys_words & set(_ALL))}"
    for pool in list(_POOLS.values()) + [WORDS]:
        side = {w: i % 2 for i, w in enumerate(pool)}
        for a in pool:
            for b in pool:
                if a < b and _stem(a) == _stem(b):
                    assert side[a] == side[b], f"near-duplicate names straddle the split: {a!r}, {b!r}"


INT_MAX = 199
PERMS3 = [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


class Sim:
    SYSTEM = {
        "en": "You are tracking the variables of a Python program. You get the initial assignments, then more "
              "statements in the sequence they run; each statement runs right after the previous one, and nothing "
              "else changes any variable. Statements such as pass or print(x) change nothing. Lists are copied "
              "with x = y[:], so two list variables never share elements. Indices start at 0. After each group of "
              "statements, answer the question with only the current value, written exactly as Python's repr "
              "prints it: integers like 7, booleans True or False, strings in single quotes like 'pine', lists "
              "like [3, 1, 2] (one space after each comma).",
    }

    def __init__(self, rng, lang, split):
        assert lang == "en"
        self.rng, self.lang = rng, lang
        pools = {t: split_pool(p, split) for t, p in _POOLS.items()}
        self.words = split_pool(WORDS, split)
        while True:
            c = {"int": rng.randint(0, 4), "bool": rng.randint(0, 3), "str": rng.randint(0, 3),
                 "list": rng.randint(0, 3)}
            if (4 <= sum(c.values()) <= 9 and sum(1 for v in c.values() if v) >= 2
                    and any(v >= 2 for v in c.values())):
                break
        self.by_type = {t: rng.sample(pools[t], n) for t, n in c.items()}
        self.typ, self.val = {}, {}
        for t, names in self.by_type.items():
            for n in names:
                self.typ[n] = t
                self.val[n] = self._fresh(t)
        self.order = list(self.typ)
        rng.shuffle(self.order)
        # idle variables: never the target of a state-changing statement, so some questions stay untouched
        idle = []
        if rng.random() < 0.7:
            for n in rng.sample(self.order, rng.choice([1, 1, 2])):
                rest = {t: [m for m in ns if m != n and m not in idle] for t, ns in self.by_type.items()}
                if sum(map(len, rest.values())) >= 3 and any(len(ns) >= 2 for ns in rest.values()):
                    idle.append(n)
        self.idle = idle
        self.act = {t: [n for n in ns if n not in idle] for t, ns in self.by_type.items()}
        self.act_names = [n for n in self.order if n not in idle]
        self.t = 0
        self.touch = {n: 0 for n in self.order}   # times the target of a state-changing statement
        self.nchg = {n: 0 for n in self.order}    # times the value actually changed
        self.last = {n: -1 for n in self.order}
        self.prev = None

    # ---------- values and rendering
    def _fresh(self, t, cur=None):
        rng = self.rng
        for _ in range(8):
            if t == "int":
                v = rng.randint(0, 99)
            elif t == "bool":
                v = rng.random() < 0.5
            elif t == "str":
                v = rng.choice(self.words)
            else:
                v = rng.sample(range(1, 10), rng.randint(3, 6))
            if cur is None or v != cur or rng.random() < 0.08:
                return v
        return v

    @staticmethod
    def _lit(v):
        return repr(v)

    def initial(self):
        return [f"{n} = {self._lit(self.val[n])}" for n in self.order]

    # ---------- events
    def _pick_pair(self, names):
        rng = self.rng
        diff = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if self.val[a] != self.val[b]]
        a, b = rng.choice(diff) if diff and rng.random() < 0.92 else rng.sample(names, 2)
        return (a, b) if rng.random() < 0.5 else (b, a)

    def event(self):
        rng, T, val = self.rng, self.act, self.val
        swap_types = [t for t, ns in T.items() if len(ns) >= 2]
        perm_types = [t for t, ns in T.items() if len(ns) >= 3]
        kinds, weights = [], []

        def add(k, w):
            kinds.append(k)
            weights.append(w)

        if swap_types:
            add("swap", 18)
            add("copy", 7)
        if perm_types:
            add("perm3", 5)
        if T["bool"]:
            add("flip", 16)
        if T["list"]:
            add("lswap", 12)
            add("lset", 4)
            add("rev", 3)
        if len(T["list"]) >= 2:
            add("xswap", 5)
        if T["int"]:
            add("inc", 12)
        if len(T["int"]) >= 2:
            add("incvar", 3)
        if len(T["bool"]) >= 2:
            add("logic", 3)
        add("lit", 8)
        add("noop", 8)
        kind = rng.choices(kinds, weights)[0]
        before = {n: (list(v) if isinstance(v, list) else v) for n, v in val.items()}
        text, targets = getattr(self, "_ev_" + kind)()
        self.t += 1
        for n in targets:
            self.touch[n] += 1
            self.last[n] = self.t
            if val[n] != before[n]:
                self.nchg[n] += 1
        return text

    def _ev_swap(self):
        t = self.rng.choice([t for t, ns in self.act.items() if len(ns) >= 2])
        a, b = self._pick_pair(self.act[t])
        self.val[a], self.val[b] = self.val[b], self.val[a]
        return f"{a}, {b} = {b}, {a}", [a, b]

    def _ev_perm3(self):
        rng = self.rng
        t = rng.choice([t for t, ns in self.act.items() if len(ns) >= 3])
        for _ in range(6):
            x = rng.sample(self.act[t], 3)
            if any(self.val[x[0]] != self.val[y] for y in x[1:]):
                break
        p = rng.choice(PERMS3)
        rhs = [x[p[i]] for i in range(3)]
        old = {n: self.val[n] for n in x}
        for i in range(3):
            self.val[x[i]] = old[rhs[i]]
        return f"{', '.join(x)} = {', '.join(rhs)}", x

    def _ev_flip(self):
        v = self.rng.choice(self.act["bool"])
        self.val[v] = not self.val[v]
        return f"{v} = not {v}", [v]

    def _ev_lswap(self):
        rng = self.rng
        L = rng.choice(self.act["list"])
        i, j = rng.sample(range(len(self.val[L])), 2)
        self.val[L][i], self.val[L][j] = self.val[L][j], self.val[L][i]
        return f"{L}[{i}], {L}[{j}] = {L}[{j}], {L}[{i}]", [L]

    def _ev_xswap(self):
        rng = self.rng
        A, B = rng.sample(self.act["list"], 2)
        i, j = rng.randrange(len(self.val[A])), rng.randrange(len(self.val[B]))
        self.val[A][i], self.val[B][j] = self.val[B][j], self.val[A][i]
        return f"{A}[{i}], {B}[{j}] = {B}[{j}], {A}[{i}]", [A, B]

    def _ev_lset(self):
        rng = self.rng
        L = rng.choice(self.act["list"])
        i = rng.randrange(len(self.val[L]))
        k = rng.randint(1, 9)
        self.val[L][i] = k
        return f"{L}[{i}] = {k}", [L]

    def _ev_rev(self):
        L = self.rng.choice(self.act["list"])
        self.val[L].reverse()
        return f"{L}.reverse()", [L]

    def _ev_lit(self):
        v = self.rng.choice(self.act_names)
        new = self._fresh(self.typ[v], self.val[v])
        self.val[v] = new
        return f"{v} = {self._lit(new)}", [v]

    def _ev_copy(self):
        t = self.rng.choice([t for t, ns in self.act.items() if len(ns) >= 2])
        x, y = self._pick_pair(self.act[t])
        if t == "list":
            self.val[x] = list(self.val[y])
            return f"{x} = {y}[:]", [x]
        self.val[x] = self.val[y]
        return f"{x} = {y}", [x]

    def _ev_inc(self):
        rng = self.rng
        v = rng.choice(self.act["int"])
        d = rng.randint(1, 9) if rng.random() < 0.8 else rng.randint(10, 30)
        cur = self.val[v]
        minus = (cur + d > INT_MAX) or (cur - d >= 0 and rng.random() < 0.4)
        self.val[v] = cur - d if minus else cur + d
        return f"{v} {'-=' if minus else '+='} {d}", [v]

    def _ev_incvar(self):
        rng = self.rng
        v, w = rng.sample(self.act["int"], 2)
        cur, d = self.val[v], self.val[w]
        ops = [o for o, ok in (("+=", cur + d <= INT_MAX), ("-=", cur - d >= 0)) if ok]
        if not ops:
            return self._ev_inc()
        op = rng.choice(ops)
        self.val[v] = cur + d if op == "+=" else cur - d
        return f"{v} {op} {w}", [v]

    def _ev_logic(self):
        rng = self.rng
        a, b = rng.sample(self.act["bool"], 2)
        form = rng.choice(["not", "and", "or"])
        if form == "not":
            self.val[a] = not self.val[b]
            return f"{a} = not {b}", [a]
        self.val[a] = (self.val[a] and self.val[b]) if form == "and" else (self.val[a] or self.val[b])
        return f"{a} = {a} {form} {b}", [a]

    def _ev_noop(self):
        rng, T = self.rng, self.act
        opts = ["pass", "print", "self"]
        if any(len(ns) >= 2 for ns in T.values()):
            opts.append("idtuple")
        if T["list"]:
            opts.append("idelem")
        if T["bool"]:
            opts.append("notnot")
        if T["int"]:
            opts.append("plus0")
        o = rng.choice(opts)
        if o == "pass":
            return "pass", []
        if o == "print":
            return f"print({rng.choice(self.order)})", []
        if o == "self":
            v = rng.choice(self.order)
            return f"{v} = {v}", []
        if o == "idtuple":
            t = rng.choice([t for t, ns in T.items() if len(ns) >= 2])
            a, b = rng.sample(T[t], 2)
            return f"{a}, {b} = {a}, {b}", []
        if o == "idelem":
            L = rng.choice(T["list"])
            i, j = rng.sample(range(len(self.val[L])), 2)
            return f"{L}[{i}], {L}[{j}] = {L}[{i}], {L}[{j}]", []
        if o == "notnot":
            v = rng.choice(T["bool"])
            return f"{v} = not not {v}", []
        v = rng.choice(T["int"])
        return f"{v} {rng.choice(['+=', '-='])} 0", []

    # ---------- questions
    def ask(self):
        rng, names = self.rng, self.order
        untouched = [n for n in names if self.touch[n] == 0 and self.nchg[n] == 0]
        recent = [n for n in names if self.touch[n] and self.t - self.last[n] < 8]
        multi = [n for n in names if self.nchg[n] >= 2]
        r = rng.random()
        if r < 0.07 and untouched:
            v, kind = rng.choice(untouched), "untouched"
        elif r < 0.19 and self.prev is not None:
            v, kind = self.prev, "repeat"
        elif r < 0.72 and recent:
            v, kind = rng.choice(recent), "recent"
        elif r < 0.90 and multi:
            v, kind = rng.choices(multi, [self.nchg[n] for n in multi])[0], "multi"
        else:
            v, kind = rng.choice(names), "any"
        self.prev = v
        meta = {"var": v, "kind": kind, "type": self.typ[v], "touches": self.touch[v], "changes": self.nchg[v],
                "since": (self.t - self.last[v]) if self.last[v] >= 0 else None}
        if self.typ[v] == "list" and rng.random() < 0.25:
            i = rng.randrange(len(self.val[v]))
            expr, ans = f"{v}[{i}]", repr(self.val[v][i])
        else:
            expr, ans = v, repr(self.val[v])
        meta["expr"] = expr
        q = rng.choice(["What is the value of {} now?", "What is {} now?"]).format(expr)
        return q, ans, meta


_check_hygiene(Sim.SYSTEM["en"])


# ---------- independent checker: run the program text with exec(), compare repr()
_N = r"([a-z][a-z_]*)"
_NN = r"[a-z][a-z_]*"
_LIT = r"(-?\d+|True|False|'[a-z]+'|\[\d+(?:, \d+)*\])"
_SUB = rf"({_NN})\[(\d+)\]"
_PATTERNS = [
    ("lit", re.compile(rf"^{_N} = {_LIT}$")),
    ("copy", re.compile(rf"^{_N} = {_N}\[:\]$")),
    ("notnot", re.compile(rf"^{_N} = not not {_N}$")),
    ("not", re.compile(rf"^{_N} = not {_N}$")),
    ("logic", re.compile(rf"^{_N} = {_N} (and|or) {_N}$")),
    ("var", re.compile(rf"^{_N} = {_N}$")),
    ("augv", re.compile(rf"^{_N} (\+=|-=) {_N}$")),
    ("augl", re.compile(rf"^{_N} (\+=|-=) (\d+)$")),
    ("tuple", re.compile(rf"^({_NN}(?:, {_NN})+) = ({_NN}(?:, {_NN})+)$")),
    ("subswap", re.compile(rf"^{_SUB}, {_SUB} = {_SUB}, {_SUB}$")),
    ("subset", re.compile(rf"^{_SUB} = (-?\d+)$")),
    ("reverse", re.compile(rf"^{_N}\.reverse\(\)$")),
    ("pass", re.compile(r"^pass$")),
    ("print", re.compile(rf"^print\({_N}\)$")),
]
_QUESTION = re.compile(rf"^What is (?:the value of )?({_NN})(?:\[(\d+)\])? now\?$")


class _Null:
    def write(self, s):
        return len(s)

    def flush(self):
        pass


def _parse(line: str):
    for kind, pat in _PATTERNS:
        m = pat.match(line)
        if m:
            return kind, m.groups()
    raise AssertionError(f"unparsed line: {line!r}")


def _names_of(kind, g):
    if kind in ("tuple",):
        return g[0].split(", ") + g[1].split(", ")
    if kind == "subswap":
        return [g[0], g[2], g[4], g[6]]
    if kind == "subset":
        return [g[0]]
    if kind == "pass":
        return []
    if kind == "lit":
        return [g[0]]
    if kind in ("augl",):
        return [g[0]]
    if kind == "augv":
        return [g[0], g[2]]
    if kind == "logic":
        return [g[0], g[1], g[3]]
    return [x for x in g]


def _check_line(line, ns, types0):
    """Grammar and semantic validity of one statement against the state it runs in."""
    kind, g = _parse(line)
    for n in _names_of(kind, g):
        assert not keyword.iskeyword(n) and not hasattr(builtins, n), f"reserved name in {line!r}"
        assert n in types0, f"unknown variable {n!r} in {line!r}"
    T = lambda n: type(ns[n])  # noqa: E731
    if kind == "copy":
        assert T(g[1]) is list and T(g[0]) is list, line
    elif kind == "var":
        assert T(g[0]) is T(g[1]), line
        assert T(g[1]) is not list or g[0] == g[1], f"list alias: {line!r}"
    elif kind in ("not", "notnot"):
        assert T(g[0]) is bool and T(g[1]) is bool, line
    elif kind == "logic":
        assert T(g[0]) is bool and T(g[1]) is bool and T(g[3]) is bool, line
    elif kind == "augv":
        assert T(g[0]) is int and T(g[2]) is int, line
    elif kind == "augl":
        assert T(g[0]) is int, line
    elif kind == "tuple":
        lhs, rhs = g[0].split(", "), g[1].split(", ")
        assert len(set(lhs)) == len(lhs) and sorted(lhs) == sorted(rhs), f"not a permutation: {line!r}"
        assert len({T(n) for n in lhs}) == 1, f"mixed types: {line!r}"
    elif kind == "subswap":
        lhs, rhs = [(g[0], g[1]), (g[2], g[3])], [(g[4], g[5]), (g[6], g[7])]
        assert lhs[0] != lhs[1] and sorted(lhs) == sorted(rhs), f"not a swap: {line!r}"
        assert all(T(n) is list for n, _ in lhs), line
    elif kind == "subset":
        assert T(g[0]) is list, line
    elif kind == "reverse":
        assert T(g[0]) is list, line
    elif kind == "lit":
        assert type(eval(g[1])) is types0[g[0]], f"literal type change: {line!r}"  # literal text from the grammar
    return kind


def _state_ok(ns, types0, line):
    names = set(ns) - {"__builtins__"}
    assert names == set(types0), f"variables changed by {line!r}: {sorted(names ^ set(types0))}"
    for n, t in types0.items():
        assert type(ns[n]) is t, f"{n} changed type to {type(ns[n]).__name__} by {line!r}"
    lists = [ns[n] for n, t in types0.items() if t is list]
    assert len({id(x) for x in lists}) == len(lists), f"lists share items after {line!r}"
    for n in types0:
        if types0[n] is list:
            assert all(type(x) is int for x in ns[n]), f"list {n} holds non-int after {line!r}"
            assert 3 <= len(ns[n]) <= 6 and all(1 <= x <= 9 for x in ns[n]), f"list {n} out of format after {line!r}"
        elif types0[n] is int:
            assert 0 <= ns[n] <= 199, f"int {n} out of range after {line!r}"


def _exec(text: str, ns: dict, null, what: str) -> None:
    """exec() the program text; a statement that fails at run time is an invalid event."""
    try:
        with contextlib.redirect_stdout(null):
            exec(text, ns)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{what} failed at run time: {type(exc).__name__}: {exc}") from exc


def replay(s: Session) -> list[str]:
    assert s.lang == "en"
    null = _Null()
    # initial lines: plain literal assignments, each variable once
    ns0: dict = {}
    seen = set()
    for line in s.initial:
        kind, g = _parse(line)
        assert kind == "lit", f"initial line must assign a literal: {line!r}"
        assert g[0] not in seen and not keyword.iskeyword(g[0]) and not hasattr(builtins, g[0]), line
        seen.add(g[0])
    _exec("\n".join(s.initial), ns0, null, "initial state")
    types0 = {n: type(v) for n, v in ns0.items() if n != "__builtins__"}
    assert set(types0) == seen
    live = {k: (list(v) if isinstance(v, list) else v) for k, v in ns0.items() if k != "__builtins__"}
    _state_ok(live, types0, "initial state")
    program = list(s.initial)
    answers = []
    for t in s.turns:
        for e in t.events:
            assert "\n" not in e and e == e.strip(), f"bad event text: {e!r}"
            _check_line(e, live, types0)
            _exec(e, live, null, repr(e))
            _state_ok(live, types0, e)
            program.append(e)
        m = _QUESTION.match(t.question)
        assert m, f"unparsed question: {t.question!r}"
        # ground truth: the whole program text, fresh namespace
        ns: dict = {}
        _exec("\n".join(program), ns, null, "program")
        v = ns[m.group(1)]
        if m.group(2) is not None:
            v = v[int(m.group(2))]
        answers.append(repr(v))
    return answers
