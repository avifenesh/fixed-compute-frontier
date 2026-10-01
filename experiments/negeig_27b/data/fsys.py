"""fsys: an interactive shell session over a small file tree; questions: where is file X, does path P exist,
what is in directory D, what is the current directory.

Commands (one per line, prefixed "$ "), each applied exactly as a real shell would:
  mv FROM TO  FROM moves into TO if TO is an existing directory, otherwise it becomes the path TO; a file moved
              onto an existing file replaces it (renames, moves, directory renames that shift every path below)
  cp FROM TO  copies a file the same way (copying onto an existing file changes no names)
  rm F / rm -r D / mkdir D / touch F / cd P
  ls, ls D, pwd, cat F, wc -l F   (no change: distractors)
Macros that permute locations (emitted as consecutive commands, a question may land in the middle):
  swap homes   two files exchange directories (two mv commands)
  swap dirs    two directories exchange names through a temporary name (three mv commands), so every file
               below them changes path
English only. All names are lowercase; files always contain a dot and directories never do. Files may sit
directly under the root "/". The SYSTEM prompt uses no single capital letters and no names from the split pools.
"""

from __future__ import annotations

import re
from collections import Counter

from common import Session, split_pool

STEMS = [
    "report", "journal", "budget", "readme", "main", "settings", "schema", "invoice", "summary", "draft", "plan",
    "todo", "changelog", "photo", "logo", "index", "setup", "deploy", "query", "model", "chart", "memo", "agenda",
    "roadmap", "minutes", "resume", "contract", "proposal", "receipt", "survey", "ledger", "roster", "manifest",
    "catalog", "inventory", "timeline", "outline", "slides", "sketch", "script", "patch", "stats", "policy",
    "pricing", "checklist", "handbook", "guide", "forecast", "quote", "letter", "brief", "digest", "recipe",
    "playlist", "lyrics", "essay", "thesis", "glossary", "dataset", "benchmark", "template", "profile", "session",
    "flyer",
]
DIRS = [
    "src", "docs", "build", "data", "logs", "tmp", "assets", "tests", "config", "scripts", "archive", "backup",
    "release", "staging", "drafts", "reports", "images", "vendor", "cache", "notes", "inbox", "outbox", "shared",
    "public", "private", "old", "lib", "bin", "web", "api", "db", "ui", "media", "exports", "imports", "invoices",
    "contracts", "photos", "music", "videos", "models", "plugins", "themes", "templates", "fixtures", "samples",
    "results", "metrics", "secrets", "projects", "clients", "sales", "finance", "design", "research", "legal", "hr",
    "ops", "billing", "travel",
]
EXTS = [".txt", ".md", ".csv", ".py", ".json", ".log", ".pdf", ".yaml", ".sh", ".png", ".xml", ".ini"]
SUFFIXES = ["_v2", "_old", "_new", "_final", "_copy", "_backup", "_draft", "-2", "-old", "_2024"]
HOLD = "_hold"  # temporary directory name used by the swap-dirs macro

assert len(STEMS) % 2 == 0 and len(DIRS) % 2 == 0 and not set(STEMS) & set(DIRS)
assert len(set(STEMS)) == len(STEMS) and len(set(DIRS)) == len(DIRS)
_PROMPT_NAMES = {"srv", "site", "menu", "apple", "apple_v2", "berry", "cherry"}  # used in the SYSTEM examples
assert not _PROMPT_NAMES & (set(STEMS) | set(DIRS)), "SYSTEM examples must not use split-pool names"


def _parent(p: str) -> str:
    return p.rsplit("/", 1)[0] or "/"


def _base(p: str) -> str:
    return p.rsplit("/", 1)[1]


def _cat(d: str, n: str) -> str:
    return "/" + n if d == "/" else d + "/" + n


def _under(p: str, d: str) -> bool:
    """p is strictly below directory d."""
    return p != d and p.startswith("/" if d == "/" else d + "/")


def _depth(p: str) -> int:
    return 0 if p == "/" else p.count("/")


class Sim:
    SYSTEM = {
        "en": "You are tracking a shell session over a small file tree. You get the initial tree and the current "
              "directory, then shell commands in order (lines starting with \"$ \"); apply each one exactly as a "
              "real shell would. Relative paths start from the current directory, \".\" is the current directory, "
              "\"..\" is the parent directory and \"/\" is the root, which always exists. \"mv FROM TO\" moves FROM "
              "into TO if TO is an existing directory; otherwise FROM becomes the new path TO (a rename, possibly "
              "into another directory). Moving a file onto an existing file replaces it. \"cp FROM TO\" copies a "
              "file the same way (copying onto an existing file changes no names). \"rm\" deletes a file, \"rm -r\" "
              "deletes a directory with everything in it, \"touch\" creates an empty file unless it already "
              "exists, \"mkdir\" creates a directory, \"cd\" changes the current directory, and \"ls\", \"pwd\", "
              "\"cat\" and \"wc -l\" change nothing. After each batch, answer the question with only the value: "
              "for \"Where is the file NAME now?\" (only asked when exactly one file has that name), its full "
              "absolute path (for example /srv/site/menu.md); for \"Does PATH exist?\" (a file or a directory), yes "
              "or no; for a directory listing, its entries sorted by name in plain ASCII order and joined by \", \", "
              "with a trailing / after each subdirectory (the / is not part of the name when sorting), for example: "
              "apple.txt, apple_v2.txt, berry/, cherry.md, or the word empty if it has no entries; for the current "
              "directory, its full absolute path.",
    }

    def __init__(self, rng, lang, split):
        assert lang == "en"
        self.rng = rng
        self.stems = split_pool(STEMS, split)
        self.dnames = split_pool(DIRS, split)
        self.dirs: set[str] = set()
        self.files: set[str] = set()
        self.queue: list[tuple] = []   # pending ops of a multi-command macro
        self.recent: list[tuple] = []  # ("f", file name) or ("d", directory path) of recent changes
        self.gone: list[str] = []      # absolute paths that used to exist
        self.revived: list[str] = []   # vacated file paths that a later mv/cp/touch filled again
        self.hot: list[str] = []       # current paths of recently created or moved files
        self.touched: set[str] = set()  # file names that changed at least once
        self.dtouched: set[str] = set()  # directories whose listing or location changed at least once
        tops = rng.sample(self.dnames, rng.randint(2, 3))
        for t in tops:
            self.dirs.add("/" + t)
        rest = [n for n in self.dnames if n not in tops]
        rng.shuffle(rest)
        n_dirs = rng.randint(6, 10)
        while len(self.dirs) < n_dirs and rest:
            parents = [d for d in sorted(self.dirs) if _depth(d) <= 2]
            self.dirs.add(_cat(rng.choice(parents), rest.pop()))
        dl = sorted(self.dirs)
        for _ in range(rng.randint(9, 15)):
            self.files.add(_cat(rng.choice(dl + ["/"]), self._new_name()))
        self.cwd = rng.choice(dl)
        self.init_dirs = set(dl)

    # ---------- naming helpers
    def _new_name(self) -> str:
        rng = self.rng
        have = {_base(f) for f in self.files}
        while True:
            n = rng.choice(self.stems)
            if rng.random() < 0.3:
                n += rng.choice(SUFFIXES)
            n += rng.choice(EXTS)
            if n not in have:
                return n

    def _new_dname(self, parent: str):
        kids = {_base(x) for x in self.dirs if _parent(x) == parent}
        opts = [n for n in self.dnames if n not in kids]
        return self.rng.choice(opts) if opts else None

    def _pick_file(self) -> str:
        hot = [p for p in self.hot if p in self.files]
        if hot and self.rng.random() < 0.5:
            return self.rng.choice(hot)
        return self.rng.choice(sorted(self.files))

    def _locs(self) -> list[str]:
        """Directories a file can live in: every directory plus the root."""
        return ["/"] + sorted(self.dirs)

    def _reuse(self, keep: str | None):
        """A vacated file path that can be filled again (undo, or a new file taking an old name), or None.
        Its name must not belong to any file other than `keep`, so names stay unique."""
        rng = self.rng
        others = {_base(g) for g in self.files if g != keep}
        cands = [p for p in self.gone[-40:] if "." in _base(p) and p not in self.files and p not in self.dirs
                 and (_parent(p) == "/" or _parent(p) in self.dirs) and _base(p) not in others]
        return rng.choice(cands) if cands else None

    def _free_dirs(self) -> list[str]:
        """Directories that may be moved or removed: not the current directory nor one of its ancestors."""
        return [d for d in sorted(self.dirs) if not (self.cwd == d or _under(self.cwd, d))]

    # ---------- Sim interface
    def initial(self):
        ents = [(d, "Directory") for d in self.dirs] + [(f, "File") for f in self.files]
        out = [f"{k} {p}" for p, k in sorted(ents)]
        out.append(f"Current directory: {self.cwd}")
        return out

    # ---------- op generators: each returns a list of ops (macros return several) or None if not possible now.
    # op = ("mv"|"cp", src, "into"|"as", dst) | ("rm"|"rmr"|"mkdir"|"touch"|"cd", path) | ("nop", kind, path)
    def _g_mv_move(self):
        f = self._pick_file()
        both = self.files | self.dirs
        ds = [d for d in self._locs() if d != _parent(f) and _cat(d, _base(f)) not in both]
        return [("mv", f, "into", self.rng.choice(ds))] if ds else None

    def _g_mv_rename(self):
        f = self._pick_file()
        if self.rng.random() < 0.2 and (p := self._reuse(f)) and p != f:
            return [("mv", f, "as", p)]
        return [("mv", f, "as", _cat(_parent(f), self._new_name()))]

    def _g_mv_move_rename(self):
        f = self._pick_file()
        if self.rng.random() < 0.2 and (p := self._reuse(f)) and p != f:
            return [("mv", f, "as", p)]
        ds = [d for d in self._locs() if d != _parent(f)]
        return [("mv", f, "as", _cat(self.rng.choice(ds), self._new_name()))] if ds else None

    def _g_mv_over(self):
        f = self._pick_file()
        gs = [g for g in sorted(self.files) if g != f]
        return [("mv", f, "as", self.rng.choice(gs))] if gs else None

    def _g_dir_move(self):
        rng = self.rng
        free = self._free_dirs()
        if not free:
            return None
        d = rng.choice(free)
        both = self.files | self.dirs
        tds = [t for t in ["/"] + sorted(self.dirs)
               if t != _parent(d) and t != d and not _under(t, d) and _depth(t) <= 3 and _cat(t, _base(d)) not in both]
        return [("mv", d, "into", rng.choice(tds))] if tds else None

    def _g_dir_rename(self):
        free = self._free_dirs()
        if not free:
            return None
        d = self.rng.choice(free)
        n = self._new_dname(_parent(d))
        return [("mv", d, "as", _cat(_parent(d), n))] if n else None

    def _g_swap_dirs(self):
        rng = self.rng
        free = self._free_dirs()
        pairs = [(a, b) for a in free for b in free if a < b and not _under(a, b) and not _under(b, a)]
        if not pairs:
            return None
        a, b = rng.choice(pairs)
        if rng.random() < 0.5:
            a, b = b, a
        tmp = _cat(_parent(a), HOLD)
        if tmp in self.dirs:
            return None
        return [("mv", a, "as", tmp), ("mv", b, "as", a), ("mv", tmp, "as", b)]

    def _g_swap_homes(self):
        rng = self.rng
        both = self.files | self.dirs
        fl = sorted(self.files)
        pairs = [(f, g) for f in fl for g in fl
                 if _parent(f) != _parent(g) and _cat(_parent(g), _base(f)) not in both
                 and _cat(_parent(f), _base(g)) not in both]
        if not pairs:
            return None
        f, g = rng.choice(pairs)
        return [("mv", f, "into", _parent(g)), ("mv", g, "into", _parent(f))]

    def _g_cp_new(self):
        rng = self.rng
        f = self._pick_file()
        if rng.random() < 0.15 and (p := self._reuse(None)):
            return [("cp", f, "as", p)]
        d = _parent(f) if rng.random() < 0.5 else rng.choice(self._locs())
        return [("cp", f, "as", _cat(d, self._new_name()))]

    def _g_cp_dup(self):
        f = self._pick_file()
        both = self.files | self.dirs
        ds = [d for d in self._locs() if d != _parent(f) and _cat(d, _base(f)) not in both]
        return [("cp", f, "into", self.rng.choice(ds))] if ds else None

    def _g_cp_over(self):
        f = self._pick_file()
        gs = [g for g in sorted(self.files) if g != f]
        return [("cp", f, "as", self.rng.choice(gs))] if gs else None

    def _g_rm(self):
        return [("rm", self._pick_file())] if len(self.files) > 5 else None

    def _g_rmr(self):
        free = self._free_dirs()
        free = [d for d in free if len(self.files) - sum(1 for f in self.files if _under(f, d)) >= 5
                and sum(1 for x in self.dirs if x != d and not _under(x, d)) >= 3]
        return [("rmr", self.rng.choice(free))] if free else None

    def _g_mkdir(self):
        rng = self.rng
        if len(self.dirs) >= 16:
            return None
        ps = [d for d in sorted(self.dirs) if _depth(d) <= 3]
        p = "/" if rng.random() < 0.1 else rng.choice(ps)
        n = self._new_dname(p)
        return [("mkdir", _cat(p, n))] if n else None

    def _g_touch_new(self):
        rng = self.rng
        d = self.cwd if rng.random() < 0.4 else rng.choice(self._locs())
        return [("touch", _cat(d, self._new_name()))]

    def _g_touch_old(self):
        return [("touch", self._pick_file())]

    def _g_cd(self):
        rng = self.rng
        r = rng.random()
        kids = [d for d in sorted(self.dirs) if _parent(d) == self.cwd]
        if r < 0.03:
            t = self.cwd
        elif r < 0.28 and self.cwd != "/":
            t = _parent(self.cwd)
        elif r < 0.5 and kids:
            t = rng.choice(kids)
        else:
            t = rng.choice([d for d in self._locs() if d != self.cwd])
        return [("cd", t)]

    def _g_nop(self):
        rng = self.rng
        k = rng.choice(["ls", "ls_dir", "pwd", "cat", "wc"])
        if k == "ls_dir":
            return [("nop", k, rng.choice(sorted(self.dirs)))]
        if k in ("cat", "wc"):
            return [("nop", k, self._pick_file())]
        return [("nop", k, "")]

    WEIGHTS = [
        ("_g_mv_move", 14), ("_g_mv_rename", 10), ("_g_mv_move_rename", 6), ("_g_mv_over", 3),
        ("_g_dir_move", 5), ("_g_dir_rename", 5), ("_g_swap_dirs", 3), ("_g_swap_homes", 4),
        ("_g_cp_new", 6), ("_g_cp_dup", 3), ("_g_cp_over", 1), ("_g_rm", 6), ("_g_rmr", 2), ("_g_mkdir", 4),
        ("_g_touch_new", 4), ("_g_touch_old", 2), ("_g_cd", 10), ("_g_nop", 8),
    ]

    def _gen(self):
        rng = self.rng
        if len(self.files) < 5:
            return getattr(self, rng.choice(["_g_touch_new", "_g_cp_new"]))()
        names, ws = zip(*self.WEIGHTS)
        for _ in range(40):
            ops = getattr(self, rng.choices(names, ws)[0])()
            if ops:
                return ops
        return [("nop", "pwd", "")]

    # ---------- rendering: paths as a user would type them from the current directory
    def _rel(self, p: str):
        pc = [] if p == "/" else p[1:].split("/")
        cc = [] if self.cwd == "/" else self.cwd[1:].split("/")
        k = 0
        while k < len(pc) and k < len(cc) and pc[k] == cc[k]:
            k += 1
        ups = len(cc) - k
        if ups > 2:
            return None
        return "/".join([".."] * ups + pc[k:]) or "."

    def _rp(self, p: str, slash: bool = False) -> str:
        rng = self.rng
        s = p
        rel = self._rel(p)
        if rel is not None and rng.random() < 0.6:
            s = rel
            if not s.startswith("..") and s != "." and rng.random() < 0.2:
                s = "./" + s
        if slash and s != "/" and rng.random() < 0.5:  # only ever passed for directories
            s += "/"
        return s

    def _render(self, op) -> str:
        k = op[0]
        if k in ("mv", "cp"):
            _, src, mode, dst = op
            a = self._rp(src)
            return f"$ {k} {a} {self._rp(dst, slash=(mode == 'into'))}"
        if k == "rm":
            return f"$ rm {self._rp(op[1])}"
        if k == "rmr":
            return f"$ rm -r {self._rp(op[1], slash=True)}"
        if k == "mkdir":
            return f"$ mkdir {self._rp(op[1])}"
        if k == "touch":
            return f"$ touch {self._rp(op[1])}"
        if k == "cd":
            return f"$ cd {self._rp(op[1], slash=True)}"
        kind, p = op[1], op[2]
        if kind == "ls_dir":
            return f"$ ls {self._rp(p, slash=True)}"
        if kind == "cat":
            return f"$ cat {self._rp(p)}"
        if kind == "wc":
            return f"$ wc -l {self._rp(p)}"
        return f"$ {kind}"

    # ---------- state bookkeeping
    def _gone_add(self, *ps):
        self.gone.extend(ps)
        del self.gone[:-60]

    def _mark(self, name: str):
        self.touched.add(name)
        self.recent.append(("f", name))
        del self.recent[:-40]

    def _markd(self, d: str):
        self.dtouched.add(d)
        self.recent.append(("d", d))
        del self.recent[:-40]

    def _hit(self, p: str):
        self.hot.append(p)
        del self.hot[:-12]
        if p in self.gone:
            self.revived.append(p)
            del self.revived[:-12]

    def _apply(self, op):
        k = op[0]
        F, D = self.files, self.dirs
        if k in ("mv", "cp"):
            _, src, mode, dst = op
            dest = _cat(dst, _base(src)) if mode == "into" else dst
            assert src != dest and dest not in D and (_parent(dest) == "/" or _parent(dest) in D), op
            assert mode == "as" or dst == "/" or dst in D, op
            if src in F:
                if k == "mv":
                    F.discard(src)
                    self._gone_add(src)
                    self._mark(_base(src))
                F.add(dest)
                self._mark(_base(dest))
                self._hit(dest)
                self._markd(_parent(dest))
                self._markd(_parent(src))
            else:
                assert k == "mv" and src in D and dest not in F and not _under(dest, src), op
                assert not (self.cwd == src or _under(self.cwd, src)), op
                self._gone_add(*[x for x in sorted(D | F) if x == src or _under(x, src)])
                cut = len(src)

                def mv(x):
                    return dest + x[cut:] if (x == src or _under(x, src)) else x

                self.dirs = {mv(x) for x in D}
                self.files = {mv(x) for x in F}
                for x in sorted(self.files):
                    if _under(x, dest):
                        self._mark(_base(x))
                        self._hit(x)
                self._markd(dest)
                self._markd(_parent(dest))
                self._markd(_parent(src))
        elif k == "rm":
            p = op[1]
            assert p in F, op
            F.remove(p)
            self._gone_add(p)
            self._mark(_base(p))
            self._markd(_parent(p))
        elif k == "rmr":
            p = op[1]
            assert p in D and not (self.cwd == p or _under(self.cwd, p)), op
            olds = [x for x in sorted(D | F) if x == p or _under(x, p)]
            self._gone_add(*olds)
            for x in olds:
                D.discard(x)
                if x in F:
                    F.discard(x)
                    self._mark(_base(x))
            self._markd(_parent(p))
        elif k == "mkdir":
            p = op[1]
            assert p not in D and p not in F and (_parent(p) == "/" or _parent(p) in D), op
            D.add(p)
            self._markd(p)
            self._markd(_parent(p))
        elif k == "touch":
            p = op[1]
            if p not in F:
                assert p not in D and (_parent(p) == "/" or _parent(p) in D), op
                F.add(p)
                self._mark(_base(p))
                self._hit(p)
                self._markd(_parent(p))
        elif k == "cd":
            assert op[1] == "/" or op[1] in D, op
            self.cwd = op[1]
        else:
            assert k == "nop", op

    def event(self):
        if not self.queue:
            self.queue = self._gen()
        op = self.queue.pop(0)
        text = self._render(op)
        self._apply(op)
        return text

    # ---------- questions
    def _q_where(self):
        rng = self.rng
        cnt = Counter(_base(f) for f in self.files)
        path_of = {_base(f): f for f in self.files if cnt[_base(f)] == 1}
        if not path_of:
            return None
        recent = [n for kind, n in self.recent[-24:] if kind == "f" and n in path_of]
        untouched = sorted(n for n in path_of if n not in self.touched)
        r = rng.random()
        if r < 0.12 and untouched:
            name, pick = rng.choice(untouched), "untouched"
        elif r < 0.8 and recent:
            name, pick = rng.choice(recent), "recent"
        else:
            name, pick = rng.choice(sorted(path_of)), "random"
        return f"Where is the file {name} now?", path_of[name], {"q": "where", "pick": pick}

    def _q_exists(self):
        rng = self.rng
        pick = "random"
        if self.revived and rng.random() < 0.08:
            p, pick = rng.choice(self.revived), "revived"
        elif rng.random() < 0.44:
            hot = [p for p in self.hot if p in self.files]
            hot += [p for kind, p in self.recent[-24:] if kind == "d" and p in self.dirs]
            untouched = sorted(f for f in self.files if _base(f) not in self.touched)
            if untouched and rng.random() < 0.15:
                p, pick = rng.choice(untouched), "untouched"
            elif hot and rng.random() < 0.6:
                p, pick = rng.choice(hot), "recent"
            else:
                p = rng.choice(sorted(self.files | self.dirs))
        elif self.gone and rng.random() < 0.6:
            p, pick = rng.choice(self.gone[-16:]), "gone"
        else:
            p = _cat(rng.choice(self._locs()), _base(rng.choice(sorted(self.files))))
        ans = "yes" if p in self.files or p in self.dirs else "no"
        return f"Does {p} exist?", ans, {"q": "exists", "pick": pick}

    def _q_list(self):
        rng = self.rng
        cand = [p for kind, p in self.recent[-24:] if kind == "d" and (p in self.dirs or p == "/")]
        untouched = sorted(d for d in self.init_dirs if d in self.dirs and d not in self.dtouched)
        r = rng.random()
        pick = "recent"
        if r < 0.10 and untouched:
            d, pick = rng.choice(untouched), "untouched"
        elif cand and r < 0.75:
            d = rng.choice(cand)
        elif r > 0.97:
            d, pick = "/", "root"
        else:
            d, pick = rng.choice(sorted(self.dirs)), "random"
        kids = sorted((_base(x), x in self.dirs) for x in self.dirs | self.files if _parent(x) == d)
        ans = ", ".join(n + "/" if isd else n for n, isd in kids) or "empty"
        return f"What is in the directory {d}?", ans, {"q": "list", "pick": pick, "n": len(kids)}

    def _q_cwd(self):
        return "What is the current directory?", self.cwd, {"q": "cwd", "pick": "cwd"}

    def ask(self):
        rng = self.rng
        for _ in range(20):
            k = rng.choices(["_q_where", "_q_exists", "_q_list", "_q_cwd"], [40, 25, 25, 10])[0]
            res = getattr(self, k)()
            if res:
                return res
        return self._q_cwd()


# ---------- independent checker: recompute every answer from the rendered text only.
# Shares nothing with Sim: its own path arithmetic, its own shell semantics, no vocabulary needed.
def replay(s: Session) -> list[str]:
    dirs: set[str] = set()
    files: set[str] = set()
    cwd = None

    def is_dir(p):
        return p == "/" or p in dirs

    def split(p):  # (parent directory, name) of an absolute path
        head, _, tail = p.rpartition("/")
        return (head or "/"), tail

    def below(p, d):  # p lies strictly inside directory d
        pre = d if d.endswith("/") else d + "/"
        return p != d and p.startswith(pre)

    def resolve(text):
        assert text and " " not in text, f"bad path {text!r}"
        parts = [] if text.startswith("/") or cwd == "/" else cwd.strip("/").split("/")
        for c in text.split("/"):
            if c in ("", "."):
                continue
            if c == "..":
                assert parts, f"{text!r} climbs above the root"
                parts.pop()
            else:
                parts.append(c)
        return "/" + "/".join(parts)

    def dest_of(src, raw):
        """Where `mv/cp src raw` puts src: into an existing directory, else the path itself."""
        dst = resolve(raw)
        if is_dir(dst):
            dst = ("" if dst == "/" else dst) + "/" + split(src)[1]
        else:
            assert not raw.endswith("/"), f"trailing slash on a non-directory: {raw!r}"
        return dst

    # initial state
    for line in s.initial:
        if (m := re.fullmatch(r"Directory (/\S+)", line)):
            dirs.add(m.group(1))
        elif (m := re.fullmatch(r"File (/\S+)", line)):
            files.add(m.group(1))
        elif (m := re.fullmatch(r"Current directory: (/\S*)", line)):
            assert cwd is None, "two current directories"
            cwd = m.group(1)
        else:
            raise AssertionError(f"unparsed initial line: {line!r}")
    assert cwd is not None and is_dir(cwd), "no valid current directory"
    for p in dirs | files:
        assert is_dir(split(p)[0]), f"parent of {p} missing"
    assert not dirs & files

    answers = []
    for t in s.turns:
        for e in t.events:
            assert e.startswith("$ "), f"unparsed event: {e!r}"
            tok = e[2:].split(" ")
            cmd, args = tok[0], tok[1:]
            if cmd in ("mv", "cp") and len(args) == 2:
                src = resolve(args[0])
                assert src in files or (cmd == "mv" and src in dirs), e
                if args[0].endswith("/"):
                    assert src in dirs, e
                dst = dest_of(src, args[1])
                assert dst != src and dst not in dirs and is_dir(split(dst)[0]), e
                if src in files:
                    if cmd == "mv":
                        files.remove(src)
                    files.add(dst)
                else:
                    assert dst not in files and not below(dst, src), e
                    assert not (cwd == src or below(cwd, src)), f"moves the current directory: {e}"
                    shift = lambda x: dst + x[len(src):] if (x == src or below(x, src)) else x  # noqa: E731
                    dirs = {shift(x) for x in dirs}
                    files = {shift(x) for x in files}
            elif cmd == "rm" and len(args) == 1:
                p = resolve(args[0])
                assert p in files, e
                files.remove(p)
            elif cmd == "rm" and len(args) == 2 and args[0] == "-r":
                p = resolve(args[1])
                assert p in dirs and not (cwd == p or below(cwd, p)), e
                dirs = {x for x in dirs if x != p and not below(x, p)}
                files = {x for x in files if not below(x, p)}
            elif cmd == "mkdir" and len(args) == 1:
                p = resolve(args[0])
                assert p not in dirs and p not in files and is_dir(split(p)[0]), e
                dirs.add(p)
            elif cmd == "touch" and len(args) == 1:
                p = resolve(args[0])
                if p not in files:
                    assert p not in dirs and is_dir(split(p)[0]), e
                    files.add(p)
            elif cmd == "cd" and len(args) == 1:
                p = resolve(args[0])
                assert is_dir(p), e
                cwd = p
            elif cmd == "ls" and len(args) <= 1:
                assert not args or is_dir(resolve(args[0])), e
            elif cmd == "pwd" and not args:
                pass
            elif cmd == "cat" and len(args) == 1:
                assert resolve(args[0]) in files, e
            elif cmd == "wc" and len(args) == 2 and args[0] == "-l":
                assert resolve(args[1]) in files, e
            else:
                raise AssertionError(f"unparsed event: {e!r}")
        q = t.question
        if (m := re.fullmatch(r"Where is the file (\S+) now\?", q)):
            hits = [p for p in files if split(p)[1] == m.group(1)]
            assert len(hits) == 1, f"file name {m.group(1)!r} is not unique: {hits}"
            answers.append(hits[0])
        elif (m := re.fullmatch(r"Does (/\S*) exist\?", q)):
            answers.append("yes" if is_dir(m.group(1)) or m.group(1) in files else "no")
        elif (m := re.fullmatch(r"What is in the directory (/\S*)\?", q)):
            d = m.group(1)
            assert is_dir(d), q
            ents = sorted((split(x)[1], x in dirs) for x in dirs | files if split(x)[0] == d)
            answers.append(", ".join(n + ("/" if isd else "") for n, isd in ents) or "empty")
        elif q == "What is the current directory?":
            answers.append(cwd)
        else:
            raise AssertionError(f"unparsed question: {q!r}")
    return answers
