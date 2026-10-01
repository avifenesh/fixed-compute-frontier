#!/usr/bin/env python3
"""Static checks of RUNBOOK.md for test_runbook.sh. CPU only, stdlib only.

  runbook_lint.py extract RUNBOOK DIR      write every fenced bash block to DIR/NN.sh, and a block whose first line is
                                           `# runbook-block: NAME` also to DIR/NAME.block; prints the block count
  runbook_lint.py func RUNBOOK NAME        print the definition of shell function NAME (`NAME() {` to the closing `}`)
  runbook_lint.py safety RUNBOOK           the lane's own guards: no teardown without --ids, no live lane box named, no kill by name
  runbook_lint.py refs RUNBOOK ROOT        every script, flag and subcommand the runbook names must exist in ROOT (the
                                           negeig_27b tree); prints one `PROBLEM ...` line per miss and `REFS scripts=N
                                           flags=M subcommands=K receipts=R`
"""
import os
import re
import sys

FENCE = re.compile(r"^```(\w*)\s*$")
MARK = re.compile(r"^#\s*runbook-block:\s*([A-Za-z0-9_-]+)\s*$")


def read_blocks(path):
    """-> (blocks, prose_lines): blocks are lists of lines of ```bash fences, prose is every line outside any fence."""
    blocks, prose, cur, lang = [], [], None, None
    for line in open(path, encoding="utf-8").read().split("\n"):
        m = FENCE.match(line)
        if m and cur is None:
            cur, lang = [], m.group(1)
            continue
        if line.startswith("```") and cur is not None:
            blocks.append((lang, cur))
            cur = None
            continue
        if cur is not None:
            cur.append(line)
        else:
            prose.append(line)
    if cur is not None:
        print("PROBLEM unterminated code fence", file=sys.stderr)
        sys.exit(1)
    return blocks, prose


def cmd_extract(rb, out):
    os.makedirs(out, exist_ok=True)
    blocks, _ = read_blocks(rb)
    n = 0
    for lang, lines in blocks:
        if lang != "bash":
            print(f"PROBLEM a ```{lang or '(no language)'} fence: every runbook block is ```bash", file=sys.stderr)
            sys.exit(1)
        n += 1
        text = "\n".join(lines) + "\n"
        with open(os.path.join(out, f"{n:02d}.sh"), "w", encoding="utf-8") as f:
            f.write(text)
        m = MARK.match(lines[0]) if lines else None
        if m:
            with open(os.path.join(out, m.group(1) + ".block"), "w", encoding="utf-8") as f:
                f.write(text)
    print(n)


def cmd_func(rb, name):
    blocks, _ = read_blocks(rb)
    for _, lines in blocks:
        for i, line in enumerate(lines):
            if re.match(rf"^{re.escape(name)}\(\)\s*\{{", line):
                j = i
                while j < len(lines) and lines[j] != "}" and not (j == i and line.rstrip().endswith("}")):
                    j += 1
                print("\n".join(lines[i:j + 1]))
                return 0
    print(f"PROBLEM no function {name}() in {rb}", file=sys.stderr)
    return 1


# subcommands: the word after the script name must be one of these, and the script must dispatch on it
SUBCMDS = {
    "ctl.sh": {"arm", "status", "hold", "release", "disarm", "logs"},
    "sd_box.sh": {"serve", "chat", "tools", "status", "finalize", "lengths", "stop"},
    "serve_arm.sh": {"up", "down"},
    "prepare_box.sh": {"all", "venv", "bfcl", "data", "nltk", "check"},
}
# names the runbook mentions that live outside the tree (or are not scripts of this lane)
EXTERNAL = {"agentic_env.py", "run.py", "preflight.js"}
RECEIPT_SUFFIX = ("OK", "FAILED", "FAIL", "EXIT", "CONFIRMED", "INCOMPLETE", "BROKEN", "INCONCLUSIVE", "WAIVED", "HIT",
                  "STARTED", "READ", "PREEMPTED")
# receipts a script builds from parts: the runbook's name -> (file, substrings that must all appear in it)
DYNAMIC = {
    "SD_CHAT_EXIT": ("box/sd_box.sh", ['SD_${name^^}_EXIT', "run_gen chat "]),
    "SD_TOOLS_EXIT": ("box/sd_box.sh", ['SD_${name^^}_EXIT', "run_gen tools "]),
    "SD_LENGTHS_OK": ("box/sd_lengths.py", ["SD_LENGTHS_{'OK' if not problems else 'FAIL'}"]),
    "SD_LENGTHS_FAIL": ("box/sd_lengths.py", ["SD_LENGTHS_{'OK' if not problems else 'FAIL'}"]),
}
ENV_NAMES = ["ALLOW_NO_REPLAY", "ALLOW_NO_NVLS", "REQUIRE_REPLAY", "SD_OUT", "SD_FG", "GPUS_PER_RUN", "ARMS",
             "NEGEIG27_STATE", "BXPULL_BWLIMIT", "GRAD_CKPT_FLAGS", "MEMRA_RELEASE_QUALIFICATION_MODE"]
TOKEN_RE = re.compile(r"(?P<script>(?<![A-Za-z0-9_\]])(?:\$\{?(?:BOX|E27R|E27)\}?/|\b(?:box|evals)/)?(?:(?:box|evals)/)?[A-Za-z0-9_]+\.(?:sh|py)\b)"
                      r"|(?P<flag>(?<![A-Za-z0-9_-])--[A-Za-z][A-Za-z0-9_-]*)")


def build_table(root):
    table = {}
    for d in (".", "box", "evals", "selfdistill"):
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        for fn in sorted(os.listdir(full)):
            if fn.endswith((".sh", ".py")) and os.path.isfile(os.path.join(full, fn)):
                table.setdefault(fn, os.path.join(full, fn))
    return table


def resolve(token, table, root):
    """-> path or None. A prefixed token must exist exactly where its prefix says; a bare one anywhere in the table."""
    t = token.replace("${", "$").replace("}", "")
    base = os.path.basename(t)
    if t.startswith(("$BOX/",)):
        return os.path.join(root, "box", base) if os.path.isfile(os.path.join(root, "box", base)) else None
    if t.startswith(("$E27/", "$E27R/")):
        rest = t.split("/", 1)[1]
        p = os.path.join(root, rest)
        return p if os.path.isfile(p) else None
    if t.startswith(("box/", "evals/")):
        p = os.path.join(root, t)
        return p if os.path.isfile(p) else None
    return table.get(base)


def dispatches(text, word):
    """The script dispatches on the command WORD: a `word)` or `word|` case arm, or a `= word` test. A function named
    cmd_word alone does not count: a handler nothing calls is not a command."""
    w = re.escape(word)
    return bool(re.search(rf"(?<![A-Za-z0-9_-]){w}[|)]", text) or re.search(rf"=\s*\"?{w}\"?\s*\]", text))


def logical_lines(lines):
    out, cur = [], ""
    for line in lines:
        if line.rstrip().endswith("\\"):
            cur += line.rstrip()[:-1] + " "
        else:
            out.append(cur + line)
            cur = ""
    if cur:
        out.append(cur)
    return out


def inline_code(prose):
    spans = []
    for line in prose:
        spans.extend(re.findall(r"`([^`]+)`", line))
    return spans


def cmd_refs(rb, root):
    blocks, prose = read_blocks(rb)
    table = build_table(root)
    texts = {}
    problems = []
    n_scripts = n_flags = n_sub = 0

    def src(path):
        if path not in texts:
            texts[path] = open(path, encoding="utf-8", errors="replace").read()
        return texts[path]

    units = []
    for _, lines in blocks:
        units.extend(logical_lines(lines))
    units.extend(inline_code(prose))
    for u in units:
        cur = None   # the script the next flag belongs to
        cur_name = None
        for m in TOKEN_RE.finditer(u):
            if m.group("script"):
                tok = m.group("script")
                base = os.path.basename(tok)
                if base in EXTERNAL:
                    cur = cur_name = None
                    continue
                path = resolve(tok, table, root)
                n_scripts += 1
                if path is None:
                    problems.append(f"PROBLEM no such script: {tok}   (in: {u.strip()[:100]})")
                    cur = cur_name = None
                    continue
                cur, cur_name = path, base
                subs = SUBCMDS.get(base)
                if subs:
                    nxt = re.match(r"\"?\s*([A-Za-z][A-Za-z0-9_-]*)", u[m.end():])
                    word = nxt.group(1) if nxt else ""
                    if word in subs:
                        n_sub += 1
                        if not dispatches(src(path), word):
                            problems.append(f"PROBLEM {base} has no '{word}' command   (in: {u.strip()[:100]})")
            else:
                flag = m.group("flag")
                if cur is None:
                    continue
                n_flags += 1
                if not re.search(re.escape(flag) + r"(?![A-Za-z0-9_-])", src(cur)):
                    problems.append(f"PROBLEM {cur_name} has no {flag}   (in: {u.strip()[:100]})")
    # receipt lines and env names the runbook quotes must exist in some script of the tree
    alltext = ""
    for d in (".", "box", "evals", "selfdistill"):
        full = os.path.join(root, d)
        if os.path.isdir(full):
            for fn in sorted(os.listdir(full)):
                p = os.path.join(full, fn)
                if os.path.isfile(p) and fn.endswith((".sh", ".py", ".example")):
                    alltext += open(p, encoding="utf-8", errors="replace").read() + "\n"
    whole = open(rb, encoding="utf-8").read()
    receipts = sorted({t for t in re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", whole) if t.split("_")[-1] in RECEIPT_SUFFIX})
    for t in receipts:
        if t in DYNAMIC:
            fn, parts = DYNAMIC[t]
            body = open(os.path.join(root, fn), encoding="utf-8", errors="replace").read()
            if not all(x in body for x in parts):
                problems.append(f"PROBLEM receipt {t}: {fn} no longer builds it")
        elif t not in alltext:
            problems.append(f"PROBLEM receipt {t} is named in the runbook but no script prints it")
    for e in ENV_NAMES:
        if e in whole and e not in alltext:
            problems.append(f"PROBLEM env {e} is named in the runbook but no script reads it")
    for p in problems:
        print(p)
    print(f"REFS scripts={n_scripts} flags={n_flags} subcommands={n_sub} receipts={len(receipts)}")
    return 1 if problems else 0


def cmd_safety(rb):
    blocks, _ = read_blocks(rb)
    problems = []
    for _, lines in blocks:
        for u in logical_lines(lines):
            t = u.strip()
            if "teardown.sh" in t and "--list" not in t and "--ids" not in t:
                problems.append(f"PROBLEM teardown.sh without --ids acts on every box of the table: {t[:100]}")
            if "h200p" in t:
                problems.append(f"PROBLEM names a box of the live lane: {t[:100]}")
            if re.search(r"\b(pkill|killall)\b", t):
                problems.append(f"PROBLEM kills by name (use a pid file): {t[:100]}")
            if re.search(r"\bsleep\s+[0-9]{2,}", t):
                problems.append(f"PROBLEM a long foreground sleep: {t[:100]}")
    for p in problems:
        print(p)
    print(f"SAFETY problems={len(problems)}")
    return 1 if problems else 0


def main(argv):
    if len(argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    mode, rb, arg = argv[1], argv[2], argv[3] if len(argv) == 4 else ""
    if mode == "safety":
        return cmd_safety(rb)
    if mode == "extract":
        cmd_extract(rb, arg)
        return 0
    if mode == "func":
        return cmd_func(rb, arg)
    if mode == "refs":
        return cmd_refs(rb, arg)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
