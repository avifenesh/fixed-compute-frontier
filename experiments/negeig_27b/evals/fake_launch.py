#!/usr/bin/env python3
"""Stand-in for `python -m sglang.launch_server` in the serve_arm.sh tests (CPU only, standard library).

Takes the real flags (--port, --dp-size, --tp-size, the rest is recorded). Serves GET /health and POST /v1/chat/completions.
What it prints to stderr is what a real server log would hold for the engagement check: with SGLANG_PLUGINS=negeig one
`negeig: installed v1 ...` line, then per rank (dp * tp) one `negeig: active v1: ...` line twice, as the real plugin's
_say() does (a bare print, and logger.info rendered by SGLang's root handler as `[date TP0] negeig: ...`), carrying the
sha256 of the NEGEIG_GATES file it was given.

Every launch appends one JSON line to $FAKE_RECORD: argv, the env variables that matter, and whether the plugin variables
were set. Knobs (env, read at start):
  FAKE_CRASH=1          exit 3 at once
  FAKE_DELAY=S          /health answers 503 for S seconds
  FAKE_ANSWER=TEXT      the chat answer (default 391)
  FAKE_ACTIVE_RANKS=N   print N active lines instead of dp*tp
  FAKE_LAYERS=N         layers on each rank (default 48 / tp)
  FAKE_SHA=HEX          print this sha256 instead of the gates file's
  FAKE_MAXW=X           the max|W| printed (default 0.5)
  FAKE_NO_ENGAGE=1      print no line even with the plugin loaded
  FAKE_FORCE_ENGAGE=1   print the lines even without the plugin (a stock server that carries the patch anyway)
  FAKE_EXTRA_LINE=1     print one more `negeig:` line that is not an active or installed line (a warning)
  FAKE_LOG_FORM=X       bare | logged | both (default both): which forms of the active line are printed
  FAKE_CHAT_STATUS=N    answer chat requests with this HTTP status
"""
import hashlib
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


port = int(arg("--port", "30000"))
dp = int(arg("--dp-size", "1"))
tp = int(arg("--tp-size", "1"))
env = os.environ
if env.get("FAKE_CRASH") == "1":
    print("fake sglang: planned crash", file=sys.stderr, flush=True)
    sys.exit(3)
if env.get("FAKE_RECORD"):
    with open(env["FAKE_RECORD"], "a") as f:
        f.write(json.dumps({"argv": sys.argv[1:], "plugins": env.get("SGLANG_PLUGINS"), "gates": env.get("NEGEIG_GATES"),
                            "pythonpath": env.get("PYTHONPATH"), "cuda_home": env.get("CUDA_HOME"),
                            "path0": env.get("PATH", "").split(":")[0], "jit_deepgemm": env.get("SGLANG_ENABLE_JIT_DEEPGEMM"),
                            "pid": os.getpid()}) + "\n")
loaded = "negeig" in env.get("SGLANG_PLUGINS", "").split(",")
if (loaded and env.get("FAKE_NO_ENGAGE") != "1") or env.get("FAKE_FORCE_ENGAGE") == "1":
    gates = env.get("NEGEIG_GATES", "")
    sha = env.get("FAKE_SHA") or (hashlib.sha256(open(gates, "rb").read()).hexdigest() if gates and os.path.isfile(gates) else "none")
    n = int(env.get("FAKE_ACTIVE_RANKS", str(dp * tp)))
    layers = int(env.get("FAKE_LAYERS", str(48 // tp)))
    form = env.get("FAKE_LOG_FORM", "both")
    ver = env.get("FAKE_INSTALLED_VERSION", "1")
    line = f"negeig: installed v{ver} on sglang 0.5.20 (pid {os.getpid()})"
    if form in ("bare", "both"):
        print(line, file=sys.stderr, flush=True)
    if form in ("logged", "both"):
        print(f"[2026-10-01 00:00:00] {line}", file=sys.stderr, flush=True)
    for r in range(n):
        line = (f"negeig: active v1: {layers} gated GDN layers on this rank, gates from file {gates} sha256={sha}, "
                f"max|W|={env.get('FAKE_MAXW', '0.5')}")
        if form in ("bare", "both"):
            print(line, file=sys.stderr, flush=True)
        if form in ("logged", "both"):
            print(f"[2026-10-01 00:00:0{r % 10} DP{r // max(1, tp)} TP{r % max(1, tp)}] {line}", file=sys.stderr, flush=True)
if env.get("FAKE_EXTRA_LINE") == "1":
    print("negeig: warning: a layer fell back to the stock kernel", file=sys.stderr, flush=True)
t0 = time.time()
delay = float(env.get("FAKE_DELAY", "0"))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200 if time.time() - t0 >= delay else 503, {})
        else:
            self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        if env.get("FAKE_RECORD"):
            with open(env["FAKE_RECORD"] + ".chat", "a") as f:
                f.write(json.dumps(body) + "\n")
        status = int(env.get("FAKE_CHAT_STATUS", "200"))
        if status != 200:
            self._send(status, {"error": "planned"})
        else:
            self._send(200, {"choices": [{"message": {"content": env.get("FAKE_ANSWER", "391")}}]})


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
