#!/usr/bin/env python3
"""Fake `python -m sglang.launch_server` for the box tests: /health after STUB_SGLANG_DELAY seconds, and a chat endpoint
that answers STUB_SGLANG_ANSWER (default 391). STUB_SGLANG_CRASH=1 exits 3 at once. STUB_SGLANG_ARGLOG appends the launch argv, STUB_SGLANG_REQLOG appends one line
per chat request (the request body) so a test can read the sampling params and chat_template_kwargs sent.
For the parity tests: with SGLANG_PLUGINS=negeig it prints the plugin's engagement line to stderr (the real plugin's
format) unless STUB_SGLANG_NO_ENGAGE=1 (STUB_SGLANG_ALWAYS_BANNER=1 prints it with no plugin at all), and GET /server_info answers what this server was started with (argv, plugins,
gates, PYTHONPATH, CUDA_VISIBLE_DEVICES) so a stub collect can record which arm it really talked to."""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 30000
if os.environ.get("STUB_SGLANG_CRASH") == "1":
    print("stub sglang: planned crash", file=sys.stderr)
    sys.exit(3)
if os.environ.get("STUB_SGLANG_ARGLOG"):  # one JSON line per launch: argv, CUDA_HOME, the first two PATH entries
    with open(os.environ["STUB_SGLANG_ARGLOG"], "a") as f:
        f.write(json.dumps({"argv": sys.argv[1:], "cuda_home": os.environ.get("CUDA_HOME", ""),
                            "path": os.environ.get("PATH", "").split(":")[:2],
                            "plugins": os.environ.get("SGLANG_PLUGINS", ""), "gates": os.environ.get("NEGEIG_GATES", ""),
                            "pythonpath": os.environ.get("PYTHONPATH", ""),
                            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "")}) + "\n")
if ("negeig" in os.environ.get("SGLANG_PLUGINS", "").split(",") and os.environ.get("STUB_SGLANG_NO_ENGAGE") != "1") \
        or os.environ.get("STUB_SGLANG_ALWAYS_BANNER") == "1":  # the latter: a "stock" server that loaded the plugin anyway
    print(f"negeig: active v1: 48 gated GDN layers on this rank, gates from {os.environ.get('NEGEIG_GATES', '?')}, max|W|=0.5",
          file=sys.stderr, flush=True)
t0 = time.time()
delay = float(os.environ.get("STUB_SGLANG_DELAY", "0"))


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
        elif self.path.startswith("/server_info"):
            self._send(200, {"argv": sys.argv[1:], "plugins": os.environ.get("SGLANG_PLUGINS", ""),
                             "gates": os.environ.get("NEGEIG_GATES", ""), "pythonpath": os.environ.get("PYTHONPATH", ""),
                             "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "")})
        else:
            self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n).decode()
        if os.environ.get("STUB_SGLANG_REQLOG"):
            with open(os.environ["STUB_SGLANG_REQLOG"], "a") as f:
                f.write(body.replace("\n", " ") + "\n")
        self._send(200, {"choices": [{"message": {"content": os.environ.get("STUB_SGLANG_ANSWER", "391")}}]})


HTTPServer(("127.0.0.1", port), H).serve_forever()
