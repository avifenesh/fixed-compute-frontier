#!/usr/bin/env python3
"""Stand-in for sglang/check_parity.py in the parity.sh tests. Same subcommands and flags as the real one; no model, no
numbers. Every call is appended to $STUB_PARITY_LOG as a JSON line (cmd, argv, CUDA_VISIBLE_DEVICES, time, start|end).

  prepare, make-random-gates   write a small file at --out (prepare insists on --tokenizer, as the box run needs it)
  collect                      GET <url>/server_info and record what the server it talked to was started with, plus
                               whether its log (--server-log) carries a "negeig: active" line: the evidence compare reads
  hf-ref                       fails when a server is (or comes up, while it sleeps STUB_HF_SLEEP seconds) on $PARITY_PORT
                               with the same CUDA_VISIBLE_DEVICES (a GPU shared between the HF job and a server); records its mode
  compare                      checks the WIRING the way the real gates would depend on it, writes the verdict JSON with
                               "wiring_errors" (the tests assert it is empty) and exits as STUB_COMPARE_EXIT[_<mode>] says

Knobs: STUB_PARITY_LOG, STUB_PREPARE_RC, STUB_GATES_RC, STUB_COLLECT_FAIL=<tag>, STUB_HF_FAIL=<tag>, STUB_HF_SLEEP,
STUB_COMPARE_EXIT / STUB_COMPARE_EXIT_eager / STUB_COMPARE_EXIT_graphs (0 PASS, 1 G2 FAIL, 2 G2 INCONCLUSIVE),
STUB_COMPARE_NOWRITE=1 (exit without writing the verdict)."""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


def log(ev, **kw):
    p = os.environ.get("STUB_PARITY_LOG")
    if not p:
        return
    with open(p, "a") as f:
        f.write(json.dumps({"t": time.time(), "ev": ev, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""), **kw}) + "\n")


def server_info(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/server_info", timeout=5) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


def write(path, doc):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--out", required=True); p.add_argument("--n"); p.add_argument("--seed")
    p.add_argument("--vocab"); p.add_argument("--tokenizer")
    p = sub.add_parser("make-random-gates"); p.add_argument("--config", required=True); p.add_argument("--out", required=True)
    p.add_argument("--target-t", type=float, default=0.5); p.add_argument("--seed")
    p = sub.add_parser("collect"); p.add_argument("--url", required=True); p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True); p.add_argument("--tag", required=True); p.add_argument("--topk")
    p.add_argument("--decode-tokens"); p.add_argument("--server-log"); p.add_argument("--timeout")
    p = sub.add_parser("hf-ref"); p.add_argument("--model", required=True); p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True); p.add_argument("--tag", required=True); p.add_argument("--mode", required=True)
    p.add_argument("--topk"); p.add_argument("--device", default="cuda:0"); p.add_argument("--dtype"); p.add_argument("--patch")
    p.add_argument("--torch-gdn", action="store_true")
    p = sub.add_parser("compare")
    for k in ("stock", "stock-repeat", "zero", "rand", "hf-stock", "hf-rand"):
        p.add_argument("--" + k)
    p.add_argument("--out"); p.add_argument("--skip")
    for k in ("gap-ratio", "gap-floor", "effect-min", "noise-factor", "effect-rel-tol", "corr-min", "top1-slack",
              "decode-ratio", "decode-floor", "min-n"):
        p.add_argument("--" + k)
    a = ap.parse_args()
    log("call", cmd=a.cmd, argv=sys.argv[1:])

    if a.cmd == "prepare":
        if not a.tokenizer or not Path(a.tokenizer).is_dir():
            print("prepare: --tokenizer must name the model dir", file=sys.stderr)
            return 4
        if int(os.environ.get("STUB_PREPARE_RC", "0")):
            return int(os.environ["STUB_PREPARE_RC"])
        write(a.out, {"prompts": [{"id": f"p{i}", "input_ids": [1000 + i]} for i in range(4)]})
        return 0

    if a.cmd == "make-random-gates":
        if not Path(a.config).is_file():
            print(f"make-random-gates: no config {a.config}", file=sys.stderr)
            return 4
        if int(os.environ.get("STUB_GATES_RC", "0")):
            return int(os.environ["STUB_GATES_RC"])
        Path(a.out).write_text(f"stub gates target_t={a.target_t}\n")
        return 0

    if a.cmd == "collect":
        if os.environ.get("STUB_COLLECT_FAIL") == a.tag:
            print(f"collect {a.tag}: planned failure", file=sys.stderr)
            return 5
        info = server_info(a.url)
        if info is None:
            print(f"collect {a.tag}: no server answers {a.url}", file=sys.stderr)
            return 6
        engaged = False
        if a.server_log and Path(a.server_log).is_file():
            engaged = "negeig: active" in Path(a.server_log).read_text(errors="replace")
        write(a.out, {"tag": a.tag, "server": info, "log_engaged": engaged, "server_log": a.server_log})
        return 0

    if a.cmd == "hf-ref":
        port = os.environ.get("PARITY_PORT", "30002")
        info = server_info(f"http://127.0.0.1:{port}")
        log("hf-start", tag=a.tag, mode=a.mode, device=a.device, server_up=info is not None,
            server_cuda=(info or {}).get("cuda_visible"))
        if info is not None and info.get("cuda_visible", "x") == os.environ.get("CUDA_VISIBLE_DEVICES", ""):
            print(f"hf-ref {a.tag}: GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} is held by a running server", file=sys.stderr)
            return 7
        # the job "works" for STUB_HF_SLEEP seconds; a server that comes up on this GPU meanwhile is a shared GPU
        deadline = time.time() + float(os.environ.get("STUB_HF_SLEEP", "0"))
        while time.time() < deadline:
            info = server_info(f"http://127.0.0.1:{port}")
            if info is not None and info.get("cuda_visible", "x") == os.environ.get("CUDA_VISIBLE_DEVICES", ""):
                print(f"hf-ref {a.tag}: a server came up on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} while this job ran",
                      file=sys.stderr)
                return 7
            time.sleep(0.2)
        if os.environ.get("STUB_HF_FAIL") == a.tag:
            print(f"hf-ref {a.tag}: planned failure", file=sys.stderr)
            return 8
        write(a.out, {"tag": a.tag, "mode": a.mode, "device": a.device})
        log("hf-end", tag=a.tag)
        return 0

    # ---- compare
    def load(path):
        return json.loads(Path(path).read_text()) if path and Path(path).is_file() else None

    arms = {k: load(getattr(a, k.replace("-", "_"))) for k in ("stock", "stock-repeat", "zero", "rand", "hf-stock", "hf-rand")}
    mode = Path(a.stock).parent.name
    errs = []
    for k in ("stock", "stock-repeat", "zero", "rand", "hf-stock", "hf-rand"):
        if arms[k] is None:
            errs.append(f"missing input {k}")
    if not errs:
        st, z, r = arms["stock"]["server"], arms["zero"]["server"], arms["rand"]["server"]
        if arms["stock-repeat"]["server"] != st:
            errs.append("stock repeat did not come from the same server as stock")
        if st["plugins"] or st["gates"]:
            errs.append(f"stock arm is not stock: plugins={st['plugins']!r} gates={st['gates']!r}")
        if arms["stock"]["log_engaged"]:
            errs.append("stock server log shows the plugin engaged")
        if "negeig" not in z["plugins"].split(",") or z["gates"] != "zeros":
            errs.append(f"zero arm wrong: plugins={z['plugins']!r} gates={z['gates']!r}")
        if "negeig" not in r["plugins"].split(","):
            errs.append(f"rand arm has no plugin: {r['plugins']!r}")
        if not r["gates"].endswith(".safetensors") or not Path(r["gates"]).is_file():
            errs.append(f"rand arm gates is not an existing safetensors file: {r['gates']!r}")
        if arms["hf-rand"]["mode"] != r["gates"]:
            errs.append(f"hf-rand mode {arms['hf-rand']['mode']!r} is not the gates file the server used {r['gates']!r}")
        if arms["hf-stock"]["mode"] != "stock":
            errs.append(f"hf-stock mode is {arms['hf-stock']['mode']!r}")
        if not (z["pythonpath"].split(":")[0] == r["pythonpath"].split(":")[0] and Path(z["pythonpath"].split(":")[0], "negeig_sglang.py").is_file()):
            errs.append(f"plugin dir not first on PYTHONPATH of the patched servers: {z['pythonpath']!r}")
        if not (arms["zero"]["log_engaged"] and arms["rand"]["log_engaged"]):
            errs.append("a patched server log has no engagement line")
        for k, s in (("stock", st), ("zero", z), ("rand", r)):
            if s["cuda_visible"] != "0":
                errs.append(f"{k} server not on GPU 0: {s['cuda_visible']!r}")

        def flags(s):  # the argv without the one value that legitimately differs between arms (none today)
            return s["argv"]
        if not (flags(st) == flags(z) == flags(r)):
            errs.append("the three servers were launched with different flags")
        for need in ("--linear-attn-backend", "--linear-attn-prefill-backend", "--linear-attn-decode-backend"):
            for k, s in (("stock", st), ("zero", z), ("rand", r)):
                av = s["argv"]
                if need not in av or av[av.index(need) + 1] != "triton":
                    errs.append(f"{k}: {need} is not triton")
        for k, s in (("stock", st), ("zero", z), ("rand", r)):
            av = s["argv"]
            if "--disable-radix-cache" not in av:
                errs.append(f"{k}: no --disable-radix-cache")
            if "--tp-size" not in av or av[av.index("--tp-size") + 1] != "1":
                errs.append(f"{k}: --tp-size is not 1")
            has_cfg = "--cuda-graph-config" in av
            if mode == "eager" and not has_cfg:
                errs.append(f"{k}: eager mode without --cuda-graph-config")
            if mode == "graphs" and has_cfg:
                errs.append(f"{k}: graphs mode with --cuda-graph-config")
            if has_cfg:
                cfg = json.loads(av[av.index("--cuda-graph-config") + 1])
                if cfg != {"decode": {"backend": "disabled"}, "prefill": {"backend": "disabled"}}:
                    errs.append(f"{k}: eager graph config is {cfg}")
    skip = set((a.skip or "").split(",")) - {""}
    code = int(os.environ.get(f"STUB_COMPARE_EXIT_{mode}", os.environ.get("STUB_COMPARE_EXIT", "0")))
    summary = {g: "PASS" for g in ("G0", "G1", "G2", "G3")}
    if code == 1:
        summary["G2"] = "FAIL"
    elif code == 2:
        summary["G2"] = "INCONCLUSIVE"
    for g in skip:
        if g in summary:
            summary[g] = "WAIVED"
            if code and summary.get("G2") in ("FAIL", "INCONCLUSIVE") and g == "G2":
                code = 0  # a waived gate does not decide the exit
    for g in ("G0", "G1", "G2", "G3"):
        print(f"{g}: {summary[g]}  stub why ({mode})")
    doc = {"mode": mode, "summary": summary, "exit": code, "wiring_errors": errs, "skip": sorted(skip),
           "extra_args": {k: v for k, v in vars(a).items() if k in ("gap_ratio", "min_n") and v is not None}}
    log("compare", mode=mode, exit=code, errors=errs)
    print(json.dumps(doc, indent=1))
    if a.out and os.environ.get("STUB_COMPARE_NOWRITE") != "1":
        write(a.out, doc)
    return code


if __name__ == "__main__":
    sys.exit(main())
