#!/usr/bin/env python3
"""Stand-in for sglang/negeig_sglang.py in the serve_arm.sh tests: merge-lora, export-gates, install-plugin, static-check.
It reads the trainable with torch (the test interpreter has it) to tell a wide file from a LoRA-only one, the way the real
export-gates does ("no negeig_w tensors in the file" on a ctrl arm). Every call appends its argv to $FAKE_CLI_LOG.
Knobs (env): FAKE_MERGE_FAIL=1, FAKE_DELTA_KEPT (0.99), FAKE_EXPORT_ACCEPTS_CTRL=1 (export-gates writes a file for a ctrl
file too), FAKE_EXPORT_OTHER_FAIL=1 (export-gates fails for another reason), FAKE_STATIC_RC (0)."""
import hashlib
import json
import os
import sys
from pathlib import Path


def opt(name):
    return sys.argv[sys.argv.index(name) + 1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def has_gate(trainable):
    import torch

    obj = torch.load(trainable, map_location="cpu", weights_only=True)
    state = obj["trainable"] if isinstance(obj.get("trainable"), dict) else obj
    return any("negeig_w" in k for k in state)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if os.environ.get("FAKE_CLI_LOG"):
        with open(os.environ["FAKE_CLI_LOG"], "a") as f:
            f.write(json.dumps(sys.argv[1:]) + "\n")
    if cmd == "merge-lora":
        if os.environ.get("FAKE_MERGE_FAIL") == "1":
            print("merge-lora: planned failure", file=sys.stderr)
            return 1
        out = Path(opt("--out"))
        if out.exists():
            print(f"merge-lora: {out} exists", file=sys.stderr)
            return 1
        out.mkdir(parents=True)
        (out / "config.json").write_text("{}")
        dk = float(os.environ.get("FAKE_DELTA_KEPT", "0.99"))
        (out / "MERGE-RECEIPT.json").write_text(json.dumps({
            "delta_kept": dk, "delta_kept_worst_tensor": dk - 0.01, "trainable_sha256": sha(opt("--trainable")),
            "scale": float(opt("--scale")), "rounding": opt("--rounding"), "seed": int(opt("--seed")), "targets": 192}))
        print(json.dumps({"out": str(out)}))
        return 0
    if cmd == "export-gates":
        if os.environ.get("FAKE_EXPORT_OTHER_FAIL") == "1":
            print("export-gates: the config has no linear_attention layers", file=sys.stderr)
            return 1
        gated = has_gate(opt("--trainable"))
        if not gated and os.environ.get("FAKE_EXPORT_ACCEPTS_CTRL") != "1":
            print("export-gates: no negeig_w tensors in the file", file=sys.stderr)
            return 1
        out = Path(opt("--out"))
        out.write_bytes(b"gates:" + sha(opt("--trainable")).encode())
        print(json.dumps({"out": str(out), "layers": 48}))
        return 0
    if cmd == "install-plugin":
        d = Path(opt("--dir"))
        d.mkdir(parents=True, exist_ok=True)
        (d / "negeig_sglang.py").write_text("def register():\n    pass\n")
        print(json.dumps({"dir": str(d)}))
        return 0
    if cmd == "static-check":
        rc = int(os.environ.get("FAKE_STATIC_RC", "0"))
        print(json.dumps({"ok": rc == 0}))
        if rc:
            print("static-check: planned failure", file=sys.stderr)
        return rc
    print(f"fake negeig cli: unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
