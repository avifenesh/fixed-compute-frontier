#!/usr/bin/env python3
"""Apply the three edits the eval needs to a clean gorilla checkout (pins.json bfcl.commit). Standard library only.

    python bfcl_patch.py --gorilla DIR [--handler tiyuvta_hermes.py] [--check]

1. copy the vendored handler to bfcl_eval/model_handler/local_inference/tiyuvta_hermes.py (sha256 pinned);
2. bfcl_eval/constants/model_config.py: import the handler and register one key, negeig/qwen3.8-27b-FC;
3. bfcl_eval/model_handler/local_inference/base_oss_handler.py: the generation cap becomes BFCL_MAX_TOKENS
   (default 4096, the stock value), so a think-on run can raise it. The negeig gates run think-off at 4096.

Every edit is anchored on a line that must occur exactly once. Zero or two hits is a hard error, never a guess, so a
different gorilla commit fails loudly instead of being patched at the wrong place. Re-running on a patched tree is a
no-op. `--check` only reports the state (exit 0 patched, 1 clean, 2 anything else).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE.parent / "pins.json").read_text(encoding="utf-8"))["bfcl"]
KEY = PINS["registry_key"]

MODEL_CONFIG = "bfcl_eval/constants/model_config.py"
BASE_OSS = "bfcl_eval/model_handler/local_inference/base_oss_handler.py"
HANDLER_DST = "bfcl_eval/model_handler/local_inference/tiyuvta_hermes.py"

IMPORT_ANCHOR = "from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler"
IMPORT_NEW = "from bfcl_eval.model_handler.local_inference.tiyuvta_hermes import TiyuvtaHermesHandler"
MAP_ANCHOR = "local_inference_model_map = {"
ENTRY = f'''    "{KEY}": ModelConfig(
        model_name="qwen3.8-27b",
        display_name="Qwen3.8-27B negeig Stage A, think-off (FC)",
        url="https://huggingface.co/Qwen/Qwen3.8-27B",
        org="tiyuvta",
        license="apache-2.0",
        model_handler=TiyuvtaHermesHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    ),'''
CAP_ANCHOR = "                4096,"
CAP_NEW = ('                int(os.getenv("BFCL_MAX_TOKENS", "4096")),'
           "  # think-on arms need room for the reasoning")


class PatchError(SystemExit):
    pass


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def only_line(lines: list, text: str, what: str) -> int:
    """Index of the one line equal to `text`; anything else is an error."""
    hits = [i for i, ln in enumerate(lines) if ln == text]
    if len(hits) != 1:
        raise PatchError(f"anchor for {what} ({text.strip()!r}) found {len(hits)} times, expected exactly 1")
    return hits[0]


def patch_model_config(src: str) -> Tuple[str, str]:
    """(new text, state) where state is 'patched' (already), or 'applied'."""
    has_import = IMPORT_NEW in src.split("\n")
    has_entry = f'    "{KEY}": ModelConfig(' in src.split("\n")
    if has_import != has_entry:
        raise PatchError(f"{MODEL_CONFIG}: half patched (import={has_import}, entry={has_entry}); restore it with git checkout")
    if has_import and has_entry:
        if src.count(f'"{KEY}"') != 1 or src.count(IMPORT_NEW) != 1:
            raise PatchError(f"{MODEL_CONFIG}: the patch is present more than once")
        return src, "patched"
    lines = src.split("\n")
    i_map = only_line(lines, MAP_ANCHOR, "the local model map")
    i_imp = only_line(lines, IMPORT_ANCHOR, "the QwenFCHandler import")
    if i_imp >= i_map:
        raise PatchError(f"{MODEL_CONFIG}: the import anchor is below the map anchor, unexpected layout")
    out = lines[: i_imp + 1] + [IMPORT_NEW] + lines[i_imp + 1: i_map + 1] + ENTRY.split("\n") + lines[i_map + 1:]
    return "\n".join(out), "applied"


def patch_base_oss(src: str) -> Tuple[str, str]:
    lines = src.split("\n")
    if CAP_NEW in lines:
        if CAP_ANCHOR in lines:
            raise PatchError(f"{BASE_OSS}: both the stock cap and the patched cap are present")
        return src, "patched"
    i = only_line(lines, CAP_ANCHOR, "the generation cap")
    if "min(" not in lines[i - 1] or "self.max_context_length" not in lines[i + 1]:
        raise PatchError(f"{BASE_OSS}: the cap line is not inside the expected min(...) call")
    if not re.search(r"^import os$", src, re.M):
        raise PatchError(f"{BASE_OSS}: no 'import os', the patched line would not run")
    lines[i] = CAP_NEW
    return "\n".join(lines), "applied"


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gorilla", required=True, help="the gorilla checkout (repo root)")
    ap.add_argument("--handler", default=str(HERE / "tiyuvta_hermes.py"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--receipt", help="write a json receipt of the file hashes before and after")
    a = ap.parse_args(argv)
    root = Path(a.gorilla) / "berkeley-function-call-leaderboard"
    if not (root / "bfcl_eval").is_dir():
        raise PatchError(f"{root}/bfcl_eval not found")
    handler = Path(a.handler).read_bytes()
    if sha256(handler) != PINS["handler_sha256"]:
        raise PatchError(f"{a.handler}: sha256 {sha256(handler)[:12]} is not the pinned {PINS['handler_sha256'][:12]}")
    paths = {"model_config": root / MODEL_CONFIG, "base_oss": root / BASE_OSS, "handler": root / HANDLER_DST}
    for k in ("model_config", "base_oss"):
        if not paths[k].is_file():
            raise PatchError(f"{paths[k]} not found")
    mc_src = paths["model_config"].read_text(encoding="utf-8")
    bo_src = paths["base_oss"].read_text(encoding="utf-8")
    mc_new, mc_state = patch_model_config(mc_src)
    bo_new, bo_state = patch_base_oss(bo_src)
    h_state = "missing"
    if paths["handler"].is_file():
        h_state = "patched" if paths["handler"].read_bytes() == handler else "different"
    states = {"model_config": mc_state, "base_oss_handler": bo_state, "handler": h_state}
    if a.check:
        print(json.dumps(states))
        if all(v == "patched" for v in states.values()):
            return 0
        return 1 if all(v in ("applied", "missing") for v in states.values()) else 2
    if h_state == "different":
        raise PatchError(f"{paths['handler']} exists and differs from the pinned handler")
    before = {k: sha256(p.read_bytes()) for k, p in paths.items() if p.is_file()}
    if mc_state == "applied":
        paths["model_config"].write_text(mc_new, encoding="utf-8")
    if bo_state == "applied":
        paths["base_oss"].write_text(bo_new, encoding="utf-8")
    if h_state == "missing":
        shutil.copyfile(a.handler, paths["handler"])
    after = {k: sha256(p.read_bytes()) for k, p in paths.items()}
    print(json.dumps({"states": states, "sha256_after": after}))
    if a.receipt:
        Path(a.receipt).write_text(json.dumps({"states": states, "sha256_before": before, "sha256_after": after,
                                               "commit": PINS["commit"], "registry_key": KEY}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
