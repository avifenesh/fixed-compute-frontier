#!/usr/bin/env python3
"""Earliest durable T84 marker, before importing NumPy, Torch, or controller."""

from __future__ import annotations

import os
import sys
import time


PROCESS_STARTED_NS = time.monotonic_ns()
TOP_LEVEL_STAGES = {"fixtures-timing", "stage0", "probes", "train"}
stage = sys.argv[1] if len(sys.argv) > 1 else None
os.environ["T84_PROCESS_START_NS"] = str(PROCESS_STARTED_NS)

if stage in TOP_LEVEL_STAGES:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ledger = os.path.join(root, "results", "t84-effective-v4-run", "ledger")
    os.makedirs(ledger, exist_ok=True)
    marker = os.path.join(
        ledger, f"early-{os.getpid()}-{PROCESS_STARTED_NS}.json"
    )
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        payload = (
            f'{{"pid":{os.getpid()},"process_started_ns":{PROCESS_STARTED_NS}}}\n'
        ).encode("ascii")
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.environ["T84_EARLY_MARKER"] = marker


from experiments.shared_score_heterogeneous_attention_t84_controller import main


if __name__ == "__main__":
    main()
