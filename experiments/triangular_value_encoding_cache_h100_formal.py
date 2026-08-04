#!/usr/bin/env python3
"""Frozen H100 admission gate for triangular value encoding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
import triton

from experiments.triangular_value_encoding_cache_h100 import benchmark


OUTPUT = Path("results/triangular-value-encoding-cache-h100-formal.json")
PREREGISTRATION = Path("results/triangular-value-encoding-cache-h100-preregistration.md")
INTEGRITY_MANIFEST = Path("results/triangular-value-encoding-cache-h100-integrity-manifest.json")
ROWS = (1, 8, 32, 128, 512, 2048)
TRIALS = 200
WARMUP = 30
SEED = 811
LATENCY_UPPER_CAP = 1.02
VALUE_MAX_ABS_CAP = 0.015625


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_integrity() -> dict[str, bool]:
    manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    paths = {
        "formal_source": Path(__file__),
        "executor_source": Path("experiments/triangular_value_encoding_cache_h100.py"),
        "attention_source": Path("experiments/triangular_value_encoding_attention.py"),
        "attention_test": Path("tests/test_triangular_value_encoding_attention.py"),
        "bridge_test": Path("tests/test_triangular_value_encoding_cache_h100.py"),
        "preregistration": PREREGISTRATION,
    }
    checks = {
        name: manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in paths.items()
    }
    checks.update({
        "python_version": manifest.get("python_version") == platform.python_version(),
        "numpy_version": manifest.get("numpy_version") == np.__version__,
        "torch_version": manifest.get("torch_version") == torch.__version__,
        "cuda_version": manifest.get("cuda_version") == torch.version.cuda,
        "triton_version": manifest.get("triton_version") == triton.__version__,
    })
    if not all(checks.values()):
        raise ValueError(f"invalid TVE H100 integrity: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if not args.formal or args.output != OUTPUT:
        raise ValueError("formal gate requires frozen --formal and output")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    integrity = validate_integrity()
    cells = [
        benchmark(rows, TRIALS, WARMUP, SEED + index * 1009)
        for index, rows in enumerate(ROWS)
    ]
    cell_gates = []
    for cell in cells:
        correctness = cell["correctness"]
        gates = {
            "control_qkv_bit_exact": correctness["canonical_control_qkv_bit_exact"],
            "control_key_cache_bit_exact": correctness["canonical_control_key_cache_bit_exact"],
            "control_value_cache_bit_exact": correctness["canonical_control_value_cache_bit_exact"],
            "candidate_qkv_bit_exact": correctness["triangular_value_encoding_qkv_bit_exact"],
            "candidate_key_cache_bit_exact": correctness["triangular_value_encoding_key_cache_bit_exact"],
            "candidate_value_cache_within_bf16_cap": (
                correctness["triangular_value_encoding_value_cache_max_abs"]
                <= VALUE_MAX_ABS_CAP
            ),
            "candidate_path_nonzero": correctness["nonzero_candidate_value_delta"],
            "latency_upper_95_at_most_1_02": (
                cell["candidate_over_control"]["upper_95"] <= LATENCY_UPPER_CAP
            ),
            "no_added_learned_weight_bytes": cell["ledger"]["learned_weight_bytes_added"] == 0,
            "no_added_kv_bytes": cell["ledger"]["kv_bytes_added"] == 0,
            "no_metadata_bits": cell["ledger"]["metadata_bits"] == 0,
        }
        cell_gates.append({"rows": cell["rows"], "gates": gates, "pass": all(gates.values())})
    payload = {
        "schema": "triangular-value-encoding-cache-h100-formal-v1",
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
        },
        "integrity": integrity,
        "integrity_manifest_sha256": sha256_file(INTEGRITY_MANIFEST),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "frozen": {
            "rows": ROWS,
            "trials": TRIALS,
            "warmup": WARMUP,
            "seed": SEED,
            "latency_upper_95_cap": LATENCY_UPPER_CAP,
            "candidate_value_max_abs_cap": VALUE_MAX_ABS_CAP,
        },
        "cells": cells,
        "cell_gates": cell_gates,
        "all_gates_pass": all(cell["pass"] for cell in cell_gates) and all(integrity.values()),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "all_gates_pass": payload["all_gates_pass"],
        "cells": cell_gates,
    }, indent=2))


if __name__ == "__main__":
    main()
