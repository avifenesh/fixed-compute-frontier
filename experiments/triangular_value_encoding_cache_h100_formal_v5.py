#!/usr/bin/env python3
"""Dependency-complete H100 admission replication for TVE."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
import triton

from experiments.triangular_value_encoding_cache_h100 import benchmark


OUTPUT = Path("results/triangular-value-encoding-cache-h100-formal-v5.json")
PREREGISTRATION = Path("results/triangular-value-encoding-cache-h100-preregistration-v5.md")
MANIFEST = Path("results/triangular-value-encoding-cache-h100-integrity-manifest-v5.json")
ROWS, TRIALS, WARMUP, SEED = (1, 8, 32, 128, 512, 2048), 200, 30, 1237


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_integrity() -> dict[str, bool]:
    manifest = json.loads(MANIFEST.read_text())
    paths = {
        "formal_source": Path(__file__),
        "executor_source": Path("experiments/triangular_value_encoding_cache_h100.py"),
        "executor_helper_source": Path("experiments/triangular_value_encoding_h100.py"),
        "attention_source": Path("experiments/triangular_value_encoding_attention.py"),
        "attention_test": Path("tests/test_triangular_value_encoding_attention.py"),
        "bridge_test": Path("tests/test_triangular_value_encoding_cache_h100.py"),
        "lm_test": Path("tests/test_triangular_value_encoding_lm_pilot.py"),
        "preregistration": PREREGISTRATION,
    }
    checks = {name: manifest.get(name + "_sha256") == sha256_file(path) for name, path in paths.items()}
    checks.update({
        "python_version": manifest.get("python_version") == platform.python_version(),
        "numpy_version": manifest.get("numpy_version") == np.__version__,
        "torch_version": manifest.get("torch_version") == torch.__version__,
        "cuda_version": manifest.get("cuda_version") == torch.version.cuda,
        "triton_version": manifest.get("triton_version") == triton.__version__,
    })
    if not all(checks.values()):
        raise ValueError(f"invalid TVE H100 v5 integrity: {checks}")
    return checks


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    integrity = validate_integrity()
    cells = [benchmark(rows, TRIALS, WARMUP, SEED + i * 1009) for i, rows in enumerate(ROWS)]
    judged = []
    for cell in cells:
        c, p = cell["correctness"], cell["propagated_executor_error"]
        gates = {
            "control_all_bit_exact": c["canonical_control_qkv_bit_exact"] and c["canonical_control_key_cache_bit_exact"] and c["canonical_control_value_cache_bit_exact"],
            "candidate_all_bit_exact_to_serving_forward": c["triangular_value_encoding_qkv_bit_exact"] and c["triangular_value_encoding_key_cache_bit_exact"] and c["triangular_value_encoding_value_cache_bit_exact"],
            "candidate_path_nonzero": c["nonzero_candidate_value_delta"],
            "algebra_rounding_below_one_bf16_step": c["candidate_algebra_rounding_max_abs"] <= 0.015625,
            "algebra_rounding_mean_at_most_2e_8": c["candidate_algebra_rounding_mean_abs"] <= 2e-8,
            "algebra_rounding_fraction_at_most_1e_5": c["candidate_algebra_rounding_mismatch_fraction"] <= 1e-5,
            "propagated_o_below_one_bf16_step": p["output_projection_max_abs"] <= 0.015625,
            "propagated_o_mean_at_most_1e_6": p["output_projection_mean_abs"] <= 1e-6,
            "proxy_nll_drift_at_most_1e_5": p["proxy_nll_abs_difference"] <= 1e-5,
            "latency_upper_95_at_most_1_025": cell["candidate_over_control"]["upper_95"] <= 1.025,
            "zero_added_state_and_cache": cell["ledger"]["learned_weight_bytes_added"] == 0 and cell["ledger"]["kv_bytes_added"] == 0 and cell["ledger"]["metadata_bits"] == 0,
        }
        judged.append({"rows": cell["rows"], "gates": gates, "pass": all(gates.values())})
    payload = {
        "schema": "triangular-value-encoding-cache-h100-formal-v5",
        "device": torch.cuda.get_device_name(0),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__},
        "integrity": integrity,
        "integrity_manifest_sha256": sha256_file(MANIFEST),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "frozen": {"rows": ROWS, "trials": TRIALS, "warmup": WARMUP, "seed": SEED},
        "cells": cells,
        "cell_gates": judged,
        "all_gates_pass": all(item["pass"] for item in judged) and all(integrity.values()),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"all_gates_pass": payload["all_gates_pass"], "cell_gates": judged}, indent=2))


if __name__ == "__main__":
    main()
