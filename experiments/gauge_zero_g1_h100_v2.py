#!/usr/bin/env python3
"""Sealed H100 re-gate for the unified one-round zero-pivot G1 contract."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import triton

from experiments.gauge_zero_g1_h100 import benchmark_cell, sha256_file


OUTPUT = Path("results/gauge-zero-g1-h100-v2.json")
PREREGISTRATION = Path("results/gauge-zero-g1-h100-v2-preregistration.md")
INTEGRITY_MANIFEST = Path("results/gauge-zero-g1-h100-v2-integrity-manifest.json")
BASE_SOURCE = Path("experiments/gauge_zero_g1_h100.py")
ATTENTION_SOURCE = Path("experiments/gauge_zero_g1_attention.py")
TEST_SOURCE = Path("tests/test_gauge_zero_g1_h100.py")
ATTENTION_TEST = Path("tests/test_gauge_zero_g1_attention.py")
FROZEN_ROWS_AND_TRIALS = ((1, 4000), (8, 400), (32, 400), (128, 400), (512, 400), (2048, 400))
WARMUP = 50
SEED = 7907


def validate_integrity() -> dict[str, bool]:
    manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    paths = {
        "source": Path(__file__),
        "base_source": BASE_SOURCE,
        "attention_source": ATTENTION_SOURCE,
        "test": TEST_SOURCE,
        "attention_test": ATTENTION_TEST,
        "preregistration": PREREGISTRATION,
    }
    checks = {
        name: manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in paths.items()
    }
    checks.update({
        "torch_version": manifest.get("torch_version") == torch.__version__,
        "triton_version": manifest.get("triton_version") == triton.__version__,
        "cuda_version": manifest.get("cuda_version") == torch.version.cuda,
    })
    if not all(checks.values()):
        raise ValueError(f"invalid gauge-zero G1 v2 integrity: {checks}")
    return checks


def main() -> None:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    integrity = validate_integrity()
    cells = [
        benchmark_cell(rows, trials, WARMUP, SEED + index * 1009)
        for index, (rows, trials) in enumerate(FROZEN_ROWS_AND_TRIALS)
    ]
    by_rows = {int(cell["rows"]): cell for cell in cells}
    gates = {
        "all_candidate_and_control_bf16_bit_exact": all(
            cell["correctness"]["candidate_bit_exact_bf16"]
            and cell["correctness"]["control_bit_exact_bf16"]
            for cell in cells
        ),
        "all_median_ratios_at_most_1_02": all(
            cell["candidate_over_control"]["median_ratio"] <= 1.02
            for cell in cells
        ),
        "all_upper_95_ratios_at_most_1_03": all(
            cell["candidate_over_control"]["upper_95"] <= 1.03
            for cell in cells
        ),
        "decode_1_and_8_upper_95_at_most_1_02": all(
            by_rows[rows]["candidate_over_control"]["upper_95"] <= 1.02
            for rows in (1, 8)
        ),
        "resident_weight_bytes_equal": all(
            cell["logical"]["resident_weight_bytes_each"] == 50_331_648
            for cell in cells
        ),
        "zero_pivot_metadata_bits": all(
            cell["logical"]["pivot_metadata_bits"] == 0 for cell in cells
        ),
    }
    payload = {
        "schema": "gauge-zero-g1-h100-v2",
        "numerical_contract": (
            "BF16 dense projection; shear and split-half RoPE in FP32; one BF16 final store"
        ),
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
        },
        "integrity": integrity,
        "integrity_manifest_sha256": sha256_file(INTEGRITY_MANIFEST),
        "frozen": {
            "rows_and_trials": FROZEN_ROWS_AND_TRIALS,
            "warmup": WARMUP,
            "seed": SEED,
            "bootstrap_replicates": 5000,
        },
        "cells": cells,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "ratios": {
            str(cell["rows"]): cell["candidate_over_control"] for cell in cells
        },
        "gates": gates,
        "all_gates_pass": payload["all_gates_pass"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
