#!/usr/bin/env python3
"""Independent one-row uncertainty confirmation for gauge-zero G1 formal v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.gauge_zero_g1_h100 import benchmark_cell, validate_integrity


FORMAL_V1 = Path("results/gauge-zero-g1-h100.json")
FORMAL_V1_SHA256 = "b92660a47e9aef6bd6e2f62dcb0ce28111f2f76dffc1a155712306e0ad063064"
OUTPUT = Path("results/gauge-zero-g1-h100-confirmation.json")
ROWS = 1
TRIALS = 4000
WARMUP = 50
SEED = 6829
PREREGISTRATION = Path("results/gauge-zero-g1-h100-confirmation-preregistration.md")
TEST_SOURCE = Path("tests/test_gauge_zero_g1_h100_confirmation.py")
INTEGRITY_MANIFEST = Path("results/gauge-zero-g1-h100-confirmation-integrity-manifest.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_v1() -> dict[str, bool]:
    payload = json.loads(FORMAL_V1.read_text())
    checks = {
        "formal_v1_hash": sha256_file(FORMAL_V1) == FORMAL_V1_SHA256,
        "formal_v1_failed": payload.get("all_gates_pass") is False,
        "only_decode_uncertainty_gate_failed": [
            key for key, value in payload.get("gates", {}).items() if not value
        ] == ["decode_1_and_8_upper_95_at_most_1_02"],
        "one_row_point_estimate_was_inside_1_02": (
            payload["cells"][0]["rows"] == 1
            and payload["cells"][0]["candidate_over_control"]["median_ratio"] <= 1.02
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid formal-v1 prerequisite: {checks}")
    return checks


def validate_confirmation_integrity() -> dict[str, bool]:
    manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    paths = {
        "source": Path(__file__),
        "test": TEST_SOURCE,
        "preregistration": PREREGISTRATION,
    }
    checks = {
        name: manifest.get(f"{name}_sha256") == sha256_file(path)
        for name, path in paths.items()
    }
    if not all(checks.values()):
        raise ValueError(f"invalid confirmation integrity: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    integrity = validate_integrity()
    confirmation_integrity = validate_confirmation_integrity()
    prerequisite = validate_v1()
    cell = benchmark_cell(ROWS, TRIALS, WARMUP, SEED)
    gates = {
        "candidate_and_control_bf16_bit_exact": (
            cell["correctness"]["candidate_bit_exact_bf16"]
            and cell["correctness"]["control_bit_exact_bf16"]
        ),
        "median_ratio_at_most_1_02": (
            cell["candidate_over_control"]["median_ratio"] <= 1.02
        ),
        "bootstrap_upper_95_at_most_1_02": (
            cell["candidate_over_control"]["upper_95"] <= 1.02
        ),
        "zero_pivot_metadata_bits": cell["logical"]["pivot_metadata_bits"] == 0,
    }
    payload = {
        "schema": "gauge-zero-g1-h100-confirmation-v1",
        "device": torch.cuda.get_device_name(0),
        "formal_v1_sha256": FORMAL_V1_SHA256,
        "integrity": integrity,
        "confirmation_integrity": confirmation_integrity,
        "prerequisite": prerequisite,
        "frozen": {
            "rows": ROWS,
            "trials": TRIALS,
            "warmup": WARMUP,
            "seed": SEED,
        },
        "cell": cell,
        "gates": gates,
        "confirmation_pass": all(gates.values()),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "ratio": cell["candidate_over_control"],
        "gates": gates,
        "confirmation_pass": payload["confirmation_pass"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
