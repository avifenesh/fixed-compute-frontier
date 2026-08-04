#!/usr/bin/env python3
"""Precommitted admission path from fused serving success to scale training."""

from __future__ import annotations

import json
from pathlib import Path

from experiments import self_product_ffn_scale as scale


PREREGISTRATION = Path("results/self-product-ffn-scale-fused-admission-preregistration.md")
TEST_SOURCE = Path("tests/test_self_product_ffn_scale_fused_admission.py")
PRECOMMIT = Path("results/self-product-ffn-scale-fused-h100-integrity-manifest.json")
MANIFEST = Path("results/self-product-ffn-scale-fused-admission-integrity-manifest.json")
FUSED_SOURCE = Path("experiments/self_product_ffn_scale_fused_h100.py")
FUSED_PREREGISTRATION = Path("results/self-product-ffn-scale-fused-h100-preregistration.md")
FUSED_TEST = Path("tests/test_self_product_ffn_scale_fused_h100.py")
FUSED_RESULT = Path("results/self-product-ffn-scale-fused-h100.json")
OUTPUT = Path("results/self-product-ffn-scale-fused-admission.json")


def run():
    precommit = json.loads(PRECOMMIT.read_text())
    precommit_checks = {
        "source": precommit.get("admission_source_sha256") == scale.base.sha256_file(Path(__file__)),
        "preregistration": precommit.get("admission_preregistration_sha256") == scale.base.sha256_file(PREREGISTRATION),
        "test": precommit.get("admission_test_sha256") == scale.base.sha256_file(TEST_SOURCE),
        "confirmation": precommit.get("confirmation_sha256") == scale.base.sha256_file(scale.CONFIRMATION),
        "data_manifest": precommit.get("data_manifest_sha256") == scale.base.sha256_file(scale.DATA_MANIFEST),
    }
    if not all(precommit_checks.values()):
        raise ValueError(f"admission path changed after fused H100 precommit: {precommit_checks}")
    manifest = json.loads(MANIFEST.read_text())
    own_paths = {
        "admission_source": Path(__file__), "admission_preregistration": PREREGISTRATION,
        "admission_test": TEST_SOURCE, "precommit": PRECOMMIT, "h100_result": FUSED_RESULT,
    }
    own_checks = {key: manifest.get(key + "_sha256") == scale.base.sha256_file(path) for key, path in own_paths.items()}
    if not all(own_checks.values()):
        raise ValueError(f"invalid fused admission integrity: {own_checks}")
    fused = json.loads(FUSED_RESULT.read_text())
    fused_embedded_checks = {
        "precommit": fused.get("hashes", {}).get("precommit") == scale.base.sha256_file(PRECOMMIT),
        "admission_source": fused.get("hashes", {}).get("admission_source") == scale.base.sha256_file(Path(__file__)),
        "admission_preregistration": fused.get("hashes", {}).get("admission_preregistration") == scale.base.sha256_file(PREREGISTRATION),
        "admission_test": fused.get("hashes", {}).get("admission_test") == scale.base.sha256_file(TEST_SOURCE),
        "confirmation": fused.get("hashes", {}).get("confirmation") == scale.base.sha256_file(scale.CONFIRMATION),
        "data_manifest": fused.get("hashes", {}).get("data_manifest") == scale.base.sha256_file(scale.DATA_MANIFEST),
        "integrity_checks": bool(fused.get("integrity_checks")) and all(fused["integrity_checks"].values()),
        "semantic_equivalence_pass": fused.get("semantic_equivalence_pass") is True,
        "h100_feasibility_pass": fused.get("h100_feasibility_pass") is True,
        "peak_allocation_pass": fused.get("peak_allocation_pass") is True,
        "fused_serving_pass": fused.get("fused_serving_pass") is True,
    }
    if not all(fused_embedded_checks.values()):
        raise ValueError(f"fused H100 artifact cannot admit scale: {fused_embedded_checks}")
    original = {
        "MANIFEST": scale.MANIFEST, "H100_SOURCE": scale.H100_SOURCE,
        "H100_PREREGISTRATION": scale.H100_PREREGISTRATION,
        "H100_TEST": scale.H100_TEST, "H100_RESULT": scale.H100_RESULT,
        "OUTPUT": scale.OUTPUT,
    }
    scale.MANIFEST = MANIFEST
    scale.H100_SOURCE = FUSED_SOURCE
    scale.H100_PREREGISTRATION = FUSED_PREREGISTRATION
    scale.H100_TEST = FUSED_TEST
    scale.H100_RESULT = FUSED_RESULT
    scale.OUTPUT = OUTPUT
    try:
        payload = scale.run()
    finally:
        for key, value in original.items():
            setattr(scale, key, value)
    payload["fused_serving_admission"] = {
        "precommit_checks": precommit_checks,
        "integrity_checks": own_checks,
        "fused_embedded_checks": fused_embedded_checks,
        "source_sha256": scale.base.sha256_file(Path(__file__)),
        "preregistration_sha256": scale.base.sha256_file(PREREGISTRATION),
        "fused_h100_result_sha256": scale.base.sha256_file(FUSED_RESULT),
    }
    scale.base.write_payload(OUTPUT, payload)
    return payload


def main() -> None:
    payload = run()
    print(json.dumps(payload["scale_decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
