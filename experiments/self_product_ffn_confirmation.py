#!/usr/bin/env python3
"""Untouched third-seed confirmation for the exact-budget self-product FFN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import transformers

from experiments import self_product_ffn_lm_screen as base
from experiments.self_product_ffn_replication import sha256_file, symmetric_envelope


SEED = 31415
PREREGISTRATION = Path("results/self-product-ffn-confirmation-preregistration.md")
TEST_SOURCE = Path("tests/test_self_product_ffn_confirmation.py")
MANIFEST = Path("results/self-product-ffn-confirmation-integrity-manifest.json")
DISCOVERY = Path("results/self-product-ffn-lm-screen.json")
REPLICATION = Path("results/self-product-ffn-replication.json")
BASE_SOURCE = Path("experiments/self_product_ffn_lm_screen.py")
BASE_INTEGRITY = Path("results/self-product-ffn-lm-integrity-manifest.json")
REPLICATION_SOURCE = Path("experiments/self_product_ffn_replication.py")
REPLICATION_INTEGRITY = Path("results/self-product-ffn-replication-integrity-manifest.json")
OUTPUT = Path("results/self-product-ffn-confirmation.json")


def validate_confirmation_protocol(args: argparse.Namespace, arms: tuple[str, ...]) -> dict[str, Any]:
    expected = {
        "arms": base.VALID_ARMS,
        "device_contains": "H100",
        "torch_version": "2.5.1+cu124",
        "cuda_version": "12.4",
        "transformers_version": "4.57.6",
        "sequence_length": 512,
        "micro_batch_size": 32,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
        "eval_batches": 128,
        "steps": 1525,
        "eval_steps": [305, 1525],
        "warmup_steps": 100,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "seed": SEED,
        "prediction_tokens": 49_971_200,
    }
    actual = {
        "arms": arms,
        "device_contains": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "eval_batch_size": args.eval_batch_size,
        "eval_batches": args.eval_batches,
        "steps": args.steps,
        "eval_steps": args.eval_steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "prediction_tokens": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length,
    }
    checks = {
        key: (expected_value in actual[key] if key == "device_contains" else actual[key] == expected_value)
        for key, expected_value in expected.items()
    }
    if not all(checks.values()):
        failed = {key: {"expected": expected[key], "actual": actual[key]} for key, valid in checks.items() if not valid}
        raise ValueError(f"invalid confirmation protocol: {json.dumps(failed, sort_keys=True)}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def confirmation_decision(
    discovery: dict[str, Any], replication: dict[str, Any], confirmation: dict[str, Any]
) -> dict[str, Any]:
    base_decision = confirmation["decision"]
    gates = {
        key: value for key, value in base_decision["gates"].items()
        if key != "candidate_activation_outside_4x_baseline_rms_at_most_1_percent"
    }
    envelope = symmetric_envelope(confirmation)
    gates.update({f"confirmation_{key}": value for key, value in envelope.items()})
    gates["revised_exploratory_seed_2718_passed"] = replication["replication_decision"]["advance_to_confirmation_seed"]
    gates["discovery_formal_failure_preserved"] = replication["replication_decision"]["discovery_formal_promotion_preserved_as_false"]
    gates["discovery_quality_and_other_original_gates_passed"] = all(
        value for key, value in discovery["decision"]["gates"].items()
        if key != "candidate_activation_outside_4x_baseline_rms_at_most_1_percent"
    )
    return {
        "gates": gates,
        "symmetric_activation_envelope": envelope,
        "per_seed_relative_improvements": {
            "1907": discovery["decision"]["candidate_relative_improvements"],
            "2718": replication["decision"]["candidate_relative_improvements"],
            "31415": confirmation["decision"]["candidate_relative_improvements"],
        },
        "advance_to_scale_test": all(gates.values()),
    }


def frozen_args() -> argparse.Namespace:
    return argparse.Namespace(
        train_file=Path("data/block-algebra-scratch/train.uint16.bin"),
        validation_file=Path("data/block-algebra-scratch/validation.uint16.bin"),
        data_manifest=Path("results/block-algebra-scratch-data-manifest.json"),
        output=OUTPUT,
        arms=",".join(base.VALID_ARMS),
        sequence_length=512,
        micro_batch_size=32,
        gradient_accumulation=2,
        eval_batch_size=32,
        eval_batches=128,
        steps=1525,
        eval_steps=[305, 1525],
        warmup_steps=100,
        learning_rate=3e-4,
        weight_decay=0.1,
        gradient_clip=1.0,
        seed=SEED,
        strict_protocol=True,
    )


def run() -> dict[str, Any]:
    replication_integrity = json.loads(REPLICATION_INTEGRITY.read_text())
    precommitted_checks = {
        "source": replication_integrity.get("confirmation_source_sha256") == sha256_file(Path(__file__)),
        "preregistration": replication_integrity.get("confirmation_preregistration_sha256") == sha256_file(PREREGISTRATION),
        "test": replication_integrity.get("confirmation_test_sha256") == sha256_file(TEST_SOURCE),
    }
    if not all(precommitted_checks.values()):
        raise ValueError(f"confirmation artifacts differ from pre-seed-2718 commitment: {precommitted_checks}")
    manifest = json.loads(MANIFEST.read_text())
    paths = {
        "source": Path(__file__),
        "preregistration": PREREGISTRATION,
        "test": TEST_SOURCE,
        "discovery": DISCOVERY,
        "replication": REPLICATION,
        "base_source": BASE_SOURCE,
        "base_integrity": BASE_INTEGRITY,
        "replication_source": REPLICATION_SOURCE,
        "replication_integrity": REPLICATION_INTEGRITY,
    }
    checks = {key: manifest.get(key + "_sha256") == sha256_file(path) for key, path in paths.items()}
    if not all(checks.values()):
        raise ValueError(f"invalid confirmation integrity manifest: {checks}")
    discovery = json.loads(DISCOVERY.read_text())
    replication = json.loads(REPLICATION.read_text())
    original_validator = base.validate_protocol
    base.validate_protocol = validate_confirmation_protocol
    try:
        payload = base.run(frozen_args())
    finally:
        base.validate_protocol = original_validator
    payload["confirmation_integrity"] = {
        "checks": checks,
        "pre_seed_2718_commitment_checks": precommitted_checks,
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "manifest_sha256": sha256_file(MANIFEST),
    }
    payload["confirmation_decision"] = confirmation_decision(discovery, replication, payload)
    base.write_payload(OUTPUT, payload)
    return payload


def main() -> None:
    payload = run()
    print(json.dumps(payload["confirmation_decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
