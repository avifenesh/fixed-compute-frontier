#!/usr/bin/env python3
"""Second-seed replication of the exact-budget self-product FFN screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from experiments import self_product_ffn_lm_screen as base


SEED = 2718
PREREGISTRATION = Path("results/self-product-ffn-replication-preregistration.md")
TEST_SOURCE = Path("tests/test_self_product_ffn_replication.py")
MANIFEST = Path("results/self-product-ffn-replication-integrity-manifest.json")
DISCOVERY = Path("results/self-product-ffn-lm-screen.json")
BASE_SOURCE = Path("experiments/self_product_ffn_lm_screen.py")
BASE_INTEGRITY = Path("results/self-product-ffn-lm-integrity-manifest.json")
CONFIRMATION_SOURCE = Path("experiments/self_product_ffn_confirmation.py")
CONFIRMATION_PREREGISTRATION = Path("results/self-product-ffn-confirmation-preregistration.md")
CONFIRMATION_TEST = Path("tests/test_self_product_ffn_confirmation.py")
OUTPUT = Path("results/self-product-ffn-replication.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_replication_protocol(args: argparse.Namespace, arms: tuple[str, ...]) -> dict[str, Any]:
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
        raise ValueError(f"invalid replication protocol: {json.dumps(failed, sort_keys=True)}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def symmetric_envelope(screen: dict[str, Any]) -> dict[str, bool]:
    candidate = screen["arms"]["parallel_self_product"]["activation_diagnostics"]
    baseline = screen["arms"]["parallel_swiglu"]["activation_diagnostics"]
    metrics = (
        "activation_rms_layer_median",
        "activation_abs_p99_layer_max",
        "activation_abs_max_layer_max",
    )
    return {
        f"{phase}_{metric}_no_greater_than_swiglu": candidate[phase][metric] <= baseline[phase][metric]
        for phase in ("initial", "terminal")
        for metric in metrics
    }


def folded_terminal(screen: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm == "parallel_swiglu":
        return screen["arms"][arm]["evaluations"]["1525"]
    return screen["arms"][arm]["terminal_ablations"]["folded_deployment"]


def batch_clustered_interval(discovery: dict[str, Any], replication: dict[str, Any], control: str) -> dict[str, float]:
    differences = []
    for screen in (discovery, replication):
        candidate = np.asarray(folded_terminal(screen, "parallel_self_product")["per_batch_loss"], dtype=np.float64)
        reference = np.asarray(folded_terminal(screen, control)["per_batch_loss"], dtype=np.float64)
        if candidate.shape != reference.shape or candidate.size != 128:
            raise ValueError("expected 128 matched validation batches per seed")
        differences.append(candidate - reference)
    values = np.stack(differences).mean(axis=0)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
    return {
        "candidate_minus_reference_mean": mean,
        "standard_error": standard_error,
        "lower_95": mean - 1.96 * standard_error,
        "upper_95": mean + 1.96 * standard_error,
        "validation_batch_clusters": int(values.size),
        "seeds_per_cluster": len(differences),
        "interpretation": "descriptive interval clustered by validation batch; not a seed-level CI",
    }


def replication_decision(discovery: dict[str, Any], replication: dict[str, Any]) -> dict[str, Any]:
    base_decision = replication["decision"]
    gates = {
        key: value for key, value in base_decision["gates"].items()
        if key != "candidate_activation_outside_4x_baseline_rms_at_most_1_percent"
    }
    discovery_envelope = symmetric_envelope(discovery)
    replication_envelope = symmetric_envelope(replication)
    clustered = {
        control: batch_clustered_interval(discovery, replication, control)
        for control in ("parallel_swiglu", "parallel_wide_silu")
    }
    gates.update({f"discovery_{key}": value for key, value in discovery_envelope.items()})
    gates.update({f"replication_{key}": value for key, value in replication_envelope.items()})
    gates.update({f"batch_clustered_interval_favors_candidate_vs_{control}": value["upper_95"] < 0.0 for control, value in clustered.items()})
    return {
        "discovery_formal_promotion_preserved_as_false": not discovery["decision"]["advance_to_second_seed_subject_to_folded_h100"],
        "gates": gates,
        "symmetric_activation_envelope": {
            "discovery": discovery_envelope,
            "replication": replication_envelope,
        },
        "batch_clustered_terminal_intervals": clustered,
        "advance_to_confirmation_seed": all(gates.values()),
        "advance_to_scale_test": False,
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
    manifest = json.loads(MANIFEST.read_text())
    paths = {
        "source": Path(__file__),
        "preregistration": PREREGISTRATION,
        "test": TEST_SOURCE,
        "discovery": DISCOVERY,
        "base_source": BASE_SOURCE,
        "base_integrity": BASE_INTEGRITY,
        "confirmation_source": CONFIRMATION_SOURCE,
        "confirmation_preregistration": CONFIRMATION_PREREGISTRATION,
        "confirmation_test": CONFIRMATION_TEST,
    }
    checks = {key: manifest.get(key + "_sha256") == sha256_file(path) for key, path in paths.items()}
    if not all(checks.values()):
        raise ValueError(f"invalid replication integrity manifest: {checks}")
    discovery = json.loads(DISCOVERY.read_text())
    if discovery.get("source_sha256") != sha256_file(BASE_SOURCE):
        raise ValueError("discovery does not embed the frozen base source hash")
    if discovery.get("integrity_manifest_sha256") != sha256_file(BASE_INTEGRITY):
        raise ValueError("discovery does not embed the frozen base integrity hash")
    original_validator = base.validate_protocol
    base.validate_protocol = validate_replication_protocol
    try:
        payload = base.run(frozen_args())
    finally:
        base.validate_protocol = original_validator
    payload["replication_integrity"] = {
        "checks": checks,
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "discovery_sha256": sha256_file(DISCOVERY),
        "manifest_sha256": sha256_file(MANIFEST),
    }
    payload["replication_decision"] = replication_decision(discovery, payload)
    base.write_payload(OUTPUT, payload)
    return payload


def main() -> None:
    payload = run()
    print(json.dumps(payload["replication_decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
