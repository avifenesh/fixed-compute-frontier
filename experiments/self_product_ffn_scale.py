#!/usr/bin/env python3
"""Scale test for the exact-budget self-product FFN."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import transformers

from experiments import self_product_ffn_lm_screen as base
from experiments import triangular_microdepth_lm_screen as triangular


HIDDEN_SIZE = 640
BASELINE_INTERMEDIATE_SIZE = 1792
WIDE = 2688
LAYERS = 16
ATTENTION_HEADS = 10
KV_HEADS = 2
HEAD_DIM = 64
SEED = 271828
VALID_ARMS = base.VALID_ARMS
PREREGISTRATION = Path("results/self-product-ffn-scale-preregistration.md")
TEST_SOURCE = Path("tests/test_self_product_ffn_scale.py")
MANIFEST = Path("results/self-product-ffn-scale-integrity-manifest.json")
CONFIRMATION = Path("results/self-product-ffn-confirmation.json")
H100_SOURCE = Path("experiments/self_product_ffn_scale_h100.py")
H100_PREREGISTRATION = Path("results/self-product-ffn-scale-h100-preregistration.md")
H100_TEST = Path("tests/test_self_product_ffn_scale_h100.py")
H100_RESULT = Path("results/self-product-ffn-scale-h100.json")
BASE_LM_SOURCE = Path("experiments/self_product_ffn_lm_screen.py")
TRIANGULAR_SOURCE = Path("experiments/triangular_microdepth_lm_screen.py")
COALESCED_SOURCE = Path("experiments/coalesced_attention_ffn_lm_screen.py")
REFLEX_SOURCE = Path("experiments/reflex_swiglu_lm_screen.py")
DATA_MANIFEST = Path("results/self-product-ffn-scale-data-manifest.json")
TRAIN_FILE = Path("data/self-product-ffn-scale/train.uint16.bin")
VALIDATION_FILE = Path("data/self-product-ffn-scale/validation.uint16.bin")
OUTPUT = Path("results/self-product-ffn-scale.json")

ORIGINAL_BUILD_MODEL = base.build_model


def configure_scale_globals() -> None:
    triangular.HIDDEN_SIZE = HIDDEN_SIZE
    triangular.BASELINE_INTERMEDIATE_SIZE = BASELINE_INTERMEDIATE_SIZE
    triangular.LAYERS = LAYERS
    triangular.ATTENTION_HEADS = ATTENTION_HEADS
    triangular.KV_HEADS = KV_HEADS
    triangular.HEAD_DIM = HEAD_DIM
    base.HIDDEN_SIZE = HIDDEN_SIZE
    base.BASELINE_INTERMEDIATE_SIZE = BASELINE_INTERMEDIATE_SIZE
    base.WIDE = WIDE
    base.PLAIN_SCALE = math.sqrt(BASELINE_INTERMEDIATE_SIZE / WIDE)


def build_scale_model(device: torch.device, arm: str):
    configure_scale_globals()
    return ORIGINAL_BUILD_MODEL(device, arm)


def validate_scale_protocol(args: argparse.Namespace) -> dict[str, Any]:
    expected = {
        "device_contains": "H100",
        "torch_version": "2.5.1+cu124",
        "cuda_version": "12.4",
        "transformers_version": "4.57.6",
        "sequence_length": 512,
        "micro_batch_size": 32,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
        "eval_batches": 128,
        "steps": 3050,
        "eval_steps": [610, 3050],
        "warmup_steps": 200,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "seed": SEED,
        "prediction_tokens": 99_942_400,
    }
    actual = {
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
    checks = {key: (value in actual[key] if key == "device_contains" else actual[key] == value) for key, value in expected.items()}
    if not all(checks.values()):
        failed = {key: {"expected": expected[key], "actual": actual[key]} for key, valid in checks.items() if not valid}
        raise ValueError(f"invalid scale protocol: {json.dumps(failed, sort_keys=True)}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def symmetric_envelope(results: dict[str, Any]) -> dict[str, bool]:
    candidate = results["parallel_self_product"]["activation_diagnostics"]
    baseline = results["parallel_swiglu"]["activation_diagnostics"]
    metrics = ("activation_rms_layer_median", "activation_abs_p99_layer_max", "activation_abs_max_layer_max")
    return {
        f"{phase}_{metric}_no_greater_than_swiglu": candidate[phase][metric] <= baseline[phase][metric]
        for phase in ("initial", "terminal") for metric in metrics
    }


def scale_decision(results: dict[str, Any], confirmation: dict[str, Any], h100_payload: dict[str, Any]) -> dict[str, Any]:
    aliased_results = {}
    for arm, result in results.items():
        evaluations = dict(result["evaluations"])
        evaluations["305"] = evaluations["610"]
        evaluations["1525"] = evaluations["3050"]
        aliased_results[arm] = {**result, "evaluations": evaluations}
    original = base.decide(aliased_results, True, bool(h100_payload["h100_feasibility_pass"]))
    gates = {
        key: value for key, value in original["gates"].items()
        if key != "candidate_activation_outside_4x_baseline_rms_at_most_1_percent"
    }
    envelope = symmetric_envelope(results)
    gates.update({f"scale_{key}": value for key, value in envelope.items()})
    gates["scale_h100_peak_allocation_pass"] = bool(h100_payload["peak_allocation_pass"])
    gates["untouched_confirmation_promoted_scale_test"] = bool(confirmation["confirmation_decision"]["advance_to_scale_test"])
    return {
        "base_quality_decision": original,
        "symmetric_activation_envelope": envelope,
        "gates": gates,
        "fixed_served_cost_scale_pass": all(gates.values()),
    }


def frozen_args() -> argparse.Namespace:
    return argparse.Namespace(
        train_file=TRAIN_FILE, validation_file=VALIDATION_FILE, data_manifest=DATA_MANIFEST,
        output=OUTPUT, arms=",".join(VALID_ARMS), sequence_length=512,
        micro_batch_size=32, gradient_accumulation=2, eval_batch_size=32,
        eval_batches=128, steps=3050, eval_steps=[610, 3050], warmup_steps=200,
        learning_rate=3e-4, weight_decay=0.1, gradient_clip=1.0, seed=SEED,
        strict_protocol=True,
    )


def run() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    manifest = json.loads(MANIFEST.read_text())
    paths = {
        "source": Path(__file__), "preregistration": PREREGISTRATION, "test": TEST_SOURCE,
        "confirmation": CONFIRMATION, "h100_source": H100_SOURCE,
        "h100_preregistration": H100_PREREGISTRATION, "h100_test": H100_TEST,
        "h100_result": H100_RESULT, "data_manifest": DATA_MANIFEST,
        "base_lm_source": BASE_LM_SOURCE, "triangular_source": TRIANGULAR_SOURCE,
        "coalesced_source": COALESCED_SOURCE, "reflex_source": REFLEX_SOURCE,
    }
    checks = {key: manifest.get(key + "_sha256") == base.sha256_file(path) for key, path in paths.items()}
    if not all(checks.values()):
        raise ValueError(f"invalid scale integrity manifest: {checks}")
    args = frozen_args()
    protocol = validate_scale_protocol(args)
    data_ledger = base.validate_data_ledger(DATA_MANIFEST, TRAIN_FILE, VALIDATION_FILE, 512)
    train_file = base.TokenFile(TRAIN_FILE, 512)
    validation_file = base.TokenFile(VALIDATION_FILE, 512)
    if train_file.sequence_count < args.steps * args.gradient_accumulation * args.micro_batch_size:
        raise ValueError("scale train file too short")
    h100_payload = json.loads(H100_RESULT.read_text())
    h100_embedded_expected = {
        "source": base.sha256_file(H100_SOURCE),
        "preregistration": base.sha256_file(H100_PREREGISTRATION),
        "test": base.sha256_file(H100_TEST),
        "lm_source": base.sha256_file(Path(__file__)),
        "lm_preregistration": base.sha256_file(PREREGISTRATION),
        "lm_test": base.sha256_file(TEST_SOURCE),
        "base_lm_source": base.sha256_file(BASE_LM_SOURCE),
        "triangular_source": base.sha256_file(TRIANGULAR_SOURCE),
        "coalesced_source": base.sha256_file(COALESCED_SOURCE),
        "reflex_source": base.sha256_file(REFLEX_SOURCE),
    }
    h100_embedded_checks = {
        key: h100_payload.get("hashes", {}).get(key) == value
        for key, value in h100_embedded_expected.items()
    }
    if not all(h100_embedded_checks.values()):
        raise ValueError(f"stale scale H100 artifact: {h100_embedded_checks}")
    expected_h100_protocol = {
        "seed": 271828,
        "warmups": 5,
        "repetitions": 30,
        "prefill_cells": [[1, 512], [8, 512], [32, 512]],
        "decode_cells": [[1, 512], [8, 512]],
        "bootstrap_repetitions": 5000,
        "weight_dtype": "torch.bfloat16",
    }
    h100_environment_checks = {
        "device": "H100" in str(h100_payload.get("device")),
        "torch_version": h100_payload.get("torch_version") == "2.5.1+cu124",
        "cuda_version": h100_payload.get("cuda_version") == "12.4",
        "transformers_version": h100_payload.get("transformers_version") == "4.57.6",
        "protocol": h100_payload.get("protocol") == expected_h100_protocol,
    }
    if not all(h100_environment_checks.values()):
        raise ValueError(f"invalid scale H100 environment/protocol: {h100_environment_checks}")
    confirmation = json.loads(CONFIRMATION.read_text())
    if not h100_payload.get("h100_feasibility_pass") or not h100_payload.get("peak_allocation_pass"):
        raise ValueError("scale H100 gate failed")
    configure_scale_globals()
    original_builder = base.build_model
    base.build_model = build_scale_model
    try:
        device = torch.device("cuda")
        payload: dict[str, Any] = {
            "candidate": "variance-matched-self-product-ffn-scale",
            "source_sha256": base.sha256_file(Path(__file__)),
            "preregistration_sha256": base.sha256_file(PREREGISTRATION),
            "integrity_checks": checks,
            "experiment_protocol": protocol,
            "data_ledger": data_ledger,
            "shape": {"hidden_size": HIDDEN_SIZE, "baseline_intermediate_size": BASELINE_INTERMEDIATE_SIZE, "wide": WIDE, "layers": LAYERS, "attention_heads": ATTENTION_HEADS, "kv_heads": KV_HEADS},
            "device": torch.cuda.get_device_name(0),
            "h100_embedded_hash_checks": h100_embedded_checks,
            "h100_environment_checks": h100_environment_checks,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "arms": {},
        }
        for arm in VALID_ARMS:
            print(json.dumps({"starting_arm": arm}), flush=True)
            reference_rms = None
            if "parallel_swiglu" in payload["arms"]:
                diagnostics = payload["arms"]["parallel_swiglu"]["activation_diagnostics"]
                reference_rms = {phase: diagnostics[phase]["activation_rms_layer_median"] for phase in ("initial", "terminal")}
            payload["arms"][arm] = base.train_arm(arm, args, train_file, validation_file, device, reference_rms)
            base.write_payload(OUTPUT, payload)
        payload["scale_decision"] = scale_decision(payload["arms"], confirmation, h100_payload)
        base.write_payload(OUTPUT, payload)
        return payload
    finally:
        base.build_model = original_builder


def main() -> None:
    payload = run()
    print(json.dumps(payload["scale_decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
